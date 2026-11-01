"""Publish speech-tokenizer Pub/Sub messages grouped by video_id."""

from __future__ import annotations

import argparse
import json
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml
from google.api_core.exceptions import AlreadyExists
from google.cloud import pubsub_v1
from google.protobuf.duration_pb2 import Duration
from google.protobuf.field_mask_pb2 import FieldMask

MAX_PUBSUB_RETENTION_SECONDS = 31 * 24 * 60 * 60
MAX_ACK_DEADLINE_SECONDS = 600


@dataclass
class WorkerPublishConfig:
    project_id: str = "speechsdk"
    subscription_id: str = "chatterbox-tts-sub"
    topic_id: str = "chatterbox-tts"
    bucket: str = "tiktok-pipeline-processed"
    clips_folder: str = "clips"
    manifest_path: str = "../../pose_cond_npy_manifest.txt"


@dataclass
class ClipSpec:
    video_id: str
    clip_id: str
    mp4_uri: str
    transcript_uri: str


@dataclass
class PublisherConfig:
    project_id: str
    topic_id: str
    subscription_id: str
    manifest_path: str
    max_video_ids_per_message: int = 8
    min_video_ids_per_message: int = 4
    enable_message_ordering: bool = True
    dry_run_limit: int = 0
    publish_batch_messages: int = 1000


def load_config(path: Path) -> tuple[WorkerPublishConfig, PublisherConfig]:
    raw = {}
    if path.exists() and path.stat().st_size > 0:
        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    worker_raw: Dict[str, Any] = raw.get("worker", {})
    worker = WorkerPublishConfig(
        project_id=str(worker_raw.get("project_id", "speechsdk")),
        subscription_id=str(worker_raw.get("subscription_id", "chatterbox-tts-sub")),
        topic_id=str(worker_raw.get("topic_id", "chatterbox-tts")),
        bucket=str(worker_raw.get("bucket", "tiktok-pipeline-processed")),
        clips_folder=str(worker_raw.get("clips_folder", "clips")),
        manifest_path=str(worker_raw.get("manifest_path", "../../pose_cond_npy_manifest.txt")),
    )
    publisher_raw: Dict[str, Any] = raw.get("publisher", {})
    publisher = PublisherConfig(
        project_id=str(publisher_raw.get("project_id", worker.project_id)),
        topic_id=str(publisher_raw.get("topic_id", worker.topic_id)),
        subscription_id=str(publisher_raw.get("subscription_id", worker.subscription_id)),
        manifest_path=str(publisher_raw.get("manifest_path", worker.manifest_path)),
        max_video_ids_per_message=int(publisher_raw.get("max_video_ids_per_message", 8)),
        min_video_ids_per_message=int(publisher_raw.get("min_video_ids_per_message", 4)),
        enable_message_ordering=bool(publisher_raw.get("enable_message_ordering", True)),
        dry_run_limit=int(publisher_raw.get("dry_run_limit", 0)),
        publish_batch_messages=int(publisher_raw.get("publish_batch_messages", 1000)),
    )
    return worker, publisher


def clip_id_from_pose_manifest_entry(line: str) -> str:
    name = Path(line.strip()).name
    if name.endswith("_poses.npy"):
        return name[: -len("_poses.npy")]
    if name.endswith(".mp4"):
        return name[: -len(".mp4")]
    if name.endswith(".json"):
        return name[: -len(".json")]
    return Path(name).stem


def video_id_from_clip_id(clip_id: str) -> str:
    video_id, sep, _ = clip_id.partition("_")
    if not sep:
        raise ValueError(f"Cannot derive video_id from clip_id: {clip_id}")
    return video_id


def clip_spec_from_id(worker: WorkerPublishConfig, clip_id: str) -> ClipSpec:
    video_id = video_id_from_clip_id(clip_id)
    base = f"gs://{worker.bucket}/{video_id}/{worker.clips_folder}/{clip_id}"
    return ClipSpec(video_id=video_id, clip_id=clip_id, mp4_uri=f"{base}.mp4", transcript_uri=f"{base}_transcript.json")


def read_manifest_groups(manifest_path: Path, worker: WorkerPublishConfig) -> "OrderedDict[str, List[dict]]":
    groups: "OrderedDict[str, List[dict]]" = OrderedDict()
    with manifest_path.open("r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            clip_id = clip_id_from_pose_manifest_entry(raw)
            spec = clip_spec_from_id(worker, clip_id)
            groups.setdefault(spec.video_id, []).append(spec.__dict__)
    return groups


def batch_video_groups(groups: "OrderedDict[str, List[dict]]", max_video_ids: int) -> Iterable[List[dict]]:
    batch: List[dict] = []
    for video_id, clips in groups.items():
        batch.append({"video_id": video_id, "clips": clips})
        if len(batch) >= max_video_ids:
            yield batch
            batch = []
    if batch:
        yield batch


def ensure_pubsub_resources(config: PublisherConfig) -> str:
    publisher = pubsub_v1.PublisherClient(
        publisher_options=pubsub_v1.types.PublisherOptions(
            enable_message_ordering=config.enable_message_ordering
        )
    )
    subscriber = pubsub_v1.SubscriberClient()
    topic_path = publisher.topic_path(config.project_id, config.topic_id)
    subscription_path = subscriber.subscription_path(config.project_id, config.subscription_id)

    retention = Duration(seconds=MAX_PUBSUB_RETENTION_SECONDS)
    try:
        topic = publisher.create_topic(
            request={
                "name": topic_path,
                "message_retention_duration": retention,
            }
        )
        print(f"Created topic {topic.name}")
    except AlreadyExists:
        try:
            publisher.update_topic(
                request={
                    "topic": {"name": topic_path, "message_retention_duration": retention},
                    "update_mask": FieldMask(paths=["message_retention_duration"]),
                }
            )
            print(f"Updated topic retention to {MAX_PUBSUB_RETENTION_SECONDS}s")
        except Exception as exc:
            print(f"Warning: could not update topic retention: {exc}")

    try:
        sub = subscriber.create_subscription(
            request={
                "name": subscription_path,
                "topic": topic_path,
                "ack_deadline_seconds": MAX_ACK_DEADLINE_SECONDS,
                "message_retention_duration": retention,
                "enable_message_ordering": config.enable_message_ordering,
            }
        )
        print(f"Created subscription {sub.name}")
    except AlreadyExists:
        try:
            subscriber.update_subscription(
                request={
                    "subscription": {
                        "name": subscription_path,
                        "ack_deadline_seconds": MAX_ACK_DEADLINE_SECONDS,
                        "message_retention_duration": retention,
                        "enable_message_ordering": config.enable_message_ordering,
                    },
                    "update_mask": FieldMask(
                        paths=["ack_deadline_seconds", "message_retention_duration", "enable_message_ordering"]
                    ),
                }
            )
            print(f"Updated subscription retention/ack deadline to Pub/Sub maximums")
        except Exception as exc:
            print(f"Warning: could not update subscription retention/ack settings: {exc}")

    return topic_path


def publish_batches(config: PublisherConfig, batches: Iterable[List[dict]], *, dry_run: bool) -> int:
    batch_settings = pubsub_v1.types.BatchSettings(
        max_bytes=10 * 1024 * 1024,
        max_latency=1.0,
        max_messages=max(1, config.publish_batch_messages),
    )
    publisher = pubsub_v1.PublisherClient(
        batch_settings=batch_settings,
        publisher_options=pubsub_v1.types.PublisherOptions(
            enable_message_ordering=config.enable_message_ordering
        )
    )
    topic_path = publisher.topic_path(config.project_id, config.topic_id)
    published = 0
    futures = []

    def flush_futures() -> None:
        nonlocal futures
        for future in futures:
            future.result()
        futures = []

    for batch in batches:
        payload = {
            "schema_version": 1,
            "videos": batch,
        }
        data = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        if dry_run:
            print(json.dumps(payload, indent=2)[:2000])
        else:
            ordering_key = batch[0]["video_id"] if config.enable_message_ordering else ""
            futures.append(publisher.publish(topic_path, data, ordering_key=ordering_key))
        published += 1
        if not dry_run and len(futures) >= max(1, config.publish_batch_messages):
            flush_futures()
        if config.dry_run_limit and published >= config.dry_run_limit:
            break

    if not dry_run:
        flush_futures()
    return published


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish clip-tokenization jobs grouped by video_id")
    parser.add_argument("--config", default="tokenization_config.yaml")
    parser.add_argument("--manifest", help="Override manifest path")
    parser.add_argument("--ensure-resources", action="store_true", help="Create/update topic/subscription retention settings")
    parser.add_argument("--dry-run", action="store_true", help="Print messages without publishing")
    parser.add_argument("--limit-messages", type=int, default=0, help="Publish/print at most this many Pub/Sub messages")
    parser.add_argument("--publish-batch-messages", type=int, help="Number of publish futures to queue before waiting")
    args = parser.parse_args()

    worker, publisher_config = load_config(Path(args.config))
    if args.manifest:
        publisher_config.manifest_path = args.manifest
    if args.limit_messages:
        publisher_config.dry_run_limit = args.limit_messages
    if args.publish_batch_messages:
        publisher_config.publish_batch_messages = args.publish_batch_messages

    manifest_path = Path(publisher_config.manifest_path)
    if not manifest_path.is_absolute():
        manifest_path = Path(__file__).resolve().parent / manifest_path
    manifest_path = manifest_path.resolve()

    groups = read_manifest_groups(manifest_path, worker)
    print(f"Loaded {sum(len(v) for v in groups.values())} clips across {len(groups)} video_ids")

    if publisher_config.max_video_ids_per_message < 1:
        raise ValueError("max_video_ids_per_message must be >= 1")
    if publisher_config.min_video_ids_per_message > publisher_config.max_video_ids_per_message:
        raise ValueError("min_video_ids_per_message cannot exceed max_video_ids_per_message")

    if args.ensure_resources and not args.dry_run:
        ensure_pubsub_resources(publisher_config)

    started = time.time()
    count = publish_batches(
        publisher_config,
        batch_video_groups(groups, publisher_config.max_video_ids_per_message),
        dry_run=args.dry_run,
    )
    elapsed = max(time.time() - started, 1e-6)
    verb = "Prepared" if args.dry_run else "Published"
    print(f"{verb} {count} messages in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
