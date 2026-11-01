"""Pub/Sub worker for tokenising TikTok clip audio with Mimi."""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import shutil
import sys
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import requests
import torch
import yaml
from datasets import Dataset, Features, LargeList, Value
from google.cloud import pubsub_v1, storage
from transformers import AutoFeatureExtractor, MimiModel
from tqdm import tqdm

warnings.filterwarnings("ignore", module="torchaudio")

try:
    import health_check as healthz
except Exception:
    class _NoopHealth:
        def start(self, **_: Any) -> None:
            pass

        def set_ready(self, *_: Any, **__: Any) -> None:
            pass

        def set_processing(self, *_: Any, **__: Any) -> None:
            pass

        def note_activity(self, *_: Any, **__: Any) -> None:
            pass

        def fail(self, message: str) -> None:
            print(f"HEALTH FAIL: {message}", file=sys.stderr)

    healthz = _NoopHealth()


TARGET_SR = 24_000
MODEL_ID = "kyutai/mimi"
PUBSUB_MAX_ACK_DEADLINE_SECONDS = 600

global_model: Optional["MimiTokenizer"] = None
global_config: Optional["WorkerConfig"] = None


@dataclass
class WorkerConfig:
    project_id: str = "speechsdk"
    subscription_id: str = "chatterbox-tts-sub"
    topic_id: str = "chatterbox-tts"
    bucket: str = "tiktok-pipeline-processed"
    clips_folder: str = "clips"
    output_folder: str = "speech-tokens"
    output_filename_template: str = "{video_id}.parquet"
    manifest_path: str = "../../pose_cond_npy_manifest.txt"
    max_duration_seconds: float = 30.0
    native_duration_threshold_seconds: float = 30.0
    min_chunk_duration_seconds: float = 5.0
    boundary_search_seconds: float = 4.0
    random_boundary_jitter_seconds: float = 0.0
    random_seed: int = 17
    download_workers: int = 16
    local_work_dir: str = "gcs_data/speech_tokenizer"
    log_interval: int = 25
    ack_deadline_seconds: int = PUBSUB_MAX_ACK_DEADLINE_SECONDS
    lease_heartbeat_seconds: int = 60
    max_lease_duration_seconds: int = 12 * 3600
    health_port: int = 8090
    mimi_model_id: str = MODEL_ID
    mimi_local_files_only: bool = False

    @classmethod
    def from_yaml(cls, path: Path) -> "WorkerConfig":
        data = _load_yaml(path)
        worker = data.get("worker", data)
        return cls(**{k: v for k, v in worker.items() if k in cls.__dataclass_fields__})


@dataclass
class ClipSpec:
    video_id: str
    clip_id: str
    mp4_uri: str
    transcript_uri: str


@dataclass
class LocalClip:
    spec: ClipSpec
    mp4_path: Path
    transcript_path: Optional[Path]
    transcript: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChunkSpec:
    video_id: str
    clip_id: str
    clip_index: int
    chunk_index: int
    chunk_id: str
    clip_uri: str
    transcript_uri: str
    start_seconds: float
    end_seconds: float
    words: List[Dict[str, Any]]
    text: str


@dataclass
class MimiTokenizer:
    model: MimiModel
    feature_extractor: Any
    device: str
    sampling_rate: int


def heartbeat(tag: str = "") -> None:
    try:
        healthz.note_activity(tag)
    except Exception:
        pass


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _parse_gcs_uri(uri: str) -> Tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError(f"Expected gs:// URI, got: {uri}")
    bucket, _, name = uri[5:].partition("/")
    if not bucket or not name:
        raise ValueError(f"Invalid GCS URI: {uri}")
    return bucket, name


def _clip_id_from_pose_manifest_entry(line: str) -> str:
    name = Path(line.strip()).name
    if name.endswith("_poses.npy"):
        return name[: -len("_poses.npy")]
    if name.endswith(".mp4"):
        return name[: -len(".mp4")]
    if name.endswith(".json"):
        return name[: -len(".json")]
    return Path(name).stem


def _video_id_from_clip_id(clip_id: str) -> str:
    video_id, sep, _ = clip_id.partition("_")
    if not sep:
        raise ValueError(f"Cannot derive video_id from clip_id: {clip_id}")
    return video_id


def clip_spec_from_id(config: WorkerConfig, clip_id: str) -> ClipSpec:
    video_id = _video_id_from_clip_id(clip_id)
    base = f"gs://{config.bucket}/{video_id}/{config.clips_folder}/{clip_id}"
    return ClipSpec(video_id, clip_id, f"{base}.mp4", f"{base}_transcript.json")


def _manifest_path(config: WorkerConfig) -> Path:
    path = Path(config.manifest_path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    return path.resolve()


def load_manifest_clip_groups(config: WorkerConfig) -> Dict[str, List[ClipSpec]]:
    groups: Dict[str, List[ClipSpec]] = {}
    with _manifest_path(config).open("r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            spec = clip_spec_from_id(config, _clip_id_from_pose_manifest_entry(raw))
            groups.setdefault(spec.video_id, []).append(spec)
    return groups


def _normalise_message(payload: Dict[str, Any], config: WorkerConfig) -> List[Tuple[str, List[ClipSpec]]]:
    if "videos" in payload:
        groups = []
        for video in payload["videos"]:
            video_id = str(video["video_id"])
            clips = []
            for clip in video.get("clips", []):
                clip_id = str(clip["clip_id"])
                default = clip_spec_from_id(config, clip_id)
                clips.append(
                    ClipSpec(
                        video_id=str(clip.get("video_id") or video_id),
                        clip_id=clip_id,
                        mp4_uri=str(clip.get("mp4_uri") or default.mp4_uri),
                        transcript_uri=str(clip.get("transcript_uri") or default.transcript_uri),
                    )
                )
            groups.append((video_id, clips))
        return groups
    if "video_ids" in payload:
        manifest_groups = load_manifest_clip_groups(config)
        return [(str(video_id), manifest_groups.get(str(video_id), [])) for video_id in payload["video_ids"]]
    if "clip_ids" in payload:
        by_video: Dict[str, List[ClipSpec]] = {}
        for clip_id in payload["clip_ids"]:
            spec = clip_spec_from_id(config, str(clip_id))
            by_video.setdefault(spec.video_id, []).append(spec)
        return sorted(by_video.items())
    raise ValueError("Message must contain 'videos', 'video_ids', or 'clip_ids'")


def _download_blob(client: storage.Client, uri: str, dest: Path, *, required: bool) -> Optional[Path]:
    bucket_name, blob_name = _parse_gcs_uri(uri)
    blob = client.bucket(bucket_name).blob(blob_name)
    if not blob.exists():
        if required:
            raise FileNotFoundError(f"Missing required object: {uri}")
        print(f"Warning: optional object not found: {uri}")
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(str(dest))
    heartbeat("gcs_download")
    return dest


def _read_json(path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        print(f"Warning: failed to read transcript {path}: {exc}")
        return {}


def download_video_clips(config: WorkerConfig, video_id: str, clips: Sequence[ClipSpec], root: Path) -> List[LocalClip]:
    client = storage.Client()
    video_root = root / video_id
    if video_root.exists():
        shutil.rmtree(video_root)
    video_root.mkdir(parents=True, exist_ok=True)

    def fetch_clip(spec: ClipSpec) -> LocalClip:
        mp4_path = video_root / f"{spec.clip_id}.mp4"
        json_path = video_root / f"{spec.clip_id}.json"
        _download_blob(client, spec.mp4_uri, mp4_path, required=True)
        transcript_path = _download_blob(client, spec.transcript_uri, json_path, required=False)
        return LocalClip(spec, mp4_path, transcript_path, _read_json(transcript_path))

    local_by_id: Dict[str, LocalClip] = {}
    with ThreadPoolExecutor(max_workers=max(1, config.download_workers)) as executor:
        future_to_spec = {executor.submit(fetch_clip, spec): spec for spec in clips}
        for future in tqdm(as_completed(future_to_spec), total=len(future_to_spec), desc=f"Downloading {video_id}"):
            spec = future_to_spec[future]
            local_by_id[spec.clip_id] = future.result()
            print(f"Downloaded {spec.clip_id}")
    return [local_by_id[spec.clip_id] for spec in clips]


def decode_audio(path: Path) -> np.ndarray:
    from torchcodec.decoders import AudioDecoder

    decoder = AudioDecoder(str(path), sample_rate=TARGET_SR)
    samples = decoder.get_all_samples()
    audio = samples.data.cpu().numpy()
    if audio.ndim > 1:
        audio = np.mean(audio, axis=tuple(range(audio.ndim - 1)))
    return np.ascontiguousarray(audio, dtype=np.float32)


def _extract_words(transcript: Dict[str, Any]) -> List[Dict[str, Any]]:
    words = transcript.get("words") or transcript.get("word_timestamps")
    if words is None and isinstance(transcript.get("segments"), list):
        words = []
        for segment in transcript["segments"]:
            words.extend(segment.get("words") or [])
    if not isinstance(words, list):
        return []
    clean = []
    for word in words:
        if not isinstance(word, dict):
            continue
        try:
            start = float(word.get("start", word.get("start_time")))
            end = float(word.get("end", word.get("end_time", start)))
        except (TypeError, ValueError):
            continue
        item = dict(word)
        item["start"] = start
        item["end"] = max(start, end)
        clean.append(item)
    return sorted(clean, key=lambda item: (float(item["start"]), float(item["end"])))


def _words_for_window(words: Sequence[Dict[str, Any]], start: float, end: float) -> Tuple[List[Dict[str, Any]], str]:
    chunk_words = []
    for word in words:
        ws = float(word["start"])
        we = float(word["end"])
        if we <= start or ws >= end:
            continue
        rebased = dict(word)
        rebased["start"] = max(0.0, ws - start)
        rebased["end"] = max(0.0, min(end, we) - start)
        rebased["clip_start"] = ws
        rebased["clip_end"] = we
        chunk_words.append(rebased)
    text = " ".join(str(w.get("word", w.get("text", ""))).strip() for w in chunk_words).strip()
    return chunk_words, text


def choose_chunk_boundaries(duration: float, words: Sequence[Dict[str, Any]], config: WorkerConfig) -> List[Tuple[float, float]]:
    max_duration = float(config.max_duration_seconds)
    if duration <= min(max_duration, float(config.native_duration_threshold_seconds)):
        return [(0.0, duration)]

    n_chunks = int(math.ceil(duration / max_duration))
    target_len = duration / n_chunks
    rng = random.Random(config.random_seed)
    boundaries = [0.0]
    word_boundaries = sorted({float(w["end"]) for w in words if 0.0 < float(w.get("end", 0.0)) < duration})

    for idx in range(1, n_chunks):
        remaining = n_chunks - idx
        prev = boundaries[-1]
        target = idx * target_len
        if config.random_boundary_jitter_seconds > 0:
            target += rng.uniform(-config.random_boundary_jitter_seconds, config.random_boundary_jitter_seconds)

        lower = max(prev + config.min_chunk_duration_seconds, duration - remaining * max_duration)
        upper = min(prev + max_duration, duration - remaining * config.min_chunk_duration_seconds)
        if lower > upper:
            lower = prev + min(max_duration, max(0.0, duration - prev) / (remaining + 1))
            upper = min(prev + max_duration, duration)

        search = max(0.0, config.boundary_search_seconds)
        candidates = [b for b in word_boundaries if lower <= b <= upper and (search == 0.0 or abs(b - target) <= search)]
        if not candidates:
            candidates = [b for b in word_boundaries if lower <= b <= upper]
        boundary = min(candidates, key=lambda b: abs(b - target)) if candidates else min(max(target, lower), upper)
        boundaries.append(float(boundary))

    boundaries.append(duration)
    return [(round(s, 4), round(e, 4)) for s, e in zip(boundaries, boundaries[1:]) if e - s > 0.01]


def build_chunks(local_clip: LocalClip, clip_index: int, audio: np.ndarray, config: WorkerConfig) -> List[Tuple[ChunkSpec, np.ndarray]]:
    duration = float(len(audio)) / float(TARGET_SR)
    words = _extract_words(local_clip.transcript)
    chunks = []
    for chunk_index, (start, end) in enumerate(choose_chunk_boundaries(duration, words, config)):
        sample_start = int(round(start * TARGET_SR))
        sample_end = int(round(end * TARGET_SR))
        chunk_words, text = _words_for_window(words, start, end)
        chunk_id = f"{local_clip.spec.clip_id}_chunk_{chunk_index:03d}"
        chunks.append(
            (
                ChunkSpec(
                    video_id=local_clip.spec.video_id,
                    clip_id=local_clip.spec.clip_id,
                    clip_index=clip_index,
                    chunk_index=chunk_index,
                    chunk_id=chunk_id,
                    clip_uri=local_clip.spec.mp4_uri,
                    transcript_uri=local_clip.spec.transcript_uri,
                    start_seconds=start,
                    end_seconds=end,
                    words=chunk_words,
                    text=text,
                ),
                np.ascontiguousarray(audio[sample_start:sample_end], dtype=np.float32),
            )
        )
    return chunks


def tokenise_audio(waveform: np.ndarray, tokenizer: MimiTokenizer) -> List[List[int]]:
    inputs = tokenizer.feature_extractor(
        raw_audio=waveform,
        sampling_rate=tokenizer.sampling_rate,
        return_tensors="pt",
    )
    input_values = inputs["input_values"].to(tokenizer.device)
    with torch.inference_mode():
        encoder_outputs = tokenizer.model.encode(input_values)

    codes = encoder_outputs.audio_codes.detach().cpu()
    while codes.ndim > 2:
        codes = codes[0]
    codebooks = codes.to(torch.int32).tolist()
    del input_values, encoder_outputs, codes
    return [[int(v) for v in codebook] for codebook in codebooks]


def _row_from_codes(chunk: ChunkSpec, mimi_codes: List[List[int]]) -> Dict[str, Any]:
    word_texts = [str(w.get("word", w.get("text", ""))) for w in chunk.words]
    word_start_seconds = [float(w.get("start", 0.0)) for w in chunk.words]
    word_end_seconds = [float(w.get("end", 0.0)) for w in chunk.words]
    word_clip_start_seconds = [float(w.get("clip_start", 0.0)) for w in chunk.words]
    word_clip_end_seconds = [float(w.get("clip_end", 0.0)) for w in chunk.words]
    row: Dict[str, Any] = {
        "row_id": f"{chunk.clip_id}_chunk_{chunk.chunk_index:03d}",
        "video_id": chunk.video_id,
        "clip_id": chunk.clip_id,
        "clip_index": chunk.clip_index,
        "chunk_index": chunk.chunk_index,
        "chunk_id": chunk.chunk_id,
        "chunk_start_seconds": float(chunk.start_seconds),
        "chunk_end_seconds": float(chunk.end_seconds),
        "chunk_duration_seconds": float(chunk.end_seconds - chunk.start_seconds),
        "mimi_codes": mimi_codes,
        "num_frames": len(mimi_codes[0]) if mimi_codes else 0,
        "word_texts": word_texts,
        "word_start_seconds": word_start_seconds,
        "word_end_seconds": word_end_seconds,
        "word_clip_start_seconds": word_clip_start_seconds,
        "word_clip_end_seconds": word_clip_end_seconds,
    }
    return row


def write_parquet(rows: List[Dict[str, Any]], path: Path) -> None:
    features = {
        "row_id": Value("string"),
        "video_id": Value("string"),
        "clip_id": Value("string"),
        "clip_index": Value("int32"),
        "chunk_index": Value("int32"),
        "chunk_id": Value("string"),
        "chunk_start_seconds": Value("float32"),
        "chunk_end_seconds": Value("float32"),
        "chunk_duration_seconds": Value("float32"),
        "mimi_codes": LargeList(LargeList(Value("int32"))),
        "num_frames": Value("int32"),
        "word_texts": LargeList(Value("string")),
        "word_start_seconds": LargeList(Value("float32")),
        "word_end_seconds": LargeList(Value("float32")),
        "word_clip_start_seconds": LargeList(Value("float32")),
        "word_clip_end_seconds": LargeList(Value("float32")),
    }
    columns = {name: [row[name] for row in rows] for name in features}
    dataset = Dataset.from_dict(columns, features=Features(features))
    heartbeat("saving_parquet")
    dataset.to_parquet(str(path))
    print(f"Wrote {len(rows)} rows to {path}")


def destination_blob_name(config: WorkerConfig, video_id: str) -> str:
    filename = config.output_filename_template.format(video_id=video_id)
    return f"{video_id}/{config.output_folder}/{filename}"


def upload_parquet(config: WorkerConfig, video_id: str, local_path: Path) -> str:
    blob_name = destination_blob_name(config, video_id)
    storage.Client().bucket(config.bucket).blob(blob_name).upload_from_filename(str(local_path))
    uri = f"gs://{config.bucket}/{blob_name}"
    print(f"Uploaded {uri}")
    heartbeat("upload_parquet")
    return uri


def mirror_parquet_locally(config: WorkerConfig, video_id: str, local_path: Path, local_output_root: Path) -> Path:
    mirrored_path = local_output_root / config.bucket / destination_blob_name(config, video_id)
    mirrored_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(local_path, mirrored_path)
    print(f"Saved local GCS mirror to {mirrored_path}")
    return mirrored_path


def encode_video(
    config: WorkerConfig,
    video_id: str,
    clips: Sequence[ClipSpec],
    model: MimiTokenizer,
    work_root: Path,
    *,
    upload: bool = True,
    local_output_root: Optional[Path] = None,
) -> int:
    local_clips = download_video_clips(config, video_id, clips, work_root / "downloads")
    rows: List[Dict[str, Any]] = []
    started = time.time()
    for clip_index, local_clip in enumerate(local_clips):
        heartbeat("decode_audio")
        audio = decode_audio(local_clip.mp4_path)
        chunks = build_chunks(local_clip, clip_index, audio, config)
        print(f"{local_clip.spec.clip_id}: {len(chunks)} chunks from {len(audio) / TARGET_SR:.2f}s")
        for chunk_spec, chunk_audio in tqdm(chunks, desc=f"Encoding {local_clip.spec.clip_id}"):
            mimi_codes = tokenise_audio(chunk_audio, model)
            rows.append(_row_from_codes(chunk_spec, mimi_codes))
            heartbeat("encoding")
            if len(rows) % config.log_interval == 0:
                elapsed = max(time.time() - started, 1e-6)
                print(f"{video_id}: encoded {len(rows)} chunks ({len(rows) / elapsed:.2f} chunks/s)")
        del audio
        gc.collect()
    if not rows:
        raise RuntimeError(f"No rows produced for {video_id}")
    local_parquet = work_root / "parquet" / f"{video_id}.parquet"
    local_parquet.parent.mkdir(parents=True, exist_ok=True)
    write_parquet(rows, local_parquet)
    if local_output_root is not None:
        mirror_parquet_locally(config, video_id, local_parquet, local_output_root)
    if upload:
        upload_parquet(config, video_id, local_parquet)
    torch.cuda.empty_cache()
    return len(rows)


def load_model(config: WorkerConfig) -> MimiTokenizer:
    print(f"Loading Mimi model: {config.mimi_model_id}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        config.mimi_model_id,
        local_files_only=config.mimi_local_files_only,
    )
    model = MimiModel.from_pretrained(
        config.mimi_model_id,
        local_files_only=config.mimi_local_files_only,
    ).to(device)
    model.eval()
    sampling_rate = int(getattr(feature_extractor, "sampling_rate", TARGET_SR))
    if sampling_rate != TARGET_SR:
        raise ValueError(f"Expected Mimi sampling rate {TARGET_SR}, got {sampling_rate}")
    print(f"Mimi model loaded on {device}")
    return MimiTokenizer(model=model, feature_extractor=feature_extractor, device=device, sampling_rate=sampling_rate)


def get_instance_info() -> Dict[str, Optional[str]]:
    headers = {"Metadata-Flavor": "Google"}

    def get_metadata(path: str) -> Optional[str]:
        try:
            return requests.get(f"http://metadata.google.internal/computeMetadata/v1/{path}", headers=headers, timeout=2).text
        except Exception:
            return None

    zone_path = get_metadata("instance/zone") or ""
    created_by = get_metadata("instance/attributes/created-by") or ""
    return {
        "instance_name": get_metadata("instance/name"),
        "zone": zone_path.split("/")[-1] if zone_path else None,
        "mig_name": created_by.split("instanceGroupManagers/")[-1] if "instanceGroupManagers/" in created_by else None,
    }


def process_payload(
    payload: Dict[str, Any],
    config: WorkerConfig,
    model: MimiTokenizer,
    *,
    upload: bool = True,
    local_output_root: Optional[Path] = None,
) -> Dict[str, int]:
    work_root = Path(config.local_work_dir)
    work_root.mkdir(parents=True, exist_ok=True)
    results: Dict[str, int] = {}
    for video_id, clips in _normalise_message(payload, config):
        print(f"Processing video_id={video_id} clips={len(clips)}")
        if not clips:
            raise ValueError(f"No clips found for video_id={video_id}")
        results[video_id] = encode_video(
            config,
            video_id,
            clips,
            model,
            work_root,
            upload=upload,
            local_output_root=local_output_root,
        )
    return results


def process_message(message: pubsub_v1.subscriber.message.Message) -> None:
    msg_id = message.message_id
    lease_stop = threading.Event()
    print(f"[msg_id={msg_id}] delivery_attempt={getattr(message, 'delivery_attempt', None)}")
    assert global_config is not None and global_model is not None
    config = global_config

    try:
        message.modify_ack_deadline(min(config.ack_deadline_seconds, PUBSUB_MAX_ACK_DEADLINE_SECONDS))
    except Exception as exc:
        print(f"[msg_id={msg_id}] WARNING: initial ack extension failed: {exc}")

    def lease_heartbeat() -> None:
        while not lease_stop.wait(timeout=max(1, config.lease_heartbeat_seconds)):
            try:
                message.modify_ack_deadline(min(config.ack_deadline_seconds, PUBSUB_MAX_ACK_DEADLINE_SECONDS))
                heartbeat("lease_extend")
            except Exception as exc:
                print(f"[msg_id={msg_id}] WARNING: lease extension failed: {exc}")

    lease_thread = threading.Thread(target=lease_heartbeat, daemon=True)
    lease_thread.start()
    try:
        healthz.set_processing(True)
        payload = json.loads(message.data.decode("utf-8") if isinstance(message.data, bytes) else message.data)
        results = process_payload(payload, config, global_model)
        print(f"[msg_id={msg_id}] complete: {results}")
        message.ack()
    except Exception as exc:
        print(f"[msg_id={msg_id}] failed: {exc}", file=sys.stderr)
        message.ack()
        healthz.fail(f"message failed: {exc}")
    finally:
        lease_stop.set()
        lease_thread.join(timeout=5)
        healthz.set_processing(False)


def run_pubsub(config: WorkerConfig) -> None:
    global global_config, global_model
    global_config = config
    healthz.start(port=config.health_port, min_mem_mb=1024, idle_grace_s=120, idle_timeout_s=config.max_lease_duration_seconds)
    healthz.set_ready(False, "loading_model")
    global_model = load_model(config)
    healthz.set_ready(True, "subscriber_running")
    print(f"Instance info: {get_instance_info()}")

    subscriber = pubsub_v1.SubscriberClient()
    sub_path = subscriber.subscription_path(config.project_id, config.subscription_id)
    flow_control = pubsub_v1.types.FlowControl(max_messages=1, max_lease_duration=config.max_lease_duration_seconds)
    future = subscriber.subscribe(sub_path, callback=process_message, flow_control=flow_control)
    print(f"Subscriber started: {sub_path}")
    try:
        future.result()
    except KeyboardInterrupt:
        future.cancel()


def run_test(args: argparse.Namespace, config: WorkerConfig) -> None:
    model = load_model(config)
    if args.message_json:
        payload = json.loads(Path(args.message_json).read_text(encoding="utf-8"))
    elif args.clip_ids:
        by_video: Dict[str, List[dict]] = {}
        for clip_id in args.clip_ids:
            spec = clip_spec_from_id(config, clip_id)
            by_video.setdefault(spec.video_id, []).append(spec.__dict__)
        payload = {"videos": [{"video_id": video_id, "clips": clips} for video_id, clips in by_video.items()]}
    elif args.clip_id:
        spec = clip_spec_from_id(config, args.clip_id)
        payload = {"videos": [{"video_id": spec.video_id, "clips": [spec.__dict__]}]}
    else:
        raise ValueError("--test requires --message_json, --clip_id, or --clip_ids")
    output_root = Path(args.local_output_root).resolve() if args.no_upload else None
    print(f"[TEST] complete: {process_payload(payload, config, model, upload=not args.no_upload, local_output_root=output_root)}")


def run_manifest_example(args: argparse.Namespace, config: WorkerConfig) -> None:
    groups = load_manifest_clip_groups(config)
    if args.video_id:
        video_id = args.video_id
    else:
        manifest_lines = [
            line.strip()
            for line in _manifest_path(config).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not manifest_lines:
            raise ValueError(f"Manifest is empty: {_manifest_path(config)}")
        if args.manifest_index < 0 or args.manifest_index >= len(manifest_lines):
            raise ValueError(f"--manifest_index must be between 0 and {len(manifest_lines) - 1}")
        clip_id = _clip_id_from_pose_manifest_entry(manifest_lines[args.manifest_index])
        video_id = _video_id_from_clip_id(clip_id)

    clips = groups.get(video_id)
    if not clips:
        raise ValueError(f"No manifest clips found for video_id={video_id}")
    if args.limit_clips:
        clips = clips[: args.limit_clips]

    output_root = Path(args.local_output_root).resolve()
    payload = {"videos": [{"video_id": video_id, "clips": [clip.__dict__ for clip in clips]}]}
    print(f"[EXAMPLE] video_id={video_id} clips={len(clips)}")
    print(f"[EXAMPLE] local mirror root={output_root}")
    model = load_model(config)
    results = process_payload(payload, config, model, upload=False, local_output_root=output_root)
    print(f"[EXAMPLE] complete: {results}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tokenise TikTok clip audio from Pub/Sub messages")
    parser.add_argument("--config", default="tokenization_config.yaml")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--message_json")
    parser.add_argument("--clip_id")
    parser.add_argument("--clip_ids", nargs="+")
    parser.add_argument("--no_upload", action="store_true")
    parser.add_argument("--example_from_manifest", action="store_true")
    parser.add_argument("--manifest_index", type=int, default=0)
    parser.add_argument("--video_id")
    parser.add_argument("--limit_clips", type=int, default=0)
    parser.add_argument("--local_output_root", default="gcs_data/local_mirror")
    args = parser.parse_args()
    config = WorkerConfig.from_yaml(Path(args.config))
    if args.example_from_manifest:
        run_manifest_example(args, config)
    elif args.test:
        run_test(args, config)
    else:
        run_pubsub(config)


if __name__ == "__main__":
    main()
