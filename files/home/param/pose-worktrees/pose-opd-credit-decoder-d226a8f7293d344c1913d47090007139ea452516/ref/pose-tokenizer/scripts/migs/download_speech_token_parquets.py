"""Download speech-token parquet outputs for video_ids derived from a manifest."""

from __future__ import annotations

import argparse
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterator, List

from google.cloud import storage
from tqdm import tqdm


def clip_id_from_manifest_entry(line: str) -> str:
    name = Path(line.strip()).name
    if name.endswith("_poses.npy"):
        return name[: -len("_poses.npy")]
    if name.endswith(".mp4"):
        return name[: -len(".mp4")]
    if name.endswith(".parquet"):
        return Path(name).stem
    return Path(name).stem


def video_id_from_manifest_entry(line: str) -> str:
    value = line.strip()
    if not value:
        raise ValueError("empty manifest line")
    name = Path(value).name
    if "_" not in name:
        return Path(name).stem
    return clip_id_from_manifest_entry(name).split("_", 1)[0]


def read_video_ids(manifest: Path, limit: int = 0) -> List[str]:
    seen = set()
    video_ids: List[str] = []
    with manifest.open("r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            video_id = video_id_from_manifest_entry(raw)
            if video_id in seen:
                continue
            seen.add(video_id)
            video_ids.append(video_id)
            if limit and len(video_ids) >= limit:
                break
    return video_ids


def blob_name_for_video_id(video_id: str) -> str:
    return f"{video_id}/speech-tokens/{video_id}.parquet"


def local_path_for_video_id(
    bucket_name: str,
    video_id: str,
    output_root: Path,
) -> Path:
    return output_root / bucket_name / blob_name_for_video_id(video_id)


def is_not_found_error(exc: Exception) -> bool:
    status = getattr(exc, "code", None)
    return status == 404 or exc.__class__.__name__ == "NotFound"


def download_one(
    bucket: storage.Bucket,
    *,
    bucket_name: str,
    video_id: str,
    output_root: Path,
    skip_existing: bool,
    download_kwargs: dict,
) -> tuple[str, str, int, Exception | None]:
    local_path = local_path_for_video_id(bucket_name, video_id, output_root)
    if skip_existing and local_path.exists() and local_path.stat().st_size > 0:
        return video_id, "exists", 0, None

    local_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = local_path.with_suffix(local_path.suffix + ".tmp")
    try:
        bucket.blob(blob_name_for_video_id(video_id)).download_to_filename(
            str(tmp_path),
            **download_kwargs,
        )
        tmp_path.replace(local_path)
        return video_id, "downloaded", local_path.stat().st_size, None
    except Exception as exc:
        tmp_path.unlink(missing_ok=True)
        if is_not_found_error(exc):
            return video_id, "missing", 0, None
        return video_id, "error", 0, exc


def download_with_progress(
    video_ids: List[str],
    *,
    bucket: storage.Bucket,
    bucket_name: str,
    output_root: Path,
    workers: int,
    skip_existing: bool,
    download_kwargs: dict,
) -> Iterator[tuple[str, str, int, Exception | None]]:
    max_pending = max(workers * 4, workers)
    video_iter = iter(video_ids)
    pending: set[Future[tuple[str, str, int, Exception | None]]] = set()

    def submit_until_full(executor: ThreadPoolExecutor) -> None:
        while len(pending) < max_pending:
            try:
                video_id = next(video_iter)
            except StopIteration:
                return
            pending.add(
                executor.submit(
                    download_one,
                    bucket,
                    bucket_name=bucket_name,
                    video_id=video_id,
                    output_root=output_root,
                    skip_existing=skip_existing,
                    download_kwargs=download_kwargs,
                )
            )

    with ThreadPoolExecutor(max_workers=workers) as executor:
        submit_until_full(executor)
        while pending:
            for future in as_completed(pending):
                pending.remove(future)
                yield future.result()
                submit_until_full(executor)
                break


def main() -> None:
    parser = argparse.ArgumentParser(description="Download speech-token parquet files for manifest video_ids")
    parser.add_argument("--manifest", default="../../pose_cond_npy_manifest.txt")
    parser.add_argument("--bucket", default="tiktok-pipeline-processed")
    parser.add_argument("--output-root", default="gcs_data/downloaded_speech_tokens")
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true", default=True, help=argparse.SUPPRESS)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip non-empty local files. Off by default so downloads start immediately.",
    )
    parser.add_argument("--missing-out", default="missing_speech_token_parquets.txt")
    parser.add_argument(
        "--checksum",
        choices=["auto", "md5", "crc32c", "none"],
        default="auto",
        help="Checksum mode passed to blob downloads (default: auto)",
    )
    parser.add_argument(
        "--single-shot-download",
        action="store_true",
        help="Force each object download to use a single request.",
    )
    args = parser.parse_args()

    manifest = Path(args.manifest)
    if not manifest.is_absolute():
        manifest = Path(__file__).resolve().parent / manifest
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = Path(__file__).resolve().parent / output_root

    video_ids = read_video_ids(manifest.resolve(), limit=args.limit)
    print(f"Found {len(video_ids)} unique video_ids")

    client = storage.Client()
    bucket = client.bucket(args.bucket)
    counts = {"downloaded": 0, "exists": 0, "missing": 0, "error": 0}
    missing: List[str] = []
    downloaded_bytes = 0
    errors_shown = 0
    workers = max(1, args.workers)
    checksum = None if args.checksum == "none" else args.checksum
    download_kwargs = {
        "checksum": checksum,
        "single_shot_download": args.single_shot_download,
    }

    print(
        f"Downloading {len(video_ids)} parquets to {output_root}/ "
        f"({workers} workers, {'skip existing' if args.skip_existing else 'overwrite'})"
    )

    start = time.monotonic()
    progress = tqdm(total=len(video_ids), desc="Downloading parquets", unit="file")
    with progress:
        completed = 0
        for video_id, status, size, error in download_with_progress(
            video_ids,
            bucket=bucket,
            bucket_name=args.bucket,
            output_root=output_root,
            workers=workers,
            skip_existing=args.skip_existing,
            download_kwargs=download_kwargs,
        ):
            counts[status] += 1
            downloaded_bytes += size
            if status == "missing":
                missing.append(video_id)
            elif error is not None and errors_shown < 10:
                print(f"ERROR: {video_id}: {error!r}")
                errors_shown += 1

            completed += 1
            if completed % 128 == 0 or completed == len(video_ids):
                elapsed = time.monotonic() - start
                mb_s = (downloaded_bytes / (1024 * 1024)) / elapsed if elapsed > 0 else 0
                progress.set_postfix(
                    ok=counts["downloaded"],
                    missing=counts["missing"],
                    err=counts["error"],
                    mb_s=f"{mb_s:.0f}",
                    refresh=False,
                )
            progress.update(1)

    elapsed = time.monotonic() - start
    mb_s = (downloaded_bytes / (1024 * 1024)) / elapsed if elapsed > 0 else 0
    print(f"Downloaded in {elapsed:.1f}s (~{mb_s:.0f} MB/s)")

    missing_path = output_root / args.missing_out
    if missing:
        missing_path.parent.mkdir(parents=True, exist_ok=True)
        missing_path.write_text("\n".join(missing) + "\n", encoding="utf-8")
        print(f"Wrote missing list to {missing_path}")

    print("Done:", counts)
    print(f"Output root: {output_root}")


if __name__ == "__main__":
    main()
