"""Download and verify only model tensors from the final OPD HF checkpoint."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
from huggingface_hub import HfApi, hf_hub_download

REPO = "InternalCan/ppm-opd-repro-slinky"
REVISION = "188691b40d2962892431bd765e6179e3dd471579"
CHECKPOINT = "checkpoint-1793330493092033558-7e901371585b"
PREFIX = "runs/opd-nextrun-b384-v3/rolling/" + CHECKPOINT
MANIFEST_SHA = "818ba3bccc3b648ff407214471711f007f43c4c8ba94f4d5d7593ff8d23b9ca7"
STAGE = Path("/home/param/models/imported-weights/.next-opd-v110-download")
DEST = Path("/home/param/models/imported-weights/next-opd-v110")
REPORT = Path("/home/param/outputs/opd-next-sft-preparation/transfer-status.json")
STARTED = time.time()

def report(status, **kwargs):
    data = dict(status=status, elapsed_seconds=round(time.time()-STARTED, 1),
                repo=REPO, revision=REVISION, destination=str(DEST), **kwargs)
    tmp = REPORT.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2)+"\n")
    tmp.replace(REPORT)
    print(json.dumps(data), flush=True)

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(16 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def main():
    if DEST.exists():
        raise RuntimeError(f"Refusing to overwrite existing destination: {DEST}")
    api = HfApi()
    info = api.repo_info(REPO, revision=REVISION)
    assert info.private and info.sha == REVISION
    files = [x for x in api.list_repo_tree(REPO, path_in_repo=PREFIX, revision=REVISION, recursive=True)
             if hasattr(x, "size") and (x.path.startswith(PREFIX+"/model/") or x.path in (PREFIX+"/MANIFEST.json", PREFIX+"/COMPLETE"))]
    model = [x for x in files if x.path.startswith(PREFIX+"/model/")]
    assert len(model) == 29 and len(files) == 31
    assert sum(x.size for x in model) == 156529359638
    total = sum(x.size for x in files)
    assert os.statvfs(DEST.parent).f_bavail * os.statvfs(DEST.parent).f_frsize > total + 50 * 1024**3
    completed = 0
    downloaded = 0
    report("downloading", files_total=len(files), bytes_total=total, files_complete=0)
    def download(item):
        path = Path(hf_hub_download(REPO, item.path, revision=REVISION, local_dir=STAGE))
        assert path.stat().st_size == item.size, item.path
        return item, path
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        pending = {pool.submit(download, item): item for item in files}
        while pending:
            done, _ = concurrent.futures.wait(pending, timeout=30, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                item, path = future.result()
                results.append((item, path))
                completed += 1
                downloaded += item.size
                pending.pop(future)
            report("downloading", files_total=len(files), bytes_total=total,
                   files_complete=completed, bytes_in_completed_files=downloaded)
    root = STAGE / PREFIX
    assert sha256(root / "MANIFEST.json") == MANIFEST_SHA
    manifest = json.loads((root / "MANIFEST.json").read_text())
    assert manifest["checkpoint_id"] == CHECKPOINT and str(manifest["student_version"]) == "110"
    assert json.loads((root / "COMPLETE").read_text())["manifest_sha256"] == MANIFEST_SHA
    assert not (root / "optimizer").exists() and not (root / "run").exists()
    report("verifying_sha256", files_total=len(files), bytes_total=total)
    verified = []
    def verify(pair):
        item, path = pair
        digest = sha256(path)
        if item.lfs:
            assert digest == item.lfs.sha256, (item.path, "HF SHA256 mismatch")
        else:
            data = path.read_bytes()
            git_digest = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
            assert git_digest == item.blob_id, (item.path, "HF Git blob mismatch")
        return {"path": item.path[len(PREFIX)+1:], "bytes": item.size, "sha256": digest}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for entry in pool.map(verify, results):
            verified.append(entry)
            report("verifying_sha256", files_total=len(files), files_verified=len(verified), bytes_total=total)
    receipt = dict(format="opd-model-only-hf-download-v1", repo_id=REPO, revision=REVISION,
                   remote_prefix=PREFIX, checkpoint_id=CHECKPOINT, student_version=110,
                   manifest_sha256=MANIFEST_SHA, model_only=True, optimizer_downloaded=False,
                   source_manifest_scope="Original full-resume provenance; only model files downloaded",
                   verified_at_unix=time.time(), files=sorted(verified, key=lambda x:x["path"]))
    (root / "DOWNLOAD_VERIFIED.json").write_text(json.dumps(receipt, indent=2)+"\n")
    root.rename(DEST)
    report("complete", files_total=len(files), files_verified=len(verified), bytes_total=total,
           model_path=str(DEST/"model"), optimizer_downloaded=False)

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        report("failed", error=f"{type(exc).__name__}: {exc}")
        raise
