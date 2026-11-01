#!/usr/bin/env python3
"""Run the selected held-out clips on landsend-0/1, one PPM per GPU pair.

Successful outputs are gathered on landsend-0 and receive PPM_COMPLETE.json
only after the subprocess exits and audio/code validation passes.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime
import json
import os
from pathlib import Path
import queue
import shlex
import socket
import subprocess
import threading
import time

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parents[2]
KEY = "/home/param/.ssh/id_ed25519_cluster"
SSH = ["ssh", "-i", KEY, "-o", "BatchMode=yes", "-o", "ConnectTimeout=15"]
RECIPES = ("aligned-media", "compact-all", "xabi-original")


def remote(host, command, **kwargs):
    argv = ["bash", "-lc", command] if host == socket.gethostname().split(".")[0] else SSH + [host, command]
    return subprocess.run(argv, check=True, **kwargs)


def validate_output(path, clip, recipe, seed):
    meta = json.loads((path / "meta.json").read_text())
    assert meta["recipe"] == recipe and meta["clip_index"] == clip["clip_index"]
    assert meta["frames"] == clip["frames"] and meta["seed"] == seed and meta["prefill_frames"] == 0
    assert not meta["ground_truth_generation_media"]
    with np.load(path / "tokens.npz", allow_pickle=False) as z:
        for name in ("audio_codes", "pose_codes"):
            x = z[name]
            assert x.shape == (32, clip["frames"]) and x.min() >= 0 and x.max() < 2048
        assert z["lane"].shape == (clip["frames"],)
        if recipe != "compact-all":
            assert np.array_equal(z["lane"], np.load(clip["lane"], allow_pickle=False))
    audio, sr = sf.read(path / "audio.wav")
    assert sr == 24000 and len(audio) == clip["frames"] * 1920 and np.isfinite(audio).all()
    for name in ("mimi.safetensors", "pose_tokens.safetensors", "words.json"):
        assert (path / name).stat().st_size > 0
    return dict(frames=clip["frames"], audio_samples=len(audio), audio_rate=sr,
                generation_seconds=meta["generation_seconds"], wall_seconds=meta["wall_seconds"])


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--selection", type=Path, required=True)
    p.add_argument("--recipes", nargs="+", choices=RECIPES, default=list(RECIPES))
    p.add_argument("--seed", type=int, default=20261028)
    p.add_argument("--nodes", nargs="+", choices=("landsend-0", "landsend-1"), default=["landsend-0", "landsend-1"])
    a = p.parse_args()
    if socket.gethostname().split(".")[0] != "landsend-0" or os.getuid() != 1003:
        raise RuntimeError("Launch this comparison controller as param UID1003 on landsend-0")
    root = a.selection.resolve().parent
    clips = json.loads(a.selection.read_text())["clips"]
    jobs = queue.Queue()
    for recipe in a.recipes:
        for i, clip in enumerate(clips):
            jobs.put((clip, recipe, a.seed + i))
    logs = root / "logs"
    logs.mkdir(exist_ok=True)
    lock, failures = threading.Lock(), []
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    events = (root / "ppm-events.jsonl").open("a", buffering=1)

    def event(state, **data):
        rec = dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(), state=state, **data)
        with lock:
            events.write(json.dumps(rec) + "\n")
            print(json.dumps(rec), flush=True)

    def slot(host, pair):
        while True:
            try:
                clip, recipe, seed = jobs.get_nowait()
            except queue.Empty:
                return
            name = f"{clip['gallery_id']}-{recipe}"
            dest = root / "outputs" / clip["gallery_id"] / recipe / "ppm"
            sentinel = dest / "PPM_COMPLETE.json"
            try:
                if sentinel.exists():
                    validate_output(dest, clip, recipe, seed)
                    event("already_complete", name=name)
                    continue
                # Refuse occupied GPUs; this controller never kills another job.
                guard = "test $(id -u) = 1003 && test -z \"$(nvidia-smi -i " + pair + " --query-compute-apps=pid --format=csv,noheader)\""
                remote(host, guard, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                ckpt = Path("/home/param/models/transcript-pose") / recipe / "export-200"
                remote(host, "test -f " + shlex.quote(str(ckpt / "COMPLETE")), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if recipe == "xabi-original" and not (root / "baseline-gpu-PASS.json").exists():
                    raise RuntimeError("Baseline GPU validation has not completed")
                dest.parent.mkdir(parents=True, exist_ok=True)
                if dest.exists() and any(dest.iterdir()):
                    raise RuntimeError(f"Incomplete existing output requires inspection: {dest}")
                args = [str(REPO / ".venv/bin/python"), str(REPO / "infer/transcript_pose/infer.py"),
                        "--recipe", recipe, "--clip-index", str(clip["clip_index"]),
                        "--start-frame", str(clip.get("start_frame", 0)), "--frames", str(clip["frames"]),
                        "--seed", str(seed), "--devs", "0,1", "--out", str(dest)]
                command = ("cd " + shlex.quote(str(REPO)) + " && source finetune/transcript_pose/environment/activate.sh && "
                           "export CUDA_VISIBLE_DEVICES=" + pair + " MIMI_DIR=/home/param/models/mimi "
                           "OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 && exec " + shlex.join(args))
                logfile = logs / (name + ".log")
                with logfile.open("w") as log:
                    argv = ["bash", "-lc", command] if host == "landsend-0" else SSH + [host, command]
                    proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT)
                    event("started", name=name, host=host, gpu_pair=pair, controller_pid=proc.pid,
                          seed=seed, git_commit=sha, output=str(dest))
                    rc = proc.wait()
                if rc:
                    raise RuntimeError(f"Inference exited {rc}; see {logfile}")
                if host != "landsend-0":
                    dest.mkdir(parents=True, exist_ok=True)
                    subprocess.run(["rsync", "-a", "-e", shlex.join(SSH), f"{host}:{dest}/", str(dest) + "/"], check=True)
                details = validate_output(dest, clip, recipe, seed)
                receipt = dict(host=host, gpu_pair=pair, clip_index=clip["clip_index"], recipe=recipe,
                               seed=seed, exit_code=0, git_commit=sha, **details)
                tmp = sentinel.with_suffix(".tmp")
                tmp.write_text(json.dumps(receipt, indent=2) + "\n")
                tmp.replace(sentinel)
                event("complete", name=name, **receipt)
            except Exception as exc:
                with lock:
                    failures.append(dict(name=name, error=str(exc)))
                event("failed", name=name, host=host, gpu_pair=pair, error=str(exc))
            finally:
                jobs.task_done()

    with ThreadPoolExecutor(max_workers=4 * len(a.nodes)) as pool:
        futures = [pool.submit(slot, host, f"{i},{i + 1}") for host in a.nodes for i in range(0, 8, 2)]
        for future in futures:
            future.result()
    events.close()
    if failures:
        raise RuntimeError(json.dumps(failures, indent=2))
    print("All requested PPM outputs completed and validated", flush=True)


if __name__ == "__main__":
    main()
