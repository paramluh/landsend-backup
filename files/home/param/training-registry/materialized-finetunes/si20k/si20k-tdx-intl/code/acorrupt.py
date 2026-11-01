"""si20k-vox TRAINING CORRUPTION of the Voxtral-side waveform (user 10-18 PT): qqa-corrupt-v2 (audiocorrupt-work/code/corrupt_lib.py,
the corruption the si20k filter judged intelligibility after), applied IN THE DATA COLLATOR (DataLoader worker), drawn fresh every pass and
seeded by (row key, epoch); never precomputed. Same function for the si20k user audio and the podcast partner audio.
Draw per row: u_split < 0.35 -> clean; else u_mode < 0.70 -> realistic preset, else random styles with severity mild 40 / medium 40 /
strong 20 (corrupt_lib's split rule, here per (row, epoch) instead of a fixed per-id rank). The engine's own randomness is keyed by
'<row key>|e<epoch>'. Far-end (echo / ducking stages) = channel B's waveform when available (si20k), else silence (podcast mono FLACs).
The engine runs in a persistent subprocess per DataLoader worker in the voicebank venv (corrupt_lib needs torchaudio, which the training venv
lacks): acorrupt_server.py; float32 [-1, 1) <-> int16 at the boundary (the corruption's own int16 output)."""
import hashlib
import json
import os
import subprocess

import numpy as np

PY = "/data/home/amartyavarma/venvs/voicebank/bin/python"
SERVER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "acorrupt_server.py")
P_CLEAN, P_REALISTIC, SEV = 0.35, 0.70, (("mild", 0.40), ("medium", 0.80), ("strong", 1.00))
_SRV = [None]


def u01(*parts):
    return int.from_bytes(hashlib.sha256("\x1f".join(str(p) for p in parts).encode()).digest()[:8], "big") / 2.0 ** 64


def draw(key, epoch):
    """-> (mode, severity): ('clean', None) | ('realistic', None) | ('random', 'mild' | 'medium' | 'strong')."""
    if u01("si20k-corrupt-split", key, epoch) < P_CLEAN:
        return "clean", None
    if u01("si20k-corrupt-mode", key, epoch) < P_REALISTIC:
        return "realistic", None
    u = u01("si20k-corrupt-sev", key, epoch)
    return "random", next(s for s, hi in SEV if u < hi)


def _server():
    if _SRV[0] is None or _SRV[0].poll() is not None:
        env = dict(os.environ, PYTHONPATH="/data/home/amartyavarma/ref/audiocorrupt", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   PATH="/data/home/amartyavarma/data/si20k-tts/filter/bin:" + os.environ.get("PATH", "/usr/bin:/bin"))
        _SRV[0] = subprocess.Popen(["nice", "-n", "10", PY, "-u", SERVER], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=env)
    return _SRV[0]


def apply(w, key, epoch, far=None):
    """float32 waveform (24 kHz) -> (corrupted float32 same length, info)."""
    mode, sev = draw(key, epoch)
    info = {"mode": mode, "sev": sev}
    if mode == "clean" or w.size == 0:
        return w, info
    a = np.clip(np.round(w.astype(np.float64) * 32768.0), -32768, 32767).astype(np.int16)
    b = None if far is None else np.clip(np.round(far.astype(np.float64) * 32768.0), -32768, 32767).astype(np.int16)
    s = _server()
    s.stdin.write((json.dumps({"key": f"{key}|e{int(epoch)}", "mode": mode, "sev": sev, "n": int(a.size), "has_b": b is not None}) + "\n").encode())
    s.stdin.write(a.tobytes())
    if b is not None:
        s.stdin.write(b[:a.size].tobytes() if b.size >= a.size else np.pad(b, (0, a.size - b.size)).tobytes())
    s.stdin.flush()
    r = json.loads(s.stdout.readline())
    y = np.frombuffer(s.stdout.read(2 * a.size), np.int16)
    assert y.size == a.size, ("corruption server reply length", y.size, a.size)
    info.update(r)
    return (y.astype(np.float32) / 32768.0), info
