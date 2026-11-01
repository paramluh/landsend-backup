"""qqa-corrupt-v2 server (runs in the voicebank venv: torch / torchaudio / audiocorrupt; one per DataLoader worker, spawned by acorrupt.py).
Protocol on stdin / stdout: request = one JSON line {"key", "mode", "sev", "n", "has_b", "voices"} + n int16 (A) [+ n int16 (B)] raw bytes;
reply = one JSON line (record summary) + n int16 bytes. corrupt_lib (audiocorrupt-work/code, qqa-corrupt-v2) imported verbatim; competing
talkers = the si20k filter's pool (clean channels of other si20k conversations, CC BY 3.0 YODAS bank voices)."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/data/home/amartyavarma/audiocorrupt-work/code")
import torch  # noqa: E402
torch.set_num_threads(1)
import corrupt_lib as CL  # noqa: E402
CL.SPEECH_LICENCE = "CC BY 3.0"
POOL = os.environ.get("SI20K_CORRUPT_POOL", "/data/home/amartyavarma/data/si20k-tts/filter/pool")
eng = CL.Engine(POOL, json.load(open(f"{POOL}/manifest.json")))
inp, out = sys.stdin.buffer, sys.stdout.buffer
while True:
    ln = inp.readline()
    if not ln:
        break
    q = json.loads(ln)
    n = int(q["n"])
    a = np.frombuffer(inp.read(2 * n), np.int16).copy()
    b = np.frombuffer(inp.read(2 * n), np.int16).copy() if q.get("has_b") else np.zeros(n, np.int16)
    try:
        y, rec = eng.corrupt(q["key"], q["mode"], q.get("sev"), a, b, tuple(q.get("voices") or (None, None)))
        r = {"ok": 1, "attempts": rec["attempts"], "final_gain_db": rec["final_gain_db"], "scenario": rec.get("scenario"), "styles": rec.get("styles")}
    except Exception as e:  # noqa: BLE001
        y, r = a, {"ok": 0, "error": repr(e)[:300]}
    out.write((json.dumps(r) + "\n").encode()); out.write(np.ascontiguousarray(y, np.int16).tobytes()); out.flush()
