"""Preset bank for the Rosie FT: 4 Rosie clone presets (one per held-out call of InternalCan/rosie-stamp-conversations-v1, Rosie's ORIGINAL
Mimi codes = what the FT was trained on as its voice-clip prefix, blank lane, 112 frames) + the 12 release Aloe presets.
  source aloe.env; cd code/ppm/aloe-m1vox-live; $ALOE_PY /workspace/aloe-live/tools/build_rosie_bank.py"""
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

sys.path.insert(0, os.getcwd())
import prompts_ppm as PP  # noqa: E402

SRC, DST = "/workspace/aloe-live/assets/presets", "/workspace/aloe-live/assets/presets-rosie"
HELD_OUT = ["d65a446f", "09bb73ac", "4cc4c9fc", "88e8e0f4"]
L = 112                                                    # 8.96 s, inside the trained 38..125 clip range

doc = json.load(open(f"{SRC}/presets.json"))
z = dict(np.load(f"{SRC}/{doc['npz']}"))
os.makedirs(f"{DST}/clips", exist_ok=True)
for f in os.listdir(f"{SRC}/clips"):
    shutil.copy(f"{SRC}/clips/{f}", f"{DST}/clips/{f}")
idx = pq.read_table("/workspace/data/rosie/index/conversations.parquet").to_pandas()
new = []
for n, sid in enumerate(HELD_OUT):
    r = idx[idx.session_id.str.startswith(sid) & (idx.seg == "seg00")].iloc[0]
    t = pq.read_table(f"/workspace/data/rosie/{r.token_shard}").slice(int(r.token_row), 1).to_pylist()[0]
    codes = np.load(io.BytesIO(t["rosie_original_codes"])).astype(np.int64)
    W = json.loads(t["words_json"])["words"]
    T = codes.shape[1]
    ro, us = np.zeros(T), np.zeros(T)
    for w in W:
        a, b = int(w["start"] * 12.5 + 1e-6), min(T, int(np.ceil(w["end"] * 12.5)))
        (ro if w["role"] == "assistant" else us)[a:b] = 1
    st = np.arange(0, T - L + 1, 4)                         # stride 4 like sysprompt.window; most Rosie speech, least partner speech
    score = np.array([ro[s:s + L].mean() - 3 * us[s:s + L].mean() for s in st])
    w0 = int(st[score.argmax()])
    c = codes[:, w0:w0 + L]
    pid = f"rosie-{sid}"
    mine, _ = PP.build_prefix("", PP.upload_clip(c))
    z[f"codes__{pid}"] = c
    for x in ("ids", "depth", "present"):
        z[f"ref_{x}__{pid}"] = np.asarray(mine[x])
    g = float(json.loads(t["original_mimi_json"])["mimi"][r.rosie_lane]["lufs_gain_db"])
    fl = f"/workspace/data/rosie/flac/{r.session_id}/seg00/Rosie_Stamp.original.flac"
    a = np.frombuffer(subprocess.run(["ffmpeg", "-v", "error", "-i", fl, "-ac", "1", "-ar", "24000", "-f", "f32le", "-"],
                                     capture_output=True, check=True).stdout, np.float32)
    sf.write(f"{DST}/clips/{pid}.wav", np.clip(a[w0 * 1920:(w0 + L) * 1920] * 10 ** (g / 20), -1, 1), 24000, subtype="PCM_16")
    txt = " ".join(w["text"] for w in W if w["role"] == "assistant" and w0 / 12.5 <= w["start"] < (w0 + L) / 12.5)
    print(f"{pid} w0={w0} rosie {ro[w0:w0 + L].mean():.2f} user {us[w0:w0 + L].mean():.2f} gain {g:+.1f} dB P={mine['P']}: {txt[:100]}")
    new.append({"id": pid, "order": n, "name": f"Rosie (clone) {n + 1}", "k": -1, "r": -1, "sw": 0, "side": "assistant",
                "src_key": f"rosie/{r.session_id}/seg00/c00", "src": "held_out_call_original_codes", "w0": w0, "L": L, "seconds": L / 12.5,
                "P": int(mine["P"]), "clip_wav": f"clips/{pid}.wav",
                "clip_wav_sha256": hashlib.sha256(open(f"{DST}/clips/{pid}.wav", "rb").read()).hexdigest()})
for m in doc["presets"]:
    m["order"] = int(m["order"]) + len(new)
doc["presets"] = new + doc["presets"]
doc["n"] = len(doc["presets"])
doc["source"] = "4 Rosie clone presets (held-out calls, Rosie ORIGINAL Mimi codes, blank lane, 112 frames) + " + doc["source"]
np.savez(f"{DST}/preset_codes.npz", **z)
doc["npz_sha256"] = hashlib.sha256(open(f"{DST}/preset_codes.npz", "rb").read()).hexdigest()
json.dump(doc, open(f"{DST}/presets.json", "w"), indent=1)
B = PP.BankPresets(DST, first="rosie-d65a446f")
print("selftest", B.selftest()["PASS"], [p["id"] for p in B.items][:6])
