#!/usr/bin/env python3
"""(aloe-posedup-live copy: + pose / session / feed checks in result.json; env PPMLIVE_PERSON = the start message's avatar person id)
NON-PREEMPTING test client for the aloe-m1vox live demo: client 'test' (the server refuses it while any session is active, and it
never takes over a browser session). Streams a 24 kHz wav as the mic in real time (80 ms chunks), records B's frames / audio.
  python test_client.py <base URL> <a.wav> <out dir> [secs] [preset id]"""
import asyncio
import json
import os
import sys
import time

import aiohttp
import numpy as np
import soundfile as sf


async def run(base, wav, out, secs, preset, params=None, live_set=None):
    x, sr = sf.read(wav, dtype="float32", always_2d=True)
    x = x.mean(1)
    assert sr == 24000, sr
    x = x[:int(secs * 24000)]
    url = base.rstrip("/").replace("https://", "wss://").replace("http://", "ws://") + "/ws"
    frames, bchunks, msgs = [], [], []
    async with aiohttp.ClientSession() as s:
        model = os.environ.get("PPMLIVE_MODEL", "")                 # 10-25: a front proxy serving several models: route to this one
        async with s.get(base.rstrip("/") + "/api/presets" + (f"?model={model}" if model else "")) as r:
            pj = await r.json()
        if pj.get("busy"):
            print("BUSY: a session is active; the test client never preempts. Exit."); return 2
        start = {"type": "start", "client": "test", "preset": preset or pj["default_preset"], "persona": None,
                 "clip": pj["default_clip"].get(preset or pj["default_preset"]), "params": params or {}}
        if os.environ.get("PPMLIVE_PERSONA"):                         # 10-25: B's persona text for a persona-trained model
            start["persona"] = os.environ["PPMLIVE_PERSONA"]
        if os.environ.get("PPMLIVE_PERSON"):                          # aloe-posedup-live: the avatar person id (feed start)
            start["person"] = os.environ["PPMLIVE_PERSON"]
        if model:
            start["model"] = model
            url += f"?model={model}"
        async with s.ws_connect(url, max_msg_size=8 << 20, heartbeat=20) as ws:
            await ws.send_str(json.dumps(start))
            t0 = time.time()
            ready = None
            while ready is None:
                m = await ws.receive(timeout=180)
                if m.type == aiohttp.WSMsgType.TEXT:
                    d = json.loads(m.data); msgs.append(d)
                    if d["type"] == "ready":
                        ready = d
                    elif d["type"] == "error":
                        print("ERROR", d); return 1
                else:
                    print("closed", m); return 1
            print(f"ready after {time.time() - t0:.1f}s: P {ready['P']} seed {ready['seed']} step {ready['step']} prompt {ready.get('prompt')}")

            async def sender():
                n = 1920
                t1 = time.time()
                for i in range(0, len(x), n):
                    if live_set and i // n == 60:
                        await ws.send_str(json.dumps(dict({"type": "set"}, **live_set)))
                    y = np.clip(x[i:i + n] * 32767, -32768, 32767).astype("<i2").tobytes()
                    await ws.send_bytes(y)
                    dt = t1 + (i // n + 1) * 0.08 - time.time()
                    if dt > 0:
                        await asyncio.sleep(dt)
                await ws.send_bytes(np.zeros(1920 * 3, "<i2").tobytes())
                await asyncio.sleep(0.5)
                await ws.send_str(json.dumps({"type": "stop"}))
            st = asyncio.ensure_future(sender())
            done = None
            while True:
                m = await ws.receive(timeout=60)
                if m.type == aiohttp.WSMsgType.BINARY:
                    bchunks.append(np.frombuffer(m.data, "<i2").astype(np.float32) / 32768)
                elif m.type == aiohttp.WSMsgType.TEXT:
                    d = json.loads(m.data)
                    if d["type"] == "f":
                        frames.append(d)
                    else:
                        msgs.append(d)
                        if d["type"] == "done":
                            done = d; break
                        if d["type"] == "error":
                            print("ERROR", d); break
                else:
                    break
            st.cancel()
    os.makedirs(out, exist_ok=True)
    b = np.concatenate(bchunks) if bchunks else np.zeros(0, np.float32)
    sf.write(f"{out}/b.wav", b, 24000)
    ms = np.asarray([f["ms"] for f in frames])
    words = "".join((f["tok"] if f["ws"] or not f["tok"] else f["tok"]) for f in frames if f["tok"])
    res = {"frames": len(frames), "b_seconds": round(len(b) / 24000, 2), "ms_p50": float(np.percentile(ms, 50)) if len(ms) else None,
           "ms_p95": float(np.percentile(ms, 95)) if len(ms) else None, "ms_max": float(ms.max()) if len(ms) else None,
           "lag_max": max((f["lag"] for f in frames), default=None), "done": done,
           "settings": [m_ for m_ in msgs if m_.get("type") == "settings"], "in_effect": ready.get("in_effect"), "rejected": ready.get("rejected"),
           "cfg_frames": sum(1 for f in frames if f.get("s", 1.0) != 1.0), "tv_mean": float(np.mean([f["tv"] for f in frames if f.get("tv") is not None])) if frames else None, "words": words, "ready": {k: ready[k] for k in ("P", "seed", "step", "setup_s", "s")}}
    # aloe-posedup-live: pose codes in every frame message from code frame 0 on (cf = f - 2), contiguous, all 32 >= 0
    pf = [f for f in frames if f.get("cf") is not None]
    okp = [f for f in pf if isinstance(f.get("pose"), list) and len(f["pose"]) == 32 and min(f["pose"]) >= 0 and max(f["pose"]) < 2048]
    res.update(session=ready.get("session"), person=ready.get("person"), voice=ready.get("voice"), feed=ready.get("feed"),
               rtf=round(float(ms.mean()) / 80.0, 3) if len(ms) else None, ms_mean=float(ms.mean()) if len(ms) else None,
               pose={"frames": len(frames), "code_frames": len(pf), "with_32_codes": len(okp), "first_cf": pf[0]["cf"] if pf else None,
                     "cf_contiguous": [f["cf"] for f in pf] == list(range(len(pf))), "cf_eq_f_minus_2": all(f["cf"] == f["f"] - 2 for f in pf),
                     "pre_cd_frames_without_pose": sum(1 for f in frames if f.get("cf") is None and f.get("pose") is None),
                     "cb0_unique": len({f["pose"][0] for f in okp})})
    np.save(f"{out}/pose.npy", np.asarray([f["pose"] for f in okp], np.int16).reshape(-1, 32))
    json.dump(res, open(f"{out}/result.json", "w"), indent=1, default=str)
    print(json.dumps(res, default=str)[:1500])
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(run(sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]) if len(sys.argv) > 4 else 20.0,
                             sys.argv[5] if len(sys.argv) > 5 and sys.argv[5] != "-" else None,
                             json.loads(sys.argv[6]) if len(sys.argv) > 6 else None, json.loads(sys.argv[7]) if len(sys.argv) > 7 else None)))
