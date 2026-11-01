#!/usr/bin/env python3
"""LOCAL TEST ENDPOINT for the videort feed (aloe-posedup-live): a WebSocket server at ws://0.0.0.0:<port>/feed that records every message
(+ its receive time) to <out jsonl> and prints a summary per session on 'end'. Format checks (start / f / end keys and types, f contiguous
from 0, 32 pose + 32 Mimi codes in range, pcm_b64 = 1920 int16 samples) -> <out jsonl>.summary.json.
usage: python -I feed_sink.py <port> <out jsonl>"""
import base64
import json
import sys
import time

from aiohttp import web, WSMsgType

PORT, OUT = int(sys.argv[1]), sys.argv[2]
S = {}


def check(sess):
    st, fr, en = sess["start"], sess["f"], sess["end"]
    st = {k: v for k, v in (st or {}).items() if k != "_rx"} or None
    bad = []
    if not st or set(st) - {"type", "session", "model", "voice", "fps", "person"} or st.get("model") != "posedup" or st.get("fps") != 12.5:
        bad.append(f"start {st}")
    if [m["f"] for m in fr] != list(range(len(fr))):
        bad.append("f not contiguous from 0")
    for m in fr:
        if set(m) - {"_rx"} != {"type", "session", "f", "t", "pose", "mimi", "pcm_b64"}:
            bad.append(f"f keys {sorted(m)}"); break
        if len(m["pose"]) != 32 or len(m["mimi"]) != 32 or not all(0 <= v < 2048 for v in m["pose"] + m["mimi"]):
            bad.append(f"codes at f {m['f']}"); break
        if len(base64.b64decode(m["pcm_b64"])) != 3840:
            bad.append(f"pcm length at f {m['f']}"); break
    if not en:
        bad.append("no end")
    lat = [m["_rx"] - m["t"] for m in fr]
    lat.sort()
    return {"session": st and st.get("session"), "start": st, "frames": len(fr), "end": bool(en), "bad": bad, "PASS": not bad and len(fr) > 0,
            "rx_minus_t_ms_p50": round(1000 * lat[len(lat) // 2], 1) if lat else None,
            "rx_minus_t_ms_max": round(1000 * lat[-1], 1) if lat else None}


async def feed(req):
    ws = web.WebSocketResponse(max_msg_size=8 << 20, heartbeat=20)
    await ws.prepare(req)
    print(f"[sink {time.strftime('%T')}] client connected {req.remote}", flush=True)
    with open(OUT, "a") as fh:
        async for msg in ws:
            if msg.type != WSMsgType.TEXT:
                continue
            rx = time.time()
            d = json.loads(msg.data)
            fh.write(json.dumps(dict(d, _rx=round(rx, 4))) + "\n"); fh.flush()
            sid = d.get("session")
            s = S.setdefault(sid, {"start": None, "f": [], "end": None})
            if d["type"] == "start":
                s["start"] = d
            elif d["type"] == "f":
                s["f"].append(dict(d, _rx=rx))
            elif d["type"] == "end":
                s["end"] = d
                r = check(s)
                print(f"[sink {time.strftime('%T')}] {json.dumps(r)}", flush=True)
                json.dump(r, open(f"{OUT}.summary-{sid}.json", "w"), indent=1)
    print(f"[sink {time.strftime('%T')}] client closed", flush=True)
    return ws


app = web.Application()
app.router.add_get("/feed", feed)
print(f"[sink] ws://0.0.0.0:{PORT}/feed -> {OUT}", flush=True)
web.run_app(app, host="0.0.0.0", port=PORT, access_log=None, print=None)
