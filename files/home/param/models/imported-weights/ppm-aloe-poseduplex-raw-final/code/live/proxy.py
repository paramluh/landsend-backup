#!/usr/bin/env python3
"""ALWAYS-UP FRONT PROXY for the live demo: owns the tunnel port, forwards HTTP + WebSocket to the backend (server.py on BACKEND).
Backend down -> the page route serves a static "restarting, back in ~60 s" page (auto-reload), /api/* -> 503 JSON, the WebSocket is
accepted and closed cleanly (1013, reason "model server restarting, back in ~60 s") after a JSON error message. Token checked here
too (constant time); no access log (the query carries the token).
  python proxy.py --port 8785 --backend 8786 --token-file ~/.s4live_token
10-17 LOAD BALANCER (--lb, default <this dir>/lb.json, keyed by this proxy's --port; no entry / one backend -> the single-backend proxy
exactly as before, e.g. instance 2's own front :8795): one public URL over several identical backends (same checkpoint, clips, page).
  * WS: the proxy waits for the client's 'start', then pins the connection to the first IDLE backend in config order (idle = /healthz
    ready and not busy, and no session of this proxy pinned there) for the connection's whole life. No idle backend: a 'browser' start
    goes to a backend held by one of this proxy's TEST sessions (the backend's takeover ends the test: tests never keep a backend from a
    real user); otherwise the busy message (error + close 1013 'busy'). Test clients never take a backend from anyone. The proxy first
    sends {"type": "route", "instance", "of", "name"} (the page shows 'instance k/n' in Call). A pin is released on the backend's
    done / taken_over / timeout / error or when the connection closes.
  * POST /api/clip (upload) goes to the backend the next session would get; the proxy keeps the raw upload and, if a session using it
    lands on another backend, re-uploads it there at start and rewrites the clip id in the start message (uploads seen only before a
    proxy restart: re-uploaded from the backend's sessions/uploads/<id>.npz).
  * Everything else (page, presets, params schema, healthz) is served by the first ready backend preferring idle ones, falling over to the
    next on connection error; /api/presets 'busy' = no idle backend, plus 'lb' {instances, idle, served_by}. /proxyhealth lists all.
  * 10-25 SESSION RESUME (server.py RESUME_S): a socket whose first message is {type:'resume', session, rtoken} goes to the backend that
    sent that session's 'ready' (remembered per session id), never to a new pick, and is re-pinned there; unknown session id (e.g. a proxy
    restart since the 'ready'): every backend in turn, busy ones first, until one answers 'resumed' (the others refuse a foreign session
    without touching their own)."""
import argparse
import asyncio
import collections
import io
import json
import os
import re
import time

import aiohttp
from aiohttp import web, WSMsgType

RESTART_HTML = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Aloe Pose-Duplex Live</title><style>:root{--bg:#fbfaf7;--fg:#1d1d1b}@media (prefers-color-scheme:dark){:root{--bg:#171716;--fg:#ecebe6}}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,sans-serif}main{max-width:640px;margin:0 auto;padding:40px 16px}</style>
</head><body><main><h1>Aloe pose-duplex: live voice</h1><p>The model server is restarting, back in about 60 s. This page reloads by itself.</p>
<p id="t"></p><script>let n=0;setInterval(()=>{n+=5;document.getElementById('t').textContent='waited '+n+' s';fetch(location.href,{cache:'no-store'}).then(r=>r.text()).then(t=>{if(t.indexOf('id="go"')>=0)location.reload();}).catch(()=>{});},5000);</script>
</main></body></html>"""
MSG = "model server restarting, back in ~60 s"
HERE = os.path.dirname(os.path.abspath(__file__))
CONN_ERR = (aiohttp.ClientConnectorError, aiohttp.ServerDisconnectedError, asyncio.TimeoutError, ConnectionError)
TERMINAL = ("done", "taken_over", "timeout", "error")                # backend messages after which its session is over (or never began)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--backend", type=int, default=8766)
    ap.add_argument("--token-file", required=True)
    ap.add_argument("--sessions", default="", help="the backend's sessions dir (upload clip previews); default <this dir>/sessions")
    ap.add_argument("--lb", default=f"{HERE}/lb.json", help="load-balancer config {\"<front port>\": [{port, name, sessions}, ...]}")
    a = ap.parse_args()
    token = open(a.token_file).read().strip()
    B = f"http://127.0.0.1:{a.backend}"
    LB = None
    try:
        ent = json.load(open(a.lb)).get(str(a.port))
        if isinstance(ent, list) and len(ent) > 1:
            LB = [dict(b, port=int(b["port"]), name=str(b.get("name") or f"instance {k + 1}")) for k, b in enumerate(ent)]
    except (OSError, ValueError, KeyError, TypeError):
        LB = None
    UPDIRS = [a.sessions or HERE + "/sessions"] + [b["sessions"] for b in (LB or []) if b.get("sessions")]
    UPDIRS = list(dict.fromkeys(UPDIRS))
    print(f"proxy :{a.port} " + ("LOAD BALANCER over " + ", ".join(f"{b['name']} :{b['port']}" for b in LB) if LB else f"backend :{a.backend}"), flush=True)

    def ok(req):
        return True                                                    # 10-10 user: no auth on the public link; the proxy adds the token upstream

    def down_response(req):
        if req.path.startswith("/api/"):
            return web.json_response({"error": MSG, "state": "restarting"}, status=503)
        return web.Response(text=RESTART_HTML, content_type="text/html", headers={"Cache-Control": "no-store"})

    async def http(req):
        if not ok(req):
            return web.Response(status=403, text="forbidden")
        sess = req.app["cs"]
        try:
            if req.query.get("proxytest") == "down":                      # test hook: behave as if the backend were down
                raise aiohttp.ServerDisconnectedError()
            body = await req.read()
            hdr = {k: v for k, v in req.headers.items() if k.lower() in ("content-type", "x-filename")}
            async with sess.request(req.method, B + str(req.rel_url.update_query(token=token)), data=body or None, headers=hdr,
                                    timeout=aiohttp.ClientTimeout(total=120)) as r:
                data = await r.read()
                h = {k: v for k, v in r.headers.items() if k.lower() in ("content-type", "cache-control")}
                return web.Response(status=r.status, body=data, headers=h)
        except CONN_ERR:
            return down_response(req)

    async def pump_pair(wc, wb, on_b2c_text=None, on_c2b_text=None, st=None):
        """forward both directions until either side ends (= the single-backend proxy's c2b / b2c). 10-25: st (dict) collects per-direction
        message counts and the longest gap between consecutive messages (early-cut diagnosis); an on_c2b_text returning True = not forwarded."""
        st = st if st is not None else {}
        st.update(t0=time.time(), c_n=0, c_bytes=0, c_gap=0.0, c_gap_at=0.0, c_last=time.time(), b_n=0, b_gap=0.0, b_gap_at=0.0, b_last=time.time())

        def tick(k):
            t = time.time(); g = t - st[k + "_last"]
            if g > st[k + "_gap"]:
                st[k + "_gap"], st[k + "_gap_at"] = g, st[k + "_last"] - st["t0"]
            st[k + "_last"] = t; st[k + "_n"] += 1

        q = asyncio.Queue()                                                # 10-25: client reads decoupled from backend sends, so a stuck
                                                                           # backend hop shows as a send block / queue, not a client gap
        async def c2b_read():
            async for m in wc:
                tick("c")
                if m.type == WSMsgType.BINARY:
                    st["c_bytes"] += len(m.data); q.put_nowait((True, m.data))
                elif m.type == WSMsgType.TEXT:
                    if on_c2b_text is not None and on_c2b_text(m.data) is True:
                        continue
                    q.put_nowait((False, m.data))
                else:
                    break
            q.put_nowait(None)

        async def c2b():
            st.update(send_max=0.0, send_max_at=0.0, q_max=0)
            while True:
                it = await q.get()
                if it is None:
                    break
                t = time.time()
                await (wb.send_bytes(it[1]) if it[0] else wb.send_str(it[1]))
                dt = time.time() - t
                if dt > st["send_max"]:
                    st["send_max"], st["send_max_at"] = dt, t - st["t0"]
                st["q_max"] = max(st["q_max"], q.qsize())
            await wb.close()

        async def b2c():
            async for m in wb:
                tick("b")
                if m.type == WSMsgType.BINARY:
                    await wc.send_bytes(m.data)
                elif m.type == WSMsgType.TEXT:
                    r = on_b2c_text(m.data) if on_b2c_text is not None else None
                    await wc.send_str(r if isinstance(r, str) else m.data)      # 10-26: a hook may rewrite a message (ready: avatar path)
                else:
                    break
            if not wc.closed:
                await wc.send_json({"type": "error", "msg": "model server connection lost (" + MSG + ")"})
                await wc.close(code=1013, message=MSG.encode())

        t0r, t1, t2 = asyncio.ensure_future(c2b_read()), asyncio.ensure_future(c2b()), asyncio.ensure_future(b2c())
        await asyncio.wait([t1, t2], return_when=asyncio.FIRST_COMPLETED)
        for t in (t0r, t1, t2):
            t.cancel()
        if not wb.closed:
            await wb.close()

    async def wsh(req):
        if not ok(req):
            return web.Response(status=403, text="forbidden")
        wc = web.WebSocketResponse(max_msg_size=8 << 20, heartbeat=20)
        await wc.prepare(req)
        try:
            if req.query.get("proxytest") == "down":
                raise ConnectionError("proxytest")
            wb = await req.app["cs"].ws_connect(B + str(req.rel_url.update_query(token=token)), max_msg_size=8 << 20, heartbeat=20)
        except Exception:  # noqa: BLE001
            await wc.send_json({"type": "error", "msg": MSG})
            await wc.close(code=1013, message=MSG.encode())
            return wc
        # 10-25: single-backend diagnostics too (page 'diag' logged here, not forwarded; per-session gaps / send blocks at the end)
        def c2b_text(s):
            if s.startswith('{"type":"diag"') or s.startswith('{"type": "diag"'):
                print(f"diag: {s[:600]}", flush=True)
                return True
        pst = {}
        await pump_pair(wc, wb, None, c2b_text, pst)
        d = time.time() - pst.get("t0", time.time())
        print(f"session end: {d:.1f} s; client->backend {pst.get('c_n')} msgs {pst.get('c_bytes', 0) / 48000:.1f} s audio, longest gap {pst.get('c_gap', 0):.1f} s "
              f"at t={pst.get('c_gap_at', 0):.1f}, last msg {time.time() - pst.get('c_last', time.time()):.1f} s before the end; backend->client {pst.get('b_n')} msgs, "
              f"longest gap {pst.get('b_gap', 0):.1f} s at t={pst.get('b_gap_at', 0):.1f}; slowest send to the backend {pst.get('send_max', 0):.2f} s, max queued {pst.get('q_max', 0)}", flush=True)
        return wc

    async def on_start(app):
        app["cs"] = aiohttp.ClientSession()

    async def on_stop(app):
        await app["cs"].close()

    app = web.Application(client_max_size=26 << 20)
    app.on_startup.append(on_start)
    app.on_cleanup.append(on_stop)

    async def health(req):
        if not ok(req):
            return web.Response(status=403, text="forbidden")
        try:
            async with req.app["cs"].get(B + "/healthz", timeout=aiohttp.ClientTimeout(total=5)) as r:
                bk = await r.json()
        except Exception as e:  # noqa: BLE001
            bk = {"down": repr(e)[:120]}
        return web.json_response({"proxy": "ok", "backend": bk})

    def npz_path(cid):
        for d in UPDIRS:
            p = f"{d}/uploads/{cid}.npz"
            if os.path.isfile(p):
                return p
        return f"{UPDIRS[0]}/uploads/{cid}.npz"

    def npz_wav(cid):
        import numpy as np
        import soundfile as sf
        w = np.load(npz_path(cid))["wav"].astype(np.float32)
        bio = io.BytesIO()
        sf.write(bio, np.clip(w, -1, 1), 24000, subtype="PCM_16", format="WAV")
        return bio.getvalue()

    async def clipwav(req):
        """B voice-prompt preview, served by the proxy itself (no backend / model involvement): id=preset:<preset id> -> clips/<id>.wav
        (the Mimi decode of the clip's codes = exactly what the model's prompt holds); id=upload:<hex> -> the upload as Mimi-encoded
        (first <= 10 s, -23 LUFS), from the backend's sessions/uploads/<hex>.npz (load balancer: any backend's). audio/wav, full body
        (no Range needed: the page plays a blob)."""
        cid = req.query.get("id", "")
        try:
            if re.fullmatch(r"preset:(aloe-\d+-\d+-[01]|\d\d_[0-9a-f]{8})", cid):   # 10-25: + the 20 video-person presets (clips/<NN>_<hex>.wav)
                # 10-24: the instance's own <state dir>/clips (e.g. the sam-sep presets' decodes) first, else the shared clips/
                own = f"{os.path.dirname((a.sessions or HERE + '/sessions').rstrip('/'))}/clips/{cid[7:]}.wav"
                data = open(own if os.path.isfile(own) else f"{HERE}/clips/{cid[7:]}.wav", "rb").read()
            elif re.fullmatch(r"start:\d\d_[0-9a-f]{8}", cid):   # 10-27: the conversation-start clip (opening mimi_local[:, :125]) decode
                data = open(f"{os.path.dirname((a.sessions or HERE + '/sessions').rstrip('/'))}/clips/start_{cid[6:]}.wav", "rb").read()
            elif re.fullmatch(r"upload:[0-9a-f]{12}", cid):
                data = npz_wav(cid[7:])
            else:
                return web.json_response({"error": "unknown clip id"}, status=404)
        except (OSError, KeyError, ValueError) as e:
            return web.json_response({"error": "clip audio not available: " + repr(e)[:120]}, status=404)
        return web.Response(body=data, content_type="audio/wav", headers={"Cache-Control": "no-store", "Accept-Ranges": "none"})

    # ------------------------------------------------------------------ load balancer (LB mode only) ------------------------------------
    N = len(LB) if LB else 0
    OWNER = [None] * N                                 # backend index -> the Pin of this proxy's session pinned there (None = free)
    UPLOADS = collections.OrderedDict()                # public upload id -> {"ids": {backend: its id}, "raw": bytes, "hdr": dict}
    UP_CAP = 1 << 30
    HOME = collections.OrderedDict()                   # 10-25 resume: session id -> (backend index, client) from the backend's 'ready'
    PROBE = {"t": 0.0, "st": None}
    LOCK = asyncio.Lock()

    class Pin:
        __slots__ = ("client", "i")

        def __init__(self, client):
            self.client, self.i = client, None

    # 10-25 (user: 'have both up'): lb.json entries may carry "host" (a backend on another node, bound to 0.0.0.0) and "model" / "model_label";
    # each session / API call goes to a backend of its model (?model= on the URL or "model" in the start message; default = the first entry's)
    M_OF = [str(b.get("model") or "default") for b in (LB or [])]
    MODELS = list(dict.fromkeys(M_OF))
    MLABEL = {m: next((str(b.get("model_label") or m) for b in LB if str(b.get("model") or "default") == m), m) for m in MODELS} if LB else {}
    MHIDDEN = {str(b.get("model") or "default") for b in (LB or []) if b.get("hidden")}   # 10-26: backends for other pages (the playground's live duplex) never show in this page's model picker
    REAL = ("browser", "playground")   # 10-26: real users (the playground's live-mic page too) are never taken over; test clients are

    def want_model(m):
        m = str(m or "")
        return m if m in MODELS else (MODELS[0] if MODELS else "default")

    def bk(i):
        return f"http://{LB[i].get('host') or '127.0.0.1'}:{LB[i]['port']}"

    async def probe_one(i):
        try:
            async with app["cs"].get(bk(i) + "/healthz", timeout=aiohttp.ClientTimeout(total=3)) as r:
                j = await r.json()
            return {"up": True, "ready": j.get("state") == "ready", "busy": bool(j.get("busy")), "state": j.get("state")}
        except Exception as e:  # noqa: BLE001
            return {"up": False, "ready": False, "busy": False, "state": "down", "err": repr(e)[:80]}

    async def probe_all(max_age=0.0):
        if PROBE["st"] is not None and time.time() - PROBE["t"] <= max_age:
            return PROBE["st"]
        st = list(await asyncio.gather(*[probe_one(i) for i in range(N)]))
        PROBE.update(t=time.time(), st=st)
        return st

    def idle_list(st, model=None):
        return [i for i in range(N) if st[i]["ready"] and not st[i]["busy"] and OWNER[i] is None and (model is None or M_OF[i] == model)]

    def http_order(st, model=None):
        """non-session calls: ready + idle first, then ready, then merely up, then the rest (config order within each); only the model's backends."""
        idle = idle_list(st, model)
        rest = [i for i in range(N) if i not in idle and (model is None or M_OF[i] == model)]
        return idle + sorted(rest, key=lambda i: (not st[i]["ready"], not st[i]["up"], i))

    def release(pin):
        if pin.i is not None and OWNER[pin.i] is pin:
            OWNER[pin.i] = None

    def remember_upload(i, body, hdr, data):
        try:
            cid = json.loads(data)["id"][7:]
        except (ValueError, KeyError, TypeError):
            return
        UPLOADS[cid] = {"ids": {i: cid}, "raw": body, "hdr": dict(hdr)}
        while len(UPLOADS) > 1 and sum(len(u["raw"]) for u in UPLOADS.values()) > UP_CAP:
            UPLOADS.popitem(last=False)

    async def upload_for(cid, i):
        """the clip id backend i knows for public upload id cid (re-uploading there if needed); unchanged if it cannot be mapped."""
        u = UPLOADS.get(cid)
        if u is None:                                                    # an upload from before this proxy started: its npz says where
            home = next((k for k, b in enumerate(LB) if b.get("sessions") and os.path.isfile(f"{b['sessions']}/uploads/{cid}.npz")), None)
            if home is None or home == i:
                return "upload:" + cid
            try:
                raw = await asyncio.get_running_loop().run_in_executor(None, npz_wav, cid)
            except Exception:  # noqa: BLE001
                return "upload:" + cid
            u = UPLOADS[cid] = {"ids": {home: cid}, "raw": raw, "hdr": {"Content-Type": "audio/wav"}}
        if i in u["ids"]:
            return "upload:" + u["ids"][i]
        try:
            async with app["cs"].post(bk(i) + "/api/clip?token=" + token, data=u["raw"], headers=u["hdr"],
                                      timeout=aiohttp.ClientTimeout(total=60)) as r:
                j = await r.json()
            u["ids"][i] = j["id"][7:]
            print(f"upload {cid} re-uploaded to {LB[i]['name']} as {j['id']}", flush=True)
            return j["id"]
        except Exception as e:  # noqa: BLE001
            print(f"upload {cid} re-upload to {LB[i]['name']} failed: {e!r}"[:300], flush=True)
            return "upload:" + cid

    async def http_lb(req):
        if not ok(req):
            return web.Response(status=403, text="forbidden")
        if req.query.get("proxytest") == "down":
            return down_response(req)
        body = await req.read()
        hdr = {k: v for k, v in req.headers.items() if k.lower() in ("content-type", "x-filename")}
        st = await probe_all(max_age=1.0)
        model = want_model(req.query.get("model"))
        for i in http_order(st, model):
            try:
                async with app["cs"].request(req.method, bk(i) + str(req.rel_url.update_query(token=token)), data=body or None, headers=hdr,
                                             timeout=aiohttp.ClientTimeout(total=120)) as r:
                    data = await r.read()
                    status = r.status
                    h = {k: v for k, v in r.headers.items() if k.lower() in ("content-type", "cache-control")}
            except CONN_ERR:
                continue
            if req.path == "/api/clip" and req.method == "POST" and status == 200:
                remember_upload(i, body, hdr, data)
            elif req.path == "/api/presets" and status == 200:
                try:
                    j = json.loads(data)
                    idle = [k + 1 for k in idle_list(st, model)]
                    j["busy"] = not idle
                    j["lb"] = {"instances": M_OF.count(model), "idle": idle, "served_by": i + 1}
                    j["model"] = model
                    j["models"] = [{"id": m, "label": MLABEL[m], "instances": M_OF.count(m), "idle": len(idle_list(st, m))} for m in MODELS if m not in MHIDDEN or m == model]
                    data = json.dumps(j).encode()
                except ValueError:
                    pass
            return web.Response(status=status, body=data, headers=h)
        return down_response(req)

    def hooks(pin, pick, model):
        """the per-connection text hooks of an LB session pinned to backend pick (fresh start or resumed)."""
        def b2c_text(s):                                                 # release the pin once the backend's session is over
            if s.startswith('{"type": "f"'):
                return
            try:
                d = json.loads(s)
                t = d.get("type")
            except (ValueError, AttributeError):
                return
            if t == "ready" and d.get("session"):                        # 10-25 resume: remember which backend runs this session
                k = str(d["session"])
                HOME[k] = (pick, pin.client)
                HOME.move_to_end(k)
                while len(HOME) > 512:
                    HOME.popitem(last=False)
                d["avatar_path"] = LB[pick].get("avatar", "avatar/")       # 10-26: each instance has its own avatar pipeline (same-origin mount)
                return json.dumps(d)
            if t in TERMINAL:
                release(pin)

        def c2b_text(s):                                                 # a further start on this connection re-pins if still free
            if s.startswith('{"type":"diag"') or s.startswith('{"type": "diag"'):   # 10-25 page diagnostics: logged here, not forwarded
                print(f"diag ({pin.client}, model {model}, {LB[pick]['name']}): {s[:600]}", flush=True)
                return True
            if '"start"' in s and OWNER[pick] is None:
                try:
                    if json.loads(s).get("type") == "start":
                        OWNER[pick] = pin
                except (ValueError, AttributeError):
                    pass
        return b2c_text, c2b_text

    async def resume_lb(req, wc, d):
        """10-25 SESSION RESUME: the backend that sent this session's 'ready' (HOME) gets the resume; unknown id: every backend in turn, busy
        ones first. A backend's first answer decides: 'resumed' -> forwarded (with anything it sent before it) and this connection becomes
        that backend's pinned LB connection; error -> the next candidate (unknown id) or forwarded (known home). Nobody answers -> close 1013
        without a message (the page's reconnect keeps trying within its 15 s)."""
        sid = str(d.get("session") or "")
        home = HOME.get(sid)
        if home is not None:
            order = [home[0]]
        else:
            st = await probe_all(max_age=1.0)
            order = sorted(range(N), key=lambda i: (not st[i]["busy"], i))
        pin = Pin(home[1] if home is not None else "browser")
        refusal = None
        for i in order:
            try:
                wb = await app["cs"].ws_connect(bk(i) + str(req.rel_url.update_query(token=token)), max_msg_size=8 << 20, heartbeat=20)
            except Exception:  # noqa: BLE001
                continue
            got, verdict = [], None
            try:
                await wb.send_str(json.dumps(d))
                t_end = time.time() + 6.0
                while verdict is None:
                    m = await wb.receive(timeout=max(0.1, t_end - time.time()))
                    if m.type == WSMsgType.BINARY:
                        got.append(m)
                    elif m.type == WSMsgType.TEXT:
                        got.append(m)
                        if not m.data.startswith('{"type": "f"'):
                            try:
                                t = json.loads(m.data).get("type")
                            except (ValueError, AttributeError):
                                t = None
                            verdict = "ok" if t == "resumed" else ("refused" if t == "error" else None)
                    else:
                        break
            except Exception:  # noqa: BLE001
                pass
            if verdict != "ok":
                if verdict == "refused":
                    refusal = got[-1].data
                if not wb.closed:
                    await wb.close()
                print(f"resume {sid}: {LB[i]['name']} {verdict or 'no answer'}", flush=True)
                continue
            OWNER[i] = pin                                               # the old connection's later release() is then a no-op
            pin.i = i
            HOME[sid] = (i, pin.client)
            print(f"resume {sid} ({pin.client}) -> {LB[i]['name']} {LB[i].get('host') or '127.0.0.1'}:{LB[i]['port']}"
                  + ("" if home is not None else " (session id unknown to this proxy: found by asking)"), flush=True)
            try:
                for m in got:
                    await (wc.send_str(m.data) if m.type == WSMsgType.TEXT else wc.send_bytes(m.data))
                b2c_text, c2b_text = hooks(pin, i, M_OF[i])
                pst = {}
                await pump_pair(wc, wb, b2c_text, c2b_text, pst)
                print(f"session end (resumed {sid}, {pin.client}, {LB[i]['name']}): {time.time() - pst.get('t0', time.time()):.1f} s; "
                      f"client->backend {pst.get('c_n')} msgs {pst.get('c_bytes', 0) / 48000:.1f} s audio, longest gap {pst.get('c_gap', 0):.1f} s; "
                      f"backend->client {pst.get('b_n')} msgs, longest gap {pst.get('b_gap', 0):.1f} s", flush=True)
            finally:
                release(pin)
            return wc
        if refusal is not None:
            await wc.send_str(refusal)
            await wc.close(code=4404, message=b"no such session")
        else:
            await wc.close(code=1013, message=b"resume: backend not reachable")
        return wc

    async def wsh_lb(req):
        if not ok(req):
            return web.Response(status=403, text="forbidden")
        wc = web.WebSocketResponse(max_msg_size=8 << 20, heartbeat=20)
        await wc.prepare(req)
        if req.query.get("proxytest") == "down":
            await wc.send_json({"type": "error", "msg": MSG})
            await wc.close(code=1013, message=MSG.encode())
            return wc
        start = resume = None                                            # 1. wait for the client's start (pings answered here;
        try:                                                             #    anything else before it is dropped, as the backend would)
            while start is None and resume is None:
                m = await wc.receive(timeout=120)
                if m.type == WSMsgType.TEXT:
                    try:
                        d = json.loads(m.data)
                    except ValueError:
                        continue
                    if isinstance(d, dict) and d.get("type") == "start":
                        start = d
                    elif isinstance(d, dict) and d.get("type") == "resume":     # 10-25: to the backend running that session
                        resume = d
                    elif isinstance(d, dict) and d.get("type") == "ping":
                        await wc.send_str(json.dumps({"type": "pong", "t": d.get("t")}))
                elif m.type == WSMsgType.BINARY:
                    continue
                else:
                    return wc
        except asyncio.TimeoutError:
            await wc.close()
            return wc
        if resume is not None:
            return await resume_lb(req, wc, resume)
        pin = Pin(str(start.get("client") or ""))
        model = want_model(start.pop("model", None) or req.query.get("model"))
        NM = M_OF.count(model)
        try:
            async with LOCK:                                             # 2. pick a backend of the model (serialised, fresh /healthz of every backend)
                st = await probe_all()
                idle = idle_list(st, model)
                pick = idle[0] if idle else None
                steal = pick is None                                     # 10-26: any pick below takes a BUSY backend over
                if pick is None and pin.client in REAL:                # a real user never waits behind a test session
                    tk = [i for i in range(N) if M_OF[i] == model and st[i]["ready"] and OWNER[i] is not None and OWNER[i].client not in REAL]
                    tk += [i for i in range(N) if M_OF[i] == model and st[i]["ready"] and OWNER[i] is None and i not in tk]   # 10-26: busy without a
                    pick = tk[0] if tk else None                         # proxy pin = a direct (agent) connection -> taken over too
                if pick is None and model in MHIDDEN and pin.client in REAL:   # 10-26 USER RULE "steal the server": a new playground call ALWAYS
                    tk = [i for i in range(N) if M_OF[i] == model and st[i]["ready"]]   # takes its backend, ending whatever holds it (test or older call)
                    pick = tk[0] if tk else None
                    if pick is not None:
                        print(f"session ({pin.client}, model {model}) STEALS {LB[pick]['name']} from {OWNER[pick].client if OWNER[pick] else 'an unpinned session'}", flush=True)
                if (pick is None and model in MHIDDEN and pin.client in REAL
                        and not any(st[k]["ready"] for k in range(N) if M_OF[k] == model)):   # 10-26 main: the playground's own backend is DOWN
                    fb = [i for i in range(N) if M_OF[i] not in MHIDDEN and st[i]["ready"] and not st[i]["busy"] and OWNER[i] is None]
                    # 10-26 main (user got 4 refusals while the instance was still busy ~40 s after their call): also take a demo instance held
                    # ONLY by a lingering session (busy with no proxy pin: detached / resume wait / post-call save) or by a test session, like
                    # the P steal rule (the start goes as a 'browser' takeover). Never one in a real user's call (OWNER client in REAL).
                    fb += [i for i in range(N) if M_OF[i] not in MHIDDEN and st[i]["ready"] and st[i]["busy"]
                           and (OWNER[i] is None or OWNER[i].client not in REAL) and i not in fb]
                    pick = fb[0] if fb else None                         # -> an idle demo instance first, else a lingering/test-held one
                    if pick is not None:
                        print(f"session ({pin.client}, model {model}) falls back to {LB[pick]['name']} (playground backend down"
                              + ("; takes over a lingering session)" if st[pick]["busy"] else ")"), flush=True)
                if pick is not None:
                    OWNER[pick] = pin
                    pin.i = pick
            if pick is None:
                if any(s["ready"] for k, s in enumerate(st) if M_OF[k] == model):
                    msg = f"all {NM} slots busy, try again in a minute" + ("" if pin.client == "browser" else " (automated clients never take over)")   # 10-19 user wording
                    await wc.send_json({"type": "error", "msg": msg})
                    await wc.close(code=1013, message=b"busy")
                else:
                    states = sorted({s["state"] for s in st})
                    await wc.send_json({"type": "error", "msg": (f"model {' / '.join(map(str, states))}; try again in about a minute"
                                                                  if any(s["up"] for s in st) else MSG)})
                    await wc.close(code=1013, message=MSG.encode())
                print(f"session ({pin.client}) refused: {[(s['state'], s['busy']) for s in st]} pinned {[o and o.client for o in OWNER]}", flush=True)
                return wc
            clip = str(start.get("clip") or "")
            if clip.startswith("upload:"):                               # 3. per-session upload -> the id this backend knows
                start["clip"] = await upload_for(clip[7:], pick)
            try:                                                         # 4. connect, tell the page which instance, forward the start
                wb = await app["cs"].ws_connect(bk(pick) + str(req.rel_url.update_query(token=token)), max_msg_size=8 << 20, heartbeat=20)
            except Exception:  # noqa: BLE001
                await wc.send_json({"type": "error", "msg": MSG})
                await wc.close(code=1013, message=MSG.encode())
                return wc
            print(f"session ({pin.client}, model {model}) -> {LB[pick]['name']} {LB[pick].get('host') or '127.0.0.1'}:{LB[pick]['port']}", flush=True)
            mi = [k for k in range(N) if M_OF[k] == model] or [pick]
            if pick not in mi:                                           # 10-26: a fallback pick (another model's backend)
                mi = mi + [pick]
            await wc.send_json({"type": "route", "instance": mi.index(pick) + 1, "of": NM, "name": LB[pick]["name"], "model": model, "model_label": MLABEL[model]})
            if steal and pin.client in REAL:                             # the backend only takes over for client 'browser' (its takeover path)
                start = dict(start, client="browser", takeover_by=pin.client)
            await wb.send_str(json.dumps(start))
            b2c_text, c2b_text = hooks(pin, pick, model)
            pst = {}
            await pump_pair(wc, wb, b2c_text, c2b_text, pst)
            d = time.time() - pst.get("t0", time.time())
            print(f"session end ({pin.client}, model {model}, {LB[pick]['name']}): {d:.1f} s; client->backend {pst.get('c_n')} msgs {pst.get('c_bytes', 0) / 48000:.1f} s audio, "
                  f"longest gap {pst.get('c_gap', 0):.1f} s at t={pst.get('c_gap_at', 0):.1f}, last msg {time.time() - pst.get('c_last', time.time()):.1f} s before the end; "
                  f"backend->client {pst.get('b_n')} msgs, longest gap {pst.get('b_gap', 0):.1f} s at t={pst.get('b_gap_at', 0):.1f}; "
                  f"slowest send to the backend {pst.get('send_max', 0):.2f} s at t={pst.get('send_max_at', 0):.1f}, max queued {pst.get('q_max', 0)}", flush=True)
            return wc
        finally:
            release(pin)

    async def health_lb(req):
        if not ok(req):
            return web.Response(status=403, text="forbidden")
        st = await probe_all()
        rows = [dict(name=b["name"], port=b["port"], host=b.get("host") or "127.0.0.1", model=M_OF[i], **st[i],
                     pinned=(OWNER[i].client if OWNER[i] is not None else None)) for i, b in enumerate(LB)]
        return web.json_response({"proxy": "ok", "lb": True, "idle": [i + 1 for i in idle_list(st)], "backends": rows,
                                  "backend": next((r for r in rows if r["ready"]), rows[0])})

    # 10-25 user ("stop asking for permissions every call"): videort's avatar player SAME-ORIGIN under /avatar/ (HTTP + WebSocket passthrough
    # to its front on this node), so the page can record the avatar from the player's canvas instead of a per-call "share this tab" prompt.
    AVATAR_UPS = {"avatar": os.environ.get("PPMLIVE_AVATAR_UPSTREAM", "http://127.0.0.1:8960"),
                  "avatar2": os.environ.get("PPMLIVE_AVATAR2_UPSTREAM", "http://127.0.0.1:8940"), "avatar3": os.environ.get("PPMLIVE_AVATAR3_UPSTREAM", "http://127.0.0.1:8945")   # 10-26 videort front P (the playground's live avatar pool)
                  }   # 10-26: instance 2's own avatar front (videort)

    async def avatar(req):
        tail = req.match_info.get("tail", "")
        url = f"{AVATAR_UPS[req.match_info.get('av', 'avatar')]}/{tail}" + (("?" + req.query_string) if req.query_string else "")
        hop = {"host", "content-length", "connection", "upgrade", "transfer-encoding", "sec-websocket-key", "sec-websocket-version",
               "sec-websocket-extensions", "sec-websocket-protocol", "accept-encoding"}
        if req.headers.get("Upgrade", "").lower() == "websocket":
            wc = web.WebSocketResponse(max_msg_size=64 << 20, heartbeat=20)
            await wc.prepare(req)
            try:
                wb = await req.app["cs"].ws_connect(url.replace("http", "ws", 1), max_msg_size=64 << 20, heartbeat=20)
            except Exception:  # noqa: BLE001
                await wc.close(code=1013, message=b"avatar player unavailable")
                return wc
            await pump_pair(wc, wb)
            return wc
        try:
            async with req.app["cs"].request(req.method, url, headers={k: v for k, v in req.headers.items() if k.lower() not in hop},
                                             data=(await req.read()) if req.can_read_body else None, allow_redirects=False) as r:
                body = await r.read()
                hdr = {k: v for k, v in r.headers.items() if k.lower() not in ("content-length", "transfer-encoding", "content-encoding", "connection")}
                return web.Response(status=r.status, body=body, headers=hdr)
        except Exception as e:  # noqa: BLE001
            return web.Response(status=502, text=f"avatar player unavailable: {type(e).__name__}")

    app.router.add_route("*", "/{av:avatar[23]?}", lambda req: web.HTTPFound("/" + req.match_info["av"] + "/" + (("?" + req.query_string) if req.query_string else "")))
    app.router.add_route("*", "/{av:avatar[23]?}/{tail:.*}", avatar)
    # 10-27 USER: "destroy that page so you never make that mistake again": the live demo page is gone; its human-facing paths redirect to
    # the playground's duplex view. The APIs / WS / clipwav / avatar mounts the playground Live mic and the speech twin use stay as they were.
    PLAYGROUND = "https://california-side-enquiry-bryan.trycloudflare.com/?model=duplex"

    def playground_url():                                               # 10-28 duplexhost: the CURRENT playground URL (genweb/URL.txt, line 1) at each request; the old constant only as a fallback
        try:
            u = open(f"{HERE}/genweb/URL.txt").read().split()[0].strip().rstrip("/")
            if u.startswith("https://"):
                return u + "/?model=duplex"
        except (OSError, IndexError):
            pass
        return PLAYGROUND
    for pth in ("/", "/index.html"):
        app.router.add_get(pth, lambda req: web.HTTPFound(playground_url()))
    app.router.add_get("/clipwav", clipwav)
    app.router.add_get("/proxyhealth", health_lb if LB else health)
    app.router.add_get("/ws", wsh_lb if LB else wsh)
    app.router.add_route("*", "/{tail:.*}", http_lb if LB else http)
    web.run_app(app, host="127.0.0.1", port=a.port, access_log=None, print=None, reuse_address=True)


if __name__ == "__main__":
    main()
