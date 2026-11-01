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
    next on connection error; /api/presets 'busy' = no idle backend, plus 'lb' {instances, idle, served_by}. /proxyhealth lists all."""
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
<title>Aloe Voxtral Live</title><style>:root{--bg:#fbfaf7;--fg:#1d1d1b}@media (prefers-color-scheme:dark){:root{--bg:#171716;--fg:#ecebe6}}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,sans-serif}main{max-width:640px;margin:0 auto;padding:40px 16px}</style>
</head><body><main><h1>Aloe Voxtral: live voice</h1><p>The model server is restarting, back in about 60 s. This page reloads by itself.</p>
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
        # 10-27 (main: paused models must never show an error): PAUSED.html next to this file, if present, replaces the "restarting" page
        # while the GPUs train (read per request; restore = delete it, no restart). Absent -> the 10-24 behaviour, unchanged.
        pf = f"{HERE}/PAUSED.html"
        if os.path.isfile(pf):
            if req.path.startswith("/api/"):
                return web.json_response({"error": "paused for training jobs, back later", "state": "paused"}, status=503)
            return web.Response(text=open(pf).read(), content_type="text/html", headers={"Cache-Control": "no-store"})
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
                    if on_b2c_text is not None:
                        on_b2c_text(m.data)
                    await wc.send_str(m.data)
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
        await pump_pair(wc, wb)
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
            if re.fullmatch(r"preset:aloe-\d+-\d+-[01]", cid):
                # 10-24: the instance's own <state dir>/clips (e.g. the sam-sep presets' decodes) first, else the shared clips/
                own = f"{os.path.dirname((a.sessions or HERE + '/sessions').rstrip('/'))}/clips/{cid[7:]}.wav"
                data = open(own if os.path.isfile(own) else f"{HERE}/clips/{cid[7:]}.wav", "rb").read()
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
                    j["models"] = [{"id": m, "label": MLABEL[m], "instances": M_OF.count(m), "idle": len(idle_list(st, m))} for m in MODELS]
                    data = json.dumps(j).encode()
                except ValueError:
                    pass
            return web.Response(status=status, body=data, headers=h)
        return down_response(req)

    async def wsh_lb(req):
        if not ok(req):
            return web.Response(status=403, text="forbidden")
        wc = web.WebSocketResponse(max_msg_size=8 << 20, heartbeat=20)
        await wc.prepare(req)
        if req.query.get("proxytest") == "down":
            await wc.send_json({"type": "error", "msg": MSG})
            await wc.close(code=1013, message=MSG.encode())
            return wc
        start = None                                                     # 1. wait for the client's start (pings answered here;
        try:                                                             #    anything else before it is dropped, as the backend would)
            while start is None:
                m = await wc.receive(timeout=120)
                if m.type == WSMsgType.TEXT:
                    try:
                        d = json.loads(m.data)
                    except ValueError:
                        continue
                    if isinstance(d, dict) and d.get("type") == "start":
                        start = d
                    elif isinstance(d, dict) and d.get("type") == "ping":
                        await wc.send_str(json.dumps({"type": "pong", "t": d.get("t")}))
                elif m.type == WSMsgType.BINARY:
                    continue
                else:
                    return wc
        except asyncio.TimeoutError:
            await wc.close()
            return wc
        pin = Pin(str(start.get("client") or ""))
        model = want_model(start.pop("model", None) or req.query.get("model"))
        NM = M_OF.count(model)
        try:
            async with LOCK:                                             # 2. pick a backend of the model (serialised, fresh /healthz of every backend)
                st = await probe_all()
                idle = idle_list(st, model)
                pick = idle[0] if idle else None
                if pick is None and pin.client == "browser":            # a real user never waits behind a test session
                    tk = [i for i in range(N) if M_OF[i] == model and st[i]["ready"] and OWNER[i] is not None and OWNER[i].client != "browser"]
                    pick = tk[0] if tk else None
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
            mi = [k for k in range(N) if M_OF[k] == model]
            await wc.send_json({"type": "route", "instance": mi.index(pick) + 1, "of": NM, "name": LB[pick]["name"], "model": model, "model_label": MLABEL[model]})
            await wb.send_str(json.dumps(start))

            def b2c_text(s):                                             # release the pin once the backend's session is over
                if s.startswith('{"type": "f"'):
                    return
                try:
                    t = json.loads(s).get("type")
                except (ValueError, AttributeError):
                    return
                if t in TERMINAL:
                    release(pin)

            def c2b_text(s):                                             # a further start on this connection re-pins if still free
                if s.startswith('{"type":"diag"') or s.startswith('{"type": "diag"'):   # 10-25 page diagnostics: logged here, not forwarded
                    print(f"diag ({pin.client}, model {model}, {LB[pick]['name']}): {s[:600]}", flush=True)
                    return True
                if '"start"' in s and OWNER[pick] is None:
                    try:
                        if json.loads(s).get("type") == "start":
                            OWNER[pick] = pin
                    except (ValueError, AttributeError):
                        pass

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

    # 10-27 user ("are we saving history" -> "yes" to a History tab): every call the backends of this front saved, newest first.
    # Read-only over the backends' sessions dirs: the client names a call only by (b = index into HDIRS, sid = a session-dir name).
    HDIRS = ([(b.get("name") or f"backend {k + 1}", b["sessions"]) for k, b in enumerate(LB) if b.get("sessions")]
             if LB else [("main", a.sessions or HERE + "/sessions")])
    try:                                                                   # optional older read-only dirs: [[name, sessions dir], ...]
        HDIRS += [(str(n), str(p)) for n, p in json.load(open(f"{HERE}/history_extra.json"))
                  if os.path.isdir(p) and os.path.realpath(p) not in [os.path.realpath(x) for _, x in HDIRS]]
    except (OSError, ValueError):
        pass
    SID_RE = re.compile(r"^\d{8}-\d{6}$")
    HCACHE = {}

    def npy_len(z, name):
        """sample count of one 1-D float member of a session.npz, from its .npy header (no decompression of the data)."""
        import ast
        with z.open(name) as f:
            major = f.read(8)[6]                                           # magic \x93NUMPY, major, minor
            hl = int.from_bytes(f.read(2 if major == 1 else 4), "little")
            d = ast.literal_eval(f.read(hl).decode("latin1"))
        return int(d["shape"][0]) if d.get("shape") else 0

    # 10-27 user ("cant see integrated lufs in history"): BS.1770 integrated loudness (pyloudnorm, as the training-data measurement) of
    # each saved call's two tracks, cached in history_lufs.json by session dir + npz mtime (precomputed offline; new calls on first open).
    LCACHE_F, LLOCK = f"{HERE}/history_lufs.json", __import__("threading").Lock()
    try:
        LCACHE = json.load(open(LCACHE_F))
    except (OSError, ValueError):
        LCACHE = {}

    def hist_lufs_calc(k, sid):
        d = f"{HDIRS[k][1]}/{sid}"
        mt = os.stat(f"{d}/session.npz").st_mtime
        c = LCACHE.get(d)
        if c and c.get("mt") == mt:
            return c
        import numpy as np
        import pyloudnorm as pyln
        with np.load(f"{d}/session.npz") as z:
            A, Bt = z["a"].astype(np.float64).reshape(-1), z["b"].astype(np.float64).reshape(-1)
        meter = pyln.Meter(24000)

        def L(x):
            if len(x) < 24000:
                return None
            v = float(meter.integrated_loudness(x))
            return round(v, 2) if np.isfinite(v) else None
        j = json.load(open(f"{d}/session.json"))
        cfg = j.get("cfg") or {}
        gdb = float(cfg.get("a_gain", cfg.get("a_gain_db", 0)) or 0)
        on = bool(cfg.get("a_lufs_on", cfg.get("a_lufs_match", False)))
        la = L(A)
        c = {"mt": mt, "a": la, "b": L(Bt), "gain_db": gdb, "match_on": on,
             "gain_changed": any("a_gain" in json.dumps(x) or "a_lufs" in json.dumps(x) for x in (j.get("param_log") or [])),
             "a_raw": round(la - gdb, 2) if la is not None and not on else None}   # saved mic track = after A gain (as the model heard it)
        with LLOCK:
            LCACHE[d] = c
            tmp = LCACHE_F + ".tmp"
            json.dump(LCACHE, open(tmp, "w"))
            os.replace(tmp, LCACHE_F)
        return c

    def hist_entry(k, sid):
        d = f"{HDIRS[k][1]}/{sid}"
        try:
            mt = os.stat(f"{d}/session.json").st_mtime
        except OSError:
            return None
        c = HCACHE.get((k, sid))
        if c and c[0] == mt:
            return c[1]
        e = None
        try:
            j = json.load(open(f"{d}/session.json"))
            cfg = j.get("cfg") or {}
            if cfg.get("client") != "test" and not cfg.get("hidden"):
                import zipfile
                with zipfile.ZipFile(f"{d}/session.npz") as z:
                    n = max(npy_len(z, "a.npy"), npy_len(z, "b.npy"))
                w = str(j.get("B_words") or "")
                e = dict(b=k, backend=HDIRS[k][0], sid=sid, label=j.get("label") or "", preset=cfg.get("preset") or "", clip=cfg.get("clip") or "",
                         persona=bool(j.get("persona")), s=round(n / 24000, 1), words=w[:240], nwords=len(w.split()), end_reason=j.get("end_reason"))
        except Exception as ex:  # noqa: BLE001
            e = dict(b=k, backend=HDIRS[k][0], sid=sid, label="", preset="", clip="", persona=False, s=None, words="", nwords=0,
                     end_reason=None, error=repr(ex)[:200])
        HCACHE[(k, sid)] = (mt, e)
        return e

    def hist_all():
        out = []
        for k, (_, sd) in enumerate(HDIRS):
            try:
                names = [x for x in os.listdir(sd) if SID_RE.match(x)]
            except OSError:
                continue
            out += [dict(e, lufs=LCACHE.get(f"{sd}/{e['sid']}")) for e in (hist_entry(k, x) for x in names) if e]
        out.sort(key=lambda e: (e["sid"], e["b"]), reverse=True)
        return out

    def hist_ok(req):
        k, sid = req.query.get("b", ""), req.query.get("sid", "")
        return (int(k), sid) if k.isdigit() and int(k) < len(HDIRS) and SID_RE.match(sid) and os.path.isdir(f"{HDIRS[int(k)][1]}/{sid}") else None

    def wav_f32(x):
        """32-bit IEEE float mono WAV, 24 kHz, the samples unchanged (as the Last call tab writes)."""
        import struct
        n = len(x)
        h = (b"RIFF" + struct.pack("<I", 50 + n * 4) + b"WAVE" + b"fmt " + struct.pack("<IHHIIHHH", 18, 3, 1, 24000, 96000, 4, 32, 0)
             + b"fact" + struct.pack("<II", 4, n) + b"data" + struct.pack("<I", n * 4))
        return h + x.tobytes()

    def hist_wav_body(k, sid, ch):
        import numpy as np
        with np.load(f"{HDIRS[k][1]}/{sid}/session.npz") as z:
            A, Bt = z["a"].astype(np.float32).reshape(-1), z["b"].astype(np.float32).reshape(-1)
        if ch == "a":
            return wav_f32(A)
        if ch == "b":
            return wav_f32(Bt)
        M = np.zeros(max(len(A), len(Bt)), np.float32)                     # the plain sum, no scaling or clipping
        M[:len(A)] += A
        M[:len(Bt)] += Bt
        return wav_f32(M)

    async def hist_list(req):
        rows = await asyncio.get_running_loop().run_in_executor(None, hist_all)
        return web.json_response({"calls": rows, "backends": [n for n, _ in HDIRS]}, headers={"Cache-Control": "no-store"})

    async def hist_json(req):
        kv = hist_ok(req)
        if kv is None:
            return web.Response(status=404, text="no such call")
        body = await asyncio.get_running_loop().run_in_executor(None, lambda: open(f"{HDIRS[kv[0]][1]}/{kv[1]}/session.json", "rb").read())
        return web.Response(body=body, content_type="application/json", headers={"Cache-Control": "no-store"})

    async def hist_lufs(req):
        kv = hist_ok(req)
        if kv is None:
            return web.Response(status=404, text="no such call")
        try:
            c = await asyncio.get_running_loop().run_in_executor(None, hist_lufs_calc, kv[0], kv[1])
        except Exception as ex:  # noqa: BLE001
            return web.json_response({"error": repr(ex)[:300]}, status=500)
        return web.json_response(c, headers={"Cache-Control": "no-store"})

    async def hist_wav(req):
        kv, ch = hist_ok(req), req.query.get("ch", "mix")
        if kv is None or ch not in ("a", "b", "mix"):
            return web.Response(status=404, text="no such call")
        body = await asyncio.get_running_loop().run_in_executor(None, hist_wav_body, kv[0], kv[1], ch)
        name = f"call-{kv[1]}-{HDIRS[kv[0]][0].replace(' ', '')}-{dict(a='you', b='model', mix='both')[ch]}.wav"
        return web.Response(body=body, content_type="audio/wav",
                            headers={"Cache-Control": "no-store", "Content-Disposition": f'inline; filename="{name}"'})

    app.router.add_get("/history/list", hist_list)
    app.router.add_get("/history/json", hist_json)
    app.router.add_get("/history/wav", hist_wav)
    app.router.add_get("/history/lufs", hist_lufs)
    app.router.add_get("/clipwav", clipwav)
    app.router.add_get("/proxyhealth", health_lb if LB else health)
    app.router.add_get("/ws", wsh_lb if LB else wsh)
    app.router.add_route("*", "/{tail:.*}", http_lb if LB else http)
    web.run_app(app, host="127.0.0.1", port=a.port, access_log=None, print=None, reuse_address=True)


if __name__ == "__main__":
    main()
