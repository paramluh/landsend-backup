#!/usr/bin/env python3
"""10-26 (main: the user got a Cloudflare 502 during a web-server restart): the playground's ALWAYS-UP front on :9205. It never restarts for
a deploy; the cloudflared tunnel points at it. It forwards every HTTP request (any method, Range for video, streamed bodies) to the app
server whose port is in genweb/APP_PORT (re-read when the file changes). A deploy (app_sup.sh) starts the new app on the other port, waits
for its health, rewrites APP_PORT, then drains the old one, so no request ever hits a closed port. If the current port refuses, the other
port is tried once; only if both fail does the user get a short 'restarting' page that retries by itself.
  front.py [--port 9205]"""
import argparse
import asyncio
import os
import socket
import time

import aiohttp
from aiohttp import web

HERE = os.path.dirname(os.path.abspath(__file__))
PORTF = f"{HERE}/APP_PORT"
PORTS = (9206, 9207)
HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailers", "transfer-encoding", "upgrade", "host",
       "content-length"}
_P = {"m": None, "p": PORTS[0]}
RESTARTING = ("<!doctype html><meta charset=utf-8><title>Playground restarting</title><body style='font:16px system-ui;background:#111;"
              "color:#ddd;padding:40px'>The playground server is restarting, back in a few seconds. This page reloads by itself."
              "<script>setTimeout(()=>location.reload(),3000)</script>")


def app_port():
    try:
        m = os.path.getmtime(PORTF)
        if m != _P["m"]:
            _P["p"], _P["m"] = int(open(PORTF).read().split()[0]), m
    except (OSError, ValueError, IndexError):
        pass
    return _P["p"]


async def handle(req):
    body = await req.read() if req.can_read_body else None
    hdr = {k: v for k, v in req.headers.items() if k.lower() not in HOP}
    first = app_port()
    for port in (first, PORTS[1] if first == PORTS[0] else PORTS[0]):
        url = f"http://127.0.0.1:{port}{req.rel_url}"
        try:
            up = await req.app["cs"].request(req.method, url, headers=hdr, data=body, allow_redirects=False,
                                             timeout=aiohttp.ClientTimeout(total=None, sock_connect=3, sock_read=900))
        except (aiohttp.ClientConnectorError, ConnectionRefusedError, asyncio.TimeoutError):
            continue
        try:
            resp = web.StreamResponse(status=up.status, reason=up.reason)
            for k, v in up.headers.items():
                if k.lower() not in HOP:
                    resp.headers[k] = v
            if up.headers.get("Content-Length") and up.status != 304:
                resp.content_length = int(up.headers["Content-Length"])
            await resp.prepare(req)
            async for chunk in up.content.iter_chunked(1 << 16):
                await resp.write(chunk)
            await resp.write_eof()
            return resp
        except (ConnectionResetError, aiohttp.ClientError):
            return resp if "resp" in locals() else web.Response(status=502)
        finally:
            up.release()
    if req.path.startswith("/api/") or req.method != "GET":
        return web.json_response({"error": "the playground server is restarting, retry in a few seconds"}, status=503,
                                 headers={"Retry-After": "3", "Cache-Control": "no-store"})
    return web.Response(text=RESTARTING, content_type="text/html", status=503, headers={"Cache-Control": "no-store"})


async def on_start(app):
    app["cs"] = aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0, force_close=False), auto_decompress=False)


async def on_stop(app):
    await app["cs"].close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=9205)
    a = ap.parse_args()
    app = web.Application(client_max_size=1 << 30)
    app.on_startup.append(on_start)
    app.on_cleanup.append(on_stop)
    app.router.add_route("*", "/{tail:.*}", handle)
    while True:                                                      # bind as soon as the old server releases the port (cut-over)
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("127.0.0.1", a.port))
            break
        except OSError:
            s.close()
            time.sleep(0.05)
    s.listen(1024)
    print(f"[front] serving :{a.port} -> app port from {PORTF} ({app_port()})", flush=True)
    web.run_app(app, sock=s, print=None, access_log=None)


if __name__ == "__main__":
    main()
