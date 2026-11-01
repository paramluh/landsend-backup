#!/usr/bin/env python3
"""FEED for videort (aloe-posedup-live, 10-25): the backend is a WebSocket CLIENT of videort's real-time video model's feed endpoint
(default ws://slinky-10:8990/feed; server.py: {STATE}/FEED_URL.txt first line, else env PPMLIVE_FEED_URL; 'off' / empty = disabled) and
sends one JSON text message per message, immediately (no batching):
  {"type": "start", "session": id, "model": "posedup", "voice": <preset id>, "fps": 12.5[, "person": <demo person id>]}
  {"type": "f", "session": id, "f": n, "t": unix_s, "pose": [32 ints], "mimi": [32 ints], "pcm_b64": <int16 LE, 1920 samples @ 24 kHz>}
      every code frame n = 0, 1, ... (step frame - the 2-frame code delay; pose and Mimi share n), listening frames included
  {"type": "end", "session": id}
NEVER BLOCKS THE SESSION: put() only hands the (already serialised) message to this module's own thread / asyncio loop
(loop.call_soon_threadsafe); the loop keeps a BOUNDED queue (drops the OLDEST frame when full = a slow endpoint), connects lazily, sends with
a 1 s timeout per message, and on any failure closes, drops what cannot be sent and retries the connection at most every 2 s (endpoint down:
every frame dropped, counted, logged at most every 10 s). After a (re)connect mid-session the session's 'start' is re-sent first.
10-29 openfix: the socket is READ by a drain task (answers the server's heartbeat pings, sees its close at once), and a failed send
reconnects immediately and re-sends (start first) instead of losing the message.
FeedLog: the session's feed messages, all of them (independent of the endpoint), appended to <session dir>/feed.jsonl by a writer thread."""
import asyncio
import os
import queue
import threading
import time

import aiohttp

DEFAULT_URL = "ws://slinky-10:8990/feed"


class Feed:
    def __init__(self, url, log, qmax=64, hold=False):
        self.url = (url or "").strip()
        self.hold = bool(hold)                                # 10-29 openfix: True (record feeds) = while the endpoint is unreachable keep
        self._retry = None                                    # the messages (bounded by qmax, oldest frame dropped) and send them, start first, once it answers
        self.log = log
        self.qmax = int(qmax)
        self.stats = {"sent": 0, "dropped": 0, "connects": 0, "errors": 0}
        self._sess = {}                                       # per-session counters {session: {"sent", "dropped"}}
        self.cur_start = None                                 # (session, serialised start) of the active session
        self.ws = None
        self.next_try = 0.0
        self.last_log = 0.0
        self.ready = threading.Event()
        self.th = threading.Thread(target=self._thread, name="feed", daemon=True)
        self.th.start()
        self.ready.wait(10)

    # ---------------------------------------------------------------- called from the session thread (never blocks)
    def enabled(self):
        return bool(self.url) and self.url.lower() != "off"

    def set_url(self, url):
        url = (url or "").strip()
        if url != self.url:
            self.log(f"feed: endpoint {self.url or 'off'} -> {url or 'off'}")
            self.url = url
            self.loop.call_soon_threadsafe(self._drop_ws)

    def put(self, kind, session, text):
        if not self.enabled():
            return
        try:
            self.loop.call_soon_threadsafe(self._enqueue, (kind, session, text))
        except RuntimeError:                                  # loop gone (interpreter shutdown)
            pass

    def session_stats(self, session):
        return dict(self._sess.get(session, {"sent": 0, "dropped": 0}), endpoint=self.url or "off")

    # ---------------------------------------------------------------- the feed thread / loop
    def _thread(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.q = asyncio.Queue()
        self.ready.set()
        self.loop.run_until_complete(self._main())

    def _count(self, session, k):
        self.stats[k] += 1
        s = self._sess.setdefault(session, {"sent": 0, "dropped": 0})
        s[k] = s.get(k, 0) + 1
        if len(self._sess) > 64:
            self._sess.pop(next(iter(self._sess)))

    def _enqueue(self, item):
        kind, session, text = item
        if kind == "start":
            self.cur_start = (session, text)
        elif kind == "end" and self.cur_start and self.cur_start[0] == session:
            self.cur_start = None
        while self.q.qsize() >= self.qmax:                    # slow / stalled endpoint: drop the OLDEST frame (start / end are tiny, kept
            old = self.q.get_nowait()                         # by preference: re-queued at the back if they were the oldest)
            if old[0] == "f":
                self._count(old[1], "dropped")
                self._note("queue full (endpoint slow)")
            else:
                self.q.put_nowait(old)
                if self.q.qsize() > 2 * self.qmax:
                    break
        self.q.put_nowait(item)

    def _note(self, why):
        now = time.time()
        if now - self.last_log > 10:
            self.last_log = now
            self.log(f"feed: dropping frames ({why}); totals {self.stats} endpoint {self.url or 'off'}")

    def _drop_ws(self):
        ws, self.ws = self.ws, None
        if ws is not None and not ws.closed:
            asyncio.ensure_future(ws.close())
        self.next_try = 0.0

    async def _send(self, text):
        await asyncio.wait_for(self.ws.send_str(text), 1.0)

    async def _drain(self, ws):
        """10-29 openfix: READ the socket (and discard what the endpoint sends: acks, pongs). aiohttp answers the server's heartbeat PINGs
        and handles its CLOSE only inside receive(): a never-read socket idle between sessions was dropped by the server's 20 s heartbeat
        ~30 s after the last message, still looked open here, and the next session's 'start' + first frames died in the closing socket."""
        try:
            async for _ in ws:
                pass
        except Exception:  # noqa: BLE001
            pass

    async def _connect(self, cs):
        self.ws = await asyncio.wait_for(cs.ws_connect(self.url, heartbeat=20, max_msg_size=8 << 20, autoping=True), 3.0)
        self.stats["connects"] += 1
        self.log(f"feed: connected to {self.url}")
        self._reader = asyncio.ensure_future(self._drain(self.ws))         # (held: the loop keeps only weak task refs)

    async def _main(self):
        async with aiohttp.ClientSession() as cs:
            while True:
                if self._retry is not None:
                    (kind, session, text), self._retry = self._retry, None
                else:
                    kind, session, text = await self.q.get()
                if not self.enabled():
                    continue
                if self.ws is None or self.ws.closed:
                    if time.time() < self.next_try:
                        if self.hold:                                   # keep it (the queue behind it keeps the rest, bounded)
                            self._retry = (kind, session, text)
                            await asyncio.sleep(max(0.05, self.next_try - time.time()))
                            continue
                        if kind == "f":
                            self._count(session, "dropped")
                            self._note("endpoint down")
                        continue
                    try:
                        await self._connect(cs)
                        if kind != "start" and self.cur_start is not None:      # mid-session (re)connect: the session's start first
                            await self._send(self.cur_start[1])
                    except Exception as e:  # noqa: BLE001
                        self.ws = None
                        self.next_try = time.time() + 2.0
                        self.stats["errors"] += 1
                        if self.hold:
                            self._retry = (kind, session, text)
                            if time.time() - self.last_log > 10:
                                self.last_log = time.time()
                                self.log(f"feed: connect failed ({type(e).__name__} {str(e)[:120]}): holding the messages, retrying every 2 s; totals {self.stats}")
                            continue
                        if kind == "f":
                            self._count(session, "dropped")
                        self._note(f"connect failed: {type(e).__name__} {str(e)[:120]}")
                        continue
                try:
                    await self._send(text)
                    if kind == "f":
                        self._count(session, "sent")
                except Exception as e:  # noqa: BLE001
                    self.stats["errors"] += 1
                    self.log(f"feed: send failed ({type(e).__name__} {str(e)[:120]}): reconnecting now and re-sending ({kind})")
                    self._drop_ws()
                    try:                                                # 10-29 openfix: a stale socket must not cost the message (a lost
                        await self._connect(cs)                         # 'start' + 2 s backoff dropped a session's first 15 frames):
                        if kind != "start" and self.cur_start is not None:      # reconnect at once, the session's start first, then it
                            await self._send(self.cur_start[1])
                        await self._send(text)
                        if kind == "f":
                            self._count(session, "sent")
                        continue
                    except Exception as e2:  # noqa: BLE001
                        self._drop_ws()
                        e = e2
                    if self.hold:
                        self._retry = (kind, session, text)
                        self.next_try = time.time() + 2.0
                        continue
                    if kind == "f":
                        self._count(session, "dropped")
                    self._note(f"send failed: {type(e).__name__} {str(e)[:120]}")
                    self.next_try = time.time() + 2.0


class FeedLog:
    """append every feed message of one session to <dir>/feed.jsonl from a writer thread (unbounded queue: the session never waits on disk)."""

    def __init__(self, path, log):
        self.path, self.log = path, log
        self.q = queue.Queue()
        self.n = 0
        self.th = threading.Thread(target=self._run, name="feedlog", daemon=True)
        self.th.start()

    def put(self, text):
        self.q.put(text)

    def close(self, timeout=30):
        self.q.put(None)
        self.th.join(timeout)
        return self.n

    def _run(self):
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(self.path, "w") as fh:
                last = time.time()
                while True:
                    try:
                        x = self.q.get(timeout=1.0)
                    except queue.Empty:
                        fh.flush(); last = time.time()
                        continue
                    if x is None:
                        break
                    fh.write(x + "\n")
                    self.n += 1
                    if time.time() - last > 1.0:
                        fh.flush(); last = time.time()
        except Exception as e:  # noqa: BLE001
            self.log(f"feedlog {self.path}: {e!r}")
