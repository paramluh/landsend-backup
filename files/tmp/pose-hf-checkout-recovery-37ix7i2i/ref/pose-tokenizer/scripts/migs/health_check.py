import http.server
import socketserver
import threading
import time
import json
import re

# ---- internal state (thread-safe) ----
_lock = threading.Lock()
_state = {
    "ready": False,
    "processing": False,
    "last_activity_ts": time.time(),
    "last_reason": "init",
    "min_mem_mb": 1024,
    "idle_grace_s": 120,
    "idle_timeout_s": 35 * 60,  # default; you can override in start()
}


def _mem_available_mb() -> int:
    try:
        with open("/proc/meminfo") as f:
            m = re.search(r"MemAvailable:\s+(\d+)\s+kB", f.read())
        return (int(m.group(1)) // 1024) if m else 0
    except Exception:
        return 0


def _snapshot():
    with _lock:
        return dict(_state)


def set_ready(value: bool, reason: str = "ready/unready"):
    with _lock:
        _state["ready"] = value
        _state["last_reason"] = reason


def set_processing(value: bool):
    with _lock:
        _state["processing"] = value


def note_activity(reason: str = "activity"):
    now = time.time()
    with _lock:
        _state["last_activity_ts"] = now
        _state["last_reason"] = reason


def fail(reason: str):
    with _lock:
        _state["ready"] = False
        _state["last_reason"] = reason


def status_dict():
    s = _snapshot()
    now = time.time()
    mem_mb = _mem_available_mb()
    time_since = now - s["last_activity_ts"]

    # Health rules:
    #   - must be "ready"
    #   - have enough free RAM
    #   - be active (recent activity or currently processing), with grace at startup
    #   - main process must be running (if PID is set)
    active_ok = (time_since <= max(s["idle_timeout_s"], s["idle_grace_s"])) or not s["ready"]
    mem_ok = mem_mb >= s["min_mem_mb"]
    overall = s["ready"] and mem_ok and active_ok

    return {
        "ready": s["ready"],
        "processing": s["processing"],
        "seconds_since_last_activity": round(time_since, 1),
        "mem_available_mb": mem_mb,
        "mem_ok": mem_ok,
        "active_ok": active_ok,
        "reason": s["last_reason"],
        "status": "ok" if overall else "unhealthy",
        "ok": overall,
    }


class _Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/healthz":
            self.send_response(404)
            self.end_headers()
            return

        d = status_dict()
        code = 200 if d["ok"] else 503
        body = json.dumps(d).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):  # silence default logs
        return


def start(port: int = 8080, *, min_mem_mb: int = 1024, idle_grace_s: int = 120, idle_timeout_s: int = 35 * 60):
    # Configure thresholds
    with _lock:
        _state["min_mem_mb"] = min_mem_mb
        _state["idle_grace_s"] = idle_grace_s
        _state["idle_timeout_s"] = idle_timeout_s

    server = socketserver.TCPServer(("0.0.0.0", port), _Handler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server  # returned for completeness (rarely needed)
