"""Keep the gallery and its Cloudflare tunnel running independently."""
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/home/param/pose-repo/.venv/bin/python"
STOP = threading.Event()
LOCK = threading.Lock()
CHILDREN = set()


def record(name, **values):
    path = ROOT / "state" / f"{name}.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(values, indent=2) + "\n")
    temporary.replace(path)


def supervise(name, command, tunnel=False):
    while not STOP.is_set():
        logpath = ROOT / "logs" / f"{name}-{time.time_ns()}.log"
        with logpath.open("x") as logfile:
            child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                     stdout=logfile, stderr=subprocess.STDOUT, start_new_session=True)
            with LOCK:
                CHILDREN.add(child)
            record(name, status="running", pid=child.pid, log=str(logpath))
            url = None
            while child.poll() is None and not STOP.wait(1):
                if tunnel and not url:
                    matches = re.findall(r"https://[a-z0-9-]+\.trycloudflare\.com", logpath.read_text())
                    if matches:
                        url = matches[0]
                        temporary = ROOT / "URL.tmp"
                        temporary.write_text(url + "\n")
                        temporary.replace(ROOT / "URL.txt")
                        record(name, status="running", pid=child.pid, url=url, log=str(logpath))
            if STOP.is_set() and child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
            child.wait()
            with LOCK:
                CHILDREN.discard(child)
            record(name, status="exited", returncode=child.returncode, log=str(logpath))
        STOP.wait(5)


def stop(signum, frame):
    STOP.set()
    with LOCK:
        for child in list(CHILDREN):
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)


if __name__ == "__main__":
    for name in ("logs", "state"):
        (ROOT / name).mkdir(exist_ok=True)
    lockfile = (ROOT / "state" / "supervisor.lock").open("a")
    fcntl.flock(lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop)
    workers = [
        threading.Thread(target=supervise, args=("gallery", [PYTHON, "-u", str(ROOT / "serve.py")])),
        threading.Thread(target=supervise, args=("cloudflared", ["/home/param/bin/cloudflared", "tunnel",
            "--no-autoupdate", "--protocol", "http2", "--url", "http://127.0.0.1:8920"], True)),
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
