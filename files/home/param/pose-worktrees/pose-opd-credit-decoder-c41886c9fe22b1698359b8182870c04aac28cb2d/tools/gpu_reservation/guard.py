#!/usr/bin/env python3
"""Enforce the param GPU reservation on landsend-1 through landsend-4."""
import argparse
import datetime
import json
import os
from pathlib import Path
import pwd
import signal
import socket
import subprocess
import time

HOSTS = {f"landsend-{i}" for i in range(1, 5)}
ALLOWED_UID = 1003
INFRASTRUCTURE = {"nvidia-persiste", "nvidia-fabricma", "nv-hostengine", "dcgm", "dcgm-exporter"}


def identify(pid):
    base = Path("/proc") / str(pid)
    try:
        stat = (base / "stat").read_text()
        fields = stat[stat.rfind(")") + 2:].split()
        return {"pid": pid, "uid": base.stat().st_uid, "start_ticks": int(fields[19]),
                "comm": (base / "comm").read_text().strip(),
                "loginuid": int((base / "loginuid").read_text()),
                "cgroup": (base / "cgroup").read_text().strip()}
    except (FileNotFoundError, ProcessLookupError):
        return None


def same_process(a, b):
    return a is not None and b is not None and all(a[k] == b[k] for k in ("pid", "uid", "start_ticks"))


def disposition(process):
    if process["uid"] == ALLOWED_UID:
        return "allowed"
    if process["uid"] == 0 or process["comm"] in INFRASTRUCTURE:
        return "review"
    return "terminate"


def send_signal(process, sig):
    if disposition(process) != "terminate" or not same_process(process, identify(process["pid"])):
        return False
    try:
        fd = os.pidfd_open(process["pid"])
    except ProcessLookupError:
        return False
    try:
        # pidfd pins the task even if the numeric PID is subsequently reused.
        if not same_process(process, identify(process["pid"])):
            return False
        signal.pidfd_send_signal(fd, sig)
        return True
    except ProcessLookupError:
        return False
    finally:
        os.close(fd)


def compute_processes():
    out = subprocess.check_output([
        "/usr/bin/nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits",
    ], text=True, stderr=subprocess.PIPE, timeout=10)
    pids = {int(line.strip()) for line in out.splitlines() if line.strip()}
    return {pid: p for pid in sorted(pids) if (p := identify(pid)) is not None}


def emit(event, **data):
    print(json.dumps({"time": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      "event": event, **data}, sort_keys=True), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--interval", type=float, default=3)
    ap.add_argument("--grace", type=float, default=3)
    ap.add_argument("--state-dir", type=Path, default=Path("/var/lib/param-gpu-reservation"))
    ap.add_argument("--audit-once", action="store_true", help="Read-only inventory; never sends signals")
    args = ap.parse_args()
    host = socket.gethostname().split(".")[0]
    if host not in HOSTS or pwd.getpwuid(ALLOWED_UID).pw_name != "param":
        raise RuntimeError("This guard is restricted to param UID1003 on landsend-1..4")
    if not 2 <= args.interval <= 5 or args.grace < 1:
        ap.error("Use a 2–5 second poll interval and at least a one-second TERM grace period")
    if args.audit_once:
        emit("audit", host=host, processes=[dict(p, action=disposition(p)) for p in compute_processes().values()])
        return
    if os.geteuid() != 0:
        raise RuntimeError("Enforcement must run as root")
    args.state_dir.mkdir(parents=True, exist_ok=True)
    pending, reviews = {}, set()
    totals = {"polls": 0, "sigterm": 0, "sigkill": 0, "errors": 0}
    emit("started", host=host, allowed_uid=ALLOWED_UID, interval=args.interval, grace=args.grace)
    while True:
        now = time.monotonic()
        error = None
        try:
            processes = compute_processes()
            for pid in list(pending):
                if not same_process(pending[pid]["process"], processes.get(pid)):
                    emit("released", process=pending.pop(pid)["process"])
            for pid, process in processes.items():
                action = disposition(process)
                if action == "review":
                    identity = (pid, process["start_ticks"])
                    if identity not in reviews:
                        emit("manual_review", process=process)
                        reviews.add(identity)
                elif action == "terminate":
                    if pid not in pending:
                        if send_signal(process, signal.SIGTERM):
                            totals["sigterm"] += 1
                            pending[pid] = {"process": process, "deadline": time.monotonic() + args.grace, "killed": False}
                            emit("sigterm", process=process)
                    elif now >= pending[pid]["deadline"] and not pending[pid]["killed"]:
                        if send_signal(process, signal.SIGKILL):
                            totals["sigkill"] += 1
                            pending[pid]["killed"] = True
                            emit("sigkill", process=process)
            totals["polls"] += 1
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            error = f"{type(exc).__name__}: {exc}"
            totals["errors"] += 1
            processes = {}
            emit("error", error=error)
        state = {"host": host, "pid": os.getpid(), "time": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 "allowed_uid": ALLOWED_UID, "interval": args.interval, "grace": args.grace,
                 "totals": totals, "error": error,
                 "processes": [dict(p, action=disposition(p)) for p in processes.values()],
                 "pending": sorted(pending)}
        tmp = args.state_dir / "status.json.tmp"
        tmp.write_text(json.dumps(state, indent=2) + "\n")
        tmp.replace(args.state_dir / "status.json")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
