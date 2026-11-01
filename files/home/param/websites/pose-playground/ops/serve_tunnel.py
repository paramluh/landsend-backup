#!/usr/bin/env python3
"""Keep the recorded URL in sync when a supervised quick tunnel restarts."""
import argparse
from pathlib import Path
import re
import signal
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, required=True)
parser.add_argument("--url-file", type=Path, required=True)
args = parser.parse_args()
process = subprocess.Popen(
    ["/home/param/bin/cloudflared", "tunnel", "--url", f"http://127.0.0.1:{args.port}",
     "--no-autoupdate", "--protocol", "http2"],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
)
def stop(signum, frame):
    process.terminate()
signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)
for line in process.stdout:
    print(line, end="", flush=True)
    match = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", line)
    if match:
        args.url_file.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.url_file.with_suffix(".tmp")
        temporary.write_text(match.group(0) + "\n")
        temporary.replace(args.url_file)
raise SystemExit(process.wait())
