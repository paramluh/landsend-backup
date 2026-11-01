#!/usr/bin/env python3
"""Record the URL produced by the self-hosted Cloudflare quick tunnel."""
import argparse
from pathlib import Path
import re
import signal
import subprocess

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=9236)
    args = parser.parse_args()
    url_file = ROOT / 'TUNNEL_URL'
    url_file.unlink(missing_ok=True)
    process = subprocess.Popen(
        ['/home/param/bin/cloudflared', 'tunnel', '--no-autoupdate', '--protocol', 'http2',
         '--url', f'http://127.0.0.1:{args.port}'],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    def stop(signum, frame):
        process.terminate()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    for line in process.stdout:
        print(line, end='', flush=True)
        match = re.search(r'https://[a-z0-9-]+\.trycloudflare\.com', line)
        if match:
            temporary = url_file.with_suffix('.tmp')
            temporary.write_text(match.group(0) + '\n')
            temporary.replace(url_file)
    raise SystemExit(process.wait())


if __name__ == '__main__':
    main()
