#!/usr/bin/env python3
"""Serve only generated dashboard assets and refresh them every 60 seconds."""
import argparse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import logging
from pathlib import Path
import threading
from urllib.parse import urlsplit

from build_stage3 import ROOT, PAGE, build

ALLOWED = {'/', '/index.html', '/dashboard.js', '/dashboard.css',
           f'/stage3_{PAGE}.html', f'/stage3_{PAGE}_artifact.html',
           f'/s3live_{PAGE}.json', f'/ver_s3_{PAGE}.txt'}


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT / 'dist'), **kwargs)

    def send_head(self):
        if urlsplit(self.path).path not in ALLOWED:
            self.send_error(404)
            return None
        return super().send_head()

    def end_headers(self):
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        super().end_headers()

    def log_message(self, fmt, *args):
        if args and str(args[1]) != '200':
            logging.info(fmt, *args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=9236)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    build()
    stop = threading.Event()

    def refresh():
        while not stop.wait(60):
            try:
                build()
            except Exception:
                logging.exception('Refresh failed; retaining the previous snapshot')

    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    thread = threading.Thread(target=refresh, daemon=True)
    thread.start()
    logging.info('Serving http://127.0.0.1:%s', args.port)
    try:
        server.serve_forever()
    finally:
        stop.set()
        server.server_close()


if __name__ == '__main__':
    main()
