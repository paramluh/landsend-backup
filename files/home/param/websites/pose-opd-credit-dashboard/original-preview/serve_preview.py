#!/usr/bin/env python3
"""Serve the original monitor preview, using only synthetic fixtures."""
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import logging
import threading

from make_demo import DIST, build


class Handler(SimpleHTTPRequestHandler):
    def list_directory(self, path):
        self.send_error(404)
        return None

    def end_headers(self):
        self.send_header('Cache-Control','no-store')
        self.send_header('X-Content-Type-Options','nosniff')
        super().end_headers()

    def log_message(self, fmt, *args):
        if len(args)>1 and str(args[1]) not in ('200','304'):
            logging.info(fmt,*args)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    build()
    stop = threading.Event()
    def refresh():
        while not stop.wait(60):
            try: build()
            except Exception: logging.exception('Demo refresh failed')
    threading.Thread(target=refresh,daemon=True).start()
    with ThreadingHTTPServer(('127.0.0.1',9236),partial(Handler,directory=str(DIST))) as server:
        try: server.serve_forever()
        finally: stop.set()
