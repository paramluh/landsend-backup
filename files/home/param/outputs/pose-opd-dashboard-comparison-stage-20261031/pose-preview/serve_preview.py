#!/usr/bin/env python3
"""Serve pose monitor assets; refresh curves every 60s and observed progress every 5s."""
from functools import partial
from http.server import ThreadingHTTPServer
import logging
from pathlib import Path
import re
import sys
import threading
from urllib.parse import unquote,urlsplit

from make_demo import DIST, build, refresh_progress
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'upstream'))
from range_http import RangeHandler


class Handler(RangeHandler):
    immutable_asset=re.compile(r'^/samples/assets/update-\d{6}-[0-9a-f]{16}/[A-Za-z0-9_.-]+$')

    def send_response(self, code, message=None):
        self.response_code=code
        return super().send_response(code,message)

    def send_head(self):
        # A cancelled range must never affect the next request on a connection.
        self._remaining=None
        self.compressed_json=False
        self.vary_encoding=False
        self.media_etag=None
        path=Path(self.translate_path(self.path))
        route=unquote(urlsplit(self.path).path)
        if route.startswith('/diagnostics/') and path.suffix.lower() in ('.mp4','.webm','.wav','.png','.jpg') and path.is_file():
            stat=path.stat()
            self.media_etag=f'"{stat.st_mtime_ns:x}-{stat.st_size:x}"'
            condition=self.headers.get('If-None-Match','')
            tags=[item.strip().removeprefix('W/') for item in condition.split(',')]
            if self.media_etag in tags or '*' in tags:
                self.send_response(304);self.end_headers();return None
            # Unhashed diagnostic filenames can change. A stale If-Range must
            # receive the complete replacement, never mixed old/new bytes.
            if_range=self.headers.get('If-Range')
            if if_range and if_range not in (self.media_etag,self.date_time_string(stat.st_mtime)):
                if 'Range' in self.headers:del self.headers['Range']
        if self.immutable_asset.fullmatch(route) and path.suffix=='.json':
            self.vary_encoding=True
            accepted={}
            for item in self.headers.get('Accept-Encoding','').split(','):
                token,*parameters=item.strip().split(';')
                quality=1.0
                for parameter in parameters:
                    if parameter.strip().startswith('q='):
                        try:quality=float(parameter.strip()[2:])
                        except ValueError:quality=0.0
                accepted[token.lower()]=quality
            zipped=path.with_name(path.name+'.gzip6.gz')
            # Media byte ranges stay identity encoded; never reinterpret their
            # offsets as positions in a compressed representation.
            if (not self.headers.get('Range') and accepted.get('gzip',accepted.get('*',0))>0
                    and path.is_file() and zipped.is_file()):
                try:stream=zipped.open('rb')
                except OSError:return super().send_head()
                stat=zipped.stat()
                self.compressed_json=True
                self.send_response(200)
                self.send_header('Content-Type',self.guess_type(str(path)))
                self.send_header('Content-Encoding','gzip')
                self.send_header('Content-Length',str(stat.st_size))
                self.send_header('Last-Modified',self.date_time_string(path.stat().st_mtime))
                self.end_headers()
                return stream
        return super().send_head()

    def list_directory(self, path):
        self.send_error(404)
        return None

    def end_headers(self):
        route=unquote(urlsplit(self.path).path)
        immutable=self.immutable_asset.fullmatch(route) and getattr(self,'response_code',0) in (200,206,304)
        diagnostic=getattr(self,'media_etag',None) and getattr(self,'response_code',0) in (200,206,304)
        policy=('public, max-age=31536000, immutable' if immutable else
                'public, max-age=0, must-revalidate' if diagnostic else 'no-store')
        self.send_header('Cache-Control',policy)
        if diagnostic:self.send_header('ETag',self.media_etag)
        if getattr(self,'vary_encoding',False):self.send_header('Vary','Accept-Encoding')
        self.send_header('X-Content-Type-Options','nosniff')
        super().end_headers()

    def copyfile(self,source,output):
        try:return super().copyfile(source,output)
        except (BrokenPipeError,ConnectionResetError):
            # Browsers abort metadata/range reads on seek or sample changes.
            # That is normal cancellation, not a server/media failure.
            return

    def log_message(self, fmt, *args):
        if len(args)>1 and str(args[1]) not in ('200','206','304'):
            logging.info(fmt,*args)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    build()
    stop = threading.Event()
    def refresh():
        tick=0
        while not stop.wait(5):
            tick+=1
            try:
                if tick%12==0:build()
                else:refresh_progress()
            except Exception:logging.exception('Dashboard refresh failed')
    threading.Thread(target=refresh,daemon=True).start()
    with ThreadingHTTPServer(('127.0.0.1',9236),partial(Handler,directory=str(DIST))) as server:
        try: server.serve_forever()
        finally: stop.set()
