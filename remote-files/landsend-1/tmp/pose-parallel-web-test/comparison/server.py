"""Read-only comparison gallery, isolated from the interactive GPU demo."""
import argparse
import hashlib
import hmac
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import secrets
from urllib.parse import urlsplit

PAGE = Path(__file__).with_name('index.html')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True, type=Path)
    parser.add_argument('--bind', default='127.0.0.1')
    parser.add_argument('--port', default=7880, type=int)
    parser.add_argument('--token-file', type=Path, help='Shared access token; defaults to DATA_ROOT/.access-token')
    args = parser.parse_args()
    root = args.data_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    token_path = args.token_file or root / '.access-token'
    if not token_path.exists():
        token_path.write_text(secrets.token_urlsafe(32) + '\n')
        token_path.chmod(0o600)
    token = token_path.read_text().strip()
    if len(token) < 24:
        raise ValueError('Use an access token of at least 24 characters')
    session = hmac.new(token.encode(), b'comparison-gallery-session-v1', hashlib.sha256).hexdigest()

    class Handler(BaseHTTPRequestHandler):
        def authorized(self):
            try:
                cookie = SimpleCookie(self.headers.get('Cookie', ''))
                value = cookie['comparison_session'].value if 'comparison_session' in cookie else ''
                return hmac.compare_digest(value, session)
            except Exception:
                return False

        def do_POST(self):
            if urlsplit(self.path).path != '/auth':
                return self.send_error(404)
            size = int(self.headers.get('Content-Length', '0'))
            if not 0 < size <= 1024:
                return self.send_error(400)
            try:
                supplied = json.loads(self.rfile.read(size)).get('token', '')
            except (ValueError, AttributeError):
                return self.send_error(400)
            if not isinstance(supplied, str) or not hmac.compare_digest(supplied, token):
                return self.send_error(403)
            self.send_response(204)
            self.send_header('Set-Cookie', f'comparison_session={session}; Path=/; HttpOnly; SameSite=Lax; Secure; Max-Age=604800')
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()

        def do_HEAD(self):
            self.serve(head=True)

        def do_GET(self):
            self.serve()

        def serve(self, head=False):
            route = urlsplit(self.path).path
            if route in ('/', '/index.html'):
                path, mime, cache = PAGE, 'text/html; charset=utf-8', 'no-cache'
            elif route == '/manifest.json':
                if not self.authorized():
                    return self.send_error(401)
                path, mime, cache = root / 'manifest.json', 'application/json', 'no-store'
            elif re.fullmatch(r'/media/[\w-]+/[\w-]+/[a-f0-9]{20}\.(mp4|jpg)', route):
                if not self.authorized():
                    return self.send_error(401)
                path = (root / route.lstrip('/')).resolve()
                if not path.is_relative_to(root):
                    return self.send_error(404)
                mime = 'video/mp4' if path.suffix == '.mp4' else 'image/jpeg'
                cache = 'private, max-age=31536000, immutable'
            else:
                return self.send_error(404)
            if not path.is_file():
                return self.send_error(404)
            size, start, end, status = path.stat().st_size, 0, path.stat().st_size - 1, 200
            value = self.headers.get('Range')
            if value:
                match = re.fullmatch(r'bytes=(\d*)-(\d*)', value)
                if not match or not any(match.groups()):
                    return self.send_error(416)
                if match[1]:
                    start = int(match[1])
                    end = min(int(match[2]) if match[2] else end, end)
                else:
                    start = max(0, size - int(match[2]))
                if start > end:
                    self.send_response(416)
                    self.send_header('Content-Range', f'bytes */{size}')
                    self.end_headers()
                    return
                status = 206
            self.send_response(status)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(end - start + 1))
            self.send_header('Accept-Ranges', 'bytes')
            self.send_header('Cache-Control', cache)
            self.send_header('X-Content-Type-Options', 'nosniff')
            if status == 206:
                self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
            self.end_headers()
            if not head:
                try:
                    with path.open('rb') as source:
                        source.seek(start)
                        remaining = end - start + 1
                        while remaining:
                            chunk = source.read(min(1024 * 1024, remaining))
                            if not chunk:
                                break
                            self.wfile.write(chunk)
                            remaining -= len(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    pass

    print(f'Comparison gallery on {args.bind}:{args.port}; data {root}', flush=True)
    ThreadingHTTPServer((args.bind, args.port), Handler).serve_forever()


if __name__ == '__main__':
    main()
