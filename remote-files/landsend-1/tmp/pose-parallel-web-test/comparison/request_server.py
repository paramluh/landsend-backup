"""Authenticated, persistent parallel request studio; GPU dispatch is separate."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import hmac
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import sys
from urllib.parse import urlsplit
import uuid

HERE = Path(__file__).resolve().parent


def durable_json(path, value):
    with path.open('w') as handle:
        json.dump(value, handle, indent=2)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())


def sync_directory(path):
    handle = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(handle)
    finally:
        os.close(handle)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gallery-root', required=True, type=Path)
    parser.add_argument('--queue-root', required=True, type=Path)
    parser.add_argument('--native-gallery-root', type=Path, default=Path('/home/param/pose-runs/native-comparison-20261028/gallery'))
    parser.add_argument('--schema-dir', type=Path, default=Path('/home/param/pose-repo/infer/transcript_pose'))
    parser.add_argument('--bind', default='127.0.0.1')
    parser.add_argument('--port', default=7880, type=int)
    args = parser.parse_args()
    sys.path.insert(0, str(args.schema_dir))
    from request_schema import normalize_request, load_request, request_options

    gallery, queue = args.gallery_root.resolve(), args.queue_root.resolve()
    native_gallery = args.native_gallery_root.resolve()
    jobs = queue / 'jobs'
    jobs.mkdir(parents=True, exist_ok=True)
    token_file = gallery / '.access-token'
    token = token_file.read_text().strip()
    if len(token) < 24:
        raise ValueError('Missing strong gallery access token')
    session = hmac.new(token.encode(), b'comparison-gallery-session-v1', hashlib.sha256).hexdigest()
    csrf = hmac.new(token.encode(), b'request-studio-csrf-v1', hashlib.sha256).hexdigest()

    def load(path):
        return json.loads(path.read_text())

    def public_requests():
        rows = []
        for path in jobs.glob('[0-9a-f]' * 32):
            try:
                state = load(path / 'state.json')
                request = normalize_request(load(path / 'request.json'))
            except (OSError, ValueError):
                continue
            row = {k: state[k] for k in ('id', 'status', 'stage', 'created_at', 'updated_at', 'started_at', 'completed_at', 'progress') if k in state}
            row.update(id=path.name, request=request)
            node, pair = state.get('assigned_node'), state.get('gpu_pair')
            if isinstance(node, str) and re.fullmatch(r'landsend-[1-4]', node):
                row['assigned_node'] = node
            if isinstance(pair, list) and len(pair) == 2 and all(type(g) is int and 0 <= g <= 7 for g in pair) and len(set(pair)) == 2:
                row['gpu_pair'] = pair
            active = state.get('active_gpus')
            if isinstance(active, list) and len(active) <= 2 and all(type(g) is int and 0 <= g <= 7 for g in active) and len(set(active)) == len(active):
                row['active_gpus'] = active
            if type(state.get('executor_pid')) is int and state['executor_pid'] > 0:
                row['executor_pid'] = state['executor_pid']
            if state.get('error'):
                row['error'] = str(state['error'])[:1200]
            if state.get('status') == 'complete' and (path / 'video.mp4').is_file():
                row['video'] = f'/requests/{path.name}/video.mp4'
                if (path / 'metadata.json').is_file():
                    row['metadata'] = load(path / 'metadata.json')
            rows.append(row)
        return sorted(rows, key=lambda r: r.get('created_at', ''), reverse=True)

    def public_capacity():
        path = queue / 'dispatcher-status.json'
        try:
            if not path.is_file() or path.stat().st_size > 262144:
                raise ValueError('Missing or oversized status')
            value = load(path)
            updated = datetime.fromisoformat(value['updated_at'].replace('Z', '+00:00'))
            if updated.tzinfo is None or value['status'] not in ('running', 'stopping', 'stopped'):
                raise ValueError('Invalid dispatcher heartbeat')
            counts = {key: value[key] for key in ('capacity_total', 'capacity_busy', 'capacity_free', 'gpus_total', 'gpus_busy', 'gpus_free', 'queued_jobs', 'running_jobs')}
            if not all(type(n) is int and 0 <= n <= 100000 for n in counts.values()):
                raise ValueError('Invalid capacity counts')
            if not 0 < counts['capacity_total'] <= 16 or counts['capacity_busy'] + counts['capacity_free'] != counts['capacity_total']:
                raise ValueError('Inconsistent GPU-pair counts')
            if not 0 < counts['gpus_total'] <= 32 or counts['gpus_busy'] + counts['gpus_free'] != counts['gpus_total']:
                raise ValueError('Inconsistent GPU counts')
            slots = []
            if not isinstance(value['slots'], list) or len(value['slots']) > 128:
                raise ValueError('Invalid GPU-pair list')
            for slot in value['slots']:
                node, pair = slot['node'], slot['gpu_pair']
                if not isinstance(node, str) or not re.fullmatch(r'landsend-[1-4]', node):
                    raise ValueError('Unexpected compute node')
                if not isinstance(pair, list) or len(pair) != 2 or not all(type(g) is int and 0 <= g <= 7 for g in pair) or len(set(pair)) != 2:
                    raise ValueError('Invalid GPU pair')
                jid = slot.get('job_id')
                stage = slot.get('stage', 'unknown')
                if jid is not None and (not isinstance(jid, str) or not re.fullmatch(r'[a-f0-9]{32}', jid)):
                    raise ValueError('Invalid assignment')
                if not isinstance(stage, str) or len(stage) > 80:
                    raise ValueError('Invalid compute stage')
                clean = {'node': node, 'gpu_pair': pair, 'job_id': jid, 'stage': stage}
                active = slot.get('active_gpus')
                if active is not None:
                    if not isinstance(active, list) or len(active) > 2 or not all(type(g) is int and 0 <= g <= 7 for g in active) or len(set(active)) != len(active):
                        raise ValueError('Invalid active GPU allocation')
                    clean['active_gpus'] = active
                slots.append(clean)
            age = (datetime.now(timezone.utc) - updated).total_seconds()
            return {'available': True, 'status': value['status'], 'updated_at': value['updated_at'],
                    'stale': age > 60 or age < -60, **counts, 'slots': slots}
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            return {'available': False, 'status': 'unavailable', 'slots': []}

    class Handler(BaseHTTPRequestHandler):
        def authorized(self):
            try:
                cookies = SimpleCookie(self.headers.get('Cookie', ''))
                value = cookies['comparison_session'].value if 'comparison_session' in cookies else ''
                return hmac.compare_digest(value, session)
            except Exception:
                return False

        def same_origin(self):
            origin = urlsplit(self.headers.get('Origin', ''))
            return origin.scheme in ('http', 'https') and origin.netloc == self.headers.get('Host')

        def send_json(self, status, data):
            raw = json.dumps(data, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(raw)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            if self.command != 'HEAD':
                self.wfile.write(raw)

        def raw_body(self):
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 8192:
                    raise ValueError('Request body must be at most 8 KB')
                return self.rfile.read(length).decode('utf-8')
            except (ValueError, UnicodeError) as error:
                raise ValueError('Invalid request JSON or length') from error

        def body(self):
            try:
                value = json.loads(self.raw_body())
                if not isinstance(value, dict):
                    raise ValueError('Request body must be a JSON object')
                return value
            except (ValueError, UnicodeError) as error:
                raise ValueError('Invalid request JSON or length') from error

        def do_POST(self):
            route = urlsplit(self.path).path
            if not self.same_origin():
                return self.send_json(403, {'error': 'Requests must originate from this site'})
            if route == '/auth':
                try:
                    supplied = self.body().get('token', '')
                except ValueError as error:
                    return self.send_json(400, {'error': str(error)})
                if not isinstance(supplied, str) or not hmac.compare_digest(supplied, token):
                    return self.send_json(403, {'error': 'Invalid access token'})
                self.send_response(204)
                self.send_header('Set-Cookie', f'comparison_session={session}; Path=/; HttpOnly; SameSite=Lax; Secure; Max-Age=604800')
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                return
            if not self.authorized() or not hmac.compare_digest(self.headers.get('X-CSRF-Token', ''), csrf):
                return self.send_json(403, {'error': 'Missing session or request token'})
            if route != '/api/requests':
                return self.send_json(404, {'error': 'Not found'})
            try:
                request = load_request(self.raw_body())
            except ValueError as error:
                return self.send_json(400, {'error': str(error)})
            with (queue / '.submit.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                if sum(r['status'] in ('queued', 'running') for r in public_requests()) >= 100:
                    return self.send_json(429, {'error': 'Queue is full; wait for a request to finish'})
                jid = uuid.uuid4().hex
                staged = jobs / ('.new-' + jid)
                staged.mkdir(mode=0o700)
                created = datetime.now(timezone.utc).isoformat(timespec='microseconds')
                state = {'id': jid, 'status': 'queued', 'stage': 'queued', 'created_at': created,
                         'updated_at': created, 'request': request}
                durable_json(staged / 'request.json', request)
                durable_json(staged / 'state.json', state)
                sync_directory(staged)
                staged.rename(jobs / jid)
                sync_directory(jobs)
            self.send_json(201, state)

        def do_HEAD(self):
            self.serve(head=True)

        def do_GET(self):
            self.serve()

        def serve(self, head=False):
            route = urlsplit(self.path).path
            if route in ('/', '/index.html', '/comparison', '/native-comparison'):
                path = HERE / ('index.html' if route in ('/comparison', '/native-comparison') else 'studio.html')
                return self.file(path, 'text/html; charset=utf-8', 'no-cache', head)
            if not self.authorized():
                return self.send_json(401, {'error': 'Authentication required'})
            if route == '/manifest.json':
                manifest = load(gallery / 'manifest.json')
                manifest['native_comparison_available'] = (native_gallery / 'manifest.json').is_file()
                return self.send_json(200, manifest)
            if route == '/native-manifest.json':
                if not (native_gallery / 'manifest.json').is_file():
                    return self.send_json(404, {'error': 'Native comparison is being prepared'})
                manifest = load(native_gallery / 'manifest.json')
                for clip in manifest['clips']:
                    if clip.get('voice_reference'):
                        clip['voice_reference']['audio'] = clip['voice_reference']['audio'].replace('/media/', '/native-media/', 1)
                    for output in clip['outputs'].values():
                        for key in ('video', 'poster'):
                            if key in output:
                                output[key] = output[key].replace('/media/', '/native-media/', 1)
                manifest['request_studio_url'] = '/'
                manifest['native_comparison_available'] = True
                return self.send_json(200, manifest)
            if re.fullmatch(r'/native-media/[\w-]+/[\w-]+/[a-f0-9]{20}\.(mp4|jpg|wav)', route):
                path = (native_gallery / route.removeprefix('/native-')).resolve()
                if not path.is_relative_to(native_gallery):
                    return self.send_error(404)
                return self.file(path, {'.mp4': 'video/mp4', '.jpg': 'image/jpeg', '.wav': 'audio/wav'}[path.suffix], 'private, max-age=3600', head)
            if route == '/api/config':
                manifest = load(gallery / 'manifest.json')
                return self.send_json(200, {'csrf': csrf, 'options': request_options(), 'models': manifest['models'],
                     'clips': [{k: c[k] for k in ('id', 'title', 'transcript', 'voice_reference', 'provenance') if k in c} for c in manifest['clips']]})
            if route == '/api/requests':
                return self.send_json(200, {'requests': public_requests()})
            if route == '/api/capacity':
                return self.send_json(200, public_capacity())
            if re.fullmatch(r'/media/[\w-]+/[\w-]+/[a-f0-9]{20}\.(mp4|jpg|wav)', route):
                path = (gallery / route.lstrip('/')).resolve()
                if not path.is_relative_to(gallery):
                    return self.send_error(404)
                return self.file(path, {'.mp4': 'video/mp4', '.jpg': 'image/jpeg', '.wav': 'audio/wav'}[path.suffix], 'private, max-age=3600', head)
            match = re.fullmatch(r'/requests/([a-f0-9]{32})/video\.mp4', route)
            if match:
                job = jobs / match[1]
                path = (job / 'video.mp4').resolve()
                try:
                    ready = load(job / 'state.json')['status'] == 'complete'
                except (OSError, ValueError, KeyError):
                    ready = False
                if not ready or not path.is_relative_to(job.resolve()):
                    return self.send_error(404)
                return self.file(path, 'video/mp4', 'private, max-age=3600', head)
            self.send_error(404)

        def file(self, path, mime, cache, head=False):
            if not path.is_file():
                return self.send_error(404)
            size = path.stat().st_size
            start, end, status = 0, size - 1, 200
            if self.headers.get('Range'):
                match = re.fullmatch(r'bytes=(\d*)-(\d*)', self.headers['Range'])
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
            self.send_header('X-Frame-Options', 'DENY')
            self.send_header('Referrer-Policy', 'same-origin')
            if status == 206:
                self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
            self.end_headers()
            if head:
                return
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

    print(f'Request studio on {args.bind}:{args.port}; queue {queue}', flush=True)
    ThreadingHTTPServer((args.bind, args.port), Handler).serve_forever()


if __name__ == '__main__':
    main()
