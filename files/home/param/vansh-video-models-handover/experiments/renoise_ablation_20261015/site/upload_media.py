import concurrent.futures
import hashlib
import json
from pathlib import Path
import sys
import termios
import urllib.error
import urllib.parse
import urllib.request


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, new_url):
        return None


original_terminal = None
if sys.stdin.isatty():
    original_terminal = termios.tcgetattr(sys.stdin.fileno())
    hidden_terminal = termios.tcgetattr(sys.stdin.fileno())
    hidden_terminal[3] &= ~termios.ECHO
    termios.tcsetattr(sys.stdin.fileno(), termios.TCSANOW, hidden_terminal)
print('Ready for upload JSON on stdin (input is hidden).', flush=True)
try:
    config = json.loads(sys.stdin.readline())
finally:
    if original_terminal is not None:
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSANOW, original_terminal)

base = config['baseUrl'].rstrip('/')
if urllib.parse.urlparse(base).scheme != 'https':
    raise ValueError('Uploads require HTTPS.')
token = config['token']
manifest = json.loads(Path(config['manifest']).read_text())
chunk_size = 8 * 1024 * 1024


def request(method, path, body=None, headers=None):
    request_headers = {'OAI-Sites-Authorization': f'Bearer {token}', **(headers or {})}
    req = urllib.request.Request(base + path, data=body, headers=request_headers, method=method)
    return urllib.request.build_opener(NoRedirect).open(req, timeout=180)


def upload_file(item):
    path = Path(item['path'])
    size = path.stat().st_size
    with path.open('rb') as source:
        sha256 = hashlib.file_digest(source, 'sha256').hexdigest()
    media_path = f"/api/bimodal-resolver/media/{item['condition']}/{item['file']}"
    try:
        with request('HEAD', media_path) as response:
            if int(response.headers['Content-Length']) == size and response.headers.get('X-File-Sha256') == sha256:
                print(f"Verified existing {item['condition']}/{item['file']}: {size} bytes", flush=True)
                return {'condition': item['condition'], 'file': item['file'], 'size': size, 'sha256': sha256, 'status': 'existing'}
    except urllib.error.HTTPError as error:
        if error.code != 404:
            raise
    query = {'condition': item['condition'], 'file': item['file']}

    def call(action, method, body=None, headers=None, **parameters):
        url = '/api/bimodal-resolver/upload?' + urllib.parse.urlencode({**query, 'action': action, **parameters})
        with request(method, url, body, headers) as response:
            return json.load(response)

    started = call('start', 'POST', b'', {'X-File-Sha256': sha256})
    upload_id = started['uploadId']
    try:
        parts = []
        with path.open('rb') as source:
            while chunk := source.read(chunk_size):
                parts.append(call('part', 'PUT', chunk, {'Content-Type': 'application/octet-stream'}, uploadId=upload_id, part=len(parts) + 1))
        completed = call('complete', 'POST', json.dumps(parts).encode(), {'Content-Type': 'application/json'}, uploadId=upload_id)
        if completed['size'] != size:
            raise ValueError(f"Unexpected uploaded size: {item['condition']}/{item['file']}")
    except Exception:
        call('abort', 'DELETE', uploadId=upload_id)
        raise
    with request('HEAD', media_path) as response:
        if int(response.headers['Content-Length']) != size or response.headers.get('X-File-Sha256') != sha256:
            raise ValueError(f"Upload verification failed: {item['condition']}/{item['file']}")
    with request('GET', media_path, headers={'Range': 'bytes=0-15'}) as response:
        first = response.read()
        with path.open('rb') as source:
            if response.status != 206 or first != source.read(16):
                raise ValueError(f"Range verification failed: {item['condition']}/{item['file']}")
    print(f"Uploaded and verified {item['condition']}/{item['file']}: {size} bytes", flush=True)
    return {'condition': item['condition'], 'file': item['file'], 'size': size, 'sha256': sha256, 'status': 'uploaded'}


with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(upload_file, manifest['files']))
print(json.dumps({'verified': len(results), 'files': results}), flush=True)
