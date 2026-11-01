"""Merge immutable generated-video results into the served sample manifest.

The CPU decoder remains the sole writer of its own manifest. Results bind to
one exact raw manifest and are ignored after a resumed update changes that hash.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile

HEX = re.compile(r'[0-9a-f]{64}\Z')
ID = re.compile(r'[A-Za-z0-9_-]+\Z')


def _positive(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def _verified_video(source, result):
    name = result['file']
    if not isinstance(name, str) or Path(name).name != name or not ID.fullmatch(Path(name).stem) or Path(name).suffix != '.mp4':
        raise ValueError('Video file must be a safe MP4 basename')
    digest = result['sha256']
    if not isinstance(digest, str) or not HEX.fullmatch(digest):
        raise ValueError('Invalid video SHA256')
    incoming = source / 'video-results' / name
    if incoming.is_symlink() or not incoming.is_file():
        raise ValueError('Complete regular video file is missing')
    size = result['bytes']
    if type(size) is not int or size <= 0 or incoming.stat().st_size != size:
        raise ValueError('Video size mismatch')
    cache = source.parent / '.sample-video-cache'
    cache.mkdir(exist_ok=True)
    target = cache / (digest + '.mp4')
    if target.is_file() and target.stat().st_size == size:
        return target
    # Copy once rather than hard-linking an externally produced file into the
    # immutable HTTP cache. A mistaken later in-place edit cannot corrupt it.
    temporary = None
    try:
        sha = hashlib.sha256()
        with incoming.open('rb') as stream, tempfile.NamedTemporaryFile(dir=cache, prefix='.verify-', delete=False) as output:
            temporary = Path(output.name)
            for block in iter(lambda: stream.read(1 << 20), b''):
                sha.update(block); output.write(block)
            output.flush(); os.fsync(output.fileno())
        if temporary.stat().st_size != size or sha.hexdigest() != digest:
            raise ValueError('Video SHA256 mismatch')
        temporary.replace(target)
        return target
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def merge_generated_videos(source, staging):
    """Return an audit; invalid/unfinished results leave the skeleton intact."""
    source, staging = Path(source), Path(staging)
    path = staging / 'manifest.json'
    manifest = json.loads(path.read_text())
    report = {'attached': [], 'rejected': []}
    for rollout in manifest.get('rollouts', []):
        sha = rollout.get('source_manifest_sha256', '')
        if not isinstance(sha, str) or not HEX.fullmatch(sha):
            continue
        for sample in rollout.get('samples', []):
            sid = sample.get('id', '')
            if not isinstance(sid, str) or not ID.fullmatch(sid):
                continue
            sidecar = source / 'video-results' / f'{sid}-{sha}.json'
            if not sidecar.is_file():
                continue
            try:
                if sidecar.is_symlink() or sidecar.stat().st_size > 65536:
                    raise ValueError('Unsafe result manifest')
                result = json.loads(sidecar.read_text())
                if (result.get('schema_version') != 1 or result.get('kind') != 'pose-opd-generated-video'
                        or result.get('complete') is not True or result.get('sample_id') != sid
                        or result.get('source_manifest_sha256') != sha or result.get('update') != rollout['update']):
                    raise ValueError('Video result identity does not match this rollout')
                fps, duration = result.get('fps'), result.get('duration_seconds')
                if not _positive(fps) or not _positive(duration):
                    raise ValueError('Invalid video clock')
                for key in ('width', 'height', 'frame_count'):
                    if type(result.get(key)) is not int or result[key] <= 0:
                        raise ValueError('Invalid video dimensions/frame count')
                if (abs(duration - float(sample['duration_seconds'])) > max(1 / fps, .005)
                        or abs(result['frame_count'] / fps - duration) > max(1 / fps, .005)
                        or result.get('video_offset_seconds', 0) != 0):
                    raise ValueError('Video must cover the full physical timeline from zero')
                verified = _verified_video(source, result)
                folder = f"assets/update-{rollout['update']:06d}-{sha[:16]}"
                filename = f"{sid}-video-{result['sha256'][:16]}.mp4"
                relative = Path(folder) / filename
                target = staging / relative; target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    try: os.link(verified, target)
                    except OSError: shutil.copy2(verified, target)
                sample['video_url'] = relative.as_posix()
                sample['video_offset_seconds'] = 0
                sample['video_metadata'] = {key: result[key] for key in ('sha256', 'duration_seconds', 'fps', 'width', 'height', 'frame_count')}
                report['attached'].append(sid)
            except (OSError, ValueError, KeyError, TypeError) as error:
                report['rejected'].append({'sample_id': sid, 'error': str(error)})
    temporary = path.with_suffix('.video.tmp')
    temporary.write_text(json.dumps(manifest, separators=(',', ':'), allow_nan=False) + '\n')
    temporary.replace(path)
    return report
