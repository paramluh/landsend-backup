"""Publish completed held-out comparison videos atomically; CPU only."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

MODELS = [
    {"id": "aligned-media", "label": "Aligned transcript", "description": "Full aligned transcript upfront; the same voice and forced framewise text as the other models."},
    {"id": "compact-all", "label": "Compact transcript", "description": "Plain transcript upfront; the same voice and forced framewise text as the other models."},
    {"id": "xabi-original", "label": "Original Xabi", "description": "Original monoplex Xabi checkpoint. Same voice and forced framewise text; no upfront transcript."},
]
MODEL_IDS = {m['id'] for m in MODELS}


def now():
    return datetime.now(timezone.utc).isoformat(timespec='microseconds')


def slug(value):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,100}', value):
        raise ValueError(f'Invalid clip ID: {value!r}')
    return value


def atomic_json(path, value):
    handle, temporary = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(handle, 'w') as output:
            json.dump(value, output, indent=2, ensure_ascii=False)
            output.write('\n')
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


@contextmanager
def manifest_lock(root):
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.publish.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def probe(path):
    result = subprocess.run(['ffprobe', '-v', 'error', '-show_streams', '-show_format', '-of', 'json', str(path)],
                            check=True, capture_output=True, text=True)
    data = json.loads(result.stdout)
    videos = [s for s in data['streams'] if s['codec_type'] == 'video']
    audio = [s for s in data['streams'] if s['codec_type'] == 'audio']
    if len(videos) != 1 or not audio:
        raise ValueError('Publish requires one video stream and an audio stream')
    v = videos[0]
    return {'seconds': float(v.get('duration', data['format']['duration'])),
            'width': v['width'], 'height': v['height'], 'fps': v['r_frame_rate'],
            'frames': int(v['nb_frames']) if v.get('nb_frames', '').isdigit() else None,
            'video_codec': v['codec_name'], 'audio_codec': audio[0]['codec_name']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True, type=Path)
    sub = parser.add_subparsers(dest='command', required=True)
    init = sub.add_parser('init')
    init.add_argument('--clips', required=True, type=Path)
    init.add_argument('--mode', choices=['controlled', 'native'], default='controlled')
    init.add_argument('--title', default='Held-out speech & pose comparison')
    init.add_argument('--note', default='Eight held-out clips, three PPM checkpoints, one shared step-2911 video model. Completed videos appear as they finish.')
    pub = sub.add_parser('publish')
    pub.add_argument('--clip-id', required=True, type=slug)
    pub.add_argument('--model', required=True, choices=sorted(MODEL_IDS))
    pub.add_argument('--video', required=True, type=Path)
    pub.add_argument('--metadata', type=Path, help='Public sample metadata only; avoid credentials and private operational details')
    state = sub.add_parser('status')
    state.add_argument('--clip-id', required=True, type=slug)
    state.add_argument('--model', required=True, choices=sorted(MODEL_IDS))
    state.add_argument('--status', required=True, choices=['pending', 'generating', 'failed'])
    state.add_argument('--message', default='')
    voice = sub.add_parser('voice')
    voice.add_argument('--clip-id', required=True, type=slug)
    voice.add_argument('--audio', required=True, type=Path)
    args = parser.parse_args()
    root = args.data_root.resolve()
    manifest_path = root / 'manifest.json'
    if args.command == 'init':
        source = json.loads(args.clips.read_text())
        clips = source.get('clips') if isinstance(source, dict) else source
        if not isinstance(clips, list) or not clips:
            raise ValueError('Expected a nonempty clip list')
        clean, seen = [], set()
        for row in clips:
            cid = slug(str(row['id']))
            if cid in seen:
                raise ValueError(f'Duplicate clip ID: {cid}')
            seen.add(cid)
            clean.append({k: row[k] for k in ('id', 'title', 'transcript', 'recording_id', 'provenance') if k in row})
            clean[-1].update(id=cid, outputs={})
        models = MODELS if args.mode == 'controlled' else [
            {'id': 'compact-all', 'label': 'Compact · native', 'description': 'Plain transcript upfront; generated framewise text, audio and pose.'},
            {'id': 'xabi-original', 'label': 'Original Xabi · native', 'description': 'No transcript input; generated text, audio and pose with the same voice and appearance.'},
        ]
        with manifest_lock(root):
            if manifest_path.exists():
                previous = json.loads(manifest_path.read_text())
                old = {c['id']: c for c in previous['clips']}
                if set(old) - seen:
                    raise ValueError('Initialization cannot remove existing clips')
                for row in clean:
                    row['outputs'] = old.get(row['id'], {}).get('outputs', {})
                    if old.get(row['id'], {}).get('voice_reference'):
                        row['voice_reference'] = old[row['id']]['voice_reference']
            atomic_json(manifest_path, {'title': args.title, 'note': args.note, 'updated_at': now(),
                                       'models': models, 'clips': clean})
        print(json.dumps({'initialized': len(clean), 'manifest': str(manifest_path)}))
        return
    if args.command == 'voice':
        info = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_streams', '-of', 'json', str(args.audio)]))
        streams = info['streams']
        if len(streams) != 1 or streams[0]['codec_type'] != 'audio' or abs(float(streams[0]['duration']) - 8) > .001:
            raise ValueError('Expected one eight-second voice reference')
        digest = hashlib.sha256(args.audio.read_bytes()).hexdigest()
        directory = root / 'media' / args.clip_id / 'voice'
        directory.mkdir(parents=True, exist_ok=True)
        temporary = directory / f'.voice.{os.getpid()}.mp4'
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(args.audio), '-vn', '-c:a', 'aac',
                        '-b:a', '128k', '-t', '8', '-movflags', '+faststart', str(temporary)], check=True)
        media_hash = hashlib.sha256(temporary.read_bytes()).hexdigest()
        target = directory / f'{media_hash[:20]}.mp4'
        temporary.replace(target)
        with manifest_lock(root):
            manifest = json.loads(manifest_path.read_text())
            clip = next(c for c in manifest['clips'] if c['id'] == args.clip_id)
            clip['voice_reference'] = {'audio': '/' + str(target.relative_to(root)), 'seconds': 8,
                                      'sha256': media_hash, 'source_wav_sha256': digest,
                                      'description': 'Identical reference supplied to every model; outside the generation window'}
            manifest['updated_at'] = now()
            atomic_json(manifest_path, manifest)
        print(json.dumps({'voice_reference': args.clip_id, 'seconds': 8}))
        return
    if args.command == 'status':
        with manifest_lock(root):
            manifest = json.loads(manifest_path.read_text())
            clip = next(c for c in manifest['clips'] if c['id'] == args.clip_id)
            if clip['outputs'].get(args.model, {}).get('status') == 'complete':
                raise ValueError('Cannot replace a published video with a pending status')
            clip['outputs'][args.model] = {'status': args.status, 'message': args.message, 'updated_at': now()}
            manifest['updated_at'] = now()
            atomic_json(manifest_path, manifest)
        return
    # Probe and copy before publishing: a partial MP4 never enters the manifest.
    with manifest_lock(root):
        manifest = json.loads(manifest_path.read_text())
        clip = next((c for c in manifest['clips'] if c['id'] == args.clip_id), None)
        if clip is None:
            raise ValueError(f'Unknown clip: {args.clip_id}')
    metadata = json.loads(args.metadata.read_text()) if args.metadata else {}
    if not isinstance(metadata, dict):
        raise ValueError('Metadata must be a JSON object')
    media = probe(args.video)
    digest = hashlib.sha256(args.video.read_bytes()).hexdigest()
    directory = root / 'media' / args.clip_id / args.model
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f'{digest[:20]}.mp4'
    if not target.exists():
        handle, temporary = tempfile.mkstemp(prefix='.upload-', suffix='.mp4', dir=directory)
        os.close(handle)
        try:
            shutil.copyfile(args.video, temporary)
            if hashlib.sha256(Path(temporary).read_bytes()).hexdigest() != digest:
                raise ValueError('Source changed while copying; retry after generation completes')
            os.replace(temporary, target)
        finally:
            Path(temporary).unlink(missing_ok=True)
    poster = target.with_suffix('.jpg')
    handle, temporary = tempfile.mkstemp(prefix='.poster-', suffix='.jpg', dir=directory)
    os.close(handle)
    try:
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(target), '-frames:v', '1',
                        '-q:v', '3', temporary], check=True)
        os.replace(temporary, poster)
    finally:
        Path(temporary).unlink(missing_ok=True)
    with manifest_lock(root):
        manifest = json.loads(manifest_path.read_text())
        clip = next(c for c in manifest['clips'] if c['id'] == args.clip_id)
        clip['outputs'][args.model] = {'status': 'complete', 'video': '/' + str(target.relative_to(root)),
                                     'poster': '/' + str(poster.relative_to(root)), 'sha256': digest,
                                     'published_at': now(), **media, 'metadata': metadata}
        manifest['updated_at'] = now()
        atomic_json(manifest_path, manifest)
    print(json.dumps({'published': args.clip_id, 'model': args.model, 'video': '/' + str(target.relative_to(root)), **media}))


if __name__ == '__main__':
    main()
