#!/usr/bin/env python3
"""Mark the assembled dataset ready only after every download and preparation succeeds."""
import fcntl
import json
import os
from pathlib import Path
import socket
import time

ROOT = Path('/home/param/datasets/aloe-duplex')
CONTROL = Path('/home/param/pose-runs/duplex-data-download-20261028')


def read(path):
    return json.loads(path.read_text())


def write(path, doc):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(doc, indent=2) + '\n'); tmp.replace(path)


def main():
    lock = (CONTROL / 'finalize.lock').open('w'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    print('Waiting for audio, SAM stores, and pose verification.', flush=True)
    requirements = [CONTROL / 'DOWNLOADS_COMPLETE.json', ROOT / 's2h-m1-samsep/VERIFIED.json', ROOT / 'pose/VERIFIED.json']
    while not all(p.exists() for p in requirements):
        for path, field in [(CONTROL / 'status.json', 'download_state'),
                            (CONTROL / 'preparation-status.json', 'phase'),
                            (CONTROL / 'pose-download/status.json', 'download_state'),
                            (CONTROL / 'pose-download/preparation-status.json', 'phase')]:
            if path.exists() and read(path).get(field) == 'failed':
                raise RuntimeError(f'Component failed: {path}')
        time.sleep(20)
    download, sam, pose = [read(p) for p in requirements]
    assert download['download_state'] == 'complete' and not download['failed']
    assert sam['verified'] and pose['verified'] and sam['rows'] == 11777 and pose['tracks'] == 13857
    audio = read(Path(pose['audio_index']))
    source_manifest = read(ROOT / 'SOURCE_MANIFEST.json')
    audio_files = {e['path'][5:]: e for e in source_manifest['files'] if e['directory'] == 'original' and e['path'].startswith('flac/')}
    needed = {}
    for rows in audio['rows'].values():
        for row in rows:
            for entry in row['A'].values():
                rel = entry['flac']; p = ROOT / 'original/flac' / rel
                assert rel in audio_files and p.is_file() and p.stat().st_size == audio_files[rel]['size'], rel
                needed[rel] = audio_files[rel]['size']
    assert len(needed) == 16214, len(needed)
    pose_manifest = read(ROOT / 'pose/SOURCE_MANIFEST.json')
    ready = dict(training_ready=True, hostname=socket.gethostname(),
                 completed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                 train_conversations=7949, eval_conversations=158, samsep_store_rows=11777,
                 valid_pose_tracks=13857, audio_tracks=16214, missing_components=[],
                 downloaded_bytes=download['bytes_total'] + sum(e['bytes'] for e in pose_manifest['sources']),
                 sources=source_manifest['sources'] + pose_manifest['sources'], paths=pose)
    write(ROOT / 'DATASET_READY.json', ready)
    write(CONTROL / 'DATASET_READY.json', ready)
    for p in [CONTROL / 'status.json', CONTROL / 'DOWNLOADS_COMPLETE.json']:
        current = read(p)
        current.update(training_ready=True, missing_components=[], pending_preparation=[], assembled_dataset=str(ROOT / 'DATASET_READY.json'))
        write(p, current)
    print(json.dumps(ready, indent=2), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        write(CONTROL / 'finalize-error.json', dict(error_type=type(exc).__name__, message=str(exc)[:500], pid=os.getpid()))
        raise
