#!/usr/bin/env python3
"""Verify/extract downloaded pose archives and write the local training index."""
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import socket
import tarfile
import time

ROOT = Path('/home/param/datasets/aloe-duplex')
POSE = ROOT / 'pose'
HF = POSE / 'hf'
CONTROL = Path('/home/param/pose-runs/duplex-data-download-20261028/pose-download')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, doc):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(doc, indent=2) + '\n'); tmp.replace(path)


def status(phase, **kw):
    doc = dict(phase=phase, hostname=socket.gethostname(), pid=os.getpid(),
               updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **kw)
    write(CONTROL / 'preparation-status.json', doc)
    print(json.dumps(doc), flush=True)


def main():
    CONTROL.mkdir(parents=True, exist_ok=True)
    lock = (CONTROL / 'prepare.lock').open('w'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status('waiting_for_verified_hf_download')
    while not (CONTROL / 'DOWNLOADS_COMPLETE.json').exists():
        if (CONTROL / 'status.json').exists():
            d = json.loads((CONTROL / 'status.json').read_text())
            if d['download_state'] == 'failed' or not Path(f"/proc/{d['pid']}").exists():
                raise RuntimeError('Pose downloader stopped before verification completed')
        time.sleep(10)
    expected = {r['path']: r for r in map(json.loads, (HF / 'indexes/FILES.jsonl').read_text().splitlines())}
    assert len(expected) == json.loads((HF / 'SUMMARY.json').read_text())['files']
    dest = POSE / 'data'; dest.mkdir(parents=True, exist_ok=True)
    extracted = set()
    status('extracting_and_verifying', expected_files=len(expected))
    for shard in json.loads((HF / 'indexes/SHARDS.json').read_text()):
        archive = HF / shard['path']
        assert archive.stat().st_size == shard['bytes'] and sha(archive) == shard['sha256']
        with tarfile.open(archive, 'r') as tar:
            for member in tar:
                rel = member.name; p = PurePosixPath(rel)
                assert member.isfile() and not p.is_absolute() and '..' not in p.parts and rel in expected
                assert rel not in extracted, rel
                entry = expected[rel]; target = dest / rel
                assert member.size == entry['bytes']
                target.parent.mkdir(parents=True, exist_ok=True)
                assert target.parent.resolve().is_relative_to(dest.resolve())
                assert not target.is_symlink(), target
                if target.is_file() and target.stat().st_size == entry['bytes'] and sha(target) == entry['sha256']:
                    extracted.add(rel); continue
                tmp = target.with_suffix(target.suffix + '.part')
                h = hashlib.sha256()
                with tar.extractfile(member) as source, tmp.open('wb') as out:
                    for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
                        out.write(block); h.update(block)
                assert h.hexdigest() == entry['sha256'], rel
                tmp.replace(target); extracted.add(rel)
        status('extracting_and_verifying', extracted_files=len(extracted), expected_files=len(expected))
    assert extracted == set(expected)
    idx = json.loads((HF / 'indexes/POSE_INDEX.original.json').read_text())
    sam = json.loads((ROOT / 's2h-m1-samsep/VERIFIED.json').read_text())
    assert sam['verified'] and idx['store_digests'] == [sam['digests'][f'm1n{k}'] for k in range(4)]
    idx['data_root'] = str(dest)
    idx['stores'] = [str(ROOT / 's2h-m1-samsep/stores' / f'm1n{k}') for k in range(4)]
    idx['local_provenance'] = dict(source_index_sha256=sha(HF / 'indexes/POSE_INDEX.original.json'),
                                    hf_source_manifest=str(POSE / 'SOURCE_MANIFEST.json'))
    for rows in idx['rows'].values():
        for row in rows.values():
            for e in row['sides'].values():
                if e['ok']:
                    assert expected[e['path']]['sha256'] == e['sha256']
    write(POSE / 'POSE_INDEX.json', idx)
    # Preserve source entries while resolving only the store roots for this installation.
    audio = json.loads((HF / 'provenance/M1VOX_INDEX_samsep.original.json').read_text())
    assert audio['digests'] == idx['store_digests']
    audio['stores'] = idx['stores']
    write(POSE / 'M1VOX_INDEX_samsep.json', audio)
    result = dict(verified=True, files=len(extracted), tracks=idx['counts']['sides_ok'],
                  counts=idx['counts'], data_root=str(dest), pose_index=str(POSE / 'POSE_INDEX.json'),
                  audio_index=str(POSE / 'M1VOX_INDEX_samsep.json'), audio_root=str(ROOT / 'original/flac'),
                  rawgain=str(HF / 'provenance/RAWGAIN_samsep.original.json'),
                  samsep_stores=idx['stores'], missing_components=[], source='InternalCan/aloe-poseduplex-v5-training-pose')
    write(POSE / 'VERIFIED.json', result)
    status('complete', tracks=result['tracks'], files=len(extracted), pose_index=result['pose_index'])


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        status('failed', error_type=type(exc).__name__, message=str(exc)[:500])
        raise
