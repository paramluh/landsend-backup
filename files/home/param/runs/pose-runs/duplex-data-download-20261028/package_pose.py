#!/usr/bin/env python3
"""Package only the user's copy; never open the original owner's data tree."""
import ast
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import time

ROOT = Path('/data/home/mr_parameter/datasets/aloe-duplex-pose-20261028')
SOURCE = ROOT / 'data'
OUT = ROOT / 'hf-package'
REPO = 'InternalCan/aloe-poseduplex-v5-training-pose'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def status(phase, **extra):
    doc = dict(phase=phase, updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **extra)
    p = ROOT / 'control/package-status.json'
    tmp = p.with_suffix('.tmp'); tmp.write_text(json.dumps(doc, indent=2) + '\n'); tmp.replace(p)
    print(json.dumps(doc), flush=True)


def checked_file(rel):
    p = PurePosixPath(rel)
    assert not p.is_absolute() and '..' not in p.parts
    src = SOURCE / rel
    assert not src.is_symlink() and src.resolve().is_relative_to(SOURCE.resolve()), src
    assert src.is_file(), src
    return src


def verify_track(item):
    rel, e = item
    p = checked_file(rel)
    assert p.stat().st_size == e['bytes'] and sha(p) == e['sha256'], rel
    with p.open('rb') as f:
        assert f.read(6) == b'\x93NUMPY'
        version = f.read(2)
        width = 2 if version[0] == 1 else 4
        header = ast.literal_eval(f.read(int.from_bytes(f.read(width), 'little')).decode('latin1').strip())
    assert header['descr'] == '<i2' and header['shape'] == (32, e['T']) and not header['fortran_order'], rel
    jr = rel.replace('.pose_tokens.npy', '.pose.json'); jp = checked_file(jr)
    meta = json.loads(jp.read_text())
    assert meta['status'] == 'ok' and meta['pose_tokens_sha256'] == e['sha256'], jr
    assert meta['speaker'] == e['spk'] and tuple(meta['pose_tokens_shape']) == (32, e['T'])
    assert str(meta['tokenizer']).startswith('InternalCan/pose-tokenizer-32cb-200k-noshift@1d6ff4f0')
    return [dict(path=rel, bytes=e['bytes'], sha256=e['sha256'], kind='pose_tokens'),
            dict(path=jr, bytes=jp.stat().st_size, sha256=sha(jp), kind='pose_metadata')]


def pack_shard(task):
    i, paths, n = task
    filename = f'pose-{i:05d}-of-{n:05d}.tar'
    final = OUT / 'shards' / filename
    tmp = final.with_suffix('.tar.tmp')
    with tarfile.open(tmp, 'w', format=tarfile.USTAR_FORMAT) as tar:
        for rel in sorted(paths):
            p = checked_file(rel)
            info = tarfile.TarInfo(rel)
            info.size = p.stat().st_size; info.mode = 0o644; info.mtime = 0
            with p.open('rb') as f:
                tar.addfile(info, f)
    tmp.replace(final)
    return dict(path='shards/' + filename, bytes=final.stat().st_size, sha256=sha(final), files=len(paths))


def main():
    copy_done = ROOT / 'control/copy.exit'
    status('waiting_for_copy_into_user_folder')
    while not copy_done.exists():
        time.sleep(10)
    assert copy_done.read_text().strip() == '0', 'Copy failed; see copy.log'
    for name in ['shards', 'indexes', 'provenance']:
        (OUT / name).mkdir(parents=True, exist_ok=True)
    idx = json.loads((ROOT / 'provenance/POSE_INDEX.original.json').read_text())
    entries = {}; tracks = []
    for store, rows in idx['rows'].items():
        for rownum, row in rows.items():
            for side, e in row['sides'].items():
                if not e['ok']:
                    continue
                rel = e['path']; entries[rel] = e
                tracks.append(dict(store=store, row=rownum, side=side, conversation=row['conv'],
                                   split=row['split'], first_chunk_frames=row['n'], **e))
    assert len(entries) == len(tracks) == 13857
    assert sum(t['split'] == 'train' for t in tracks) == 13578
    assert sum(t['split'] == 'eval' for t in tracks) == 279
    status('verifying_user_copy', tracks=len(entries), token_bytes=sum(e['bytes'] for e in entries.values()))
    files = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for result in pool.map(verify_track, sorted(entries.items())):
            files.extend(result)
    for rel in ['pose_manifest.csv', 'manifest.csv', 'README.md', 'NOTE_local.md']:
        if (SOURCE / rel).is_file():
            p = checked_file(rel)
            files.append(dict(path=rel, bytes=p.stat().st_size, sha256=sha(p), kind='source_metadata'))
    files.sort(key=lambda x: x['path'])
    (OUT / 'indexes/FILES.jsonl').write_text(''.join(json.dumps(f) + '\n' for f in files))
    (OUT / 'indexes/TRACKS.jsonl').write_text(''.join(json.dumps(t) + '\n' for t in tracks))
    for p in (ROOT / 'provenance').iterdir():
        if p.is_file():
            shutil.copyfile(p, OUT / 'provenance' / p.name)
    # Keep the source index unchanged, including missing-pose records and exact row identities.
    shutil.copyfile(ROOT / 'provenance/POSE_INDEX.original.json', OUT / 'indexes/POSE_INDEX.original.json')
    n = 32; partitions = [[] for _ in range(n)]
    for f in files:
        # Both files of a track stay together; assignment is stable.
        key = f['path'].replace('.pose_tokens.npy', '').replace('.pose.json', '')
        partitions[int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % n].append(f['path'])
    status('packaging_verified_user_copy', files=len(files), shards=n)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        shards = list(pool.map(pack_shard, [(i, p, n) for i, p in enumerate(partitions)]))
    (OUT / 'indexes/SHARDS.json').write_text(json.dumps(shards, indent=2) + '\n')
    summary = dict(format='aloe-poseduplex-v5-training-pose-v1', source_run='30074', source_checkpoint='export-498',
                   conversations=8107, train_conversations=7949, eval_conversations=158,
                   valid_pose_tracks=13857, train_pose_tracks=13578, eval_pose_tracks=279,
                   missing_pose_sides=2357, token_bytes=sum(e['bytes'] for e in entries.values()),
                   files=len(files), shard_bytes=sum(s['bytes'] for s in shards),
                   pose_tokenizer=idx['pose_tokenizer'], normalization=idx['normalization'], token_rate_hz=12.5,
                   source_index_sha256=sha(ROOT / 'provenance/POSE_INDEX.original.json'),
                   source_access='Original owner files were copied read-only into the user folder before verification, packaging, or upload.',
                   verified=True)
    (OUT / 'SUMMARY.json').write_text(json.dumps(summary, indent=2) + '\n')
    (OUT / 'README.md').write_text('''---
pretty_name: Exact Aloe pose-duplex v5 training pose tracks
---

# Aloe pose-duplex v5 training pose data

Private, byte-preserving copy of the valid pose tracks referenced by job 30074's
original POSE_INDEX.json (the fine-tune producing export-498).

- 8,107 conversations: 7,949 train and 158 held out.
- 13,857 valid pose tracks: 13,578 train and 279 evaluation.
- The original index also retains 2,357 sides without valid pose. Their pose
  inputs/loss were absent in training; no replacement pose is invented here.
- Tokens are int16 [32, T] at 12.5 Hz, using
  InternalCan/pose-tokenizer-32cb-200k-noshift@1d6ff4f0 and the original frame-0
  root/shoulder-width normalization. Whole source tracks are preserved; the
  original fine-tune uses first chunks (at most 7,500 frames).

`shards/*.tar` contain the original `files/none/<session>/<segment>/` paths,
pose-token files, and their pose JSON sidecars. Extract all shards into one
data directory. `indexes/FILES.jsonl` records every extracted file's SHA256 and
size; `indexes/SHARDS.json` records archive checksums. All pose-token SHA256s
were checked against the original training index before packaging.

`indexes/POSE_INDEX.original.json` and `provenance/` preserve original training
metadata. Their old absolute paths are provenance, not paths to execute. For
a local training index, update `data_root` to the extracted directory and
`stores` to the four local SAM-separated stores; preserve row IDs and digests.
The copied global manifest CSVs describe the broader source corpus; the exact
files shipped here are defined by FILES.jsonl and TRACKS.jsonl.

Companion HF datasets: `InternalCan/aloe-v2vox-speech-export253` (original row
stores, transcripts, and raw audio) and `InternalCan/stage2-sam-sep-v1-mimi`
at revision ecb72b2de2347bfbbc3b71b0236e2986b0c44572 (SAM-isolated audio tokens).
Raw audio and SAM audio tokens are not duplicated in this pose package.

All original-owner access was read-only copying. Verification, archive creation,
and upload ran exclusively against the copy under the requesting user's folder.
''')
    status('uploading_private_hf_dataset', repo=REPO, **summary)
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(REPO, repo_type='dataset', private=True, exist_ok=True)
    assert api.dataset_info(REPO).private, 'Refusing to upload this data to a public repository'
    result = api.upload_folder(repo_id=REPO, repo_type='dataset', folder_path=OUT,
                               commit_message='Add verified exact pose tracks for Aloe pose-duplex v5 training')
    remote = api.dataset_info(REPO, revision=result.oid, files_metadata=True)
    local_files = {p.relative_to(OUT).as_posix(): p for p in OUT.rglob('*') if p.is_file() and '.cache' not in p.relative_to(OUT).parts}
    for f in remote.siblings:
        if f.rfilename == '.gitattributes':
            continue
        p = local_files.pop(f.rfilename)
        assert p.stat().st_size == f.size, f.rfilename
        if f.lfs:
            assert sha(p) == f.lfs.sha256, f.rfilename
        else:
            h = hashlib.sha1(f'blob {p.stat().st_size}\0'.encode() + p.read_bytes()).hexdigest()
            assert h == f.blob_id, f.rfilename
    assert not local_files, sorted(local_files)
    complete = dict(repo=REPO, revision=result.oid, url=f'https://huggingface.co/datasets/{REPO}',
                    private=True, files=len(remote.siblings), bytes=sum(f.size for f in remote.siblings), verified=True)
    (ROOT / 'control/UPLOAD_COMPLETE.json').write_text(json.dumps(complete, indent=2) + '\n')
    status('complete', **complete)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        status('failed', error_type=type(exc).__name__, message=str(exc)[:500])
        raise
