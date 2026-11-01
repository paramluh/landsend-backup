#!/usr/bin/env python3
"""Recreate the archived SAM store once its downloaded inputs are verified."""
import fcntl
import json
import os
from pathlib import Path
import socket
import sys
import time
from collections import Counter
from multiprocessing import Pool

CONTROL = Path(__file__).resolve().parent
ROOT = Path('/home/param/datasets/aloe-duplex')


def state(phase, **extra):
    doc = dict(phase=phase, hostname=socket.gethostname(), pid=os.getpid(),
               updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **extra)
    p = CONTROL / 'preparation-status.json'
    tmp = p.with_suffix('.tmp'); tmp.write_text(json.dumps(doc, indent=2) + '\n'); tmp.replace(p)
    print(json.dumps(doc), flush=True)


def main():
    lock = (CONTROL / 'prepare.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = json.loads((CONTROL / 'manifest.json').read_text())
    required = {(e['repo'], e['path']) for e in manifest['files']
                if e['directory'] == 'samsep' or e['path'].startswith('stores/')}
    state('waiting_for_verified_source_files', required_files=len(required))
    while True:
        done = set()
        if (CONTROL / 'verified.jsonl').exists():
            for line in (CONTROL / 'verified.jsonl').read_text().splitlines():
                try:
                    row = json.loads(line); done.add((row['repo'], row['path']))
                except json.JSONDecodeError:
                    pass
        if required <= done:
            break
        if (CONTROL / 'status.json').exists():
            download = json.loads((CONTROL / 'status.json').read_text())
            if download['download_state'] in ('failed', 'complete') or not Path(f"/proc/{download['pid']}").exists():
                raise RuntimeError('Downloader stopped before required inputs were verified')
        time.sleep(15)
    import build_samsep_stores as build
    import verify_samsep_stores as verify
    import pyarrow.parquet as pq
    import s3rows as F
    (ROOT / 's2h-m1-samsep').mkdir(exist_ok=True)
    marker = ROOT / 's2h-m1-samsep' / 'VERIFIED.json'
    if marker.exists():
        state('complete', result=str(marker)); return
    summary = Path(build.DATA) / 'build_summary.json'
    if not summary.exists():
        state('building_samsep_stores')
        sys.argv = [str(CONTROL / 'build_samsep_stores.py'), '--workers', '4']
        build.main()
    summ = json.loads(summary.read_text())
    expected = ['7318ffb3ea75cbcc3ff48b63b6cf8b88', '31ecb2273542310096936c0baf49efa8',
                '1759200f7818c6017dfb6a46e37fa5df', '611b2e7edb1503cc600a6489e49dcf40']
    assert summ['rows_kept'] == 11777 and summ['rows_dropped'] == 163 and summ['convs_dropped'] == 122
    assert [summ['new_digests'][f'm1n{k}'] for k in range(4)] == expected
    state('verifying_all_code_rows_and_metadata')
    keep, rowmap = verify.rowmap()
    noncode = {}
    first = Counter()
    idx = pq.read_table(f'{build.HF}/index/pairs.parquet', columns=['session', 'seg', 'token_shard']).to_pydict()
    sam = {f'{a}/{b}': sh for a, b, sh in zip(idx['session'], idx['seg'], idx['token_shard'])}
    by_shard = {}
    for k in range(4):
        checks, _ = verify.check_noncode(k, keep[k])
        assert all(v for v in checks.values() if isinstance(v, bool)) and checks['rows_jsonl_bad'] == 0
        noncode[f'm1n{k}'] = checks
        t = pq.read_table(f'{build.SRC}/m1n{k}/rows.parquet').to_pydict()
        for nr, old in enumerate(keep[k]):
            conv = t['src_conversation'][old]
            item = (k, nr, old, int(t['frame_start'][old]), int(t['n_frames'][old]), int(t['src_frames'][old]),
                    t['row_key'][old], t['split'][old], int(t['chunk_index'][old]))
            by_shard.setdefault(sam[conv], {}).setdefault(conv, []).append(item)
            if item[-1] == 0:
                first[item[-2]] += 1
    assert dict(first) == {'train': 7949, 'eval': 158}, first
    counts = Counter()
    with Pool(4) as pool:
        for shard, stats, _, _, _ in pool.imap_unordered(verify.check_codes, sorted(by_shard.items())):
            counts.update(stats)
    n = counts['rows']
    assert n == 11777 and counts['T_eq_src_frames'] == n
    assert all(counts[f'{side}_{check}'] == n for side in F.SIDES for check in ('eq_iso_own', 'pad_ok', 'range_ok'))
    result = dict(verified=True, source_revision=build.HF_REV, rows=n, first_chunks=dict(first),
                  digests=summ['new_digests'], code_counts=dict(counts), metadata_checks=noncode,
                  scope='All rows and both lanes: isolated-source code equality, padding, ranges, dimensions, and original non-code record equality',
                  not_included=['pose-token add-on', 'trainer launch configuration'])
    marker.write_text(json.dumps(result, indent=2) + '\n')
    state('complete', result=str(marker), rows=n, first_chunks=dict(first))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        state('failed', error_type=type(exc).__name__, message=str(exc)[:500])
        raise
