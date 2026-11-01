"""Import independently verified early-scan chunks into a final scan plan.

Run after final HF sources have been content-verified and before final scanning.
Preview summaries and shard assignments are deliberately not imported. A cache
may still be producing other chunks: only stable, completed pairs are eligible.
Changed sources, policies, code/models or candidate data are rejected and left
for the final scanner to recompute.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd

from hf_sources import write_json
from measurements import ARCHIVE, MODELS
from video_pipeline import code_config, digest, input_fingerprint, task_path
from video_scan import CHUNK_FRAMES, FPS, STEP, timestamp_allowed


def track_map(path):
    rows = json.loads(Path(path).read_text())
    result = {}
    for row in rows:
        name = row['track']
        if (Path(name).is_absolute() or any(part in ('.', '..') for part in name.split('/'))
                or name != f"{row['recording_id']}/speaker_{row['speaker']}"):
            raise ValueError(f'Invalid track identity: {name}')
        if name in result:
            raise ValueError(f'Duplicate track in {path}: {name}')
        result[name] = row
    return result


def validate_candidate(path, track, chunk, config):
    """Return a stable receipt bound to current inputs and actual parquet rows."""
    path = Path(path)
    receipt_path = path.with_suffix('.complete.json')
    original_receipt = receipt_path.read_bytes()
    receipt = json.loads(original_receipt)
    pose_path = track['pose_paths'].get(str(chunk), track['pose_paths'].get(chunk))
    fingerprint = input_fingerprint(track, chunk, pose_path, config)
    if (receipt.get('complete') is not True or receipt.get('track') != track['track']
            or receipt.get('chunk') != chunk or receipt.get('fingerprint') != fingerprint):
        raise ValueError('Checkpoint identity or current input/code fingerprint differs')
    if receipt.get('sha256') != digest(path):
        raise ValueError('Candidate checksum differs')
    rows = pd.read_parquet(path)
    if receipt.get('rows') != len(rows):
        raise ValueError('Candidate row count differs')
    if len(rows):
        expected = {key: track[key] for key in ('track', 'uuid', 'recording_id', 'speaker', 'session_id')}
        expected.update(bank=track.get('bank', 'bank'), cohort=track.get('cohort', ''), chunk=chunk,
                        pose_valid=True, in_eval=False, in_bb=False, in_light=False)
        for key, value in expected.items():
            if key not in rows or not rows[key].eq(value).all():
                raise ValueError(f'Candidate identity/policy field differs: {key}')
        frames = pd.to_numeric(rows.frame_index, errors='coerce').to_numpy(float)
        if (not np.isfinite(frames).all() or not np.equal(frames, np.floor(frames)).all()
                or len(set(frames)) != len(frames)
                or not ((frames >= chunk * CHUNK_FRAMES) & (frames < (chunk + 1) * CHUNK_FRAMES)).all()
                or not np.equal((frames - chunk * CHUNK_FRAMES) % STEP, 0).all()):
            raise ValueError('Candidate frame clock, sampling, or uniqueness differs')
        seconds = frames / FPS
        if not np.allclose(rows.time_s.to_numpy(float), seconds, rtol=0, atol=1e-9):
            raise ValueError('Candidate time differs from frame clock')
        if not all(timestamp_allowed(track, value) for value in seconds):
            raise ValueError('Candidate frame violates corpus exclusions')
        base = track['source'].split('#frame=')[0]
        if rows.source.tolist() != [f'{base}#frame={int(frame)}' for frame in frames]:
            raise ValueError('Candidate source differs from exact planned HF video')
        if rows.candidate_id.tolist() != [f"{track['track']}#frame={int(frame)}" for frame in frames]:
            raise ValueError('Candidate frame identity differs')
    if receipt_path.read_bytes() != original_receipt or digest(path) != receipt['sha256']:
        raise ValueError('Candidate or receipt changed during validation')
    if input_fingerprint(track, chunk, pose_path, config) != fingerprint:
        raise ValueError('Source changed during validation')
    return receipt


def publish_candidate(source, destination, receipt):
    """Publish data then receipt atomically without modifying the cache."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix='.import-', suffix='.parquet', dir=destination.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        shutil.copyfile(source, temporary)
        if digest(temporary) != receipt['sha256']:
            raise ValueError('Candidate changed while being copied')
        os.replace(temporary, destination)
        write_json(destination.with_suffix('.complete.json'), receipt)
    finally:
        temporary.unlink(missing_ok=True)


def import_caches(output_dir, caches, config):
    output_dir = Path(output_dir).resolve()
    final_path = output_dir / 'tracks.json'
    final_hash = digest(final_path)
    final = track_map(final_path)
    report = {'created_at': datetime.now(timezone.utc).isoformat(), 'final_tracks_sha256': final_hash,
              'code_inputs': config, 'imported_chunks': 0, 'already_valid_chunks': 0,
              'absent_chunks': 0, 'rejected_chunks': 0, 'caches': []}
    with ExitStack() as locks:
        # Cover both distributed supervisors and direct single/sharded scanners.
        lock_paths = [output_dir / 'distributed-scan.lock', output_dir / 'scan.lock',
                      *sorted((output_dir / 'scan-shards').glob('*/scan.lock'))]
        for path in lock_paths:
            handle = locks.enter_context(path.open('a'))
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for cache in sorted({Path(path).resolve() for path in caches}):
            if cache == output_dir:
                raise ValueError('Cache must differ from final output directory')
            source_plan = cache / 'tracks.json'
            source_hash = digest(source_plan)
            original = track_map(source_plan)
            item = {'cache': str(cache), 'tracks_sha256': source_hash,
                    'tracks_not_in_final': sorted(set(original) - set(final)), 'chunks': []}
            report['caches'].append(item)
            for name in sorted(set(original) & set(final)):
                track = final[name]
                for chunk in track['usable_chunks']:
                    source = task_path(cache, name, chunk)
                    destination = task_path(output_dir, name, chunk)
                    record = {'track': name, 'chunk': chunk}
                    if not source.is_file() or not source.with_suffix('.complete.json').is_file():
                        record['status'] = 'absent'
                        report['absent_chunks'] += 1
                    else:
                        try:
                            if original[name] != track:
                                raise ValueError('Preview track dict differs from final plan')
                            receipt = validate_candidate(source, track, chunk, config)
                            valid_destination = False
                            if destination.is_file() and destination.with_suffix('.complete.json').is_file():
                                try:
                                    validate_candidate(destination, track, chunk, config)
                                    valid_destination = True
                                except (OSError, ValueError, KeyError, AttributeError):
                                    pass
                            if valid_destination:
                                record['status'] = 'already-valid'
                                report['already_valid_chunks'] += 1
                            else:
                                if digest(source_plan) != source_hash or digest(final_path) != final_hash:
                                    raise ValueError('Track plan changed during import')
                                publish_candidate(source, destination, receipt)
                                validate_candidate(destination, track, chunk, config)
                                record['status'] = 'imported'
                                record['sha256'] = receipt['sha256']
                                report['imported_chunks'] += 1
                        except (OSError, ValueError, KeyError, AttributeError) as error:
                            record.update(status='rejected', reason=f'{type(error).__name__}: {error}')
                            report['rejected_chunks'] += 1
                    item['chunks'].append(record)
            if digest(source_plan) != source_hash:
                raise RuntimeError(f'Preview track plan changed during import: {cache}')
        if digest(final_path) != final_hash:
            raise RuntimeError('Final track plan changed during import')
        report['complete'] = True  # Import audit finished, not a bank/scan completion claim.
        write_json(output_dir / 'scan-cache-import.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--cache', type=Path, action='append', required=True)
    parser.add_argument('--archive-root', type=Path, default=ARCHIVE)
    parser.add_argument('--model-root', type=Path, default=MODELS)
    args = parser.parse_args()
    report = import_caches(args.output_dir, args.cache, code_config(args.archive_root, args.model_root))
    print(json.dumps({key: report[key] for key in ('imported_chunks', 'already_valid_chunks', 'absent_chunks', 'rejected_chunks')}))


if __name__ == '__main__':
    main()
