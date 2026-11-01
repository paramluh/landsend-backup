"""Validate and aggregate only a scope's existing shared measurement checkpoints.

This reads shared per-track checkpoints without changing them or waiting for an
unrelated global watcher. Scope tables and provenance are written separately.
"""
from concurrent.futures import ThreadPoolExecutor
import argparse
import hashlib
import json
from pathlib import Path
import time

from hf_sources import write_json
from measure_upstream import measurement_config, source_fingerprint


def read_checkpoint(track, data_root, checkpoint_root, config):
    path = checkpoint_root / 'tracks' / (track.replace('/', '__') + '.json')
    raw = path.read_bytes()
    value = json.loads(raw)
    if (value.get('complete') is not True or value.get('track') != track
            or value.get('fingerprint') != source_fingerprint(track, data_root, config)):
        raise ValueError(f'Measurement checkpoint is incomplete or stale: {track}')
    for key in ('p14', 'p59'):
        if not isinstance(value.get(key), list) or any(row.get('track') != track for row in value[key]):
            raise ValueError(f'Measurement rows have the wrong track: {track}/{key}')
    if not isinstance(value.get('teeth'), dict) or value['teeth'].get('track') != track:
        raise ValueError(f'Teeth measurement has the wrong track: {track}')
    if path.read_bytes() != raw:
        raise ValueError(f'Measurement checkpoint changed during validation: {track}')
    return value, hashlib.sha256(raw).hexdigest()


def aggregate(tracks_file, data_root, checkpoint_root, output_dir, workers=32):
    tracks_file, data_root = Path(tracks_file), Path(data_root)
    checkpoint_root, output_dir = Path(checkpoint_root), Path(output_dir)
    if output_dir.resolve() == checkpoint_root.resolve():
        raise ValueError('Scoped aggregates must not overwrite the shared measurement directory')
    tracks_bytes = tracks_file.read_bytes()
    tracks = [line for line in tracks_bytes.decode().splitlines() if line]
    if not tracks or len(tracks) != len(set(tracks)):
        raise ValueError('Scoped tracks must be nonempty and unique')
    if any(Path(track).is_absolute() or '..' in Path(track).parts for track in tracks):
        raise ValueError('Invalid scoped track path')
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'complete.json').unlink(missing_ok=True)
    config = measurement_config()
    started = time.monotonic()
    tables = {'p14': [], 'p59': [], 'teeth': []}
    checkpoint_hashes, failures = {}, []
    def read(track):
        try:
            return track, read_checkpoint(track, data_root, checkpoint_root, config), None
        except (OSError, ValueError, KeyError, TypeError) as error:
            return track, None, f'{type(error).__name__}: {error}'
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for number, (track, result, error) in enumerate(pool.map(read, tracks), 1):
            if error:
                failures.append({'track': track, 'error': error})
            else:
                value, checkpoint_hashes[track] = result
                tables['p14'].extend(value['p14'])
                tables['p59'].extend(value['p59'])
                tables['teeth'].append(value['teeth'])
            if number % 500 == 0 or number == len(tracks):
                print(json.dumps({'validated_tracks': number, 'tracks': len(tracks), 'failures': len(failures)}), flush=True)
    write_json(output_dir / 'failures.json', failures)
    if failures:
        raise RuntimeError(f'{len(failures)} scoped measurement checkpoints need completion; see {output_dir / "failures.json"}')
    if measurement_config() != config or tracks_file.read_bytes() != tracks_bytes:
        raise ValueError('Scoped track list or measurement implementation changed during aggregation')
    import pandas as pd
    for name, rows in tables.items():
        destination = output_dir / f'{name}.parquet'
        temporary = destination.with_suffix('.parquet.tmp')
        pd.DataFrame(rows).to_parquet(temporary, index=False)
        temporary.replace(destination)
    report = {'complete': True, 'tracks': len(tracks), 'tracks_sha256': hashlib.sha256(tracks_bytes).hexdigest(),
              'checkpoint_root': str(checkpoint_root.resolve()), 'measurement_config': config,
              'checkpoint_sha256': checkpoint_hashes, 'rows': {key: len(rows) for key, rows in tables.items()},
              'seconds': time.monotonic() - started}
    write_json(output_dir / 'complete.json', report)
    print(json.dumps({key: report[key] for key in ('complete', 'tracks', 'rows', 'seconds')}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('tracks', 'data-root', 'checkpoint-root', 'output-dir'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=32)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    aggregate(args.tracks, args.data_root, args.checkpoint_root, args.output_dir, args.workers)


if __name__ == '__main__':
    main()
