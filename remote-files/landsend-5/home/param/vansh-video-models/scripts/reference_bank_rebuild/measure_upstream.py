"""Measure every downloaded upstream track, with per-track restart checkpoints."""
import argparse
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import time

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
from measurements import ARCHIVE, MODELS, initialize_measurements

MODULES = None
DATA = OUT = None
MISSING = set()
CONFIG = None


def measurement_config(archive=ARCHIVE, models=MODELS):
    paths = [Path(__file__), Path(__file__).with_name('measurements.py'),
             Path(archive) / 'node/ref_audit/audit.py', Path(archive) / 'node/ref_audit/audit59.py',
             Path(archive) / 'node/mouth_open_pilot/teeth.py', Path(archive) / 'node/mouth_open_pilot/allteeth.py',
             Path(archive) / 'node/mouth_open_pilot/scan.py',
             Path(models) / 'face_landmarker.task', Path(models) / 'hand_landmarker.task']
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def source_fingerprint(track, data, config):
    from upstream import SLOTS, REFERENCE_PREFIX, SELECTOR_FOLDER
    folder = Path(data) / REFERENCE_PREFIX / track / SELECTOR_FOLDER
    entries = {}
    for name in ('manifest.json', 'selected_poses.parquet', *(f'{slot}.png' for slot in SLOTS)):
        path = folder / name
        if not path.is_file():
            entries[name] = None
            continue
        stat = path.stat()
        entries[name] = [stat.st_size, stat.st_mtime_ns]
        if not name.endswith('.png'):
            entries[name].append(hashlib.sha256(path.read_bytes()).hexdigest())
    return hashlib.sha256(json.dumps({'config': config, 'inputs': entries}, sort_keys=True).encode()).hexdigest()


def checkpoint_valid(track, data, output, config):
    path = Path(output) / 'tracks' / (track.replace('/', '__') + '.json')
    if not path.is_file():
        return False
    try:
        row = json.loads(path.read_text())
        return (row.get('complete') is True and row.get('track') == track
                and row.get('fingerprint') == source_fingerprint(track, data, config))
    except (ValueError, OSError):
        return False


def initialize(archive, models, data, output, missing_reports=()):
    global MODULES, DATA, OUT, MISSING, CONFIG
    MODULES = initialize_measurements(archive, models, data)
    DATA, OUT = Path(data), Path(output)
    MISSING = {key for report in missing_reports for key in json.loads(Path(report).read_text())['missing']}
    CONFIG = measurement_config(archive, models)


def json_value(x):
    import numpy as np
    if isinstance(x, np.generic):
        return x.item()
    raise TypeError(type(x).__name__)


def measure_track(track):
    import cv2
    import pandas as pd
    folder = DATA / 'reference-images-512x960-v1' / track / MODULES.audit.SUB
    fingerprint = source_fingerprint(track, DATA, CONFIG)
    prefix = folder.relative_to(DATA).as_posix()
    manifest = json.loads((folder / 'manifest.json').read_text()) if (folder / 'manifest.json').is_file() else {'images': []}
    if not (folder / 'manifest.json').is_file() and f'{prefix}/manifest.json' not in MISSING:
        raise FileNotFoundError(f'Unaccounted missing manifest: {track}')
    pose = folder / 'selected_poses.parquet'
    pose_rows = pd.read_parquet(pose) if pose.is_file() else pd.DataFrame(columns=['category'])
    if not pose.is_file() and f'{prefix}/selected_poses.parquet' not in MISSING:
        raise FileNotFoundError(f'Unaccounted missing poses: {track}')
    categories = {row.category for row in pose_rows.itertuples()}
    for row in manifest['images']:
        if not (folder / (row['category'] + '.png')).is_file() and f'{prefix}/{row["category"]}.png' not in MISSING:
            raise FileNotFoundError(f'Manifest image is missing: {track}/{row["category"]}')
        if row['category'] not in categories and pose.is_file():
            raise ValueError(f'Selected image has no image_keypoints: {track}/{row["category"]}')
    # HF source README: image_keypoints are already on the 512x960 canvas.
    p14 = MODULES.audit.do_track(track)
    p59 = MODULES.audit59.do_track(track)
    tooth = {'track': track, 'ok': False}
    mouth = next((r for r in p14 if r['slot'] == 'mouth_open' and r.get('mp_ok')), None)
    if mouth:
        img = cv2.cvtColor(cv2.imread(str(folder / 'mouth_open.png')), cv2.COLOR_BGR2RGB)
        s = 1.25 * max(mouth['bx1'] - mouth['bx0'], mouth['by1'] - mouth['by0'])
        cx, cy = (mouth['bx0'] + mouth['bx1']) / 2, (mouth['by0'] + mouth['by1']) / 2
        crop = cv2.resize(MODULES.pscan.crop(img, cx, cy, s)[0], (288, 288), interpolation=cv2.INTER_CUBIC)
        lm = MODULES.allteeth.landmarks(crop)
        if lm is not None:
            try:
                tooth.update(ok=True, **MODULES.allteeth.rows(crop, lm))
            except (ValueError, cv2.error, IndexError, FloatingPointError) as error:
                tooth['measurement_error'] = type(error).__name__
    if source_fingerprint(track, DATA, CONFIG) != fingerprint:
        raise RuntimeError(f'Inputs changed during measurement: {track}')
    payload = {'track': track, 'p14': p14, 'p59': p59, 'teeth': tooth, 'complete': True, 'fingerprint': fingerprint}
    dest = OUT / 'tracks' / (track.replace('/', '__') + '.json')
    tmp = dest.with_suffix('.tmp')
    tmp.write_text(json.dumps(payload, default=json_value) + '\n')
    tmp.replace(dest)
    return {'track': track, 'images': len(manifest['images'])}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tracks', type=Path, required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--archive-root', type=Path, default=ARCHIVE)
    p.add_argument('--model-root', type=Path, default=MODELS)
    p.add_argument('--workers', type=int, default=32)
    p.add_argument('--limit', type=int)
    p.add_argument('--missing-report', type=Path, action='append', default=[])
    args = p.parse_args()
    tracks = [t.strip() for t in args.tracks.read_text().splitlines() if t.strip()]
    if args.limit:
        tracks = tracks[:args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'tracks').mkdir(exist_ok=True)
    config = measurement_config(args.archive_root, args.model_root)
    pending = [t for t in tracks if not checkpoint_valid(t, args.data_root, args.output_dir, config)]
    print(json.dumps({'tracks': len(tracks), 'pending': len(pending), 'workers': args.workers}), flush=True)
    started = time.monotonic()
    failures = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize,
                             initargs=(args.archive_root, args.model_root, args.data_root, args.output_dir, args.missing_report)) as pool:
        futures = {pool.submit(measure_track, t): t for t in pending}
        for n, future in enumerate(as_completed(futures), 1):
            try:
                future.result()
            except Exception as error:
                failures.append({'track': futures[future], 'type': type(error).__name__, 'error': str(error)})
            if n % 100 == 0 or n == len(pending):
                print(json.dumps({'finished': n, 'pending_total': len(pending), 'errors': len(failures),
                                  'seconds': round(time.monotonic() - started, 1)}), flush=True)
    (args.output_dir / 'failures.json').write_text(json.dumps(failures, indent=2) + '\n')
    if failures:
        raise SystemExit(f'{len(failures)} incomplete tracks; inspect failures.json and retry')
    import pandas as pd
    tables = {'p14': [], 'p59': [], 'teeth': []}
    for t in tracks:
        payload = json.loads((args.output_dir / 'tracks' / (t.replace('/', '__') + '.json')).read_text())
        assert payload['complete'] and payload['track'] == t
        tables['p14'].extend(payload['p14'])
        tables['p59'].extend(payload['p59'])
        tables['teeth'].append(payload['teeth'])
    for name, rows in tables.items():
        pd.DataFrame(rows).to_parquet(args.output_dir / (name + '.parquet'), index=False)
    summary = {'tracks': len(tracks), 'rows': {k: len(v) for k, v in tables.items()},
               'seconds': time.monotonic() - started, 'complete': True}
    (args.output_dir / 'complete.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
