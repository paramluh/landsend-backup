"""Independently check final selections against the strict Richmond slot rules."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import copy
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import struct

import cv2
import numpy as np
import pandas as pd

from finalize_bank import HF_PREFIX, validate_selection
from upstream import REFERENCE_PREFIX, SELECTOR_FOLDER, SLOTS
from video_scan import _design, fit_selector_proxy, selector_features

RULES_ROOT = Path('/mnt/paramfs/vansh/richmond-7/reference-bank')


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(row, key):
    value = row.get(key)
    require(isinstance(value, (int, float, np.number)) and math.isfinite(value),
            f'Missing/nonfinite measurement: {key}={value!r}')
    return float(value)


def boolean(row, key):
    value = row.get(key)
    require(isinstance(value, (bool, np.bool_)), f'Missing/nonboolean measurement: {key}={value!r}')
    return bool(value)


def check_quality(prop, row):
    require(boolean(row, 'mp_ok'), 'No detected face')
    require(number(row, 'blur') >= 60, 'Face blur < 60')
    left, right = number(row, 'blinkL'), number(row, 'blinkR')
    limit = .6 if prop == 4 else .5
    require(not (left > limit and right > limit), 'Both eyes closed')
    if prop <= 4:
        require(number(row, 'hand_in_face') == 0, 'Selector hand keypoints overlap face')
        require(number(row, 'oof') == 0, 'Face landmark outside frame')
        yaw = number(row, 'yaw')
        if prop <= 3:
            require(number(row, 'bx0') >= 2 and number(row, 'by0') >= 2
                    and number(row, 'bx1') <= 510 and number(row, 'by1') <= 958,
                    'Face box within two pixels of frame edge')
        if prop == 1:
            require(abs(yaw) < 10, 'Front |yaw| >= 10')
            require(max(left, right) < .35, 'Front eye blink >= .35')
            return number(row, 'blur')
        if prop in (2, 3):
            require(20 <= yaw * (-1 if prop == 2 else 1) <= 50, 'Side yaw outside signed [20,50]')
            require(number(row, 'score') >= .3, 'Side selector score < .3')
            return number(row, 'score')
        require(abs(yaw) <= 15, 'Mouth |yaw| > 15')
        require(boolean(row, 't_ok'), 'Missing successful teeth measurement')
        require(number(row, 'up') >= 1 and number(row, 'lo') >= 1 and number(row, 'span') >= .4,
                'Mouth teeth thresholds failed')
        return row.get('jaw', np.nan)

    hands = number(row, 'n_hands')
    require(hands in (1, 2), 'Expected one or two measured hands')
    hands = int(hands)
    sharpness = [row.get(f'h{i}_sharp', np.nan) for i in range(hands)]
    if prop in (5, 6, 8, 9):
        require(all(number(row, f'h{i}_face_ov') <= .3 for i in range(hands)), 'Hand overlaps face')
    if prop in (5, 6):
        sharpness = [number(row, f'h{i}_sharp') for i in range(hands)]
        require(hands == 2, 'Both hands required')
        require(all(number(row, f'h{i}_inframe') >= 20 for i in range(2)), 'Hand landmarks outside frame')
        require(min(sharpness) >= (80 if prop == 5 else 60), 'Hand sharpness threshold failed')
        if prop == 5:
            require(all(number(row, f'h{i}_ext') >= 3 for i in range(2)), 'Fewer than three extended fingers')
            require(number(row, 'hh_dist') > .15, 'Visible hands are touching')
        else:
            require(boolean(row, 'gate'), 'Strict contact gate failed')
    else:
        targets = [row.get(f'h{i}_target') for i in range(hands)]
        require(all(target in ('hair', 'face', 'clothing', 'none') for target in targets),
                'Missing hand region measurement')
        require(('hair' if prop == 7 else 'clothing') in targets, 'Required hand region absent')
        sharpness = [value for value in sharpness if isinstance(value, (int, float, np.number)) and math.isfinite(value)]
        require(sharpness and max(sharpness) >= 80, 'No hand with sharpness >= 80')
    return min(sharpness) if prop == 5 else max(sharpness)


def check_crop(row, keypoints):
    x, y, width, height = (number(row, key) for key in ('cx0', 'cy0', 'cw', 'ch'))
    if int(row['property']) >= 5:
        require((x, y, width, height) == (0, 0, 512, 960) and number(row, 'scale') == .5,
                'Body references must halve the full 512x960 frame')
        return
    points = np.asarray(keypoints, dtype=np.float32).reshape(133, 3)[23:91]
    points = points[(points[:, 2] > .3) & np.isfinite(points).all(axis=1)]
    require(len(points) >= 10, 'Fewer than ten confident selector face keypoints')
    # Match selector float32 arithmetic before integer rounding, including top padding.
    x0, y0 = points[:, :2].min(0)
    x1, y1 = points[:, :2].max(0)
    y0 = y0 - .35 * (y1 - y0)
    growth = max(1., (x1 - x0) / 256, (y1 - y0) / 480)
    require(growth <= 1.1, 'Padded selector face requires more than 10% growth')
    native_width = math.ceil(256 * growth - 1e-9)
    require(width in (native_width, min(282, native_width + 2)), 'Face crop does not use native/minimal geometry')
    require(height == round(width * 480 / 256), 'Face crop aspect ratio differs')
    center_x, center_y = (x0 + x1) / 2, (y0 + y1) / 2
    require(x == round(min(max(center_x - width / 2, 0), 512 - width))
            and y == round(min(max(center_y - height / 2, 0), 960 - height)),
            'Face crop is not centered and shifted into the frame')
    require(x0 >= x and y0 >= y and x1 <= x + width and y1 <= y + height,
            'Face crop does not contain the padded selector face')
    require(math.isclose(number(row, 'scale'), 256 / width, rel_tol=0, abs_tol=1e-12), 'Wrong face crop scale')


def thumbnail(value):
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    require(array.shape == (1920,) and np.isfinite(array).all() and (array >= 0).all() and (array <= 255).all(),
            'Missing/invalid full-frame 32x60 gray thumbnail')
    return array


def check_duplicates(rows, thumbnails):
    groups = defaultdict(list)
    for row in rows:
        prop = int(row['property'])
        groups[(row['bank'], row['uuid'], 7 if prop >= 7 else prop)].append(row)
    checked = 0
    for (_, _, group), members in groups.items():
        for i, row in enumerate(members):
            for other in members[:i]:
                scanned = row['origin'] == 'video-scan' or other['origin'] == 'video-scan'
                if not scanned and group != 7:
                    continue
                same_recording = row['recording_id'] == other['recording_id']
                same_track = same_recording and int(row['speaker']) == int(other['speaker'])
                gap = abs(int(row['frame_index']) - int(other['frame_index']))
                pair = f'iid={row["iid"]}/{other["iid"]}'
                if scanned and same_track:
                    require(gap >= 250, f'Scanned same-track references separated by <10 seconds: {pair}')
                if group == 7 and same_recording:
                    require(gap >= 50, f'Hair/clothing references separated by <2 seconds: {pair}')
                mad = float(np.abs(thumbnails[int(row['iid'])] - thumbnails[int(other['iid'])]).mean())
                require(mad >= 6, f'Near-duplicate actor/property-group references, thumbnail MAD={mad}: {pair}')
                checked += 1
    return checked


def indexed_table(path, keys):
    frame = pd.read_parquet(path)
    require(set(keys) <= set(frame) and not frame.duplicated(keys).any(), f'Duplicate/missing evidence identity: {path}')
    return {tuple(row[key] for key in keys): row for row in frame.to_dict('records')}


class Evidence:
    def __init__(self, measurements, source_root, scan_dir, fit_score):
        self.source_root, self.scan_dir = Path(source_root), Path(scan_dir)
        self.hashes = {}
        self.tables = {}
        for name in ('p14', 'p59', 'teeth'):
            path = Path(measurements) / f'{name}.parquet'
            self.hashes[str(path)] = digest(path)
            self.tables[name] = indexed_table(path, ['track'] if name == 'teeth' else ['track', 'slot'])
        self.fit_score, self.proxy = Path(fit_score), None
        self.reset_caches()

    def reset_caches(self):
        self.upstream_track = lru_cache(maxsize=1)(self.read_upstream_track)
        self.scan_chunk = lru_cache(maxsize=1)(self.read_scan_chunk)

    def fork(self):
        result = copy.copy(self)
        result.hashes = {}
        result.reset_caches()
        return result

    def read_upstream_track(self, track):
        folder = self.source_root / REFERENCE_PREFIX / track / SELECTOR_FOLDER
        manifest = json.loads((folder / 'manifest.json').read_text())
        images = {row['category']: row for row in manifest['images']}
        require(len(images) == len(manifest['images']), f'Duplicate manifest category: {track}')
        poses = indexed_table(folder / 'selected_poses.parquet', ['category'])
        return folder, manifest, images, poses

    def read_scan_chunk(self, track, chunk):
        path = self.scan_dir / 'candidates' / track / f'{chunk:06d}.parquet'
        receipt = json.loads(path.with_suffix('.complete.json').read_text())
        actual = digest(path)
        require(receipt['sha256'] == actual and receipt['track'] == track and int(receipt['chunk']) == chunk,
                f'Candidate checkpoint receipt mismatch: {path}')
        self.hashes[str(path)] = actual
        return indexed_table(path, ['track', 'frame_index'])

    def get(self, row):
        track = f'{row["recording_id"]}/speaker_{int(row["speaker"])}'
        prop, frame = int(row['property']), int(row['frame_index'])
        if row['origin'] == 'ref-image':
            folder, manifest, images, poses = self.upstream_track(track)
            slot = row['slot']
            require(manifest.get('person_uuid', row['uuid']) == row['uuid'], 'Manifest actor identity mismatch')
            image, pose = images[slot], poses[(slot,)]
            require(int(image['frame_index']) == frame and int(pose.get('frame_index', frame)) == frame,
                    'Source pose/manifest frame differs from selection')
            require((pose.get('image_width', 512), pose.get('image_height', 960)) == (512, 960), 'Wrong selector pose canvas')
            path = folder / f'{slot}.png'
            raw = path.read_bytes()
            actual = hashlib.sha256(raw).hexdigest()
            require(actual == row['source_sha256'] == image['sha256'], 'Upstream image checksum differs')
            if pose.get('image_sha256'):
                require(actual == pose['image_sha256'], 'Source pose image checksum differs')
            decoded = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            require(decoded is not None and decoded.shape == (960, 512, 3), 'Wrong upstream source dimensions')
            thumb = cv2.resize(cv2.cvtColor(decoded, cv2.COLOR_BGR2GRAY), (32, 60), interpolation=cv2.INTER_AREA)
            measured = dict(self.tables['p14' if prop <= 4 else 'p59'][(track, slot)])
            require(not boolean(measured, 'missing'), 'Upstream image was missing at measurement time')
            require(number(measured, 'frame_index') == frame, 'Upstream measurement frame differs')
            if prop == 4:
                teeth = self.tables['teeth'][(track,)]
                measured.update(teeth, t_ok=teeth['ok'])
            if prop in (2, 3):
                require(number(measured, 'score') == float(image['score']), 'Measured selector score differs from manifest')
            if prop == 6:
                measured['gate'] = measured.get('interaction_gate_passed')
                require(boolean(measured, 'gate') == boolean(image, 'interaction_gate_passed'), 'Contact evidence differs from manifest')
            keypoints = np.stack(pose['image_keypoints']).astype(np.float32)
            return measured, keypoints, thumbnail(thumb)

        measured = dict(self.scan_chunk(track, frame // 7500)[(track, frame)])
        require(measured['uuid'] == row['uuid'] and measured['bank'] == row['bank'], 'Scan actor/bank identity mismatch')
        require(measured['source'] == row['source'], 'Scan source identity differs')
        require(boolean(measured, 'pose_valid'), 'Invalid scan pose')
        for flag in ('in_eval', 'in_bb', 'in_light'):
            require(not boolean(measured, flag), f'Prohibited scan source: {flag}')
        keypoints = np.asarray(measured['kp'], np.float32).reshape(133, 3)
        if prop in (2, 3):
            if self.proxy is None:
                self.hashes[str(self.fit_score)] = digest(self.fit_score)
                self.proxy = fit_selector_proxy(self.fit_score)
            features = np.asarray([selector_features(keypoints)])
            measured['score'] = float((_design(features, -1 if prop == 2 else 1) @ self.proxy)[0])
        require(not boolean(row, 'prepared_image'), 'Strict audit needs the decoded full frame, before cropping')
        raw = Path(row['image_path']).read_bytes()
        require(hashlib.sha256(raw).hexdigest() == row['decoded_image_sha256'], 'Decoded scan image checksum differs')
        decoded = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        require(decoded is not None and decoded.shape == (960, 512, 3), 'Wrong decoded scan dimensions')
        thumb = thumbnail(cv2.resize(cv2.cvtColor(decoded, cv2.COLOR_BGR2GRAY), (32, 60), interpolation=cv2.INTER_AREA))
        require(np.array_equal(thumb, thumbnail(measured['thumb'])), 'Decoded scan thumbnail differs from measured frame')
        return measured, keypoints, thumb


def check_latent_headers(selection, root):
    expected = set()
    for (bank, actor, prop), group in selection.groupby(['bank', 'uuid', 'property']):
        path = Path(root) / bank / actor / f'property_{int(prop)}' / 'latents.safetensors'
        expected.add(path.relative_to(root).as_posix())
        with path.open('rb') as handle:
            prefix = handle.read(8)
            require(len(prefix) == 8, f'Truncated latent header: {path}')
            size = struct.unpack('<Q', prefix)[0]
            require(2 <= size <= 1024 * 1024, f'Invalid latent header size: {path}')
            header = json.loads(handle.read(size))
        tensor, meta = header['video'], header['__metadata__']
        require(tensor['shape'] == [len(group), 48, 30, 16] and tensor['dtype'] == 'BF16', f'Invalid latent shape/dtype: {path}')
        require(meta == {'uuid': actor, 'property': str(int(prop)), 'slot': SLOTS[int(prop) - 1], 'n': str(len(group))},
                f'Latent identity metadata differs: {path}')
        require(tensor['data_offsets'] == [0, len(group) * 48 * 30 * 16 * 2]
                and path.stat().st_size == 8 + size + tensor['data_offsets'][1], f'Wrong latent tensor byte extent: {path}')
    actual = {path.relative_to(root).as_posix() for bank in ('bank', 'beatles-eval')
              for path in (Path(root) / bank).glob('*/property_*/latents.safetensors')}
    require(actual == expected, 'Published latent inventory differs from selected groups')
    return len(expected)


def check_rows(rows, evidence):
    thumbnails = {}
    for row in rows:
        try:
            measured, keypoints, thumb = evidence.get(row)
            rank = check_quality(int(row['property']), measured)
            if pd.isna(rank):
                require(pd.isna(row.get('rank_metric')), 'Missing measured rank was replaced')
            else:
                require(math.isclose(number(row, 'rank_metric'), rank, rel_tol=1e-10, abs_tol=1e-10), 'Selected rank differs from exact measurement')
            if int(row['property']) in (2, 3):
                require(math.isclose(number(row, 'sel_score'), number(measured, 'score'), rel_tol=1e-10, abs_tol=1e-10),
                        'Selected score differs from exact measurement')
            for key in ('blur', 'blinkL', 'blinkR'):
                require(number(row, key) == number(measured, key), f'Selected {key} differs from measurement')
            check_crop(row, keypoints)
            if 'thumb' in row and row['thumb'] is not None:
                require(np.array_equal(thumbnail(row['thumb']), thumb), 'Selected thumbnail differs from source evidence')
            thumbnails[int(row['iid'])] = thumb
        except (ValueError, KeyError, TypeError, OSError) as error:
            raise ValueError(f'iid={row["iid"]}, actor={row["uuid"]}, property={row["property"]}: {error}') from error
    return thumbnails, evidence.hashes


def validate(selection_path, measurements_dir, source_root, scan_dir, fit_score, training_csv=None, bank_root=None, workers=1):
    require(workers >= 1, 'Workers must be positive')
    selection = pd.read_parquet(selection_path)
    validate_selection(selection)
    require(len(selection) > 0, 'Empty final selection')
    for _, group in selection.groupby(['bank', 'uuid', 'property']):
        require(len(group) <= 20 and sorted(group['index']) == list(range(len(group))), 'Slot cap/indices invalid')
    evidence = Evidence(measurements_dir, source_root, scan_dir, fit_score)
    if (selection.origin.eq('video-scan') & selection.property.isin([2, 3])).any():
        evidence.hashes[str(evidence.fit_score)] = digest(evidence.fit_score)
        evidence.proxy = fit_selector_proxy(evidence.fit_score)
    ordered = selection.sort_values(['origin', 'recording_id', 'speaker', 'frame_index'])
    if workers == 1:
        thumbnails, _ = check_rows(ordered.to_dict('records'), evidence)
    else:
        cv2.setNumThreads(1)
        groups = [group.to_dict('records') for _, group in ordered.groupby(['origin', 'recording_id', 'speaker'], sort=False)]
        executor = ThreadPoolExecutor(max_workers=workers)
        thumbnails = {}
        try:
            for checked, hashes in executor.map(lambda rows: check_rows(rows, evidence.fork()), groups):
                thumbnails.update(checked)
                evidence.hashes.update(hashes)
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
    comparisons = check_duplicates(selection.to_dict('records'), thumbnails)
    coverage = None
    if training_csv is not None:
        actors = set(pd.read_csv(training_csv).uuid)
        represented = set(selection.loc[selection.bank.eq('bank'), 'uuid'])
        missing = sorted(actors - represented)
        require(not missing, f'Training actors with no references: {missing}')
        coverage = {'training_csv': str(training_csv), 'sha256': digest(training_csv), 'actors': len(actors), 'missing': missing}
    report = {'strict_quality_valid': True, 'selection_sha256': digest(selection_path), 'images': len(selection),
              'actors': int(selection.uuid.nunique()), 'images_by_origin': dict(Counter(selection.origin)),
              'duplicate_pairs_checked': comparisons, 'evidence_sha256': evidence.hashes, 'training_coverage': coverage,
              'scope': 'Exact stored measurements, source identity, slot predicates, crop geometry, selected-set dedup; source exclusions and encoder values remain covered by finalize_bank.'}
    if bank_root is not None:
        report['latent_headers_checked'] = check_latent_headers(selection, Path(bank_root))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('selection', 'measurements-dir', 'source-root', 'scan-dir', 'fit-score', 'report'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--training-csv', type=Path)
    parser.add_argument('--bank-root', type=Path)
    parser.add_argument('--rules-root', type=Path, default=RULES_ROOT)
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    try:
        report = validate(args.selection, args.measurements_dir, args.source_root, args.scan_dir,
                          args.fit_score, args.training_csv, args.bank_root, args.workers)
        report['rules_sha256'] = {name: digest(args.rules_root / name) for name in ('sinks.md', 'RECIPE.md')}
    except (ValueError, KeyError, TypeError, OSError) as error:
        report = {'strict_quality_valid': False, 'error': str(error)}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_suffix(args.report.suffix + '.tmp')
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    temporary.replace(args.report)
    print(json.dumps({key: value for key, value in report.items() if key != 'evidence_sha256'}), flush=True)
    if not report['strict_quality_valid']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
