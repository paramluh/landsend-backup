"""
Fresh actor-level upstream-image planning and strict reference selection.

Chunk clocks are the archived corpus convention [300*i, 300*(i+1)] seconds.
No prior bank or encoded tensor is read by this module.
"""

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


CSV_ROOT = Path('/mnt/paramfs/vansh/richmond-7/data/csvs/current')
BLACKBANDS_CSV = Path('/mnt/paramfs/vansh/richmond-7/evals/refs-audit/data/drop_union_aloe-h4.csv')
REFERENCE_PREFIX = 'reference-images-512x960-v1'
SELECTOR_FOLDER = 'reference_images/reference-images-512x960-strict-contact-poses-v8'
SLOTS = ('front', 'face_image_left', 'face_image_right', 'mouth_open', 'hands_visible',
         'hands_interacting', 'hair_clothing_01', 'hair_clothing_02', 'hair_clothing_03')
CHUNK_SECONDS, EVAL_MARGIN, CAP = 300, 5.0, 20
SOURCE_WIDTH, SOURCE_HEIGHT, CROP_WIDTH, CROP_HEIGHT = 512, 960, 256, 480
FACE_TOP_PAD, CROP_GROW = 0.35, 1.1
BEATLES_HELD_ROWS = (3, 6, 11, 14, 16, 17, 22, 23, 25, 26, 27, 30, 32, 33, 37, 38)


@dataclass
class Corpus:
    actors: list
    tracks: list
    all_clips: list
    ranges_eval: dict
    ranges_blackbands: dict
    ranges_lighting: dict
    ranges_excess: dict
    source_hashes: dict


def read_csv(path, required):
    with Path(path).open(newline='') as handle:
        reader = csv.DictReader(handle)
        missing = set(required) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f'{path}: missing columns {sorted(missing)}')
        rows = list(reader)
    for row in rows:
        if 'speaker' in row:
            row['speaker'] = int(row['speaker'])
            row['track'] = f"{row['recording_id']}/speaker_{row['speaker']}"
        if 'chunk' in row:
            row['chunk'] = int(row['chunk'])
    return rows


def ranges(rows, margin=0.0):
    result = defaultdict(set)
    for row in rows:
        result[row['track']].add((CHUNK_SECONDS * row['chunk'] - margin,
                                  CHUNK_SECONDS * (row['chunk'] + 1) + margin))
    return {track: sorted(values) for track, values in result.items()}


def in_ranges(time_s, track, intervals):
    return any(start <= time_s <= end for start, end in intervals.get(track, ()))


def timestamp_allowed(corpus, track, time_s, cohort):
    excluded = any(in_ranges(time_s, track, intervals) for intervals in (
        corpus.ranges_eval, corpus.ranges_blackbands, corpus.ranges_lighting,
    ))
    return not excluded and (cohort != 'eval-unseen' or in_ranges(time_s, track, corpus.ranges_excess))


def load_corpus(csv_root=CSV_ROOT, blackbands_csv=BLACKBANDS_CSV, reserved_csvs=()):
    csv_root = Path(csv_root)
    identity = ('uuid', 'session_id', 'recording_id', 'speaker', 'chunk')
    paths = {
        'all': csv_root / 'all.csv',
        'studio': csv_root / 'aloe-studio-no-blackbands-high-quality.csv',
        'app': csv_root / 'aloe-app-selected-clips.csv',
        'all_app': csv_root / 'all-app.csv',
        'all_studio': csv_root / 'all-studio.csv',
        'seen': csv_root / 'aloe-studio-seen-eval.csv',
        'unseen': csv_root / 'aloe-studio-unseen-eval.csv',
        'beatles': csv_root / 'aloe-eval-beatles.csv',
        'excess': csv_root / 'aloe-unseen-excess-chunks.csv',
        'blackbands': Path(blackbands_csv),
    }
    rows = {name: read_csv(path, ('recording_id', 'speaker', 'chunk') if name == 'blackbands' else identity)
            for name, path in paths.items()}
    for name in ('all_app', 'all_studio'):
        if any(row.get('blackbands') not in ('yes', 'no') for row in rows[name]):
            raise ValueError(f'{paths[name]} requires fully labelled blackbands=yes/no')
    if any(row.get('lighting_rejection') not in ('0', '1') for row in rows['all_app']):
        raise ValueError(f"{paths['all_app']} requires lighting_rejection=0/1; no lighting exclusions may be omitted")
    train = {row['uuid'] for row in rows['studio'] + rows['app']}
    unseen = {row['uuid'] for row in rows['unseen']}
    if train & unseen:
        raise ValueError(f'Training and unseen actors overlap: {sorted(train & unseen)}')
    if {row['uuid'] for row in rows['excess']} - unseen:
        raise ValueError('Unseen excess allowlist contains an actor outside the unseen split')
    reserved = rows['seen'] + rows['unseen'] + rows['beatles']
    for index, path in enumerate(reserved_csvs):
        paths[f'extra_reserved_{index}'] = Path(path)
        reserved += read_csv(path, identity)
    blackbands = rows['blackbands'] + [row for name in ('all_app', 'all_studio')
                                      for row in rows[name] if row['blackbands'] == 'yes']
    lighting = [row for row in rows['all_app'] if row['lighting_rejection'] == '1']
    by_track, chunks = {}, defaultdict(set)
    for row in rows['all']:
        track = row['track']
        record = {key: row[key] for key in ('track', 'uuid', 'session_id', 'recording_id', 'speaker')}
        if not record['uuid'] or not record['session_id']:
            raise ValueError(f'Missing track identity or session: {track}')
        if track in by_track and by_track[track] != record:
            raise ValueError(f'Conflicting track identity: {track}')
        by_track[track] = record
        chunks[track].add(row['chunk'])
    required_tracks = {row['track'] for name in ('studio', 'app', 'excess') for row in rows[name]}
    if required_tracks - by_track.keys():
        raise ValueError(f'Selected tracks absent from all.csv: {sorted(required_tracks - by_track.keys())}')
    for name in ('studio', 'app', 'excess'):
        for row in rows[name]:
            if by_track[row['track']]['uuid'] != row['uuid']:
                raise ValueError(f"Actor mismatch between {name} and all.csv: {row['track']}")
    excess_chunks = defaultdict(set)
    for row in rows['excess']:
        excess_chunks[row['track']].add(row['chunk'])
    unavailable = {(row['track'], row['chunk']) for row in reserved + blackbands + lighting}
    selected_tracks = {track for track, row in by_track.items() if row['uuid'] in train} | set(excess_chunks)
    tracks = []
    for track in sorted(selected_tracks):
        record = dict(by_track[track])
        cohort = 'eval-unseen' if record['uuid'] in unseen else 'train'
        allowed = excess_chunks[track] if cohort == 'eval-unseen' else chunks[track]
        usable = sorted(chunk for chunk in allowed if (track, chunk) not in unavailable)
        record.update(cohort=cohort, bank='bank', usable_chunks=usable, usable_chunk_count=len(usable))
        tracks.append(record)
    counts = Counter(row['uuid'] for row in tracks)
    actors = [{'uuid': actor, 'cohort': 'eval-unseen' if actor in unseen else 'train',
               'eligible_track_count': counts[actor]} for actor in sorted(train | unseen)]
    return Corpus(actors, tracks, rows['all'], ranges(reserved, EVAL_MARGIN), ranges(blackbands),
                  ranges(lighting), ranges(rows['excess']),
                  {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths.values()})


def write_inventory(corpus, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (('actors', corpus.actors), ('tracks', corpus.tracks)):
        with (output_dir / f'{name}.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows({key: json.dumps(value) if isinstance(value, list) else value
                              for key, value in row.items()} for row in rows)
    keys = [f"{REFERENCE_PREFIX}/{row['track']}/{SELECTOR_FOLDER}/{filename}"
            for row in corpus.tracks for filename in ('manifest.json', 'selected_poses.parquet',
                                                       *(f'{slot}.png' for slot in SLOTS))]
    (output_dir / 'upstream-files.txt').write_text('\n'.join(keys) + '\n')
    (output_dir / 'tracks.txt').write_text('\n'.join(row['track'] for row in corpus.tracks) + '\n')
    summary = {'actors': len(corpus.actors), 'actors_by_cohort': dict(Counter(row['cohort'] for row in corpus.actors)),
               'tracks': len(corpus.tracks), 'upstream_keys': len(keys),
               'actors_without_eligible_tracks': [row['uuid'] for row in corpus.actors if not row['eligible_track_count']],
               'source_hashes': corpus.source_hashes,
               'chunk_clock': '300-second chunks; reserved chunks excluded with inclusive +/-5-second margins',
               'selection_policy': 'fresh strict rules for every actor; no old bank reuse',
               'blackband_policy': 'legacy exclusion CSV plus final all-app and all-studio blackbands=yes rows'}
    (output_dir / 'inventory.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


def face_box(keypoints):
    import numpy as np
    face = np.asarray(keypoints)[23:91]
    face = face[(face[:, 2] > 0.3) & np.isfinite(face).all(axis=1)]
    if len(face) < 10:
        return None
    x0, y0 = face[:, :2].min(axis=0)
    x1, y1 = face[:, :2].max(axis=0)
    return x0, y0 - FACE_TOP_PAD * (y1 - y0), x1, y1


def crop(box):
    import math
    width, height = box[2] - box[0], box[3] - box[1]
    scale = max(1.0, width / CROP_WIDTH, height / CROP_HEIGHT)
    if scale > CROP_GROW:
        return None
    width = math.ceil(CROP_WIDTH * scale - 1e-9)
    height = round(width * CROP_HEIGHT / CROP_WIDTH)
    if width > 282 or height > 529:
        return None
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    for candidate_width in (width, min(282, width + 2)) if scale < CROP_GROW else (width,):
        candidate_height = round(candidate_width * CROP_HEIGHT / CROP_WIDTH)
        x0 = round(min(max(cx - candidate_width / 2, 0), SOURCE_WIDTH - candidate_width))
        y0 = round(min(max(cy - candidate_height / 2, 0), SOURCE_HEIGHT - candidate_height))
        if box[0] >= x0 and box[1] >= y0 and box[2] <= x0 + candidate_width and box[3] <= y0 + candidate_height:
            return x0, y0, candidate_width, candidate_height
    return None


def image_thumbnail(payload):
    import io

    import cv2
    import numpy as np
    from PIL import Image

    with Image.open(io.BytesIO(payload)) as image:
        pixels = np.asarray(image.convert('RGB'))
    if pixels.shape != (SOURCE_HEIGHT, SOURCE_WIDTH, 3):
        raise ValueError('Reference images must have 512x960 RGB geometry')
    return cv2.resize(cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY), (32, 60),
                      interpolation=cv2.INTER_AREA).reshape(-1).tolist()


def checked_thumbnail(value):
    import numpy as np

    thumbnail = np.asarray(value, dtype=np.float32).reshape(-1)
    if len(thumbnail) != 32 * 60 or not np.isfinite(thumbnail).all():
        raise ValueError('Missing/invalid reference thumbnail; rebuild the upstream plan')
    return thumbnail


def build_plan(corpus, data_root, known_missing=()):
    import numpy as np
    import pandas as pd

    data_root, known_missing = Path(data_root), set(known_missing)
    rows, issues = [], []

    def available(key):
        if (data_root / key).is_file():
            return True
        confirmed = key in known_missing
        filename = key.rsplit('/', 1)[-1]
        issues.append({'kind': 'confirmed_hf_missing' if confirmed else 'input_not_downloaded',
                       'asset_type': 'metadata' if filename in ('manifest.json', 'selected_poses.parquet') else 'image',
                       'source_key': key})
        return False

    for record in corpus.tracks:
        track, cohort = record['track'], record['cohort']
        prefix = f'{REFERENCE_PREFIX}/{track}/{SELECTOR_FOLDER}'
        manifest_key, poses_key = f'{prefix}/manifest.json', f'{prefix}/selected_poses.parquet'
        manifest_present, poses_present = available(manifest_key), available(poses_key)
        images, keypoints, pose_records = {}, {}, {}
        if manifest_present:
            manifest = json.loads((data_root / manifest_key).read_text())
            if manifest.get('person_uuid', record['uuid']) != record['uuid']:
                raise ValueError(f'Actor identity mismatch in {manifest_key}')
            for image in manifest['images']:
                slot = image['category']
                if slot in images:
                    raise ValueError(f'Duplicate manifest category: {manifest_key}: {slot}')
                images[slot] = image
        if poses_present:
            poses = pd.read_parquet(data_root / poses_key)
            if not {'category', 'image_keypoints'}.issubset(poses.columns):
                raise ValueError(f'Missing category/image_keypoints columns: {poses_key}')
            for pose in poses.to_dict('records'):
                slot = pose['category']
                if slot in keypoints:
                    raise ValueError(f'Duplicate pose category: {poses_key}: {slot}')
                value = np.stack(pose['image_keypoints']).astype(np.float32)
                if value.shape != (133, 3):
                    raise ValueError(f'Expected image_keypoints[133,3], got {value.shape}: {poses_key}')
                if (pose.get('image_width', SOURCE_WIDTH), pose.get('image_height', SOURCE_HEIGHT)) != (SOURCE_WIDTH, SOURCE_HEIGHT):
                    raise ValueError(f'image_keypoints canvas must be 512x960: {poses_key}')
                # HF distinguishes keypoints (256x480) from image_keypoints (already 512x960).
                keypoints[slot], pose_records[slot] = value, pose
        for property_number, slot in enumerate(SLOTS, 1):
            source = f'{prefix}/{slot}.png'
            image = images.get(slot)
            row = {key: record[key] for key in ('track', 'uuid', 'session_id', 'recording_id', 'speaker', 'cohort', 'bank')}
            row.update(property=property_number, slot=slot, iid=len(rows), source=source,
                       has_manifest_image=image is not None, has_image=False, has_pose=slot in keypoints,
                       source_incomplete=not (manifest_present and poses_present),
                       source_status='metadata-missing' if not (manifest_present and poses_present) else 'selector-empty',
                       frame_index=None, time_s=None, sel_score=None, gate=None, source_sha256=None, thumb=None,
                       in_eval=False, in_bb=False, in_light=False, in_excess=False,
                       face_box=False, crop_fit=False, cx0=None, cy0=None, cw=None, ch=None, scale=None)
            if image is not None:
                row['has_image'] = available(source)
                row['source_incomplete'] |= not row['has_image']
                if manifest_present and poses_present:
                    row['source_status'] = 'complete' if row['has_image'] else 'png-missing'
                frame_index = int(image['frame_index'])
                if frame_index < 0 or frame_index != float(image['frame_index']):
                    raise ValueError(f'Invalid frame_index: {source}')
                time_s = frame_index / 25.0
                if abs(time_s - float(image['time_s'])) > 1 / 25 + 1e-6:
                    raise ValueError(f'25fps frame/time mismatch: {source}')
                row.update(frame_index=frame_index, time_s=time_s, sel_score=image.get('score'),
                           gate=image.get('interaction_gate_passed'),
                           in_eval=in_ranges(time_s, track, corpus.ranges_eval),
                           in_bb=in_ranges(time_s, track, corpus.ranges_blackbands),
                           in_light=in_ranges(time_s, track, corpus.ranges_lighting),
                           in_excess=in_ranges(time_s, track, corpus.ranges_excess))
                if poses_present and slot not in keypoints:
                    issues.append({'kind': 'missing_pose_category', 'source_key': poses_key, 'slot': slot})
                    row['source_incomplete'], row['source_status'] = True, 'pose-category-missing'
                pose = pose_records.get(slot, {})
                if pose and pose.get('frame_index', frame_index) != frame_index:
                    issues.append({'kind': 'pose_manifest_frame_mismatch', 'source_key': poses_key, 'slot': slot})
                if row['has_image']:
                    payload = (data_root / source).read_bytes()
                    actual_sha256 = hashlib.sha256(payload).hexdigest()
                    row['source_sha256'] = actual_sha256
                    expected_hashes = [('manifest', image.get('sha256'))]
                    if pose:
                        expected_hashes.append(('parquet', pose.get('image_sha256')))
                    for location, expected in expected_hashes:
                        if not expected:
                            issues.append({'kind': 'missing_image_checksum', 'source_key': source, 'location': location})
                        elif expected != actual_sha256:
                            issues.append({'kind': 'image_checksum_mismatch', 'source_key': source,
                                           'location': location, 'expected': expected, 'actual': actual_sha256})
                    if all(expected == actual_sha256 for _, expected in expected_hashes):
                        row['thumb'] = image_thumbnail(payload)
                if property_number <= 4:
                    box = face_box(keypoints[slot]) if slot in keypoints else None
                    window = crop(box) if box is not None else None
                    row.update(face_box=box is not None, crop_fit=window is not None)
                    if window is not None:
                        row.update(zip(('cx0', 'cy0', 'cw', 'ch'), window))
                        row['scale'] = CROP_WIDTH / window[2]
                else:
                    row.update(cx0=0, cy0=0, cw=SOURCE_WIDTH, ch=SOURCE_HEIGHT, scale=0.5,
                               crop_fit=True, face_box=True)
            row['excluded'] = (not row['has_image'] or not row['has_pose'] or row['in_eval'] or row['in_bb']
                               or row['in_light'] or (cohort == 'eval-unseen' and not row['in_excess']))
            row['candidate'] = not row['excluded'] and row['crop_fit']
            rows.append(row)
    plan = pd.DataFrame(rows)
    plan['frame_index'] = plan['frame_index'].astype('Int64')
    diagnostics = {'rows': len(rows), 'candidates': int(plan['candidate'].sum()),
                   'issue_counts': dict(Counter(issue['kind'] for issue in issues)), 'issues': issues,
                   'fatal_input_issues': sum(issue['kind'] != 'confirmed_hf_missing' for issue in issues),
                   'source_status_counts': {str(key): int(value) for key, value in plan.source_status.value_counts().items()},
                   'incomplete_source_tracks': sorted(set(plan.loc[plan.source_incomplete, 'track'])),
                   'actors_with_incomplete_source': sorted(set(plan.loc[plan.source_incomplete, 'uuid'])),
                   'keypoint_canvas': 'image_keypoints already uses 512x960; no additional scaling',
                   'png_checksums': 'Available manifest PNGs are checked against manifest.sha256 and, when present, the corresponding parquet.image_sha256',
                   'source_hashes': corpus.source_hashes}
    return plan, diagnostics


def select_upstream(plan, p14, p59, teeth):
    import numpy as np
    import pandas as pd

    def unique(frame, keys, name):
        missing = set(keys) - set(frame.columns)
        if missing or frame.duplicated(keys).any():
            raise ValueError(f'{name}: missing identity columns {sorted(missing)} or duplicate {keys}')

    unique(plan, ['track', 'slot'], 'plan')
    unique(p14, ['track', 'slot'], 'p14')
    unique(p59, ['track', 'slot'], 'p59')
    unique(teeth, ['track'], 'teeth')
    plan = plan.reset_index(drop=True).copy()
    measured = pd.concat([p14[['track', 'slot']], p59[['track', 'slot']]])
    missing = plan.loc[plan.candidate, ['track', 'slot']].merge(measured, how='left', indicator=True)
    if (missing['_merge'] != 'both').any():
        raise ValueError(f"Missing measurements for {(missing['_merge'] != 'both').sum()} candidate images")
    measurement_frames = pd.concat([p14[['track', 'slot', 'frame_index']], p59[['track', 'slot', 'frame_index']]])
    frame_check = plan.loc[plan.candidate, ['track', 'slot', 'frame_index']].merge(
        measurement_frames, on=['track', 'slot'], suffixes=('_plan', '_measured'), validate='one_to_one',
    )
    if frame_check.frame_index_measured.isna().any() or not frame_check.frame_index_plan.eq(frame_check.frame_index_measured).all():
        raise ValueError('Measurement frame_index differs from the planned source image')
    mouth_tracks = set(p14.loc[p14.slot.eq('mouth_open') & p14.mp_ok.eq(True), 'track'])
    if mouth_tracks - set(teeth.track):
        raise ValueError(f'Teeth measurements missing for {len(mouth_tracks - set(teeth.track))} detected mouths')
    a = p14.merge(teeth.rename(columns={'ok': 't_ok'}).assign(slot='mouth_open'), on=['track', 'slot'], how='left')
    d = p59.copy()

    def column(frame, key):
        return frame[key] if key in frame else pd.Series(np.nan, index=frame.index)

    def finite(frame, *keys):
        return pd.concat([np.isfinite(pd.to_numeric(column(frame, key), errors='coerce')) for key in keys], axis=1).all(axis=1)

    a['f_face'] = a.mp_ok.eq(True) & a.missing.eq(False)
    a['f_eyes'] = finite(a, 'blinkL', 'blinkR') & ~((a.blinkL > 0.5) & (a.blinkR > 0.5))
    a['f_eyes_full'] = finite(a, 'blinkL', 'blinkR') & ~((a.blinkL > 0.6) & (a.blinkR > 0.6))
    a['f_sharp'] = a.blur >= 60
    a['f_hand'] = a.hand_in_face.eq(0)
    a['f_eyes_open'] = finite(a, 'blinkL', 'blinkR') & a[['blinkL', 'blinkR']].max(axis=1).lt(0.35)
    a['f_cut'] = finite(a, 'oof', 'bx0', 'by0', 'bx1', 'by1') & ~((a.oof > 0) | (a.bx0 < 2) | (a.by0 < 2)
                                                                                | (a.bx1 > SOURCE_WIDTH - 2) | (a.by1 > SOURCE_HEIGHT - 2))
    turned_yaw = a.yaw * np.where(a.slot.eq('face_image_left'), -1, 1)
    a['f_front'] = a.yaw.abs() < 10
    a['f_score'] = a.score >= 0.3
    a['f_turn'] = turned_yaw.between(20, 50)
    a['f_teeth'] = (a.up >= 1.0) & (a.lo >= 1.0) & (a.span >= 0.4) & a.t_ok.eq(True)
    a['f_mouth_yaw'] = a.yaw.abs() <= 15
    a['f_oof0'] = a.oof.eq(0)
    filters = {
        1: ['f_face', 'f_front', 'f_eyes', 'f_eyes_open', 'f_hand', 'f_sharp', 'f_cut'],
        2: ['f_face', 'f_score', 'f_turn', 'f_eyes', 'f_hand', 'f_sharp', 'f_cut'],
        4: ['f_face', 'f_teeth', 'f_eyes_full', 'f_hand', 'f_sharp', 'f_mouth_yaw', 'f_oof0'],
    }
    filters[3] = filters[2]
    a['rank'] = np.select([a.slot.eq('front'), a.slot.eq('mouth_open')], [a.blur, a.jaw], a.score)
    p14_columns = ['track', 'slot', 'rank', 'yaw', 'blinkL', 'blinkR', 'blur', 'jaw', 'up', 'lo', 'span', 'hand_in_face']
    m14 = a[p14_columns + sorted({name for values in filters.values() for name in values})]

    def hand(key):
        return column(d, key).fillna(0)

    d['f_face_ok'] = (d.mp_ok.eq(True) & d.missing.eq(False) & finite(d, 'blinkL', 'blinkR')
                      & ~((d.blinkL > 0.5) & (d.blinkR > 0.5)) & (d.blur >= 60))
    d['f_two_complete_sharp'] = (d.n_hands.eq(2) & (hand('h0_inframe') >= 20) & (hand('h1_inframe') >= 20)
                                & (hand('h0_sharp') >= 80) & (hand('h1_sharp') >= 80))
    d['f_fingers3'] = (hand('h0_ext') >= 3) & (hand('h1_ext') >= 3)
    d['f_not_touching'] = hand('hh_dist') > 0.15
    d['f_no_hand_on_face'] = ~((hand('h0_face_ov') > 0.3) | (hand('h1_face_ov') > 0.3))
    d['f_gate'] = column(d, 'interaction_gate_passed').eq(True)
    d['f_two_inframe'] = d.n_hands.eq(2) & (hand('h0_inframe') >= 20) & (hand('h1_inframe') >= 20)
    d['f_hair'] = column(d, 'h0_target').eq('hair') | column(d, 'h1_target').eq('hair')
    d['f_sharp_hand'] = (hand('h0_sharp') >= 80) | (hand('h1_sharp') >= 80)
    d['f_p6_sharp'] = np.fmin(hand('h0_sharp'), hand('h1_sharp')) >= 60
    d['f_clothing'] = column(d, 'h0_target').eq('clothing') | column(d, 'h1_target').eq('clothing')
    filters.update({
        5: ['f_two_complete_sharp', 'f_fingers3', 'f_not_touching', 'f_no_hand_on_face', 'f_face_ok'],
        6: ['f_gate', 'f_two_inframe', 'f_p6_sharp', 'f_no_hand_on_face', 'f_face_ok'],
        7: ['f_hair', 'f_sharp_hand', 'f_face_ok'],
        8: ['f_sharp_hand', 'f_no_hand_on_face', 'f_clothing', 'f_face_ok'],
    })
    filters[9] = filters[8]
    d['rank'] = np.where(d.slot.eq('hands_visible'), np.fmin(hand('h0_sharp'), hand('h1_sharp')),
                         np.fmax(hand('h0_sharp'), hand('h1_sharp')))
    p59_columns = ['track', 'slot', 'rank', 'n_hands', 'h0_sharp', 'h1_sharp', 'hh_dist', 'blinkL', 'blinkR', 'blur']
    m59 = d.reindex(columns=p59_columns + sorted({name for key in (5, 6, 7, 8) for name in filters[key]}))
    full = plan.merge(pd.concat([m14, m59]), on=['track', 'slot'], how='left', validate='one_to_one')
    filter_columns = sorted({name for values in filters.values() for name in values})
    full[filter_columns] = full[filter_columns].fillna(False).astype(bool)
    full['pass'] = False
    for property_number, names in filters.items():
        mask = full.property.eq(property_number)
        full.loc[mask, 'pass'] = full.loc[mask, names].all(axis=1)
    full['pass'] &= full.candidate
    full['sel'], full['index'] = False, -1
    passing = full[full['pass']].sort_values(['rank', 'track'], ascending=[False, True], na_position='last', kind='stable')
    passing['session_round'] = passing.groupby(['uuid', 'property', 'session_id']).cumcount()
    passing['session_rank'] = passing.groupby(['uuid', 'property', 'session_id'])['rank'].transform('max')
    passing = passing.sort_values(['uuid', 'property', 'session_round', 'session_rank', 'session_id', 'track'],
                                  ascending=[True, True, True, False, True, True], na_position='last', kind='stable')
    counts, hair = Counter(), defaultdict(list)
    full['dedup_rejected'] = False
    for index, row in passing.iterrows():
        actor = row['bank'], row.uuid
        group = (*actor, int(row.property))
        if counts[group] >= CAP:
            continue
        thumbnail = checked_thumbnail(row.get('thumb'))
        if row.property >= 7:
            # Time is recording-local; appearance duplicates are actor-wide.
            if any((recording == row.recording_id and abs(seconds - row.time_s) < 2)
                   or np.abs(thumbnail - other).mean() < 6
                   for recording, seconds, other in hair[actor]):
                full.loc[index, 'dedup_rejected'] = True
                continue
            hair[actor].append((row.recording_id, row.time_s, thumbnail))
        full.loc[index, ['sel', 'index']] = True, counts[group]
        counts[group] += 1
    columns = ['uuid', 'bank', 'cohort', 'property', 'slot', 'index', 'recording_id', 'speaker', 'session_id',
               'frame_index', 'time_s', 'source', 'source_sha256', 'cx0', 'cy0', 'cw', 'ch', 'scale', 'rank', 'sel_score', 'gate',
               'yaw', 'blinkL', 'blinkR', 'blur', 'jaw', 'up', 'lo', 'span', 'hand_in_face', 'n_hands',
               'h0_sharp', 'h1_sharp', 'hh_dist', 'iid', 'thumb']
    selected = full[full.sel].sort_values(['uuid', 'property', 'index'])[columns].rename(columns={'rank': 'rank_metric'})
    selected['origin'] = 'ref-image'
    selected['frame_index'] = selected['frame_index'].astype('int64')
    if full.loc[full.sel, ['in_eval', 'in_bb', 'in_light', 'excluded']].any().any():
        raise ValueError('Selected a prohibited or unusable image')
    return full, selected


def select_beatles(plan, p14, p59, teeth, beatles_rows, held_rows=BEATLES_HELD_ROWS):
    if not held_rows or min(held_rows) < 0 or max(held_rows) >= len(beatles_rows):
        raise ValueError('Beatles held row indices are outside the provided CSV')
    held_recordings = {beatles_rows[index]['recording_id'] for index in held_rows}
    source_tracks = {row['track'] for row in beatles_rows}
    subset = plan[plan.track.isin(source_tracks)].copy()
    if set(subset.track) != source_tracks:
        raise ValueError('The upstream plan is missing a Beatles donor/held track')
    subset['in_eval'] = subset.recording_id.isin(held_recordings)
    subset['bank'], subset['cohort'] = 'beatles-eval', 'beatles-eval'
    # This separate benchmark permits donor Beatles recordings; its held recordings remain forbidden in full.
    subset['excluded'] = (~subset.has_image | ~subset.has_pose | subset.in_bb | subset.in_light | subset.in_eval)
    subset['candidate'] = ~subset.excluded & subset.crop_fit
    full, selected = select_upstream(subset, p14[p14.track.isin(source_tracks)],
                                     p59[p59.track.isin(source_tracks)], teeth[teeth.track.isin(source_tracks)])
    if set(selected.recording_id) & held_recordings:
        raise ValueError('Beatles selected references overlap held recordings')
    receipt = {'held_rows': list(held_rows), 'held_recordings': sorted(held_recordings),
               'source_track_count': len(source_tracks), 'selected_donor_tracks': int(selected[['recording_id', 'speaker']].drop_duplicates().shape[0]),
               'selected_images': len(selected), 'actors': len({row['uuid'] for row in beatles_rows}),
               'policy': 'Fixed historical 16 rollout rows; references only from other Beatles recordings, with fresh strict gates and blackband/lighting exclusions'}
    return full, selected, receipt


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command', required=True)
    for name in ('inventory', 'plan'):
        command = subparsers.add_parser(name)
        command.add_argument('--csv-root', type=Path, default=CSV_ROOT)
        command.add_argument('--blackbands-csv', type=Path, default=BLACKBANDS_CSV)
        command.add_argument('--reserved-csv', type=Path, action='append', default=[])
        command.add_argument('--output-dir', type=Path, required=True)
        if name == 'plan':
            command.add_argument('--data-root', type=Path, required=True)
            command.add_argument('--missing-report', type=Path, action='append', default=[])
    selection = subparsers.add_parser('select')
    for name in ('plan', 'p14', 'p59', 'teeth', 'actors', 'output-dir'):
        selection.add_argument(f'--{name}', type=Path, required=True)
    beatles = subparsers.add_parser('select-beatles')
    for name in ('plan', 'p14', 'p59', 'teeth', 'output-dir'):
        beatles.add_argument(f'--{name}', type=Path, required=True)
    beatles.add_argument('--beatles-csv', type=Path, default=CSV_ROOT / 'aloe-eval-beatles.csv')
    beatles.add_argument('--held-rows', type=int, nargs='+', default=BEATLES_HELD_ROWS)
    args = parser.parse_args()
    if args.command == 'inventory':
        corpus = load_corpus(args.csv_root, args.blackbands_csv, args.reserved_csv)
        print(json.dumps(write_inventory(corpus, args.output_dir)))
        return
    import pandas as pd
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.command == 'plan':
        corpus = load_corpus(args.csv_root, args.blackbands_csv, args.reserved_csv)
        known_missing = {key for report in args.missing_report for key in json.loads(report.read_text())['missing']}
        plan, diagnostics = build_plan(corpus, args.data_root, known_missing)
        (args.output_dir / 'input-diagnostics.json').write_text(json.dumps(diagnostics, indent=2) + '\n')
        if diagnostics['fatal_input_issues']:
            raise ValueError(f"{diagnostics['fatal_input_issues']} incomplete/mismatched inputs; see input-diagnostics.json")
        plan.to_parquet(args.output_dir / 'plan.parquet', index=False)
        print(json.dumps({key: value for key, value in diagnostics.items() if key != 'issues'}))
    elif args.command == 'select-beatles':
        tables = [pd.read_parquet(getattr(args, name)) for name in ('plan', 'p14', 'p59', 'teeth')]
        rows = read_csv(args.beatles_csv, ('uuid', 'recording_id', 'speaker'))
        full, selected, receipt = select_beatles(*tables, rows, args.held_rows)
        full.to_parquet(args.output_dir / 'beatles-selection-full.parquet', index=False)
        selected.to_parquet(args.output_dir / 'beatles-selection.parquet', index=False)
        selected.to_csv(args.output_dir / 'beatles-selection.csv', index=False)
        (args.output_dir / 'beatles-holdout.json').write_text(json.dumps(receipt, indent=2) + '\n')
        print(json.dumps(receipt))
    else:
        full, selected = select_upstream(*(pd.read_parquet(getattr(args, name)) for name in ('plan', 'p14', 'p59', 'teeth')))
        full.to_parquet(args.output_dir / 'selection_full.parquet', index=False)
        selected.to_parquet(args.output_dir / 'upstream-selection.parquet', index=False)
        selected.to_csv(args.output_dir / 'upstream-selection.csv', index=False)
        actors = pd.read_csv(args.actors)
        counts = selected.groupby(['uuid', 'property']).size().unstack(fill_value=0)
        counts = counts.reindex(index=actors.uuid, columns=range(1, 10), fill_value=0)
        summary = {'actors': len(actors), 'images': len(selected), 'actors_with_images': int((counts.sum(axis=1) > 0).sum()),
                   'actors_without_images': counts.index[counts.sum(axis=1).eq(0)].tolist(),
                   'actors_without_eligible_tracks': actors.loc[actors.eligible_track_count.eq(0), 'uuid'].tolist(),
                   'actors_with_incomplete_source': sorted(set(full.loc[full.source_incomplete, 'uuid'])),
                   'incomplete_source_track_count': int(full.loc[full.source_incomplete, 'track'].nunique()),
                   'slots_per_actor': {str(key): int(value) for key, value in counts.gt(0).sum(axis=1).value_counts().sort_index().items()},
                   'slot_image_counts': {str(key): int(value) for key, value in counts.sum().items()}}
        (args.output_dir / 'upstream-coverage.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(json.dumps(summary))


if __name__ == '__main__':
    main()
