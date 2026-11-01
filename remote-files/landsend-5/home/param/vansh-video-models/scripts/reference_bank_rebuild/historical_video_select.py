"""Historical studio video quality, crop, dedup and ranking over verified scans.

The frozen scanner remains untouched. Its valid-pose, finite-keypoint and current
corpus exclusions still apply; these are explicit safeguards beyond the old scan.
"""
import argparse
import ast
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from historical_studio_select import POLICY_ID, sha256
from measurements import ARCHIVE, MODELS
from video_pipeline import (atomic_table, code_config, digest, input_fingerprint,
                            load_plan, plan_identity, read_table, task_path, write_json)
from video_scan import fit_selector_proxy, timestamp_allowed

ARCHIVED_SELECTOR = ARCHIVE / 'studio-actors/v2/select_v2.py'
ARCHIVED_SHA256 = 'dff8511c0d748da675c25e076a5050def6c3ec5d5b00032ab37766dcae9b2d61'


def assigned(node, name):
    return isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)


def archive_blocks():
    if sha256(ARCHIVED_SELECTOR) != ARCHIVED_SHA256:
        raise ValueError('Historical video selector differs from audited archive')
    tree = ast.parse(ARCHIVED_SELECTOR.read_bytes(), filename=str(ARCHIVED_SELECTOR))
    feature_functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in ('sfeat', 'design')]
    start = next(i for i, node in enumerate(tree.body) if assigned(node, 'K'))
    end = next(i for i, node in enumerate(tree.body) if assigned(node, 'rd'))
    score = ast.Module(body=feature_functions + tree.body[start:end], type_ignores=[])
    start = next(i for i, node in enumerate(tree.body) if assigned(node, 'v1n'))
    end = next(i for i, node in enumerate(tree.body) if assigned(node, 'N'))
    select = ast.Module(body=tree.body[start:end], type_ignores=[])
    return (compile(score, str(ARCHIVED_SELECTOR), 'exec'), compile(select, str(ARCHIVED_SELECTOR), 'exec'),
            {'score_ast_sha256': hashlib.sha256(ast.dump(score).encode()).hexdigest(),
             'select_ast_sha256': hashlib.sha256(ast.dump(select).encode()).hexdigest()})


def validate_candidates(frames, tracks_by_name=None):
    if frames.duplicated(['track', 'frame_index']).any():
        raise ValueError('Duplicate candidate track/frame pairs')
    if frames.groupby('uuid').bank.nunique().gt(1).any():
        raise ValueError('Historical actor merge requires one bank per actor')
    for row in frames.to_dict('records'):
        if row['track'] != f"{row['recording_id']}/speaker_{row['speaker']}":
            raise ValueError('Candidate recording/speaker identity differs from track')
        frame, seconds = row['frame_index'], row['time_s']
        if not np.isfinite(frame) or frame < 0 or int(frame) != frame or not np.isfinite(seconds):
            raise ValueError('Candidate frame/time must be finite and nonnegative')
        if abs(seconds - frame / 25.) > 1e-6:
            raise ValueError('Candidate frame_index differs from the 25-fps source clock')
        points = np.asarray(row['kp'], np.float32).reshape(133, 3)
        if not np.isfinite(points).all():
            raise ValueError('Candidate keypoints must be finite')
        if row.get('pose_valid') is not True:
            raise ValueError('Candidate pose_valid must be true under current scan safeguards')
        if any(row.get(name, False) for name in ('in_eval', 'in_bb', 'in_light')):
            raise ValueError('Candidate violates source holdouts or exclusions')
        if tracks_by_name is not None:
            if row['track'] not in tracks_by_name:
                raise ValueError('Candidate track is not selected by the current plan')
            track = tracks_by_name[row['track']]
            if any(row[key] != track[key] for key in ('uuid', 'recording_id', 'speaker', 'session_id')):
                raise ValueError('Candidate identity differs from selected source track')
            if row.get('bank', 'bank') != track.get('bank', 'bank') or row.get('cohort', '') != track.get('cohort', ''):
                raise ValueError('Candidate bank/cohort differs from selected source track')
            chunk = int(row['chunk'])
            if chunk not in track['usable_chunks'] or not 7500 * chunk <= frame < 7500 * (chunk + 1) or frame % 12:
                raise ValueError('Candidate frame is outside the eligible sampled chunk')
            expected = f"{str(track['source']).split('#frame=')[0]}#frame={int(frame)}"
            if row['source'] != expected or not expected.startswith('hf://'):
                raise ValueError('Candidate HF source/frame differs from target plan')
            if not timestamp_allowed(track, seconds):
                raise ValueError('Candidate timestamp violates current corpus exclusions')


def select_historical_scanned(candidates, upstream_selected, fit_score_path=None,
                              proxy_weights=None, tracks_by_name=None):
    frames = candidates.reset_index(drop=True).copy()
    if frames.empty:
        return pd.DataFrame()
    validate_candidates(frames, tracks_by_name)
    if proxy_weights is None:
        if fit_score_path is None:
            raise ValueError('fit_score_path or proxy_weights is required')
        proxy_weights = fit_selector_proxy(fit_score_path)
    if not np.isfinite(proxy_weights).all():
        raise ValueError('Selector proxy weights must be finite')
    score_code, select_code, _ = archive_blocks()
    namespace = {'np': np, 'pd': pd, 'F': frames, 'WS': proxy_weights, 'CAP': 20,
                 'GAP': 10., 'MAD': 6., 'W': 512, 'HH': 960, 'CW': 256, 'CH': 480,
                 'PAD': .35, 'GROW': 1.1}
    exec(score_code, namespace)
    passes, ranks = namespace['pass_'], namespace['rank']
    original_passing = {prop: int(passes[prop].sum()) for prop in range(1, 10)}
    # Keep old thresholds while rejecting missing operands that a negated
    # comparison could otherwise accept. Optional hand fields still fill zero.
    for prop in range(1, 10):
        needed = ['blinkL', 'blinkR', 'blur']
        if prop <= 4:
            needed += ['yaw', 'hand_in_face', 'oof']
            needed += ['jaw'] if prop == 4 else ['bx0', 'by0', 'bx1', 'by1']
        finite = pd.concat([np.isfinite(pd.to_numeric(frames.get(key, pd.Series(np.nan, index=frames.index)), errors='coerce'))
                            for key in needed], axis=1).all(axis=1)
        passes[prop] &= finite
    passing = np.logical_or.reduce([passes[prop].to_numpy() for prop in range(1, 10)])
    for index in frames.index[passing]:
        thumb = np.asarray(frames.loc[index, 'thumb'], np.float32).reshape(-1)
        if len(thumb) != 1920 or not np.isfinite(thumb).all():
            raise ValueError(f'Invalid thumbnail for passing candidate {index}')
    upstream = upstream_selected.copy()
    for (bank, actor, prop), group in upstream.groupby(['bank', 'uuid', 'property']):
        if len(group) > 20 or sorted(group['index']) != list(range(len(group))):
            raise ValueError('Upstream selection indices/cap are invalid')
    namespace['v1'] = upstream
    exec(select_code, namespace)
    result = []
    for addition in namespace['new']:
        index, prop = addition['iid_frame'], addition['property']
        row = frames.loc[index].to_dict()
        row.update(property=prop, slot=namespace.get('SLOTS', (
            'front', 'face_image_left', 'face_image_right', 'mouth_open', 'hands_visible',
            'hands_interacting', 'hair_clothing_01', 'hair_clothing_02', 'hair_clothing_03'))[prop - 1],
                   index=addition['index'], rank_metric=ranks[prop].iloc[index],
                   sel_score=frames.loc[index, f'score{prop}'] if prop in (2, 3) else np.nan,
                   origin='video-scan')
        row['cx0'], row['cy0'], row['cw'], row['ch'] = namespace['boxes'].get(index) if prop <= 4 else (0, 0, 512, 960)
        row['scale'] = 256 / row['cw'] if prop <= 4 else .5
        row.pop('kp', None)
        row.pop('thumb', None)
        result.append(row)
    selected = pd.DataFrame(result)
    selected.attrs['candidate_rows'] = len(frames)
    selected.attrs['archived_passing_frames_per_property'] = original_passing
    selected.attrs['finite_gate_rejections_per_property'] = {p: original_passing[p] - int(passes[p].sum()) for p in range(1, 10)}
    selected.attrs['passing_frames_per_property'] = {p: int(passes[p].sum()) for p in range(1, 10)}
    selected.attrs['dedup_kept_per_property'] = {p: sum(len(v) for (_, q), v in namespace['lists'].items() if q == p) for p in range(1, 10)}
    return selected


def select(args):
    tracks, target_report = load_plan(args)
    if digest(args.upstream) != target_report['upstream_sha256']:
        raise ValueError('Upstream selection changed after target planning')
    upstream = read_table(args.upstream)
    config = json.loads((args.output_dir / 'scan-code-inputs.json').read_text())
    if config != code_config(args.archive_root, args.model_root):
        raise ValueError('Scan code/model fingerprints changed before historical selection')
    scan_report = json.loads((args.output_dir / 'scan-summary.json').read_text())
    if scan_report.get('plan_sha256') != plan_identity(args.output_dir):
        raise ValueError('Scan plan changed or distributed checkpoints were not aggregated')
    grouped, lookup = defaultdict(list), {row['track']: row for row in tracks}
    for track in tracks:
        grouped[(track.get('bank', 'bank'), track['uuid'])].append(track)
    output, missing, checkpoint_hashes = [], [], {}
    passing, dedup, finite_rejections, archive_passing = (defaultdict(int) for _ in range(4))
    candidate_rows = 0
    for actor_number, ((bank, actor), actor_tracks) in enumerate(sorted(grouped.items()), 1):
        chunks = []
        for track in actor_tracks:
            for chunk in track['usable_chunks']:
                path = task_path(args.output_dir, track['track'], chunk)
                receipt_path = path.with_suffix('.complete.json')
                if not path.is_file() or not receipt_path.is_file():
                    missing.append({'track': track['track'], 'chunk': chunk})
                    continue
                meta = json.loads(receipt_path.read_text())
                pose_path = track['pose_paths'].get(str(chunk), track['pose_paths'].get(chunk))
                fingerprint = input_fingerprint(track, chunk, pose_path, config)
                if meta['fingerprint'] != fingerprint or meta['sha256'] != digest(path):
                    raise ValueError(f'Scan checkpoint source/code/checksum changed: {path}')
                if meta.get('track') != track['track'] or meta.get('chunk') != chunk:
                    raise ValueError(f'Scan checkpoint identity differs from selected target: {path}')
                rows = pd.read_parquet(path)
                if len(rows) != meta['rows']:
                    raise ValueError(f'Scan checkpoint row count differs: {path}')
                checkpoint_hashes[str(path)] = meta['sha256']
                if len(rows):
                    chunks.append(rows)
        if chunks:
            old = upstream[upstream.uuid.eq(actor) & upstream.bank.eq(bank)]
            rows = select_historical_scanned(pd.concat(chunks, ignore_index=True), old, args.fit_score, tracks_by_name=lookup)
            candidate_rows += rows.attrs.get('candidate_rows', 0)
            for prop, count in rows.attrs.get('finite_gate_rejections_per_property', {}).items():
                finite_rejections[prop] += count
            for prop, count in rows.attrs.get('archived_passing_frames_per_property', {}).items():
                archive_passing[prop] += count
            for prop, count in rows.attrs.get('passing_frames_per_property', {}).items():
                passing[prop] += count
            for prop, count in rows.attrs.get('dedup_kept_per_property', {}).items():
                dedup[prop] += count
            if len(rows):
                output.append(rows)
        if actor_number % 25 == 0 or actor_number == len(grouped):
            print(json.dumps({'historical_selected_actors_processed': actor_number, 'total_actors': len(grouped)}), flush=True)
    write_json(args.output_dir / 'selection-missing.json', missing)
    if missing and not args.allow_incomplete:
        raise ValueError(f'{len(missing)} scan chunks incomplete')
    selected = pd.concat(output, ignore_index=True) if output else pd.DataFrame(columns=['uuid', 'property', 'index', 'track', 'frame_index'])
    selected = selected.sort_values(['uuid', 'property', 'index'], kind='stable').reset_index(drop=True)
    atomic_table(selected, args.output_dir / 'video-selected.parquet')
    atomic_table(selected, args.output_dir / 'video-selected.csv')
    summary = {'rows': len(selected), 'actors': int(selected.uuid.nunique()), 'missing_chunks': len(missing),
               'complete': target_report['complete'] and not missing,
               'slot_counts': {str(k): int(v) for k, v in selected.groupby('property').size().items()},
               'fit_score_sha256': digest(args.fit_score), 'selection_policy': POLICY_ID,
               'candidate_rows': candidate_rows, 'archived_passing_frames_per_property': dict(archive_passing),
               'finite_gate_rejections_per_property': dict(finite_rejections),
               'passing_frames_per_property': dict(passing), 'dedup_kept_per_property': dict(dedup),
               'dedup_note': 'Archived 10s separation includes upstream refs; MAD<6 compares accepted scan peers; P7–9 share peer groups'}
    write_json(args.output_dir / 'selection-summary.json', summary)
    guard_path = args.output_dir.parent.parent / 'studio-only/historical-scan-candidate-guard-audit.json'
    guard_audit = None
    if guard_path.is_file():
        guard = json.loads(guard_path.read_text())
        if digest(guard['plan']) != guard['plan_sha256']:
            raise ValueError('Candidate guard audit source plan changed')
        audited_tracks = {row['track']: row for row in json.loads(Path(guard['plan']).read_text())}
        if any(audited_tracks.get(row['track']) != row for row in tracks):
            raise ValueError('Candidate guard audit does not cover the exact selected track inputs')
        guard_audit = {'path': str(guard_path), 'sha256': digest(guard_path), 'counts': guard['counts'],
                       'audited_tracks': guard['tracks'], 'audited_chunks': guard['chunks'],
                       'current_tracks_exactly_covered': True}
    receipt = {'candidate_guard_audit': guard_audit, 'policy_id': POLICY_ID, 'archived_selector': str(ARCHIVED_SELECTOR),
               'archived_selector_sha256': ARCHIVED_SHA256, 'wrapper_sha256': sha256(__file__),
               'executed_blocks': archive_blocks()[2], 'upstream_sha256': digest(args.upstream),
               'fit_score_sha256': digest(args.fit_score), 'scan_code_inputs': config,
               'scan_plan_sha256': plan_identity(args.output_dir), 'candidate_checksums': checkpoint_hashes,
               'quality_and_ranking': 'Exact pinned studio v2 filter/crop/rank/dedup/append blocks',
               'candidate_deviation': 'Current scanner excludes pose_valid=False and nonfinite keypoints before measurement; archived v2 recorded pose_valid but did not reject it. Current corpus exclusions and finite required measurement operands are also retained. This is historical quality/ranking over currently valid HF inputs, not byte-identical historical candidate or tensor reproduction.',
               'selected_sha256': digest(args.output_dir / 'video-selected.parquet'), 'summary': summary}
    write_json(args.output_dir / 'historical-video-policy.json', receipt)
    (args.output_dir / 'archived-studio-select-v2.py').write_bytes(ARCHIVED_SELECTOR.read_bytes())
    print(json.dumps(summary), flush=True)
    return 0 if summary['complete'] or args.allow_incomplete else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--allow-incomplete', action='store_true')
    parser.add_argument('--archive-root', type=Path, default=ARCHIVE)
    parser.add_argument('--model-root', type=Path, default=MODELS)
    parser.add_argument('--fit-score', type=Path, default=ARCHIVE / 'fit_score.parquet')
    return select(parser.parse_args())


if __name__ == '__main__':
    raise SystemExit(main())
