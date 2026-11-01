"""Replay the pinned historical studio selector on freshly verified studio inputs.

This is an explicit alternative policy. It never reads the old bank, changes the
frozen scanner, or chooses an output directory implicitly. The archived program
is executed only through selection/ranking, before its first filesystem write.
"""

import argparse
import ast
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from upstream import (BEATLES_HELD_ROWS, BLACKBANDS_CSV, REFERENCE_PREFIX, SELECTOR_FOLDER, SLOTS,
                      load_corpus, read_csv, select_upstream, timestamp_allowed)

POLICY_ID = 'historical-studio-v1-filters-and-ranking-v1'
ARCHIVED_SELECTOR = Path('/mnt/paramfs/vansh/richmond-7/reference-bank/code/studio-actors/select.py')
ARCHIVED_SHA256 = 'bbcbd57c1b5edebe7bc760f8452e12f782a7a6d88b9b8f66cf72c41e0a323be6'
ARCHIVED_BEATLES = ARCHIVED_SELECTOR.parent.parent / 'beatles-eval/plan.py'
BEATLES_SHA256 = '58d68913aaae742663e8fd24213197aff2ac17416716eefb90ddcea2ddeb2b17'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def archive_prefix():
    source = ARCHIVED_SELECTOR.read_bytes()
    if hashlib.sha256(source).hexdigest() != ARCHIVED_SHA256:
        raise ValueError('Historical studio selector differs from the audited archived source')
    tree = ast.parse(source, filename=str(ARCHIVED_SELECTOR))
    boundary = next(index for index, node in enumerate(tree.body)
                    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Attribute)
                    and node.value.func.attr == 'to_parquet')
    # Imports are supplied in the isolated namespace, so the pandas facade stays
    # local to this invocation and never monkey-patches process-global pandas.
    body = [node for node in tree.body[:boundary] if not isinstance(node, (ast.Import, ast.ImportFrom))]
    prefix = ast.Module(body=body, type_ignores=[])
    return compile(prefix, str(ARCHIVED_SELECTOR), 'exec'), hashlib.sha256(ast.dump(prefix).encode()).hexdigest()


class InputTables:
    def __init__(self, plan, p14, p59, teeth):
        self.tables = {
            '/home/vansh/eval/ref_bank/plan.parquet': plan,
            '/home/vansh/eval/ref_audit/train.parquet': p14,
            '/home/vansh/eval/ref_bank/new_p14.parquet': p14.iloc[:0],
            '/home/vansh/eval/ref_audit/train59.parquet': p59,
            '/home/vansh/eval/ref_bank/new_p59.parquet': p59.iloc[:0],
            '/home/vansh/eval/mouth_open_pilot/allteeth_refs.parquet': teeth,
            '/home/vansh/eval/ref_bank/new_teeth.parquet': teeth.iloc[:0],
        }

    def read_parquet(self, path):
        if str(path) not in self.tables:
            raise ValueError(f'Unapproved historical input read: {path}')
        return self.tables[str(path)].copy(deep=True)

    def __getattr__(self, name):
        return getattr(pd, name)


def validate_plan(plan):
    if plan.iid.duplicated().any():
        raise ValueError('Duplicate planned image identity')
    for row in plan.to_dict('records'):
        if row['track'] != f"{row['recording_id']}/speaker_{row['speaker']}":
            raise ValueError('Planned track differs from recording/speaker identity')
        prop = row['property']
        if prop not in range(1, 10) or row['slot'] != SLOTS[prop - 1]:
            raise ValueError('Planned property and slot disagree')
        expected = f"{REFERENCE_PREFIX}/{row['track']}/{SELECTOR_FOLDER}/{row['slot']}.png"
        if row['source'] != expected:
            raise ValueError('Planned source differs from canonical track/slot image key')
        if not row['candidate']:
            continue
        digest = row['source_sha256']
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('Candidate source_sha256 is missing or malformed')
        fields = ('frame_index', 'time_s', 'cx0', 'cy0', 'cw', 'ch', 'scale')
        if not all(np.isfinite(row[name]) for name in fields):
            raise ValueError('Candidate frame/time/crop fields must be finite')
        if row['frame_index'] < 0 or int(row['frame_index']) != row['frame_index']:
            raise ValueError('Candidate frame_index must be a nonnegative integer')
        if abs(row['time_s'] - row['frame_index'] / 25.) > 1e-6:
            raise ValueError('Candidate frame_index differs from 25-fps source clock')
        if any(row[name] for name in ('in_eval', 'in_bb', 'in_light', 'excluded')):
            raise ValueError('Candidate includes an excluded source')
        if not row['crop_fit'] or not row['has_image'] or not row['has_pose']:
            raise ValueError('Candidate lacks a verified image, pose or fitting crop')
        if row['cohort'] == 'eval-unseen' and not row['in_excess']:
            raise ValueError('Unseen candidate is outside the excess allowlist')
        if (row['cx0'] < 0 or row['cy0'] < 0 or row['cw'] <= 0 or row['ch'] <= 0
                or row['cx0'] + row['cw'] > 512 or row['cy0'] + row['ch'] > 960
                or abs(row['scale'] - 256 / row['cw']) > 1e-9):
            raise ValueError('Candidate crop is outside the archived geometry limits')
        if prop <= 4 and not (256 <= row['cw'] <= 282 and 480 <= row['ch'] <= 529):
            raise ValueError('Candidate face crop exceeds archived growth limits')
        if prop >= 5 and (row['cx0'], row['cy0'], row['cw'], row['ch'], row['scale']) != (0, 0, 512, 960, .5):
            raise ValueError('Candidate hand reference is not the full-frame downsample')


def finite_gate_inputs(plan, p14, p59):
    """Reject missing operands that archived negated comparisons could admit.

    A missing optional hand remains optional under historical P6. No stricter
    hand-count, region or sharpness requirement is added. Hair duplicate metrics
    may be +inf, their documented sentinel when no competing image exists.
    """
    allowed = {}
    for row in p14.to_dict('records'):
        slot = row['slot']
        keys = ['blinkL', 'blinkR', 'blur', 'hand_in_face', 'yaw', 'oof']
        if slot != 'mouth_open':
            keys += ['bx0', 'by0', 'bx1', 'by1']
        if slot in ('face_image_left', 'face_image_right'):
            keys += ['score']
        if slot == 'mouth_open':
            keys += ['jaw']
        allowed[row['track'], slot] = all(np.isfinite(row.get(key, np.nan)) for key in keys)
    for row in p59.to_dict('records'):
        keys = ['blinkL', 'blinkR', 'blur']
        # Historical hand operands deliberately fill missing hands with zero.
        allowed[row['track'], row['slot']] = all(np.isfinite(row.get(key, np.nan)) for key in keys)
    return pd.Series([allowed.get((row.track, row.slot), False) for row in plan.itertuples()], index=plan.index)


def select_historical_studio(plan, p14, p59, teeth):
    # Keep the current input contract, including exact measurement frame IDs and
    # unique joins. Its strict scientific outputs are deliberately discarded.
    _, schema = select_upstream(plan, p14, p59, teeth)
    plan = plan.reset_index(drop=True).copy()
    validate_plan(plan)
    finite = finite_gate_inputs(plan, p14, p59)
    plan['candidate'] &= finite
    code, _ = archive_prefix()
    namespace = {'np': np, 'pd': InputTables(plan, p14, p59, teeth)}
    exec(code, namespace)
    full = namespace['M']
    full['finite_gate_inputs'] = finite.to_numpy()
    columns = [name if name != 'rank_metric' else 'rank' for name in schema.columns if name != 'origin']
    selected = full[full.sel].sort_values(['uuid', 'property', 'index'])[columns].rename(columns={'rank': 'rank_metric'})
    selected['origin'] = 'ref-image'
    selected['frame_index'] = selected.frame_index.astype('int64')
    if full.loc[full.sel, ['in_eval', 'in_bb', 'in_light', 'excluded']].any().any():
        raise ValueError('Historical selection includes a prohibited image')
    return full, selected


def select_historical_beatles(plan, p14, p59, teeth, beatles_rows, held_rows=BEATLES_HELD_ROWS):
    if not held_rows or min(held_rows) < 0 or max(held_rows) >= len(beatles_rows):
        raise ValueError('Beatles held row indices are outside the provided CSV')
    source_tracks = {row['track'] for row in beatles_rows}
    held_recordings = {beatles_rows[index]['recording_id'] for index in held_rows}
    subset = plan[plan.track.isin(source_tracks)].copy()
    if set(subset.track) != source_tracks:
        raise ValueError('Historical upstream plan is missing a Beatles donor/held track')
    subset['in_eval'] = subset.recording_id.isin(held_recordings)
    subset['bank'], subset['cohort'] = 'beatles-eval', 'beatles-eval'
    subset['excluded'] = (~subset.has_image | ~subset.has_pose | subset.in_bb | subset.in_light | subset.in_eval)
    subset['candidate'] = ~subset.excluded & subset.crop_fit
    full, schema = select_historical_studio(subset, p14[p14.track.isin(source_tracks)],
                                           p59[p59.track.isin(source_tracks)], teeth[teeth.track.isin(source_tracks)])
    # The separate original Beatles script ranks with Python's stable sort.
    # Replay that exact block, with fixed existing holds rather than choosing
    # new rollout tracks from today's measurements.
    if sha256(ARCHIVED_BEATLES) != BEATLES_SHA256:
        raise ValueError('Archived Beatles selector differs from audited source')
    tree = ast.parse(ARCHIVED_BEATLES.read_bytes(), filename=str(ARCHIVED_BEATLES))
    start = next(i for i, node in enumerate(tree.body) if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == 'groups' for target in node.targets))
    end = next(i for i, node in enumerate(tree.body[start:], start) if isinstance(node, ast.With))
    prefix = ast.Module(body=tree.body[start:end], type_ignores=[])
    rows = full.to_dict('records')
    for row in rows:
        row['ok'] = row['pass']
    held = set(subset.loc[subset.recording_id.isin(held_recordings), 'track'])
    namespace = {'defaultdict': defaultdict, 'M': rows, 'held': held, 'CAP': 20, 'SLOTS': list(SLOTS)}
    exec(compile(prefix, str(ARCHIVED_BEATLES), 'exec'), namespace)
    order = {row['iid']: row['index'] for row in namespace['out']}
    full['sel'] = full.iid.isin(order)
    full['index'] = full.iid.map(order).fillna(-1).astype('int64')
    columns = [name if name != 'rank_metric' else 'rank' for name in schema.columns if name != 'origin']
    selected = full[full.sel].sort_values(['uuid', 'property', 'index'])[columns].rename(columns={'rank': 'rank_metric'})
    selected['origin'] = 'ref-image'
    selected['frame_index'] = selected.frame_index.astype('int64')
    if set(selected.recording_id) & held_recordings:
        raise ValueError('Historical Beatles references overlap held recordings')
    receipt = {'held_rows': list(held_rows), 'held_recordings': sorted(held_recordings),
               'source_track_count': len(source_tracks),
               'selected_donor_tracks': int(selected[['recording_id', 'speaker']].drop_duplicates().shape[0]),
               'selected_images': len(selected), 'actors': len({row['uuid'] for row in beatles_rows}),
               'policy': 'Historical studio filter flags and archived Beatles stable Python ranking; fixed historical 16 rollout rows, entire held recordings excluded; current blackband/lighting/crop/source checks',
               'policy_id': POLICY_ID, 'archived_beatles_selector_sha256': BEATLES_SHA256,
               'executed_ranking_ast_sha256': hashlib.sha256(ast.dump(prefix).encode()).hexdigest()}
    return full, selected, receipt


def validate_scope(plan, actors, csv_root, blackbands_csv):
    manifest_path = csv_root / 'studio-scope.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('scope') != 'studio-only' or manifest.get('complete') is not True:
        raise ValueError('An audited, complete studio-only scope is required')
    for name, evidence in manifest['files'].items():
        if sha256(csv_root / name) != evidence['derived_sha256']:
            raise ValueError(f'Studio scope CSV changed: {name}')
    corpus = load_corpus(csv_root, blackbands_csv)
    tracks = {row['track']: row for row in corpus.tracks}
    if set(plan.track) != set(tracks) or len(plan) != 9 * len(tracks):
        raise ValueError('Historical selector requires the complete scoped nine-slot plan')
    if actors.uuid.duplicated().any() or set(actors.uuid) != {row['uuid'] for row in corpus.actors}:
        raise ValueError('Actor inventory differs from the complete studio scope')
    for row in plan.to_dict('records'):
        record = tracks[row['track']]
        if any(row[key] != record[key] for key in ('uuid', 'session_id', 'recording_id', 'speaker', 'bank', 'cohort')):
            raise ValueError('Plan identity differs from the studio-only corpus')
        if row['candidate'] and not timestamp_allowed(corpus, row['track'], row['time_s'], row['cohort']):
            raise ValueError('Candidate violates current corpus holdouts or exclusions')
    return corpus, manifest_path


def coverage(full, selected, actors):
    counts = selected.groupby(['uuid', 'property']).size().unstack(fill_value=0)
    counts = counts.reindex(index=actors.uuid, columns=range(1, 10), fill_value=0)
    return {'actors': len(actors), 'images': len(selected), 'actors_with_images': int(counts.sum(axis=1).gt(0).sum()),
            'actors_without_images': counts.index[counts.sum(axis=1).eq(0)].tolist(),
            'actors_without_eligible_tracks': actors.loc[actors.eligible_track_count.eq(0), 'uuid'].tolist(),
            'actors_with_incomplete_source': sorted(set(full.loc[full.source_incomplete, 'uuid'])),
            'incomplete_source_track_count': int(full.loc[full.source_incomplete, 'track'].nunique()),
            'slots_per_actor': {str(key): int(value) for key, value in counts.gt(0).sum(axis=1).value_counts().sort_index().items()},
            'slot_image_counts': {str(key): int(value) for key, value in counts.sum().items()}, 'selection_policy': POLICY_ID}


POLICY_DOCUMENT = """# Historical studio reference-bank selection

This bank applies policy `historical-studio-v1-filters-and-ranking-v1` to newly
verified Hugging Face source images and newly computed measurements. Its quality
thresholds intentionally follow the studio selection code used before the later
strict app expansion. Where the archived global RECIPE.md or sinks.md disagree,
this document and the pinned `archived-studio-select.py` govern studio selection.
This is a fresh rebuild of that policy, not a claim of byte-identical old tensors.

P1–4 require face blur >=40 and fewer than three hand landmarks on the face.
Front yaw is strictly within 15 degrees; turns span 20 through 60 degrees;
mouth yaw is at most 20 degrees. The original per-property eye, face-boundary,
turn-score and teeth gates apply. P5 hand sharpness is >=60 for both hands,
with the original complete-hand, finger-extension, separation and face checks.
P6 requires the upstream strict contact gate and a valid sharp face; it does not
add later hand-count, in-frame or hand-sharpness gates. P7–9 require best-hand
sharpness >=60 with their original hair/face-overlap/duplicate checks; P8–9 do
not add the later clothing-region predicate. Face sharpness for P5–9 is >=40.

The original studio rank and session round-robin cap of 20 per actor/property
are replayed exactly. Missing operands that could make negated comparisons pass
are rejected. Optional hand measurements remain optional where the historical
policy does not require them. Input hash, frame, clock, geometry and identity
checks fail closed. Current studio-only actor and track membership, blackband
and lighting exclusions, evaluation margins, and unseen excess allowlists apply.
No app source is eligible. Source gaps and empty slots remain explicit.

Video additions use the pinned original studio v2 quality, face-crop, per-video
10-second/MAD deduplication and rank-first track merge, capped at 20 total
upstream-plus-video references per slot. Raw measurements retain the original
every-12th-frame sampling at 25 fps, 512x960 RGB geometry, pose scaling and
contact/hand/teeth measurement prefilters. The current scanner additionally
rejects pose_valid=False and nonfinite keypoints before measurement; the old
scanner recorded pose_valid but did not reject it. Those current safety guards
are intentionally retained. Therefore this rebuild reproduces historical quality
and ranking over currently valid HF inputs, not the unrestricted historical
candidate set. The 2026-10-18 source-guard audit covered all 410 selected studio
tracks and 929 chunks: 18 of 580,430 eligible sampled frames had invalid poses;
zero had nonfinite keypoints. Those 18 frames remain excluded. The audit is
hash-bound by the selection receipts. See historical-video-policy.json and
archived-studio-select-v2.py. Coverage is reported per actor; nine populated
slots are not guaranteed for every actor.

The separate Beatles evaluation bank uses these same historical studio flags
and the original Beatles stable Python ranking. Its existing 16 rollout rows
remain fixed, and their entire recordings are forbidden as references. Donors
come only from the other Beatles recordings. Holds are not recomputed from the
new measurements. See `beatles-holdout.json` and `archived-beatles-plan.py`.

The image/latent contract is unchanged: P1–4 use the planned face crop;
P5–9 downsample the full 512x960 image to 256x480. Muon first-frame latents are
bf16 [N,48,30,16] in each actor/property `latents.safetensors`. This selector
only publishes image selections; encoding and final source/tensor validation
remain required. `selection-policy.json` pins code, inputs, scope and outputs.
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'p14', 'p59', 'teeth', 'actors', 'csv-root', 'output-dir'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--blackbands-csv', type=Path, default=BLACKBANDS_CSV)
    args = parser.parse_args()
    tables = [pd.read_parquet(getattr(args, name)) for name in ('plan', 'p14', 'p59', 'teeth')]
    actors = pd.read_csv(args.actors)
    corpus, scope_manifest = validate_scope(tables[0], actors, args.csv_root, args.blackbands_csv)
    full, selected = select_historical_studio(*tables)
    strict_full, strict = select_upstream(*tables)
    beatles_rows = read_csv(args.csv_root / 'aloe-eval-beatles.csv', ('uuid', 'recording_id', 'speaker'))
    beatles_full, beatles_selected, beatles_receipt = select_historical_beatles(*tables, beatles_rows)
    summary = coverage(full, selected, actors)
    strict_summary = coverage(strict_full, strict, actors)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    full.to_parquet(args.output_dir / 'selection_full.parquet', index=False)
    selected.to_parquet(args.output_dir / 'upstream-selection.parquet', index=False)
    selected.to_csv(args.output_dir / 'upstream-selection.csv', index=False)
    (args.output_dir / 'upstream-coverage.json').write_text(json.dumps(summary, indent=2) + '\n')
    beatles_full.to_parquet(args.output_dir / 'beatles-selection-full.parquet', index=False)
    beatles_selected.to_parquet(args.output_dir / 'beatles-selection.parquet', index=False)
    beatles_selected.to_csv(args.output_dir / 'beatles-selection.csv', index=False)
    (args.output_dir / 'beatles-holdout.json').write_text(json.dumps(beatles_receipt, indent=2) + '\n')
    guard_path = args.csv_root.parent / 'historical-scan-candidate-guard-audit.json'
    guard_audit = ({'path': str(guard_path), 'sha256': sha256(guard_path),
                    'counts': json.loads(guard_path.read_text())['counts']} if guard_path.is_file() else None)
    receipt = {'candidate_guard_audit': guard_audit, 'policy_id': POLICY_ID, 'status': 'applied selection policy; encoding and final validation required',
               'archived_selector': str(ARCHIVED_SELECTOR), 'archived_selector_sha256': ARCHIVED_SHA256,
               'executed_prefix_ast_sha256': archive_prefix()[1], 'wrapper_sha256': sha256(__file__),
               'inputs': {name: {'path': str(getattr(args, name)), 'sha256': sha256(getattr(args, name))}
                          for name in ('plan', 'p14', 'p59', 'teeth', 'actors')},
               'scope_manifest': {'path': str(scope_manifest), 'sha256': sha256(scope_manifest)},
               'corpus_source_hashes': corpus.source_hashes,
               'historical_quality_and_ranking': 'Pinned original studio select.py, through its first output write boundary',
               'current_safety_checks': 'Verified source identity/hash, exact measurement frame, finite required operands, current studio-only corpus exclusions, existing crop geometry',
               'finite_operand_rejections': int((tables[0].candidate & ~full.finite_gate_inputs).sum()),
               'strict_comparison': {'images': len(strict), 'slots_per_actor': strict_summary['slots_per_actor']},
               'historical_result': summary,
               'beatles_policy': beatles_receipt,
               'outputs': {name: sha256(args.output_dir / name) for name in ('selection_full.parquet', 'upstream-selection.parquet', 'upstream-selection.csv', 'upstream-coverage.json', 'beatles-selection-full.parquet', 'beatles-selection.parquet', 'beatles-selection.csv', 'beatles-holdout.json')}}
    (args.output_dir / 'selection-policy.json').write_text(json.dumps(receipt, indent=2) + '\n')
    (args.output_dir / 'archived-studio-select.py').write_bytes(ARCHIVED_SELECTOR.read_bytes())
    (args.output_dir / 'archived-beatles-plan.py').write_bytes(ARCHIVED_BEATLES.read_bytes())
    (args.output_dir / 'RECIPE-historical-studio.md').write_text(POLICY_DOCUMENT)
    print(json.dumps({'upstream': summary, 'beatles': beatles_receipt}))


if __name__ == '__main__':
    main()
