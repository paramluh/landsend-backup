"""Independent archived-selector comparisons; never change the running bank."""

import ast
import copy
import csv
import hashlib
import io
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd
import pyarrow.parquet as pq

import historical_studio_select
from historical_studio_select import select_historical_studio
from upstream import BEATLES_HELD_ROWS, REFERENCE_PREFIX, SELECTOR_FOLDER, SLOTS, select_upstream


ARCHIVED_SELECTOR = Path('/mnt/paramfs/vansh/richmond-7/reference-bank/code/studio-actors/select.py')
ARCHIVED_BEATLES = ARCHIVED_SELECTOR.parent.parent / 'beatles-eval/plan.py'


def fixtures(tracks=1):
    plans, faces, hands, teeth = [], [], [], []
    for index in range(tracks):
        track, session = f'track{index:02d}/speaker_1', f'session{index % 3}'
        teeth.append(dict(track=track, ok=True, up=2., lo=2., span=.8))
        for property_number, slot in enumerate(SLOTS, 1):
            source = f'{REFERENCE_PREFIX}/{track}/{SELECTOR_FOLDER}/{slot}.png'
            plans.append(dict(track=track, uuid='actor', bank='bank', cohort='train', property=property_number,
                              slot=slot, recording_id=f'track{index:02d}', speaker=1, session_id=session,
                              frame_index=1000, time_s=40., source=source,
                              source_sha256=hashlib.sha256(source.encode()).hexdigest(), has_image=True,
                              has_pose=True, crop_fit=True, cx0=0, cy0=0,
                              cw=256 if property_number <= 4 else 512,
                              ch=480 if property_number <= 4 else 960, scale=1. if property_number <= 4 else .5,
                              sel_score=.8, gate=True, iid=len(plans), candidate=True, in_eval=False,
                              in_bb=False, in_light=False, in_excess=True, excluded=False))
            common = dict(track=track, slot=slot, mp_ok=True, missing=False, blinkL=.1, blinkR=.1,
                          blur=100. + index, frame_index=1000)
            if property_number <= 4:
                faces.append(dict(**common, score=.5 + index / 100,
                                  hand_in_face=0, yaw=-25. if property_number == 2 else 25. if property_number == 3 else 0.,
                                  oof=0, bx0=50, by0=50, bx1=400, by1=800, jaw=.6 + index / 100))
            else:
                hands.append(dict(**common, n_hands=2, h0_inframe=21, h1_inframe=21, h0_sharp=100. + index,
                                  h1_sharp=100. + index, h0_ext=4, h1_ext=4, hh_dist=.3, h0_face_ov=0.,
                                  h1_face_ov=0., interaction_gate_passed=True,
                                  h0_target='hair' if property_number == 7 else 'clothing', h1_target='none',
                                  hc_min_mad=10., hc_min_dt=5.))
    return tuple(pd.DataFrame(rows) for rows in (plans, faces, hands, teeth))


def archived_selection(plan, p14, p59, teeth):
    """Execute archived filters and ranking directly, intercepting only file reads.

    Stop before its first write; no archived filesystem path is ever modified.
    This oracle does not call the new wrapper's archive-loading helper.
    """
    source = ARCHIVED_SELECTOR.read_text()
    tree = ast.parse(source, filename=str(ARCHIVED_SELECTOR))
    boundary = next(index for index, node in enumerate(tree.body)
                    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Attribute)
                    and node.value.func.attr == 'to_parquet')
    code = compile(ast.Module(body=tree.body[:boundary], type_ignores=[]), str(ARCHIVED_SELECTOR), 'exec')
    sources = {
        '/home/vansh/eval/ref_bank/plan.parquet': plan,
        '/home/vansh/eval/ref_audit/train.parquet': p14,
        '/home/vansh/eval/ref_bank/new_p14.parquet': p14.iloc[:0],
        '/home/vansh/eval/mouth_open_pilot/allteeth_refs.parquet': teeth,
        '/home/vansh/eval/ref_bank/new_teeth.parquet': teeth.iloc[:0],
        '/home/vansh/eval/ref_audit/train59.parquet': p59,
        '/home/vansh/eval/ref_bank/new_p59.parquet': p59.iloc[:0],
    }
    namespace = {}
    with patch.object(pd, 'read_parquet', side_effect=lambda path: sources[str(path)].copy(deep=True)):
        exec(code, namespace)
    return namespace['M']


def compare_archive(test, tables):
    full, selected = select_historical_studio(*tables)
    archived = archived_selection(*tables)
    columns = ['iid', 'track', 'property', 'pass', 'sel', 'index', 'rank']
    pd.testing.assert_frame_equal(full[columns].reset_index(drop=True),
                                  archived[columns].reset_index(drop=True), check_exact=True)
    old = archived[archived.sel].sort_values(['uuid', 'property', 'index'])
    test.assertEqual(selected.iid.tolist(), old.iid.tolist())
    plan = tables[0].set_index('iid')
    # The historical policy must not substitute another frame, crop or source.
    for row in selected.to_dict('records'):
        original = plan.loc[row['iid']]
        for key in ('source', 'source_sha256', 'frame_index', 'time_s', 'recording_id', 'speaker',
                    'session_id', 'uuid', 'cx0', 'cy0', 'cw', 'ch', 'scale'):
            test.assertEqual(row[key], original[key], key)
    return full, selected


def archived_beatles_selection(tables, beatles_rows, held_rows):
    """Execute original Beatles ranking with the established fixed holdout.

    Only holdout discovery is overridden: use the fixed rows and exclude both
    speakers of held recordings. The historical Python ranking runs unchanged.
    """
    full = archived_selection(*tables)
    tree = ast.parse(ARCHIVED_BEATLES.read_text(), filename=str(ARCHIVED_BEATLES))
    group_start = next(i for i, node in enumerate(tree.body) if isinstance(node, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == 'groups' for t in node.targets))
    first_write = next(i for i, node in enumerate(tree.body) if isinstance(node, ast.With))
    csv_text = io.StringIO(newline='')
    writer = csv.DictWriter(csv_text, fieldnames=list(beatles_rows[0]))
    writer.writeheader()
    writer.writerows(beatles_rows)
    namespace = {'__file__': str(ARCHIVED_BEATLES)}
    def read(path, *args, **kwargs):
        if str(path) != '/home/vansh/vansh-video-models/csvs/aloe-eval-beatles.csv':
            raise AssertionError(f'Unexpected archived file access: {path}')
        return io.StringIO(csv_text.getvalue())
    with patch('builtins.open', side_effect=read), patch.object(pq, 'read_table', return_value=SimpleNamespace(
            to_pylist=lambda: copy.deepcopy(full.to_dict('records')))):
        exec(compile(ast.Module(body=tree.body[:group_start], type_ignores=[]), str(ARCHIVED_BEATLES), 'exec'), namespace)
    held_recordings = {beatles_rows[index]['recording_id'] for index in held_rows}
    namespace['held'] = {row['track'] for row in full.to_dict('records') if row['recording_id'] in held_recordings}
    namespace['rollout'] = list(held_rows)
    # Extra corpus lighting exclusions are retained, as required by this rebuild.
    for row in namespace['M']:
        row['ok'] &= not row['in_light']
    exec(compile(ast.Module(body=tree.body[group_start:first_write], type_ignores=[]), str(ARCHIVED_BEATLES), 'exec'), namespace)
    return namespace['out']


class HistoricalStudioChecks(unittest.TestCase):
    def test_archived_thresholds_differ_from_strict_expansion(self):
        cases = [
            (1, {'blur': 40.}),
            (1, {'yaw': 14., 'blinkL': .4, 'blinkR': .4}),
            (2, {'yaw': -60.}),
            (3, {'hand_in_face': 2}),
            (4, {'yaw': 20., 'hand_in_face': 2, 'blur': 40.}),
            (5, {'h0_sharp': 60., 'h1_sharp': 60., 'blur': 40.}),
            (6, {'n_hands': 0, 'h0_inframe': 0, 'h1_inframe': 0, 'h0_sharp': 0.,
                 'h1_sharp': 0., 'h0_face_ov': 1., 'h1_face_ov': 1., 'blur': 40.}),
            (7, {'h0_sharp': 60., 'h1_sharp': 0.}),
            (8, {'h0_target': 'none', 'h1_target': 'none'}),
            (9, {'h0_target': 'none', 'h1_target': 'none'}),
        ]
        for prop, changes in cases:
            with self.subTest(property=prop, changes=changes):
                tables = fixtures()
                measurements = tables[1 if prop <= 4 else 2]
                for key, value in changes.items():
                    measurements.loc[measurements.slot.eq(SLOTS[prop - 1]), key] = value
                full, _ = compare_archive(self, tables)
                self.assertTrue(full.loc[full.property.eq(prop), 'pass'].item())
                strict, _ = select_upstream(*tables)
                self.assertFalse(strict.loc[strict.property.eq(prop), 'pass'].item())

    def test_archived_rejection_boundaries_are_preserved(self):
        for prop, changes in [(1, {'blur': 39.99}), (2, {'yaw': -60.01}), (3, {'hand_in_face': 3}),
                              (5, {'h0_sharp': 59.99}), (6, {'interaction_gate_passed': False}),
                              (6, {'blur': 39.99}), (7, {'h0_sharp': 59.99, 'h1_sharp': 0.})]:
            with self.subTest(property=prop, changes=changes):
                tables = fixtures()
                measurements = tables[1 if prop <= 4 else 2]
                for key, value in changes.items():
                    measurements.loc[measurements.slot.eq(SLOTS[prop - 1]), key] = value
                full, _ = compare_archive(self, tables)
                self.assertFalse(full.loc[full.property.eq(prop), 'pass'].item())

    def test_exclusions_crop_and_cap_session_round_robin(self):
        tables = fixtures(25)
        plan = tables[0]
        for prop, field, value in ((1, 'in_eval', True), (2, 'in_bb', True),
                                  (3, 'in_light', True), (4, 'crop_fit', False)):
            mask = plan.track.eq('track24/speaker_1') & plan.property.eq(prop)
            plan.loc[mask, field] = value
            plan.loc[mask, 'candidate'] = False
            if field != 'crop_fit':
                plan.loc[mask, 'excluded'] = True
        full, selected = compare_archive(self, tables)
        self.assertFalse(full.loc[full.track.eq('track24/speaker_1') & full.property.le(4), 'pass'].any())
        self.assertEqual(len(selected), 180)
        for _, group in selected.groupby('property'):
            self.assertEqual(group['index'].tolist(), list(range(20)))
            self.assertEqual(sorted(group.session_id.value_counts().tolist()), [6, 7, 7])

    def test_archive_tied_rank_order_is_preserved(self):
        tables = fixtures(25)
        tables[1][['blur', 'jaw', 'score']] = [100., .8, .8]
        tables[2][['h0_sharp', 'h1_sharp']] = [100., 100.]
        compare_archive(self, tables)

    def test_frame_and_source_identity_fail_closed(self):
        tables = fixtures()
        tables[1].loc[tables[1].slot.eq('front'), 'frame_index'] = 1001
        with self.assertRaisesRegex(ValueError, 'frame_index'):
            select_historical_studio(*tables)
        for key, bad in [('source', 'reference-images-512x960-v1/other/speaker_1/front.png'),
                         ('source_sha256', 'not-a-sha256'), ('track', 'other/speaker_1')]:
            with self.subTest(key=key):
                tables = fixtures()
                tables[0].loc[0, key] = bad
                with self.assertRaises(ValueError):
                    select_historical_studio(*tables)

    def test_nonfinite_required_quality_measurements_are_not_selected(self):
        for prop, key in ((1, 'blinkL'), (2, 'bx0'), (4, 'blinkR'), (6, 'blinkL')):
            with self.subTest(property=prop, key=key):
                tables = fixtures()
                measurements = tables[1 if prop <= 4 else 2]
                measurements[key] = measurements[key].astype(float)
                measurements.loc[measurements.slot.eq(SLOTS[prop - 1]), key] = float('nan')
                full, selected = select_historical_studio(*tables)
                self.assertFalse(full.loc[full.property.eq(prop), 'pass'].item())
                self.assertFalse(selected.property.eq(prop).any())

    def test_nonfinite_source_frame_time_and_crop_are_rejected(self):
        for key, value in [('frame_index', float('nan')), ('time_s', float('inf')),
                           ('cx0', float('nan')), ('cw', float('inf')), ('scale', float('nan'))]:
            with self.subTest(key=key):
                tables = fixtures()
                tables[0][key] = tables[0][key].astype(float)
                tables[0].loc[0, key] = value
                with self.assertRaises(ValueError):
                    select_historical_studio(*tables)

    def test_p6_missing_hand_measurements_preserve_gate_and_face_only_policy(self):
        tables = fixtures(3)
        hands = tables[2]
        mask = hands.track.eq('track00/speaker_1') & hands.slot.eq('hands_interacting')
        for key in ('n_hands', 'h0_inframe', 'h1_inframe', 'h0_sharp', 'h1_sharp',
                    'h0_face_ov', 'h1_face_ov'):
            hands[key] = hands[key].astype(float)
            hands.loc[mask, key] = float('nan')
        full, selected = compare_archive(self, tables)
        p6 = selected[selected.property.eq(6)]
        self.assertEqual(len(p6), 3)
        self.assertTrue(full.loc[full.track.eq('track00/speaker_1') & full.property.eq(6), 'pass'].item())
        self.assertTrue(pd.isna(p6.iloc[-1].rank_metric))
        self.assertEqual(p6.iloc[-1].recording_id, 'track00')

    def test_actual_studio_sample_matches_archived_execution(self):
        root = Path('/mnt/paramfs/param/outputs/reference-bank-rebuild-20261018/studio-only')
        if not (root / 'plan/plan.parquet').is_file():
            self.skipTest('Current verified studio plan unavailable')
        plan = pd.read_parquet(root / 'plan/plan.parquet')
        tracks = plan.loc[plan.candidate, 'track'].drop_duplicates().head(16).tolist()
        plan = plan[plan.track.isin(tracks)].copy()
        measured = [pd.read_parquet(root / 'measurements' / f'{name}.parquet', filters=[('track', 'in', tracks)])
                    for name in ('p14', 'p59', 'teeth')]
        _, selected = compare_archive(self, (plan, *measured))
        self.assertEqual(len(tracks), 16)
        self.assertGreater(len(selected), 0)

    def test_beatles_fixed_holdout_whole_recordings_and_original_ranking(self):
        tables = list(fixtures(40))
        # A second speaker in a held recording is not one of the held CSV rows,
        # but its entire recording must still be excluded as a reference donor.
        held_recording = f'track{BEATLES_HELD_ROWS[0]:02d}'
        old_track, other_track = f'{held_recording}/speaker_1', f'{held_recording}/speaker_2'
        extra = tables[0][tables[0].track.eq(old_track)].copy()
        extra['track'], extra['speaker'] = other_track, 2
        extra['iid'] = range(len(tables[0]), len(tables[0]) + len(extra))
        extra['source'] = [f'{REFERENCE_PREFIX}/{other_track}/{SELECTOR_FOLDER}/{slot}.png' for slot in extra.slot]
        extra['source_sha256'] = extra.source.map(lambda value: hashlib.sha256(value.encode()).hexdigest())
        tables[0] = pd.concat([tables[0], extra], ignore_index=True)
        for i in (1, 2, 3):
            more = tables[i][tables[i].track.eq(old_track)].copy()
            more['track'] = other_track
            tables[i] = pd.concat([tables[i], more], ignore_index=True)
        # Ties exercise the archived stable Python order, unlike pandas sorting.
        tables[1][['blur', 'jaw', 'score']] = [100., .8, .8]
        tables[2][['h0_sharp', 'h1_sharp']] = [100., 100.]
        plan = tables[0]
        plan['in_eval'], plan['excluded'], plan['candidate'] = True, True, False
        plan.loc[plan.track.eq('track00/speaker_1') & plan.property.eq(8), 'in_bb'] = True
        plan.loc[plan.track.eq('track00/speaker_1') & plan.property.eq(9), 'in_light'] = True
        beatles = plan.drop_duplicates('track')[['uuid', 'recording_id', 'speaker', 'session_id', 'track']].to_dict('records')
        full, selected, receipt = historical_studio_select.select_historical_beatles(*tables, beatles)
        expected = archived_beatles_selection(tables, beatles, BEATLES_HELD_ROWS)
        self.assertEqual(selected.iid.tolist(), [row['iid'] for row in expected])
        self.assertEqual(selected['index'].tolist(), [row['index'] for row in expected])
        self.assertEqual(set(selected.bank), {'beatles-eval'})
        held = {beatles[index]['recording_id'] for index in BEATLES_HELD_ROWS}
        self.assertFalse(set(selected.recording_id) & held)
        self.assertNotIn(other_track, set(selected.recording_id + '/speaker_' + selected.speaker.astype(str)))
        self.assertEqual(receipt['held_rows'], list(BEATLES_HELD_ROWS))
        self.assertEqual(set(receipt['held_recordings']), held)
        self.assertFalse(full.loc[full.sel, ['in_bb', 'in_light']].any().any())
        for _, group in selected.groupby('property'):
            self.assertEqual(group['index'].tolist(), list(range(20)))

    def test_actual_beatles_sample_matches_archived_ranking(self):
        root = Path('/mnt/paramfs/param/outputs/reference-bank-rebuild-20261018/studio-only')
        if not (root / 'plan/plan.parquet').is_file():
            self.skipTest('Current verified studio plan unavailable')
        beatles = pd.read_csv(root / 'csvs/aloe-eval-beatles.csv').to_dict('records')
        for row in beatles:
            row['track'] = f"{row['recording_id']}/speaker_{row['speaker']}"
        tracks = sorted({row['track'] for row in beatles})
        plan = pd.read_parquet(root / 'plan/plan.parquet', filters=[('track', 'in', tracks)])
        measurements = [pd.read_parquet(root / 'measurements' / f'{name}.parquet', filters=[('track', 'in', tracks)])
                        for name in ('p14', 'p59', 'teeth')]
        _, selected, receipt = historical_studio_select.select_historical_beatles(plan, *measurements, beatles)
        expected = archived_beatles_selection((plan, *measurements), beatles, BEATLES_HELD_ROWS)
        self.assertEqual(selected.iid.tolist(), [row['iid'] for row in expected])
        self.assertEqual(selected['index'].tolist(), [row['index'] for row in expected])
        self.assertEqual(len(beatles), 40)
        self.assertEqual(receipt['held_rows'], list(BEATLES_HELD_ROWS))


if __name__ == '__main__':
    unittest.main(verbosity=2)
