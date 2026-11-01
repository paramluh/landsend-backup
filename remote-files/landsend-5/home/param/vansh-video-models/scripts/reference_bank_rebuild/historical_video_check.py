"""Independent archived studio-v2 video selector oracle, without scan/decode jobs."""

import ast
import hashlib
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from historical_video_select import select_historical_scanned
from video_scan import fit_selector_proxy, score_candidates


ARCHIVE = Path('/mnt/paramfs/vansh/richmond-7/reference-bank/code/studio-actors/v2/select_v2.py')
WEIGHTS = np.r_[np.zeros(12), .5]
UPSTREAM_COLUMNS = ['bank', 'uuid', 'recording_id', 'speaker', 'property', 'index', 'time_s']


def candidate(seconds=0., track='r/speaker_1', **changes):
    points = np.zeros((133, 3), np.float32)
    points[23:91, 0] = np.linspace(200, 300, 68)
    points[23:91, 1] = np.linspace(200, 320, 68)
    points[23:91, 2] = 1
    frame = round(seconds * 25)
    recording, speaker = track.split('/speaker_')
    row = dict(uuid='actor', bank='bank', cohort='train', track=track, recording_id=recording,
               speaker=int(speaker), session_id=f'session-{recording}', frame_index=frame, time_s=frame / 25,
               chunk=frame // 7500, source=f'hf://buckets/test/reference-videos/{track}/full.mp4#frame={frame}',
               candidate_id=f'{track}#frame={frame}', kp=points.reshape(-1).tolist(),
               thumb=[(frame / 25 * 2) % 256] * 1920, pose_valid=True, mp_ok=True,
               yaw=0., blinkL=.1, blinkR=.1, blur=100., hand_in_face=0., oof=0.,
               bx0=200., by0=200., bx1=300., by1=320., jaw=.8,
               t_ok=False, up=0., lo=0., span=0., n_hands=0., h0_sharp=0., h1_sharp=0.,
               h0_inframe=0., h1_inframe=0., h0_ext=0., h1_ext=0., hh_dist=0.,
               h0_face_ov=0., h1_face_ov=0., h0_target='none', h1_target='none', gate=False,
               in_eval=False, in_bb=False, in_light=False, in_excess=True)
    row.update(changes)
    return row


def lookup(rows):
    result = {}
    for row in rows:
        item = {key: row[key] for key in ('track', 'uuid', 'recording_id', 'speaker', 'session_id', 'bank', 'cohort')}
        item.update(source=row['source'].split('#frame=')[0], usable_chunks=[0, 1, 2], excluded_ranges=[])
        result[row['track']] = item
    return result


def oracle(candidates, upstream, weights):
    """Compile actual archived functions/statements, without wrapper helpers or I/O."""
    tree = ast.parse(ARCHIVE.read_text(), filename=str(ARCHIVE))
    def assignment(name):
        return next(i for i, node in enumerate(tree.body) if isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == name for target in node.targets))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in ('sfeat', 'design')]
    namespace = dict(np=np, pd=pd, F=candidates.reset_index(drop=True).copy(), v1=upstream.copy(), WS=weights,
                     CAP=20, GAP=10., MAD=6., W=512, HH=960, CW=256, CH=480, PAD=.35, GROW=1.1)
    blocks = [functions + tree.body[assignment('K'):assignment('rd')],
              tree.body[assignment('v1n'):assignment('N')]]
    for body in blocks:
        exec(compile(ast.Module(body=body, type_ignores=[]), str(ARCHIVE), 'exec'), namespace)
    return namespace


def compare(test, rows, upstream=None, weights=WEIGHTS, tracks=None):
    candidates = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    if upstream is None:
        upstream = pd.DataFrame(columns=UPSTREAM_COLUMNS)
    if tracks is None:
        tracks = lookup(candidates.to_dict('records'))
    expected = oracle(candidates, upstream, weights)
    actual = select_historical_scanned(candidates, upstream, proxy_weights=weights, tracks_by_name=tracks)
    identities = []
    for row in expected['new']:
        frame = expected['F'].iloc[row['iid_frame']]
        identities.append((frame.track, frame.frame_index, row['property'], row['index']))
    test.assertEqual(list(actual[['track', 'frame_index', 'property', 'index']].itertuples(index=False, name=None))
                     if len(actual) else [], identities)
    test.assertEqual(actual.attrs['passing_frames_per_property'],
                     {p: int(expected['pass_'][p].sum()) for p in range(1, 10)})
    test.assertEqual(actual.attrs['dedup_kept_per_property'],
                     {p: sum(len(v) for (_, prop), v in expected['lists'].items() if prop == p) for p in range(1, 10)})
    for row, old in zip(actual.to_dict('records'), expected['new']):
        prop, index = old['property'], old['iid_frame']
        box = expected['boxes'][index] if prop <= 4 else (0, 0, 512, 960)
        test.assertEqual(tuple(row[key] for key in ('cx0', 'cy0', 'cw', 'ch')), box)
        test.assertEqual(row['scale'], 256 / box[2] if prop <= 4 else .5)
        test.assertEqual(row['rank_metric'], expected['rank'][prop].iloc[index])
        source = expected['F'].iloc[index]
        for key in ('source', 'time_s', 'recording_id', 'speaker', 'session_id', 'uuid', 'frame_index'):
            test.assertEqual(row[key], source[key], key)
    return actual


class HistoricalVideoChecks(unittest.TestCase):
    def test_historical_thresholds_against_archived_filters(self):
        hands = dict(yaw=70., n_hands=2., h0_inframe=21., h1_inframe=21., h0_ext=3., h1_ext=3.,
                     h0_sharp=60., h1_sharp=60., hh_dist=.2)
        cases = [(1, dict(blur=40., hand_in_face=2., yaw=14., blinkL=.4, blinkR=.4)),
                 (2, dict(yaw=-60.)), (3, dict(yaw=60.)),
                 (4, dict(yaw=20., t_ok=True, up=1., lo=1., span=.4, hand_in_face=2.)),
                 (5, hands), (6, dict(yaw=70., gate=True, n_hands=0., blur=40.)),
                 (6, dict(yaw=70., gate=True, n_hands=float('nan'), h0_sharp=float('nan'),
                          h1_sharp=float('nan'), blur=40.)),
                 (7, dict(yaw=70., h0_target='hair', h0_sharp=60.)),
                 (8, dict(yaw=70., h0_sharp=100., h0_target='none')),
                 (9, dict(yaw=70., h0_sharp=100., h0_target='none'))]
        for prop, values in cases:
            with self.subTest(property=prop):
                rows = [candidate(**values)]
                selected = compare(self, rows)
                self.assertEqual(selected.attrs['passing_frames_per_property'][prop], 1)
                strict = score_candidates(pd.DataFrame(rows), proxy_weights=WEIGHTS)
                self.assertFalse(strict.loc[0, f'pass{prop}'])

    def test_ten_second_boundary_and_thumbnail_mad_with_upstream(self):
        old = pd.DataFrame([dict(bank='bank', uuid='actor', recording_id='r', speaker=1,
                                 property=1, index=0, time_s=.08)])
        rows = [candidate(9.6, blur=140.), candidate(10.08, blur=130.),
                candidate(20.16, blur=120., thumb=[20.16] * 1920), candidate(30.24, blur=110.)]
        selected = compare(self, rows, old)
        self.assertEqual(selected.time_s.tolist(), [10.08, 30.24])
        self.assertEqual(selected['index'].tolist(), [1, 2])

    def test_hair_clothing_siblings_share_dedup_and_a_frame_is_used_once(self):
        values = dict(yaw=70., h0_sharp=100., h0_target='none')
        rows = [candidate(0., **dict(values, h0_target='hair', thumb=[0.] * 1920)),
                candidate(5.76, **dict(values, thumb=[50.] * 1920)),
                candidate(20.16, **dict(values, thumb=[0.] * 1920)),
                candidate(30.24, **dict(values, thumb=[90.] * 1920))]
        selected = compare(self, rows)
        self.assertEqual(list(zip(selected.property, selected.time_s)), [(7, 0.), (8, 30.24)])
        self.assertFalse(selected.duplicated(['track', 'frame_index']).any())

    def test_rank_first_track_round_robin_appends_without_exceeding_twenty(self):
        old = pd.DataFrame([dict(bank='bank', uuid='actor', recording_id='old', speaker=1,
                                 property=1, index=i, time_s=i * 20.) for i in range(18)])
        rows = [candidate(0., track='a/speaker_1', blur=200.), candidate(20.16, track='a/speaker_1', blur=190.),
                candidate(0., track='b/speaker_1', blur=150.), candidate(20.16, track='b/speaker_1', blur=140.)]
        selected = compare(self, rows, old)
        self.assertEqual(selected.track.tolist(), ['a/speaker_1', 'b/speaker_1'])
        self.assertEqual(selected['index'].tolist(), [18, 19])
        rows[2]['blur'] = 200.
        selected = compare(self, rows, old)
        self.assertEqual(selected.track.tolist(), ['b/speaker_1', 'a/speaker_1'])

    def test_native_and_grown_crops_match_archive_and_oversized_face_fails(self):
        rows = []
        for number, width in enumerate((100., 280., 400.)):
            row = candidate(number * 10.08, blur=100. - number)
            points = np.asarray(row['kp'], np.float32).reshape(133, 3)
            points[23:91, 0] = np.linspace(50., 50. + width, 68)
            row['kp'] = points.reshape(-1).tolist()
            rows.append(row)
        selected = compare(self, rows)
        self.assertEqual(selected.property.tolist(), [1, 1])
        self.assertEqual(selected.cw.tolist(), [256, 280])
        self.assertEqual(selected.ch.tolist(), [480, 525])
        self.assertEqual(selected.attrs['passing_frames_per_property'][1], 2)

    def test_current_source_pose_and_holdout_guards_fail_closed(self):
        for changes in ({'pose_valid': False}, {'in_eval': True}, {'in_bb': True}, {'in_light': True},
                        {'source': 'hf://buckets/test/wrong.mp4#frame=0'}, {'frame_index': 1, 'time_s': .04}):
            with self.subTest(changes=changes):
                base = candidate()
                with self.assertRaises(ValueError):
                    select_historical_scanned(pd.DataFrame([dict(base, **changes)]),
                                              pd.DataFrame(columns=UPSTREAM_COLUMNS), proxy_weights=WEIGHTS,
                                              tracks_by_name=lookup([base]))
        base = candidate()
        tracks = lookup([base])
        tracks[base['track']]['excluded_ranges'] = [(0., 5.)]
        with self.assertRaises(ValueError):
            select_historical_scanned(pd.DataFrame([base]), pd.DataFrame(columns=UPSTREAM_COLUMNS),
                                      proxy_weights=WEIGHTS, tracks_by_name=tracks)

    def test_actual_completed_candidate_subset_matches_archive(self):
        root = Path('/mnt/paramfs/param/outputs/reference-bank-rebuild-20261018/studio-only')
        paths = sorted((root / 'video-scan/candidates').glob('*/*/*.parquet'))
        if not paths:
            self.skipTest('No completed current studio candidate chunks available')
        tracks = {row['track']: row for row in json.loads((root / 'video-scan/tracks.json').read_text())}
        quality = ['blinkL', 'blinkR', 'blur', 'yaw', 'hand_in_face', 'oof', 'bx0', 'by0', 'bx1', 'by1', 'jaw']
        selected_rows = None
        for path in paths[:8]:
            receipt_path = path.with_suffix('.complete.json')
            if not receipt_path.is_file():
                continue
            receipt = json.loads(receipt_path.read_text())
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), receipt['sha256'])
            rows = pd.read_parquet(path)
            if not set(quality) <= set(rows.columns):
                continue
            finite = np.isfinite(rows[quality].astype(float)).all(axis=1)
            sample = rows[finite].head(200).copy()
            if len(sample):
                selected_rows = sample
                break
        self.assertIsNotNone(selected_rows, 'No finite measured candidate subset found')
        weights = fit_selector_proxy(ARCHIVE.parent.parent.parent / 'fit_score.parquet')
        additions = compare(self, selected_rows, weights=weights, tracks=tracks)
        self.assertGreater(len(additions), 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
