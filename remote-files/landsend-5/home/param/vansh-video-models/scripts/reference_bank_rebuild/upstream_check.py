"""Small synthetic checks for held-out exclusion, strict slot gates and session diversity."""

import json
import hashlib
import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from upstream import Corpus, REFERENCE_PREFIX, SELECTOR_FOLDER, SLOTS, build_plan, crop, select_beatles, select_upstream


class UpstreamChecks(unittest.TestCase):
    def test_crop_contains_face_without_upsampling(self):
        window = crop((100, 100, 380, 450))
        self.assertEqual(window[2:], (280, 525))
        self.assertLessEqual(window[0], 100)
        self.assertGreaterEqual(window[0] + window[2], 380)
        self.assertIsNone(crop((100, 100, 390, 450)))
        self.assertEqual(crop((0, 0, 100, 100)), (0, 0, 256, 480))
        self.assertEqual(crop((100, 100, 381.5, 450))[2:], (282, 529))

    def test_plan_excludes_reserved_blackband_lighting_and_unseen_outside_allowlist(self):
        tracks = [dict(track=f'rec{i}/speaker_1', recording_id=f'rec{i}', speaker=1, uuid=f'actor{i}',
                       session_id=f'session{i}', cohort=cohort, bank='bank')
                  for i, cohort in enumerate(('train', 'eval-unseen'))]
        corpus = Corpus([], tracks, [], {'rec0/speaker_1': [(95, 105)]},
                        {'rec0/speaker_1': [(480, 520)]}, {'rec0/speaker_1': [(690, 710)]},
                        {'rec1/speaker_1': [(400, 600)]}, {})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            keypoints = np.zeros((133, 3), np.float32)
            keypoints[23:91] = [220, 300, 0.9]
            keypoints[23] = [180, 200, 0.9]
            keypoints[24] = [280, 350, 0.9]
            payload = io.BytesIO()
            Image.new('RGB', (512, 960), (40, 80, 120)).save(payload, format='PNG')
            png = payload.getvalue()
            digest = hashlib.sha256(png).hexdigest()
            for track in tracks:
                path = root / REFERENCE_PREFIX / track['track'] / SELECTOR_FOLDER
                path.mkdir(parents=True)
                images = []
                for slot, time_s in zip(SLOTS, (100, 500, 700, 1100, 500, 500, 500, 500, 500)):
                    images.append(dict(category=slot, frame_index=time_s * 25, time_s=time_s,
                                       score=0.8, interaction_gate_passed=True, sha256=digest))
                    (path / f'{slot}.png').write_bytes(png)
                (path / 'manifest.json').write_text(json.dumps({'images': images}))
                pd.DataFrame([{'category': slot, 'image_keypoints': keypoints.tolist(),
                               'image_sha256': digest}
                              for slot in SLOTS]).to_parquet(path / 'selected_poses.parquet')
            plan, diagnostics = build_plan(corpus, root)
            self.assertEqual(diagnostics['fatal_input_issues'], 0)
            self.assertEqual(len(plan.iloc[0]['thumb']), 1920)
            train = plan[plan.uuid.eq('actor0')].set_index('property')
            unseen = plan[plan.uuid.eq('actor1')].set_index('property')
            self.assertFalse(train.loc[1, 'candidate'])
            self.assertTrue(train.loc[1, 'in_eval'])
            self.assertFalse(train.loc[2, 'candidate'])
            self.assertTrue(train.loc[2, 'in_bb'])
            self.assertFalse(train.loc[3, 'candidate'])
            self.assertTrue(train.loc[3, 'in_light'])
            self.assertTrue(train.loc[4, 'candidate'])
            self.assertFalse(unseen.loc[1, 'candidate'])
            self.assertTrue(unseen.loc[2, 'candidate'])
            self.assertEqual(unseen.loc[2, 'frame_index'], 12500)
            missing = f'{REFERENCE_PREFIX}/rec0/speaker_1/{SELECTOR_FOLDER}/front.png'
            (root / missing).unlink()
            _, diagnostic = build_plan(corpus, root)
            self.assertEqual(diagnostic['fatal_input_issues'], 1)
            _, diagnostic = build_plan(corpus, root, {missing})
            self.assertEqual(diagnostic['fatal_input_issues'], 0)
            self.assertEqual(diagnostic['issue_counts'], {'confirmed_hf_missing': 1})
            (root / missing).write_bytes(b'corrupt-png')
            _, diagnostic = build_plan(corpus, root)
            self.assertEqual(diagnostic['issue_counts'], {'image_checksum_mismatch': 2})

    def test_strict_gates_and_session_round_robin(self):
        plans, faces, hands, teeth = [], [], [], []
        for index in range(25):
            track = f'track{index:02d}/speaker_1'
            session = f'session{index % 3}'
            teeth.append(dict(track=track, ok=True, up=2.0, lo=2.0, span=0.8))
            for property_number, slot in enumerate(SLOTS, 1):
                frame_index = 1000 + property_number * 125
                plans.append(dict(track=track, uuid='actor', bank='bank', cohort='train', property=property_number,
                                  slot=slot, recording_id=f'track{index:02d}', speaker=1, session_id=session,
                                  frame_index=frame_index, time_s=frame_index / 25, source=f'{track}/{slot}.png',
                                  thumb=np.random.default_rng(len(plans)).integers(0, 256, 1920).tolist(),
                                  source_sha256='fixture', has_image=True, has_pose=True, crop_fit=True,
                                  cx0=0, cy0=0, cw=256, ch=480, scale=1.0, sel_score=0.8, gate=True,
                                  iid=len(plans), candidate=True, in_eval=False, in_bb=False, in_light=False,
                                  excluded=False))
                common = dict(track=track, slot=slot, mp_ok=True, missing=False, blinkL=0.1, blinkR=0.1,
                              blur=100 + index, frame_index=frame_index)
                if property_number <= 4:
                    faces.append(dict(**common, score=0.8, hand_in_face=0, yaw=-25 if property_number == 2 else 25 if property_number == 3 else 0,
                                      oof=0, bx0=50, by0=50, bx1=400, by1=800, jaw=0.9))
                else:
                    hands.append(dict(**common, n_hands=2, h0_inframe=21, h1_inframe=21, h0_sharp=100 + index,
                                      h1_sharp=100 + index, h0_ext=4, h1_ext=4, hh_dist=0.3,
                                      h0_face_ov=0, h1_face_ov=0, interaction_gate_passed=True,
                                      h0_target='hair' if property_number == 7 else 'clothing', h1_target='none',
                                      hc_min_mad=10, hc_min_dt=5))
        p14, p59 = pd.DataFrame(faces), pd.DataFrame(hands)
        bad = p14.track.eq('track24/speaker_1')
        p14.loc[bad & p14.slot.eq('front'), 'yaw'] = 12
        p14.loc[bad & p14.slot.eq('face_image_left'), 'yaw'] = -55
        p14.loc[bad & p14.slot.eq('face_image_right'), 'hand_in_face'] = 1
        p14.loc[bad & p14.slot.eq('mouth_open'), 'yaw'] = 18
        bad = p59.track.eq('track24/speaker_1')
        p59.loc[bad & p59.slot.eq('hands_visible'), 'h1_sharp'] = 70
        p59.loc[bad & p59.slot.eq('hands_interacting'), 'n_hands'] = 1
        p59.loc[bad & p59.slot.eq('hair_clothing_01'), 'h0_target'] = 'clothing'
        p59.loc[bad & p59.slot.eq('hair_clothing_02'), 'h0_target'] = 'hair'
        p59.loc[bad & p59.slot.eq('hair_clothing_03'), ['h0_sharp', 'h1_sharp']] = 70
        full, selected = select_upstream(pd.DataFrame(plans), p14, p59, pd.DataFrame(teeth))
        self.assertFalse(full[full.track.eq('track24/speaker_1')]['pass'].any())
        self.assertEqual(len(selected), 180)
        for _, group in selected.groupby('property'):
            self.assertEqual(group['index'].tolist(), list(range(20)))
            self.assertEqual(sorted(group.session_id.value_counts().tolist()), [6, 7, 7])
        with self.assertRaisesRegex(ValueError, 'Missing measurements'):
            select_upstream(pd.DataFrame(plans), p14.iloc[1:], p59, pd.DataFrame(teeth))
        beatles_rows = [{'uuid': 'actor', 'recording_id': f'track{index:02d}',
                         'speaker': 1, 'track': f'track{index:02d}/speaker_1'} for index in range(25)]
        marked = pd.DataFrame(plans)
        marked['in_eval'], marked['excluded'], marked['candidate'] = True, True, False
        _, beatles, receipt = select_beatles(marked, p14, p59, pd.DataFrame(teeth), beatles_rows, [0, 1, 2, 3])
        self.assertEqual(len(beatles), 180)
        self.assertFalse(set(beatles.recording_id) & set(receipt['held_recordings']))
        self.assertEqual(set(beatles.bank), {'beatles-eval'})

    def test_hair_dedup_uses_selected_actor_images_and_exact_boundaries(self):
        plans, hands = [], []
        cases = [('a', 7, 0., 0, 120), ('b', 7, 20., 0, 110),
                 ('a', 8, 1.96, 100, 120), ('a', 9, 2., 100, 120),
                 ('c', 9, 40., 105, 110), ('d', 9, 40., 106, 100)]
        for recording, prop, seconds, shade, sharp in cases:
            track, slot = f'{recording}/speaker_1', SLOTS[prop - 1]
            frame_index = round(seconds * 25)
            plans.append(dict(track=track, uuid='actor', bank='bank', cohort='train', property=prop,
                              slot=slot, recording_id=recording, speaker=1, session_id=recording,
                              frame_index=frame_index, time_s=seconds, source='fixture', source_sha256='fixture',
                              thumb=[shade] * 1920, cx0=0, cy0=0, cw=512, ch=960, scale=.5,
                              sel_score=1., gate=False, iid=len(plans), candidate=True,
                              in_eval=False, in_bb=False, in_light=False, excluded=False))
            hands.append(dict(track=track, slot=slot, frame_index=frame_index, mp_ok=True, missing=False,
                              blinkL=.1, blinkR=.1, blur=100, n_hands=1, h0_inframe=21, h1_inframe=0,
                              h0_sharp=sharp, h1_sharp=0, h0_ext=4, h1_ext=0, hh_dist=0,
                              h0_face_ov=0, h1_face_ov=0, interaction_gate_passed=False,
                              h0_target='hair' if prop == 7 else 'clothing', h1_target='none',
                              hc_min_dt=0, hc_min_mad=0))
        faces = pd.DataFrame(columns=['track', 'slot', 'frame_index', 'mp_ok', 'missing', 'blinkL', 'blinkR',
                                      'blur', 'hand_in_face', 'oof', 'bx0', 'by0', 'bx1', 'by1', 'yaw', 'score', 'jaw'])
        teeth = pd.DataFrame(columns=['track', 'ok', 'up', 'lo', 'span'])
        full, selected = select_upstream(pd.DataFrame(plans), faces, pd.DataFrame(hands), teeth)
        self.assertEqual(selected.iid.tolist(), [0, 3, 5])
        self.assertEqual(full.loc[full.dedup_rejected, 'iid'].tolist(), [1, 2, 4])
        with self.assertRaisesRegex(ValueError, 'thumbnail'):
            select_upstream(pd.DataFrame(plans).drop(columns='thumb'), faces, pd.DataFrame(hands), teeth)


if __name__ == '__main__':
    unittest.main()
