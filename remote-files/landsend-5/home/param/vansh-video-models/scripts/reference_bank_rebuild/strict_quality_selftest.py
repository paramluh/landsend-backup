"""Synthetic strict-quality checks; no corpus, models, or network access."""
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest

import cv2
import numpy as np
import pandas as pd

from strict_quality_check import (HF_PREFIX, REFERENCE_PREFIX, SELECTOR_FOLDER, SLOTS,
                                  check_crop, check_duplicates, check_latent_headers, check_quality, validate)
from upstream import crop, face_box


def measurement(prop):
    return dict(mp_ok=True, missing=False, blur=100., blinkL=.1, blinkR=.1,
                yaw=-30. if prop == 2 else 30. if prop == 3 else 0.,
                hand_in_face=0, oof=0, bx0=200., by0=200., bx1=300., by1=320.,
                score=.8, jaw=.4, t_ok=True, up=2., lo=2., span=.8,
                n_hands=2, h0_sharp=100., h1_sharp=100., h0_inframe=21, h1_inframe=21,
                h0_ext=4, h1_ext=4, hh_dist=.3, h0_face_ov=0., h1_face_ov=0., gate=True,
                h0_target='hair' if prop == 7 else 'clothing', h1_target='none')


def points():
    value = np.zeros((133, 3), np.float32)
    value[23:91, 0] = np.linspace(200, 300, 68)
    value[23:91, 1] = np.linspace(200, 320, 68)
    value[23:91, 2] = 1
    return value


class StrictQualityChecks(unittest.TestCase):
    def test_all_nine_rules_and_historical_failures(self):
        failures = [(1, 'yaw', 13.13), (2, 'yaw', -56.), (3, 'hand_in_face', 1),
                    (4, 'yaw', 18.), (5, 'h1_sharp', 70.), (6, 'n_hands', 1),
                    (7, 'h0_target', 'clothing'), (8, 'h0_face_ov', .4), (9, 'h0_target', 'hair')]
        for prop, field, value in failures:
            with self.subTest(prop=prop):
                check_quality(prop, measurement(prop))
                bad = measurement(prop)
                bad[field] = value
                with self.assertRaises(ValueError):
                    check_quality(prop, bad)

    def test_missing_measurements_fail_closed(self):
        for prop, field in [(1, 'blinkL'), (4, 't_ok'), (5, 'h1_face_ov'), (6, 'gate'), (8, 'h0_target')]:
            with self.subTest(prop=prop, field=field):
                row = measurement(prop)
                del row[field]
                with self.assertRaisesRegex(ValueError, 'Missing'):
                    check_quality(prop, row)
        row = measurement(7)
        row.update(n_hands=1, h1_sharp=np.nan, h1_target=None)
        check_quality(7, row)
        row = measurement(7)
        row['h1_sharp'] = np.nan
        check_quality(7, row)
        row = measurement(4)
        row['jaw'] = np.nan
        self.assertTrue(np.isnan(check_quality(4, row)), 'Missing rank is allowed; required quality measurements are not')

    def test_crop_requires_selector_face_and_native_geometry(self):
        row = dict(property=1, cx0=122, cy0=0, cw=256, ch=480, scale=1.)
        check_crop(row, points())
        with self.assertRaisesRegex(ValueError, 'confident'):
            check_crop(row, np.zeros((133, 3)))
        with self.assertRaisesRegex(ValueError, '10%'):
            large = points()
            large[24, 0] = 500
            check_crop(row, large)
        with self.assertRaisesRegex(ValueError, 'centered'):
            check_crop(dict(row, cx0=124), points())
        boundary = points()
        boundary[23:91, 0] = np.linspace(201.000015, 301.99997, 68, dtype=np.float32)
        x, y, width, height = crop(face_box(boundary))
        check_crop(dict(property=1, cx0=x, cy0=y, cw=width, ch=height, scale=256 / width), boundary)

    def test_dedup_cross_recording_mad_and_exact_boundaries(self):
        a = dict(iid=0, bank='bank', uuid='u', property=7, recording_id='a', speaker=1,
                 frame_index=0, origin='ref-image')
        b = dict(a, iid=1, property=8, frame_index=50)
        thumbs = {0: np.zeros(1920), 1: np.full(1920, 6.)}
        check_duplicates([a, b], thumbs)
        with self.assertRaisesRegex(ValueError, '<2 seconds'):
            check_duplicates([a, dict(b, frame_index=49)], thumbs)
        with self.assertRaisesRegex(ValueError, 'MAD'):
            check_duplicates([a, dict(b, recording_id='other')], {0: thumbs[0], 1: np.full(1920, 5.9)})
        scan = dict(b, origin='video-scan', frame_index=250)
        check_duplicates([a, scan], thumbs)
        with self.assertRaisesRegex(ValueError, '<10 seconds'):
            check_duplicates([a, dict(scan, frame_index=249)], thumbs)
        with self.assertRaisesRegex(ValueError, 'MAD'):
            check_duplicates([a, dict(scan, recording_id='other')], {0: thumbs[0], 1: thumbs[0]})

    def test_preencode_checks_raw_evidence_and_source_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            actor = '00cc8905-1a01-4d05-8352-bb77cb51de0f'
            track = 'recording/speaker_1'
            folder = root / 'sources' / REFERENCE_PREFIX / track / SELECTOR_FOLDER
            folder.mkdir(parents=True)
            image = np.zeros((960, 512, 3), np.uint8)
            cv2.imwrite(str(folder / 'front.png'), image)
            checksum = hashlib.sha256((folder / 'front.png').read_bytes()).hexdigest()
            (folder / 'manifest.json').write_text(json.dumps({'person_uuid': actor, 'images': [
                dict(category='front', frame_index=1000, time_s=40., sha256=checksum)]}))
            pd.DataFrame([dict(category='front', image_keypoints=points().tolist(), image_sha256=checksum,
                               frame_index=1000, image_width=512, image_height=960)]).to_parquet(folder / 'selected_poses.parquet')
            measured = dict(measurement(1), track=track, slot='front', frame_index=1000)
            measures = root / 'measurements'
            measures.mkdir()
            pd.DataFrame([measured]).to_parquet(measures / 'p14.parquet')
            pd.DataFrame(columns=['track', 'slot']).to_parquet(measures / 'p59.parquet')
            pd.DataFrame(columns=['track']).to_parquet(measures / 'teeth.parquet')
            row = dict(uuid=actor, bank='bank', cohort='train', property=1, slot=SLOTS[0], index=0,
                       iid=0, recording_id='recording', speaker=1, session_id='session', frame_index=1000,
                       time_s=40., source=HF_PREFIX + str((folder / 'front.png').relative_to(root / 'sources')),
                       source_sha256=checksum, cx0=122, cy0=0, cw=256, ch=480, scale=1., rank_metric=100.,
                       blur=100., blinkL=.1, blinkR=.1, origin='ref-image', thumb=[0] * 1920)
            path = root / 'selection.parquet'
            pd.DataFrame([row]).to_parquet(path)
            report = validate(path, measures, root / 'sources', root / 'scan', root / 'unused-fit.parquet')
            self.assertTrue(report['strict_quality_valid'])
            measured['yaw'] = 13.13
            pd.DataFrame([measured]).to_parquet(measures / 'p14.parquet')
            with self.assertRaisesRegex(ValueError, 'Front'):
                validate(path, measures, root / 'sources', root / 'scan', root / 'unused-fit.parquet')
            measured['yaw'] = 0.
            pd.DataFrame([measured]).to_parquet(measures / 'p14.parquet')
            candidate = dict(measurement(1), uuid=actor, bank='bank', track=track, recording_id='recording',
                             speaker=1, frame_index=1250, source=HF_PREFIX + 'videos/full.mp4#frame=1250',
                             pose_valid=True, in_eval=False, in_bb=False, in_light=False,
                             kp=points().reshape(-1).tolist(), thumb=[10] * 1920)
            candidate_path = root / 'scan' / 'candidates' / track / '000000.parquet'
            candidate_path.parent.mkdir(parents=True)
            pd.DataFrame([candidate]).to_parquet(candidate_path)
            candidate_path.with_suffix('.complete.json').write_text(json.dumps({
                'sha256': hashlib.sha256(candidate_path.read_bytes()).hexdigest(), 'track': track, 'chunk': 0}))
            decoded_path = root / 'decoded.png'
            cv2.imwrite(str(decoded_path), np.full((960, 512, 3), 10, np.uint8))
            scanned = dict(row, iid=1, index=1, origin='video-scan', frame_index=1250, time_s=50.,
                           source=candidate['source'], image_path=str(decoded_path), prepared_image=False,
                           decoded_image_sha256=hashlib.sha256(decoded_path.read_bytes()).hexdigest(), thumb=[10] * 1920)
            pd.DataFrame([row, scanned]).to_parquet(path)
            report = validate(path, measures, root / 'sources', root / 'scan', root / 'unused-fit.parquet')
            self.assertEqual(report['images'], 2)
            self.assertEqual(report['duplicate_pairs_checked'], 1)
            parallel = validate(path, measures, root / 'sources', root / 'scan', root / 'unused-fit.parquet', workers=2)
            self.assertEqual(parallel, report)
            cv2.imwrite(str(decoded_path), np.full((960, 512, 3), 11, np.uint8))
            with self.assertRaisesRegex(ValueError, 'checksum'):
                validate(path, measures, root / 'sources', root / 'scan', root / 'unused-fit.parquet')

    def test_published_header_inventory_and_tensor_extent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'bank' / 'actor' / 'property_1' / 'latents.safetensors'
            path.parent.mkdir(parents=True)
            header = {'video': {'dtype': 'BF16', 'shape': [1, 48, 30, 16], 'data_offsets': [0, 46080]},
                      '__metadata__': {'uuid': 'actor', 'property': '1', 'slot': 'front', 'n': '1'}}
            raw = json.dumps(header).encode()
            path.write_bytes(struct.pack('<Q', len(raw)) + raw + bytes(46080))
            selection = pd.DataFrame([dict(bank='bank', uuid='actor', property=1)])
            self.assertEqual(check_latent_headers(selection, root), 1)
            path.write_bytes(path.read_bytes()[:-1])
            with self.assertRaisesRegex(ValueError, 'byte extent'):
                check_latent_headers(selection, root)


if __name__ == '__main__':
    unittest.main()
