"""Exact serial parity for ordered IDs, diagnostics and real source tracks."""

from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import sys

import numpy as np
import pandas as pd
from PIL import Image

import parallel_plan
from parallel_plan import parallel_build_plan
from upstream import Corpus, REFERENCE_PREFIX, SELECTOR_FOLDER, SLOTS, build_plan, load_corpus


def assert_parity(test, corpus, source_root, known_missing=(), workers=2, chunk_size=1):
    serial, serial_report = build_plan(corpus, source_root, known_missing)
    parallel, parallel_report = parallel_build_plan(corpus, source_root, known_missing,
                                                    workers=workers, chunk_size=chunk_size)
    pd.testing.assert_frame_equal(serial, parallel, check_exact=True)
    test.assertEqual(serial_report, parallel_report)
    with tempfile.TemporaryDirectory() as directory:
        paths = [Path(directory) / f'{name}.parquet' for name in ('serial', 'parallel')]
        serial.to_parquet(paths[0], index=False)
        parallel.to_parquet(paths[1], index=False)
        test.assertEqual(paths[0].read_bytes(), paths[1].read_bytes())
    return serial, serial_report


class ParallelPlanChecks(unittest.TestCase):
    def test_mixed_missing_exclusions_checksum_issues_and_global_ids(self):
        # Unsorted corpus order must survive process completion order. One whole
        # chunk is empty, exercising DataFrame inference across null columns.
        tracks = [dict(track=f'{name}/speaker_1', recording_id=name, speaker=1, uuid=f'actor{index}',
                       session_id=f'session{index}', cohort='eval-unseen' if index == 2 else 'train', bank='bank')
                  for index, name in enumerate(('z-first', 'a-empty', 'm-last'))]
        corpus = Corpus([], tracks, [], {'z-first/speaker_1': [(0, 50)]}, {}, {},
                        {'m-last/speaker_1': [(0, 100)]}, {'fixture.csv': 'fixture-hash'})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            known_missing = set()
            for index, track in enumerate(tracks):
                prefix = f"{REFERENCE_PREFIX}/{track['track']}/{SELECTOR_FOLDER}"
                path = root / prefix
                if index == 1:
                    known_missing |= {f'{prefix}/manifest.json', f'{prefix}/selected_poses.parquet'}
                    continue
                path.mkdir(parents=True)
                keypoints = np.zeros((133, 3), np.float32)
                keypoints[23:91] = [220, 300, .9]
                keypoints[23], keypoints[24] = [180, 200, .9], [280, 350, .9]
                images, poses = [], []
                for slot_index, slot in enumerate(SLOTS):
                    payload = io.BytesIO()
                    Image.new('RGB', (512, 960), (index * 50, slot_index * 20, 100)).save(payload, format='PNG')
                    value = payload.getvalue()
                    (path / f'{slot}.png').write_bytes(value)
                    checksum = hashlib.sha256(value).hexdigest()
                    images.append(dict(category=slot, frame_index=1000, time_s=40., score=float('nan'),
                                       interaction_gate_passed=slot_index % 2 == 0, sha256=checksum))
                    poses.append(dict(category=slot, image_keypoints=keypoints.tolist(), image_sha256=checksum))
                (path / 'manifest.json').write_text(json.dumps({'images': images}))
                pd.DataFrame(poses).to_parquet(path / 'selected_poses.parquet')
            # A checksum issue is diagnostic, not silently dropped by a reducer.
            corrupt = root / REFERENCE_PREFIX / tracks[-1]['track'] / SELECTOR_FOLDER / 'hands_visible.png'
            corrupt.write_bytes(b'corrupt')
            frame, report = assert_parity(self, corpus, root, known_missing)
            self.assertEqual(frame.iid.tolist(), list(range(27)))
            self.assertEqual(frame.track.iloc[::9].tolist(), [row['track'] for row in tracks])
            self.assertEqual(report['issue_counts'], {'confirmed_hf_missing': 2, 'image_checksum_mismatch': 2})
            self.assertEqual(report['fatal_input_issues'], 2)

    def test_bad_corpus_invalidates_prior_completion_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker = root / 'parallel-plan.complete.json'
            marker.write_text('{"complete": true}')
            with patch.object(sys, 'argv', ['parallel_plan.py', '--output-dir', str(root),
                                           '--csv-root', str(root / 'missing-csvs'), '--data-root', str(root)]):
                with self.assertRaises(FileNotFoundError):
                    parallel_plan.main()
            self.assertFalse(marker.exists())

    def test_identity_error_propagates_from_worker(self):
        record = dict(track='track/speaker_1', recording_id='track', speaker=1, uuid='actor',
                      session_id='session', cohort='train', bank='bank')
        corpus = Corpus([], [record], [], {}, {}, {}, {}, {})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / REFERENCE_PREFIX / record['track'] / SELECTOR_FOLDER
            path.mkdir(parents=True)
            (path / 'manifest.json').write_text(json.dumps({'person_uuid': 'wrong', 'images': []}))
            with self.assertRaisesRegex(ValueError, 'Actor identity mismatch'):
                parallel_build_plan(corpus, directory, workers=2, chunk_size=1)

    def test_actual_small_studio_subset_is_byte_identical(self):
        root = Path('/mnt/paramfs/param/outputs/reference-bank-rebuild-20261018')
        csvs = root / 'studio-only/csvs'
        if not csvs.is_dir():
            self.skipTest('Local audited studio source fixture unavailable')
        corpus = load_corpus(csvs)
        special = {'7168b3e51724067fee4adcb52a7fbaae6ae711cb/speaker_2',
                   '866721cf41e1ac227eb4bb21d0da4d9b218bd153/speaker_1'}
        subset = corpus.tracks[:4] + [r for r in corpus.tracks if r['cohort'] == 'eval-unseen'][:2]
        subset += [r for r in corpus.tracks if r['track'] in special]
        # Reversed track order additionally checks global ID assignment.
        corpus = replace(corpus, tracks=list(reversed(subset)))
        known_missing = {key for name in ('metadata', 'image') for key in json.loads(
            (root / f'hf-upstream-{name}-receipts.jsonl.missing.json').read_text())['missing']}
        frame, _ = assert_parity(self, corpus, root / 'sources', known_missing, workers=2, chunk_size=2)
        self.assertEqual(len(frame), 72)


if __name__ == '__main__':
    unittest.main()
