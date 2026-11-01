"""Focused CPU regression checks for importing independent early-scan chunks."""
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from hf_sources import write_json
from import_scan_checkpoints import import_caches
from video_pipeline import digest, input_fingerprint, task_path


class ScanImportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.cache, self.final = self.root / 'preview', self.root / 'final'
        self.cache.mkdir()
        self.final.mkdir()
        video, pose = self.root / 'full.mp4', self.root / 'poses.safetensors'
        video.write_bytes(b'verified source video fixture')
        pose.write_bytes(b'verified source pose fixture')
        self.track = {'track': 'recording/speaker_1', 'recording_id': 'recording', 'speaker': 1,
                      'uuid': 'actor', 'session_id': 'session', 'bank': 'bank', 'cohort': 'train',
                      'usable_chunks': [0, 1], 'video_path': str(video),
                      'pose_paths': {'0': str(pose), '1': str(pose)}, 'excluded_ranges': [],
                      'source': 'hf://buckets/test/recording/speaker_1/video/original/full.mp4'}
        self.config = {'rebuild/video_scan.py': 'exact-code-hash', 'models/face_landmarker.task': 'exact-model-hash'}
        for folder in (self.cache, self.final):
            write_json(folder / 'tracks.json', [self.track])
        row = {key: self.track[key] for key in ('track', 'uuid', 'recording_id', 'speaker', 'session_id', 'bank', 'cohort')}
        row.update(chunk=0, frame_index=12, time_s=.48, pose_valid=True, in_eval=False, in_bb=False,
                   in_light=False, source=self.track['source'] + '#frame=12',
                   candidate_id=self.track['track'] + '#frame=12')
        self.source = task_path(self.cache, self.track['track'], 0)
        self.source.parent.mkdir(parents=True)
        pd.DataFrame([row]).to_parquet(self.source, index=False)
        self.receipt = {'track': self.track['track'], 'chunk': 0, 'rows': 1, 'complete': True,
                        'fingerprint': input_fingerprint(self.track, 0, pose, self.config), 'sha256': digest(self.source)}
        write_json(self.source.with_suffix('.complete.json'), self.receipt)

    def imported(self, config=None):
        return import_caches(self.final, [self.cache], self.config if config is None else config)

    def assert_rejected(self, report):
        self.assertEqual(report['imported_chunks'], 0)
        self.assertEqual(report['rejected_chunks'], 1)
        self.assertFalse(task_path(self.final, self.track['track'], 0).exists())

    def test_import_without_preview_summary_and_resume_identical_content(self):
        # The other chunk is unfinished, and no preview-wide summary exists.
        report = self.imported()
        self.assertEqual(report['imported_chunks'], 1)
        self.assertEqual(report['absent_chunks'], 1)
        destination = task_path(self.final, self.track['track'], 0)
        self.assertEqual(destination.read_bytes(), self.source.read_bytes())
        before = destination.stat().st_mtime_ns
        again = self.imported()
        self.assertEqual(again['already_valid_chunks'], 1)
        self.assertEqual(destination.stat().st_mtime_ns, before)
        self.assertFalse((self.final / 'scan-summary.json').exists())

    def test_changed_final_track_is_rejected(self):
        changed = deepcopy(self.track)
        changed['excluded_ranges'] = [[0, 1]]
        write_json(self.final / 'tracks.json', [changed])
        self.assert_rejected(self.imported())

    def test_changed_code_or_model_is_rejected(self):
        changed = {**self.config, 'rebuild/video_scan.py': 'new-code-hash'}
        self.assert_rejected(self.imported(changed))
        changed = {**self.config, 'models/face_landmarker.task': 'new-model-hash'}
        self.assert_rejected(self.imported(changed))

    def test_changed_pose_bytes_are_rejected(self):
        Path(self.track['pose_paths']['0']).write_bytes(b'changed source poses')
        self.assert_rejected(self.imported())

    def test_changed_video_stat_is_rejected(self):
        video = Path(self.track['video_path'])
        stat = video.stat()
        os.utime(video, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
        self.assert_rejected(self.imported())

    def test_candidate_checksum_corruption_is_rejected(self):
        with self.source.open('ab') as handle:
            handle.write(b'changed candidate bytes')
        self.assert_rejected(self.imported())

    def test_wrong_row_identity_is_rejected_even_with_updated_checksum(self):
        rows = pd.read_parquet(self.source)
        rows['speaker'] = 2
        rows.to_parquet(self.source, index=False)
        self.receipt['sha256'] = digest(self.source)
        write_json(self.source.with_suffix('.complete.json'), self.receipt)
        self.assert_rejected(self.imported())

    def test_unfinished_receipt_is_rejected(self):
        self.receipt['complete'] = False
        write_json(self.source.with_suffix('.complete.json'), self.receipt)
        self.assert_rejected(self.imported())

    def test_tracks_not_selected_by_final_plan_are_not_imported(self):
        write_json(self.final / 'tracks.json', [])
        report = self.imported()
        self.assertEqual(report['imported_chunks'], 0)
        self.assertEqual(report['caches'][0]['tracks_not_in_final'], [self.track['track']])


if __name__ == '__main__':
    unittest.main(verbosity=2)
