"""Bounded checks that completed-scan reuse cannot bypass source or chunk checks."""
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from download_and_scan import file_identity
from hf_sources import BUCKET, write_json
from reuse_completed_scan import RECEIPT, prepare, verify
from video_pipeline import digest, input_fingerprint, plan_identity, task_path


class CompletedScanReuseTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.cache, self.output, self.sources = (self.root / name for name in ('old/video-scan', 'new/video-scan', 'sources'))
        for path in (self.cache, self.output, self.sources):
            path.mkdir(parents=True)
        self.video, self.pose = self.sources / 'full.mp4', self.sources / 'poses.safetensors'
        self.video.write_bytes(b'verified-HF-video')
        self.pose.write_bytes(b'verified-HF-pose')
        self.config = {'rebuild/video_scan.py': 'unchanged-code', 'models/face_landmarker.task': 'unchanged-model'}
        self.track = {'track': 'recording/speaker_1', 'recording_id': 'recording', 'speaker': 1,
                      'uuid': 'actor', 'session_id': 'session', 'bank': 'bank', 'cohort': 'train',
                      'usable_chunks': [0], 'video_path': str(self.video), 'pose_paths': {'0': str(self.pose)},
                      'excluded_ranges': [], 'source': f'hf://buckets/{BUCKET}/video/full.mp4',
                      'pose_sources': {'0': f'hf://buckets/{BUCKET}/poses/poses.safetensors'},
                      'video_source_metadata': {'size': self.video.stat().st_size, 'xet_hash': 'a' * 64}}
        for path in (self.cache, self.output):
            write_json(path / 'tracks.json', [self.track])
            write_json(path / 'targets-summary.json', {'complete': False, 'missing_actors': 1})
            (path / 'downloads.tsv').write_text(f'video/full.mp4\t{self.video}\nposes/poses.safetensors\t{self.pose}\n')
        self.receipts = self.cache.parent / 'hf-video-receipts.jsonl'
        lines = []
        for key, path in [('video/full.mp4', self.video), ('poses/poses.safetensors', self.pose)]:
            lines.append({'status': 'reused', 'bucket': BUCKET, 'source_key': key, 'local_path': str(path),
                          'size': path.stat().st_size, 'xet_hash': 'a' * 64, 'sha256': digest(path),
                          'verified_at': datetime.now(timezone.utc).isoformat(), 'local_stat': list(file_identity(path))})
        self.receipts.write_text(''.join(json.dumps(row) + '\n' for row in lines))
        write_json(str(self.receipts) + '.summary.json', {'requested': 2, 'failures': [], 'missing': []})
        old_report = self.scan_report(self.cache)
        write_json(self.cache / 'scan-summary.json', old_report)
        write_json(self.cache / 'scan-code-inputs.json', self.config)
        write_json(self.cache / 'download-scan-status.json', {'stage': 'finished', 'available_targets_complete': True,
                   'plan_sha256': plan_identity(self.cache), 'scan_summary': old_report, 'selected_inputs_verified': 2})
        row = {key: self.track[key] for key in ('track', 'uuid', 'recording_id', 'speaker', 'session_id', 'bank', 'cohort')}
        row.update(chunk=0, frame_index=12, time_s=.48, pose_valid=True, in_eval=False, in_bb=False, in_light=False,
                   source=self.track['source'] + '#frame=12', candidate_id=self.track['track'] + '#frame=12')
        self.candidate = task_path(self.cache, self.track['track'], 0)
        self.candidate.parent.mkdir(parents=True)
        pd.DataFrame([row]).to_parquet(self.candidate, index=False)
        write_json(self.candidate.with_suffix('.complete.json'), {'complete': True, 'track': self.track['track'],
                   'chunk': 0, 'rows': 1, 'sha256': digest(self.candidate),
                   'fingerprint': input_fingerprint(self.track, 0, self.pose, self.config)})

    def scan_report(self, path):
        return {'completed_chunks': 1, 'candidate_rows': 1, 'missing_sources': 0, 'failures': 0,
                'plan_sha256': plan_identity(path), 'available_targets_complete': True, 'complete': False}

    def prepare(self):
        return prepare(self.cache, self.output, self.sources, self.receipts, self.config)

    def finish(self, node_count=5):
        write_json(self.output / 'scan-summary.json', self.scan_report(self.output))
        write_json(self.output / 'scan-code-inputs.json', self.config)
        write_json(self.output / 'distributed-scan.json', {'aggregation_verified': True, 'available_targets_complete': True,
                   'nodes': [{'node': str(n)} for n in range(node_count)],
                   'returncodes': {str(n): 0 for n in range(node_count)}})
        return verify(self.cache, self.output, self.sources, self.receipts, self.config)

    def test_verified_import_then_five_node_receipt_preserves_source_gaps(self):
        result = self.prepare()
        self.assertFalse(result['complete'])
        self.assertEqual(result['expected_chunks'], 1)
        self.assertEqual(self.candidate.read_bytes(), task_path(self.output, self.track['track'], 0).read_bytes())
        finished = self.finish()
        self.assertTrue(finished['complete'])
        self.assertTrue(finished['available_targets_complete'])
        self.assertFalse(finished['source_complete'])
        self.assertIn(str(self.receipts), finished['evidence_sha256'])

    def test_missing_candidate_fails_instead_of_starting_recomputation(self):
        self.candidate.unlink()
        with self.assertRaisesRegex(ValueError, 'every selected chunk'):
            self.prepare()
        self.assertFalse((self.output / RECEIPT).exists())

    def test_policy_and_candidate_changes_fail_closed(self):
        changed = deepcopy(self.track)
        changed['excluded_ranges'] = [[0, 1]]
        write_json(self.output / 'tracks.json', [changed])
        with self.assertRaisesRegex(ValueError, 'absent or differs'):
            self.prepare()
        write_json(self.output / 'tracks.json', [self.track])
        with self.candidate.open('ab') as handle:
            handle.write(b'corrupt-candidate')
        with self.assertRaisesRegex(ValueError, 'every selected chunk'):
            self.prepare()

    def test_changed_local_input_or_failed_download_summary_rejects_reuse(self):
        write_json(str(self.receipts) + '.summary.json', {'requested': 2, 'failures': ['failed'], 'missing': []})
        with self.assertRaisesRegex(ValueError, 'receipt set'):
            self.prepare()
        write_json(str(self.receipts) + '.summary.json', {'requested': 2, 'failures': [], 'missing': []})
        self.video.write_bytes(b'changed-video-content')
        with self.assertRaisesRegex(ValueError, 'changed after downloader verification'):
            self.prepare()

    def test_post_import_evidence_change_and_nonfive_node_result_reject(self):
        self.prepare()
        with self.assertRaisesRegex(ValueError, 'five-node'):
            self.finish(node_count=4)
        self.assertFalse(json.loads((self.output / RECEIPT).read_text())['complete'])
        self.prepare()
        with self.receipts.open('a') as handle:
            handle.write('\n')
        with self.assertRaises(ValueError):
            self.finish()
        self.assertFalse(json.loads((self.output / RECEIPT).read_text())['complete'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
