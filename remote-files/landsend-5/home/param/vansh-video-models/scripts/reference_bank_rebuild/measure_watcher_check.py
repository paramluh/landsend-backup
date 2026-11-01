"""Regression for download completion racing the last active measurement."""
from concurrent.futures import Future
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from hf_sources import write_json
import measure_watcher
from upstream import REFERENCE_PREFIX, SELECTOR_FOLDER, SLOTS


class WatcherCompletionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        self.output = root / 'measurements'
        self.output.mkdir()
        self.receipts = root / 'receipts.jsonl'
        self.summary = Path(str(self.receipts) + '.summary.json')
        write_json(str(self.receipts) + '.missing.json', {'missing': []})
        tracks = root / 'tracks.txt'
        tracks.write_text('recording-a/speaker_1\nrecording-b/speaker_1\n')
        self.keys = {}
        for track in tracks.read_text().splitlines():
            prefix = f'{REFERENCE_PREFIX}/{track}/{SELECTOR_FOLDER}'
            write_json(root / prefix / 'manifest.json', {'images': []})
            self.keys[track] = [f'{prefix}/manifest.json', f'{prefix}/selected_poses.parquet',
                                *[f'{prefix}/{slot}.png' for slot in SLOTS]]
        self.receipts.write_text(''.join(json.dumps({'source_key': key}) + '\n'
                                        for key in self.keys['recording-a/speaker_1']))
        self.args = SimpleNamespace(tracks=tracks, data_root=root, output_dir=self.output,
                                    receipts=[self.receipts], workers=1, shard=0, num_shards=1,
                                    run_id='completion-race')
        self.submitted = []
        self.pool = MagicMock()
        self.pool.__enter__.return_value = self.pool
        self.pool.submit.side_effect = self.submit

    def submit(self, function, track):
        self.submitted.append(track)
        future = Future()
        future.set_result(None)
        return future

    def watch(self, wait):
        with ExitStack() as patches:
            patches.enter_context(patch.object(measure_watcher, 'ProcessPoolExecutor', return_value=self.pool))
            patches.enter_context(patch.object(measure_watcher, 'measurement_config', return_value={}))
            patches.enter_context(patch.object(measure_watcher, 'checkpoint_valid', return_value=False))
            patches.enter_context(patch.object(measure_watcher, 'wait', side_effect=wait))
            patches.enter_context(patch.object(measure_watcher.time, 'sleep'))
            return measure_watcher.watch(self.args, self.output)

    def test_final_receipts_published_while_waiting_are_drained_before_failure(self):
        waited = 0
        def finish(active, **kwargs):
            nonlocal waited
            waited += 1
            if waited == 1:
                # These receipts were absent when this iteration read the file.
                with self.receipts.open('a') as handle:
                    handle.writelines(json.dumps({'source_key': key}) + '\n'
                                      for key in self.keys['recording-b/speaker_1'])
                write_json(self.summary, {'requested': 22, 'missing': [], 'failures': []})
            return set(active), set()
        self.watch(finish)
        self.assertEqual(self.submitted, ['recording-a/speaker_1', 'recording-b/speaker_1'])
        self.assertEqual(waited, 2)
        report = json.loads((self.output / 'watch-complete.json').read_text())
        self.assertTrue(report['complete'])
        self.assertEqual(report['tracks'], 2)

    def test_exhausted_receipts_still_fail_for_a_genuinely_unready_track(self):
        def finish(active, **kwargs):
            write_json(self.summary, {'requested': 22, 'missing': [], 'failures': []})
            return set(active), set()
        with self.assertRaisesRegex(RuntimeError, '1 tracks not ready'):
            self.watch(finish)
        self.assertEqual(self.submitted, ['recording-a/speaker_1'])
        self.assertFalse((self.output / 'watch-complete.json').exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
