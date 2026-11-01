"""Bounded checks for joining early source writers without stopping them."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from run_rebuild import join_prefetch_downloads


class PrefetchJoinTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.registry = self.root / 'prefetch-downloads.json'
        self.receipts = self.root / 'receipts.jsonl'
        self.summary = Path(str(self.receipts) + '.summary.json')
        self.job = {'receipts': str(self.receipts), 'requested': 10, 'pid': 123456}
        self.register()
        self.status = Mock()

    def register(self):
        self.registry.write_text(json.dumps({'jobs': [self.job]}))

    def finish(self, **changes):
        self.summary.write_text(json.dumps({'requested': 10, 'missing': [], 'failures': [], **changes}))

    def test_successful_summary_is_sufficient_even_if_pid_has_exited(self):
        self.finish()
        with patch('run_rebuild.process_alive') as alive, patch('run_rebuild.time.sleep') as sleep:
            join_prefetch_downloads(self.registry, self.status)
        alive.assert_not_called()
        sleep.assert_not_called()
        self.assertEqual(self.status.call_args.kwargs['completed_jobs'], 1)
        self.assertFalse(self.status.call_args.kwargs['running'])

    def test_failed_missing_or_stale_summary_fails_closed(self):
        for changes in ({'failures': [{'key': 'failed'}]}, {'missing': ['missing']}, {'requested': 9}):
            with self.subTest(changes=changes):
                self.finish(**changes)
                with self.assertRaises(RuntimeError), patch('run_rebuild.time.sleep') as sleep:
                    join_prefetch_downloads(self.registry, self.status)
                sleep.assert_not_called()
                self.assertTrue(self.status.call_args.kwargs['failed'])

    def test_dead_downloader_without_summary_fails_actionably(self):
        with patch('run_rebuild.process_alive', return_value=False), patch('run_rebuild.time.sleep') as sleep:
            with self.assertRaisesRegex(RuntimeError, 'exited without .*resume that download'):
                join_prefetch_downloads(self.registry, self.status)
        sleep.assert_not_called()

    def test_live_downloader_waits_and_is_never_killed(self):
        with patch('run_rebuild.process_alive', return_value=True), \
                patch('run_rebuild.time.sleep', side_effect=lambda _: self.finish()) as sleep, \
                patch('run_rebuild.os.kill') as kill:
            join_prefetch_downloads(self.registry, self.status)
        sleep.assert_called_once_with(30)
        kill.assert_not_called()
        self.assertTrue(self.status.call_args_list[0].kwargs['running'])
        self.assertFalse(self.status.call_args.kwargs['running'])

    def test_queued_job_can_start_but_new_jobs_cannot_appear_after_join(self):
        self.job['pid'] = None
        self.register()
        def start_then_finish(_):
            self.job['pid'] = 123456
            self.register()
            self.finish()
        with patch('run_rebuild.time.sleep', side_effect=start_then_finish):
            join_prefetch_downloads(self.registry, self.status)
        self.summary.unlink()
        def add_job(_):
            self.registry.write_text(json.dumps({'jobs': [self.job, {
                'receipts': str(self.root / 'new.jsonl'), 'requested': 1, 'pid': None}]}))
        with patch('run_rebuild.process_alive', return_value=True), patch('run_rebuild.time.sleep', side_effect=add_job):
            with self.assertRaisesRegex(ValueError, 'job set changed'):
                join_prefetch_downloads(self.registry, self.status)


if __name__ == '__main__':
    unittest.main(verbosity=2)
