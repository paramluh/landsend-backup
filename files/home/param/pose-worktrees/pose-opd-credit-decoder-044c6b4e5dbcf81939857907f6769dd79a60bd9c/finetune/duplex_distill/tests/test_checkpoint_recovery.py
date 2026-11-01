"""Exercise trainer recovery policy without importing its GPU runtime."""
import ast
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, call, patch

from finetune.duplex_distill.checkpoints import CheckpointSaveError


def helpers():
    path = Path(__file__).resolve().parents[1] / 'train.py'
    names = {'wandb_resume_options', 'checkpoint_event', 'checkpoint_with_retries'}
    functions = [node for node in ast.parse(path.read_text()).body
                 if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in functions} == names
    sleep = Mock()
    namespace = {'Path': Path, 'json': json, 'time': SimpleNamespace(sleep=sleep)}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace, sleep


class CheckpointRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.functions, self.sleep = helpers()
        self.save = self.functions['checkpoint_with_retries']
        self.resume = self.functions['wandb_resume_options']
        self.positional = (object(), {'main': object()}, Path('/unused-run'), 37, {'steps': 284})
        self.log = Mock()
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.output = self.root / 'new-run'
        self.source = self.root / 'original-run'
        self.checkpoint = self.source / 'checkpoints' / 'step-000037' / 'resume'
        self.checkpoint.mkdir(parents=True)
        self.output.mkdir()
        self.project = 'param-pose-llm'

    def tearDown(self):
        self.temp.cleanup()

    def metadata(self, root, **overrides):
        record = {'id': 'preserve-current-id', 'project': self.project}
        record.update(overrides)
        (root / 'wandb.json').write_text(json.dumps(record))

    def invoke(self, save_fn, *, required, **kwargs):
        return self.save(save_fn, CheckpointSaveError, self.positional,
                         required=required, log=self.log, **kwargs)

    def test_fresh_run_does_not_reuse_existing_id(self):
        self.metadata(self.output)
        self.assertEqual(self.resume(self.output, None, self.project), {})

    def test_resume_to_new_directory_uses_original_wandb_id(self):
        self.metadata(self.source)
        result = self.resume(self.output, self.checkpoint, self.project)
        self.assertEqual(result, {'id': 'preserve-current-id', 'resume': 'must'})
        self.assertFalse((self.output / 'wandb.json').exists())

    def test_resume_finds_wandb_after_checkpoints_move_to_models(self):
        self.metadata(self.source)
        moved = self.root / 'models/trained-weights/original-run/checkpoints/step-000037/resume'
        moved.mkdir(parents=True)
        (moved / 'train_state.json').write_text(json.dumps({'run_dir': str(self.source)}))
        self.assertEqual(self.resume(self.output, moved, self.project),
                         {'id': 'preserve-current-id', 'resume': 'must'})

    def test_resume_same_directory_accepts_one_record(self):
        self.metadata(self.source)
        self.assertEqual(self.resume(self.source, self.checkpoint, self.project),
                         {'id': 'preserve-current-id', 'resume': 'must'})

    def test_resume_accepts_matching_source_and_output_records(self):
        self.metadata(self.source)
        self.metadata(self.output)
        self.assertEqual(self.resume(self.output, self.checkpoint, self.project)['id'],
                         'preserve-current-id')

    def test_resume_requires_recorded_id(self):
        with self.assertRaisesRegex(ValueError, 'existing wandb.json'):
            self.resume(self.output, self.checkpoint, self.project)

    def test_resume_rejects_project_mismatch_in_either_record(self):
        for bad_root in (self.source, self.output):
            with self.subTest(directory=bad_root.name):
                self.metadata(self.source)
                self.metadata(self.output)
                self.metadata(bad_root, project='different-project')
                with self.assertRaisesRegex(ValueError, 'project'):
                    self.resume(self.output, self.checkpoint, self.project)

    def test_resume_rejects_missing_empty_or_nonstring_id(self):
        for invalid in (None, '', 123):
            with self.subTest(id=invalid):
                self.metadata(self.source, id=invalid)
                with self.assertRaisesRegex(ValueError, 'invalid'):
                    self.resume(self.output, self.checkpoint, self.project)

    def test_resume_rejects_conflicting_run_ids(self):
        self.metadata(self.source)
        self.metadata(self.output, id='another-run')
        with self.assertRaisesRegex(ValueError, 'different runs'):
            self.resume(self.output, self.checkpoint, self.project)

    def test_resume_rejects_unparseable_metadata(self):
        (self.source / 'wandb.json').write_text('{incomplete')
        with self.assertRaises(ValueError):
            self.resume(self.output, self.checkpoint, self.project)

    def test_periodic_synchronized_failure_returns_without_retry(self):
        error = CheckpointSaveError('file writes', ['rank3 disk full'])
        save_fn = Mock(side_effect=error)
        self.assertIsNone(self.invoke(save_fn, required=False))
        save_fn.assert_called_once_with(*self.positional, attempt=0)
        self.sleep.assert_not_called()
        failure = self.log.call_args_list[-1]
        self.assertEqual(failure.args, ('save_failed',))
        self.assertTrue(failure.kwargs['training_can_continue'])
        self.assertFalse(any(c.args[0] == 'save_complete' for c in self.log.call_args_list))

    def test_required_save_retries_distinct_attempts_then_returns_success(self):
        result = {'resume_path': '/unused-run/checkpoints/step-000037-retry-02/resume'}
        save_fn = Mock(side_effect=[CheckpointSaveError('directory creation', ['rank0 permission']),
                                    CheckpointSaveError('file writes', ['rank5 disk full']), result])
        self.assertIs(self.invoke(save_fn, required=True), result)
        self.assertEqual(save_fn.call_args_list,
                         [call(*self.positional, attempt=i) for i in range(3)])
        self.assertEqual(self.sleep.call_args_list, [call(5), call(5)])
        self.assertEqual([c.args[0] for c in self.log.call_args_list],
                         ['save_start', 'save_failed', 'save_start', 'save_failed',
                          'save_start', 'save_complete'])
        self.assertTrue(all(c.kwargs['step'] == 37 for c in self.log.call_args_list))

    def test_required_save_exhaustion_raises_without_false_success(self):
        error = CheckpointSaveError('preflight', ['not enough space'])
        save_fn = Mock(side_effect=error)
        with self.assertRaises(CheckpointSaveError) as raised:
            self.invoke(save_fn, required=True, attempts=2, retry_seconds=.25)
        self.assertIs(raised.exception, error)
        self.assertEqual(save_fn.call_count, 2)
        self.sleep.assert_called_once_with(.25)
        failures = [c for c in self.log.call_args_list if c.args[0] == 'save_failed']
        self.assertEqual(len(failures), 2)
        self.assertTrue(all(not c.kwargs['training_can_continue'] for c in failures))
        self.assertFalse(any(c.args[0] == 'save_complete' for c in self.log.call_args_list))

    def test_arbitrary_errors_are_not_treated_as_synchronized_io_failures(self):
        for required in (False, True):
            with self.subTest(required=required):
                error = RuntimeError('collective transport lost')
                save_fn = Mock(side_effect=error)
                with self.assertRaises(RuntimeError) as raised:
                    self.invoke(save_fn, required=required)
                self.assertIs(raised.exception, error)
                save_fn.assert_called_once_with(*self.positional, attempt=0)
                self.sleep.assert_not_called()

    def test_logger_disk_failure_does_not_prevent_save_collective(self):
        self.log.side_effect = OSError('event log disk full')
        result = {'path': '/verified-checkpoint'}
        save_fn = Mock(return_value=result)
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertIs(self.invoke(save_fn, required=True), result)
        save_fn.assert_called_once_with(*self.positional, attempt=0)
        fallback = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual([record['event'] for record in fallback], ['save_start', 'save_complete'])
        self.assertTrue(all(record['checkpoint_log_error'] == 'event log disk full' for record in fallback))
        self.sleep.assert_not_called()


    def test_logger_and_stdout_failures_do_not_prevent_save_collective(self):
        self.log.side_effect = OSError('event log disk full')
        result = {'path': '/verified-checkpoint'}
        save_fn = Mock(return_value=result)
        with patch('builtins.print', side_effect=OSError('stdout closed')):
            self.assertIs(self.invoke(save_fn, required=True), result)
        save_fn.assert_called_once_with(*self.positional, attempt=0)
        self.sleep.assert_not_called()


if __name__ == '__main__':
    unittest.main()
