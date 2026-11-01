"""Independent in-memory evaluation cadence preserves checkpoints and RNG."""
import ast
import json
from pathlib import Path
import random
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
import uuid

import numpy as np
import torch

from distillation.checkpoints import restore_rng, rng_state, write_json_atomic
from distillation.joint_runtime import checkpoint_compatibility, full_gt_checkpoint_due, full_gt_evaluation_due

TRAINER = Path(__file__).resolve().parents[1] / 'learn_dmd_joint.py'
SOURCE = ast.parse(TRAINER.read_text())


class FullGTEvaluationCadenceTests(unittest.TestCase):
    def test_student_cadence_save_priority_and_resumed_cursor(self):
        branch = next(node for node in ast.walk(SOURCE) if isinstance(node, ast.If)
            and any(isinstance(child, ast.Name) and child.id == 'due' for child in ast.walk(node.test)))
        for step, warming, manual, expected in (
                (0, True, False, []), (400, True, False, []), (399, False, False, []),
                (400, False, False, ['eval']), (800, False, False, ['save', 'ack']),
                (1200, False, False, ['eval']), (1200, False, True, ['save', 'ack']),
                (2912, False, False, []), (3200, False, False, ['save', 'ack']),
                (3600, False, False, ['eval']), (8000, False, False, ['save', 'ack'])):
            with self.subTest(step=step, warming=warming, manual=manual):
                calls = []
                counters = {'student_updates': step, 'rounds': step + 150}
                namespace = {'full_gt': True, 'full_gt_evaluation_due': full_gt_evaluation_due,
                    'due': full_gt_checkpoint_due(counters, 800, 8000, warming=warming),
                    'counters': counters, 'config': {'save_every_n_steps': 800, 'eval_every_n_steps': 400},
                    'warming': warming, 'request': {'save': manual}, 'run_dir': '/run',
                    'save_checkpoint': lambda: calls.append('save'),
                    'evaluate_current_state': lambda: calls.append('eval'),
                    'root_action': lambda action: action(),
                    'acknowledge_save_request': lambda *_: calls.append('ack')}
                exec(compile(ast.Module(body=[branch], type_ignores=[]), str(TRAINER), 'exec'), namespace)
                self.assertEqual(calls, expected)

    def test_evaluation_only_receipt_and_rng_restore_on_success_or_failure(self):
        function = next(node for node in ast.walk(SOURCE)
                        if isinstance(node, ast.FunctionDef) and node.name == 'evaluate_current_state')
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            (root / 'latest.json').write_text('original latest checkpoint')
            counters = {'student_updates': 400, 'rounds': 550}
            for fail in (False, True):
                with self.subTest(fail=fail):
                    calls = []
                    def probe(directory, *, identity):
                        calls.append(identity)
                        self.assertIsNone(identity['checkpoint_id'])
                        self.assertEqual(identity['evaluation_kind'], 'in_memory')
                        self.assertEqual(identity['counters'], counters)
                        self.assertEqual(identity['last_durable_checkpoint'], '/run/saved')
                        torch.rand(3); random.random(); np.random.rand(3)
                        if fail:
                            raise RuntimeError('probe failed')
                        return [{'weights': 'raw', 'file': 'artifact.json'}]
                    def track(directory, actual_counters):
                        torch.rand(3); random.random(); np.random.rand(3)
                        receipt = json.loads((directory / 'evaluation.complete.json').read_text())
                        self.assertEqual(actual_counters, counters)
                        self.assertEqual(receipt['evaluation_id'], calls[-1]['evaluation_id'])
                    namespace = {'rng_state': rng_state, 'restore_rng': restore_rng,
                        'time': time, 'uuid': uuid, 'Path': Path, '__file__': str(TRAINER),
                        'run_dir': root, 'counters': counters, 'last_checkpoint': None, 'resume': '/run/saved',
                        'signature': {'guidance': 2.5}, 'write_json_atomic': write_json_atomic,
                        'root_action': lambda action: action(), 'root_tracking_action': lambda action: action(),
                        'evaluate': probe, 'tracker': SimpleNamespace(evaluation=track),
                        'rank': 0, 'json': json, 'print': Mock()}
                    exec(compile(ast.Module(body=[function], type_ignores=[]), str(TRAINER), 'exec'), namespace)
                    before = rng_state()
                    expected = (torch.rand(3), random.random(), np.random.rand(3))
                    restore_rng(before)
                    if fail:
                        with self.assertRaisesRegex(RuntimeError, 'probe failed'):
                            namespace['evaluate_current_state']()
                    else:
                        namespace['evaluate_current_state']()
                    torch.testing.assert_close(torch.rand(3), expected[0], rtol=0, atol=0)
                    self.assertEqual(random.random(), expected[1])
                    np.testing.assert_array_equal(np.random.rand(3), expected[2])
                    directory = next((root / 'evaluation-runs').glob('*' + calls[-1]['evaluation_id']))
                    files = {path.name for path in directory.iterdir()}
                    self.assertEqual(files, {'evaluation.identity.json'} if fail else
                        {'evaluation.identity.json', 'evaluation.complete.json'})
                    self.assertEqual((root / 'latest.json').read_text(), 'original latest checkpoint')
            self.assertEqual(len(list((root / 'evaluation-runs').iterdir())), 2)

    def test_eval_cadence_is_runtime_only_but_objective_identity_stays_strict(self):
        saved = {'schema': 'full-gt-nine-ref-v1-single-target',
                 'training': {'teacher_guidance_scale': 2.5, 'save_every_n_steps': 800}}
        new = {**saved, 'training': {**saved['training'], 'eval_every_n_steps': 400}}
        self.assertEqual(checkpoint_compatibility(saved), checkpoint_compatibility(new))
        new['training']['teacher_guidance_scale'] = 1.
        self.assertNotEqual(checkpoint_compatibility(saved), checkpoint_compatibility(new))


if __name__ == '__main__':
    unittest.main()
