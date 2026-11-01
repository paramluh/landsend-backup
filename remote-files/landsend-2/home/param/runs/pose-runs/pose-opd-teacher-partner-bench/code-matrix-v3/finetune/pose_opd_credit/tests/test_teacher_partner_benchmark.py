"""Candidate selection and weight ownership across actor-cache replacement."""
from contextlib import nullcontext, redirect_stderr
import io
import types
import unittest
from unittest.mock import patch
import weakref

import torch

from finetune.pose_opd_credit.teacher_partner_benchmark import (
    _release_engines, _storage_identity, arguments,
)


class BenchmarkCandidatesTests(unittest.TestCase):
    def test_default_and_multi_candidate_cli(self):
        required = ['--config', '/tmp/config.json', '--output', '/tmp/benchmark']
        one = arguments(required + ['--instances', '8', '--teacher-replay', 'block'])
        self.assertEqual(one.batch_sizes, [8])
        self.assertEqual(one.replay_backends, ['block'])
        many = arguments(required + ['--batch-sizes', '2,16,32', '--replay-backends', 'sequential,block'])
        self.assertEqual(many.batch_sizes, [2, 16, 32])
        self.assertEqual(many.replay_backends, ['sequential', 'block'])
        for options in (['--batch-sizes', '2,2'], ['--batch-sizes', '65'],
                        ['--replay-backends', 'block,invalid']):
            with self.assertRaises(SystemExit), redirect_stderr(io.StringIO()):
                arguments(required + options)

    def test_release_drops_graph_cycles_and_hooks_but_preserves_weight_storage(self):
        model = torch.nn.Module()
        model.inp = torch.nn.Linear(4, 4)
        baseline = model.inp.register_forward_hook(lambda *_: None)
        initial_hooks = set(model.inp._forward_hooks)
        before = _storage_identity(model)
        closed = []

        class Engine:
            def __init__(self):
                self.model = model
                self.hook = model.inp.register_forward_hook(self.on_input)
                self.graphs = {'graph': object()}
                self._opd_partner_blocks = {6: types.SimpleNamespace(engine=self, graphs={'block': object()})}
            def on_input(self, *_):
                pass
            def close(self):
                closed.append(True)
                self.hook.remove()

        actor = Engine()
        reference = weakref.ref(actor)
        engines = {'student': actor}
        del actor
        loaded = {'student': {'model': model, 'initial_hook_ids': initial_hooks, 'storage_identity': before}}
        with patch('finetune.pose_opd_credit.teacher_partner_benchmark._sync'), \
                patch('torch.cuda.device', lambda *_: nullcontext()), patch('torch.cuda.empty_cache'):
            _release_engines(torch, engines, loaded, [torch.device('cpu')])
        self.assertEqual(closed, [True])
        self.assertEqual(engines, {})
        self.assertIsNone(reference())
        self.assertEqual(_storage_identity(model), before)
        self.assertEqual(set(model.inp._forward_hooks), initial_hooks)
        baseline.remove()

    def test_partial_constructor_hook_is_released_after_oom(self):
        model = torch.nn.Module()
        model.inp = torch.nn.Linear(4, 4)
        loaded = {'student': {'model': model, 'initial_hook_ids': set(),
                              'storage_identity': _storage_identity(model)}}
        # Simulate a constructor that registered its hook before failing.
        model.inp.register_forward_hook(lambda *_: None)
        with patch('finetune.pose_opd_credit.teacher_partner_benchmark._sync'), \
                patch('torch.cuda.device', lambda *_: nullcontext()), patch('torch.cuda.empty_cache'):
            _release_engines(torch, {}, loaded, [torch.device('cpu')])
        self.assertEqual(len(model.inp._forward_hooks), 0)


if __name__ == '__main__':
    unittest.main()
