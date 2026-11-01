import ast
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from distillation.batched_training import (concatenate_batch_trees, generate_pooled_targets,
    objective_checkpoint, regroup_prepared_rows, slice_batch_tree)
from distillation.full_gt_dmd import FullGTStabilizer, generate_resolver


def rows(ids):
    value = torch.tensor(ids, dtype=torch.float32)[:, None]
    return {'latents': value, 'audio': value + 100, 'weight': torch.ones(len(ids)),
            'history_t': .2, 'refs': value + 200, 'ref_keep': value > 0}


class BatchedTrainingTests(unittest.TestCase):
    def test_real_full_gt_generation_keeps_detached_pool_and_differentiable_student(self):
        class Net(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.tensor(.1))
                self.reference = None

            @contextmanager
            def references(self, refs, keep):
                self.reference = self.scale * (refs * keep[:, None, :, None, None]).mean(2)
                try:
                    yield
                finally:
                    self.reference = None

            def audio_tokens(self, audio):
                return audio

            def frame_audio(self, tokens, ofi, drop):
                return tokens

            def forward_pass(self, frames, plan, ext, audio, audio_mask, heads):
                return torch.stack([self.scale * frame[0] + self.reference
                                    for frame in frames[:heads]], dim=2), [], []

        net, rollout = Net(), FullGTStabilizer(history_dropout=(0., 0., 0., 0.), trim=False, cache_context=False)
        generator = torch.Generator().manual_seed(39)
        def make_rows():
            value = torch.randn(4, 1, 8, 2, 2, generator=generator)
            return {'latents': value, 'pose': value + 1,
                    'ofi': torch.arange(8)[None].expand(4, -1),
                    'refs': torch.randn(4, 1, 9, 2, 2, generator=generator),
                    'ref_keep': torch.ones(4, 9, dtype=torch.bool), 'drop': torch.zeros(4, dtype=torch.bool),
                    'audio': torch.zeros(4, 2, 3), 'weight': torch.ones(4),
                    'generation_noise': torch.randn(value.shape, generator=generator)}
        groups = [[(kind, make_rows()) for kind in ('stabilizer', 'resolver')] for _ in range(4)]
        def generate(batch, kind):
            noise, context = rollout.prepare_context(batch, noise=batch['generation_noise'])
            if kind == 'stabilizer':
                return rollout.generate(net, batch, noise, context)
            return generate_resolver(net, batch, context, noise[:, :, -1:])
        pools = generate_pooled_targets(groups, generate, 16)
        for group, pool in zip(groups, pools):
            for (kind, batch), (y, context) in zip(group, pool):
                self.assertIsNone(y.grad_fn)
                self.assertFalse(y.requires_grad)
                self.assertTrue(all(value.grad_fn is None and not value.requires_grad for value in context.values()))
                torch.testing.assert_close(context['history'], batch['latents'][:, :, 1:7])
                torch.testing.assert_close(context['noise'], batch['generation_noise'][:, :, 1:7])
                direct, _ = generate(batch, kind)
                torch.testing.assert_close(y, direct.detach())
                self.assertTrue(direct.requires_grad)
                direct.square().mean().backward()
                self.assertIsNone(net.reference)
        self.assertGreater(float(net.scale.grad.abs()), 0)

    def test_pool_keeps_four_distinct_updates_and_every_context_field(self):
        groups = [[(kind, rows(range(update * 4, update * 4 + 4)))
                   for kind in ('stabilizer', 'resolver')] for update in range(4)]
        calls = []

        def generate(batch, kind):
            self.assertFalse(torch.is_grad_enabled())
            calls.append((kind, len(batch['latents'])))
            x = batch['latents']
            return x * 2, {'history': x + 3, 'noise': x - 7, 'keep': x > 2,
                           'refs': batch['refs'], 'drop': x[:, 0] > 8}

        pools = generate_pooled_targets(groups, generate, 16)
        self.assertEqual(calls, [('stabilizer', 16), ('resolver', 16)])
        for group, pool in zip(groups, pools):
            for (_, batch), (y, context) in zip(group, pool):
                torch.testing.assert_close(y, batch['latents'] * 2)
                torch.testing.assert_close(context['history'], batch['latents'] + 3)
                torch.testing.assert_close(context['noise'], batch['latents'] - 7)
                torch.testing.assert_close(context['keep'], batch['latents'] > 2)
                torch.testing.assert_close(context['refs'], batch['refs'])
                torch.testing.assert_close(context['drop'], batch['latents'][:, 0] > 8)

    def test_pool_cap_and_prepared_audio_regroup_preserve_row_order(self):
        kinds = ['stabilizer'] * 2 + ['resolver'] * 2
        raw = [{'objective': kind} for kind in kinds]
        original = [rows([index * 2, index * 2 + 1]) for index in range(4)]
        original[2]['audio'].fill_(72)
        original[3]['audio'].fill_(91)
        prepared = regroup_prepared_rows(raw, original, {'stabilizer': 2, 'resolver': 4})
        self.assertEqual([len(item['latents']) for item in prepared], [2, 2, 4])
        self.assertEqual(prepared[-1]['audio'].flatten().tolist(), [72, 72, 91, 91])
        calls = []
        groups = [[('stabilizer', prepared[0]), ('stabilizer', prepared[1]), ('resolver', prepared[2])]]
        result = generate_pooled_targets(groups, lambda batch, kind:
            (calls.append((kind, len(batch['latents']))) or batch['latents'], None), 2)
        self.assertEqual(calls, [('stabilizer', 2), ('stabilizer', 2), ('resolver', 2), ('resolver', 2)])
        torch.testing.assert_close(result[0][-1][0], original[2]['latents'].new_tensor([[4], [5], [6], [7]]))

    def test_checkpoint_policy_restores_shared_compiled_callable_on_exception(self):
        fn = lambda: None
        net = SimpleNamespace(checkpoint='block', block_fn=fn)
        with self.assertRaisesRegex(RuntimeError, 'test'):
            with objective_checkpoint(net, 'none'):
                self.assertEqual(net.checkpoint, 'none')
                self.assertIs(net.block_fn, fn)
                raise RuntimeError('test')
        self.assertEqual(net.checkpoint, 'block')
        self.assertIs(net.block_fn, fn)

    def test_trainer_four_sequential_updates_then_student_objective_microbatches(self):
        # Execute the actual nested training orchestration with tiny CPU
        # objectives: this exercises the production call order and regrouping,
        # while model, metrics, timings and distributed reduction are replaced.
        source = ast.parse((Path(__file__).resolve().parents[1] / 'learn_dmd_joint.py').read_text())
        main = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
        fn = next(node for node in main.body if isinstance(node, ast.FunctionDef) and node.name == 'train_round')
        events = []
        kinds = ['stabilizer', 'stabilizer', 'resolver', 'resolver']
        def raw_group(update):
            return [{'objective': kind, 'global_valid_rows': 160,
                     'global_objective_rows': {'stabilizer': 160, 'resolver': 160},
                     'rows': rows([update * 4 + (index % 2) * 2 + lane for lane in range(2)])}
                    for index, kind in enumerate(kinds)]
        def regroup(raw, prepared, role):
            sizes = {'stabilizer': 2 if role == 'student' else 4, 'resolver': 4}
            output = regroup_prepared_rows(raw, prepared, sizes)
            selected = [raw[0], raw[1], raw[2]] if role == 'student' else [raw[0], raw[2]]
            return selected, output
        def generate(batch, kind, critic_pool=False):
            events.append(('pool' if critic_pool else 'student_forward', kind, len(batch['latents'])))
            return batch['latents'] * 2, {'noise': batch['latents'] + 2}
        def critic_update(raw, prepared, pool):
            events.append(('critic_update', int(prepared[0]['latents'][0, 0])))
            for batch, (y, context) in zip(prepared, pool):
                torch.testing.assert_close(y, batch['latents'] * 2)
                torch.testing.assert_close(context['noise'], batch['latents'] + 2)
        nets = {role: SimpleNamespace(checkpoint='block') for role in ('student', 'teacher', 'critic')}
        score_roles, observations = [], []
        def score(net, batch, kind, x, times, context, **kwargs):
            role = next(role for role, candidate in nets.items() if candidate is net)
            self.assertNotEqual(role, 'student', 'Diagnostics must not score the student again')
            score_roles.append(role)
            return x
        ns = dict(full_gt=True, data=SimpleNamespace(objective_schedule=kinds),
                  config={'critic_updates': 4, 'normalizer_epsilon': 1e-8},
                  prepare_group=lambda raw: [item['rows'] for item in raw], regroup=regroup,
                  generate_pooled_targets=generate_pooled_targets, generate=generate,
                  rollout_pool_rows=16, critic_update=critic_update, timed=lambda name: nullcontext(),
                  torch=torch, gradients={'student': SimpleNamespace(allocate=lambda: events.append(('allocate',)))},
                  objective_checkpoint=objective_checkpoint, checkpoint_for=lambda role, kind: 'none', nets=nets,
                  corruption=lambda y, kind: (y, torch.zeros(len(y), 1), y, y),
                  score=score,
                  dmd_frames=lambda y, *args: y, add_dmd_diagnostics=lambda *args: None,
                  sums=None, training_metrics=SimpleNamespace(add_dmd=lambda *args: observations.append(args)),
                  backward=lambda frames, batch, global_rows, kind, metric:
                      events.append(('backward', kind, len(batch['latents']), global_rows)),
                  update=lambda role: events.append(('update', role)), counters={'student_rows_seen': 0})
        exec(compile(ast.Module(body=[fn], type_ignores=[]), '<train-round>', 'exec'), ns)
        ns['train_round'](raw_group(9), False, [raw_group(index) for index in range(4)])
        self.assertEqual(events[:6], [('pool', 'stabilizer', 16), ('pool', 'resolver', 16),
                                     ('critic_update', 0), ('critic_update', 4),
                                     ('critic_update', 8), ('critic_update', 12)])
        self.assertEqual([event for event in events if event[0] == 'backward'],
                         [('backward', 'stabilizer', 2, 160), ('backward', 'stabilizer', 2, 160),
                          ('backward', 'resolver', 4, 160)])
        self.assertEqual(events[-1], ('update', 'student'))
        self.assertEqual(ns['counters']['student_rows_seen'], 160)
        self.assertEqual(score_roles, ['teacher', 'critic'] * 3)
        self.assertEqual(len(observations), 3)


if __name__ == '__main__':
    unittest.main()
