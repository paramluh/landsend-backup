"""CPU controls for paired endpoint sensitivity and generated-context replay."""
from contextlib import contextmanager
import copy
import json
import random
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from distillation.endpoint_sensitivity import (CATEGORIES, _eager_blocks, _wavefront,
                                               run_endpoint_sensitivity, render_endpoint_sensitivity)
from distillation.joint_dmd import generate_stabilizer


class LinearContextNet(torch.nn.Module):
    """Simple causal query with a reference cache rebuilt only on context entry."""
    def __init__(self, response=True):
        super().__init__()
        self.child = torch.nn.Linear(1, 1)
        self.shards = None
        self.response = response
        self.reference_builds = 0
        self.reference = None

    @contextmanager
    def references(self, refs, keep):
        previous = self.reference
        self.reference_builds += 1
        coefficients = torch.arange(1, 10, dtype=torch.float32)[None, None, :, None, None]
        self.reference = (refs * keep[:, None, :, None, None] * coefficients).sum(2).clone()
        try:
            yield self
        finally:
            self.reference = previous

    def audio_tokens(self, audio):
        return audio

    def frame_audio(self, tokens, ofi, drop):
        return torch.zeros((tokens.shape[0], ofi.shape[1], 1, 1))

    def forward_pass(self, frames, plan, ext, audio, audio_mask, heads, **kwargs):
        assert ext is None
        # Deliberately consume all RNGs: the experiment must restore them.
        torch.rand(())
        random.random()
        np.random.random()
        outputs = []
        for slot in range(heads):
            specs = next(specs[slot - start] for start, specs in plan if start <= slot < start + len(specs))
            sink = frames[specs[0][1]][0]
            previous = [frames[entry[1]][0] for entry in specs[1:-1]]
            output = self.reference + .7 * sink + .5 * sum(previous)
            outputs.append(output if self.response else torch.zeros_like(output))
        return torch.stack(outputs, 2), [], []


class CachedToyNet(LinearContextNet):
    """Two causal layers supporting the production generator's external K/V."""
    def __init__(self):
        super().__init__()
        self.transformer = SimpleNamespace(blocks=[None, None])

    def forward_pass(self, frames, plan, ext, audio, audio_mask, heads, keep=(), **kwargs):
        values = []
        for value, pose, time, slot, valid in frames:
            if not torch.is_tensor(valid):
                valid = torch.full((value.shape[0],), valid)
            values.append(torch.where(valid[:, None, None, None], value, torch.full_like(value, .3)))
        keys, vals = [], []
        for layer in range(2):
            raw = [value + .001 * layer for value in values]
            keys.append(torch.stack([raw[index] for index in keep], 1) if keep else None)
            vals.append(torch.stack([raw[index] for index in keep], 1) if keep else None)
            output = list(values)
            for start, specs in plan:
                for offset, spec in enumerate(specs):
                    sources = [raw[entry[1]] if entry[0] in ('f', 'r') else ext[layer][2 * entry[1] + 1]
                               for entry in spec]
                    index = start + offset
                    reference = self.reference if frames[index][3] else torch.zeros_like(self.reference)
                    output[index] = values[index] + .02 * (sum(sources) / len(sources) + reference)
            values = output
        velocity = torch.stack(values[:heads], 2) if heads else None
        return velocity, keys, vals


def rows():
    generator = torch.Generator().manual_seed(77)
    latent = torch.randn(2, 1, 11, 2, 3, generator=generator) * .01
    return {'latents': latent, 'pose': torch.zeros_like(latent),
            'ofi': torch.arange(11)[None].repeat(2, 1),
            'refs': torch.randn(2, 1, 9, 2, 3, generator=generator) * .01,
            'ref_keep': torch.tensor([[True, True, False, False, False, False, False, False, False],
                                      [True] * 9]),
            'drop': torch.tensor([False, True]), 'audio': torch.zeros(2, 12, 1),
            'sample_weight': torch.tensor([1., 2.])}


CONFIG = {'band_steps': [1, 1, 1, 1], 'sensitivity_seeds': [1729], 'sensitivity_rms': .01}


class EndpointSensitivityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_zero_response_and_slot_order(self):
        result = run_endpoint_sensitivity(LinearContextNet(False), rows(), CONFIG)
        self.assertEqual([(p['objective'], p['slot']) for p in result['panels']],
                         [('stabilizer', i) for i in range(4)] + [('resolver', 0)])
        self.assertEqual([p['remaining_steps'] for p in result['panels']], [1, 2, 3, 4, 1])
        for panel in result['panels']:
            self.assertEqual(panel['categories'], CATEGORIES[panel['objective']])
            for summary in panel['summary'].values():
                self.assertEqual(summary['mean'], 0.)
                self.assertEqual(summary['count'], 2)
            self.assertEqual(panel['records'][0]['baseline_replay_error_rms_per_row'], [0., 0.])
        json.dumps(result, allow_nan=False)

    def test_all_references_average_within_sample_and_fresh_cache(self):
        net = LinearContextNet()
        result = run_endpoint_sensitivity(net, rows(), CONFIG)
        panel = result['panels'][-1]
        references = [case for case in panel['records'][0]['cases'] if case['kind'] == 'reference']
        self.assertEqual([case['index'] for case in references], list(range(9)))
        # Resolver dt=.2; sample means over coefficient sets {1,2} and {1,...,9}.
        expected = (.2 * 1.5 * 1 + .2 * 5 * 2) / 3
        self.assertAlmostEqual(panel['summary']['reference bank mean']['mean'], expected, places=4)
        self.assertEqual(panel['summary']['reference bank mean']['count'], 2)
        self.assertIsNone(references[2]['paired_gain_per_row'][0])
        self.assertGreater(net.reference_builds, 100)
        self.assertIsNone(net.reference)

    def test_generated_context_never_reads_temporal_gt(self):
        source = rows()
        changed = copy.deepcopy(source)
        changed['latents'][:, :, 1:] = 1e8
        first = run_endpoint_sensitivity(LinearContextNet(), source, CONFIG)
        second = run_endpoint_sensitivity(LinearContextNet(), changed, CONFIG)
        self.assertEqual(first, second)
        for panel in first['panels']:
            self.assertTrue(all(error < 1e-6 for error in panel['records'][0]['baseline_replay_error_rms_per_row']))
        # Endpoint integration recomputes target-only, without AR multiplication.
        for panel in first['panels']:
            duration = panel['start_t'] - panel['endpoint_t']
            self.assertAlmostEqual(panel['summary']['previous 1']['mean'], duration * .5, places=4)
            self.assertAlmostEqual(panel['summary']['sink']['mean'], duration * .7, places=4)

    def test_baseline_matches_cached_production_wavefront(self):
        source, net, seed = rows(), CachedToyNet(), 123
        noise = torch.randn(2, 1, 10, 2, 3, generator=torch.Generator().manual_seed(seed))
        _, completed = _wavefront(net, source, seed, 2)
        actual, context = generate_stabilizer(net, source, [2] * 4, noise=noise, num_targets=4,
                                              prefix_frames=6, history_dropout=(), static_layout=False)
        expected = torch.stack([completed[index] for index in range(7, 11)], 2)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-6),
                        (actual - expected).abs().max())
        self.assertTrue(torch.allclose(context['history'], torch.stack([completed[index] for index in (4, 5, 6)], 2),
                                       atol=1e-6, rtol=1e-6))

    def test_actual_injected_rms_and_paired_signs(self):
        source = rows()
        source['refs'].fill_(1e4)
        result = run_endpoint_sensitivity(LinearContextNet(), source, CONFIG)
        case = next(case for case in result['panels'][-1]['records'][0]['cases'] if case['kind'] == 'reference')
        self.assertEqual([trial['sign'] for trial in case['trials']], [1, -1])
        actual = case['trials'][0]['input_rms_per_row'][0]
        self.assertNotAlmostEqual(actual, CONFIG['sensitivity_rms'], places=7)
        self.assertEqual(case['trials'][0]['input_rms_per_query'][0], [actual])

    def test_rng_modes_and_half_amplitude_validation(self):
        net = LinearContextNet()
        net.train()
        net.child.eval()
        torch.manual_seed(123)
        random.seed(456)
        np.random.seed(789)
        torch_state, py_state, np_state = torch.get_rng_state().clone(), random.getstate(), np.random.get_state()
        result = run_endpoint_sensitivity(net, rows(), dict(CONFIG, sensitivity_validate_half_rms=True))
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))
        self.assertEqual(random.getstate(), py_state)
        self.assertEqual(np.random.get_state()[0], np_state[0])
        self.assertTrue(np.array_equal(np.random.get_state()[1], np_state[1]))
        self.assertEqual(np.random.get_state()[2:], np_state[2:])
        self.assertTrue(net.training)
        self.assertFalse(net.child.training)
        half = result['half_amplitude_validation']
        self.assertEqual(half['requested_input_rms'], .005)
        for full_panel, half_panel in zip(result['panels'], half['panels']):
            for category in full_panel['categories']:
                self.assertAlmostEqual(full_panel['summary'][category]['mean'],
                                       half_panel['summary'][category]['mean'], places=4)

    def test_missing_references_and_figure_keys(self):
        source = rows()
        source['ref_keep'].zero_()
        result = run_endpoint_sensitivity(LinearContextNet(False), source, CONFIG)
        for panel in result['panels']:
            self.assertIsNone(panel['summary']['reference bank mean']['mean'])
            self.assertEqual(panel['summary']['reference bank mean']['count'], 0)
        with tempfile.TemporaryDirectory() as directory:
            paths = render_endpoint_sensitivity(result, directory)
            self.assertEqual(set(paths), {'stabilizer_context_gain', 'resolver_context_gain'})
            self.assertTrue(all(path.stat().st_size > 0 for path in paths.values()))

    def test_eager_block_dispatch_restored_on_success_and_exception(self):
        net = LinearContextNet(False)
        compiled, eager = object(), object()
        net.block_fn, net._block = compiled, eager
        with _eager_blocks(net):
            self.assertIs(net.block_fn, eager)
        self.assertIs(net.block_fn, compiled)
        with self.assertRaisesRegex(RuntimeError, 'probe failed'):
            with _eager_blocks(net):
                self.assertIs(net.block_fn, eager)
                raise RuntimeError('probe failed')
        self.assertIs(net.block_fn, compiled)
        run_endpoint_sensitivity(net, rows(), CONFIG)
        self.assertIs(net.block_fn, compiled)


if __name__ == '__main__':
    unittest.main()
