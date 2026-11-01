import copy
from contextlib import ExitStack
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from models.pose.audio import RMSNorm


class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(6, 6)
        self.norm = RMSNorm(6)

    def forward(self, x):
        return self.norm(self.proj(x))


class StreamingPreviewTests(unittest.TestCase):
    def setUp(self):
        previous = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous)

    def test_resident_layers_keep_dtypes_and_refresh_cached_weights(self):
        from bimodal.streaming import ResidentLayers

        torch.manual_seed(923)
        source = []
        for index in range(3):
            block = TinyBlock()
            audio = TinyBlock() if index != 1 else None
            with torch.no_grad():
                block.proj.weight.add_(index * .27)
                block.norm.weight.copy_(torch.arange(6) * .0007 + 1.0003 + index * .0031)
                if audio is not None:
                    audio.norm.weight.copy_(torch.arange(6) * .0009 + 1.0005 + index * .0023)
                    audio.proj.bfloat16()
            source.append(SimpleNamespace(block=block, audio_block=audio, hidden_block=None,
                                          freqs=torch.randn(5, 2, dtype=torch.float32)))
        original = [{role: copy.deepcopy(getattr(layer, role).state_dict()) for role in ('block', 'audio_block')
                     if getattr(layer, role) is not None} for layer in source]
        resident = ResidentLayers(source, torch.device('cpu'))
        self.assertEqual(len(resident), len(source))
        sample = torch.randn(2, 4, 6)
        resident_ids, audio_ids = set(), set()
        with torch.no_grad(), torch.autocast('cpu', dtype=torch.bfloat16, cache_enabled=True):
            expected = []
            x = sample
            for layer in source:
                x = layer.block(x)
                if layer.audio_block is not None:
                    x = x + layer.audio_block(x)
                expected.append(x.clone())
            for _ in range(3):
                outputs = []
                x = sample
                for index, layer in enumerate(resident):
                    self.assertTrue(torch.is_autocast_enabled('cpu'))
                    self.assertFalse(torch.is_autocast_cache_enabled())
                    resident_ids.add(id(layer.block))
                    self.assertIsNone(layer.hidden_block)
                    self.assertEqual(layer.audio_block is None, index == 1)
                    torch.testing.assert_close(layer.freqs, source[index].freqs, rtol=0, atol=0)
                    for role in ('block', 'audio_block'):
                        module = getattr(layer, role)
                        if module is None:
                            continue
                        if role == 'audio_block':
                            audio_ids.add(id(module))
                        for name, value in module.state_dict().items():
                            self.assertEqual(value.dtype, original[index][role][name].dtype)
                            torch.testing.assert_close(value, original[index][role][name], rtol=0, atol=0)
                    x = layer.block(x)
                    if layer.audio_block is not None:
                        x = x + layer.audio_block(x)
                    outputs.append(x)
                self.assertTrue(torch.is_autocast_cache_enabled())
                for actual, reference in zip(outputs, expected):
                    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        self.assertEqual(len(resident_ids), 1)
        self.assertEqual(len(audio_ids), 1)
        for index, layer in enumerate(source):
            for role, state in original[index].items():
                for name, value in getattr(layer, role).state_dict().items():
                    torch.testing.assert_close(value, state[name], rtol=0, atol=0)

    def test_cpu_construction_redirects_only_known_counter(self):
        from bimodal.streaming import CPUConstruction

        with CPUConstruction():
            counter = torch.zeros((), dtype=torch.float64, device='cuda')
            self.assertEqual(counter.device.type, 'cpu')
            with self.assertRaises(RuntimeError):
                torch.zeros(2, device='cuda')
            with self.assertRaises(RuntimeError):
                torch.ones((), device='cuda')

    def test_resolver_uses_retained_full_audio_table_and_reuses_sliding_cache(self):
        from bimodal.audio_tables import resolver_tokens
        from bimodal.cached_resolver import ResolverNet
        from models.pose.wan_universal import InitialLayer

        class RecordingEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.inputs = []
                self.null_token = torch.zeros(1, 1, 1, 1)

            def forward(self, latents):
                self.inputs.append(latents.clone())
                return latents

        encoder = RecordingEncoder()
        initial = SimpleNamespace(audio_encoder=encoder, audio_cross_attention_codes=1, num_clean=3)
        initial._window_audio = lambda *args: InitialLayer._window_audio(initial, *args)
        resolver = SimpleNamespace(initial=initial, mimi=object(), num_codebooks=1, H=2)
        full_codes = torch.arange(900)[None, None]
        retained = full_codes[:, :, :750].clone()
        waveform = full_codes[:, :, :128].clone()
        decoded_tables = []

        def decode(mimi, codes, num_codebooks):
            self.assertIs(mimi, resolver.mimi)
            self.assertEqual(num_codebooks, 1)
            decoded_tables.append(codes.clone())
            return codes.transpose(1, 2).float()

        real_autocast = torch.autocast

        def cpu_autocast(device_type, **kwargs):
            return real_autocast('cpu' if device_type == 'cuda' else device_type, **kwargs)

        with patch('torch.autocast', cpu_autocast), \
                patch('bimodal.audio_tables.audio_latents_from_codes', decode), \
                patch('bimodal.cached_resolver.audio_latents_from_codes', decode):
            cache = {}
            tokens = resolver_tokens(resolver, [retained], waveform, 1, cache)
            torch.testing.assert_close(decoded_tables[-1], retained, rtol=0, atol=0)
            self.assertEqual(encoder.inputs[-1][0, -1, 0].item(), 749)
            for frame in (1, 63):
                self.assertIs(resolver_tokens(resolver, [retained], waveform, frame, cache), tokens)
            self.assertEqual(len(encoder.inputs), 1)
            for frame in (1, 63):
                actual = ResolverNet.audio(resolver, waveform, frame, tokens)
                expected = ResolverNet.audio(resolver, retained, frame)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertEqual(set(cache), {0})
            for frame in (375, 376):
                start = frame - 374
                tokens = resolver_tokens(resolver, [retained], full_codes, frame, cache)
                torch.testing.assert_close(decoded_tables[-1], full_codes[:, :, 2 * start:2 * start + 750],
                                           rtol=0, atol=0)
                self.assertEqual(set(cache), {start})
                count = len(encoder.inputs)
                self.assertIs(resolver_tokens(resolver, [retained], full_codes, frame, cache), tokens)
                self.assertEqual(len(encoder.inputs), count)
                actual = ResolverNet.audio(resolver, full_codes, frame, tokens)
                expected = ResolverNet.audio(resolver, full_codes, frame)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_cached_wavefront_and_resolver_match_streamed_cpu(self):
        from bimodal.cached_resolver import ResolverNet
        from bimodal.sampler import WavefrontNet
        from bimodal.streaming import StreamedResolverNet, StreamedWavefrontNet

        real_autocast = torch.autocast

        def cpu_autocast(device_type, **kwargs):
            if device_type == 'cuda':
                device_type = 'cpu'
                if kwargs.get('dtype') == torch.float32:
                    kwargs.update(dtype=torch.bfloat16, enabled=False)
            return real_autocast(device_type, **kwargs)

        def attention(q, k, v):
            return F.scaled_dot_product_attention(q.transpose(1, 2).float(), k.transpose(1, 2).float(),
                                                v.transpose(1, 2).float()).transpose(1, 2).to(q.dtype)

        with ExitStack() as stack:
            stack.enter_context(patch('torch.autocast', cpu_autocast))
            for module in ('cached_universal', 'bimodal.sampler', 'bimodal.cached_resolver', 'models.pose.audio'):
                stack.enter_context(patch(module + '.flash_attn_func', attention))
            stack.enter_context(torch.no_grad())
            reference = WavefrontNet(tiny_models(2, 4), (1, 2), 2, 'cpu', False, 1, .2)
            streamed = StreamedWavefrontNet(tiny_models(2, 4), (1, 2), 2, 'cpu', .2)
            sink = torch.randn(1, 2, 1, 2)
            for net in (reference, streamed):
                net.pool_k.zero_()
                net.pool_v.zero_()
                net.start_clip(sink, False)
            torch.testing.assert_close(streamed.pool_k, reference.pool_k, rtol=0, atol=0)
            torch.testing.assert_close(streamed.pool_v, reference.pool_v, rtol=0, atol=0)
            for _ in range(2):
                for name in ('wave_latent', 'wave_pose', 'wave_audio'):
                    value = torch.randn_like(getattr(reference, name))
                    getattr(reference, name).copy_(value)
                    getattr(streamed, name).copy_(value)
                for net in (reference, streamed):
                    net.wave_valid.copy_(torch.tensor([False, True, True, False]))
                    net.wave_ints.copy_(torch.tensor([1, 1, 1, 0]))
                torch.testing.assert_close(streamed._wave(), reference._wave(), rtol=0, atol=0)
            streamed.blocks.close()

            reference = ResolverNet(tiny_models(2, 1))
            streamed = StreamedResolverNet(tiny_models(2, 1), 'cpu')
            latents, poses = (torch.randn(1, 2, 3, 1, 2) for _ in range(2))
            ofi = torch.tensor([[0, -1, 1]])
            with real_autocast('cpu', dtype=torch.bfloat16, cache_enabled=True):
                expected_context = reference.context(latents, poses, ofi)
                actual_context = streamed.context(latents, poses, ofi)
                for actual, expected in zip(actual_context, expected_context):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                for _ in range(2):
                    x, pose = (torch.randn(1, 2, 1, 2) for _ in range(2))
                    audio = torch.randn(1, 1, 2, 12)
                    t = torch.tensor(300.)
                    expected = reference.velocity(x, pose, 2, t, expected_context, audio)
                    actual = streamed.velocity(x, pose, 2, t, actual_context, audio)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            streamed.blocks.close()


def tiny_models(history, present):
    from bimodal.streaming import CPUConstruction
    from inference.loader import Layered
    from models.pose.audio import AudioAdapter
    from models.pose.wan_universal import FinalLayer, InitialLayer, TransformerLayer
    from models.wan.model import WanModel

    torch.manual_seed(117)
    transformer = WanModel(patch_size=(1, 1, 1), in_dim=2, out_dim=2, dim=12, ffn_dim=24,
                           freq_dim=8, num_heads=2, num_layers=3).bfloat16()
    encoder = nn.Module()
    encoder.null_token = nn.Parameter(torch.randn(1, 1, 1, 12))
    encoder.no_audio_tokens = nn.Parameter(torch.randn(1, 1, 2, 12))
    adapter = AudioAdapter(hidden_dim=12, num_attn_heads=2, num_layers=3,
                           injection_offset=0, injection_interval=2, initialization='random')
    with torch.no_grad():
        for index, block in enumerate(adapter.blocks):
            block.norm_q.weight.fill_(1.0007 + .0013 * index)
            block.norm_k.weight.fill_(1.0009 + .0011 * index)
    with CPUConstruction():
        initial = InitialLayer(transformer, encoder, nn.Conv3d(2, 12, 1), nn.Parameter(torch.randn(12)),
                               history, present, 1, 0., False)
    layers = [initial]
    layers += [TransformerLayer(block, adapter.get_block(i), transformer.freqs, history, present, False, False)
               for i, block in enumerate(transformer.blocks)]
    layers += [FinalLayer(transformer)]
    return SimpleNamespace(net=Layered(layers).eval(), mimi=nn.Module(), num_history=history,
                           num_present=present, num_codebooks=1)


if __name__ == '__main__':
    unittest.main()
