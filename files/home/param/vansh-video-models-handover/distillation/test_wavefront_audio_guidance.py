import copy
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from bimodal.sampler import WavefrontNet, generate_wavefront
from inference.loader import Layered
from models.pose.audio import AudioAdapter
from models.pose.wan_universal import FinalLayer, InitialLayer, TransformerLayer
from models.wan.model import WanModel


class TokenEncoder(nn.Module):
    def __init__(self, dimension, tokens):
        super().__init__()
        self.null_token = nn.Parameter(torch.randn(1, 1, 1, dimension))
        self.no_audio_tokens = nn.Parameter(torch.randn(1, 1, tokens, dimension))

    def forward(self, tokens):
        return tokens


def small_models(num_refs, history=3, present=4):
    dimension = 32
    transformer = WanModel(patch_size=(1, 1, 1), in_dim=2, out_dim=2, dim=dimension, ffn_dim=64,
                           freq_dim=8, num_heads=2, num_layers=3).cuda().bfloat16()
    encoder = TokenEncoder(dimension, 3).cuda()
    adapter = AudioAdapter(hidden_dim=dimension, num_attn_heads=2, num_layers=3,
                           injection_offset=0, injection_interval=1, initialization='random').cuda()
    initial = InitialLayer(
        transformer, encoder, nn.Conv3d(2, dimension, 1).cuda(), nn.Parameter(torch.randn(dimension, device='cuda')),
        num_history=history, num_present=present, audio_cross_attention_codes=2, audio_dropout_p=0.0,
        compile_block=False, num_refs=num_refs,
        ref_patch_embedding=copy.deepcopy(transformer.patch_embedding) if num_refs else None,
    )
    layers = [initial]
    for index, block in enumerate(transformer.blocks):
        layers.append(TransformerLayer(block, adapter.get_block(index), transformer.freqs.cuda(), history, present,
                                       compile_block=False, checkpoint=False, num_refs=num_refs))
    layers.append(FinalLayer(transformer, num_refs=num_refs))
    return SimpleNamespace(net=Layered(layers).cuda().eval().requires_grad_(False), mimi=None,
                           num_history=history, num_present=present, num_codebooks=1)


@unittest.skipUnless(torch.cuda.is_available(), 'wavefront inference requires CUDA')
class WavefrontAudioGuidanceTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(734)
        self.gradient_mode = torch.no_grad()
        self.gradient_mode.__enter__()
        self.addCleanup(self.gradient_mode.__exit__, None, None, None)

    def make_wave(self, num_refs=2, batch=2, guidance_scale=1.0, compile=False, cuda_graph=False,
                  guidance_telemetry=False):
        models = small_models(num_refs)
        cached = WavefrontNet(models, (2, 2), 2, 'cuda', compile, batch, .2, guidance_scale=guidance_scale,
                              guidance_telemetry=guidance_telemetry)
        if num_refs:
            cached.refs = (torch.randn(batch, num_refs, 2, 2, 2, device='cuda'),
                           torch.ones(batch, num_refs, dtype=torch.bool, device='cuda'))
            cached.refs[1][0, -1] = False
        sink = torch.randn(batch, 2, 2, 2, device='cuda')
        cached.start_clip(sink, cuda_graph)
        cached.wave_latent.normal_()
        cached.wave_pose.normal_()
        cached.wave_valid.fill_(True)
        cached.wave_ints.copy_(torch.tensor(
            [*cached.fixed, *range(cached.pad_history0, cached.pad_history0 + cached.H), 1], device='cuda'))
        tokens = torch.randn(batch, cached.P * cached.K, cached.initial.dim, device='cuda')
        indices = torch.tensor([[0, *([-1] * cached.H), *range(1, cached.P + 1)]] * batch, device='cuda')
        cached.wave_audio.copy_(cached.initial._window_audio(tokens, indices, torch.arange(batch, device='cuda')))
        return models, cached, sink, tokens, indices

    def test_reference_cache_matches_full_window_with_training_audio_dropout(self):
        for num_refs in (0, 2, 9):
            with self.subTest(num_refs=num_refs):
                models, cached, sink, tokens, indices = self.make_wave(num_refs=num_refs)
                batch = cached.B
                context = torch.zeros(batch, 2, 1 + cached.H, 2, 2, device='cuda')
                context[:, :, 0] = sink
                latent = torch.cat((context, cached.wave_latent), dim=2)
                pose = torch.cat((torch.zeros_like(context), cached.wave_pose), dim=2)
                schedule = torch.zeros(batch, 1 + cached.H + cached.P, device='cuda')
                schedule[:, -cached.P:] = cached.t_table[cached.slot_steps + 1]
                inputs = (latent, schedule, tokens, pose, indices, torch.arange(batch, device='cuda'),
                          cached.wave_valid[None].expand(batch, -1))
                if num_refs:
                    inputs += (cached.refs[0].transpose(1, 2), cached.refs[1])
                conditioned = models.net(inputs)[:, :, -cached.P:].float()
                cached.initial.training, cached.initial.audio_dropout_p = True, 1.0
                unconditioned = models.net(inputs)[:, :, -cached.P:].float()
                cached.initial.training, cached.initial.audio_dropout_p = False, 0.0
                expected = conditioned.clone()
                for scale in (1.0, 1.5, 2.0, 2.5, 3.0, 5.0):
                    cached.guidance_scale = scale
                    expected[:, :, -1:] = unconditioned[:, :, -1:] + scale * (
                        conditioned[:, :, -1:] - unconditioned[:, :, -1:])
                    torch.testing.assert_close(cached._wave(), expected, rtol=.025, atol=.025)

    def test_guidance_only_changes_highest_noise_slot_and_keeps_cache(self):
        _, cached, _, _, _ = self.make_wave()
        conditioned = cached._wave()
        audio = cached.wave_audio.clone()
        cached.wave_audio.copy_(cached.initial.audio_encoder.no_audio_tokens.expand_as(cached.wave_audio))
        unconditioned = cached._wave()
        cached.wave_audio.copy_(audio)
        keys, values = cached.pool_k.clone(), cached.pool_v.clone()
        for scale in (0.0, 1.5, 2.0, 2.5, 3.0, 5.0):
            cached.guidance_scale = scale
            guided = cached._wave()
            torch.testing.assert_close(guided[:, :, :-1], conditioned[:, :, :-1], rtol=.01, atol=.01)
            expected = unconditioned[:, :, -1] + scale * (conditioned[:, :, -1] - unconditioned[:, :, -1])
            torch.testing.assert_close(guided[:, :, -1], expected, rtol=.025, atol=.025)
        torch.testing.assert_close(cached.wave_audio, audio, rtol=0, atol=0)
        torch.testing.assert_close(cached.pool_k, keys, rtol=0, atol=0, equal_nan=True)
        torch.testing.assert_close(cached.pool_v, values, rtol=0, atol=0, equal_nan=True)

    def test_guidance_does_not_normalize_or_rescale_velocity(self):
        _, cached, _, _, _ = self.make_wave(guidance_scale=1.5, guidance_telemetry=True)
        conditioned = torch.ones_like(cached.wave_latent)
        conditioned[:, 1].fill_(2)
        unconditioned = torch.ones_like(conditioned)
        unconditioned[:, 0].fill_(-3)
        with patch.object(cached.final, 'unpatchify', return_value=torch.cat((conditioned, unconditioned))):
            for scale in (1.5, 2.0, 2.5, 3.0, 5.0):
                with self.subTest(scale=scale):
                    cached.guidance_scale = scale
                    expected = conditioned.clone()
                    expected[:, :, -1:] = unconditioned[:, :, -1:] + scale * (
                        conditioned[:, :, -1:] - unconditioned[:, :, -1:])
                    guided = cached._wave()
                    torch.testing.assert_close(guided, expected, rtol=0, atol=0)
                    guided_norm = torch.linalg.vector_norm(guided[:, :, -1].flatten(1), dim=1)
                    conditioned_norm = torch.linalg.vector_norm(conditioned[:, :, -1].flatten(1), dim=1)
                    self.assertTrue(torch.all(guided_norm > conditioned_norm).item())
                    guided_squared = (4 * scale - 3) ** 2 + (scale + 1) ** 2
                    expected_metrics = torch.tensor([
                        math.sqrt(20), math.sqrt(40), math.sqrt(4 * guided_squared),
                        math.sqrt(68) * (scale - 1), (6 * scale - 1) / math.sqrt(5 * guided_squared),
                    ], device='cuda').expand(cached.B, -1)
                    torch.testing.assert_close(cached.guidance_metrics, expected_metrics)

    def test_telemetry_does_not_change_velocity(self):
        _, cached, _, _, _ = self.make_wave(guidance_scale=1.5, guidance_telemetry=True)
        measured = cached._wave()
        self.assertTrue(torch.isfinite(cached.guidance_metrics).all().item())
        cached.guidance_metrics = None
        unmeasured = cached._wave()
        torch.testing.assert_close(measured, unmeasured, rtol=0, atol=0)

    def test_compiled_cuda_graph_refreshes_references_between_clips(self):
        _, cached, sink, _, _ = self.make_wave(batch=1, guidance_scale=2.5, compile=True, cuda_graph=True,
                                               guidance_telemetry=True)
        self.assertIsNone(cached.guidance_trace)
        self.assertEqual(cached.guidance_trace_count, 0)
        state = cached.wave_latent.clone()
        expected = state + cached._wave() * cached.dt
        expected_metrics = cached.guidance_metrics.clone()
        cached.step()
        torch.testing.assert_close(cached.wave_latent, expected, rtol=.015, atol=.015)
        torch.testing.assert_close(cached.guidance_metrics, expected_metrics, rtol=.015, atol=.015)
        graph = cached.graphs[0]
        previous = expected.clone()
        cached.refs[0].normal_(mean=3.0)
        cached.refs[1].fill_(True)
        cached.start_clip(sink, True)
        self.assertIs(cached.graphs[0], graph)
        cached.wave_latent.copy_(state)
        expected = state + cached._wave() * cached.dt
        expected_metrics = cached.guidance_metrics.clone()
        self.assertGreater(float((expected - previous).abs().max()), 1e-4)
        cached.step()
        torch.testing.assert_close(cached.wave_latent, expected, rtol=.015, atol=.015)
        torch.testing.assert_close(cached.guidance_metrics, expected_metrics, rtol=.015, atol=.015)

    def test_guidance_trace_excludes_warmup_drain_and_padded_batch(self):
        _, cached, _, _, _ = self.make_wave(num_refs=0, batch=2, guidance_scale=1.5, cuda_graph=True,
                                            guidance_telemetry=True)
        self.assertIsNone(cached.guidance_trace)
        for length in (4, 2):
            with self.subTest(length=length):
                latents = torch.randn(2, length, 2, 2, device='cuda')
                poses = torch.randn_like(latents)
                tokens = torch.zeros(1, length * cached.K, cached.initial.dim, device='cuda')
                codes = torch.zeros(1, 2 * length, device='cuda', dtype=torch.long)
                with patch.object(cached, 'set_wave_audio', lambda *args: cached.wave_audio.zero_()):
                    outputs = generate_wavefront(cached, [(latents, poses, tokens, codes)], [17], 'cuda', True)
                self.assertEqual(len(outputs), 1)
                self.assertEqual(cached.guidance_trace.shape, (length - 1, cached.m, 1, 5))
                self.assertEqual(cached.guidance_trace_count, (length - 1) * cached.m)
                self.assertTrue(torch.isfinite(cached.guidance_trace).all().item())
                torch.testing.assert_close(cached.guidance_trace_timesteps, torch.tensor([1.0, .9], device='cuda'))

    def test_guided_rollout_matches_full_windows_through_startup_history_and_drain(self):
        models, cached, _, _, _ = self.make_wave(num_refs=9, batch=1, guidance_scale=2.5)
        length, seed = 8, 167
        latents = torch.randn(2, length, 2, 2, device='cuda')
        poses = torch.randn_like(latents)
        tokens = torch.randn(1, length * cached.K, cached.initial.dim, device='cuda')
        codes = torch.zeros(1, 2 * length, device='cuda', dtype=torch.long)

        def set_audio(code_tables, audio_codes, last_frame, frames):
            positions = cached.K * (frames[:, None] - 1).clamp(min=0) + cached.token_offsets
            null = cached.initial.audio_encoder.null_token.expand(1, cached.P, 1, -1)
            cached.wave_audio.copy_(torch.cat((tokens[:, positions], null), dim=2))

        with patch.object(cached, 'set_wave_audio', set_audio):
            actual = generate_wavefront(cached, [(latents, poses, tokens, codes)], [seed], 'cuda', False)[0]
        noise = torch.randn(latents.shape, device='cuda', generator=torch.Generator(device='cuda').manual_seed(seed))
        clean = [latents[:, 0]]
        wave = torch.zeros_like(cached.wave_latent)
        for round_index in range(2 - cached.P, length):
            entering = round_index + cached.P - 1
            next_state = noise[:, entering:entering + 1][None] if entering < length else torch.zeros_like(wave[:, :, :1])
            wave = torch.cat((wave[:, :, 1:], next_state), dim=2)
            present = list(range(round_index, round_index + cached.P))
            history = list(range(round_index - cached.H, round_index))
            indices = [0] + [frame if frame >= 1 else -1 for frame in history]
            indices += [frame if 1 <= frame < length else -1 for frame in present]
            indices = torch.tensor([indices], device='cuda')
            valid = (indices[:, -cached.P:] >= 1)
            context = torch.zeros(1, 2, 1 + cached.H, 2, 2, device='cuda')
            context[:, :, 0] = latents[:, 0]
            for slot, frame in enumerate(history, start=1):
                if frame >= 1:
                    context[:, :, slot] = clean[frame]
            pose = poses[:, indices[0].clamp(min=0)][None]
            for substep in range(cached.m):
                schedule = torch.zeros_like(indices, dtype=torch.float32)
                schedule[:, 1:1 + cached.H] = (indices[:, 1:1 + cached.H] >= 1) * 200
                schedule[:, -cached.P:] = cached.t_table[cached.slot_steps + substep] * valid
                inputs = (torch.cat((context, wave), dim=2), schedule, tokens, pose, indices,
                          torch.zeros(1, device='cuda', dtype=torch.long), valid,
                          cached.refs[0].transpose(1, 2), cached.refs[1])
                conditional = models.net(inputs)[:, :, -cached.P:].float()
                cached.initial.training, cached.initial.audio_dropout_p = True, 1.0
                unconditional = models.net(inputs)[:, :, -cached.P:].float()
                cached.initial.training, cached.initial.audio_dropout_p = False, 0.0
                conditional[:, :, -1:] = unconditional[:, :, -1:] + cached.guidance_scale * (
                    conditional[:, :, -1:] - unconditional[:, :, -1:])
                wave = wave + conditional * cached.dt
            if round_index >= 1:
                clean.append(wave[0, :, 0].clone())
        expected = torch.stack(clean, dim=1)
        torch.testing.assert_close(actual, expected, rtol=.04, atol=.04)


if __name__ == '__main__':
    unittest.main()
