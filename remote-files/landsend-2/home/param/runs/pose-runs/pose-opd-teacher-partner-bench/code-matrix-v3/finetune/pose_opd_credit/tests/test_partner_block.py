"""Packed row offsets, causal KV refresh, and rewind-boundary state checks.

Small CPU attention is an indexing oracle; actual BF16 GPU model parity is a
separate required benchmark before selecting the block optimization.
"""
import copy
from contextlib import nullcontext
import math
import sys
import types
import unittest
from unittest.mock import patch

import torch
from torch import nn

from finetune.pose_opd_credit.partner_block import CommittedBlockReplay, _layer


def rms(value, weight, epsilon):
    return (value.float() * torch.rsqrt(value.float().square().mean(-1, keepdim=True) + epsilon)).to(value.dtype) * weight


def qk_prep(q, k, qw, kw, epsilon, cosine, sine):
    dim = len(qw)
    return (rms(q.view(*q.shape[:-1], -1, dim), qw, epsilon),
            rms(k.view(*k.shape[:-1], -1, dim), kw, epsilon))


FUSED = types.SimpleNamespace(
    rms=rms, qk_prep=qk_prep,
    add_rms=lambda a, b, w, e: (a + b, rms(a + b, w, e)),
    silu_mul=lambda g, u: torch.nn.functional.silu(g) * u,
)


class Attention:
    def flash_attn_with_kvcache(self, q, keys, values, *, k, v, cache_seqlens,
                               softmax_scale, causal, num_splits):
        assert causal and num_splits == 1
        batch, count, heads, dim = q.shape
        out = torch.empty_like(q)
        for row in range(batch):
            first = int(cache_seqlens[row])
            keys[row, first:first + count].copy_(k[row])
            values[row, first:first + count].copy_(v[row])
            for frame in range(count):
                end = first + frame + 1
                kk = keys[row, :end].repeat_interleave(heads // keys.shape[-2], dim=-2)
                vv = values[row, :end].repeat_interleave(heads // values.shape[-2], dim=-2)
                weights = torch.einsum('hd,thd->ht', q[row, frame].float(), kk.float()) * softmax_scale
                out[row, frame] = torch.einsum('ht,thd->hd', weights.softmax(-1), vv.float()).to(q.dtype)
        return out


class Norm(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size, dtype=torch.bfloat16))
        self.variance_epsilon = 1e-5


class Layer(nn.Module):
    def __init__(self, size=8):
        super().__init__()
        self.input_layernorm, self.post_attention_layernorm = Norm(size), Norm(size)
        self.self_attn = nn.Module()
        at = self.self_attn
        at.q_proj, at.k_proj, at.v_proj, at.o_proj = [nn.Linear(size, out, bias=False).bfloat16()
                                                    for out in (8, 4, 4, 8)]
        at.q_norm, at.k_norm = Norm(4), Norm(4)
        at.head_dim, at.scaling = 4, .5
        self.mlp = nn.Module()
        self.mlp.gate_proj = nn.Linear(size, 16, bias=False).bfloat16()
        self.mlp.up_proj = nn.Linear(size, 16, bias=False).bfloat16()
        self.mlp.down_proj = nn.Linear(16, size, bias=False).bfloat16()


class CodeEmbedding(nn.Module):
    def __init__(self, size=8):
        super().__init__()
        self.table = nn.Embedding(64, size).bfloat16()

    def forward(self, codes):
        return (self.table(codes.clamp_min(0)) * (codes >= 0)[..., None]).sum(-2) / 32


class Input(nn.Module):
    def __init__(self):
        super().__init__()
        self.text = nn.ModuleList([nn.Embedding(32, 8).bfloat16() for _ in range(2)])
        self.codes = nn.ModuleList([CodeEmbedding() for _ in range(2)])

    def forward(self, ids, codes, present):
        return sum((self.text[side](ids[side]) + self.codes[side](codes[side])) * present[side, :, None]
                   for side in range(2)).to(torch.bfloat16)


class FakeActor:
    def __init__(self):
        cpu = torch.device('cpu')
        self.d0 = self.dl = cpu
        self.sdevs = [cpu]
        self.pose = self.cfg_rows = False
        self.NI = self.R = 2
        self.H, self._opd_rewind_window = 8, 8
        self._opd_sampler_backend = 'inverse_cdf'
        self._opd_input_mode = self._opd_pending_frame = self._opd_force = None
        self._opd_next_frame = 10
        self._opd_lane_force = torch.full((2,), -1, dtype=torch.long)
        self.Pv = {cpu: torch.tensor([3, 7])}
        self.Tv = {cpu: torch.empty(2, dtype=torch.long)}
        self.set_f(10)
        self.laneB = torch.randint(0, 32, (2, 40))
        self.depB = torch.randint(0, 64, (2, 40, 32))
        self.idsA = torch.zeros_like(self.laneB)
        self.depA = torch.full_like(self.depB, -1)
        self.presA, self.presB = torch.zeros(2, 40), torch.ones(2, 40)
        self._opd_audio = torch.randn(2, 40, 8).bfloat16()
        self.hn_prev = torch.randn(2, 8).bfloat16()
        self.hv_s, self.z0 = torch.empty(2, 8).bfloat16(), torch.empty(2, 16)
        self.pk = {cpu: torch.empty(2, 33, dtype=torch.long)}
        self.m = nn.Module()
        self.m.inp = Input()
        self.m.head = nn.Module()
        self.m.head.norm = Norm(8)
        self.m.tower_s = nn.Module()
        tower = self.m.tower_s
        tower.text_emb = nn.Embedding(32, 8).bfloat16()
        tower.code_emb = CodeEmbedding()
        tower.cond = nn.Linear(8, 8, bias=False).bfloat16()
        tower.norm, tower.cb0_head = Norm(8), nn.Linear(8, 16, bias=False).bfloat16()
        self.bb, self.ts = types.SimpleNamespace(L=[Layer(), Layer()]), types.SimpleNamespace(L=[Layer(), Layer()])
        self.segs = [(cpu, [0, 1])]
        self.bbK = [torch.randn(2, 40, 1, 4).bfloat16() for _ in range(2)]
        self.bbV = [torch.randn(2, 40, 1, 4).bfloat16() for _ in range(2)]
        self.tsK = [torch.randn(2, 40, 1, 4).bfloat16() for _ in range(2)]
        self.tsV = [torch.randn(2, 40, 1, 4).bfloat16() for _ in range(2)]
        self.rope = {cpu: (torch.ones(1, 40, 4).bfloat16(), torch.zeros(1, 40, 4).bfloat16())}
        self.rope_t = self.rope[cpu]
        self.fa, self.history, self.saved = Attention(), [], {}

    def set_f(self, frame):
        for device in self.sdevs:
            self.Tv[device].copy_(self.Pv[device] + frame)

    def _supply_text(self, tokens):
        self.history.append(tokens.clone())

    def _remember_frame(self, frame):
        self.saved[frame] = (self.hn_prev.clone(), [tokens.clone() for tokens in self.history])


def sequential_reference(actor, start, stop):
    rows = torch.arange(actor.NI)
    for frame in range(start, stop):
        actor.set_f(frame)
        positions = actor.Tv[actor.d0]
        lengths = positions.to(torch.int32)
        cosine, sine = (x.index_select(1, positions) for x in actor.rope_t)
        tokens, codes = actor.laneB[rows, positions], actor.depB[rows, positions]
        if frame > start:
            tower = actor.m.tower_s
            previous = actor.depB[rows, positions - 1]
            hidden = (tower.text_emb(tokens) + tower.code_emb(previous) + tower.cond(actor.hn_prev))[None]
            for index, layer in enumerate(actor.ts.L):
                hidden = _layer(actor, layer, hidden, cosine, sine, actor.tsK[index], actor.tsV[index],
                                lengths, batch=2, frames=1, fused=FUSED)
            actor.hv_s.copy_(rms(hidden[0], tower.norm.weight, tower.norm.variance_epsilon))
            actor.z0.copy_((actor.hv_s @ tower.cb0_head.weight.T).float())
        ids = torch.stack((actor.idsA[rows, positions], tokens))
        depth = torch.stack((actor.depA[rows, positions], codes))
        present = torch.stack((actor.presA[rows, positions], actor.presB[rows, positions]))
        hidden = (actor.m.inp(ids, depth, present) + actor._opd_audio[rows, positions])[None]
        for index, layer in enumerate(actor.bb.L):
            hidden = _layer(actor, layer, hidden, cosine, sine, actor.bbK[index], actor.bbV[index],
                            lengths, batch=2, frames=1, fused=FUSED)
        actor._supply_text(tokens)
        actor.hn_prev.copy_(rms(hidden[0], actor.m.head.norm.weight, actor.m.head.norm.variance_epsilon))
        actor._remember_frame(frame + 1)
    actor._opd_next_frame = stop


class PartnerBlockTests(unittest.TestCase):
    def test_packed_rows_match_sequential_causal_replay_and_every_snapshot(self):
        torch.manual_seed(413)
        actor = FakeActor()
        reference = copy.deepcopy(actor)
        original = copy.deepcopy(actor)
        with torch.no_grad(), patch.dict(sys.modules, {'fused_triton': FUSED}), \
                patch('torch.cuda.device', lambda *_: nullcontext()):
            sequential_reference(reference, 10, 16)
            block = CommittedBlockReplay(actor, 6, graphs=False)
            receipt = block.replay(10, 16)
        self.assertEqual(receipt['new_sampling_draws'], 0)
        self.assertEqual(actor._opd_next_frame, 16)
        torch.testing.assert_close(actor.hn_prev, reference.hn_prev, atol=.016, rtol=.016)
        torch.testing.assert_close(actor.z0, reference.z0, atol=.016, rtol=.016)
        for name in ('bbK', 'bbV', 'tsK', 'tsV'):
            for mine, expected, before in zip(getattr(actor, name), getattr(reference, name), getattr(original, name)):
                torch.testing.assert_close(mine, expected, atol=.016, rtol=.016)
                for row, prefix in enumerate((3, 7)):
                    first = prefix + 10 + int(name.startswith('ts'))
                    torch.testing.assert_close(mine[row, :first], before[row, :first], atol=0, rtol=0)
                    torch.testing.assert_close(mine[row, prefix + 16:], before[row, prefix + 16:], atol=0, rtol=0)
        self.assertEqual(sorted(actor.saved), list(range(11, 17)))
        for frame, (hidden, history) in actor.saved.items():
            torch.testing.assert_close(hidden, reference.saved[frame][0], atol=.016, rtol=.016)
            self.assertEqual(len(history), frame - 10)
            for mine, expected in zip(history, reference.saved[frame][1]):
                torch.testing.assert_close(mine, expected, atol=0, rtol=0)

    def test_requires_matching_rewound_cursor(self):
        with patch.dict(sys.modules, {'fused_triton': FUSED}):
            block = CommittedBlockReplay(FakeActor(), 6, graphs=False)
        with self.assertRaisesRegex(ValueError, 'Rewind'):
            block.replay(11, 17)


if __name__ == '__main__':
    unittest.main()
