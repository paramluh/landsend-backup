#!/usr/bin/env python3
"""STREAMING MIMI DECODER (codes -> 24 kHz audio, one 80 ms frame per call, B streams), CUDA-graph capturable: every state lives in a
static device buffer, the frame index in a device scalar, no host branch / sync per frame.

transformers' MimiModel.decode has no streaming state for the SEANet convolutions (only the transformer KV cache), so this re-runs
the SAME modules' weights frame by frame with explicit states (the Moshi streaming rules):
  quantizer   semantic RVQ (codebook 0) + acoustic RVQ (1..31): sum of codebook rows, output_proj (k1 conv) each, summed
              (= MimiSplitResidualVectorQuantizer.decode, without its torch.tensor(0.0) host constant).
  upsample    ConvTranspose1d (12.5 -> 25 Hz, k4 s2, depthwise, no bias): output of this frame = conv_T(x)[:2 s] + the tail carried
              from the previous frame; the last k - s outputs are the new tail (trim_right_ratio 1: the full-sequence path trims them
              off the right end, so the first T * s outputs are identical).
  transformer 8 layers, RoPE on the absolute 25 Hz step, causal sliding window 250 (= create_sliding_window_causal_mask: keys in
              (q - 250, q]): K / V in a 256-slot ring per layer with the slot's absolute position (masked SDPA, fp32).
  SEANet      causal Conv1d (stride 1): state = the last (k - 1) * dilation inputs (full path: that many zeros of left padding);
              ConvTranspose1d (8, 6, 5, 4): tail carried as for the upsample, bias added on emitted samples only.
=> the concatenated output of frames 0..N-1 equals MimiModel.decode(codes[:, :, :N]) (fp32 round-off), test: tests/test_mimi_dec.py."""
import math

import torch
import torch.nn.functional as F

SPF24 = 1920
RING = 256


class MimiDecStream:
    def __init__(self, mimi, B, dev):
        from transformers.models.mimi.modeling_mimi import MimiConv1d, MimiConvTranspose1d, MimiResnetBlock
        self.mimi, self.B, self.dev = mimi, int(B), dev
        cfg = mimi.config
        self.win = int(cfg.sliding_window)
        assert self.win <= RING - 2
        q = mimi.quantizer
        self.sem, self.ac = q.semantic_residual_vector_quantizer, q.acoustic_residual_vector_quantizer
        self.nsem = int(q.num_semantic_quantizers)
        with torch.no_grad():
            self.cb_sem = torch.stack([l.codebook.embed for l in self.sem.layers]).to(dev).contiguous()   # [1, 2048, 256]
            self.cb_ac = torch.stack([l.codebook.embed for l in self.ac.layers]).to(dev).contiguous()     # [31, 2048, 256]
        self.nq = self.cb_sem.shape[0] + self.cb_ac.shape[0]
        # transformer
        tr = mimi.decoder_transformer
        self.layers = tr.layers
        self.nh, self.D = cfg.num_attention_heads, cfg.head_dim
        H = cfg.hidden_size
        self.K = [torch.zeros(B, self.nh, RING, self.D, device=dev) for _ in self.layers]
        self.V = [torch.zeros(B, self.nh, RING, self.D, device=dev) for _ in self.layers]
        self.slot_pos = torch.full((RING,), -10 ** 9, dtype=torch.long, device=dev)
        inv = tr.rotary_emb.inv_freq.to(dev).float()
        pos = torch.arange(65536, device=dev).float()
        fr = pos[:, None] * inv[None]
        emb = torch.cat([fr, fr], -1)
        sc = float(getattr(tr.rotary_emb, "attention_scaling", 1.0))
        self.cos, self.sin = (emb.cos() * sc).contiguous(), (emb.sin() * sc).contiguous()
        self.k = torch.zeros(1, dtype=torch.long, device=dev)                       # frame index (device)
        self.ar2 = torch.arange(2, device=dev)
        self.ring_ar = torch.arange(RING, device=dev)
        # upsample + SEANet states
        self.ups = mimi.upsample
        k_, s_ = self.ups.conv.kernel_size[0], self.ups.conv.stride[0]
        self.ups_tail = torch.zeros(B, H, k_ - s_, device=dev)
        self.ops = []                                                                # (kind, module, state or None)
        for L in mimi.decoder.layers:
            if isinstance(L, MimiConv1d):
                self.ops.append(("conv", L, self._conv_state(L)))
            elif isinstance(L, MimiConvTranspose1d):
                kk, ss = L.conv.kernel_size[0], L.conv.stride[0]
                self.ops.append(("convT", L, torch.zeros(B, L.conv.out_channels, kk - ss, device=dev)))
            elif isinstance(L, MimiResnetBlock):
                st = [self._conv_state(c) if isinstance(c, MimiConv1d) else None for c in L.block]
                assert not isinstance(L.shortcut, MimiConv1d), "conv shortcut not supported"
                self.ops.append(("res", L, st))
            else:
                self.ops.append(("act", L, None))

    def _conv_state(self, L):
        assert L.causal and L.conv.stride[0] == 1, "decoder convs are causal stride 1"
        n = (L.conv.kernel_size[0] - 1) * L.conv.dilation[0]
        assert L.pad_mode == "constant"
        return torch.zeros(self.B, L.conv.in_channels, n, device=self.dev) if n else None

    def reset(self):
        for t in self.K + self.V:
            t.zero_()
        self.slot_pos.fill_(-10 ** 9)
        self.k.zero_()
        self.ups_tail.zero_()
        for kind, _, st in self.ops:
            for s in (st if isinstance(st, list) else [st]):
                if s is not None:
                    s.zero_()

    # ---- pieces
    @staticmethod
    def _sconv(L, x, st):
        if st is not None:
            xc = torch.cat([st, x], -1)
            st.copy_(xc[..., xc.shape[-1] - st.shape[-1]:])
        else:
            xc = x
        return L.conv(xc)

    @staticmethod
    def _sconvT(conv, x, tail, bias=True):
        y = F.conv_transpose1d(x, conv.weight, None, conv.stride, 0, 0, conv.groups, conv.dilation)
        n = x.shape[-1] * conv.stride[0]
        kt = tail.shape[-1]
        head = y[..., :kt] + tail
        y = torch.cat([head, y[..., kt:]], -1)
        tail.copy_(y[..., n:])
        out = y[..., :n]
        if bias and conv.bias is not None:
            out = out + conv.bias[None, :, None]
        return out

    def _quant(self, codes):
        """codes [B, 32] local ids -> [B, 512, 1] (= quantizer.decode)."""
        cs, ca = codes[:, :self.nsem], codes[:, self.nsem:]
        es = sum(F.embedding(cs[:, i], self.cb_sem[i]) for i in range(self.nsem))                         # [B, 256]
        ea = F.embedding(ca + torch.arange(ca.shape[1], device=codes.device)[None] * self.cb_ac.shape[1],
                         self.cb_ac.reshape(-1, self.cb_ac.shape[-1])).sum(1)
        qs = self.sem.output_proj(es[:, :, None]) if self.sem.output_proj is not None else es[:, :, None]
        qa = self.ac.output_proj(ea[:, :, None]) if self.ac.output_proj is not None else ea[:, :, None]
        return qs + qa

    @staticmethod
    def _rot(x, c, s):
        h = x.shape[-1] // 2
        return x * c + torch.cat([-x[..., h:], x[..., :h]], -1) * s

    def _transformer(self, x):
        """x [B, 2, 512] at 25 Hz steps 2k, 2k + 1 (device k) -> [B, 2, 512]."""
        B = x.shape[0]
        pos = self.k * 2 + self.ar2                                                    # [2]
        slots = pos % RING
        self.slot_pos.index_copy_(0, slots, pos)
        c, s = self.cos.index_select(0, pos)[None, None], self.sin.index_select(0, pos)[None, None]   # [1, 1, 2, D]
        sp = self.slot_pos[None]                                                       # [1, R]
        mask = (sp <= pos[:, None]) & (sp > pos[:, None] - self.win)                   # [2, R]
        for i, Ly in enumerate(self.layers):
            at = Ly.self_attn
            h = Ly.input_layernorm(x)
            q = at.q_proj(h).view(B, 2, self.nh, self.D).transpose(1, 2)
            k = at.k_proj(h).view(B, 2, self.nh, self.D).transpose(1, 2)
            v = at.v_proj(h).view(B, 2, self.nh, self.D).transpose(1, 2)
            q, k = self._rot(q, c, s), self._rot(k, c, s)
            self.K[i].index_copy_(2, slots, k)
            self.V[i].index_copy_(2, slots, v)
            a = F.scaled_dot_product_attention(q, self.K[i], self.V[i], attn_mask=mask[None, None], scale=at.scaling)
            a = at.o_proj(a.transpose(1, 2).reshape(B, 2, -1))
            x = x + Ly.self_attn_layer_scale(a)
            x = x + Ly.mlp_layer_scale(Ly.mlp(Ly.post_attention_layernorm(x)))
        return x

    @torch.no_grad()
    def step(self, codes):
        """codes [B, 32] LOCAL ids (0..2047), long, on dev -> audio [B, 1920] fp32; advances the frame index."""
        e = self._quant(codes)                                                         # [B, 512, 1]
        e = self._sconvT(self.ups.conv, e, self.ups_tail, bias=False)                  # [B, 512, 2]
        e = self._transformer(e.transpose(1, 2)).transpose(1, 2)
        for kind, L, st in self.ops:
            if kind == "conv":
                e = self._sconv(L, e, st)
            elif kind == "convT":
                e = self._sconvT(L.conv, e, st)
            elif kind == "res":
                r = e
                for c_, s_ in zip(L.block, st):
                    e = self._sconv(c_, e, s_) if s_ is not None or hasattr(c_, "conv") else c_(e)
                e = r + e
            else:
                e = L(e)
        self.k.add_(1)
        assert e.shape[-1] == SPF24, e.shape
        return e[:, 0]
