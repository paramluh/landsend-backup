#!/usr/bin/env python3
"""VoxStream with a FUSED per-frame layer stack (the self-play front sits on the critical path when the hearing delay is 3 frames):
same math as vox_stream.VoxStream.layers (Voxtral Realtime encoder layer: RMSNorm -> q/k/v (q, v biased) -> RoPE -> sliding-window
FA3 -> o (biased) -> residual -> RMSNorm -> SiLU(gate) * up -> down (biased) -> residual), with
  * q / k / v as ONE GEMM (concatenated weights, k's bias 0), gate / up as ONE GEMM;
  * the trainer's fused Triton kernels (train/vendor/fused_triton via m.FO): rms, add_rms (residual add + next norm in one kernel);
  * RoPE on q and k in one pass.
~10 kernels per layer instead of ~40. Not bit-identical (bf16 reduction order in the fused norms); tests/test_front.py compares it with
VoxStream (cosine)."""
import torch
import torch.nn.functional as F

from . import vox_stream as VS


class VoxStreamFast(VS.VoxStream):
    def __init__(self, enc, B, max_frames, dev, fa, FO, feature_extractor, *,
                 cache_mode='contiguous', rewind_frames=8, max_block_frames=8):
        super().__init__(enc, B, max_frames, dev, fa, feature_extractor,
                         cache_mode=cache_mode, rewind_frames=rewind_frames, max_block_frames=max_block_frames)
        self.FO = FO
        self.W = []
        with torch.no_grad():
            for Ly in enc.layers:
                at, mlp = Ly.self_attn, Ly.mlp
                Hq = at.q_proj.weight.shape[0]
                wqkv = torch.cat([at.q_proj.weight, at.k_proj.weight, at.v_proj.weight], 0).contiguous()
                bq = at.q_proj.bias if at.q_proj.bias is not None else torch.zeros(Hq, device=dev, dtype=wqkv.dtype)
                bk = at.k_proj.bias if at.k_proj.bias is not None else torch.zeros(at.k_proj.weight.shape[0], device=dev, dtype=wqkv.dtype)
                bv = at.v_proj.bias if at.v_proj.bias is not None else torch.zeros(at.v_proj.weight.shape[0], device=dev, dtype=wqkv.dtype)
                bqkv = torch.cat([bq, bk, bv]).contiguous()
                wgu = torch.cat([mlp.gate_proj.weight, mlp.up_proj.weight], 0).contiguous()
                assert mlp.gate_proj.bias is None and mlp.up_proj.bias is None
                self.W.append((wqkv, bqkv, wgu, Hq, at.k_proj.weight.shape[0]))
        self.inter = enc.layers[0].mlp.gate_proj.weight.shape[0]

    def layers(self, h, L):
        FO, B = self.FO, h.shape[0]
        t0 = self.kd * VS.EPF
        pos = t0 + torch.arange(L, device=self.dev)
        c, s = self.cos.index_select(0, pos)[None, :, None], self.sin.index_select(0, pos)[None, :, None]
        sl = t0.to(torch.int32).expand(B).contiguous()
        lys = self.enc.layers
        x = FO.rms(h, lys[0].self_attn_layer_norm.weight, float(lys[0].self_attn_layer_norm.variance_epsilon))
        nq, nk = self.nh, self.kvh
        for i, Ly in enumerate(lys):
            at, mlp = Ly.self_attn, Ly.mlp
            wqkv, bqkv, wgu, Hq, Hk = self.W[i]
            qkv = F.linear(x, wqkv, bqkv)
            qk = qkv[..., :Hq + Hk].view(B, L, nq + nk, self.D)
            qk = self._rot(qk, c, s)
            q, k = qk[:, :, :nq], qk[:, :, nq:]
            v = qkv[..., Hq + Hk:].view(B, L, nk, self.D)
            a = self._attention(q, self.K[i], self.V[i], k=k, v=v, cache_seqlens=sl, softmax_scale=at.scaling,
                                causal=True, window_size=(VS.WIN - 1, 0))
            o = at.o_proj(a.reshape(B, L, -1))
            h, x2 = FO.add_rms(h, o, Ly.final_layer_norm.weight, float(Ly.final_layer_norm.variance_epsilon))
            gu = F.linear(x2, wgu)
            y = mlp.down_proj(F.silu(gu[..., :self.inter]) * gu[..., self.inter:])     # = eager: silu rounded, then * up
            nxt = lys[i + 1].self_attn_layer_norm if i + 1 < len(lys) else self.enc.norm
            h, x = FO.add_rms(h, y, nxt.weight, float(nxt.variance_epsilon))
        return x
