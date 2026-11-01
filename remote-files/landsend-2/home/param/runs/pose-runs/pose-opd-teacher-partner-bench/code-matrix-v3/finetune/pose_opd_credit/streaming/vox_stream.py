"""Streaming Voxtral frontend from the selected student's archived live engine.

Vendored from code/live_deps/qwen-1ch/infer/s4fast/vox_stream.py in
ppm-aloe-poseduplex-raw-final. Arithmetic is unchanged; encoder/feature-extractor
and FA3 are injected instead of importing deployment-specific absolute paths.
"""
import math
import time
import torch
import torch.nn.functional as F

from .paged_kv import allocate_cache

SR24, SR16 = 24000, 16000
SPF24, SPF16 = 1920, 1280          # samples per 80 ms model frame
EPF = 4                            # encoder frames per model frame
MPF = 8                            # mel frames per model frame
N_FFT, HOP = 400, 160
LOOK24 = 72                        # 24k lookahead samples needed beyond the frame end (module doc)
CTX24 = 330                        # 24k history before the frame start used to rebuild the 16k window (multiple of 3)
WIN = 750                          # sliding window (encoder frames)


def log(*a):
    print(f"[s4front {time.strftime('%H:%M:%S')}]", *a, flush=True)


# ------------------------------------------------------------------------------------------------ resample (= hera / torchaudio)
_RK = {}


def _sinc_kernel(orig_freq, new_freq, width_mult=6, rolloff=0.99, device=None, dtype=torch.float32):
    """= hera streaming_encoder._sinc_kernel (torchaudio sinc_interp_hann)."""
    g = math.gcd(int(orig_freq), int(new_freq))
    orig, new = int(orig_freq) // g, int(new_freq) // g
    base = min(orig, new) * rolloff
    width = math.ceil(width_mult * orig / base)
    idx = torch.arange(-width, width + orig, device=device, dtype=dtype)[None, None] / orig
    t = torch.arange(0, -new, -1, device=device, dtype=dtype)[:, None, None] / new + idx
    t = (t * base).clamp_(-width_mult, width_mult)
    window = torch.cos(t * math.pi / width_mult / 2) ** 2
    t = t * math.pi
    kernel = torch.where(t == 0, torch.ones_like(t), t.sin() / t) * window * (base / orig)
    return kernel, width, orig, new


def resample(wav, orig_freq=SR24, new_freq=SR16):
    """= hera streaming_encoder.resample: (batch, samples) polyphase, zero padded both ends."""
    key = (int(orig_freq), int(new_freq), wav.device.type, wav.device.index, wav.dtype)
    if key not in _RK:
        _RK[key] = _sinc_kernel(orig_freq, new_freq, device=wav.device, dtype=wav.dtype)
    kernel, width, orig, new = _RK[key]
    length = wav.shape[-1]
    padded = F.pad(wav, (width, width + orig))
    out = F.conv1d(padded[:, None], kernel, stride=orig)
    out = out.transpose(1, 2).reshape(wav.shape[0], -1)
    return out[..., : math.ceil(new * length / orig)]


class VoxStream:
    """streaming Voxtral encoder for B parallel audio streams (module doc). State: frame counter k (host + device scalar), conv caches,
    per-layer KV caches of max_frames * 4 encoder frames. step(wav_win) -> latents [B, 4, 1280] of frame k, k += 1.
    wav_win [B, CTX24 + 1920 + LOOK24] = 24k samples [1920 k - CTX24, 1920 (k + 1) + LOOK24) (zeros before the start)."""

    def __init__(self, enc, B, max_frames, dev, fa, feature_extractor, *,
                 cache_mode='contiguous', rewind_frames=8, max_block_frames=8):
        self.enc, self.B, self.dev = enc, int(B), dev
        cfg = enc.config
        self.Hd, self.nh, self.D = cfg.hidden_size, cfg.num_attention_heads, cfg.head_dim
        self.kvh = getattr(cfg, "num_key_value_heads", self.nh) or self.nh
        assert cfg.sliding_window == WIN
        self.fe = feature_extractor
        self.melW = torch.as_tensor(feature_extractor.mel_filters, device=dev, dtype=torch.float32)
        self.lmax = feature_extractor.global_log_mel_max
        self.win = torch.hann_window(N_FFT, device=dev)
        self.fa = fa
        self.maxT = int(max_frames) * EPF + 8
        self.cache_mode = cache_mode
        self.K, self.V, self.page_table, self.cache_layout = allocate_cache(
            batch=self.B, positions=self.maxT, heads=self.kvh, head_dim=self.D,
            layers=len(enc.layers), device=dev, mode=cache_mode, attention_window=WIN,
            rewind_positions=int(rewind_frames) * EPF, append_positions=int(max_block_frames) * EPF)
        self.cache_capacity_positions = self.maxT if self.cache_layout is None else self.cache_layout.capacity_positions
        pos = torch.arange(self.maxT, device=dev)[None]
        c, s = enc.rotary_emb(torch.zeros(1, 1, 8, dtype=torch.bfloat16, device=dev), pos)
        self.cos, self.sin = c[0].contiguous(), s[0].contiguous()                       # [maxT, D] bf16 (= HF)
        self.mel_prev = torch.zeros(B, cfg.num_mel_bins, 2, dtype=torch.bfloat16, device=dev)
        self.c1_prev = torch.zeros(B, self.Hd, 1, dtype=torch.bfloat16, device=dev)
        self.kd = torch.zeros(1, dtype=torch.long, device=dev)                          # frame index (device)
        self.k = 0
        self.ar4 = torch.arange(EPF, device=dev)

    def reset(self):
        self.mel_prev.zero_(); self.c1_prev.zero_(); self.kd.zero_(); self.k = 0

    def _attention(self, q, keys, values, **kwargs):
        if self.cache_layout is not None and q.shape[1] > self.cache_layout.append_positions:
            raise ValueError('Voxtral attention block exceeds the paged cache append allowance')
        return self.fa.flash_attn_with_kvcache(q, keys, values, page_table=self.page_table, **kwargs)

    # ---- mel of model frames k .. k+n-1 from a 24k window
    def mel(self, wav_win, n, first):
        """wav_win [B, CTX24 + 1920 n + LOOK24] -> log-mel [B, 128, 8 n] (fp32). first: frame k == 0 (reflect pad at the start)."""
        w16 = resample(wav_win.float())                                                 # [B, 2/3 len]
        o = CTX24 * 2 // 3                                                               # 16k index of the frame start inside w16
        if first:
            seg = w16[:, o:o + SPF16 * n + 40]
            seg = torch.cat([seg[:, 1:201].flip(-1), seg], -1)                         # torch reflect pad of 200 at the start
        else:
            seg = w16[:, o - 200:o + SPF16 * n + 40]
        assert seg.shape[-1] == 200 + SPF16 * n + 40, seg.shape
        st = torch.stft(seg, N_FFT, HOP, window=self.win, center=False, return_complex=True)   # [B, 201, 8n]
        assert st.shape[-1] == MPF * n, st.shape
        mag = st.abs() ** 2
        ms = self.melW.T @ mag
        ls = torch.clamp(ms, min=1e-10).log10()
        ls = torch.clamp(ls, min=float(self.lmax) - 8.0)                               # = maximum(ls, global max - 8): graph-safe
        return (ls + 4.0) / 4.0

    def embed(self, mel):
        """causal conv1 / conv2 with the streaming caches: mel [B, 128, 8n] -> [B, 4n, 1280]."""
        em = self.enc.embedder
        x = torch.cat([self.mel_prev, mel.to(torch.bfloat16)], -1)
        self.mel_prev.copy_(x[..., -2:])
        h1 = F.gelu(F.conv1d(x, em.conv1.weight, em.conv1.bias))                          # [B, H, 8n]
        x2 = torch.cat([self.c1_prev, h1], -1)
        self.c1_prev.copy_(x2[..., -1:])
        h2 = F.gelu(F.conv1d(x2, em.conv2.weight, em.conv2.bias, stride=2))               # [B, H, 4n]
        return h2.permute(0, 2, 1)

    @staticmethod
    def _rms(x, w, eps):
        xf = x.float()
        return w * (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype)

    @staticmethod
    def _rot(x, c, s):
        h = x.shape[-1] // 2
        r = torch.cat([-x[..., h:], x[..., :h]], -1)
        return x * c + r * s

    def layers(self, h, L):
        """h [B, L, H] at encoder positions 4k .. 4k + L - 1 (device scalar kd) -> final-normed [B, L, H]."""
        B = h.shape[0]
        t0 = self.kd * EPF
        pos = t0 + torch.arange(L, device=self.dev)
        c, s = self.cos.index_select(0, pos)[None, :, None], self.sin.index_select(0, pos)[None, :, None]
        sl = t0.to(torch.int32).expand(B).contiguous()
        for i, Ly in enumerate(self.enc.layers):
            at, mlp = Ly.self_attn, Ly.mlp
            x = self._rms(h, Ly.self_attn_layer_norm.weight, Ly.self_attn_layer_norm.variance_epsilon)
            q = at.q_proj(x).view(B, L, self.nh, self.D)
            k = at.k_proj(x).view(B, L, self.kvh, self.D)
            v = at.v_proj(x).view(B, L, self.kvh, self.D)
            q, k = self._rot(q, c, s), self._rot(k, c, s)
            a = self._attention(q, self.K[i], self.V[i], k=k, v=v, cache_seqlens=sl, softmax_scale=at.scaling,
                                causal=True, window_size=(WIN - 1, 0))
            h = h + at.o_proj(a.reshape(B, L, -1))
            x = self._rms(h, Ly.final_layer_norm.weight, Ly.final_layer_norm.variance_epsilon)
            h = h + mlp.down_proj(mlp.act_fn(mlp.gate_proj(x)) * mlp.up_proj(x))
        return self._rms(h, self.enc.norm.weight, self.enc.norm.variance_epsilon)

    @torch.no_grad()
    def encode(self, wav_win, n=1):
        """n model frames from the 24k window [1920 k - CTX24, 1920 (k + n) + LOOK24) -> latents [B, 4n, 1280]; advances k by n."""
        first = self.k == 0
        mel = self.mel(wav_win, n, first)
        h = self.embed(mel)
        out = self.layers(h, EPF * n)
        self.kd.add_(n)
        self.k += n
        return out

    def step_graphable(self, wav_win):
        """one frame, k >= 1 only (no host branch on k): for CUDA-graph capture."""
        mel = self.mel(wav_win, 1, False)
        h = self.embed(mel)
        out = self.layers(h, EPF)
        self.kd.add_(1)
        return out


