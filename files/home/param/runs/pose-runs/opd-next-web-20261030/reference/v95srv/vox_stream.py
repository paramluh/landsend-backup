#!/usr/bin/env python3
"""STAGE-4 streaming audio front end for channel A: the frozen Voxtral Realtime encoder (mistralai/Voxtral-Mini-4B-Realtime-2602,
audio_tower only) run FRAME BY FRAME (80 ms = 1920 samples at 24 kHz -> 4 encoder frames at 50 Hz), plus the Mimi encoder streamed on
the same audio (A's codes, undelayed), plus the adapter (4 encoder frames -> 1 model frame).

EXACTNESS vs the full-sequence path (hera streaming_encoder.StreamingEncoder.encode_frames = what the trainer computes over a whole row):
  resample 24k -> 16k  polyphase sinc (hera's resample = torchaudio's kernel): 16k sample j = 2q + ph depends on 24k samples
                       [3q - 10, 3q + 12]; zero history before the start (as the reference pads).
  log-mel              STFT n_fft 400 / hop 160 / hann, center=True in the reference (reflect pad 200 at the START = frame 0; the
                       reference also reflect-pads at the END, which only touches the row's LAST model frame). Global log-mel max 1.5
                       (fixed, so the mel is causal: no per-utterance max). Mel frame i = 16k samples [160 i - 200, 160 i + 200).
                       -> model frame k (mel frames 8k .. 8k+7) needs 16k samples up to 1280 (k + 1) + 39, i.e. 24k samples up to
                       1920 (k + 1) + 72: a LOOKAHEAD of 72 samples = 3 ms. Streaming = process frame k when audio through
                       1920 (k + 1) + 72 has arrived (3 ms extra latency, exact).
  conv1 (k3, s1) / conv2 (k3, s2)   causal (left pads 2 / 1 zeros): streaming caches of the last 2 mel frames / the last conv1 output.
  32 encoder layers    RoPE on the absolute encoder-frame index, causal SLIDING-WINDOW attention of 750 frames (15 s, HF's
                       create_sliding_window_causal_mask: keys in (q - 750, q]) -> FA3 flash_attn_with_kvcache window_size (749, 0)
                       on a static per-layer KV cache (graph-capturable: the frame index lives in a device scalar).
So the streaming latents equal the full-sequence latents up to bf16 kernel round-off (different GEMM / conv shapes), except the very
last frame of a finite file (end reflect pad). parity: test_front.py (streaming vs HF full-sequence encoder).

MIMI: transformers MimiModel.encode(use_streaming=True) with its padding cache + encoder KV cache, one 1920-sample frame per call
(the Mimi encoder is causal). For offline drivers the full-file encode is also available (encode_full).

REUSED (other agents' modules, imported): stage4/voxtral_front.py (weights path + loader _load_encoder, loudness rule lufs_gain_db /
  apply_gain, load_row_audio), stage4/vadapter_s4q.py (VAdapter = the trainer's adapter, init_rule). make_adapter() = a seeded fresh
  VAdapter with target_rms = the trainer's rule (emb_word_rms) for plumbing before a stage-4 checkpoint; load_adapter_from_export() reads
  vadapter.* of a stage-4 export."""
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from pathlib import Path
S4 = '/home/param/pose-repo/finetune/duplex_distill/production/lib'
MIMI_PATH = "/home/param/models/mimi"
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


# ------------------------------------------------------------------------------------------------ loading
def _s4mod(name):
    """import stage4/<name>.py BY FILE PATH under the name 's4_<name>' -- never by putting stage4/ on sys.path: stage4/ holds verbatim
    copies of stage-3 modules (data_s3q, model_s3q, sysprompt, ...) that would shadow the stage-3 ones the inference stack imports."""
    import importlib.util
    key = f"s4_{name}"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, f"{S4}/{name}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[key] = mod
        spec.loader.exec_module(mod)
    return sys.modules[key]


def _vf():
    """agent 2's stage4/voxtral_front (the trainer's front end): weights path, loader, loudness rule, full-sequence encode."""
    return _s4mod("voxtral_front")


def vox_path():
    return os.environ.get("VOXTRAL_DIR") or _vf().VOXTRAL_PATH


def load_audio_tower(dev, dtype=torch.bfloat16, attn="sdpa"):
    """the frozen VoxtralRealtimeEncoder exactly as the trainer loads it (voxtral_front._load_encoder: audio_tower.safetensors cut out of
    mistralai/Voxtral-Mini-4B-Realtime-2602), eval, no grad. attn only matters for the full-sequence reference (the streaming path runs
    its own FA3 sliding-window attention)."""
    VF = _vf()
    enc = VF._load_encoder(vox_path(), dev, dtype, attn)
    assert enc.config.sliding_window == WIN
    log(f"Voxtral encoder via stage4/voxtral_front._load_encoder from {vox_path()}: {sum(p.numel() for p in enc.parameters()) / 1e6:.0f}M params")
    return enc


def mel_filters_t(dev):
    """the feature extractor's slaney mel filter bank [201, 128] fp32 + its global log-mel max."""
    from transformers.models.voxtral_realtime.feature_extraction_voxtral_realtime import VoxtralRealtimeFeatureExtractor
    fe = VoxtralRealtimeFeatureExtractor.from_pretrained(vox_path())
    assert (fe.n_fft, fe.hop_length, fe.sampling_rate) == (N_FFT, HOP, SR16), (fe.n_fft, fe.hop_length, fe.sampling_rate)
    return fe, torch.from_numpy(fe.mel_filters).to(dev, torch.float32), fe.global_log_mel_max


def emb_word_rms(m, c=0):
    """= model_s4q.emb_word_rms (the trainer's target_rms rule): mean over word-piece rows (< 151643) of channel c's text table's row RMS."""
    W = m.m.s3in.emb[c].weight
    return float(W[:151643].float().pow(2).mean(-1).sqrt().double().mean())


def make_adapter(H, dev, seed=1234, target_rms=1.0, hidden=4096):
    """agent 1's stage4/vadapter_s4q.VAdapter (hera AudioAdapter, 4 x 1280 -> hidden -> H, RMS x target_rms), fresh init = the trainer's
    init_rule (weights N(0, 1/fan_in), biases 0), seeded (plumbing only, before a stage-4 checkpoint exists)."""
    VA = _s4mod("vadapter_s4q")
    ad = VA.VAdapter(VA.ENC_HIDDEN, H, hidden, VA.STACK, target_rms)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in ad.named_parameters(prefix="vadapter"):
            kind, std = VA.init_rule(name, tuple(p.shape))
            p.copy_(torch.zeros_like(p) if kind == "zeros" else torch.randn(p.shape, generator=g) * std)
    return ad.to(dev, torch.bfloat16).eval().requires_grad_(False)


def load_adapter_from_export(ck, H, dev, hidden=4096):
    """vadapter.* of a stage-4 trainer export (DCP) -> VAdapter (None if the export has none)."""
    import torch.distributed.checkpoint as dcp
    md = dcp.FileSystemReader(ck).read_metadata()
    keys = [k for k in md.state_dict_metadata if k.startswith("vadapter.")]
    if not keys:
        return None
    ad = make_adapter(H, dev, hidden=hidden)
    sd = {f"vadapter.{k}": v for k, v in ad.state_dict().items() if f"vadapter.{k}" in keys}
    dcp.load(sd, checkpoint_id=ck, no_dist=True)
    ad.load_state_dict({k[len("vadapter."):]: v for k, v in sd.items()}, strict=False)
    missing = [k for k in ad.state_dict() if f"vadapter.{k}" not in sd]
    assert not [k for k in missing if k != "target_rms"], ("export lacks adapter tensors", missing)
    return ad.to(dev, torch.bfloat16).eval().requires_grad_(False)


# ------------------------------------------------------------------------------------------------ the streaming Voxtral encoder
class VoxStream:
    """streaming Voxtral encoder for B parallel audio streams (module doc). State: frame counter k (host + device scalar), conv caches,
    per-layer KV caches of max_frames * 4 encoder frames. step(wav_win) -> latents [B, 4, 1280] of frame k, k += 1.
    wav_win [B, CTX24 + 1920 + LOOK24] = 24k samples [1920 k - CTX24, 1920 (k + 1) + LOOK24) (zeros before the start)."""

    def __init__(self, enc, B, max_frames, dev, fa=None):
        self.enc, self.B, self.dev = enc, int(B), dev
        cfg = enc.config
        self.Hd, self.nh, self.D = cfg.hidden_size, cfg.num_attention_heads, cfg.head_dim
        self.kvh = getattr(cfg, "num_key_value_heads", self.nh) or self.nh
        assert cfg.sliding_window == WIN
        self.fe, self.melW, self.lmax = mel_filters_t(dev)
        self.win = torch.hann_window(N_FFT, device=dev)
        if fa is None:
            import pinned_fa3
            fa = pinned_fa3._load_module().flash_attn_interface
        self.fa = fa
        self.maxT = int(max_frames) * EPF + 8
        self.K = [torch.zeros(B, self.maxT, self.kvh, self.D, dtype=torch.bfloat16, device=dev) for _ in enc.layers]
        self.V = [torch.zeros(B, self.maxT, self.kvh, self.D, dtype=torch.bfloat16, device=dev) for _ in enc.layers]
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
            a = self.fa.flash_attn_with_kvcache(q, self.K[i], self.V[i], k=k, v=v, cache_seqlens=sl, softmax_scale=at.scaling,
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


def window(wav24, k, n=1):
    """the 24k window of frames k .. k+n-1 from a full waveform (np / torch [B, S] or [S]); zeros outside [0, S)."""
    x = torch.as_tensor(wav24)
    if x.dim() == 1:
        x = x[None]
    lo, hi = SPF24 * k - CTX24, SPF24 * (k + n) + LOOK24
    S = x.shape[-1]
    a, b = max(lo, 0), min(hi, S)
    out = torch.zeros(x.shape[0], hi - lo, dtype=torch.float32, device=x.device)
    if b > a:
        out[:, a - lo:b - lo] = x[:, a:b].float()
    return out


def window_plain(wav24, f):
    """the 1920 samples of frame f (Mimi's input), zeros past the end."""
    return window(wav24, f, 1)[:, CTX24:CTX24 + SPF24]


@torch.no_grad()
def full_reference(enc, wav24, n_frames, dev):
    """the full-sequence path = what the trainer computes (voxtral_front.VoxtralFront.encode's steps, hera's encode_frames: resample,
    centred log-mel, HF encoder forward over the whole input) -> [B, 4 n, 1280]."""
    fe, _, _ = mel_filters_t(dev)
    x = torch.as_tensor(wav24, dtype=torch.float32, device=dev)
    if x.dim() == 1:
        x = x[None]
    want = n_frames * SPF24
    x = F.pad(x[..., :want], (0, max(0, want - x.shape[-1])))
    w16 = resample(x)[..., : n_frames * SPF16]
    mels = fe._torch_extract_fbank_features(w16, device=str(w16.device), center=True).to(dev, torch.bfloat16)
    lat = enc(input_features=mels).last_hidden_state
    assert lat.shape[1] == n_frames * EPF, (lat.shape, n_frames)
    return lat


# ------------------------------------------------------------------------------------------------ Mimi (A's codes)
class MimiStream:
    """streaming Mimi encode of B streams: push(frame [B, 1920]) -> codes [B, 32] (local ids 0..2047)."""

    def __init__(self, dev, mimi=None, nq=32):
        from transformers import MimiModel
        self.mimi = mimi if mimi is not None else MimiModel.from_pretrained(MIMI_PATH).to(dev).eval()
        self.mimi.requires_grad_(False)
        self.dev, self.nq = dev, nq
        self.reset()

    def reset(self):
        self.pc, self.kv = None, None

    @torch.no_grad()
    def push(self, frame):
        x = torch.as_tensor(frame, dtype=torch.float32, device=self.dev)
        if x.dim() == 1:
            x = x[None]
        o = self.mimi.encode(x[:, None], num_quantizers=self.nq, encoder_past_key_values=self.kv, padding_cache=self.pc,
                             use_streaming=True, return_dict=True)
        self.kv, self.pc = o.encoder_past_key_values, o.padding_cache
        c = o.audio_codes                                                                # [B, nq, 1]
        assert c.shape[-1] == 1, c.shape
        return c[..., 0]

    @torch.no_grad()
    def encode_full(self, wav24):
        x = torch.as_tensor(wav24, dtype=torch.float32, device=self.dev)
        if x.dim() == 1:
            x = x[None]
        n = x.shape[-1] // SPF24
        c = self.mimi.encode(x[:, None, :n * SPF24], num_quantizers=self.nq).audio_codes          # [B, nq, n]
        return c

    @torch.no_grad()
    def decode(self, codes):
        return self.mimi.decode(torch.as_tensor(codes, device=self.dev)[None] if codes.dim() == 2 else codes).audio_values[:, 0]


def lufs_normalize(wav, sr=SR24):
    """whole-file loudness match to the Mimi codes' preprocessing (voxtral_front.lufs_gain_db: ONE BS.1770-4 integrated loudness over the
    whole input -> gain to -23 LUFS, peak-capped at 0.995) -> (wav, info)."""
    VF = _vf()
    g, lin, status = VF.lufs_gain_db(np.asarray(wav, np.float32), sr)
    return VF.apply_gain(wav, g), {"lufs_in": lin, "gain_db": g, "status": status, "rule": "voxtral_front.lufs_gain_db"}


class RunningLoudness:
    """LIVE loudness match for a streamed input (no whole-episode measurement exists at inference). The training audio got ONE gain per
    (episode, speaker) = -23 LUFS integrated (BS.1770-4: K-weighting, 400 ms blocks with 75 % overlap, absolute gate -70 LUFS, relative
    gate -10 LU), peak-capped at 0.995. Live estimate of the same quantity:
      * K-weighting filters = pyloudnorm.Meter(sr)'s two biquads, run causally with filter state across 80 ms frames;
      * 100 ms sub-block energies -> 400 ms block loudness every 100 ms (the 75 % overlap of BS.1770), kept in a growing history
        (the integrated measure of everything heard so far = what the offline rule converges to on a long session);
      * gated integrated loudness L of the history (both gates, as BS.1770) once >= `warm_s` of gated speech exists; before that the
        gain is `g0_db` (a fixed default for the capture chain, e.g. measured once per microphone);
      * target gain g = -23 - L, slewed at <= `slew_db_s` dB/s (no audible pumping, the adapter never sees a jump) and capped so the
        running peak x gain <= 0.995.
    The latent / code input of frame f uses the gain known BEFORE frame f (causal). Converges to the offline per-episode gain within a few
    seconds of speech; the residual early error is a level offset of a few dB, which the Mimi codes of training also span across episodes."""

    def __init__(self, sr=SR24, target=-23.0, g0_db=0.0, warm_s=3.0, slew_db_s=3.0, peak=0.995):
        import pyloudnorm as pyln
        from scipy.signal import lfilter_zi
        self.sr, self.target, self.peak = sr, target, peak
        self.filt = [(f.b, f.a) for f in pyln.Meter(sr)._filters.values()]
        self.zi = [lfilter_zi(b, a) * 0.0 for b, a in self.filt]
        self.sub = int(0.1 * sr)
        self.buf = np.zeros(0)
        self.e100 = []                                       # mean square per 100 ms sub-block (K-weighted)
        self.blocks = []                                     # 400 ms block mean squares
        self.g_db, self.g0, self.warm_s, self.slew = float(g0_db), float(g0_db), warm_s, slew_db_s
        self.pk = 0.0

    def _integrated(self):
        z = np.asarray(self.blocks)
        lk = -0.691 + 10 * np.log10(z + 1e-20)
        z = z[lk > -70.0]
        if len(z) * 0.1 < self.warm_s:
            return None
        lr = -0.691 + 10 * np.log10(z.mean()) - 10.0
        z2 = z[-0.691 + 10 * np.log10(z) > lr]
        return float(-0.691 + 10 * np.log10(z2.mean())) if len(z2) else None

    def push(self, frame):
        """frame: float32 [1920] raw input -> (gain-applied frame, gain_db used). Gain = state before this frame (causal)."""
        from scipy.signal import lfilter
        x = np.asarray(frame, np.float64)
        g = self.g_db
        y = x
        for i, (b, a) in enumerate(self.filt):
            y, self.zi[i] = lfilter(b, a, y, zi=self.zi[i])
        self.buf = np.concatenate([self.buf, y])
        while len(self.buf) >= self.sub:
            self.e100.append(float(np.mean(self.buf[:self.sub] ** 2))); self.buf = self.buf[self.sub:]
            if len(self.e100) >= 4:
                self.blocks.append(float(np.mean(self.e100[-4:])))
        self.pk = max(self.pk, float(np.abs(x).max()) if len(x) else 0.0)
        L = self._integrated()
        want = self.g0 if L is None else self.target - L
        if self.pk > 0:
            want = min(want, 20 * np.log10(self.peak / self.pk))
        step = self.slew * len(x) / self.sr
        self.g_db = float(np.clip(want, g - step, g + step))
        return (x * 10 ** (g / 20)).astype(np.float32), g
