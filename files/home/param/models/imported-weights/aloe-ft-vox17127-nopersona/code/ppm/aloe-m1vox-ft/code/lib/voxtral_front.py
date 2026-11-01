"""STAGE-4 channel-A audio front end: frozen Voxtral Realtime encoder (+ the loudness match to the Mimi codes).

API (everything a trainer needs)
--------------------------------
    import voxtral_front as VF
    front = VF.VoxtralFront(device="cuda")              # frozen, bf16, eval, requires_grad False, NOT an nn.Module
    wav = VF.load_row_audio(entry, root)                # float32 [1920 * n_frames] at 24 kHz, loudness-matched (see below)
    lat = front.encode(wav, n_frames)                   # bf16 [B, 4 * n_frames, 1280]   (no_grad; B = 1 for a 1-D wav)
    lat.view(B, n_frames, 4 * 1280)                     # the hera stacking: model frame f <- encoder frames 4f..4f+3

  * encode() takes a 24 kHz waveform [S] or [B, S] (float, any device; moved to the front's device), pads/crops it to exactly
    1920 * n_frames samples, resamples to 16 kHz (torchaudio-equivalent sinc, hera's kernel), computes Voxtral's log-mel ON THE
    GPU (same math as VoxtralRealtimeFeatureExtractor._torch_extract_fbank_features, asserted in self_test), runs the 32-layer
    causal encoder (sliding window 750 = 15 s) and returns last_hidden_state. Batched rows must share n_frames: pad shorter rows
    with zeros to the longest and slice their latents to 4 * own_n (the encoder is causal, so zero tail padding changes nothing
    before it except the very last mel frame's centre-pad, i.e. latents of a row's final frame only).
  * The front is deliberately not an nn.Module of the model (hera's AudioFrontEnd reasoning): FSDP would shard / all-gather /
    recompute 1.9 GB of weights that never change. Keep it per-rank, outside FSDP; feed the model the latents tensor.

Loudness (must match the Mimi codes)
------------------------------------
  The raw per-speaker flacs on GCS are PRE-normalisation. The stored Mimi codes were encoded (upstream worker
  mimi_incr_worker.apply_lufs_gain, recipe recorded in every speaker's streaming_mimi_codes/episode.mimi.json):
      ffmpeg -f f32le -ac 1 -ar 24000 per chunk, chunks concatenated  ->  ONE BS.1770-4 integrated loudness (pyloudnorm) over the
      WHOLE episode of that speaker  ->  ONE pure gain to -23 LUFS, capped so the peak stays <= 0.995 ('ok+peak_limited'),
      skipped for < 0.45 s  ->  zero-pad to a 1920 multiple  ->  Mimi streaming encode.
  lufs_gain_db() reproduces that measurement from the whole-episode audio; load_row_audio() uses the gain RECORDED in
  episode.mimi.json (lufs.lufs_gain_db, 3 decimals) because a row only has its own chunk. Verified in voxtral_tools/verify_lufs_mimi.py.
  => the stage-4 dataset must carry lufs_gain_db per (episode, speaker) (from episode.mimi.json); a row's chunk flac alone cannot
     reproduce it.

Timing / causality
------------------
  model frame f = 24 kHz samples [1920 f, 1920 (f+1)) = 16 kHz [1280 f, 1280 (f+1)) = mel frames 8f..8f+7 = encoder frames 4f..4f+3.
  The encoder is causal (causal convs + causal sliding-window attention). The STFT is centred (n_fft 400, hop 160, as the
  realtime processor does), so mel frame m sees 16 kHz samples up to 160 m + 200, and encoder frame j (conv1 k3, conv2 k3 s2,
  both left-padded) sees mel up to 2j + 1, i.e. audio up to 320 j + 360. The last encoder frame of model frame f therefore
  sees 40 samples (2.5 ms) of frame f + 1 (plus < 0.5 ms of resampler lookahead) -- negligible, and the same lookahead the
  realtime model has when streaming. Effective latency (how late a word's identity becomes readable) is measured in
  voxtral_tools/latency_probe.py; see S4_ASR_LAG there.
"""
_ALOE_ROOT = __import__("os").environ["ALOE_ROOT"]        # release: every path comes from aloe.env (ALOE_ROOT)
import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F

VOXTRAL_PATH = os.environ.get("S4_VOXTRAL_PATH") or os.environ.get("VOXTRAL_DIR") or f"{_ALOE_ROOT}/models/voxtral-realtime"   # aloe.env VOXTRAL_DIR
ENCODER_FILE = "audio_tower.safetensors"          # audio_tower.* only, cut out of mistralai/Voxtral-Mini-4B-Realtime-2602 model.safetensors
PREFIX = "audio_tower."
ATTN_IMPLEMENTATION = os.environ.get("S4_VOXTRAL_ATTN", "flex_attention")

SR_24K, SR_16K = 24000, 16000
SAMPLES_PER_FRAME_24K, SAMPLES_PER_FRAME_16K = 1920, 1280
ENC_FRAMES_PER_FRAME = 4                          # 12.5 Hz model frame = 4 x 50 Hz encoder frames
HIDDEN = 1280
SLIDING_WINDOW = 750                              # 15 s at 50 Hz (asserted against the checkpoint config)

LUFS_TARGET, LUFS_PEAK, LUFS_MIN_SEC = -23.0, 0.995, 0.45


# ------------------------------------------------------------------------------------------------ loudness (Mimi match)
def lufs_gain_db(wave, sr=SR_24K, target=LUFS_TARGET, peak=LUFS_PEAK, min_sec=LUFS_MIN_SEC):
    """The upstream Mimi preprocessing gain for ONE speaker's WHOLE episode (all chunks concatenated, float32 mono 24 kHz).

    pyloudnorm BS.1770-4 integrated loudness -> pure gain to `target`, capped so max|x| * gain <= `peak`; no gain for audio
    shorter than `min_sec` or with non-finite loudness (silence). Returns (gain_db, lufs_in, status)."""
    import pyloudnorm as pyln
    x = np.asarray(wave, dtype=np.float32).reshape(-1)
    if len(x) < min_sec * sr:
        return 0.0, float("nan"), "skipped_short"
    lufs_in = float(pyln.Meter(sr).integrated_loudness(x.astype(np.float64)))
    if not math.isfinite(lufs_in):
        return 0.0, lufs_in, "skipped_silent"
    gain = target - lufs_in
    pk = float(np.max(np.abs(x)))
    if pk > 0 and pk * 10.0 ** (gain / 20.0) > peak:
        return 20.0 * math.log10(peak / pk), lufs_in, "ok+peak_limited"
    return gain, lufs_in, "ok"


def apply_gain(wave, gain_db):
    """pure gain in dB on a float waveform (numpy or torch), float32 out."""
    g = 10.0 ** (float(gain_db) / 20.0)
    if isinstance(wave, torch.Tensor):
        return wave.to(torch.float32) * g
    return (np.asarray(wave, np.float32) * np.float32(g)).astype(np.float32)


def read_flac(path):
    """float32 mono 24 kHz, the same values ffmpeg -f f32le gives for integer PCM flac (int / 2^(bits-1))."""
    import soundfile as sf
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    assert sr == SR_24K, (path, sr)
    if x.ndim == 2:
        x = x.mean(1)
    return x


def episode_gain_db(episode_mimi_json):
    """the gain the Mimi codes were made with, as recorded upstream (rounded to 0.001 dB)."""
    d = json.load(open(episode_mimi_json)) if isinstance(episode_mimi_json, str) else episode_mimi_json
    lf = d["lufs"]
    assert float(lf["lufs_target"]) == LUFS_TARGET, lf
    return float(lf["lufs_gain_db"])


def load_row_audio(entry, root, gain_db=None):
    """A stage-4 test row's channel-A audio: float32 torch [1920 * n_frames] at 24 kHz, loudness-matched to its Mimi codes.

    entry: one record of testdata/manifest.json ('flac' and 'episode_mimi_json' relative to root, 'n_frames').
    Row frame f = samples [1920 f, 1920 (f+1)) of the chunk flac (chunks are exact 600 s = 7500-frame windows of the episode).
    The tail is zero-padded to a 1920 multiple, as the encoder worker did."""
    n = int(entry["n_frames"])
    if gain_db is None:
        gain_db = episode_gain_db(os.path.join(root, entry["episode_mimi_json"]))
    x = read_flac(os.path.join(root, entry["flac"]))
    want = n * SAMPLES_PER_FRAME_24K
    assert len(x) > want - SAMPLES_PER_FRAME_24K, (entry["row_key"], len(x), want)
    x = apply_gain(x[:want], gain_db)
    if len(x) < want:
        x = np.concatenate([x, np.zeros(want - len(x), np.float32)])
    return torch.from_numpy(x)


# ------------------------------------------------------------------------------------------------ resample (hera, verbatim math)
_RESAMPLE_KERNELS = {}


def _sinc_kernel(orig_freq, new_freq, width_mult=6, rolloff=0.99, device=None, dtype=torch.float32):
    """torchaudio's sinc_interp_hann kernel (torchaudio is not in the venv); hera streaming_encoder.py verbatim."""
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


def resample(wav, orig_freq, new_freq):
    """Polyphase resample of (batch, samples), matching torchaudio.functional.resample."""
    key = (int(orig_freq), int(new_freq), wav.device.type, wav.device.index, wav.dtype)
    if key not in _RESAMPLE_KERNELS:
        _RESAMPLE_KERNELS[key] = _sinc_kernel(orig_freq, new_freq, device=wav.device, dtype=wav.dtype)
    kernel, width, orig, new = _RESAMPLE_KERNELS[key]
    length = wav.shape[-1]
    padded = F.pad(wav, (width, width + orig))
    out = F.conv1d(padded[:, None], kernel, stride=orig)
    out = out.transpose(1, 2).reshape(wav.shape[0], -1)
    return out[..., : math.ceil(new * length / orig)]


# ------------------------------------------------------------------------------------------------ encoder
def _load_encoder(path, device, dtype, attn_implementation):
    from safetensors.torch import load_file
    from transformers import VoxtralRealtimeConfig
    from transformers.models.voxtral_realtime.modeling_voxtral_realtime import VoxtralRealtimeEncoder

    cfg = VoxtralRealtimeConfig.from_pretrained(path).audio_config
    assert cfg.sliding_window == SLIDING_WINDOW and cfg.hidden_size == HIDDEN, (cfg.sliding_window, cfg.hidden_size)
    sd = load_file(os.path.join(path, ENCODER_FILE), device="cpu")
    sd = {k[len(PREFIX):]: v.to(dtype) for k, v in sd.items() if k.startswith(PREFIX)}
    try:
        from accelerate import init_empty_weights   # leaves buffers (rotary inv_freq) real; see hera _empty_init
        ctx = init_empty_weights()
    except ImportError:
        import contextlib
        ctx = contextlib.nullcontext()
    with ctx:
        enc = VoxtralRealtimeEncoder._from_config(cfg, attn_implementation=attn_implementation)
    enc.load_state_dict(sd, strict=True, assign=True)
    enc = enc.to(device=device)
    # non-persistent buffers (rotary inv_freq) stay float32 as constructed; parameters are `dtype`
    enc.eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    return enc


class VoxtralFront:
    """Frozen Voxtral Realtime audio encoder: 24 kHz waveform -> [B, 4n, 1280] latents. Not an nn.Module (keep it out of FSDP)."""

    def __init__(self, path=VOXTRAL_PATH, device="cuda", dtype=torch.bfloat16, attn_implementation=ATTN_IMPLEMENTATION):
        from transformers.models.voxtral_realtime.feature_extraction_voxtral_realtime import VoxtralRealtimeFeatureExtractor
        self.device, self.dtype = torch.device(device), dtype
        self.fe = VoxtralRealtimeFeatureExtractor.from_pretrained(path)
        assert self.fe.sampling_rate == SR_16K and self.fe.hop_length == 160 and self.fe.n_fft == 400, vars(self.fe)
        self.encoder = _load_encoder(path, self.device, dtype, attn_implementation)
        self._mel_fb = torch.from_numpy(self.fe.mel_filters).to(self.device, torch.float32)   # [201, 128]
        self._window = torch.hann_window(self.fe.n_fft, device=self.device)

    @property
    def hidden_size(self):
        return HIDDEN

    def log_mel(self, wav16):
        """[B, S16] float32 on device -> [B, 128, S16 // 160] float32. FeatureExtractor._torch_extract_fbank_features(center=True),
        kept on the GPU (the extractor's own copy round-trips through the CPU)."""
        fe = self.fe
        stft = torch.stft(wav16, fe.n_fft, fe.hop_length, window=self._window, return_complex=True, center=True)
        mag = stft[..., :-1].abs() ** 2
        mel = self._mel_fb.T @ mag
        log_spec = torch.clamp(mel, min=1e-10).log10()
        log_spec = torch.maximum(log_spec, torch.tensor(fe.global_log_mel_max, device=log_spec.device) - 8.0)
        return (log_spec + 4.0) / 4.0

    @torch.no_grad()
    def encode(self, wav24, n_frames):
        """24 kHz waveform [S] or [B, S] -> bf16 [B, 4 * n_frames, 1280] (no grad). Exact frame bookkeeping: 1920 n samples at
        24 kHz -> 1280 n at 16 kHz -> 8 n mel frames -> 4 n encoder frames."""
        if wav24.dim() == 1:
            wav24 = wav24.unsqueeze(0)
        wav24 = wav24.to(self.device, torch.float32)
        want24 = n_frames * SAMPLES_PER_FRAME_24K
        wav24 = F.pad(wav24[..., :want24], (0, max(0, want24 - wav24.shape[-1])))
        wav16 = resample(wav24, SR_24K, SR_16K)[..., : n_frames * SAMPLES_PER_FRAME_16K]
        mel = self.log_mel(wav16).to(self.dtype)
        lat = self.encoder(input_features=mel).last_hidden_state
        assert lat.shape[1] == n_frames * ENC_FRAMES_PER_FRAME, (lat.shape, n_frames)
        return lat

    @torch.no_grad()
    def encode_stacked(self, wav24, n_frames):
        """encode() reshaped to the model clock: [B, n_frames, 4 * 1280] (frame f = encoder frames 4f..4f+3, concatenated)."""
        lat = self.encode(wav24, n_frames)
        return lat.reshape(lat.shape[0], n_frames, ENC_FRAMES_PER_FRAME * HIDDEN)


@torch.no_grad()
def self_test(front, seconds=7.0, seed=0):
    """GPU log-mel == the feature extractor's; resample length bookkeeping; causality (future audio does not change past latents
    beyond the 1-frame lookahead). Returns a dict of max abs errors."""
    g = torch.Generator().manual_seed(seed)
    n = int(seconds * 12.5)
    w24 = torch.randn(1, n * SAMPLES_PER_FRAME_24K, generator=g) * 0.1
    w16 = resample(w24.to(front.device), SR_24K, SR_16K)[..., : n * SAMPLES_PER_FRAME_16K]
    ref = front.fe._torch_extract_fbank_features(w16[0].float().cpu(), device="cpu", center=True)
    mine = front.log_mel(w16)[0].cpu()
    out = {"mel_maxerr": float((ref - mine).abs().max()), "mel_frames": int(mine.shape[-1])}
    a = front.encode(w24, n)
    w24b = w24.clone(); w24b[..., (n // 2) * SAMPLES_PER_FRAME_24K:] = torch.randn(w24b.shape[-1] - (n // 2) * SAMPLES_PER_FRAME_24K, generator=g) * 0.1
    b = front.encode(w24b, n)
    d = (a.float() - b.float()).abs().amax(-1)[0]           # [4n]
    changed = torch.nonzero(d > 1e-3).flatten()
    out["first_changed_enc_frame"] = int(changed[0]) if len(changed) else -1
    out["perturb_start_enc_frame"] = (n // 2) * ENC_FRAMES_PER_FRAME
    out["latent_shape"] = list(a.shape)
    return out
