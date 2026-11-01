"""STAGE-4 Voxtral ADAPTER (ported from ~/ref/hera/audio_adapter.py, canopy-labs-internal/hera ca4a93a; same parameter names, so a hera
AudioAdapter state dict - e.g. InternalCan/hera3-adapter-v1 - loads strictly). CPU-importable (no torch.distributed at import).

  latents [B, 4n, 1280]  (frozen Voxtral Realtime encoder, 50 Hz; 4 encoder frames = one 80 ms model frame = 1920 samples at 24 kHz)
  -> stack 4 consecutive encoder frames: [B, n, 5120]  (frame f of the model = encoder frames 4f .. 4f + 3, in order, concatenated)
  -> GatedMLP (SwiGLU): proj( silu(linear1 x) * linear2 x ),  5120 -> S4_VADAPT_HIDDEN (4096) -> 5120   (nn.Linear WITH bias, as hera)
  -> RMSNorm (no affine) x target_rms     (target_rms = a fixed fp32 BUFFER, not trained: the RMS of channel A's text-embedding rows,
                                           so the adapter's output enters the summed trunk input at the scale A's text did)
  NOT zero-init: the adapter REPLACES channel A's text embedding (a zero start = a dead channel; hera's rule).

INIT (S4_VADAPT_INIT): 'fresh' (default, user decision 10-07) = linear weights ~ N(0, 1/fan_in), biases 0, seeded per tensor name
(H2.init_rows_, identical on every rank / any sharding); 'hera:<dir or .safetensors>' = strict load of a hera AudioAdapter state dict
(keys audio_adapter.* or bare), sizes must match (1280 x 4 -> 4096 -> 5120); target_rms then STILL comes from S4_VADAPT_RMS unless
S4_VADAPT_RMS=ckpt (keep the checkpoint's)."""
import os

import torch
import torch.nn.functional as F
from torch import nn

ENC_HIDDEN, STACK = 1280, 4
SAMPLES_PER_FRAME_24K = 1920


class GatedMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(input_dim, hidden_dim)
        self.proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.proj(F.silu(self.linear1(x)) * self.linear2(x))


def stack_frames(latents, stack=STACK):
    """[B, t, h] -> [B, t // stack, h * stack]: model frame f = encoder frames stack*f .. stack*f + stack - 1 concatenated in time order."""
    b, t, h = latents.shape
    assert t % stack == 0, f"{t} encoder frames is not a multiple of {stack}"
    return latents.reshape(b, t // stack, h * stack)


class VAdapter(nn.Module):
    """hera AudioAdapter (module doc). forward(latents [B, 4n, 1280]) -> [B, n, H] in latents' dtype."""

    def __init__(self, encoder_hidden=ENC_HIDDEN, output_dim=5120, hidden_dim=4096, stack=STACK, target_rms=1.0, norm="rms"):
        super().__init__()
        self.stack = int(stack)
        self.mlp = GatedMLP(encoder_hidden * self.stack, hidden_dim or output_dim, output_dim)
        self.norm = nn.RMSNorm(output_dim, elementwise_affine=False)
        self.register_buffer("target_rms", torch.tensor(float(target_rms)))
        # stage4-asr2 10-13 (user-approved recipe change): norm="gain" = NO per-frame RMSNorm; out = mlp(x) * gain, gain ONE learnable scalar
        # (shape [1] so FSDP2 can shard it; fp32 master / Adam state like every adapter param), initialised by the trainer so the MEAN
        # per-frame output RMS on held-out clips = target_rms. norm="rms" = the original hera rule (every frame RMS = target_rms).
        self.norm_mode = str(norm)
        assert self.norm_mode in ("rms", "gain", "none"), self.norm_mode     # 10-13 user: "none" = the RAW mlp output (no norm, no scale)
        if self.norm_mode == "gain":
            self.gain = nn.Parameter(torch.ones(1))

    def forward(self, latents):
        x = stack_frames(latents, self.stack)
        if self.norm_mode == "none":
            out = self.mlp(x)
        elif self.norm_mode == "gain":
            out = self.mlp(x) * self.gain
        else:
            out = self.norm(self.mlp(x)) * self.target_rms
        return out.to(latents.dtype)


def init_rule(name, shape):
    """fresh init: weights N(0, 1/fan_in), biases 0 (name = FQN under the model, e.g. 'vadapter.mlp.linear1.weight')."""
    if name.endswith(".bias"):
        return "zeros", 0.0
    if name.endswith(".gain"):                                          # stage4-asr2 gain mode: 1.0 (re-set by the trainer's gain init)
        return "ones", 1.0
    assert name.endswith(".weight") and len(shape) == 2, (name, shape)
    return "normal", 1.0 / float(shape[1]) ** 0.5


def read_hera_state(path):
    """hera AudioAdapter tensors from a safetensors file or a directory of them -> {mlp.linear1.weight, ..., target_rms}."""
    from safetensors import safe_open
    files = [path] if path.endswith(".safetensors") else sorted(os.path.join(path, f) for f in os.listdir(path) if f.endswith(".safetensors"))
    assert files, ("no safetensors under", path)
    out = {}
    for fp in files:
        with safe_open(fp, framework="pt") as f:
            for k in f.keys():
                kk = k[len("audio_adapter."):] if k.startswith("audio_adapter.") else (k[len("context.audio_adapter."):] if k.startswith("context.audio_adapter.") else None)
                if kk is None and k.split(".")[0] in ("mlp", "norm", "target_rms"):
                    kk = k
                if kk is not None:
                    out[kk] = f.get_tensor(k)
    assert out, ("no audio_adapter tensors in", path)
    return out


def check_hera_shapes(sd, model):
    want = {k: tuple(v.shape) for k, v in model.state_dict().items()}
    got = {k: tuple(v.shape) for k, v in sd.items()}
    assert set(got) == set(want), ("hera adapter keys differ", sorted(set(got) ^ set(want)))
    for k in want:
        assert got[k] == want[k], ("hera adapter shape differs", k, got[k], want[k])
    return True
