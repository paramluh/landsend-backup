"""FRAME-ALIGNED RoPE for interleaved backbones (10-17, user-approved; flag ILP_ROPE_POS_DIV). SELF-CONTAINED (torch only): copy this file
VERBATIM into any trainer that needs the same RoPE (e.g. ppm/next-arch), so every implementation is identical.

An interleaved row puts S slots per frame at backbone positions S*p + k (k = 0..S-1), per packed segment (restarting at 0). With div = S the
backbone RoPE uses the FLOAT positions (S*p + k) / div = p + k/S: every RoPE frequency divided uniformly by div (rope_theta unchanged), so two
frames N apart get exactly the summed model's angle at every frequency, and slot 0 of frame p sits exactly at p ((S*p)/S is exact in fp32).
cos / sin are computed in fp32 from the float positions (elementwise product with inv_freq: no integer table, no matmul / TF32), then cast to
the activation dtype, = the HF Qwen3RotaryEmbedding recipe (emb = cat(freqs, freqs), x attention_scaling). div = 1: callers keep their
original HF path (bit-identical old behaviour); frame_rope(div=1) also reproduces that path bit for bit (tests/test_rope_frame.py).
The 4B tower and the 0.6B decoder keep their own (frame-index / depth-index) RoPE: this is for the backbone only."""
import torch


def frame_positions(pos_int, div):
    """integer backbone positions [..., N] (per-segment 0..S*T_s - 1) -> fp32 positions pos / div."""
    return pos_int.to(torch.float32) / float(div)


def frame_rope(inv_freq, pos_int, div, dtype, attention_scaling=1.0):
    """inv_freq [D/2] (the rotary module's buffer, fp32), pos_int [B, N] or [N] integer backbone positions, div (slots per frame, e.g. 3)
    -> (cos, sin) [B, N, D] in `dtype`: angle[n, i] = inv_freq[i] * pos_int[n] / div, computed in fp32."""
    pos = frame_positions(pos_int if pos_int.dim() == 2 else pos_int[None], div)                  # [B, N] fp32
    with torch.autocast(device_type=pos.device.type, enabled=False):
        freqs = pos[..., None] * inv_freq.to(device=pos.device, dtype=torch.float32)[None, None, :]   # [B, N, D/2] fp32, elementwise
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * attention_scaling
        sin = emb.sin() * attention_scaling
    return cos.to(dtype), sin.to(dtype)
