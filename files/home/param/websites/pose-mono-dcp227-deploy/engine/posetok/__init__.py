# VENDORED for aloe-pose-monoplex-infer from pose-demo-website demo-v3 pose/ (working copy video/demo-v3-1019 @ ffbd8fc). Edits: weights from
# $ELECTRON_ENCODER / $POSE_TOKENIZER_DIR or the pinned HF revisions (sha256 still asserted); package renamed pose -> posetok.
"""posev3 pose tokens -> electron v3 pose latents, the path our sample folders were made with:
PoseTokenizer.decode (TF32 off) -> denormalise against frame 0 -> raw_to_recon -> render_pose_canvas 480x256
-> FastElectronEncoder(encoder-v2-pose), last frame repeated once (2F + 1 frames). Needs a CUDA GPU (the raster
kernel is JIT-built with ninja on first use).

    import pose; pose.load(); out = pose.tokens_to_latents(codes, frame0_pose)
    python -m pose POSE.safetensors FRAME0.safetensors OUT.safetensors

Weights (HF, private): vanshjjw/electron-vae encoder-v2-pose.safetensors (VANSH_HF_TOKEN, else the default token),
InternalCan/pose-tokenizer-32cb-200k-noshift (default token).
"""
import hashlib
import os

import numpy as np
import torch
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors.torch import load_file

from .encoder import ElectronEncoder
from .fast_encoder import FastElectronEncoder
from .kinematic import WHOLEBODY_INDEX, offsets_to_positions, raw_to_recon
from .pose_tokenizer import PoseTokenizer
from .raster import render_pose_canvas

ENCODER = ('vanshjjw/electron-vae', '9f18375922780ad10fbc209af78c467a67866e69', 'encoder-v2-pose.safetensors')
TOKENIZER = ('InternalCan/pose-tokenizer-32cb-200k-noshift', '1d6ff4f0de89408cb054da618e309d4334596252')
H, W = 480, 256
SEL = WHOLEBODY_INDEX
LOCAL_ENCODER = os.environ.get('ELECTRON_ENCODER')   # a local copy of ENCODER's file; else hf_hub_download (private repo: needs an HF token with access)
_M = {}


def load():
    """Encoder first: its constructor sets cudnn.benchmark, which the tokenizer decode then also runs under."""
    if not _M:
        net = ElectronEncoder(640)
        ckpt = LOCAL_ENCODER or hf_hub_download(ENCODER[0], ENCODER[2], revision=ENCODER[1])
        assert hashlib.sha256(open(ckpt, 'rb').read()).hexdigest().startswith('5f73f266'), ckpt   # pin: pose/SOURCES.md
        net.load_state_dict({k.removeprefix('student.'): v for k, v in load_file(ckpt).items()}, strict=True)
        _M['enc'] = FastElectronEncoder(net)
        tdir = os.environ.get('POSE_TOKENIZER_DIR') or snapshot_download(TOKENIZER[0], revision=TOKENIZER[1])
        _M['tok'] = PoseTokenizer.from_pretrained(tdir, device='cuda')
    return _M


def global_state(kp0):
    """frame-0 root (shoulder midpoint, normalised y, x) and shoulder width: the tokenizer's normalisation.
    kp0 [133, 3] float32 (x, y, conf) px."""
    p = np.asarray(kp0, np.float32)[SEL]
    yx = np.stack([p[:, 1] / H, p[:, 0] / W], -1)
    root0, sw0 = (yx[5] + yx[6]) / 2, abs(yx[5, 1] - yx[6, 1])
    return torch.from_numpy(np.array([root0[0], root0[1], sw0 if sw0 > 0 else 1.0], np.float32))


def decode(tok, codes):
    flags = torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = False
    dec = tok.decode([c[None].cuda() for c in codes.unbind(0)]).float()[0].cpu()
    torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = flags
    return dec


@torch.no_grad()
def tokens_to_latents(codes, frame0_pose):
    """codes int [32, F] posev3 tokens (odd F trimmed to even), frame0_pose float [133, 3] (y, x, conf) px ->
    {'latents' bf16 [2F/4 + 1, 48, 30, 16], 'n_frames' 2F + 1, 'valid_frames' 2F}."""
    m = load()
    codes = torch.as_tensor(codes).long()
    codes = codes[:, :codes.shape[1] // 2 * 2]
    assert codes.ndim == 2 and codes.shape[0] == 32 and codes.shape[1] > 0, codes.shape
    yxc = torch.as_tensor(frame0_pose).float().numpy()
    kp0 = torch.from_numpy(np.ascontiguousarray(yxc[:, [1, 0, 2]]))          # [133, 3] (x, y, conf) px
    gs = global_state(kp0.numpy())

    dec = decode(m['tok'], codes)                                            # [2F, 165]
    T = len(dec)
    assert T == 2 * codes.shape[1] and torch.isfinite(dec).all()
    f = dec.view(T, 55, 3)
    g = gs.expand(T, 3)
    pos = offsets_to_positions(g[:, :2], f[..., :2] * g[:, None, 2:3])
    kp = torch.zeros(T, 133, 3)
    kp[:, SEL, :2] = torch.stack([pos[..., 1] * W, pos[..., 0] * H], -1)
    kp[:, SEL, 2] = f[..., 2]

    recon, g = raw_to_recon(torch.cat([kp, kp[-1:]]), kp0, H, W)
    canvas = render_pose_canvas(recon[None].cuda(), g[None].cuda(), H, W)
    z = m['enc'].encode(canvas)[0].transpose(0, 1).to(torch.bfloat16).contiguous().cpu()
    assert z.shape == (T // 4 + 1, 48, 30, 16), z.shape
    return {'latents': z, 'n_frames': torch.tensor(T + 1), 'valid_frames': torch.tensor(T)}
