"""H5 codebook analysis helpers: data loading, CPU decode (pinned posev3 tokenizer), head/body metrics."""
import json
import os
import sys

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, "/data/home/amartyavarma/ref/pose-tokenizer")
from pose_tokenizer import PoseTokenizer  # noqa: E402
from pose_tokenizer.data.kinematic import offsets_to_positions  # noqa: E402

TOK = ("InternalCan/pose-tokenizer-32cb-200k-noshift", "1d6ff4f0de89408cb054da618e309d4334596252")   # = demo-v3 TOKENIZER
CTL = "/data/home/amartyavarma/video/studio-posemono/pose-ft-samples/s20/control"
REQ = "/data/home/amartyavarma/ppm/ppm-pose-server/requests"
RUNS = "/data/home/amartyavarma/video/studio-posemono/sam8b-eval-cfg/runs"
ROWS = ["S-05", "S-13", "S-15", "S-20", "E-03", "E-05", "E-09", "E-12", "E-13", "E-15", "E-18"]
SETS = ["fs", "d1", "d2", "d3"]
PF = 25          # prefill token frames
HEAD = [0, 1, 2, 3, 4]
BODY = [7, 8, 9, 10, 11, 12]
HANDS = list(range(13, 55))

_T = {}


def tok():
    if not _T:
        _T["p"] = PoseTokenizer.from_pretrained(TOK[0], device="cpu", revision=TOK[1])
        _T["p"].model.eval()
    return _T["p"]


def load_all():
    gt = {}
    for r in ROWS:
        z = np.load(f"{RUNS}/{r}/codes.npz")
        gt[r] = z["gt_pose"].astype(np.int64)
    gen = {}
    for s in SETS:
        m = json.load(open(f"{CTL}/xs17641_{s}.json"))
        gen[s] = {}
        for r in ROWS:
            rid, k = m[r]
            z = np.load(f"{REQ}/{rid}/s{k}/codes.npz")
            assert int(z["prefill_frames"]) == PF, (s, r, int(z["prefill_frames"]))
            gen[s][r] = z["pose"].astype(np.int64)
    return gt, gen


@torch.no_grad()
def contribs(codes):
    """codes [B, 32, F] -> per-codebook latent contributions [32, B, D, F] (RVQ: z_q = sum)."""
    q = tok().model.quantizer
    c = torch.as_tensor(codes).long()
    out = []
    for i in range(32):
        out.append(q.quantizers[i].out_proj(q.quantizers[i].decode_code(c[:, i])))
    return torch.stack(out)


@torch.no_grad()
def dec_lat(zq):
    """zq [B, D, F] -> offsets r3 [B, 2F, 55, 3] (numpy, shoulder-width units, conf in [..., 2])."""
    p = tok()
    kp = p.model.postprocess(p.model.decoder(zq))
    kp = p._destandardise(kp).float().numpy()
    B, T, _ = kp.shape
    return kp.reshape(B, T, 55, 3)


def dec_codes(codes, ncb=32):
    c = contribs(codes)
    return dec_lat(c[:ncb].sum(0))


def positions(r3):
    """r3 [..., T, 55, 3] -> positions [..., T, 55, 2] (y, x) with root 0, width 1."""
    off = r3[..., :2]
    return np.asarray(offsets_to_positions(np.zeros(off.shape[:-2] + (2,), np.float32), off))


def feats(pos, ear_scale=None, sw_scale=None):
    """pos [T, 55, 2] -> dict of head rel (normalised by ear dist), yaw proxy, body/hands rel (normalised by shoulder width)."""
    sm = (pos[:, 5] + pos[:, 6]) / 2
    rel = pos - sm[:, None]
    ear = np.linalg.norm(pos[:, 3] - pos[:, 4], axis=-1)
    sw = np.linalg.norm(pos[:, 5] - pos[:, 6], axis=-1)
    es = ear_scale if ear_scale is not None else float(np.median(ear))
    ss = sw_scale if sw_scale is not None else float(np.median(sw))
    yaw = (pos[:, 0, 1] - (pos[:, 3, 1] + pos[:, 4, 1]) / 2) / np.maximum(np.abs(pos[:, 3, 1] - pos[:, 4, 1]), 1e-3)
    return {"head": rel[:, HEAD] / es, "yaw": yaw, "body": rel[:, BODY] / ss, "hands": rel[:, HANDS] / ss,
            "head_sw": rel[:, HEAD] / ss, "es": es, "ss": ss}


def jitter(x, sig=2.0):
    """RMS of the high-frequency residual (x - gauss(x, sig)) along time axis 0, per point norm."""
    hf = x - gaussian_filter1d(x, sig, axis=0, mode="nearest")
    if hf.ndim == 1:
        return float(np.sqrt(np.mean(hf ** 2)))
    return float(np.sqrt(np.mean(np.sum(hf ** 2, -1))))


def lowpass_std(x, sig=12.0):
    lp = gaussian_filter1d(x, sig, axis=0, mode="nearest")
    if lp.ndim == 1:
        return float(np.std(lp))
    return float(np.sqrt(np.mean(np.sum((lp - lp.mean(0)) ** 2, -1))))


def code_stats(seqs, alpha=0.5):
    """seqs: list of int arrays (one codebook, several clips) -> unigram, entropy (bits), change rate, distinct."""
    cnt = np.zeros(2048)
    ch, n = 0, 0
    for s in seqs:
        np.add.at(cnt, s, 1)
        ch += int(np.sum(s[1:] != s[:-1]))
        n += len(s) - 1
    p = cnt / cnt.sum()
    nz = p[p > 0]
    return {"cnt": cnt, "H": float(-(nz * np.log2(nz)).sum()), "chg": ch / max(n, 1), "distinct": int((cnt > 0).sum())}


def kl(cnt_p, cnt_q, alpha=0.5):
    p = (cnt_p + alpha) / (cnt_p + alpha).sum()
    q = (cnt_q + alpha) / (cnt_q + alpha).sum()
    return float((p * np.log2(p / q)).sum())
