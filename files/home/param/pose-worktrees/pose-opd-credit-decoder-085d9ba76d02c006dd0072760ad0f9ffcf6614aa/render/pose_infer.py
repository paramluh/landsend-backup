"""Speech + pose inference for the stage-2 model: text + a dataset clip, prefill its first N seconds, generate the rest.

python pose_infer.py --clip <clip key> --prefill 3.0 [--text "..."] [--ckpt EXPORT_DIR] [--out DIR] [--device cuda|cpu]
  clip key: v3 "<clip_id>/<segment_id>" or v2 "<clip_id>/<kind>/<segment>" (as in the pose caches' audio_id.txt)
  --text defaults to the clip's own transcript (all of it, including the prefilled part).

Sequence (positions from 0 at SOT): SOT text EOT, then speech slot s carries PAD + audio codes + pose codes of
frame s-3. The hidden state at slot s predicts frame s-2: audio cb0 (2049-way, END stops), audio cb1..31 via the
audio light tower, pose cb0 (2048-way) and pose cb1..31 via the pose light tower. Prefill = slots 0..P+2 with the
clip's real codes for frames < P, in one forward pass; generation starts at frame P.
Outputs: gen.mp4 (skeleton + audio), gt.mp4 (ground truth), gen.wav, codes.npz, meta.json.
"""
import argparse
import glob
import hashlib
import io
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "train"))
sys.path.append(os.path.expanduser("~/ref/pose-tokenizer"))     # appended: the repo has its own train.py
import pyarrow.parquet as pq  # noqa: E402
import train_pose as TP  # noqa: E402  (PoseModel, liger stub, attention registrations)
from sam_lib import _sample, load_tokenizer, mimi_decode  # noqa: E402

CB, DEPTH, END = 2048, 32, 2048
SOT, EOT, PAD = 151937, 151938, 151944
CODE_DELAY = 3
FPS = 12.5
DATA = os.path.expanduser("~/data/TikTok-pose-data")


def find_clip(key):
    """Load mimi + pose + metadata for a clip key straight from the source parquet (must be on this node)."""
    parts = key.split("/")
    clip_id = parts[0]
    video_id = clip_id.split("_")[0]
    g = int(hashlib.md5(video_id.encode()).hexdigest(), 16) % 2048
    ver = "pose-v3" if len(parts) == 2 else "pose-v2"
    path = f"{DATA}/{ver}/{g}-of-2048.parquet"
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not on this node; slice {g} lives on landsend-{g % 5}")
    pf = pq.ParquetFile(path)
    names = pf.schema_arrow.names
    kcols = ["clip_id", "segment_id"] if "segment_id" in names else ["clip_id", "kind", "segment"]
    for rg in range(pf.num_row_groups):
        t = pf.read_row_group(rg, columns=kcols).to_pylist()
        for i, r in enumerate(t):
            if "/".join(str(r[c]) for c in kcols) == key:
                row = pf.read_row_group(rg, columns=["pose", "mimi"]).slice(i, 1).to_pylist()[0]
                m = pq.read_table(io.BytesIO(row["mimi"])).to_pylist()[0]
                p = pq.read_table(io.BytesIO(row["pose"])).to_pylist()[0]
                audio = np.asarray(m["mimi_codes"], dtype=np.int64)                       # [32, F]
                pose = np.stack([np.asarray(p[f"tokens_cb{c}"], dtype=np.int64) for c in range(DEPTH)])
                return {"audio": audio, "pose": pose, "words": m["word_texts"],
                        "word_starts": m["word_start_seconds"], "global_state": np.asarray(p["global_state"], np.float32),
                        "orig_hw": (int(p["orig_h"]), int(p["orig_w"])), "path": path, "row_group": rg, "row": i}
    raise KeyError(f"{key} not found in {path}")


def load_model(ckpt, device):
    model = TP.PoseModel.from_pretrained(TP.BASE, dtype=torch.bfloat16)
    model.resize_token_embeddings(151936 + 2500, mean_resizing=False)
    model.lm_head = nn.Linear(model.config.hidden_size, CB + 1, bias=False, dtype=torch.bfloat16)
    model.add_pose()
    from safetensors.torch import load_file
    missing, unexpected = model.load_state_dict(load_file(os.path.join(ckpt, "model.safetensors")), strict=False)
    missing = [k for k in missing if not k.endswith("light_llm.lm_head.weight")]   # tied to the light embeddings
    assert not missing and not unexpected, (missing[:5], unexpected[:5])
    for cfg in (model.config, model.light_llm.config, model.pose_light.light_llm.config):
        cfg._attn_implementation = "sdpa"
        cfg.use_cache = True
    return model.to(device, torch.bfloat16).eval()


@torch.no_grad()
def light_sample(tower, h, cb0, temp, topk):
    """tower: object with projection, cb0_embedding, light_llm, light_heads. Returns codes [B, 31] for cb1..31."""
    w = tower.light_heads.weight.view(DEPTH - 1, CB, -1)
    seq = torch.stack([tower.projection(h), tower.cb0_embedding(cb0)], 1).to(h.dtype)
    lo = tower.light_llm.model(inputs_embeds=seq, use_cache=True)
    out = []
    for d in range(1, DEPTH):
        c = _sample(lo.last_hidden_state[:, -1] @ w[d - 1].T, temp, topk)
        out.append(c)
        if d == DEPTH - 1:
            break
        e = tower.light_llm.get_input_embeddings()((d - 1) * CB + c)
        lo = tower.light_llm.model(inputs_embeds=e[:, None], past_key_values=lo.past_key_values, use_cache=True)
    return torch.stack(out, 1)


@torch.no_grad()
def generate(model, text_ids, audio, pose, prefill_frames, max_frames, temp=0.8, topk=250, seed=0,
             pose_temp=None, pose_topk=None, pose_rep=1.0, rep_window=0):
    """audio/pose: LongTensor [32, F_real]; returns (audio [32, F], pose [32, F], ended).
    Pose cb0 has its own temperature/top-k and an optional repetition penalty over its previous codes
    (HF-style: positive logits / rep, negative * rep; rep_window 0 = whole history)."""
    pose_temp = temp if pose_temp is None else pose_temp
    pose_topk = topk if pose_topk is None else pose_topk
    dev = next(model.parameters()).device
    torch.manual_seed(seed)
    P = min(prefill_frames, audio.size(1))
    offs = torch.arange(DEPTH, device=dev) * CB
    emb = model.get_input_embeddings()
    out_a = torch.zeros(DEPTH, max_frames, dtype=torch.long, device=dev)
    out_p = torch.zeros(DEPTH, max_frames, dtype=torch.long, device=dev)
    out_a[:, :P], out_p[:, :P] = audio[:, :P].to(dev), pose[:, :P].to(dev)

    def slot_embed(t):          # input at the slot carrying frame t (t < 0: no codes)
        x = emb(torch.tensor([PAD], device=dev))
        if t >= 0:
            x = x + model.code_embedding(out_a[:, t] + offs).sum(0, keepdim=True).to(x.dtype)
            x = x + model.pose_code_embedding(out_p[:, t] + offs).sum(0, keepdim=True).to(x.dtype)
        return x

    prefix = emb(torch.tensor([SOT] + list(text_ids) + [EOT], device=dev))
    slots = torch.cat([slot_embed(s - CODE_DELAY) for s in range(P + CODE_DELAY)], 0)   # slots 0..P+2
    o = model.model(inputs_embeds=torch.cat([prefix, slots], 0)[None], use_cache=True)
    cache, h = o.past_key_values, o.last_hidden_state[:, -1]
    t, ended = P, False
    while t < max_frames:
        a0 = _sample(model.lm_head(h), temp, topk)
        if int(a0) == END:
            ended = True
            break
        out_a[0, t] = a0
        out_a[1:, t] = light_sample(model, h, a0, temp, topk)[0]
        lp = model.pose_lm_head(h).float()
        if pose_rep != 1.0 and t > 0:
            hist = out_p[0, max(0, t - rep_window) if rep_window else 0:t].unique()
            v = lp[0, hist]
            lp[0, hist] = torch.where(v > 0, v / pose_rep, v * pose_rep)
        p0 = _sample(lp, pose_temp, pose_topk)
        out_p[0, t] = p0
        out_p[1:, t] = light_sample(model.pose_light, h, p0, pose_temp, pose_topk)[0]
        o = model.model(inputs_embeds=slot_embed(t)[None], past_key_values=cache, use_cache=True)
        cache, h = o.past_key_values, o.last_hidden_state[:, -1]
        t += 1
    return out_a[:, :t].cpu(), out_p[:, :t].cpu(), ended


def render(pose_codes, wav, global_state, orig_hw, path, prefill_s=0.0, label=None):
    """Decode pose tokens to a 25 fps skeleton video, mux with 24 kHz audio. Frames before prefill_s get a marker."""
    import cv2
    import imageio_ffmpeg
    import soundfile as sf
    from pose_tokenizer import PoseTokenizer
    from pose_tokenizer.data.kinematic import (BODY_CONNECTIONS, FACE_CONNECTIONS, LEFT_HAND_CONNECTIONS,
                                               RIGHT_HAND_CONNECTIONS, offsets_to_positions)
    if not hasattr(render, "tok"):
        render.tok = PoseTokenizer.from_pretrained("InternalCan/pose-tokenizer-32cb-200k-noshift",
                                                   revision="1d6ff4f0de89408cb054da618e309d4334596252", device="cpu")
    with torch.no_grad():
        recon = render.tok.decode([pose_codes[c].long() for c in range(DEPTH)]).float().numpy().reshape(-1, 55, 3)
    gs = np.repeat(global_state[:1], len(recon), 0)           # constant per clip (frame-0 root + shoulder width)
    pos = offsets_to_positions(gs[:, :2], recon[:, :, :2] * gs[:, 2:3, None])
    conf = recon[:, :, 2]
    H0, W0 = orig_hw
    h = 480
    w = int(round(h * W0 / H0 / 2) * 2)
    parts = [(FACE_CONNECTIONS, (80, 220, 255)), (BODY_CONNECTIONS, (90, 210, 120)),
             (LEFT_HAND_CONNECTIONS, (240, 110, 90)), (RIGHT_HAND_CONNECTIONS, (90, 150, 245))]
    tmp_v, tmp_a = path + ".v.mp4", path + ".a.wav"
    wr = imageio_ffmpeg.write_frames(tmp_v, (w, h), fps=25, codec="libx264", pix_fmt_out="yuv420p", quality=7,
                                     macro_block_size=2)
    wr.send(None)
    for f in range(len(pos)):
        img = np.full((h, w, 3), 22, np.uint8)
        y, x = pos[f, :, 0] * h, pos[f, :, 1] * w
        for conns, col in parts:
            for i, j in conns:
                if conf[f, i] > 0.25 and conf[f, j] > 0.25 and np.isfinite([x[i], y[i], x[j], y[j]]).all():
                    cv2.line(img, (int(x[i]), int(y[i])), (int(x[j]), int(y[j])), col, 2, cv2.LINE_AA)
        text = label or ("prefill (real)" if f / 25 < prefill_s else "generated")
        cv2.putText(img, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
        wr.send(np.ascontiguousarray(img))
    wr.close()
    sf.write(tmp_a, wav, 24000)
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", tmp_v, "-i", tmp_a, "-c:v", "copy",
                    "-c:a", "aac", "-b:a", "96k", "-shortest", "-movflags", "+faststart", path], check=True)
    os.remove(tmp_v)
    os.remove(tmp_a)


def latest_export():
    runs = sorted(glob.glob(os.path.expanduser("~/sam/runs/stage2-pose-fast/export-step-*")),
                  key=lambda p: int(p.rsplit("-", 1)[1]))
    return runs[-1] if runs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--prefill", type=float, default=3.0, help="seconds of the real clip to prefill")
    ap.add_argument("--text", default=None)
    ap.add_argument("--ckpt", default=None, help="consolidated export dir (default: newest export-step-*)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max_s", type=float, default=30.0)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--topk", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pose_temp", type=float, default=None, help="pose sampling temperature (default: --temp)")
    ap.add_argument("--pose_topk", type=int, default=None, help="pose top-k (default: --topk)")
    ap.add_argument("--pose_rep", type=float, default=1.0, help="repetition penalty on pose cb0 history")
    ap.add_argument("--rep_window", type=int, default=0, help="frames of pose history for the penalty (0 = all)")
    ap.add_argument("--tag", default="", help="suffix for the output folder")
    a = ap.parse_args()
    ckpt = a.ckpt or latest_export()
    out = a.out or os.path.join(HERE, "out-pose", a.clip.replace("/", "__") + f"__p{a.prefill:g}" + (f"__{a.tag}" if a.tag else ""))
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    clip = find_clip(a.clip)
    text = a.text or " ".join(w.strip() for w in clip["words"] if w.strip())
    tok = load_tokenizer()
    ids = tok(text, add_special_tokens=False)["input_ids"]
    model = load_model(ckpt, a.device)
    t1 = time.time()
    P = int(round(a.prefill * FPS))
    audio, pose = torch.tensor(clip["audio"]), torch.tensor(clip["pose"])
    ga, gp, ended = generate(model, ids, audio, pose, P, int(a.max_s * FPS), a.temp, a.topk, a.seed,
                             a.pose_temp, a.pose_topk, a.pose_rep, a.rep_window)
    t2 = time.time()
    wav = mimi_decode(ga[None], device=a.device)[0]
    wav_gt = mimi_decode(audio[None], device=a.device)[0]
    import soundfile as sf
    sf.write(os.path.join(out, "gen.wav"), wav, 24000)
    render(gp, wav, clip["global_state"], clip["orig_hw"], os.path.join(out, "gen.mp4"), prefill_s=P / FPS)
    render(pose, wav_gt, clip["global_state"], clip["orig_hw"], os.path.join(out, "gt.mp4"), label="ground truth")
    np.savez(os.path.join(out, "codes.npz"), audio=ga.numpy(), pose=gp.numpy(), prefill_frames=P)
    meta = {"clip": a.clip, "text": text, "prefill_s": P / FPS, "prefill_frames": P, "gen_frames": int(ga.size(1)) - P,
            "total_s": ga.size(1) / FPS, "real_s": audio.size(1) / FPS, "ended_with_END": ended, "ckpt": ckpt,
            "source": {k: clip[k] for k in ("path", "row_group", "row")}, "device": a.device,
            "load_s": round(t1 - t0, 1), "gen_s": round(t2 - t1, 1), "temp": a.temp, "topk": a.topk, "seed": a.seed,
            "pose_temp": a.pose_temp, "pose_topk": a.pose_topk, "pose_rep": a.pose_rep, "rep_window": a.rep_window}
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=1)
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
