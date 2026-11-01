"""xabi-view (10-27): GT media + row layout for ~8 v2pure store clips (Xabi's data = Aloe pose release v2), read-only on the stores / cache.
Per clip: B side's stored Mimi codes -> infer_ppm.Mimi('cpu').decode (the GT-cell decode, models/mimi), B side's stored SAM pose codes ->
infer_ppm.decode_pose (pose tokenizer @ 1d6ff4f0, the cache item's frame-0 root + shoulder width), drawn with the GT cell's stick draw
(s20 stick_loop.draw: POSE_EDGES / POSE_COL, alpha by confidence, 25 fps; no prefill flash: the whole clip is real) on a canvas with the
source frame's aspect (480 px high), muxed H.264 + AAC. Words = the store row's B words (the word-lane source, clip-local times).
Row layout = the trainer's own micro (data_aloe_pose.Stores.conv, aloe_v2pure.env: A empty, voice-clip rule, CD 2).
   run.sh build.py OUTDIR      (compute node, CPU)"""
import json
import os
import subprocess
import sys

import numpy as np
import torch

torch.set_num_threads(4)
OUT = sys.argv[1]
os.makedirs(f"{OUT}/media", exist_ok=True)
sys.path.insert(0, "/data/home/amartyavarma/ppm/aloe-v2pure-ft/code")
import data_aloe_pose as A  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

D = A.D
CACHE = "/data/home/amartyavarma/aloe-pose/sam-cache-active-v2pure"
ROOT = A.STORE_DIR
COLS = ["row", "row_key", "split", "n_frames", "recording_id", "b_side", "src_cache_shard", "src_item", "src_key", "clip_start_sec",
        "clip_end_sec", "w_kept", "n_clips_rec"]

# ---------------- selection: 3 held-out + 5 train clips, distinct recordings, lengths spread around the 12.6 s median
tabs = []
for k in range(8):
    t = pq.read_table(f"{ROOT}/p{k}/rows.parquet", columns=COLS).to_pandas()
    t["store"] = k
    tabs.append(t)
import pandas as pd  # noqa: E402

T = pd.concat(tabs, ignore_index=True)
print("rows", len(T), T.groupby("split").size().to_dict(), "median s", float(np.median(T.n_frames) / 12.5), flush=True)
rng = np.random.default_rng(20261027)
PLAN = [("eval", 6.0), ("eval", 12.6), ("eval", 24.0), ("train", 3.5), ("train", 9.0), ("train", 12.6), ("train", 17.0), ("train", 34.0)]
used, pick = set(), []
for sp, sec in PLAN:
    n0 = sec * 12.5
    c = T[(T.split == sp) & (abs(T.n_frames - n0) <= 0.06 * n0) & (T.w_kept >= 4) & (~T.recording_id.isin(used))]
    r = c.iloc[int(rng.integers(len(c)))]
    used.add(r.recording_id)
    pick.append(r)
    print("pick", sp, sec, r.row_key, r.n_frames, r.b_side, flush=True)

sys.path.insert(0, "/data/home/amartyavarma/ppm/code")
import infer_ppm as I  # noqa: E402
import cv2  # noqa: E402
import imageio_ffmpeg  # noqa: E402
import soundfile as sf  # noqa: E402

sys.path.insert(0, "/data/home/amartyavarma/ref/pose-tokenizer")
from pose_tokenizer import PoseTokenizer  # noqa: E402

ptok = PoseTokenizer.from_pretrained(I.POSE_TOK[0], device="cpu", revision=I.POSE_TOK[1])
mimi = I.Mimi("cpu")
FF = imageio_ffmpeg.get_ffmpeg_exe()
from transformers import AutoTokenizer  # noqa: E402

TOK = AutoTokenizer.from_pretrained("/data/home/amartyavarma/models/Qwen3-32B")
cfg = A.aloe_cfg()
S = A.Stores()


def draw(p, cf, H, W):
    """= s20 stick_loop.draw (the GT cell), minus the prefill flash / label (all real here)."""
    img = np.full((H, W, 3), 24, np.uint8)
    for g_, edges in I.POSE_EDGES.items():
        col = I.POSE_COL[g_]
        for i, j in edges:
            al = 0.15 + 0.85 * min(cf[i], cf[j])
            if al < 0.2:
                continue
            cl = tuple(int(v * al) for v in col)
            cv2.line(img, (int(p[i, 1] * W), int(p[i, 0] * H)), (int(p[j, 1] * W), int(p[j, 0] * H)), cl, 3 if g_ == "body" else 1, cv2.LINE_AA)
    cv2.putText(img, "REAL", (8, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 200, 255), 1, cv2.LINE_AA)
    return img


def tokname(t):
    t = int(t)
    if t == D.PAD_ID:
        return "PAD"
    if t == D.NW_ID:
        return "NW"
    nm = {151676: "<sosa>", 151677: "<eosa>", 151678: "<sosb>", 151679: "<eosb>", 151680: "<sop>", 151681: "<eop>", 151672: "BOUNDARY"}
    if t in nm:
        return nm[t]
    s = TOK.convert_ids_to_tokens(t)
    if s is None:
        return f"<{t}>"
    return s.replace("Ġ", "␣").replace("Ċ", "\\n")


clips, layout, stats = [], None, {"first_word_frame0": 0, "voice_drawn_no_clip": 0, "voice_clip": 0}
for ci, r in enumerate(pick):
    k, rr, side = int(r.store), int(r.row), r.b_side
    sw = 1 if side == "user" else 0
    node, st, sib, ps, pz, b_ok = S.open(k)
    row = st[rr]
    assert row["row_key"] == r.row_key
    n = int(row["n_frames"])
    codes = np.asarray(row[f"codes_{side}"][:, :n]).astype(np.int64)
    pose = np.asarray(pz[side][rr][:, :n]).astype(np.int64)
    words = row[f"words_{side}"]
    it = pq.read_table(f"{CACHE}/{r.src_cache_shard}/items.parquet").slice(int(r.src_item), 1).to_pylist()[0]
    assert it["key"] == r.src_key and int(it["n"]) == n, (it["key"], r.src_key, it["n"], n)
    oh, ow = float(it["orig_h"]), float(it["orig_w"])
    H = 480
    W = int(round(H * ow / oh / 2)) * 2
    tag = f"c{ci + 1}-{r.split}-{r.recording_id[:8]}-{row['row_key'][-6:]}"
    wav = mimi.decode(codes)
    wp = f"{OUT}/media/{tag}.wav"
    sf.write(wp, wav, 24000)
    norm = {"frame0_root": it["frame0_root"], "frame0_width": it["frame0_width"], "orig_w": ow, "orig_h": oh}
    pos, conf = I.decode_pose(ptok, pose, norm, "cpu")
    NF = pos.shape[0]
    fr = np.stack([draw(pos[f], conf[f], H, W) for f in range(NF)])
    mp = f"{OUT}/media/{tag}.mp4"
    subprocess.run([FF, "-y", "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", "25", "-i", "-",
                    "-i", wp, "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264", "-crf", "23", "-preset", "veryfast", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-b:a", "96k", "-t", f"{NF / 25:.2f}", "-movflags", "+faststart", mp + ".tmp.mp4"],
                   input=np.ascontiguousarray(fr).tobytes(), check=True)
    os.replace(mp + ".tmp.mp4", mp)
    os.remove(wp)
    m = S.conv((k, rr, sw), cfg, 0)
    pv = json.loads(m["prov"])
    stats["voice_clip"] += int(bool(pv["clips"]))
    stats["voice_drawn_no_clip"] += int(bool(pv["voice_drawn"]) and not pv["clips"])
    stats["first_word_frame0"] += int(bool(words) and int(words[0]["start_frame"]) == 0)
    rms = float(np.sqrt(np.mean(wav ** 2))) if len(wav) else 0.0
    cw = {"tag": tag, "row_key": row["row_key"], "store": k, "row": rr, "split": r.split, "rid": r.recording_id, "side": side,
          "speaker": row[f"{side}_speaker_dir"], "n_frames": n, "dur_s": n / 12.5, "orig_w": ow, "orig_h": oh, "canvas": [W, H],
          "clip_start_sec": float(r.clip_start_sec), "clip_end_sec": float(r.clip_end_sec), "n_clips_rec": int(r.n_clips_rec),
          "words": [{"t": w["text"], "s": float(w["start"]), "e": float(w["end"]), "f": int(w["start_frame"])} for w in words],
          "cache_text": it["text"], "P": int(m["P"]), "T": int(len(m["pid"])), "voice": pv["clips"], "voice_drawn": pv["voice_drawn"],
          "rms": rms, "pose_conf_mean": float(conf.mean()), "mp4_bytes": os.path.getsize(mp), "frame0_ok": bool(it["frame0_ok"]),
          "crossbleed": it["audio_crossbleed_status"], "lufs": it["lufs_status"]}
    clips.append(cw)
    print("done", tag, n, "P", m["P"], "voice", pv["clips"][:1], "bytes", cw["mp4_bytes"], "rms", round(rms, 4), "conf", round(cw["pose_conf_mean"], 3),
          "size", ow, oh, flush=True)
    if layout is None and int(m["P"]) > 5 and n >= 60:
        P = int(m["P"])
        F = 27                                                       # conversation positions P .. P + 26 = frames 0..26 (~2.1 s)
        off = np.arange(D.DEPTH) * D.CBS
        dep, pdp = np.asarray(m["depth"][1]), np.asarray(m["pdepth"])
        loc = lambda a: [int(x - o) if x >= 0 else -1 for x, o in zip(a, off)]  # noqa: E731
        sysblk = [{"pos": i, "tok": tokname(m["ids"][1][i]), "id": int(m["ids"][1][i]), "mimi": int(dep[i, 0] >= 0)} for i in range(P)]
        conv = []
        for j in range(F):
            p_ = P + j
            conv.append({"pos": p_, "f": j, "t": round(j / 12.5, 2), "a_id": int(m["ids"][0][p_]), "a_present": float(m["present"][0][p_]),
                         "b_tok": tokname(m["ids"][1][p_]), "b_id": int(m["ids"][1][p_]), "mimi": loc(dep[p_]), "pose": loc(pdp[p_])})
        layout = {"tag": tag, "row_key": row["row_key"], "P": P, "T": int(len(m["pid"])), "n": n, "sys": sysblk, "conv": conv, "voice": pv["clips"],
                  "sys_mimi_frames": int((dep[:P, 0] >= 0).sum()), "kind": "pose", "cd": int(D.CD)}

# cheap counts on 500 held-out clips of the two row-level quirks (first word at frame 0 -> no NEW_WORD marker; voice drawn but none found)
ev = T[T.split == "eval"].sample(n=500, random_state=7)          # 500 random held-out clips (light IO)
fw0 = vd = vc = ne = 0
for _, r in ev.iterrows():
    k, rr, side = int(r.store), int(r.row), r.b_side
    if int(r.n_frames) < 2:
        continue
    m = S.conv((k, rr, 1 if side == "user" else 0), cfg, 0)
    pv = json.loads(m["prov"])
    P = int(m["P"])
    ne += 1
    vc += int(bool(pv["clips"]))
    vd += int(bool(pv["voice_drawn"]) and not pv["clips"])
    lane = np.asarray(m["ids"][1][P:])
    first = np.flatnonzero(lane != D.PAD_ID)
    fw0 += int(len(first) > 0 and first[0] == 0 and lane[0] != D.NW_ID)
print("eval quirks", {"n": ne, "voice_clip": vc, "voice_drawn_no_clip": vd, "first_piece_at_frame0_no_NW": fw0}, flush=True)
json.dump({"clips": clips, "layout": layout, "stats": stats,
           "eval_quirks": {"n": ne, "voice_clip": vc, "voice_drawn_no_clip": vd, "first_piece_at_frame0_no_NW": fw0},
           "counts": {"rows": len(T), "by_split": T.groupby("split").size().to_dict(),
                      "frames": T.groupby("split").n_frames.sum().astype(int).to_dict(),
                      "recs": T.groupby("split").recording_id.nunique().to_dict(), "median_s": float(np.median(T.n_frames) / 12.5),
                      "p10_s": float(np.percentile(T.n_frames, 10) / 12.5), "p90_s": float(np.percentile(T.n_frames, 90) / 12.5),
                      "max_s": float(T.n_frames.max() / 12.5)}},
          open(f"{OUT}/data.json", "w"), indent=1)
print("ALL DONE", flush=True)
