"""aloe-nopose-ft (10-16): the NO-POSE ABLATION of job 22091 = THIS file (22091's trainer) with env QQA_POSE_B=0 (default 1 = 22091's
maths unchanged). QQA_POSE_B=0 changes ONLY the pose path, exactly as 21980's no-pose aloe trainer (aloe-ft/code/train_aloe.py) runs it:
  DATA      data_aloe_pose with QQA_POSE_B=0: same stores / items / plan / split / rank sharding / speech micro, no pose arrays, kind 'speech'.
  MODEL     no pose input on B (inp.pose never called: pdepth absent), no pose tower, no pose decoder, no decp_bb link; loss = text + W_CB0 cb0
            + the speech decoder nanosteps (x W_CB). Pose knobs W_POSE_CB0 / W_POSE_CB / LR_TOWER_P / DEC_P_LR / LR_DECP_BB must be 0 (asserted).
  OPT       groups trunk / spec / proj / audio / tower_s (21980's); speech decoder nanostep AdamW only.
  FROZEN    pose parts (tower_p, dec_p, pose_head, decp_bb, inp.pose) in no optimizer, never run, asserted grad-free and bit-identical (= 21980).
  LOG       rank 0 prints the resolved-config diff vs 22091's (QQA_REF_CFG = 22091's export-430/train_cfg.json).
aloe-pose-ft TRAINER (10-16): 21980's aloe-ft trainer (train_aloe.py) with POSE ON CHANNEL B, the pose parts TRAINING.
= aloe-ft/code/train_aloe.py with ONLY these changes (everything else, incl. the qqa recipe / tokens_step normalisers / plan rule, unchanged):
  DATA      data_aloe_pose: one micro per QUALIFYING B track per row (pose ok, no crossbleed into B), kind 'pose' = 21980's speech micro +
            B's pose at PPM's pose delay (= CD) exactly as PPM's pose rows (pdepth summed into the backbone input via inp.pose, pose cb0
            labels, pose 0.6B frames = all pose frames).
  MODEL     PPM's pose path as train_ppm.run_micro runs it on pose rows: speech tower on Mimi B + POSE TOWER on pdepth (cb0 CE x
            W_POSE_CB0), speech decoder + POSE DECODER nanosteps (x W_POSE_CB), D33 backbone link decp_bb (alpha 1.0) into the pose
            decoder's slot 0 (detached + accumulated input gradient fed back as hv_p's). D32 pose head stays OFF (PPM main D36): frozen.
  OPT       + groups tower_p (QQA_LR_TOWER_P) and decp_bb (QQA_LR_DECP_BB); inp.pose stays in 'audio' (model_ppm.group_of, as PPM);
            + the pose decoder's own nanostep AdamW (QQA_DEC_P_LR, same shape: warmup x K, linear decay to 0 at steps x K).
  NORM      pose cb0 and pose light normalised like cb0 / light: the STEP's totals over all ranks / WORLD (tokens_step).
  EVAL      + pose cb0 CE / top-1, pose light CE (eval_speech/pose_* keys; combined + per orientation as before).
  FROZEN    only pose_head (asserted grad-free and bit-identical).
Original aloe-ft doc follows.
aloe-ft TRAINER (10-16): PPM full fine-tune on Model 1's Aloe handover rows, BOTH sides as B (data_aloe: every train row twice per
epoch, once per orientation; loss on B only; PPM's sysprompt rule with both speakers' persona text), with qqa-ft's recipe (qqa.env).
= qqa-ft/code/train_qqa.py (job 21823 / 21832) with ONLY these changes:
  DATA      data_aloe plan (one 7,500-frame row = one micro per rank per step, identical plan on all ranks, node-local copies of all 4 stores
            on every node); the last step's spare ranks run DUMMY micros (zero loss weights, zero counts, zero decoder chunk weights: they add
            nothing to any loss / gradient / statistic; asserted zero).
  EVAL      the stores' 240 eval rows x both orientations (480 micros, round-robin over ranks, dummies on the short ranks), reported per
            orientation (eval_asstB = data assistant on B, eval_userB = data user on B) and combined (eval), at step 0, every
            QQA_EVAL_EVERY steps, every export step and the final step.
  EXPORTS   intermediate weights-only exports at QQA_EXPORT_STEPS -> NODE-LOCAL bf16 shards (ppm-local-bf16-shards-v1, world = this run's,
            rank<R>.pt on each node's ALOE_LOCAL_EXPORT_ROOT/export-<step>, rep.pt + meta.json on rank 0's node; = ckpt_local_export's
            format, readable by train_qqa.load_init / infer_ppm load_weights), so /data never holds them; final = qqa's consolidated DCP.
  LOGGING   train/loss (the optimised per-step token-mean loss, text + cb0, mean over ranks) emitted.
Original qqa-ft doc follows.
qqa-ft TRAINER: run C's speech-duplex quick-QA fine-tune (s4-qqa job 21169) on a PPM checkpoint, in PPM's own model / speech-row maths.
A trimmed copy of ppm/code/train_ppm.py (job 21765's code, snapshot in qqa-ft/code): same model (model_ppm), same per-micro forward / loss
terms (B text CE with PPM's factorised weights, speech-tower cb0 CE, 0.6B speech decoder by NANOSTEPS), FSDP2, bf16 compute / fp32 masters.
Differences from train_ppm (all forced by the run-C recipe, each listed in qqa-ft/STATUS.md's mapping table):
  DATA      speech micros only: ONE packed micro per rank per step = run C's conversations of that step (qqa-ft/plan/PLAN-<host>.json, run C's
            order), each conversation a PPM speech row (data_qqa: swap 0, empty system block, CD 2) in its own attention segment.
  LOSS NORM run C's S4_PACK_NORM=tokens_step: every target token of the step weighs the same. Per loss term the normaliser = the STEP's total
            over all ranks / WORLD (FSDP averages over ranks): text (B text targets), cb0 (B cb0 labels), and for the 0.6B decoder update j
            the step's total chunk-j frames over ranks / WORLD (decoder all-reduce = mean over ranks); its input gradient into the tower is the
            chunk gradients weighted M_j / N (M_j = chunk j frames over all ranks, N = all light frames) = d(token mean)/d(hv).
            Step-0 PACKED-LOSS GATE (run C's S4_PACK_LOSSCHECK): with micro-level token normalisers, each packed loss term (text / cb0 / light)
            == the token-weighted mean of the conversations' standalone losses (max rel err over ranks < 2e-3, asserted).
  INIT      every weight from a PPM local inference copy (ckpt_local_export.py: bf16 dim-0 FSDP shards of world 128 + rep.pt), resharded to
            this world; fresh AdamW (run C: fresh Adam from its init export).
  LR        run C's groups / peaks mapped (env QQA_LR_*), run C's schedule: linear warmup over QQA_WARMUP steps then linear decay to 0 at the
            last step (sched_s3q wsd, floor 0); decoder nanostep updates on the same shape in update units (warmup x K, decay to 0 at steps x K).
  FROZEN    pose parts (tower_p, dec_p, pose_head, decp_bb, inp.pose) are in no optimizer, never run, asserted grad-free and bit-identical.
  EVAL      B normalised word CE (+ text / cb0 / light CE) on run C's 96 held-out conversations (4 per rank, unpacked) at step 0, every
            QQA_EVAL_EVERY steps and the final step.
  SAVE      no optimizer checkpoint; weights-only consolidated bf16 DCP export at the final step (QQA_EXPORT, readable by infer_ppm) and on
            STOP_AFTER_SAVE (control file / SIGUSR1: export at the next step boundary, then exit cleanly)."""
import datetime
import hashlib
import json
import math
import os
import shutil
import signal
import socket
import sys
import time
import zoneinfo

import numpy as np
import torch
import torch.distributed as dist

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import model_ppm as M  # noqa: E402
import data_qqa as Q  # noqa: E402
import data_aloe_pose as A  # noqa: E402
import sched_s3q as SCH  # noqa: E402

D = Q.D


def E(k, d=None):
    v = os.environ.get(k, d)
    assert v is not None, ("missing env", k)
    return v


CFG = {
    "run_tag": E("QQA_RUN"), "log_root": E("QQA_LOG_ROOT"), "init": E("QQA_INIT"), "plan_dir": E("QQA_PLAN_DIR"),
    "export": E("QQA_EXPORT"), "control_dir": E("QQA_CONTROL_DIR"), "export_min_free_gb": float(E("QQA_EXPORT_MIN_FREE_GB", "300")),
    # 10-16 scaling (export keys only): weights-only exports at these optimizer steps -> <dirname(export)>/export-<step>; an export waits
    # (polling every 15 s, up to QQA_EXPORT_WAIT_MIN) for /data space above the floor instead of being refused (training maths unchanged)
    "export_steps": [int(x) for x in E("QQA_EXPORT_STEPS", "").split(",") if x], "export_wait_min": float(E("QQA_EXPORT_WAIT_MIN", "0")),
    "steps": int(E("QQA_STEPS")), "warmup": int(E("QQA_WARMUP")), "eval_every": int(E("QQA_EVAL_EVERY")),
    "w_text": float(E("QQA_W_TEXT_CE")), "w_cat_word": float(E("QQA_W_CAT_WORD")), "w_tim": float(E("QQA_W_TIM_CE")),
    "w_nw": float(E("QQA_W_NW_CE")), "w_eob": float(E("QQA_W_EOB_CE")), "w_cb0": float(E("QQA_W_CB0")), "w_cb": float(E("QQA_W_CB")),
    "w_pcb0": float(E("QQA_W_POSE_CB0")), "w_pcb": float(E("QQA_W_POSE_CB")), "decp_bb_alpha": float(E("QQA_DECP_BB_ALPHA")),
    "lr": {"trunk": float(E("QQA_LR_TRUNK")), "spec": float(E("QQA_LR_SPEC")), "proj": float(E("QQA_LR_PROJ")),
           "audio": float(E("QQA_LR_AUDIO")), "tower_s": float(E("QQA_LR_TOWER")), "tower_p": float(E("QQA_LR_TOWER_P")),
           "decp_bb": float(E("QQA_LR_DECP_BB"))},
    "clip": {"trunk": float(E("QQA_CLIP_TRUNK")), "spec": float(E("QQA_CLIP_SPEC")), "proj": float(E("QQA_CLIP_PROJ")),
             "audio": float(E("QQA_CLIP_AUDIO")), "tower_s": float(E("QQA_CLIP_TOWER")), "tower_p": float(E("QQA_CLIP_TOWER_P")),
             "decp_bb": float(E("QQA_CLIP_DECP_BB"))},
    "adam_beta2": float(E("QQA_ADAM_BETA2")),
    "dec_k": int(E("QQA_DEC_K")), "dec_lr": float(E("QQA_DEC_LR")), "dec_clip": float(E("QQA_DEC_CLIP")),
    "dec_p_lr": float(E("QQA_DEC_P_LR")), "dec_p_clip": float(E("QQA_DEC_P_CLIP")), "dec_seed": int(E("QQA_DEC_SEED")),
    "dec_micro_frames": int(E("QQA_DEC_MICRO_FRAMES", "300")), "dec_mb_max": int(E("QQA_DEC_MB_MAX", "470")),
    "dec_even": E("QQA_DEC_EVEN", "1") == "1", "dec_dyn": E("QQA_DEC_DYN", "1") == "1", "dec_fast": E("QQA_DEC_FAST", "1") == "1",
    "dec_compile_mode": E("QQA_DEC_COMPILE_MODE", "default"), "eval_light_max": int(E("QQA_EVAL_LIGHT_MAX", "2048")),
    "losscheck": int(E("QQA_LOSSCHECK", "1")), "losscheck_tol": float(E("QQA_LOSSCHECK_TOL", "2e-3")),
    "dl_workers": int(E("QQA_DL_WORKERS", "3")), "data": A.aloe_cfg(), "cd": D.CD,
    "local_export_root": E("ALOE_LOCAL_EXPORT_ROOT"), "stores": A.STORES,
    "pose_b": E("QQA_POSE_B", "1") == "1",                        # aloe-nopose-ft: 0 = the no-pose ablation (module doc)
}
POSE = CFG["pose_b"]
assert POSE == A.POSE_B, "trainer and data must agree on QQA_POSE_B"
if not POSE:
    _pk = {"w_pcb0": CFG["w_pcb0"], "w_pcb": CFG["w_pcb"], "lr_tower_p": CFG["lr"]["tower_p"], "dec_p_lr": CFG["dec_p_lr"], "lr_decp_bb": CFG["lr"]["decp_bb"]}
    assert all(v == 0.0 for v in _pk.values()), ("QQA_POSE_B=0 needs every pose weight / LR at 0", _pk)
CFG["sched"] = {"warmup": CFG["warmup"], "horizon": CFG["steps"], "min_lr_rate": 0.0, "s3_sched": "wsd", "wsd_decay_start": CFG["warmup"],
                "wsd_decay_end": CFG["steps"], "wsd_floor": 0.0, "wsd_shape": "linear"}
CFG["dec_sched"] = {"warmup": CFG["warmup"] * CFG["dec_k"], "horizon": CFG["steps"] * CFG["dec_k"], "min_lr_rate": 0.0, "s3_sched": "wsd",
                    "wsd_decay_start": CFG["warmup"] * CFG["dec_k"], "wsd_decay_end": CFG["steps"] * CFG["dec_k"], "wsd_floor": 0.0, "wsd_shape": "linear"}
if POSE:
    TRAIN_GROUPS = ("trunk", "spec", "proj", "audio", "tower_s", "tower_p", "decp_bb")
    FROZEN_PREFIX = ("pose_head.",)                               # PPM main D36: the D32 pose head is off (frozen, unused)
else:                                                             # no-pose ablation = 21980 (train_aloe.py): pose parts frozen, never run
    TRAIN_GROUPS = ("trunk", "spec", "proj", "audio", "tower_s")
    FROZEN_PREFIX = ("tower_p.", "dec_p.", "pose_head.", "decp_bb.", "inp.pose.")
RANK, WORLD, LOCAL = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), int(os.environ["LOCAL_RANK"])
LW = int(os.environ.get("LOCAL_WORLD_SIZE", "8"))
HOST = socket.gethostname().split(".")[0]
PT = zoneinfo.ZoneInfo("America/Los_Angeles")
LOGDIR = f"{CFG['log_root']}/{CFG['run_tag']}"
GiB = 2 ** 30
assert CFG["cd"] == 2, ("PPM code delay must be 2 (PPM_CD)", CFG["cd"])


def now_pt():
    return datetime.datetime.now(PT).strftime("%m-%d %H:%M:%S PT")


def log(*a):
    if RANK == 0:
        print(f"[{now_pt()}]", *a, flush=True)


def allsum(v, dev):
    t = torch.tensor(v, dtype=torch.float64, device=dev)
    dist.all_reduce(t)
    return t.tolist()


def frozen_name(n):
    n = n.replace("_checkpoint_wrapped_module.", "")
    return n.startswith(FROZEN_PREFIX)


def group_of(n):
    return None if frozen_name(n) else M.group_of(n)


# ------------------------------------------------------------------------------------------------ micro -> device + weights
ARR = ("ids", "depth", "present", "pdepth", "pid", "sel", "y", "word", "pad", "eob", "nw", "cb0", "light", "pcb0", "plight") if POSE else \
    ("ids", "depth", "present", "pid", "sel", "y", "word", "pad", "eob", "nw", "cb0", "light")


def prep(b, dev, S_text, S_cb0, S_pcb0=None):
    """= train_ppm.prep with the normalisers given: S_text (text targets), S_cb0 (cb0 labels), S_pcb0 (pose cb0 labels). tokens_step:
    the step totals / WORLD; micro-level ('tokens', loss check): this micro's own totals."""
    kd = "pose" if POSE else "speech"
    assert b["kind"] == kd, (b["kind"], kd)
    t = {"kind": kd, "src": kd, "segs": [int(x) for x in b["segs"]]}
    for k in ARR:
        t[k] = torch.as_tensor(np.asarray(b[k])).to(dev, non_blocking=True)
    t["pid"] = t["pid"][None]
    w = {k: t[k].float() for k in ("word", "pad", "eob", "nw")}
    S = float(S_text)
    t["wt"] = ((CFG["w_cat_word"] * w["word"] + CFG["w_tim"] * w["pad"] + CFG["w_eob"] * w["eob"] + CFG["w_nw"] * w["nw"]) / S).float()
    t["wn"] = (CFG["w_text"] * w["word"] / S).float()
    m = (t["cb0"] >= 0).float()
    t["cb0_w"] = m / float(S_cb0)
    if POSE:
        m = (t["pcb0"] >= 0).float()
        t["pcb0_w"] = m / float(S_pcb0)
    t["dummy"] = bool(b.get("dummy", 0))
    if t["dummy"]:                                                # aloe: a dummy micro carries zero weight in every loss term
        t["wt"], t["wn"], t["cb0_w"] = t["wt"] * 0.0, t["wn"] * 0.0, t["cb0_w"] * 0.0
        if POSE:
            t["pcb0_w"] = t["pcb0_w"] * 0.0
    return t


TK = ("word_ce", "word_norm", "n_word", "word_hit", "pad_ce", "n_pad", "nw_ce", "n_nw", "eob_ce", "n_eob")
CB_S, CB_P = (8, 16, 31), (1, 8, 16, 31)                            # per-codebook light CE logged (speech decoder / pose decoder), as PPM


def text_vec(t, st):
    w, p, n_, e = (t[k].float() for k in ("word", "pad", "nw", "eob"))
    return [((st[0] * w).sum()), ((st[3] * w).sum()), (w.sum()), ((st[1] * w).sum()), ((st[0] * p).sum()), (p.sum()),
            ((st[0] * n_).sum()), (n_.sum()), ((st[0] * e).sum()), (e.sum())]


def summarize(v, pre):
    d = dict(zip(TK, v[:len(TK)]))
    m = lambda a, b: d[a] / max(d[b], 1.0)  # noqa: E731
    k = f"{pre}_speech"
    out = {f"{k}/word_ce_norm": m("word_norm", "n_word"), f"{k}/word_ce": m("word_ce", "n_word"), f"{k}/word_top1": m("word_hit", "n_word"),
           f"{k}/pad_ce": m("pad_ce", "n_pad"), f"{k}/nw_ce": m("nw_ce", "n_nw"),
           f"{k}/text_ce": (d["word_ce"] + d["pad_ce"] + d["nw_ce"] + d["eob_ce"]) / max(d["n_word"] + d["n_pad"] + d["n_nw"] + d["n_eob"], 1.0),
           f"{k}/n_word": d["n_word"]}
    x = v[len(TK):]
    out[f"{k}/cb0_ce"] = x[0] / max(x[1], 1.0); out[f"{k}/cb0_top1"] = x[2] / max(x[1], 1.0)
    out[f"{k}/light_ce"] = x[3] / max(x[4], 1.0)
    for j, c in enumerate(CB_S):
        out[f"{k}/light_ce_cb{c}"] = x[5 + j] / max(x[4] / 31.0, 1.0)
    if not POSE:
        return out
    o = 5 + len(CB_S)                                             # pose (PPM summarize's pose keys): cb0 sum / n / hits, light sum / n, cb 1/8/16/31
    out[f"{k}/pose_cb0_ce"] = x[o] / max(x[o + 1], 1.0); out[f"{k}/pose_cb0_top1"] = x[o + 2] / max(x[o + 1], 1.0)
    out[f"{k}/pose_light_ce"] = x[o + 3] / max(x[o + 4], 1.0)
    for j, c in enumerate(CB_P):
        out[f"{k}/pose_light_ce_cb{c}"] = x[o + 5 + j] / max(x[o + 4] / 31.0, 1.0)
    return out


# ------------------------------------------------------------------------------------------------ nanosteps (token-weighted over the step)
def dec_lam():
    return SCH.make_lambda(True, CFG["dec_sched"])


class DecOpt:
    """= train_ppm.DecOpt (replicated decoder: grads = views of ONE flat fp32 buffer, one all-reduce per update), LR = peak x the run-C
    shaped schedule over decoder updates (micros x K + j). aloe-pose: one per decoder (dec_s: QQA_DEC_LR / QQA_DEC_CLIP, dec_p: QQA_DEC_P_LR /
    QQA_DEC_P_CLIP), the same schedule shape (both decoders run on every micro)."""

    def __init__(self, dec, peak, clip):
        self.ps = list(dec.parameters())
        assert not hasattr(self.ps[0], "to_local"), "replicated decoder expected (PPM_DEC_MODE=replicate)"
        self.flat = torch.zeros(sum(p.numel() for p in self.ps), dtype=torch.float32, device=self.ps[0].device)
        o = 0
        for p in self.ps:
            p.grad = self.flat[o:o + p.numel()].view_as(p); o += p.numel()
        self.opt = torch.optim.AdamW([{"params": self.ps, "lr": 0.0}], betas=(0.9, CFG["adam_beta2"]), eps=1e-8, weight_decay=0.0, fused=True)
        self.micros, self.peak, self.clip = 0, float(peak), float(clip)
        self.lam = dec_lam()

    def lr(self, s):
        return self.peak * self.lam(int(s))


def _mb_plan(m):
    """= train_ppm._mb_plan (dec_even / dec_dyn speed path; same chunk gradient)."""
    MB = CFG["dec_micro_frames"]
    mm = max(m, 1)
    if not CFG["dec_even"]:
        return [(a0, MB) for a0 in range(0, -(-mm // MB) * MB, MB)]
    cap = CFG["dec_mb_max"] or MB
    nmb = -(-mm // cap)
    sz = -(-mm // nmb)
    if not CFG["dec_dyn"]:
        sz = -(-sz // 32) * 32
    plan = [(i * sz, sz) for i in range(nmb) if i * sz < mm]
    return [(a0, max(z, 16)) for a0, z in plan]


def nanosteps_tw(dec, do, hv, depth, light, dummy=False, zb=None):
    """train_ppm.nanosteps (replicated, one decoder) with tokens_step weights: chunk j's frames weigh 1 / (31 x M_j / WORLD) (M_j = chunk-j
    frames summed over ranks) -> after the all-reduce mean the update's gradient = the mean CE over ALL ranks' chunk-j frames; the input
    gradient accumulates chunk gradients x M_j / N (N = all ranks' light frames) = d(token mean x WORLD / ...)/d hv, i.e. the tower /
    backbone receive the token-weighted decoder loss. zb (pose decoder, D33 link input): detached + accumulated exactly as hv (train_ppm
    _nano_setup / nanosteps_pair). -> (acc, stats, lrs, (M, N), acc_zb or None)."""
    K, dev = CFG["dec_k"], hv.device
    hd = hv.detach().requires_grad_()
    zd = zb.detach().requires_grad_() if zb is not None else None
    n = int(light.numel())
    perm = torch.randperm(max(n, 1), generator=torch.Generator().manual_seed(CFG["dec_seed"] * 1000003 + do.micros * 1009 + RANK))
    sizes = [int(perm[j::K].numel()) if n else 0 for j in range(K)]
    Mg = torch.tensor([0] * K if dummy else sizes, dtype=torch.float64, device=dev)      # aloe: a dummy micro adds no frames to any update
    dist.all_reduce(Mg)
    Mj = Mg.tolist(); N = float(sum(Mj))
    perm_d = perm.pin_memory().to(dev, non_blocking=True)
    acc = torch.zeros(hd.shape, dtype=torch.float32, device=dev)
    acc_z = torch.zeros(zd.shape, dtype=torch.float32, device=dev) if zd is not None else None
    st = {"sum": torch.zeros((), device=dev, dtype=torch.float32), "n": 0, "cb": torch.zeros(31, device=dev, dtype=torch.float32)}
    lrs = []
    for j in range(K):
        m = sizes[j]
        idx = perm_d[j::K] if m else perm_d[:0]
        hd.grad = None
        if zd is not None:
            zd.grad = None
        do.flat.zero_()
        den = 31.0 * Mj[j] / WORLD
        for a0, sz in _mb_plan(m):
            mv = max(0, min(sz, m - a0))
            fl = light.index_select(0, idx[a0:a0 + mv]) if mv else light[:1]
            if fl.numel() < sz:
                fl = torch.cat([fl, light[:1].expand(sz - fl.numel())])
            wf = torch.zeros(sz, dtype=torch.float32, device=dev)
            if mv and not dummy:
                wf[:mv] = 1.0 / den
            l_, s_, num_, cb_ = dec(hd, depth, fl, wf, n_valid=mv) if zd is None else dec(hd, depth, fl, wf, n_valid=mv, zb=zd)
            if mv and not dummy:
                st["sum"] += s_; st["n"] += num_; st["cb"] += cb_.detach().float()
            l_.backward()
        if hd.grad is not None and m and not dummy:
            acc += hd.grad.float() * (Mj[j] / N)
        if zd is not None and zd.grad is not None and m and not dummy:
            acc_z += zd.grad.float() * (Mj[j] / N)
        dist.all_reduce(do.flat)
        do.flat.div_(WORLD)
        lr = do.lr(do.micros * K + j)
        for g in do.opt.param_groups:
            g["lr"] = lr
        torch.nn.utils.clip_grad_norm_(do.ps, do.clip)
        do.opt.step()
        lrs.append(lr)
    hd.grad = None
    if zd is not None:
        zd.grad = None
    do.micros += 1
    return acc, st, lrs, (Mj, N), acc_z


class DecFast:
    def __enter__(self):
        self.prev = M.DEC_FAST["active"]; M.DEC_FAST["active"] = CFG["dec_fast"]

    def __exit__(self, *a):
        M.DEC_FAST["active"] = self.prev


def run_micro(model, b, rot, dos, grad, info=None):
    """= train_ppm.run_micro for a POSE micro (NM = 1, w_task 1): forward (speech + pose towers, D33 link), loss = text + W_CB0 cb0 +
    W_POSE_CB0 pose cb0; nanosteps of BOTH decoders (train: speech decoder on hv_s / Mimi B / light, then pose decoder on hv_p / pdepth /
    plight with the link zb_p; each decoder's own update sequence, = train_ppm with dec_pair's interleave off: same maths) or a seeded
    <= eval_light_max frame decoder eval each; ONE backward. -> (stats vector, loss value, extra)."""
    with (torch.enable_grad() if grad else torch.no_grad()):
        out = model(b, rot, rot, info=info)
    assert "phead" not in out, "D32 pose head must stay off (PPM main D36)"
    if not POSE:
        assert "hv_p" not in out and "zb_p" not in out, "no-pose ablation: the pose path must not run"
    loss = out["text"] + CFG["w_cb0"] * out["cb0"] + (CFG["w_pcb0"] * out["pcb0"] if POSE else 0.0)
    v = text_vec(b, out["text_stats"])
    extra = {}
    dummy = b.get("dummy", False)
    pairs = [("dec_s", model.dec_s, out["hv_s"], b["depth"][1], b["light"], CFG["w_cb"], None)]
    if POSE:
        pairs.append(("dec_p", model.dec_p, out["hv_p"], b["pdepth"], b["plight"], CFG["w_pcb"], out.get("zb_p")))
        assert pairs[1][6] is not None, "D33 decp_bb link expected (PPM main: on)"
    dvals = []
    roots, grads = [loss], [None]
    for name, dec, hv, depth, light, wcb, zb in pairs:
        if grad:
            with DecFast():
                up, acc, lrs, (Mj, N), upz = nanosteps_tw(dec, dos[name], hv, depth, light, dummy=dummy, zb=zb)
            roots.append(hv); grads.append((up * wcb).to(hv.dtype))
            if zb is not None:
                roots.append(zb); grads.append((upz * wcb).to(zb.dtype))
            dvals.append((acc["sum"], float(acc["n"]), acc["cb"]))
            extra.update({f"{name}_lrs": lrs, f"{name}_M": Mj, f"{name}_N": N})
        else:
            with torch.no_grad():
                fl = light[torch.randperm(light.numel(), generator=torch.Generator().manual_seed(1234))[:CFG["eval_light_max"]].to(light.device)]
                wz = torch.zeros(fl.numel(), device=light.device)
                _, s_, num_, cb_ = dec(hv, depth, fl, wz) if zb is None else dec(hv, depth, fl, wz, zb=zb)
            dvals.append((float(s_), float(num_), cb_.float()))
    v += [out["cb0_sum"], out["cb0_n"], out["cb0_hit"], dvals[0][0], dvals[0][1]]
    v += [dvals[0][2][c - 1] for c in CB_S]
    if POSE:
        v += [out["pcb0_sum"], out["pcb0_n"], out["pcb0_hit"], dvals[1][0], dvals[1][1]]
        v += [dvals[1][2][c - 1] for c in CB_P]
    if grad:
        torch.autograd.backward(roots, grads)
    dvc = loss.device
    v = torch.stack([torch.as_tensor(x, device=dvc).double().reshape(()) for x in v]).tolist()
    if dummy:                                                     # aloe: a dummy micro contributes nothing to any statistic
        assert float(loss) == 0.0, ("dummy micro with a non-zero loss", float(loss))
        v = [0.0] * len(v)
    return v, float(loss), extra


@torch.no_grad()
def micro_losses(model, mic, rot, dev):
    """one micro with MICRO-level token normalisers -> (text, cb0, light mean CE over ALL its light frames, pose cb0, pose light mean CE
    over ALL its pose frames), counts."""
    oc = np.asarray(mic["own"]).sum(0)
    b = prep(mic, dev, oc[0], oc[1], oc[3] if POSE else None)
    out = model(b, rot, rot)
    res = []
    decs = [(model.dec_s, out["hv_s"], b["depth"][1], b["light"], None)]
    if POSE:
        decs.append((model.dec_p, out["hv_p"], b["pdepth"], b["plight"], out["zb_p"]))
    for dec, hv, depth, light, zb in decs:
        s_tot, n_tot = 0.0, 0
        for a0 in range(0, light.numel(), 1024):
            fl = light[a0:a0 + 1024]
            wz = torch.zeros(fl.numel(), device=dev)
            _, s_, num_, _ = dec(hv, depth, fl, wz) if zb is None else dec(hv, depth, fl, wz, zb=zb)
            s_tot += float(s_); n_tot += int(num_)
        res.append((s_tot / max(n_tot, 1), n_tot))
    if not POSE:                                                  # = 21980's micro_losses
        return [float(out["text"]), float(out["cb0"]), res[0][0]], [float(oc[0]), float(oc[1]), float(res[0][1])]
    return ([float(out["text"]), float(out["cb0"]), res[0][0], float(out["pcb0"]), res[1][0]],
            [float(oc[0]), float(oc[1]), float(res[0][1]), float(oc[3]), float(res[1][1])])


def pack_loss_check(model, b, rot, dev):
    """run C's S4_PACK_LOSSCHECK: each packed loss term == the token-weighted mean of the standalone conversations' losses. Same number of
    forwards on every rank (FSDP): ranks with fewer conversations repeat conversation 0 (ignored)."""
    model.eval()
    singles = b["singles"]
    Pv, _ = micro_losses(model, b, rot, dev)
    K = len(singles)
    t = torch.tensor([float(K)], device=dev); dist.all_reduce(t, op=dist.ReduceOp.MAX); Kmax = int(t)
    NT = 5 if POSE else 3
    num, den = [0.0] * NT, [0.0] * NT
    for k in range(Kmax):
        L, N = micro_losses(model, singles[k if k < K else 0], rot, dev)
        if k < K:
            for i in range(NT):
                num[i] += N[i] * L[i]; den[i] += N[i]
    model.train()
    exp = [num[i] / den[i] for i in range(NT)]
    err = [abs(Pv[i] - exp[i]) / max(abs(exp[i]), 1e-12) for i in range(NT)]
    return Pv, exp, err, K


# ------------------------------------------------------------------------------------------------ init / export
@torch.no_grad()
def load_init(model, path, dev):
    """every parameter <- a ppm-local-bf16-shards-v1 copy (world-W0 dim-0 FSDP2 shards rank<r>.pt, concatenated in rank order = infer_ppm
    load_weights; rep.pt = the replicated decoders), resharded onto this world's FSDP2 shards (each rank reads only the rows it holds)."""
    meta = json.load(open(f"{path}/meta.json"))
    assert meta["format"] == "ppm-local-bf16-shards-v1" and os.path.isfile(f"{path}/COMPLETE"), path
    W0 = int(meta["world"])
    params = {n.replace("_checkpoint_wrapped_module.", ""): p for n, p in model.named_parameters()}
    # global row offsets of every tensor in each source rank file (FSDP2 = torch.chunk along dim 0)
    srcs = {}

    def src(r):
        if r not in srcs:
            srcs[r] = torch.load(f"{path}/rank{r}.pt", map_location="cpu", mmap=True, weights_only=True)
        return srcs[r]
    rep = torch.load(f"{path}/rep.pt", map_location="cpu", mmap=True, weights_only=True)
    done, nbytes = set(), 0
    for n, p in params.items():
        if n in rep:
            p.copy_(rep[n].to(device=dev, dtype=p.dtype)); done.add(n); continue
        R = p.shape[0]
        c0 = -(-R // W0)                                          # source chunk rows (torch.chunk)
        a, r = M._rows(p)
        loc = M._loc(p)
        got = 0
        if r:
            for s in range(a // c0, min(W0, -(-(a + r) // c0))):
                lo, hi = s * c0, min((s + 1) * c0, R)
                t = src(s)[n]
                assert t.shape[0] == hi - lo and tuple(t.shape[1:]) == tuple(p.shape[1:]), (n, s, tuple(t.shape), lo, hi)
                x0, x1 = max(lo, a), min(hi, a + r)
                if x1 > x0:
                    loc[x0 - a:x1 - a].copy_(t[x0 - lo:x1 - lo].to(device=dev, dtype=loc.dtype)); got += x1 - x0
                    nbytes += (x1 - x0) * t[0].numel() * 2
        assert got == r, ("rows not covered", n, got, r)
        done.add(n)
    miss = sorted(set(params) - done)
    assert not miss, ("init: parameters not in the checkpoint", miss[:6])
    srcs.clear()
    nonfin = torch.tensor([float(sum(int(not torch.isfinite(M._loc(p)).all()) for p in params.values()))], device=dev)
    dist.all_reduce(nonfin)
    assert float(nonfin) == 0, "non-finite init weights"
    t = torch.tensor([float(nbytes)], device=dev); dist.all_reduce(t)
    return meta, float(t) / 1e9


def tensor_digest(model, names):
    """sha256 over THIS rank's local values (FSDP shard / replicated tensor) of the named parameters -> hex (compared per rank)."""
    h = hashlib.sha256()
    for n, p in model.named_parameters():
        if n not in names:
            continue
        h.update(n.encode()); h.update(M._loc(p).detach().float().cpu().numpy().tobytes())
    return h.hexdigest()


def export_weights(model, step, path, min_free_gb):
    """consolidated bf16 DCP of every model tensor (= safety_ppm.export_weights' format, infer_ppm 'dcp' kind) at `path`; never deletes."""
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_model_state_dict
    msd = get_model_state_dict(model)
    size_gb = sum(v.numel() * 2 for v in msd.values()) / 1e9
    ok = [None]
    if RANK == 0:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tw = time.time()
        while True:
            free = shutil.disk_usage(os.path.dirname(path)).free / 1e9
            if free - size_gb >= min_free_gb or time.time() - tw > 60.0 * CFG["export_wait_min"] or os.path.exists(path):
                break
            time.sleep(15.0)
        if time.time() - tw > 1.0:
            log(f"[export] waited {time.time() - tw:.0f}s for /data space ({free:.0f} GB free)")
        ok = [(free - size_gb >= min_free_gb and not os.path.exists(path), free)]
    dist.broadcast_object_list(ok, src=0)
    good, free = ok[0]
    if not good:
        log(f"!!! [export] REFUSED {path}: {size_gb:.1f} GB with {free:.0f} GB free (floor {min_free_gb}) or the path exists")
        return None
    t0 = time.time()
    dcp.save({k: v.to(torch.bfloat16) for k, v in msd.items()}, checkpoint_id=path)
    dist.barrier()
    if RANK == 0:
        json.dump({"step": int(step), "decp_bb": 1, "decp_bb_alpha": 1.0, "run": CFG["run_tag"], "init": CFG["init"], "cfg": CFG},
                  open(f"{path}/train_cfg.json", "w"), indent=1, default=str)
        open(f"{path}/COMPLETE", "w").write(str(step))
    dist.barrier()
    log(f"[export] {path}: consolidated bf16 weights ({size_gb:.1f} GB, {time.time() - t0:.0f}s); /data free before {free:.0f} GB")
    return path


def export_local(model, step, path):
    """aloe: weights-only NODE-LOCAL export (ppm-local-bf16-shards-v1 = ckpt_local_export's format, world = WORLD): each rank writes its
    FSDP2 dim-0 shards (names without the checkpoint wrapper) to <path>/rank<R>.pt on its own node; rank 0 also writes rep.pt (the
    replicated tensors) + meta.json; COMPLETE on rank 0's node once every rank is done. Nothing on /data."""
    t0 = time.time()
    os.makedirs(path, exist_ok=True)
    sh, rp = {}, {}
    for n, p in model.named_parameters():
        k = n.replace("_checkpoint_wrapped_module.", "")
        if hasattr(p, "to_local"):
            sh[k] = p.to_local().detach().to(torch.bfloat16).cpu().contiguous().clone()
        elif RANK == 0:
            rp[k] = p.detach().to(torch.bfloat16).cpu().contiguous().clone()
    torch.save(sh, f"{path}/rank{RANK}.pt.tmp"); os.replace(f"{path}/rank{RANK}.pt.tmp", f"{path}/rank{RANK}.pt")
    nb = sum(t.numel() * 2 for t in sh.values()) + sum(t.numel() * 2 for t in rp.values())
    hosts = [None] * WORLD; dist.all_gather_object(hosts, HOST)
    if RANK == 0:
        torch.save(rp, f"{path}/rep.pt")
        json.dump({"format": "ppm-local-bf16-shards-v1", "src": "aloe-ft train_aloe export_local", "step": int(step), "world": WORLD, "hosts": hosts,
                   "run_tag": CFG["run_tag"], "init": CFG["init"], "cfg": CFG, "save_time": now_pt()}, open(f"{path}/meta.json", "w"), indent=1, default=str)
    dist.barrier()
    if RANK == 0:
        open(f"{path}/COMPLETE", "w").write(str(step))
    t = torch.tensor([float(nb)], device=next(model.parameters()).device); dist.all_reduce(t)
    log(f"[export-local] {path} on every node: {float(t) / 1e9:.1f} GB bf16 shards (world {WORLD}), {time.time() - t0:.0f}s")


# ------------------------------------------------------------------------------------------------ main
STOP = {"flag": False}


def _usr1(signum, frame):
    STOP["flag"] = True


def make_opt(model):
    by = {g: [] for g in TRAIN_GROUPS}
    frozen, dec = [], []
    for n, p in model.named_parameters():
        g = group_of(n)
        if g is None:
            frozen.append(n)
        elif g in ("dec_s", "dec_p"):
            dec.append(n)
        else:
            assert g in by, (n, g)
            by[g].append(p)
    opt = torch.optim.AdamW([{"params": by[g], "lr": CFG["lr"][g], "name": g} for g in TRAIN_GROUPS if by[g]], betas=(0.9, CFG["adam_beta2"]),
                            eps=1e-8, weight_decay=0.0, fused=True)
    lam = SCH.make_lambda(True, CFG["sched"])                     # every trained group warms up (run C: warmup_groups = all its groups)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, [lam for _ in opt.param_groups])
    return opt, sched, by, frozen, dec


def main():
    dev = torch.device("cuda", LOCAL)
    torch.cuda.set_device(dev)
    dist.init_process_group("nccl", device_id=dev, timeout=datetime.timedelta(seconds=int(E("INIT_TIMEOUT", "1800"))))
    os.makedirs(LOGDIR, exist_ok=True)
    signal.signal(signal.SIGUSR1, _usr1)
    log(f"aloe-ft {CFG['run_tag']}: world {WORLD} (lw {LW}); host {HOST}; POSE_B {int(POSE)}; cfg {json.dumps(CFG, default=str)}")
    ref = os.environ.get("QQA_REF_CFG")
    if ref and RANK == 0:                                         # aloe-nopose-ft: resolved-config diff vs the reference run (22091)
        rc = json.load(open(ref))["cfg"]

        def flat(d, pre=""):
            o = {}
            for k_, v_ in d.items():
                if isinstance(v_, dict):
                    o.update(flat(v_, f"{pre}{k_}."))
                else:
                    o[f"{pre}{k_}"] = v_
            return o
        a_, b_ = flat(json.loads(json.dumps(rc, default=str))), flat(json.loads(json.dumps(CFG, default=str)))
        diff = [(k_, a_.get(k_, "<absent>"), b_.get(k_, "<absent>")) for k_ in sorted(set(a_) | set(b_)) if a_.get(k_, "<absent>") != b_.get(k_, "<absent>")]
        log(f"[cfgdiff] resolved config vs {ref} ({len(diff)} keys differ; every other key identical):")
        for k_, x_, y_ in diff:
            log(f"[cfgdiff]   {k_}: ref {x_!r} -> this {y_!r}")
    # ---- data (aloe plan: identical on every rank; every train row x both orientations, one row per rank per step)
    PL = A.make_plan(WORLD)
    assert PL["steps"] == CFG["steps"], ("QQA_STEPS != the plan's steps (one epoch over the doubled rows)", CFG["steps"], PL["steps"])
    for k in range(A.NST):
        A.Stores().open(k)                                        # staged store digests == the build's (asserted)
    rows, ev_items = A.rank_items(PL, RANK, WORLD)
    fallback = PL["items"][0]
    ndum = sum(x is None for x in rows)
    nconv = allsum([float(len(rows) - ndum), float(len(ev_items)), float(ndum)], dev)
    assert int(nconv[0]) == len(PL["items"]) and int(nconv[1]) == len(PL["eval"]), nconv
    dg = hashlib.sha256(json.dumps(PL["items"]).encode()).hexdigest()[:16]
    dga = [None] * WORLD; dist.all_gather_object(dga, dg)
    assert len(set(dga)) == 1, "plan differs across ranks"
    log(f"[data] aloe-pose plan {dg} (POSE_B {int(POSE)}; no-pose: same items, no pose arrays): {PL['n_train_rows']:,} train rows -> {int(nconv[0]):,} qualifying B-track micros (pose ok, no bleed into B) "
        f"in {CFG['steps']} steps x {WORLD} ranks (1 row / rank / step; {int(nconv[2])} dummy micros on the last step); eval {PL['n_eval_rows']} rows "
        f"-> {int(nconv[1])} B micros; data cfg {CFG['data']}")
    # ---- model
    c32, c4, c06 = M.cfg32(64), M.cfg4b(36), M.cfg06(28)
    model, _ = M.build(c32, c4, c06, dev, log, partial=False, proj_r=2048, tower_vocab=M.TOWER_VOCAB, init=False, dec_ac=False,
                       dec_mode="replicate", pose_head=True, decp_bb=True)
    model.pose_head_on = False                                    # PPM main D36: the D32 pose head is off (frozen)
    model.decp_bb_alpha = CFG["decp_bb_alpha"]                    # PPM main D33: alpha 1.0, no fade
    t0 = time.time()
    meta, gb = load_init(model, CFG["init"], dev)
    log(f"[init] {CFG['init']}: PPM step {meta['step']} (run {meta.get('run_tag')}, saved {meta.get('save_time')}), world {meta['world']} shards "
        f"resharded to {WORLD}; {gb:.1f} GB read; {time.time() - t0:.0f}s")
    if CFG["dec_fast"]:
        for dec_ in ((model.dec_s, model.dec_p) if POSE else (model.dec_s,)):
            M.compile_decoder(dec_, CFG["dec_compile_mode"], whole=True, dynamic=CFG["dec_dyn"])
    rot = M.Rot(c32, dev)
    opt, sched, by, frozen, decn = make_opt(model)
    dos = {"dec_s": DecOpt(model.dec_s, CFG["dec_lr"], CFG["dec_clip"])}
    if POSE:
        dos["dec_p"] = DecOpt(model.dec_p, CFG["dec_p_lr"], CFG["dec_p_clip"])
    else:                                                         # no-pose: dec_p params are frozen (no nanostep AdamW over them)
        dec = [n for n in decn if not frozen_name(n)]
        assert all(n.replace("_checkpoint_wrapped_module.", "").startswith("dec_s.") for n in dec), dec[:4]
    n_all = len(list(model.parameters()))
    assert sum(len(v) for v in by.values()) + sum(len(d.ps) for d in dos.values()) + len(frozen) == n_all, "every parameter in exactly one optimizer or frozen"
    if POSE:
        assert sorted({n.replace("_checkpoint_wrapped_module.", "").split(".")[0] for n in frozen}) == ["pose_head"], ("frozen set", frozen[:4])
    else:
        fpre = {p_ for p_ in FROZEN_PREFIX for n in frozen if n.replace("_checkpoint_wrapped_module.", "").startswith(p_)}
        assert fpre == set(FROZEN_PREFIX), ("no-pose frozen set must cover every pose part", sorted(fpre))
    fz_set = set(frozen)
    fz0 = tensor_digest(model, fz_set)
    log(f"[opt] main AdamW groups {[(pg['name'], len(pg['params']), pg['lr']) for pg in opt.param_groups]}; decoder nanostep AdamW "
        f"dec_s {len(dos['dec_s'].ps)} tensors (peak {CFG['dec_lr']:g}, clip {CFG['dec_clip']:g}), "
        + (f"dec_p {len(dos['dec_p'].ps)} tensors (peak {CFG['dec_p_lr']:g}, clip {CFG['dec_p_clip']:g}) " if POSE else "dec_p NONE (frozen) ")
        + f"(K {CFG['dec_k']}); inp.pose {'in audio' if POSE else 'FROZEN'} ({sum('inp.pose' in n for n, _ in model.named_parameters())} tensor); FROZEN {len(frozen)} tensors ({sorted({n.split('.')[0] for n in frozen})}) digest {fz0[:16]}; "
        f"schedule {SCH.describe(dict(CFG['sched'], warmup_groups=list(TRAIN_GROUPS)))}; decoder schedule = the same shape over updates "
        f"(warmup {CFG['dec_sched']['warmup']}, decay to 0 at {CFG['dec_sched']['wsd_decay_end']})")
    # ---- loaders + eval set
    from torch.utils.data import DataLoader
    W_ = CFG["dl_workers"]

    def _wi(_):
        torch.set_num_threads(1); os.environ["OMP_NUM_THREADS"] = "1"
    ld = iter(DataLoader(A.AloeDS(rows, CFG["data"], RANK, fallback, singles_at=0 if CFG["losscheck"] else -1), batch_size=None,
                         shuffle=False, num_workers=W_, prefetch_factor=2 if W_ else None, persistent_workers=bool(W_), pin_memory=False,
                         worker_init_fn=_wi if W_ else None, collate_fn=lambda x: x))
    ev = A.eval_micros(ev_items, CFG["data"], RANK)
    t_ = torch.tensor([float(len(ev))], device=dev); dist.all_reduce(t_, op=dist.ReduceOp.MAX); n_ev = int(t_)
    ev += [dict(ev[0], dummy=1, own=np.zeros_like(ev[0]["own"])) for _ in range(n_ev - len(ev))]   # same forwards on every rank (FSDP)
    log(f"[eval] {n_ev} eval micros per rank (dummies pad the short ranks); P mean {np.mean([int(m['P'][0]) for m in ev]):.0f}; "
        + (f"pose frames / micro {np.mean([len(m['plight']) for m in ev]):.0f}" if POSE else "no pose (POSE_B 0)"))
    jl = open(f"{LOGDIR}/metrics.jsonl", "a") if RANK == 0 else None

    def emit(d, step):
        if RANK == 0:
            jl.write(json.dumps({"step": step, "time_pt": now_pt(), **d}) + "\n"); jl.flush()

    def evaluate(step):
        model.eval()
        t0_ = time.time()
        acc = {0: None, 1: None}
        for m in ev:
            oc = np.asarray(m["own"]).sum(0)
            v, _, _ = run_micro(model, prep(m, dev, max(oc[0], 1.0), max(oc[1], 1.0), max(oc[3], 1.0) if POSE else None), rot, dos, grad=False)
            if m.get("dummy"):
                v = [0.0] * len(v)
            sw = int(m["swap"])
            acc[sw] = v if acc[sw] is None else [x + y for x, y in zip(acc[sw], v)]
        nv = len(next(a for a in acc.values() if a is not None))
        acc = {k: (a if a is not None else [0.0] * nv) for k, a in acc.items()}
        g0, g1 = allsum(acc[0], dev), allsum(acc[1], dev)
        d = summarize([x + y for x, y in zip(g0, g1)], "eval")
        d.update(summarize(g0, "eval_asstB")); d.update(summarize(g1, "eval_userB"))
        model.train()
        d["eval/time_s"] = time.time() - t0_
        emit(d, step)
        log(f"EVAL step {step}: " + " | ".join(f"{k} {v:.4f}" for k, v in d.items()))
        for k_, v_ in d.items():
            assert math.isfinite(v_), (k_, v_)
        return d

    evaluate(0)
    import gc
    gc.collect(); gc.freeze(); gc.disable()
    times = []
    stopped = None
    for step in range(CFG["steps"]):
        torch.cuda.synchronize(); ts = time.time()
        torch.cuda.reset_peak_memory_stats()
        tw = time.time()
        b = next(ld)
        wait = time.time() - tw
        assert int(b["step"]) == step, ("packed micro order", b["step"], step)
        if step == 0 and CFG["losscheck"]:
            Pv, ex, er, K = pack_loss_check(model, b, rot, dev)
            ea = [None] * WORLD; dist.all_gather_object(ea, er)
            worst = [max(e[i] for e in ea) for i in range(len(er))]
            log(f"[pack] LOSS CHECK step 0: packed micro loss (micro token-normalised) vs token-weighted standalone conversations, max rel err over "
                f"{WORLD} ranks: text {worst[0]:.2e} cb0 {worst[1]:.2e} light {worst[2]:.2e} "
                + (f"pose_cb0 {worst[3]:.2e} pose_light {worst[4]:.2e} " if POSE else "")
                + f"(rank 0: {K} conversations, packed {[round(x, 5) for x in Pv]}, "
                f"weighted standalone {[round(x, 5) for x in ex]})")
            assert max(worst) < CFG["losscheck_tol"], ("packed loss != token-weighted standalone losses", worst)
        oc = np.asarray(b["own"]).sum(0)
        tot = allsum([float(x) for x in oc[:5]], dev)
        bt = prep(b, dev, tot[0] / WORLD, tot[1] / WORLD, tot[3] / WORLD if POSE else None)  # tokens_step normalisers
        if step == 0:
            log(f"[pack] tokens_step: step 0 totals over {WORLD} micros: text targets {int(tot[0])}, cb0 labels {int(tot[1])}, light frames "
                + (f"{int(tot[2])}, pose cb0 labels {int(tot[3])}, pose frames {int(tot[4])}" if POSE else f"{int(tot[2])}") + f"; per-micro normaliser = total / {WORLD}")
        t_m = time.time()
        v, loss_v, ex_ = run_micro(model, bt, rot, dos, grad=True)
        torch.cuda.synchronize(); t_micro = time.time() - t_m
        assert abs(ex_["dec_s_N"] - tot[2]) < 0.5, ("speech decoder frames != the step's light frames", ex_["dec_s_N"], tot[2])
        if POSE:
            assert abs(ex_["dec_p_N"] - tot[4]) < 0.5, ("pose decoder frames != the step's pose frames", ex_["dec_p_N"], tot[4])
        for n_, p_ in model.named_parameters():
            if n_ in fz_set:
                assert p_.grad is None, ("a frozen parameter got a gradient", n_)
        gn = {}
        for pg in opt.param_groups:
            g_ = torch.nn.utils.clip_grad_norm_(pg["params"], CFG["clip"][pg["name"]])
            gn[pg["name"]] = float(g_.full_tensor() if hasattr(g_, "full_tensor") else g_)
        lr_used = {pg["name"]: pg["lr"] for pg in opt.param_groups}
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize(); dt = time.time() - ts; times.append(dt)
        sv = allsum(list(v) + [loss_v], dev)
        d = summarize(sv[:-1], "train")
        d["train/loss"] = sv[-1] / WORLD
        d["train/dummies"] = float(nconv[2]) if step == CFG["steps"] - 1 else 0.0
        pk = torch.tensor([torch.cuda.max_memory_allocated() / GiB, dt, wait, t_micro], device=dev)
        dist.all_reduce(pk, op=dist.ReduceOp.MAX)
        d.update({f"train/gn_{k}": x for k, x in gn.items()})
        d.update({f"train/lr_{k}": x for k, x in lr_used.items()})
        d.update({"train/dec_lr_first": ex_["dec_s_lrs"][0], "train/dec_lr_last": ex_["dec_s_lrs"][-1], "train/step_time": dt, "train/peak_gib": float(pk[0]),
                  "train/data_wait": float(pk[2]), "train/micro_s": float(pk[3]), "train/text_targets": tot[0], "train/cb0_labels": tot[1],
                  "train/light_frames": tot[2]})
        if POSE:
            d.update({"train/dec_p_lr_first": ex_["dec_p_lrs"][0], "train/dec_p_lr_last": ex_["dec_p_lrs"][-1],
                      "train/pose_cb0_labels": tot[3], "train/pose_frames": tot[4]})
        for k_, v_ in d.items():
            assert math.isfinite(v_), (k_, v_, "non-finite: fail loudly")
        done = step + 1
        emit(d, done)
        eta = (CFG["steps"] - done) * float(np.mean(times[-20:]))
        log(f"step {done}/{CFG['steps']} {dt:.1f}s (micro {float(pk[3]):.1f} wait {float(pk[2]):.1f}) peak {float(pk[0]):.1f} GiB | "
            + " ".join(f"{k.split('/', 1)[1]}={v:.4f}" for k, v in d.items() if k.startswith("train_speech/") and k.endswith(("word_ce_norm", "cb0_ce", "light_ce", "pad_ce")) and "light_ce_cb" not in k)
            + " " + " ".join(f"{k.split('/')[1]}={v:.4g}" for k, v in d.items() if "/gn_" in k or "/lr_trunk" in k or "/lr_tower_p" in k or "dec_lr_last" in k or "dec_p_lr_last" in k)
            + f" | ETA {eta / 60:.1f} min")
        gc.collect(1)
        if (CFG["eval_every"] and done % CFG["eval_every"] == 0) or done == CFG["steps"] or done in CFG["export_steps"]:
            evaluate(done)
        if done in CFG["export_steps"] and done < CFG["steps"]:
            export_local(model, done, f"{CFG['local_export_root']}/export-{done}")
        st_ = torch.tensor([float(STOP["flag"] or os.path.exists(f"{CFG['control_dir']}/STOP_AFTER_SAVE"))], device=dev)
        dist.all_reduce(st_, op=dist.ReduceOp.MAX)
        if float(st_) > 0 and done < CFG["steps"]:
            log(f"[control] STOP_AFTER_SAVE at step {done}: weights-only export, then exit")
            export_weights(model, done, f"{CFG['export']}-stop{done}", CFG["export_min_free_gb"])
            stopped = done
            break
    fz1 = tensor_digest(model, fz_set)
    same = torch.tensor([float(fz1 == fz0)], device=dev); dist.all_reduce(same, op=dist.ReduceOp.MIN)
    log(f"[frozen] {'pose_head (PPM D36 off)' if POSE else 'pose parts (tower_p, dec_p, pose_head, decp_bb, inp.pose)'} bit-identical to the init on every rank: {bool(float(same))} (rank 0 {fz1[:16]})")
    assert float(same) == 1.0, "a frozen parameter changed"
    if stopped is None:
        export_local(model, CFG["steps"], f"{CFG['local_export_root']}/export-{CFG['steps']}")   # aloe: node-local copy first (never refused)
        export_weights(model, CFG["steps"], CFG["export"], CFG["export_min_free_gb"])
    log(f"done {stopped or CFG['steps']} steps; mean step {np.mean(times):.2f}s; step times {[round(x, 1) for x in times]}")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
