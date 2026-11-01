"""gesturemetric (2) CPU-mini contrastive speech <-> gesture judge, trained on REAL Aloe train convs only (judge/feat/train).
Windows = 4 s = 50 token steps (100 pose frames); own-speech windows only (>= MIN_WORDS own word onsets inside).
POSE encoder: per frame 15 points (x, y) - window-mean shoulder midpoint (kills framing / posture identity), 2 hand spreads, 15 conf masks,
  velocities -> pairs of frames -> 50 steps x 154 -> Linear 128 + pos -> 2-layer transformer (d 128, 4 heads, ff 256) -> attention pool.
SPEECH encoder: per step sum of PCA-128 Qwen3-0.6B layer-14 states of own words with onset at that step + onset count, + own Mimi cb0-7
  (8 x 2048 x 16 embeddings, concat 128) -> same transformer + pool. Cosine similarity, learned temperature.
Loss: InfoNCE both ways over G=16 windows >= 4 s apart from the SAME conversation side (speaker identity, camera, room constant ->
  only content / timing can match); half the groups are TIGHT (16 consecutive moments, same conversation phase). Modality dropout (words or audio zeroed, p 0.2 each) so one model also gives words-only / audio-only.
Eval (judge/feat/eval, held-out convs): top-1@16 pose->speech and speech->pose (chance 6.25 %), full / words-only / audio-only,
  shuffled pairing (pose of side X vs speech of another conv's side), pose shifted +2 s, and TIGHT groups (16 moments within ~1-2 min:
  rules out matching by slow posture / topic drift over the conversation).
usage: train2.py [--steps N] [--hard] [--tag T] [--patience N evals] [--feat-dir DIR (default feat)] [--d 128] [--layers 2] [--ff 256]
       [--batch 8 groups] [--eval-every 500] [--max-min M (wall cap, minutes)] [--pose-keep 5-10,13,14 (default all 15 points)]
       -> judge/ckpt_<tag>.pt, judge/eval2_<tag>.json
       (+ judge/pca.npz for the default feat dir, judge/pca_<tag>.npz otherwise). env GM_DEVICE=cuda trains / evaluates on the GPU.
Mimi embeddings are 8 x (D / 8) (= 16 at the default D 128); the ckpt stores the arch so score2.py rebuilds it.
--pose-keep: point indices kept (0-12 = COCO body 0-12, 13 / 14 = left / right hand centroid); every other point's x, y, velocity and
  conf-mask features are zeroed in train AND eval (hand spread i kept iff point 13 + i kept). Stored in the ckpt (score2.py reuses it)."""
import glob
import json
import os
import sys
import time

J = os.path.dirname(os.path.abspath(__file__))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

WIN, MIN_WORDS, G, MIN_GAP = 50, 3, 16, 50
PATIENCE = 2
D, PCA_D = 128, 128
LAYERS, FF, HEADS = 2, 256, 4
FEAT = "feat"
POSE_KEEP = None              # None = all 15 pose points (unchanged default)
DEV = torch.device(os.environ.get("GM_DEVICE", "cpu"))


def set_arch(cfg):
    global D, LAYERS, FF, HEADS
    D, LAYERS, FF, HEADS = int(cfg.get("D", 128)), int(cfg.get("LAYERS", 2)), int(cfg.get("FF", 256)), int(cfg.get("HEADS", 4))


def set_pose_keep(spec):
    """'5-10,13,14' / list of ints / None / '' / 'all' -> POSE_KEEP (None = all points)."""
    global POSE_KEEP
    if spec is None or (isinstance(spec, str) and spec.strip() in ("", "all")):
        POSE_KEEP = None
        return
    if isinstance(spec, str):
        ks = []
        for part in spec.split(","):
            a, _, b = part.strip().partition("-")
            ks += list(range(int(a), int(b or a) + 1))
        spec = ks
    ks = sorted(set(int(k) for k in spec))
    assert ks and all(0 <= k < 15 for k in ks), f"bad --pose-keep {spec}"
    POSE_KEEP = None if ks == list(range(15)) else ks


def arch():
    return {"D": D, "LAYERS": LAYERS, "FF": FF, "HEADS": HEADS}


# ------------------------------------------------------------------------------------------------ data
def load_split(split):
    tr = []
    for f in sorted(glob.glob(os.path.join(J, FEAT, split, "*.npz"))):
        if f.endswith(".tmp.npz"):
            continue
        z = np.load(f, allow_pickle=True)
        T = int(z["T"])
        tr.append({"name": os.path.basename(f)[:-4], "conv": str(z["conv"]), "pose": z["pose"].astype(np.float32)[:2 * T],
                   "w_step": z["w_step"], "w_vec": z["w_vec"].astype(np.float32), "mimi": z["mimi"].astype(np.int64)[:T], "T": T})
    return tr


def fit_pca(tracks, n=30000, seed=0):
    rng = np.random.default_rng(seed)
    X = np.concatenate([t["w_vec"] for t in tracks if len(t["w_vec"])])
    X = X[rng.choice(len(X), min(n, len(X)), replace=False)]
    mu = X.mean(0)
    _, s, vt = np.linalg.svd(X - mu, full_matrices=False)
    P = vt[:PCA_D].T / (s[:PCA_D] / np.sqrt(len(X)))[None]          # whitened
    return mu.astype(np.float32), P.astype(np.float32)


def prepare(tracks, mu, P):
    for t in tracks:
        T = t["T"]
        wf = np.zeros((T, PCA_D + 1), np.float32)
        if len(t["w_step"]):
            z = (t["w_vec"] - mu) @ P
            ok = (t["w_step"] >= 0) & (t["w_step"] < T)
            np.add.at(wf, t["w_step"][ok], np.concatenate([z[ok], np.ones((ok.sum(), 1), np.float32)], 1))
        t["wf"] = wf
        cnt = np.concatenate([[0], np.cumsum(wf[:, -1])])
        t["valid"] = np.flatnonzero(cnt[WIN:T + 1] - cnt[:T + 1 - WIN] >= MIN_WORDS)     # window starts with >= MIN_WORDS onsets


def pose_window(p):
    """p [B, 100, 47] raw -> [B, 50, 154]."""
    pts, spr, m = p[..., :30].reshape(*p.shape[:2], 15, 2), p[..., 30:32], p[..., 32:47]
    sh = (m[..., 5] * m[..., 6])[..., None]                               # both shoulders confident
    mid = (pts[..., 5, :] + pts[..., 6, :]) / 2
    c = (mid * sh).sum(1, keepdims=True) / np.maximum(sh.sum(1, keepdims=True), 1)   # window-mean shoulder midpoint [B, 1, 2]
    rel = (pts - c[:, :, None]) * m[..., None]
    vel = np.zeros_like(rel)
    vel[:, 1:] = (rel[:, 1:] - rel[:, :-1]) * (m[:, 1:] * m[:, :-1])[..., None]
    f = np.concatenate([rel.reshape(*p.shape[:2], 30), spr, m, vel.reshape(*p.shape[:2], 30)], -1)   # [B, 100, 77]
    if POSE_KEEP is not None:      # zero dropped points: x, y [2k, 2k+1], conf [32+k], vel [47+2k, 48+2k]; spread i iff hand 13+i dropped
        drop = [k for k in range(15) if k not in POSE_KEEP]
        f[..., [c for k in drop for c in (2 * k, 2 * k + 1, 32 + k, 47 + 2 * k, 48 + 2 * k)] + [30 + i for i in (0, 1) if 13 + i in drop]] = 0
    return f.reshape(p.shape[0], WIN, 154)


def pick_group(valid, rng, g=G, gap=MIN_GAP):
    """g window starts from valid, pairwise >= gap apart (None if impossible)."""
    if len(valid) < g:
        return None
    cand = rng.permutation(valid)
    out = []
    for s in cand:
        if all(abs(s - o) >= gap for o in out):
            out.append(s)
            if len(out) == g:
                return np.sort(np.array(out))
    return None


def pick_tight(valid, rng, g=G, gap=MIN_GAP):
    """g consecutive valid window starts >= gap apart from a random point (span ~1-2 min): same conversation phase, so slow drift of
    posture / topic cannot separate them - only moment-level content / timing can."""
    if len(valid) < g:
        return None
    v = np.sort(valid)
    i = rng.integers(0, len(v))
    out = [v[i]]
    for s in v[i + 1:]:
        if s - out[-1] >= gap:
            out.append(s)
            if len(out) == g:
                return np.array(out)
    return None


def batch(tracks, starts_list, pose_off=0):
    """-> pose [N, 50, 154], words [N, 50, 129], mimi [N, 50, 8] for (track, start) pairs; pose_off shifts pose windows (steps)."""
    P, Wd, M = [], [], []
    for t, starts in starts_list:
        for s in starts:
            ps = int(np.clip(s + pose_off, 0, t["T"] - WIN))
            P.append(t["pose"][2 * ps:2 * ps + 2 * WIN])
            Wd.append(t["wf"][s:s + WIN]); M.append(t["mimi"][s:s + WIN])
    return (torch.from_numpy(pose_window(np.stack(P))).to(DEV), torch.from_numpy(np.stack(Wd)).to(DEV), torch.from_numpy(np.stack(M)).to(DEV))


# ------------------------------------------------------------------------------------------------ model
class Enc(nn.Module):
    def __init__(self, din):
        super().__init__()
        self.inp = nn.Linear(din, D)
        self.pos = nn.Parameter(torch.zeros(1, WIN, D))
        layer = nn.TransformerEncoderLayer(D, HEADS, FF, dropout=0.1, batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, LAYERS)
        self.att = nn.Linear(D, 1)
        self.out = nn.Linear(D, D)

    def forward(self, x, extra=None):
        h = self.inp(x) + self.pos
        if extra is not None:
            h = h + extra
        h = self.tf(h)
        a = torch.softmax(self.att(h), 1)
        return F.normalize(self.out((a * h).sum(1)), dim=-1)


class Judge(nn.Module):
    def __init__(self):
        super().__init__()
        self.pose = Enc(154)
        self.speech = Enc(PCA_D + 1)
        self.mimi = nn.ModuleList([nn.Embedding(2048, D // 8) for _ in range(8)])
        self.logit_scale = nn.Parameter(torch.tensor(np.log(1 / 0.07), dtype=torch.float32))

    def enc_speech(self, w, m, use_w=True, use_a=True):
        if not use_w:
            w = torch.zeros_like(w)
        ae = torch.cat([e(m[..., i]) for i, e in enumerate(self.mimi)], -1)
        if not use_a:
            ae = torch.zeros_like(ae)
        return self.speech(w, ae)

    def enc_pose(self, p):
        return self.pose(p)


def nparams(m):
    return sum(p.numel() for p in m.parameters())


# ------------------------------------------------------------------------------------------------ eval
@torch.no_grad()
def evaluate(model, tracks, n_groups=8, seed=123, modes=("full", "words", "audio"), extra=True):
    model.eval()
    rng = np.random.default_rng(seed)
    res = {}
    groups = []
    for t in tracks:
        for _ in range(n_groups):
            g = pick_group(t["valid"], rng)
            if g is not None:
                groups.append((t, g))
    def acc(pe, se):
        s = pe @ se.T
        r = torch.arange(len(s), device=s.device)
        return float((s.argmax(1) == r).float().mean()), float((s.argmax(0) == r).float().mean())
    for mode in modes:
        a = []
        for t, g in groups:
            p, w, m = batch(tracks, [(t, g)])
            a.append(acc(model.enc_pose(p), model.enc_speech(w, m, mode != "audio", mode != "words")))
        res[mode] = {"p2s": round(100 * np.mean([x[0] for x in a]), 2), "s2p": round(100 * np.mean([x[1] for x in a]), 2),
                     "groups": len(a)}
    if extra:
        # shuffled pairing: pose group of side i vs speech group of a side from another conversation
        a = []
        for i, (t, g) in enumerate(groups):
            t2, g2 = groups[(i + 7 * n_groups) % len(groups)]
            if t2["conv"] == t["conv"]:
                continue
            p, _, _ = batch(tracks, [(t, g)])
            _, w, m = batch(tracks, [(t2, g2)])
            a.append(acc(model.enc_pose(p), model.enc_speech(w, m)))
        res["shuffled"] = {"p2s": round(100 * np.mean([x[0] for x in a]), 2), "s2p": round(100 * np.mean([x[1] for x in a]), 2),
                           "groups": len(a)}
        a = []
        for t, g in groups:
            p, _, _ = batch(tracks, [(t, g)], pose_off=25)
            _, w, m = batch(tracks, [(t, g)])
            a.append(acc(model.enc_pose(p), model.enc_speech(w, m)))
        res["pose_shift_2s"] = {"p2s": round(100 * np.mean([x[0] for x in a]), 2), "s2p": round(100 * np.mean([x[1] for x in a]), 2)}
        # tight groups (16 moments within ~1-2 min): full / words-only / audio-only
        tg = []
        for t in tracks:
            for _ in range(n_groups):
                g = pick_tight(t["valid"], rng)
                if g is not None:
                    tg.append((t, g))
        for mode in ("full", "words", "audio"):
            a = []
            for t, g in tg:
                p, w, m = batch(tracks, [(t, g)])
                a.append(acc(model.enc_pose(p), model.enc_speech(w, m, mode != "audio", mode != "words")))
            res["tight_" + mode] = {"p2s": round(100 * np.mean([x[0] for x in a]), 2), "s2p": round(100 * np.mean([x[1] for x in a]), 2),
                                    "groups": len(a)}
    model.train()
    return res


# ------------------------------------------------------------------------------------------------ train
def main():
    steps = int(sys.argv[sys.argv.index("--steps") + 1]) if "--steps" in sys.argv else 4000
    hard = "--hard" in sys.argv
    tag = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else ("hard" if hard else "base")
    global PATIENCE, FEAT
    PATIENCE = int(sys.argv[sys.argv.index("--patience") + 1]) if "--patience" in sys.argv else 2
    arg = lambda k, d: type(d)(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d  # noqa: E731
    FEAT = arg("--feat-dir", "feat")
    set_arch({"D": arg("--d", 128), "LAYERS": arg("--layers", 2), "FF": arg("--ff", 256), "HEADS": arg("--heads", 4)})
    set_pose_keep(arg("--pose-keep", ""))
    B, every, max_s = arg("--batch", 8), arg("--eval-every", 500), 60 * arg("--max-min", 1e9)
    torch.manual_seed(0)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "3")))
    t0 = time.time()
    TR, EV = load_split("train"), load_split("eval")
    mu, P = fit_pca(TR)
    prepare(TR, mu, P); prepare(EV, mu, P)
    np.savez(os.path.join(J, "pca.npz" if FEAT == "feat" else f"pca_{tag}.npz"), mu=mu, P=P)
    for t in TR + EV:
        t.pop("w_vec", None)             # only needed for the PCA fit / prepare (saves RAM on the full set)
    TR = [t for t in TR if len(t["valid"]) >= 3 * G]
    print(f"train sides {len(TR)} eval sides {len(EV)} load {time.time() - t0:.0f}s", flush=True)
    model = Judge().to(DEV)
    print(f"params {nparams(model) / 1e6:.2f}M arch {arch()} batch {B}x{G} device {DEV} pose_keep {POSE_KEEP or 'all'}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 200) * max(0.0, 1 - s / steps))
    rng = np.random.default_rng(0)
    best, best_state, log = -1, None, []
    t_tr = time.time()
    for step in range(steps):
        sl = []
        while len(sl) < B:
            t = TR[rng.integers(len(TR))]
            g = pick_tight(t["valid"], rng) if rng.random() < 0.5 else pick_group(t["valid"], rng)   # half the groups: same phase
            if g is not None:
                sl.append((t, g))
        p, w, m = batch(TR, sl)
        use_w, use_a = True, True
        r = rng.random()
        if r < 0.2:
            use_a = False
        elif r < 0.4:
            use_w = False
        pe = model.enc_pose(p).view(B, G, D)
        se = model.enc_speech(w, m, use_w, use_a).view(B, G, D)
        scale = model.logit_scale.exp().clamp(max=100)
        lg = scale * torch.einsum("bid,bjd->bij", pe, se)
        lab = torch.arange(G, device=DEV).expand(B, G)
        loss = (F.cross_entropy(lg.reshape(-1, G), lab.reshape(-1)) + F.cross_entropy(lg.transpose(1, 2).reshape(-1, G), lab.reshape(-1))) / 2
        if hard:   # extra negatives: the same window's speech shifted by 1-3 s (12-38 steps)
            sh = []
            for t, g in sl:
                off = rng.integers(12, 38, len(g)) * rng.choice([-1, 1], len(g))
                sh.append((t, np.clip(g + off, 0, t["T"] - WIN)))
            _, w2, m2 = batch(TR, sh)
            se2 = model.enc_speech(w2, m2, use_w, use_a).view(B, G, D)
            lg2 = scale * (pe * se2).sum(-1, keepdim=True)
            loss = loss + F.cross_entropy(torch.cat([lg, lg2], -1).reshape(-1, G + 1), lab.reshape(-1)) / 2
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if step % 50 == 0:
            print(f"step {step} loss {loss.item():.3f} scale {scale.item():.1f} {time.time() - t0:.0f}s", flush=True)
        over = time.time() - t_tr > max_s
        if (step + 1) % every == 0 or step + 1 == steps or over:
            ev = evaluate(model, EV, n_groups=4, modes=("full",), extra=False)
            log.append({"step": step + 1, **ev})
            print(f"EVAL step {step + 1}: {json.dumps(ev)} {time.time() - t0:.0f}s", flush=True)
            if ev["full"]["p2s"] > best:
                best = ev["full"]["p2s"]
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            elif len(log) > PATIENCE and all(x["full"]["p2s"] < best for x in log[-PATIENCE:]):
                print("early stop (eval stopped improving)", flush=True)
                break
            if over:
                print(f"wall cap {max_s / 60:.0f} min reached: stop", flush=True)
                break
    model.load_state_dict(best_state)
    ck = os.path.join(J, f"ckpt_{tag}.pt")
    torch.save({"state": best_state, "pca_mu": mu, "pca_P": P, "tag": tag, "arch": arch(), "feat": FEAT, "pose_keep": POSE_KEEP}, ck + ".tmp")
    os.replace(ck + ".tmp", ck)
    final = evaluate(model, EV, n_groups=8)
    out = {"tag": tag, "params_M": round(nparams(model) / 1e6, 3), "arch": arch(), "batch": B, "feat": FEAT, "pose_keep": POSE_KEEP,
           "train_sides": len(TR), "eval_sides": len(EV), "log": log,
           "final": final, "wall_s": round(time.time() - t0)}
    eo = os.path.join(J, f"eval2_{tag}.json")
    json.dump(out, open(eo + ".tmp", "w"), indent=1)
    os.replace(eo + ".tmp", eo)
    print("FINAL", json.dumps(final), flush=True)


if __name__ == "__main__":
    main()
