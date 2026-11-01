"""ROSIE-FT FULL COTRAIN TRAINER (10-30, recipe v2: replaces the LoRA recipe of train_rosie_cotrain.py): FULL fine-tune of export-253's
backbone on Rosie Stamp speech (data_rosie) co-trained 1:1 per backbone step with zoom-strangers-5k text (data_zoom), one host, torchrun.
Same data / plan / zoom path / init (resharded ppm-local shards, the init's own vadapter, vgain theta sidecar) / eval / merged export as
train_rosie_cotrain.py. What differs:

  GROUPS    main (one update per backbone step, WSD warmup QQA_WARMUP then linear decay to 0):
              trunk  (inp.emb, layers, head.norm, head.lm_head)  LR 2e-6, clip 0.5   <- stored bf16 on the GPU, fp32 MASTER + AdamW m / v in
                     pinned HOST memory (HostMasterAdamW: chunked H2D -> fused-equivalent AdamW maths on the GPU -> D2H, two-stream
                     pipelined); its gradients are bf16 (FSDP2 reduce-scatters in fp32, then casts to the bf16 param dtype).
              spec 6e-5, proj 3e-5, audio 6e-5 (fp32 masters + AdamW on the GPU), clip 1.0 each.
            tower_s: TOWER NANOSTEPS, K_T (default 4) updates per backbone step, own AdamW (LR 1.5e-5 per nanostep, clip 1.0), own WSD
              (warmup QQA_WARMUP x K_T, horizon steps x K_T). tower_s is its OWN FSDP2 root (not a child of the backbone's root), run on the
              DETACHED final-normed backbone states hn of the rosie micro.
            dec_s (0.6B light decoder, replicated): K_D (16) nanostep updates per backbone step, LR 6e-5, own WSD (warmup x K_D, steps x K_D).
            FROZEN: vadapter, vgain (= 29577).
  ROSIE MICRO (one per rank per step, COTRAIN_GROUP_MODE=mixed only):
            1. backbone forward (grad) -> text CE (B, factorised weights / the step's rosie text total) and hn (captured from the head unit);
            2. the micro's cb0-labelled positions are split by a seeded permutation (step, logical rank) into K_T chunks; light frames
               follow their position's chunk and are split again into K_D / K_T decoder sub-chunks. For tower nanostep j: tower forward on
               hn.detach() over the whole micro (causal tower, every position), cb0 CE on chunk j only (weight W_CB0 / (C_j / W), C_j =
               chunk j's labels over all ranks), hv -> K_D / K_T decoder nanosteps on hv.detach() over chunk j's light frames (each its own
               all-reduce + AdamW update; their input gradient = the chunk-j mean light CE gradient), ONE tower backward (roots cb0 loss_j,
               hv with W_CB x the decoder's gradient) -> tower grads (clip, AdamW nanostep) and dL_j/dhn; dhn += (C_j / C) dL_j/dhn
               (C_j / C = the chunk's share of the step's labels; light frames follow positions, light_frac 1: the light shares equal it,
               logged). The tower / decoder weights each nanostep sees are the current ones (= the decoder's 'towers' input gradient'
               pattern of the template, one level up).
            3. ONE backbone backward: roots (text loss, hn) with grads (None, dhn).
  ZOOM MICRO: backbone forward + text CE (the step's zoom text total) + backward; never touches tower_s / dec_s. 1:1 per backbone step
            (ROSIE_LOGICAL_WORLD = ZOOM_LOGICAL_WORLD = 6 on 6 GPUs: 6 + 6 micros).
  TESTS     COTRAIN_TEST_DEPTH, QQA_TEST_STEPS, ROSIE_TEST_REPEAT, ROSIE_TEST_GRADSPLIT (zoom-only pass: exact-zero audio grads, no tower /
            decoder grads, non-zero trunk / proj), ROSIE_TEST_EXPORT0 (merged export at step 0, before any update: compare with the init via
            tools/compare_exports.py), ROSIE_TEST_MERGECHECK (reload the final export, eval micros equal up to bf16), ROSIE_MEM_CAP_GB."""
import datetime
import importlib
import json
import math
import os
import socket
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rosie_paths  # noqa: E402
RP = rosie_paths.patch()

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import torch.nn as nn  # noqa: E402

import model_ppm as M  # noqa: E402
import data_zoom as Z  # noqa: E402
import audio_s4q as AU  # noqa: E402
import sched_s3q as SCH  # noqa: E402

DATA = importlib.import_module(os.environ.get("ROSIE_DATA_MODULE", "data_rosie"))
BF = torch.bfloat16


def E(k, d=None):
    v = os.environ.get(k, d)
    assert v is not None, ("missing env", k)
    return v


RANK, WORLD, LOCAL = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), int(os.environ["LOCAL_RANK"])
HOST = socket.gethostname().split(".")[0]
MAIN_GROUPS = ("trunk", "spec", "proj", "audio")
HOST_GROUPS = ("trunk", "spec")                                   # bf16 GPU shards + host fp32 master / m / v (ROSIE_TRUNK_OFFLOAD=1)
FROZEN_GROUPS = ("vadapter", "vgain")
CFG = {
    "trainer": "train_rosie_full.py", "run_tag": E("ROSIE_RUN"), "run_root": E("ROSIE_RUN_ROOT", "/workspace/rosie-ft/runs"),
    "init": E("ROSIE_INIT", "/workspace/rosie-ft/init/export-253"), "data_root": E("ROSIE_DATA_ROOT", "/workspace/data/rosie"),
    "data_module": DATA.__name__, "epochs": int(E("ROSIE_EPOCHS", "2")), "light_frac": float(E("ROSIE_LIGHT_FRAC", "1.0")),
    "steps_env": E("QQA_STEPS", "auto"), "warmup": int(E("QQA_WARMUP", "2")), "eval_every": int(E("QQA_EVAL_EVERY", "0")),
    "export_every": int(E("ROSIE_EXPORT_EVERY", "0")), "export_steps": [int(x) for x in E("QQA_EXPORT_STEPS", "").split(",") if x],
    "w_text": float(E("QQA_W_TEXT_CE", "5")), "w_cat_word": float(E("QQA_W_CAT_WORD", "1")), "w_tim": float(E("QQA_W_TIM_CE", "1")),
    "w_nw": float(E("QQA_W_NW_CE", "1")), "w_eob": float(E("QQA_W_EOB_CE", "1")), "w_cb0": float(E("QQA_W_CB0", "1")), "w_cb": float(E("QQA_W_CB", "1")),
    "lr": {"trunk": float(E("QQA_LR_TRUNK", "2e-6")), "spec": float(E("QQA_LR_SPEC", "6e-5")), "proj": float(E("QQA_LR_PROJ", "3e-5")),
           "audio": float(E("QQA_LR_AUDIO", "6e-5")), "tower_s": float(E("QQA_LR_TOWER", "1.5e-5")), "dec_s": float(E("QQA_DEC_LR", "6e-5"))},
    "clip": {"trunk": float(E("QQA_CLIP_TRUNK", "0.5")), "spec": float(E("QQA_CLIP_SPEC", "1.0")), "proj": float(E("QQA_CLIP_PROJ", "1.0")),
             "audio": float(E("QQA_CLIP_AUDIO", "1.0")), "tower_s": float(E("QQA_CLIP_TOWER", "1.0")), "dec_s": float(E("QQA_DEC_CLIP", "1.0"))},
    "adam_beta2": float(E("QQA_ADAM_BETA2", "0.95")), "k_tower": int(E("TOWER_K", "4")), "dec_k": int(E("QQA_DEC_K", "16")),
    "dec_seed": int(E("QQA_DEC_SEED", "20261011")), "dec_mb_max": int(E("QQA_DEC_MB_MAX", "470")), "dec_dyn": E("QQA_DEC_DYN", "1") == "1",
    "dec_fast": E("QQA_DEC_FAST", "1") == "1", "dec_compile_mode": E("QQA_DEC_COMPILE_MODE", "default"),
    "eval_light_max": int(E("QQA_EVAL_LIGHT_MAX", "2048")), "dl_workers": int(E("QQA_DL_WORKERS", "3")), "cd": int(os.environ["PPM_CD"]),
    "trunk_offload": E("ROSIE_TRUNK_OFFLOAD", "1") == "1", "offload_chunk_m": int(E("ROSIE_OFFLOAD_CHUNK_M", "64")),
    "vadapter": {"enc_hidden": 1280, "hidden": 4096, "stack": 4, "norm": "none"}, "vox_model": RP["voxtral"],
    "zoom_src": Z.SRC, "zoom_table": Z.TABLE, "zoom_eval_max": int(E("ZOOM_EVAL_MAX", "100")), "rosie_eval_max": int(E("ROSIE_EVAL_MAX", "0")),
    "test_steps": int(E("QQA_TEST_STEPS", "0")), "test_depth": E("COTRAIN_TEST_DEPTH", ""), "test_repeat": E("ROSIE_TEST_REPEAT", "0") == "1",
    "test_gradsplit": E("ROSIE_TEST_GRADSPLIT", "0") == "1", "test_mergecheck": E("ROSIE_TEST_MERGECHECK", "0") == "1",
    "test_export0": E("ROSIE_TEST_EXPORT0", "0") == "1", "wandb": E("ROSIE_WANDB", "0") == "1", "paths": RP,
}
KT, KD = CFG["k_tower"], CFG["dec_k"]
assert KD % KT == 0, ("QQA_DEC_K must be a multiple of TOWER_K (decoder sub-chunks per tower chunk)", KD, KT)
KS = KD // KT
assert CFG["cd"] == 2
MODE = E("COTRAIN_GROUP_MODE", "mixed")
assert MODE == "mixed", "tower / decoder nanosteps: every rank runs exactly one rosie micro per step (mixed mode)"
LWR, ZL = int(E("ROSIE_LOGICAL_WORLD", "6")), int(E("ZOOM_LOGICAL_WORLD", E("ROSIE_LOGICAL_WORLD", "6")))
assert LWR == WORLD, ("one rosie micro per rank per step: ROSIE_LOGICAL_WORLD must equal the world size", LWR, WORLD)
W_ROSIE, W_ZOOM = float(E("COTRAIN_W_ROSIE", "1")), float(E("COTRAIN_W_ZOOM", "1"))
GZ = -(-ZL // WORLD)
SLOTS = [("rosie", RANK)] + [("zoom", RANK + WORLD * s if RANK + WORLD * s < ZL else None) for s in range(GZ)]
CFG["groups"] = {"mode": MODE, "rosie_logical_world": LWR, "zoom_logical_world": ZL, "w_rosie": W_ROSIE, "w_zoom": W_ZOOM, "slots_rank0": [("rosie", 0)] +
                 [("zoom", WORLD * s if WORLD * s < ZL else None) for s in range(GZ)]}
LOGDIR = f"{CFG['run_root']}/{CFG['run_tag']}"
GiB = 2 ** 30


def now():
    return datetime.datetime.now().strftime("%m-%d %H:%M:%S")


def log(*a):
    if RANK == 0:
        print(f"[{now()}]", *a, flush=True)


def allsum(v, dev):
    t = torch.tensor(v, dtype=torch.float64, device=dev)
    dist.all_reduce(t)
    return t.tolist()


def clean(n):
    return n.replace("_checkpoint_wrapped_module.", "")


def wsd(warm, steps):
    return SCH.make_lambda(True, {"warmup": warm, "horizon": steps, "min_lr_rate": 0.0, "s3_sched": "wsd", "wsd_decay_start": warm,
                                  "wsd_decay_end": steps, "wsd_floor": 0.0, "wsd_shape": "linear"})


# ------------------------------------------------------------------------------------------------ model (backbone root + tower root)
class Net:
    """model = PPM without tower_s (FSDP2 root: inp / layers / head / vadapter / vgain units, dec_s replicated & ignored); tower = tower_s as
    its OWN FSDP2 root (layers + tower unit). named(): every parameter under its export FQN."""

    def __init__(self, model, tower):
        self.model, self.tower = model, tower

    def named(self):
        for n, p in self.model.named_parameters():
            yield clean(n), p
        for n, p in self.tower.named_parameters(prefix="tower_s"):
            yield clean(n), p


def build(c32, c4, c06, dev):
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
    mp = MixedPrecisionPolicy(param_dtype=BF, reduce_dtype=torch.float32)
    mpn = MixedPrecisionPolicy(param_dtype=BF, reduce_dtype=torch.float32, cast_forward_inputs=False)
    with torch.device("meta"):
        model = M.PPM(c32, c4, c06, 2048, M.TOWER_VOCAB, vadapter=CFG["vadapter"], nopose=True)
    tower = model.tower_s
    del model.tower_s
    nb = 0
    for mn, mod in list(model.named_modules()):
        for pn, p in list(mod.named_parameters(recurse=False)):
            full = f"{mn}.{pn}" if mn else pn
            g = M.group_of(full)
            if g in FROZEN_GROUPS:
                setattr(mod, pn, nn.Parameter(torch.empty_like(p, dtype=BF if g == "vadapter" else p.dtype), requires_grad=False))
            elif g in HOST_GROUPS and CFG["trunk_offload"]:
                setattr(mod, pn, nn.Parameter(torch.empty_like(p, dtype=BF), requires_grad=True)); nb += 1
    for t_ in model.inp.aud:                                       # FSDP2: one original dtype per unit -> the fp32 audio tables and the
        fully_shard(t_, mp_policy=mp, reshard_after_forward=True)  #   fp32 text-head projector are their own units (bf16 trunk around)
    fully_shard(model.inp, mp_policy=mp, reshard_after_forward=True)
    for L in model.layers:
        fully_shard(L, mp_policy=mp, reshard_after_forward=True)
    fully_shard(model.head.proj, mp_policy=mp, reshard_after_forward=True)
    fully_shard(model.head, mp_policy=mpn, reshard_after_forward=True)
    fully_shard(model.vadapter, mp_policy=mp, reshard_after_forward=True)
    fully_shard(model.vgain, mp_policy=MixedPrecisionPolicy(param_dtype=torch.float32, reduce_dtype=torch.float32, cast_forward_inputs=False),
                reshard_after_forward=True)
    fully_shard(model, mp_policy=mpn, reshard_after_forward=True, ignored_params=set(model.dec_s.parameters()))
    for L in tower.layers:
        fully_shard(L, mp_policy=mp, reshard_after_forward=True)
    fully_shard(tower, mp_policy=mpn, reshard_after_forward=True)
    model.to_empty(device=dev)
    tower.to_empty(device=dev)
    model.dec_s.model.rotary_emb = type(model.dec_s.model.rotary_emb)(config=c06, device=dev)
    net = Net(model, tower)
    for n, p in net.named():
        g = M.group_of(n)
        assert p.requires_grad == (g not in FROZEN_GROUPS), n
    log(f"[build] backbone {c32.num_hidden_layers} layers (root A) | tower_s {c4.num_hidden_layers} layers (own FSDP2 root B) | dec_s "
        f"{c06.num_hidden_layers} layers replicated; trunk tensors stored bf16 with host fp32 masters: {nb}; frozen: vadapter (bf16), vgain")
    return net


# ------------------------------------------------------------------------------------------------ init (= train_rosie_cotrain.load_init over Net)
SKIPPED_INIT = {}


def read_theta(path, meta):
    if "vgain_theta_fp32" in meta:
        return float(meta["vgain_theta_fp32"]), str(meta.get("vgain_theta_fp32_src", "meta.json"))
    j = f"{path}/vgain_theta_fp32.json"
    if os.path.isfile(j):
        d = json.load(open(j))
        return float(d["vgain_theta_fp32"]), str(d.get("source", "vgain_theta_fp32.json"))
    return None, None


@torch.no_grad()
def load_init(net, path, dev):
    meta = json.load(open(f"{path}/meta.json"))
    assert meta["format"] == "ppm-local-bf16-shards-v1" and os.path.isfile(f"{path}/COMPLETE"), path
    W0 = int(meta["world"])
    params = dict(net.named())
    srcs = {}

    def src(r):
        if r not in srcs:
            srcs[r] = torch.load(f"{path}/rank{r}.pt", map_location="cpu", mmap=True, weights_only=True)
        return srcs[r]
    rep = torch.load(f"{path}/rep.pt", map_location="cpu", mmap=True, weights_only=True)
    ck = set(src(0).keys()) | set(rep.keys())
    skipped = sorted(ck - set(params))
    deep = [k for k in skipped if CFG["test_depth"] and k.startswith(("layers.", "tower_s.layers.", "dec_s.model.layers."))]
    bad = [k for k in skipped if k not in set(deep)]
    assert not bad, ("init keys not in the model", bad[:6])
    SKIPPED_INIT["TEST_deeper_layers"] = len(deep)
    done, nbytes = set(), 0
    for n, p in params.items():
        if n in rep:
            p.copy_(rep[n].to(device=dev, dtype=p.dtype)); done.add(n); continue
        R = p.shape[0]
        c0 = -(-R // W0)
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
                    nbytes += (x1 - x0) * t[0].numel() * t.element_size()
        assert got == r, ("rows not covered", n, got, r)
        done.add(n)
    assert not (set(params) - done)
    srcs.clear()
    th, src_ = read_theta(path, meta)
    if th is not None and M._rows(net.model.vgain.theta)[1]:
        M._loc(net.model.vgain.theta).fill_(th)
    net.model.vadapter.target_rms.fill_(float("nan"))
    nonfin = torch.tensor([float(sum(int(not torch.isfinite(M._loc(p)).all()) for p in params.values()))], device=dev)
    dist.all_reduce(nonfin)
    assert float(nonfin) == 0
    t = torch.tensor([float(nbytes)], device=dev); dist.all_reduce(t)
    return meta, float(t) / 1e9, th, src_


def s_value(net):
    g = net.model.vgain.theta
    return math.exp(float((g.full_tensor() if hasattr(g, "full_tensor") else g).detach().float().reshape(-1)[0]))


# ------------------------------------------------------------------------------------------------ export
@torch.no_grad()
def export_merged(net, step, path, theta, steps):
    """full model, ppm-local-bf16-shards-v1 (world = this world): FSDP2 local shards (backbone + tower roots) per rank, dec_s in rep.pt,
    meta.json (+ layers / tower_layers / dec_layers / proj_r / tower_vocab, vgain_theta_fp32) + vgain_theta_fp32.json, COMPLETE last."""
    t0 = time.time()
    os.makedirs(path, exist_ok=True)
    sh, rp = {}, {}
    for n, p in net.named():
        if hasattr(p, "to_local"):
            sh[n] = p.to_local().detach().to(BF).cpu().contiguous().clone()
        elif RANK == 0:
            rp[n] = p.detach().to(BF).cpu().contiguous().clone()
    torch.save(sh, f"{path}/rank{RANK}.pt.tmp"); os.replace(f"{path}/rank{RANK}.pt.tmp", f"{path}/rank{RANK}.pt")
    nb = sum(t.numel() * 2 for t in sh.values()) + sum(t.numel() * 2 for t in rp.values())
    hosts = [None] * WORLD; dist.all_gather_object(hosts, HOST)
    if RANK == 0:
        torch.save(rp, f"{path}/rep.pt")
        c = json.loads(json.dumps(CFG, default=str))
        c.update({"layers": DEPTHS[0], "tower_layers": DEPTHS[1], "dec_layers": DEPTHS[2], "proj_r": 2048, "tower_vocab": M.TOWER_VOCAB, "steps": steps,
                  "log_root": CFG["run_root"]})
        json.dump({"format": "ppm-local-bf16-shards-v1", "src": "rosie-ft train_rosie_full export_merged", "step": int(step), "world": WORLD, "hosts": hosts,
                   "run_tag": CFG["run_tag"], "init": CFG["init"], "cfg": c, "vgain_theta_fp32": theta, "save_time": now()},
                  open(f"{path}/meta.json", "w"), indent=1, default=str)
        json.dump({"vgain_theta_fp32": theta, "source": f"rosie-ft {CFG['run_tag']} step {step} (vgain frozen: = the init's theta)"},
                  open(f"{path}/vgain_theta_fp32.json", "w"), indent=1)
    dist.barrier()
    if RANK == 0:
        open(f"{path}/COMPLETE", "w").write(str(step))
    t = torch.tensor([float(nb)], device=dev_of(net)); dist.all_reduce(t)
    log(f"[export] {path}: {float(t) / 1e9:.2f} GB bf16 shards (world {WORLD}), {time.time() - t0:.0f}s")


def dev_of(net):
    return next(net.model.parameters()).device


# ------------------------------------------------------------------------------------------------ trunk optimizer: bf16 params, host fp32 master + m / v
class HostMasterAdamW:
    """AdamW (decoupled wd 0, bias-corrected, = torch.optim.AdamW's maths in fp32) for bf16 FSDP2 shards whose fp32 master and m / v live in
    pinned host memory. step(): local shards walked in chunks of CH elements; per chunk H2D (master, m, v) on a copy stream -> fp32 AdamW on
    the GPU (grad = bf16 shard grad x clip coefficient, in fp32) -> bf16 param <- master -> D2H (master, m, v) on a second stream; two staging
    buffers, so copies overlap compute. grad_sq(): this rank's sum of squares of the grads (fp32)."""

    def __init__(self, params, lr, betas, eps, chunk_m, dev):
        self.ps = [p for p in params]
        self.lr, (self.b1, self.b2), self.eps, self.t = float(lr), betas, float(eps), 0
        self.CH = int(chunk_m) * 2 ** 20
        self.dev = dev
        self.host = []
        for p in self.ps:
            loc = M._loc(p).detach()
            assert loc.dtype == BF and loc.is_contiguous(), (loc.dtype, loc.is_contiguous())
            n = loc.numel()
            mst = torch.empty(n, dtype=torch.float32, pin_memory=True)
            mst.copy_(loc.reshape(-1).float().cpu())
            self.host.append((mst, torch.zeros(n, dtype=torch.float32, pin_memory=True), torch.zeros(n, dtype=torch.float32, pin_memory=True)))
        self.buf = [[torch.empty(self.CH, dtype=torch.float32, device=dev) for _ in range(3)] for _ in range(2)]
        self.s_in, self.s_out = torch.cuda.Stream(dev), torch.cuda.Stream(dev)
        self.ev_free = [None, None]

    def host_bytes(self):
        return sum(sum(t.numel() * 4 for t in h) for h in self.host)

    def grad_sq(self):
        s = torch.zeros((), dtype=torch.float32, device=self.dev)
        for p in self.ps:
            if p.grad is not None:
                g = M._loc(p.grad)
                if g.numel():
                    s += g.float().pow(2).sum()
        return s

    @torch.no_grad()
    def step(self, lr_scale, coef):
        self.t += 1
        lr = self.lr * float(lr_scale)
        bc1, bc2 = 1.0 - self.b1 ** self.t, 1.0 - self.b2 ** self.t
        step_size, bc2s = lr / bc1, math.sqrt(bc2)
        main = torch.cuda.current_stream(self.dev)
        k = 0
        for p, (mst, m, v) in zip(self.ps, self.host):
            if p.grad is None:
                continue
            loc = M._loc(p).view(-1)
            g = M._loc(p.grad).view(-1)
            n = loc.numel()
            for a in range(0, n, self.CH):
                b = min(n, a + self.CH)
                L = b - a
                bi = k % 2
                W_, M_, V_ = (x[:L] for x in self.buf[bi])
                with torch.cuda.stream(self.s_in):
                    if self.ev_free[bi] is not None:
                        self.s_in.wait_event(self.ev_free[bi])
                    W_.copy_(mst[a:b], non_blocking=True); M_.copy_(m[a:b], non_blocking=True); V_.copy_(v[a:b], non_blocking=True)
                    ev_in = torch.cuda.Event(); ev_in.record(self.s_in)
                main.wait_event(ev_in)
                gg = g[a:b].float()
                if coef != 1.0:
                    gg.mul_(coef)
                M_.mul_(self.b1).add_(gg, alpha=1.0 - self.b1)
                V_.mul_(self.b2).addcmul_(gg, gg, value=1.0 - self.b2)
                den = V_.sqrt().div_(bc2s).add_(self.eps)
                W_.addcdiv_(M_, den, value=-step_size)
                loc[a:b].copy_(W_)
                del gg, den
                ev_c = torch.cuda.Event(); ev_c.record(main)
                with torch.cuda.stream(self.s_out):
                    self.s_out.wait_event(ev_c)
                    mst[a:b].copy_(W_, non_blocking=True); m[a:b].copy_(M_, non_blocking=True); v[a:b].copy_(V_, non_blocking=True)
                    ev_f = torch.cuda.Event(); ev_f.record(self.s_out)
                self.ev_free[bi] = ev_f
                k += 1
        torch.cuda.synchronize(self.dev)


# ------------------------------------------------------------------------------------------------ micro -> device
ARR = ("ids", "depth", "present", "pid", "sel", "y", "word", "pad", "eob", "nw", "cb0", "light")
ZERO_LAT = {}


def zero_lat(dev):
    if dev not in ZERO_LAT:
        ZERO_LAT[dev] = torch.zeros(1, 4, 1280, dtype=BF, device=dev)
    return ZERO_LAT[dev]


def prep(b, dev, S_text, lat=None, lam=1.0):
    """= train_cotrain.prep's text weights (the tower / decoder weights are built per nanostep). kind 'text' for the backbone forward:
    PPM.forward then stops after the text head (the tower is not a child of the backbone root)."""
    zoom = bool(b.get("zoom", 0))
    t = {"kind": "text", "segs": [int(x) for x in b["segs"]], "zoom": zoom}
    for k in ARR:
        t[k] = torch.as_tensor(np.asarray(b[k])).to(dev, non_blocking=True)
    t["pid"] = t["pid"][None]
    w = {k: t[k].float() for k in ("word", "pad", "eob", "nw")}
    S = float(S_text)
    t["wt"] = ((CFG["w_cat_word"] * w["word"] + CFG["w_tim"] * w["pad"] + CFG["w_eob"] * w["eob"] + CFG["w_nw"] * w["nw"]) / S).float() * lam
    t["wn"] = (CFG["w_text"] * w["word"] / S).float() * lam
    if zoom:
        assert bool((t["depth"] < 0).all()) and bool((t["cb0"] < 0).all()) and t["light"].numel() == 0
        t["vpos"] = torch.zeros(1, dtype=torch.int64, device=dev)
        t["vlat"] = zero_lat(dev)
        t["vmul"] = 0.0
    else:
        P_, n_ = (int(x) for x in np.asarray(b["vin"]).reshape(2))
        t["vpos"] = torch.arange(P_, P_ + n_, device=dev)
        assert lat is not None and tuple(lat.shape) == (1, 4 * n_, 1280)
        t["vlat"] = lat
        assert float(t["present"][0, P_:].abs().sum()) == 0.0 and bool((t["depth"][0] < 0).all()) and bool((t["ids"][0, P_:] == M.PAD).all())
    t["dummy"] = bool(b.get("dummy", 0))
    if t["dummy"]:
        t["wt"], t["wn"] = t["wt"] * 0.0, t["wn"] * 0.0
    return t


TK = ("word_ce", "word_norm", "n_word", "word_hit", "pad_ce", "n_pad", "nw_ce", "n_nw", "eob_ce", "n_eob")
CB_S = (8, 16, 31)
NVA = len(TK) + 5 + len(CB_S)
NVZ = len(TK)


def text_vec(t, st):
    w, p, n_, e = (t[k].float() for k in ("word", "pad", "nw", "eob"))
    return [float((st[0] * w).sum()), float((st[3] * w).sum()), float(w.sum()), float((st[1] * w).sum()), float((st[0] * p).sum()), float(p.sum()),
            float((st[0] * n_).sum()), float(n_.sum()), float((st[0] * e).sum()), float(e.sum())]


def summarize(v, pre):
    d = dict(zip(TK, v[:len(TK)]))
    m = lambda a, b: d[a] / max(d[b], 1.0)  # noqa: E731
    k = f"{pre}_rosie"
    out = {f"{k}/word_ce_norm": m("word_norm", "n_word"), f"{k}/word_ce": m("word_ce", "n_word"), f"{k}/word_top1": m("word_hit", "n_word"),
           f"{k}/pad_ce": m("pad_ce", "n_pad"), f"{k}/nw_ce": m("nw_ce", "n_nw"),
           f"{k}/text_ce": (d["word_ce"] + d["pad_ce"] + d["nw_ce"] + d["eob_ce"]) / max(d["n_word"] + d["n_pad"] + d["n_nw"] + d["n_eob"], 1.0),
           f"{k}/n_word": d["n_word"]}
    x = v[len(TK):]
    out[f"{k}/cb0_ce"] = x[0] / max(x[1], 1.0); out[f"{k}/cb0_top1"] = x[2] / max(x[1], 1.0)
    out[f"{k}/light_ce"] = x[3] / max(x[4], 1.0)
    for j, c in enumerate(CB_S):
        out[f"{k}/light_ce_cb{c}"] = x[5 + j] / max(x[4] / 31.0, 1.0)
    return out


def summarize_zoom(v, pre):
    d = dict(zip(TK, v[:len(TK)]))
    m = lambda a, b: d[a] / max(d[b], 1.0)  # noqa: E731
    k = f"{pre}_zoom"
    return {f"{k}/word_ce_norm": m("word_norm", "n_word"), f"{k}/word_ce": m("word_ce", "n_word"), f"{k}/word_top1": m("word_hit", "n_word"),
            f"{k}/pad_ce": m("pad_ce", "n_pad"), f"{k}/eob_ce": m("eob_ce", "n_eob"),
            f"{k}/text_ce": (d["word_ce"] + d["pad_ce"] + d["nw_ce"] + d["eob_ce"]) / max(d["n_word"] + d["n_pad"] + d["n_nw"] + d["n_eob"], 1.0),
            f"{k}/n_word": d["n_word"]}


# ------------------------------------------------------------------------------------------------ nanostep optimizers
class DecOpt:
    """replicated decoder: grads = views of one flat fp32 buffer, one all-reduce + clip + AdamW per nanostep (= the template's)."""

    def __init__(self, dec, peak, clip, lam):
        self.ps = list(dec.parameters())
        self.flat = torch.zeros(sum(p.numel() for p in self.ps), dtype=torch.float32, device=self.ps[0].device)
        o = 0
        for p in self.ps:
            p.grad = self.flat[o:o + p.numel()].view_as(p); o += p.numel()
        self.opt = torch.optim.AdamW([{"params": self.ps, "lr": 0.0}], betas=(0.9, CFG["adam_beta2"]), eps=1e-8, weight_decay=0.0, fused=True)
        self.n, self.peak, self.clip, self.lam = 0, float(peak), float(clip), lam

    def update(self, no_step=False):
        dist.all_reduce(self.flat)
        self.flat.div_(WORLD)
        lr = self.peak * self.lam(self.n)
        if no_step:
            return lr, float(self.flat.abs().max())
        for g in self.opt.param_groups:
            g["lr"] = lr
        gn = float(torch.nn.utils.clip_grad_norm_(self.ps, self.clip))
        self.opt.step()
        self.n += 1
        return lr, gn


class TowerOpt:
    def __init__(self, tower, peak, clip, lam):
        self.ps = [p for p in tower.parameters()]
        self.opt = torch.optim.AdamW([{"params": self.ps, "lr": 0.0}], betas=(0.9, CFG["adam_beta2"]), eps=1e-8, weight_decay=0.0, fused=True)
        self.n, self.peak, self.clip, self.lam = 0, float(peak), float(clip), lam

    def update(self, no_step=False):
        lr = self.peak * self.lam(self.n)
        g_ = torch.nn.utils.clip_grad_norm_(self.ps, self.clip if not no_step else 1e30)
        gn = float(g_.full_tensor() if hasattr(g_, "full_tensor") else g_)
        if not no_step:
            for g in self.opt.param_groups:
                g["lr"] = lr
            self.opt.step()
            self.n += 1
        self.opt.zero_grad(set_to_none=True)
        return lr, gn


class DecFast:
    def __enter__(self):
        self.prev = M.DEC_FAST["active"]; M.DEC_FAST["active"] = CFG["dec_fast"]

    def __exit__(self, *a):
        M.DEC_FAST["active"] = self.prev


def _mb_plan(m):
    cap = CFG["dec_mb_max"]
    mm = max(m, 1)
    nmb = -(-mm // cap)
    sz = -(-mm // nmb)
    if not CFG["dec_dyn"]:
        sz = -(-sz // 32) * 32
    return [(a0, max(z, 16)) for a0, z in [(i * sz, sz) for i in range(nmb) if i * sz < mm]]


def dec_sub(dec, do, hv, depth, light, subs, Msub, dummy, no_step=False):
    """KS decoder nanosteps on hv.detach() over the light-frame index lists `subs` (indices into `light`), Msub = their global sizes. ->
    (dL/dhv of the chunk-mean light CE [fp32], stats sum / n / per-codebook, lrs, grad norms)."""
    dev = hv.device
    hd = hv.detach().requires_grad_()
    N = float(sum(Msub))
    acc = torch.zeros(hd.shape, dtype=torch.float32, device=dev)
    st = {"sum": torch.zeros((), device=dev), "n": 0, "cb": torch.zeros(31, device=dev)}
    lrs, gns = [], []
    for i, idx in enumerate(subs):
        m = int(idx.numel())
        hd.grad = None
        do.flat.zero_()
        den = 31.0 * max(Msub[i], 1.0) / WORLD
        for a0, sz in _mb_plan(m):
            mv = max(0, min(sz, m - a0))
            fl = light.index_select(0, idx[a0:a0 + mv]) if mv else light[:1]
            if fl.numel() < sz:
                fl = torch.cat([fl, light[:1].expand(sz - fl.numel())])
            wf = torch.zeros(sz, dtype=torch.float32, device=dev)
            if mv and not dummy:
                wf[:mv] = 1.0 / den
            l_, s_, num_, cb_ = dec(hd, depth, fl, wf, n_valid=mv)
            if mv and not dummy:
                st["sum"] += s_; st["n"] += num_; st["cb"] += cb_.detach().float()
            l_.backward()
        if hd.grad is not None and m and not dummy:
            acc += hd.grad.float() * (Msub[i] / max(N, 1.0))
        lr, gn = do.update(no_step)
        lrs.append(lr); gns.append(gn)
    hd.grad = None
    return acc, st, lrs, gns


def partition(b, step, lrank):
    """the rosie micro's tower chunks (cb0-labelled positions, seeded permutation, KT chunks) and decoder sub-chunks (light frames by their
    position's chunk, KS each) -> (pos lists [KT], light-index lists [KT][KS])."""
    cb0, light = np.asarray(b["cb0"]), np.asarray(b["light"])
    labpos = np.nonzero(cb0 >= 0)[0]
    rng = np.random.default_rng(np.random.SeedSequence([CFG["dec_seed"], int(step), int(lrank), 7]))
    perm = rng.permutation(len(labpos))
    pos = [np.sort(labpos[perm[j::KT]]) for j in range(KT)]
    ch = np.full(len(cb0), -1, np.int64)
    for j in range(KT):
        ch[pos[j]] = j
    lc = ch[light]
    free = np.nonzero(lc < 0)[0]
    lc[free] = np.arange(len(free)) % KT
    subs = []
    for j in range(KT):
        li = np.nonzero(lc == j)[0]
        li = li[rng.permutation(len(li))]
        subs.append([np.sort(li[i::KS]) for i in range(KS)])
    return pos, subs


def make_front(dev):
    stub = os.environ.get("VOX_TEST_STUB_FRONT", "0") == "1"
    return AU.make_front({"s4_audio": "stub" if stub else "flac", "s4_vox_model": CFG["vox_model"], "s4_enc_hidden": 1280}, dev)


def encode(front, b):
    ref = json.loads(b["aref"])
    lat = front.encode([torch.from_numpy(np.ascontiguousarray(b["wav"]))], [{"row_key": ref["row_key"], "n": int(ref["n"])}])
    assert torch.isfinite(lat).all()
    return lat


class StepDS:
    """element t = [rosie micro (plan item [t][RANK]), zoom micro(s)] of this rank (= train_rosie_cotrain.StepDS)."""

    def __init__(self, plan, slots, steps, root, light_frac, repeat=False):
        self.plan, self.slots, self.T, self.root, self.lf, self.repeat = plan, list(slots), int(steps), root, float(light_frac), repeat
        self.rd, self.zst, self.zr = None, None, None

    def __len__(self):
        return self.T

    def __getitem__(self, t):
        t = int(t)
        tt = 0 if self.repeat else t
        out = []
        for g, (kind, L) in enumerate(self.slots):
            if kind == "rosie":
                if self.rd is None:
                    self.rd = DATA.RosieData(self.root, "train", light_frac=self.lf)
                it = self.plan["train"][tt][L]
                m = self.rd.micro(tuple(it) if it is not None else tuple(self.plan["fallback"]), dummy=it is None)
            else:
                if self.zst is None:
                    tbz, _ = Z.load_table()
                    self.zst, self.zr = Z.Stream(tbz["train"]), Z.Renderer("train")
                e, k, pk = self.zst.get(tt * ZL + (L if L is not None else 0))
                m = Z.as_noop_speech(Z.chat_pack_micro(self.zr, pk), dummy=L is None)
            m.update(step=t, slot=g, lrank=(-1 if L is None else int(L)))
            out.append(m)
        return out


DEPTHS = [64, 36, 28]


def main():
    dev = torch.device("cuda", LOCAL)
    torch.cuda.set_device(dev)
    cap = float(os.environ.get("ROSIE_MEM_CAP_GB", "0"))
    if cap > 0:
        torch.cuda.set_per_process_memory_fraction(min(1.0, cap * 1e9 / torch.cuda.get_device_properties(dev).total_memory), dev)
    dist.init_process_group("nccl", device_id=dev, timeout=datetime.timedelta(seconds=int(E("INIT_TIMEOUT", "1800"))))
    os.makedirs(LOGDIR, exist_ok=True)
    plan = DATA.make_plan(LWR, epochs=CFG["epochs"], root=CFG["data_root"])
    steps = plan["steps"] if CFG["steps_env"] == "auto" else int(CFG["steps_env"])
    assert 0 < steps <= plan["steps"]
    CFG["steps"] = steps
    CFG["export_steps"] = sorted(set(CFG["export_steps"]) | (set(range(CFG["export_every"], steps, CFG["export_every"])) if CFG["export_every"] else set()))
    lam_main, lam_tower, lam_dec = wsd(CFG["warmup"], steps), wsd(CFG["warmup"] * KT, steps * KT), wsd(CFG["warmup"] * KD, steps * KD)
    tbz, zmeta = Z.load_table()
    n_ep0 = len(Z.epoch_packs(tbz["train"], 0))
    CFG["plan"] = {"n_train": plan["n_train"], "n_eval": plan["n_eval"], "steps_per_epoch": plan["steps_per_epoch"], "epochs": plan["epochs"],
                   "zoom_packs_per_epoch": n_ep0, "zoom_table_key": zmeta["cfg_key"]}
    log(f"rosie-ft FULL cotrain {CFG['run_tag']}: world {WORLD} host {HOST}; per rank per step: 1 rosie micro + {GZ} zoom micro(s) (rank 0 slots "
        f"{SLOTS}); W_ROSIE {W_ROSIE:g} W_ZOOM {W_ZOOM:g}; tower nanosteps K_T {KT}, decoder nanosteps K_D {KD} ({KS} per tower nanostep)")
    log(f"[data] ROSIE {plan['n_train']} train items -> {plan['steps_per_epoch']} steps / epoch x {plan['epochs']} = {plan['steps']}; RUN {steps} steps; "
        f"eval {plan['n_eval']} items | ZOOM {n_ep0} packs / epoch, {steps * ZL} in the run ({steps * ZL / n_ep0:.3f} epochs), table key {zmeta['cfg_key']}")
    log(f"[cfg] {json.dumps(CFG, default=str)}")
    if RANK == 0:
        json.dump(CFG, open(f"{LOGDIR}/cfg.json", "w"), indent=1, default=str)
    nl, nt, nd = (int(x) for x in (CFG["test_depth"] or "64,36,28").split(","))
    DEPTHS[:] = [nl, nt, nd]
    c32, c4, c06 = M.cfg32(nl), M.cfg4b(nt), M.cfg06(nd)
    net = build(c32, c4, c06, dev)
    model, tower = net.model, net.tower
    t0 = time.time()
    meta, gb, theta, th_src = load_init(net, CFG["init"], dev)
    assert theta is not None
    if "bf16" in str(th_src):
        log(f"!!! [init] WARNING vgain theta {theta:.9g} (s {math.exp(theta):.9g}) is export-253's BF16 theta ({th_src}); s may be ~0.3 % off")
    log(f"[init] {CFG['init']}: step {meta.get('step')} run {meta.get('run_tag')} world {meta.get('world')} -> resharded onto {WORLD}; {gb:.2f} GB "
        f"read in {time.time() - t0:.0f}s; skipped {SKIPPED_INIT}")
    CFG["vgain_theta_fp32"] = theta
    CAP = {}
    model.head.register_forward_hook(lambda mod, inp, out: CAP.__setitem__("hn", out[2]))
    front = make_front(dev)
    if CFG["dec_fast"]:
        M.compile_decoder(model.dec_s, CFG["dec_compile_mode"], whole=True, dynamic=CFG["dec_dyn"])
    rot = M.Rot(c32, dev)
    # ---- optimizers
    by = {g: [] for g in MAIN_GROUPS}
    frozen = []
    for n, p in net.named():
        g = M.group_of(n)
        if g in FROZEN_GROUPS:
            frozen.append(p)
        elif g in MAIN_GROUPS:
            by[g].append(p)
        else:
            assert g in ("tower_s", "dec_s"), (n, g)
    kw = dict(betas=(0.9, CFG["adam_beta2"]), eps=1e-8, weight_decay=0.0, fused=True)
    t0 = time.time()
    if CFG["trunk_offload"]:
        hopt = {g: HostMasterAdamW(by[g], CFG["lr"][g], (0.9, CFG["adam_beta2"]), 1e-8, CFG["offload_chunk_m"], dev) for g in HOST_GROUPS}
        gpu_groups = ("proj", "audio")
    else:
        hopt = {}
        gpu_groups = MAIN_GROUPS
    gopt = torch.optim.AdamW([{"params": by[g], "lr": 0.0, "name": g} for g in gpu_groups if by[g]], **kw)
    hb = sum(o.host_bytes() for o in hopt.values())
    log(f"[opt] trunk + spec: {'HostMasterAdamW (bf16 GPU shards, fp32 master + m + v pinned host ' + f'{hb / GiB:.2f} GiB on rank 0, built in {time.time() - t0:.0f}s)' if hopt else 'GPU AdamW fp32'}"
        f" LR {CFG['lr']['trunk']:g} clip {CFG['clip']['trunk']:g} | GPU AdamW {[(pg['name'], len(pg['params'])) for pg in gopt.param_groups]} | "
        f"tower_s nanostep AdamW LR {CFG['lr']['tower_s']:g} x {KT} | dec_s nanostep AdamW LR {CFG['lr']['dec_s']:g} x {KD}; WSD warmup {CFG['warmup']} "
        f"(main), {CFG['warmup'] * KT} (tower), {CFG['warmup'] * KD} (dec); note: the template's WSD gives LR 0 at the first update of each optimizer")
    to = TowerOpt(tower, CFG["lr"]["tower_s"], CFG["clip"]["tower_s"], lam_tower)
    do = DecOpt(model.dec_s, CFG["lr"]["dec_s"], CFG["clip"]["dec_s"], lam_dec)
    cnt = {g: (len(v), sum(p.numel() for p in v)) for g, v in by.items()}
    cnt["tower_s (nanostep)"] = (len(to.ps), sum(p.numel() for p in to.ps)); cnt["dec_s (nanostep)"] = (len(do.ps), sum(p.numel() for p in do.ps))
    cnt["FROZEN vadapter+vgain"] = (len(frozen), sum(p.numel() for p in frozen))
    log("[params] " + ", ".join(f"{g} ({a}, {b / 1e6:.2f}M)" for g, (a, b) in cnt.items()))

    def frozen_fp():
        s = torch.zeros(2, dtype=torch.float64, device=dev)
        for p in frozen:
            x = M._loc(p).detach().double(); s[0] += x.sum(); s[1] += (x * x).sum()
        dist.all_reduce(s)
        return s.tolist()
    fz0 = frozen_fp()
    log(f"[mem after init] allocated {torch.cuda.memory_allocated(dev) / GiB:.2f} reserved {torch.cuda.memory_reserved(dev) / GiB:.2f} GiB (rank 0)")
    # ---- data
    from torch.utils.data import DataLoader
    W_ = CFG["dl_workers"]
    ds = StepDS(plan, SLOTS, steps, CFG["data_root"], CFG["light_frac"], repeat=CFG["test_repeat"])
    ld = iter(DataLoader(ds, batch_size=None, shuffle=False, num_workers=W_, prefetch_factor=2 if W_ else None, persistent_workers=bool(W_),
                         pin_memory=False, worker_init_fn=(lambda _: torch.set_num_threads(1)) if W_ else None, collate_fn=lambda x: x))
    rd_e = DATA.RosieData(CFG["data_root"], "eval", light_frac=1.0)
    its = plan["eval"][:CFG["rosie_eval_max"]] if CFG["rosie_eval_max"] else plan["eval"]
    # eval lists PER KIND, each padded with dummies OF ITS OWN KIND to the max over ranks, rosie first then zoom: every rank runs the same
    # sequence of backbone / tower calls (the tower is a separate FSDP2 root: a kind mismatch across ranks deadlocks the collectives)
    evr = [rd_e.micro(tuple(it)) for it in its[RANK::WORLD]]
    evz = []
    if CFG["zoom_eval_max"]:
        rend = Z.Renderer("eval")
        evz = [Z.as_noop_speech(Z.chat_pack_micro(rend, pk)) for pk in Z.eval_packs(tbz["eval"])[:CFG["zoom_eval_max"]][RANK::WORLD]]
    t_ = torch.tensor([float(len(evr)), float(len(evz))], device=dev); dist.all_reduce(t_, op=dist.ReduceOp.MAX)
    nr_, nz_ = int(t_[0]), int(t_[1])
    if nr_:
        src_r = evr[0] if evr else rd_e.micro(tuple(plan["eval"][0]))
        evr += [dict(src_r, dummy=1, own=np.zeros_like(src_r["own"])) for _ in range(nr_ - len(evr))]
    if nz_:
        src_z = evz[0] if evz else Z.as_noop_speech(Z.chat_pack_micro(Z.Renderer("eval"), Z.eval_packs(tbz["eval"])[0]))
        evz += [dict(src_z, dummy=1, own_zoom=0.0) for _ in range(nz_ - len(evz))]
    ev = evr + evz
    n_ev = len(ev)
    c_ = allsum([float(sum(1 for m in ev if not m.get("dummy") and not m.get("zoom"))), float(sum(1 for m in ev if not m.get("dummy") and m.get("zoom")))], dev)
    log(f"[eval] {n_ev} eval micros per rank: rosie {int(c_[0])} held-out chunks, zoom {int(c_[1])} eval packs")
    jl = open(f"{LOGDIR}/metrics.jsonl", "a") if RANK == 0 else None
    wb = None
    if CFG["wandb"] and RANK == 0:
        import wandb
        wb = wandb.init(project=os.environ.get("WANDB_PROJECT", "rosie-ft"), name=CFG["run_tag"], config=json.loads(json.dumps(CFG, default=str)))

    def emit(d, step):
        if RANK == 0:
            jl.write(json.dumps({"step": step, "time": now(), **d}) + "\n"); jl.flush()
            if wb is not None:
                wb.log(d, step=step)

    @torch.no_grad()
    def eval_one(m):
        """-> (rosie vec [NVA], zoom vec [NVZ], text loss): micro-normalised, current weights."""
        if m.get("zoom"):
            bt = prep(m, dev, max(float(m["own_zoom"]), 1.0))
            out = model(bt, rot, rot); CAP.pop("hn", None)
            vz = text_vec(bt, out["text_stats"])
            return [0.0] * NVA, ([0.0] * NVZ if m.get("dummy") else vz), float(out["text"])
        oc = np.asarray(m["own"]).sum(0)
        bt = prep(m, dev, max(oc[0], 1.0), lat=encode(front, m))
        out = model(bt, rot, rot)
        hn = CAP.pop("hn")
        ctx = M.Ctx(bt["segs"], dev)
        lab = bt["cb0"]
        w0 = (lab >= 0).float() / max(oc[1], 1.0)
        hv, l0, s0, n0, k0 = tower(bt["ids"][1], bt["depth"][1], hn, bt["pid"], ctx, rot, lab, w0)
        light = bt["light"]
        fl = light[torch.randperm(light.numel(), generator=torch.Generator().manual_seed(1234))[:CFG["eval_light_max"]].to(dev)]
        _, s_, num_, cb_ = model.dec_s(hv, bt["depth"][1], fl, torch.zeros(fl.numel(), device=dev))
        v = text_vec(bt, out["text_stats"]) + [float(s0), float(n0), float(k0), float(s_), float(num_)] + [float(cb_[c - 1]) for c in CB_S]
        return ([0.0] * NVA if m.get("dummy") else v), [0.0] * NVZ, float(out["text"]) + float(l0)

    def evaluate(step):
        model.eval(); tower.eval()
        t0_ = time.time()
        acc, accz = [0.0] * NVA, [0.0] * NVZ
        for m in ev:
            v, vz, _ = eval_one(m)
            acc = [x + y for x, y in zip(acc, v)]; accz = [x + y for x, y in zip(accz, vz)]
        g, gz = allsum(acc, dev), allsum(accz, dev)
        d = summarize(g, "eval"); d.update(summarize_zoom(gz, "eval"))
        d["eval/vox_a_scale"] = s_value(net)
        model.train(); tower.train()
        d["eval/time_s"] = time.time() - t0_
        emit(d, step)
        log(f"EVAL step {step}: " + " | ".join(f"{k} {v:.4f}" for k, v in d.items()))
        for k_, v_ in d.items():
            assert math.isfinite(v_), (k_, v_)
        return d

    def micro_losses(ms):
        model.eval(); tower.eval()
        out = [eval_one(m)[2] for m in ms]
        model.train(); tower.train()
        return out

    MC = evr[:1] + evz[:1]                                         # merge check: one micro of each kind (same kinds on every rank)
    evaluate(0)
    if CFG["test_export0"]:
        export_merged(net, 0, f"{LOGDIR}/export-0", theta, steps)
    import gc
    gc.collect(); gc.freeze(); gc.disable()
    times = []
    for step in range(steps):
        torch.cuda.synchronize(); ts = time.time()
        torch.cuda.reset_peak_memory_stats()
        tw = time.time()
        bs = next(ld)
        wait = time.time() - tw
        assert len(bs) == len(SLOTS) and all(int(b["step"]) == step for b in bs)
        br, bz = bs[0], bs[1:]
        assert not br.get("zoom") and all(b.get("zoom") for b in bz)
        ocr = np.asarray(br["own"]).sum(0)
        pos, subs = partition(br, step, RANK)
        dmy = bool(br.get("dummy", 0))
        cnts = [0.0 if dmy else float(len(p)) for p in pos] + [0.0 if dmy else float(len(s)) for ss in subs for s in ss]
        tot = allsum([float(ocr[0]), float(ocr[1]), float(ocr[2]), float(sum(float(b.get("own_zoom", 0.0)) for b in bz))] + cnts, dev)
        Cj = tot[4:4 + KT]
        Msub = [tot[4 + KT + j * KS:4 + KT + (j + 1) * KS] for j in range(KT)]
        C, Mtot = max(sum(Cj), 1.0), max(sum(sum(x) for x in Msub), 1.0)
        share_dev = max(abs(Cj[j] / C - sum(Msub[j]) / Mtot) for j in range(KT))
        if step == 0:
            log(f"[pack] step 0: ROSIE text {int(tot[0])} cb0 labels {int(tot[1])} light {int(tot[2])}; tower chunks {[int(x) for x in Cj]}, light per "
                f"chunk {[int(sum(x)) for x in Msub]} (max share deviation {share_dev:.2e}) | ZOOM text {int(tot[3])}")

        def rosie_pass(no_step=False):
            """rosie micro: backbone fwd, KT tower nanosteps (each with KS decoder nanosteps), ONE backbone backward. -> (vec, loss, info)."""
            lat = encode(front, br)
            bt = prep(br, dev, max(tot[0], 1.0) / WORLD, lat=lat, lam=W_ROSIE)
            del lat
            out = model(bt, rot, rot)
            hn = CAP.pop("hn")
            hn_d = hn.detach().requires_grad_()
            ctx = M.Ctx(bt["segs"], dev)
            lab = bt["cb0"]
            light = bt["light"]
            dhn = torch.zeros(hn.shape, dtype=torch.float32, device=dev)
            cb0_st, dst, lrs_t, gn_t, lrs_d, gn_d, lcb = None, {"sum": 0.0, "n": 0, "cb": torch.zeros(31, device=dev)}, [], [], [], [], 0.0
            for j in range(KT):
                w0 = torch.zeros(lab.shape[0], dtype=torch.float32, device=dev)
                if not dmy and len(pos[j]):
                    w0[torch.as_tensor(pos[j], device=dev)] = W_ROSIE * CFG["w_cb0"] / (max(Cj[j], 1.0) / WORLD)
                hv, l0, s0, n0, k0 = tower(bt["ids"][1], bt["depth"][1], hn_d, bt["pid"], ctx, rot, lab, w0)
                if j == 0:
                    cb0_st = (float(s0), float(n0), float(k0))
                lcb += float(l0.detach()) * Cj[j] / C
                sl = [torch.as_tensor(s, device=dev, dtype=torch.long) for s in subs[j]]
                with DecFast():
                    acc, st, lr_d, g_d = dec_sub(model.dec_s, do, hv, bt["depth"][1], light, sl, Msub[j], dmy, no_step)
                dst["sum"] += float(st["sum"]); dst["n"] += int(st["n"]); dst["cb"] += st["cb"]
                lrs_d += lr_d; gn_d += g_d
                torch.autograd.backward([l0, hv], [None, (acc * (CFG["w_cb"] * W_ROSIE)).to(hv.dtype)])
                dhn += hn_d.grad.float() * (Cj[j] / C)
                hn_d.grad = None
                lr_t, g_t = to.update(no_step)
                lrs_t.append(lr_t); gn_t.append(g_t)
                del hv, l0, acc
            torch.autograd.backward([out["text"], hn], [None, dhn.to(hn.dtype)])
            v = text_vec(bt, out["text_stats"]) + list(cb0_st) + [dst["sum"], float(dst["n"])] + [float(dst["cb"][c - 1]) for c in CB_S]
            if dmy:
                v = [0.0] * NVA
            return v, float(out["text"].detach()) + lcb, {"lrs_t": lrs_t, "gn_t": gn_t, "lrs_d": lrs_d, "gn_d": gn_d}

        def zoom_pass():
            vz, lz = [0.0] * NVZ, 0.0
            for b in bz:
                bt = prep(b, dev, max(tot[3], 1.0) / WORLD, lam=W_ZOOM)
                out = model(bt, rot, rot)
                CAP.pop("hn", None)
                out["text"].backward()
                if not b.get("dummy"):
                    vz = [x + y for x, y in zip(vz, text_vec(bt, out["text_stats"]))]
                lz += float(out["text"].detach())
            return vz, lz

        if step == 0 and CFG["test_gradsplit"]:
            gradsplit(net, by, zoom_pass, do, to, dev)
        t_m = time.time()
        va, loss_r, info = rosie_pass()
        torch.cuda.synchronize(); t_r = time.time() - t_m
        vz, loss_z = zoom_pass()
        torch.cuda.synchronize(); t_z = time.time() - t_m - t_r
        # ---- main update (trunk host-master + GPU groups), per-group clip
        t_o = time.time()
        lam = lam_main(step)
        gn = {}
        for g, o in hopt.items():
            gsq = o.grad_sq(); dist.all_reduce(gsq)
            gn[g] = float(gsq.sqrt())
            o.step(lam, min(1.0, CFG["clip"][g] / (gn[g] + 1e-6)))
        for pg in gopt.param_groups:
            g_ = torch.nn.utils.clip_grad_norm_(pg["params"], CFG["clip"][pg["name"]])
            gn[pg["name"]] = float(g_.full_tensor() if hasattr(g_, "full_tensor") else g_)
            pg["lr"] = CFG["lr"][pg["name"]] * lam
        gopt.step()
        gopt.zero_grad(set_to_none=True)
        for g in hopt:
            for p in by[g]:
                p.grad = None
        torch.cuda.synchronize(); t_opt = time.time() - t_o
        dt = time.time() - ts; times.append(dt)
        sv = allsum(list(va) + list(vz) + [loss_r, loss_z], dev)
        d = summarize(sv[:NVA], "train"); d.update(summarize_zoom(sv[NVA:NVA + NVZ], "train"))
        d["train/loss_rosie"], d["train/loss_zoom"] = sv[-2] / WORLD, sv[-1] / WORLD
        d["train/loss"] = d["train/loss_rosie"] + d["train/loss_zoom"]
        pk = torch.tensor([torch.cuda.max_memory_allocated() / GiB, dt, wait, t_r, t_z, t_opt], device=dev)
        dist.all_reduce(pk, op=dist.ReduceOp.MAX)
        d.update({f"train/gn_{k}": x for k, x in gn.items()})
        d.update({f"train/lr_{g}": CFG["lr"][g] * lam for g in MAIN_GROUPS})
        d.update({"train/lr_tower_last": info["lrs_t"][-1], "train/gn_tower_mean": float(np.mean(info["gn_t"])), "train/lr_dec_last": info["lrs_d"][-1],
                  "train/gn_dec_mean": float(np.mean(info["gn_d"])), "train/step_time": dt, "train/peak_gib": float(pk[0]), "train/data_wait": float(pk[2]),
                  "train/t_rosie": float(pk[3]), "train/t_zoom": float(pk[4]), "train/t_opt": float(pk[5]), "train/light_share_dev": share_dev,
                  "train/text_targets": tot[0], "train/cb0_labels": tot[1], "train/light_frames": tot[2], "train/zoom_text_targets": tot[3]})
        for k_, v_ in d.items():
            assert math.isfinite(v_), (k_, v_)
        done = step + 1
        emit(d, done)
        eta = (steps - done) * float(np.mean(times[-20:]))
        log(f"step {done}/{steps} {dt:.1f}s (rosie {float(pk[3]):.1f} zoom {float(pk[4]):.1f} opt {float(pk[5]):.1f} wait {float(pk[2]):.1f}) peak "
            f"{float(pk[0]):.1f} GiB | " + " ".join(f"{k.split('/', 1)[0][6:]}.{k.split('/', 1)[1]}={v:.4f}" for k, v in d.items()
                                                    if k.startswith(("train_rosie/", "train_zoom/")) and k.endswith(("word_ce_norm", "cb0_ce", "light_ce", "pad_ce")))
            + f" loss_rosie={d['train/loss_rosie']:.4f} loss_zoom={d['train/loss_zoom']:.4f} "
            + " ".join(f"{k.split('/')[1]}={v:.4g}" for k, v in d.items() if "/gn_" in k or k in ("train/lr_trunk", "train/lr_tower_last", "train/lr_dec_last"))
            + f" | ETA {eta / 60:.1f} min")
        gc.collect(1)
        last = done == steps or (CFG["test_steps"] and done >= CFG["test_steps"])
        if (CFG["eval_every"] and done % CFG["eval_every"] == 0) or last or done in CFG["export_steps"]:
            evaluate(done)
        if done in CFG["export_steps"] or (last and (not CFG["test_steps"] or CFG["test_mergecheck"])):
            p_ = f"{LOGDIR}/export-{done}"
            ml = micro_losses(MC) if CFG["test_mergecheck"] else None
            export_merged(net, done, p_, theta, steps)
            if ml is not None and last:
                load_init(net, p_, dev)
                ml2 = micro_losses(MC)
                w_ = torch.tensor([max(abs(a - b) / max(abs(a), 1e-9) for a, b in zip(ml, ml2))], device=dev)
                dist.all_reduce(w_, op=dist.ReduceOp.MAX)
                log(f"[mergecheck] {p_} reloaded: eval-micro losses rank 0 before {ml} after {ml2}; max relative diff {float(w_):.3e} (bf16 rounding of the fp32 tensors)")
                assert float(w_) < 5e-3
        if last:
            break
    fz1 = frozen_fp()
    assert fz1 == fz0, ("vadapter / vgain changed", fz0, fz1)
    log(f"[frozen] vadapter + vgain fingerprint unchanged ({fz0[0]:.6e}, {fz0[1]:.6e})")
    log(f"done {len(times)} steps; mean step {np.mean(times):.2f}s (excl. step 1: {np.mean(times[1:]) if len(times) > 1 else float('nan'):.2f}s)")
    dist.barrier()
    dist.destroy_process_group()


def gradsplit(net, by, zoom_pass, do, to, dev):
    """TEST: the step-0 zoom micros alone (backward only, no update): exact-zero gradient on audio (inp.aud), no tower_s / dec_s gradient at
    all (they are never run on zoom micros), non-zero trunk / proj gradients. Grads cleared afterwards."""
    do.flat.zero_()
    zoom_pass()
    aud = max([float(M._loc(p.grad).abs().max()) if (p.grad is not None and M._loc(p.grad).numel()) else 0.0 for p in by["audio"]] + [0.0])
    tw = sum(1 for p in to.ps if p.grad is not None)
    fl = float(do.flat.abs().max())
    gtr = sum(float(M._loc(p.grad).float().pow(2).sum()) for p in by["trunk"] if p.grad is not None)
    gpr = sum(float(M._loc(p.grad).float().pow(2).sum()) for p in by["proj"] if p.grad is not None)
    t = torch.tensor([aud, float(tw), fl], device=dev); dist.all_reduce(t, op=dist.ReduceOp.MAX)
    s = allsum([gtr, gpr], dev)
    log(f"[gradsplit] zoom-only pass: max |grad audio| {float(t[0]):.3e}, tower_s tensors with a grad {int(t[1])}, decoder buffer max {float(t[2]):.3e} "
        f"(want 0 / 0 / 0); grad L2 trunk {math.sqrt(s[0]):.4e} proj {math.sqrt(s[1]):.4e} (want > 0)")
    assert float(t[0]) == 0.0 and int(t[1]) == 0 and float(t[2]) == 0.0 and s[0] > 0 and s[1] > 0
    for g in by.values():
        for p in g:
            p.grad = None
    log("[gradsplit] PASS")


if __name__ == "__main__":
    main()
