"""D36 resume / equivalence test on a TINY-width PPM (hidden 256, 1 layer each), real micros of this node's stores, the real trainer
functions (train_ppm.load / save / make_opt / run_micro / summarize, safety_ppm). torchrun x 8, one process per phase; node-local /scratch only.

usage: torchrun ... tests/test_d36_ckpt.py <code dir> <phase> <root>
  a       (head ON, D35 full detach s = 0 from step 0) random init (head NON-zero), 2 optimizer steps, save dcp-2 = a synthetic checkpoint whose
          head has weights + main-AdamW state (= the prod situation at the next restart)
  on      (knob unset, new code)  resume dcp-2, step 3: record stats / losses / LRs / grads / weights after                    reference
  on2     (knob unset, new code)  the same again                                                                               noise floor
  old     (knob unset, the pre-D36 code = <code dir>)  the same                                                                (c) knob off == old code
  off     (PPM_POSEHEAD_OFF=1)    resume dcp-2 (digest incl. the set-aside head AdamW state), step 3 recorded as above; head weights == the
          checkpoint's, no optimizer state / group / LambdaLR entry, no grad, unchanged by the step; save dcp-3           (a) + (b) vs on
  off2    (OFF=1)  resume dcp-3 (a checkpoint saved with the head OFF: no head AdamW state, 7 LR groups), step 4, save dcp-4    2nd restart
  revive  (knob unset)  resume dcp-4: head weights == dcp-2's, AdamW fresh, LambdaLR entry re-inserted, warmup restarts (LR 0 then 2e-6);
          steps 5, 6: head trains again, its stats come back
Determinism (test only): FA3 varlen replaced by a per-segment SDPA-math attention (as tests/test_d32.py), torch deterministic algorithms."""
import hashlib
import json
import math
import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
CODE, PHASE, ROOT = sys.argv[1], sys.argv[2], sys.argv[3]
for p in (os.path.join(CODE, "vendor"), os.path.join(CODE, "lib"), CODE):
    sys.path.insert(0, p)
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.nn.attention import SDPBackend, sdpa_kernel  # noqa: E402

torch.use_deterministic_algorithms(True, warn_only=True)
import train_ppm as T  # noqa: E402

M, D, CFG, SF = T.M, T.D, T.CFG, T.SF
assert os.path.abspath(os.path.dirname(M.__file__)) == os.path.abspath(CODE), (M.__file__, CODE)
assert os.path.abspath(os.path.dirname(SF.__file__)) == os.path.abspath(CODE), (SF.__file__, CODE)
H2 = M.H2
PH_OFF = bool(getattr(T, "PH_OFF", False))
HK = ("pose_head.weight", "pose_head.bias")


def _att(qs, ks, vs, scale, causal):
    with sdpa_kernel(SDPBackend.MATH):
        return F.scaled_dot_product_attention(qs, ks, vs, is_causal=causal, scale=scale)


class _DetAttn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, cu, scale, causal):
        ctx.save_for_backward(q, k, v)
        ctx.cu, ctx.scale, ctx.causal = [int(x) for x in cu.tolist()], float(scale), bool(causal)
        rep = q.shape[1] // k.shape[1]
        out = torch.empty_like(q)
        for a, b in zip(ctx.cu[:-1], ctx.cu[1:]):
            for g in range(k.shape[1]):
                sl = slice(g * rep, (g + 1) * rep)
                o = _att(q[a:b, sl].transpose(0, 1)[None], k[a:b, g:g + 1].expand(-1, rep, -1).transpose(0, 1)[None],
                         v[a:b, g:g + 1].expand(-1, rep, -1).transpose(0, 1)[None], ctx.scale, ctx.causal)
                out[a:b, sl] = o[0].transpose(0, 1)
        return out

    @staticmethod
    def backward(ctx, go):
        q, k, v = ctx.saved_tensors
        rep = q.shape[1] // k.shape[1]
        dq, dk, dv = torch.zeros_like(q), torch.zeros_like(k), torch.zeros_like(v)
        for a, b in zip(ctx.cu[:-1], ctx.cu[1:]):
            for g in range(k.shape[1]):
                sl = slice(g * rep, (g + 1) * rep)
                with torch.enable_grad():
                    qs = q[a:b, sl].detach().requires_grad_(); ks = k[a:b, g:g + 1].detach().requires_grad_(); vs = v[a:b, g:g + 1].detach().requires_grad_()
                    o = _att(qs.transpose(0, 1)[None], ks.expand(-1, rep, -1).transpose(0, 1)[None], vs.expand(-1, rep, -1).transpose(0, 1)[None],
                             ctx.scale, ctx.causal)
                    gq, gk, gv = torch.autograd.grad(o, (qs, ks, vs), go[a:b, sl].transpose(0, 1)[None])
                dq[a:b, sl] = gq; dk[a:b, g:g + 1] = gk; dv[a:b, g:g + 1] = gv
        return dq, dk, dv, None, None, None


M.fa3_op.fa3_varlen_func = lambda q, k, v, cu_q, cu_k, max_q, max_k, scale, causal: _DetAttn.apply(q, k, v, cu_q, scale, causal)


def tiny():
    c32, c4 = M.cfg32(1), M.cfg4b(1)
    for c in (c32, c4):
        c.hidden_size, c.intermediate_size, c.num_attention_heads, c.num_key_value_heads, c.head_dim = 256, 512, 2, 1, 128
    c06 = H2.qwen3_light_config({"num_hidden_layers": 1, "hidden_size": 256, "intermediate_size": 512, "num_attention_heads": 2,
                                 "num_key_value_heads": 1, "head_dim": 128})
    return c32, c4, c06


def log(*a):
    if T.RANK == 0:
        print(f"[d36-{PHASE}]", *a, flush=True)


@torch.no_grad()
def rand_init(model):
    for n, p in model.named_parameters():
        n = n.replace("_checkpoint_wrapped_module.", "")
        kind = "zeros" if n in ("head.proj.D", "head.proj.U") else ("ones" if n.endswith("norm.weight") or n.endswith("layernorm.weight") else "normal")
        a, _ = M._rows(p)
        H2.init_rows_(M._loc(p), a, tuple(p.shape), kind, 0.02, n, 7)       # the head: NON-zero (so 'kept as-is' is a real check)


def micros(step):
    dcfg = CFG["data"]
    if not hasattr(micros, "c"):
        LW, LOCAL, RANK, WORLD = T.LW, T.LOCAL, T.RANK, T.WORLD
        pnode = D.PS.load_node(CFG["pose_root"])
        pplan = D.PS.plan(pnode, CFG["pose_seed"], LW, LOCAL, 1, LW, LOCAL, 1)
        snode = D.R3.load_node(CFG["speech_root"]); sidx = D.DS.load_index(CFG["speech_root"])
        splan = D.speech_plan(snode, sidx, CFG["speech_seed"], LW, LOCAL, CFG["speech_eval_frac"], CFG["speech_eval_salt"], 1)
        micros.c = {"pose": D.PoseDS(pnode, pplan["rows"][:16], dcfg, RANK), "speech": D.SpeechDS(CFG["speech_root"], splan["rows"][:32], dcfg, RANK),
                    "text": D.TextDS(CFG["text_root"], "train", [j * WORLD + RANK for j in range(16)], dcfg)}
    ks = {"pose": 0, "speech": 0, "text": 0}
    out = []
    for kind in T.SEQ:
        out.append((kind, micros.c[kind][step * T.SEQ.count(kind) + ks[kind]])); ks[kind] += 1
    return out


def ldig(t):
    return hashlib.sha256((t.to_local() if hasattr(t, "to_local") else t).detach().float().contiguous().cpu().numpy().tobytes()).hexdigest()


def gdig(t):
    allh = [None] * T.WORLD
    dist.all_gather_object(allh, ldig(t))
    return hashlib.sha256("".join(allh).encode()).hexdigest()


def head_local(model):
    return {n: M._loc(p).detach().clone() for n, p in model.named_parameters() if n in HK}


def one_step(model, rot, opt, sched, dos, step, dev, rec, dump=None):
    """= train_ppm.main's optimizer step (micros, per-group clip, AdamW, sched); records everything for the comparisons."""
    if CFG["pose_head"]:                                              # = main(): D35 backbone-grad scale for this step (env FROM 0 RAMP 0: s = 0)
        model.pose_head_bbgrad = T.posehead_bbgrad(step)
    if CFG["decp_bb"]:                                                # = main(): D33 link scale
        model.decp_bb_alpha = T.decp_bb_alpha(step)
    st = {"step": step + 1, "groups": [pg["name"] for pg in opt.param_groups], "lr_used": {pg["name"]: pg["lr"] for pg in opt.param_groups}, "micros": [],
          "bbgrad_s": getattr(model, "pose_head_bbgrad", "absent")}
    for kind, mb in micros(step):
        v, loss = T.run_micro(model, T.prep(mb, dev), rot, dos, grad=True)
        sv = T.allsum(v, dev)
        st["micros"].append({"kind": kind, "v": v, "loss": loss, "sum": sv, "summ": T.summarize(kind, sv, "train")})
    if PH_OFF:                                                        # = main's per-step D36 assertion
        assert model.pose_head.weight.grad is None and model.pose_head.bias.grad is None, "D36: the OFF pose head got a gradient"
    st["head_grad_none"] = all(p.grad is None for n, p in model.named_parameters() if n in HK)
    main_ps = {id(p) for g in opt.param_groups for p in g["params"]}
    st["grad"] = {n: gdig(p.grad) for n, p in model.named_parameters() if p.grad is not None and id(p) in main_ps}
    if dump:
        torch.save({n: M._loc(p.grad).detach().cpu() for n, p in model.named_parameters() if p.grad is not None and id(p) in main_ps},
                   f"{dump}/grad-r{T.RANK}.pt")
    gn = {}
    for pg in opt.param_groups:
        g_ = torch.nn.utils.clip_grad_norm_(pg["params"], CFG["clip"][pg["name"]])
        gn[pg["name"]] = float(g_.full_tensor() if hasattr(g_, "full_tensor") else g_)
    st["gn"] = gn
    opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
    st["lr_next"] = {pg["name"]: lr_ for pg, lr_ in zip(opt.param_groups, sched.get_last_lr())}
    st["w_after"] = {n: gdig(p) for n, p in model.named_parameters()}
    if dump:
        torch.save({n: M._loc(p).detach().cpu() for n, p in model.named_parameters() if hasattr(p, "to_local") or T.RANK == 0}, f"{dump}/w-r{T.RANK}.pt")
    rec.setdefault("steps", []).append(st)
    log(f"step {step + 1}: groups {st['groups']} gn {gn} losses {[round(m['loss'], 6) for m in st['micros']]}")
    return st


def main():
    dev = torch.device("cuda", T.LOCAL)
    torch.cuda.set_device(dev)
    import datetime
    dist.init_process_group("nccl", device_id=dev, timeout=datetime.timedelta(seconds=1800))
    c32, c4, c06 = tiny()
    model, _ = M.build(c32, c4, c06, dev, log, partial=True, proj_r=CFG["proj_r"], tower_vocab=CFG["tower_vocab"], init=False,
                       dec_mode=CFG["dec_mode"], pose_head=True, decp_bb=bool(CFG["decp_bb"]))
    if PH_OFF:                                                        # = main()
        model.pose_head_on = False
    rand_init(model)
    rot = M.Rot(c32, dev)
    opt, sched, by = T.make_opt(model)
    dos = {"dec_s": T.DecOpt(model.dec_s, "dec_s"), "dec_p": T.DecOpt(model.dec_p, "dec_p")}
    n_main = sum(len(v) for v in by.values())
    hosts = [None] * T.WORLD; dist.all_gather_object(hosts, T.HOST)
    rec = {"phase": PHASE, "code": CODE, "ph_off": PH_OFF, "n_main_params": n_main, "n_params": len(list(model.parameters())),
           "groups_built": [pg["name"] for pg in opt.param_groups], "cfg_d36": {k: CFG.get(k) for k in ("pose_head_off", "pose_head_off_from_step")}}
    dump = None
    if PHASE in ("on", "on2", "old", "off"):
        dump = f"{ROOT}/dump-{PHASE}"
        if T.RANK == 0:
            os.makedirs(dump, exist_ok=True)
        dist.barrier()
    if PHASE == "a":
        assert not PH_OFF and rec["groups_built"][-2:] == ["pose_head", "decp_bb"], rec["groups_built"]
        for s in range(2):
            one_step(model, rot, opt, sched, dos, s, dev, rec)
        nst = sum(1 for n, p in model.named_parameters() if n in HK and p in opt.state)
        assert nst == 2, nst
        T.save(model, opt, sched, dos, 2, f"{ROOT}/dcp-2", hosts)
    elif PHASE in ("on", "on2", "old", "off"):
        start = T.load(model, opt, sched, dos, f"{ROOT}/dcp-2", hosts)
        assert start == 2
        sd2 = torch.load(f"{ROOT}/dcp-2/rank{T.RANK}.pt", map_location=dev)
        hl = head_local(model)
        rec["head_w_eq_ckpt"] = bool(all(torch.equal(hl[n], sd2["model"][n]) for n in HK))
        rec["head_in_opt_state"] = sum(1 for n, p in model.named_parameters() if n in HK and p in opt.state)
        rec["groups_after_load"] = [pg["name"] for pg in opt.param_groups]
        rec["sched_after_load"] = {"base_lrs": list(sched.base_lrs), "last_lr": list(sched._last_lr), "last_epoch": sched.last_epoch}
        rec["frozen"] = sorted(n for n, p in model.named_parameters() if n in HK and not any(p is q for g in opt.param_groups for q in g["params"]))
        st = one_step(model, rot, opt, sched, dos, 2, dev, rec, dump)
        hl2 = head_local(model)
        rec["head_w_unchanged_by_step"] = bool(all(torch.equal(hl[n], hl2[n]) for n in HK))
        rec["n_pose_stats"] = len(st["micros"][0]["v"])
        if PH_OFF:
            assert rec["head_w_eq_ckpt"] and rec["head_in_opt_state"] == 0 and rec["frozen"] == sorted(HK) and st["head_grad_none"], rec
            assert rec["head_w_unchanged_by_step"] and "pose_head" not in rec["groups_after_load"] and len(sched.base_lrs) == len(opt.param_groups)
            assert CFG["pose_head_off_from_step"] == 2, CFG["pose_head_off_from_step"]
            assert not any("pose_head" in k for m in st["micros"] for k in m["summ"]) and "pose_head" not in st["gn"] and "pose_head" not in st["lr_next"]
            T.save(model, opt, sched, dos, 3, f"{ROOT}/dcp-3", hosts)
    elif PHASE == "off2":
        start = T.load(model, opt, sched, dos, f"{ROOT}/dcp-3", hosts)
        sd2 = torch.load(f"{ROOT}/dcp-2/rank{T.RANK}.pt", map_location=dev)
        ts3 = json.load(open(f"{ROOT}/dcp-3/train_state.json"))
        hl = head_local(model)
        rec.update(start=start, head_w_eq_dcp2=bool(all(torch.equal(hl[n], sd2["model"][n]) for n in HK)), off_from=CFG["pose_head_off_from_step"],
                   ckpt3_cfg_off=ts3["cfg"].get("pose_head_off"), ckpt3_sched_groups=len(ts3["sched"]["base_lrs"]),
                   head_in_opt_state=sum(1 for n, p in model.named_parameters() if n in HK and p in opt.state))
        assert PH_OFF and start == 3 and rec["head_w_eq_dcp2"] and rec["off_from"] == 2 and rec["ckpt3_cfg_off"] == 1 and rec["head_in_opt_state"] == 0, rec
        assert rec["ckpt3_sched_groups"] == len(opt.param_groups) == 7, (rec, len(opt.param_groups))
        one_step(model, rot, opt, sched, dos, 3, dev, rec)
        assert bool(all(torch.equal(hl[n], head_local(model)[n]) for n in HK))
        T.save(model, opt, sched, dos, 4, f"{ROOT}/dcp-4", hosts)
    elif PHASE == "revive":
        start = T.load(model, opt, sched, dos, f"{ROOT}/dcp-4", hosts)
        sd2 = torch.load(f"{ROOT}/dcp-2/rank{T.RANK}.pt", map_location=dev)
        hl = head_local(model)
        rec.update(start=start, head_w_eq_dcp2=bool(all(torch.equal(hl[n], sd2["model"][n]) for n in HK)), from_step=CFG["pose_head_from_step"],
                   head_in_opt_state=sum(1 for n, p in model.named_parameters() if n in HK and p in opt.state),
                   groups_after_load=[pg["name"] for pg in opt.param_groups])
        assert not PH_OFF and start == 4 and rec["head_w_eq_dcp2"] and rec["from_step"] == 4 and rec["head_in_opt_state"] == 0, rec
        assert rec["groups_after_load"] == rec["groups_built"] and "pose_head" in rec["groups_after_load"]
        s5 = one_step(model, rot, opt, sched, dos, 4, dev, rec)
        s6 = one_step(model, rot, opt, sched, dos, 5, dev, rec)
        hl6 = head_local(model)
        lr = CFG["lr"]["pose_head"]; w = CFG["pose_head_warmup"]
        rec["head_lr"] = [s5["lr_used"]["pose_head"], s5["lr_next"]["pose_head"], s6["lr_next"]["pose_head"]]
        rec["head_moved"] = bool(any(not torch.equal(hl[n], hl6[n]) for n in HK))
        hk = [k for k in s5["micros"][0]["summ"] if "pose_head" in k]
        rec["head_stats_keys"] = hk; rec["head_ce_step5"] = s5["micros"][0]["summ"].get("train_pose/pose_head_cb0_ce")
        assert s5["lr_used"]["pose_head"] == 0.0 and abs(rec["head_lr"][1] - lr / w) < 1e-15 and abs(rec["head_lr"][2] - 2 * lr / w) < 1e-15, rec["head_lr"]
        assert rec["head_moved"] and len(hk) == 2 and math.isfinite(rec["head_ce_step5"]), rec
    if T.RANK == 0:
        json.dump(rec, open(f"{ROOT}/{PHASE}.json", "w"))
        log("RESULT", json.dumps({k: v for k, v in rec.items() if k != "steps"})[:3000])
    dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
