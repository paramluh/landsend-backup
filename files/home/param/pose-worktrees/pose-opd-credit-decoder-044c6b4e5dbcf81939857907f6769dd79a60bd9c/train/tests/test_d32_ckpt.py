"""D32 test 5: checkpoint save / resume / export with the pose head, on a TINY-width PPM (hidden 256: a full-width checkpoint does not fit a spare
node's /scratch), real micros of this node's stores, the real trainer functions (train_ppm.save / load / make_opt / run_micro, safety_ppm,
infer_ppm.load_weights, ckpt_local_export.py). torchrun, phases (each a fresh process; env PPM_POSE_HEAD as noted):
  a  (head OFF) fresh init, 2 optimizer steps, save dcp-2                         = a pre-D32 checkpoint
  b  (head ON)  resume dcp-2: ONLY pose_head.weight/.bias may be missing; head = 0, no AdamW state for it, LR group appended to the sched,
                pose_head_from_step = 2; steps 3, 4: head CE at step 3 = ln 2048, head LR 0 -> 2e-6 -> 4e-6 (warmup from its first step);
                save dcp-4; record the full digest (weights + 3 AdamWs) and the sched
  c  (head ON)  resume dcp-4: per-rank digests (load_local) + the full digest == b's, from_step = 2 read back, sched identical; step 5 LR 6e-6;
                consolidated export w-5 (with the head)
  infer (1 process) ckpt_local_export of dcp-5... (run by tests/test_d32_ckpt.sh) then infer_ppm.load_weights of w-5 and local-5 into a
                head-less tiny PPM: both load, equal to each other and to the trainer's weights (bf16)."""
import json
import math
import os
import sys

sys.path.insert(0, "/data/home/amartyavarma/ppm/code")
PHASE = sys.argv[1]
ROOT = "/scratch/amartyavarma/ppm/d32test/ck"
import numpy as np  # noqa: E402
import torch  # noqa: E402

if PHASE != "infer":
    import torch.distributed as dist  # noqa: E402
    import train_ppm as T  # noqa: E402
    M, D, CFG, SF = T.M, T.D, T.CFG, T.SF
else:
    import model_ppm as M  # noqa: E402
H2 = M.H2
LN = math.log(2048)


def tiny():
    c32, c4 = M.cfg32(1), M.cfg4b(1)
    for c in (c32, c4):
        c.hidden_size, c.intermediate_size, c.num_attention_heads, c.num_key_value_heads, c.head_dim = 256, 512, 2, 1, 128
    c06 = H2.qwen3_light_config({"num_hidden_layers": 1, "hidden_size": 256, "intermediate_size": 512, "num_attention_heads": 2,
                                 "num_key_value_heads": 1, "head_dim": 128})
    return c32, c4, c06


def log(*a):
    if PHASE == "infer" or T.RANK == 0:
        print(f"[ckpt-{PHASE}]", *a, flush=True)


@torch.no_grad()
def rand_init(model):
    for n, p in model.named_parameters():
        n = n.replace("_checkpoint_wrapped_module.", "")
        kind = "zeros" if (n.startswith("pose_head.") or n in ("head.proj.D", "head.proj.U")) else \
            ("ones" if n.endswith("norm.weight") or n.endswith("layernorm.weight") else "normal")
        a, _ = M._rows(p)
        H2.init_rows_(M._loc(p), a, tuple(p.shape), kind, 0.02, n, 7)


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


def one_step(model, rot, opt, sched, dos, step, dev):
    """= train_ppm.main's optimizer step (micros, per-group clip, AdamW, sched). -> (head CE of the pose micro or None, group LRs used)."""
    used = [pg["lr"] for pg in opt.param_groups]
    hce = None
    for kind, mb in micros(step):
        v, _ = T.run_micro(model, T.prep(mb, dev), rot, dos, grad=True)
        if kind == "pose" and CFG["pose_head"]:
            s_ = T.allsum(v, dev)
            o_ = len(T.TK) + 10 + len(T.CB_S) + len(T.CB_P)
            hce = s_[o_] / s_[o_ + 1]
    for pg in opt.param_groups:
        torch.nn.utils.clip_grad_norm_(pg["params"], CFG["clip"][pg["name"]])
    opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
    return hce, used


def main():
    dev = torch.device("cuda", T.LOCAL)
    torch.cuda.set_device(dev)
    import datetime
    dist.init_process_group("nccl", device_id=dev, timeout=datetime.timedelta(seconds=1800))
    c32, c4, c06 = tiny()
    model, _ = M.build(c32, c4, c06, dev, log, partial=True, proj_r=CFG["proj_r"], tower_vocab=CFG["tower_vocab"], init=False,
                       dec_mode=CFG["dec_mode"], pose_head=bool(CFG["pose_head"]))
    rand_init(model)
    rot = M.Rot(c32, dev)
    opt, sched, _ = T.make_opt(model)
    dos = {"dec_s": T.DecOpt(model.dec_s, "dec_s"), "dec_p": T.DecOpt(model.dec_p, "dec_p")}
    hosts = [None] * T.WORLD; dist.all_gather_object(hosts, T.HOST)
    gi = [i for i, pg in enumerate(opt.param_groups) if pg["name"] == "pose_head"]
    res = {}
    if PHASE == "a":
        assert not CFG["pose_head"] and not gi
        for s in range(2):
            one_step(model, rot, opt, sched, dos, s, dev)
        T.save(model, opt, sched, dos, 2, f"{ROOT}/dcp-2", hosts)
        log("saved dcp-2 (no head); groups", [pg["name"] for pg in opt.param_groups])
    elif PHASE == "b":
        assert CFG["pose_head"] and gi == [len(opt.param_groups) - 1]
        start = T.load(model, opt, sched, dos, f"{ROOT}/dcp-2", hosts)
        hw, hb = model.pose_head.weight, model.pose_head.bias
        z = float(hw.full_tensor().abs().max()) + float(hb.full_tensor().abs().max())
        nst = sum(1 for p in (hw, hb) if p in opt.state)
        res.update(start=start, head_absmax_after_load=z, head_adam_states=nst, from_step=CFG["pose_head_from_step"], base_lrs=list(sched.base_lrs),
                   n_groups=len(opt.param_groups))
        assert start == 2 and z == 0.0 and nst == 0 and CFG["pose_head_from_step"] == 2 and len(sched.base_lrs) == len(opt.param_groups)
        assert sched.base_lrs[-1] == CFG["lr"]["pose_head"]
        steps = []
        for s in (2, 3):
            hce, used = one_step(model, rot, opt, sched, dos, s, dev)
            steps.append({"step": s + 1, "head_ce": hce, "head_lr_used": used[-1], "head_lr_next": sched.get_last_lr()[-1], "trunk_lr_used": used[0]})
        res["steps"] = steps
        assert abs(steps[0]["head_ce"] - LN) < 1e-5, steps[0]
        assert steps[0]["head_lr_used"] == 0.0 and abs(steps[0]["head_lr_next"] - 2e-6) < 1e-15 and abs(steps[1]["head_lr_next"] - 4e-6) < 1e-15, steps
        assert float(hw.full_tensor().abs().max()) > 0, "head weights moved after its first non-zero LR step"
        T.save(model, opt, sched, dos, 4, f"{ROOT}/dcp-4", hosts)
        dl = [None] * T.WORLD; dist.all_gather_object(dl, T.digest(model, [opt, dos["dec_s"].opt, dos["dec_p"].opt]))
        res["digest"] = dl                                         # every rank's digest of its local state (weights + 3 AdamWs)
        res["sched"] = {k: v for k, v in sched.state_dict().items() if k != "lr_lambdas"}
        res["head_adam_states_at_save"] = sum(1 for p in (hw, hb) if p in opt.state)
    elif PHASE == "c":
        start = T.load(model, opt, sched, dos, f"{ROOT}/dcp-4", hosts)
        b = json.load(open(f"{ROOT}/b.json"))
        dg = [None] * T.WORLD; dist.all_gather_object(dg, T.digest(model, [opt, dos["dec_s"].opt, dos["dec_p"].opt]))
        sd = {k: v for k, v in sched.state_dict().items() if k != "lr_lambdas"}
        res.update(start=start, digest_equal=dg == b["digest"], sched_equal=json.dumps(sd, sort_keys=True) == json.dumps(b["sched"], sort_keys=True),
                   from_step=CFG["pose_head_from_step"])
        assert start == 4 and dg == b["digest"] and res["sched_equal"] and CFG["pose_head_from_step"] == 2, res
        hce, used = one_step(model, rot, opt, sched, dos, 4, dev)
        res["step5"] = {"head_ce": hce, "head_lr_used": used[-1], "head_lr_next": sched.get_last_lr()[-1]}
        assert abs(sched.get_last_lr()[-1] - 6e-6) < 1e-15
        T.save(model, opt, sched, dos, 5, f"{ROOT}/dcp-5", hosts)
        SF.export_weights(model, 5, ROOT, "export", 1, "all", T.RANK, log)
        sd_ = {n: M._loc(p).detach().to(torch.bfloat16).cpu() for n, p in model.named_parameters() if not hasattr(p, "to_local")}
        full = {}
        for n, p in model.named_parameters():                      # the trainer's weights (bf16) for the infer comparison
            t_ = p.full_tensor() if hasattr(p, "full_tensor") else p
            full[n] = t_.detach().to(torch.bfloat16).cpu()
        if T.RANK == 0:
            torch.save(full, f"{ROOT}/trainer_w5_bf16.pt")
        del sd_
    if T.RANK == 0:
        json.dump(res, open(f"{ROOT}/{PHASE}.json", "w"))
        log("RESULT", json.dumps(res)[:3000])
    dist.barrier(); dist.destroy_process_group()


def infer():
    sys.path.insert(0, "/data/home/amartyavarma/ppm/code")
    import infer_ppm as I
    c32, c4, c06 = tiny()
    ref = torch.load(f"{ROOT}/trainer_w5_bf16.pt")
    assert "pose_head.weight" in ref
    out = {}
    for path in (f"{ROOT}/export/w-5", f"{ROOT}/local-5"):
        ci = I.ckpt_info(path)
        torch.set_default_dtype(torch.bfloat16)
        with torch.device("meta"):
            m = M.PPM(c32, c4, c06, 2048, M.TOWER_VOCAB)              # the inference model: no pose_head
        torch.set_default_dtype(torch.float32)
        m.to_empty(device="cuda:0")
        if ci["kind"] == "dcp":
            assert "pose_head.weight" in ci["shapes"]
        I.load_weights(m, ci)
        got = {n: p.detach().cpu() for n, p in m.named_parameters()}
        assert not any(n.startswith("pose_head") for n in got)
        bad = [n for n in got if not torch.equal(got[n], ref[n])]
        out[os.path.basename(path)] = {"tensors": len(got), "differ_from_trainer": bad[:4]}
        assert not bad, (path, bad[:4])
    print("[ckpt-infer] PASS", out, flush=True)


if __name__ == "__main__":
    infer() if PHASE == "infer" else main()
