"""D33 test 6: checkpoint save / resume / export with the decp_bb link, on a TINY-width PPM (hidden 256, as tests/test_d32_ckpt.py), the
pre-dumped real micros (tests/dump_d33_micros.py), the real trainer functions (train_ppm.save / load / make_opt / run_micro, safety_ppm,
ckpt_local_export.py, infer_ppm). Production config otherwise (D32 pose head ON). torchrun phases (each a fresh process):
  a    (link OFF) fresh init, 2 optimizer steps, save dcp-2                          = a pre-D33 checkpoint
  b    (link ON)  resume dcp-2: ONLY decp_bb.weight/.bias may be missing; link = 0, no AdamW state, LR group appended, decp_bb_from_step = 2;
                  steps 3, 4: link LR 0 -> 2e-6 -> 4e-6 (warmup from its first step); trunk LR at step 3 = the schedule's (resume LR fix);
                  save dcp-4; full digest (weights + 3 AdamWs) + sched
  c    (link ON, PPM_DECP_BB_ALPHA=0.5) resume dcp-4: digests == b's, from_step 2 read back, sched identical, alpha change logged; step 5
                  (link LR used 4e-6); save dcp-5; consolidated export w-5 (+ train_cfg.json alpha 0.5)
  neg1 (link OFF) resume dcp-5 must be REFUSED (the checkpoint has the link)
  neg2 (link ON, PPM_LR_DECP_BB changed) resume dcp-5 must be REFUSED (only alpha / fade are resume-changeable)
  infer (1 process, after ckpt_local_export of dcp-5): infer_ppm.ckpt_info of w-5 and local-5 -> decp_bb present, alpha 0.5; load_weights
                  into a tiny PPM with the link: equal to the trainer's weights (bf16); infer's AR pose decoder (DecoderAR with zb) gives the
                  same cb1 logits as the training decoder's full pass (Decoder.inputs with zb) on the same inputs."""
import json
import math
import os
import pickle
import sys

sys.path.insert(0, "/data/home/amartyavarma/ppm/code")
PHASE = sys.argv[1]
ROOT = "/scratch/amartyavarma/ppm/d33test/ck"
MICROS = os.environ.get("D33_MICROS", "/data/home/amartyavarma/ppm/teststore/d33/micros")
import torch  # noqa: E402

if PHASE != "infer":
    import torch.distributed as dist  # noqa: E402
    import train_ppm as T  # noqa: E402
    M, D, CFG, SF = T.M, T.D, T.CFG, T.SF
else:
    import model_ppm as M  # noqa: E402
H2 = M.H2


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
        kind = "zeros" if (n.startswith(("pose_head.", "decp_bb.")) or n in ("head.proj.D", "head.proj.U")) else \
            ("ones" if n.endswith("norm.weight") or n.endswith("layernorm.weight") else "normal")
        a, _ = M._rows(p)
        H2.init_rows_(M._loc(p), a, tuple(p.shape), kind, 0.02, n, 7)


REC = {}


def micros(step):
    if not REC:
        REC.update(pickle.load(open(f"{MICROS}/rank{T.RANK}.pkl", "rb")))
        assert REC["dcfg"] == CFG["data"]
    ks = {"pose": 0, "speech": 0, "text": 0}
    out = []
    for kind in T.SEQ:
        out.append((kind, REC[kind][step * T.SEQ.count(kind) + ks[kind]])); ks[kind] += 1
    return out


def one_step(model, rot, opt, sched, dos, step, dev):
    """= train_ppm.main's optimizer step. -> {group: LR used}."""
    if CFG["decp_bb"]:
        model.decp_bb_alpha = T.decp_bb_alpha(step)
    used = {pg["name"]: pg["lr"] for pg in opt.param_groups}
    for kind, mb in micros(step):
        T.run_micro(model, T.prep(mb, dev), rot, dos, grad=True)
    for pg in opt.param_groups:
        torch.nn.utils.clip_grad_norm_(pg["params"], CFG["clip"][pg["name"]])
    opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
    return used


def close(a, b):
    return abs(a - b) <= 1e-15 + 1e-9 * abs(b)


def main():
    dev = torch.device("cuda", T.LOCAL)
    torch.cuda.set_device(dev)
    import datetime
    dist.init_process_group("nccl", device_id=dev, timeout=datetime.timedelta(seconds=1800))
    c32, c4, c06 = tiny()
    model, _ = M.build(c32, c4, c06, dev, log, partial=True, proj_r=CFG["proj_r"], tower_vocab=CFG["tower_vocab"], init=False,
                       dec_mode=CFG["dec_mode"], pose_head=bool(CFG["pose_head"]), **({"decp_bb": True} if CFG["decp_bb"] else {}))
    rand_init(model)
    rot = M.Rot(c32, dev)
    opt, sched, _ = T.make_opt(model)
    dos = {"dec_s": T.DecOpt(model.dec_s, "dec_s"), "dec_p": T.DecOpt(model.dec_p, "dec_p")}
    hosts = [None] * T.WORLD; dist.all_gather_object(hosts, T.HOST)
    names = [pg["name"] for pg in opt.param_groups]
    res = {"groups": names}
    lrt = CFG["lr"]["trunk"]
    if PHASE == "a":
        assert not CFG["decp_bb"] and "decp_bb" not in names and CFG["pose_head"]
        for s in range(2):
            one_step(model, rot, opt, sched, dos, s, dev)
        T.save(model, opt, sched, dos, 2, f"{ROOT}/dcp-2", hosts)
    elif PHASE == "b":
        assert CFG["decp_bb"] and names[-1] == "decp_bb"
        start = T.load(model, opt, sched, dos, f"{ROOT}/dcp-2", hosts)
        w, bi = model.decp_bb.weight, model.decp_bb.bias
        z = float(w.full_tensor().abs().max()) + float(bi.full_tensor().abs().max())
        nst = sum(1 for p in (w, bi) if p in opt.state)
        res.update(start=start, link_absmax_after_load=z, link_adam_states=nst, from_step=CFG["decp_bb_from_step"], n_base_lrs=len(sched.base_lrs))
        assert start == 2 and z == 0.0 and nst == 0 and CFG["decp_bb_from_step"] == 2 and len(sched.base_lrs) == len(names)
        assert sched.base_lrs[-1] == CFG["lr"]["decp_bb"]
        steps = []
        for s in (2, 3):
            used = one_step(model, rot, opt, sched, dos, s, dev)
            steps.append({"step": s + 1, "link_lr_used": used["decp_bb"], "trunk_lr_used": used["trunk"], "head_lr_used": used["pose_head"],
                          "link_lr_next": sched.get_last_lr()[-1], "link_wnorm": float(w.full_tensor().norm())})
        res["steps"] = steps
        assert steps[0]["link_lr_used"] == 0.0 and close(steps[1]["link_lr_used"], 2e-6) and close(steps[1]["link_lr_next"], 4e-6), steps
        assert close(steps[0]["trunk_lr_used"], lrt * 2 / 100) and close(steps[1]["trunk_lr_used"], lrt * 3 / 100), ("resume LR fix", steps)
        assert steps[0]["link_wnorm"] == 0.0 and steps[1]["link_wnorm"] > 0, "link weights move after its first non-zero LR step"
        T.save(model, opt, sched, dos, 4, f"{ROOT}/dcp-4", hosts)
        dl = [None] * T.WORLD; dist.all_gather_object(dl, T.digest(model, [opt, dos["dec_s"].opt, dos["dec_p"].opt]))
        res["digest"] = dl
        res["sched"] = {k: v for k, v in sched.state_dict().items() if k != "lr_lambdas"}
        res["link_adam_states_at_save"] = sum(1 for p in (w, bi) if p in opt.state)
    elif PHASE == "c":
        assert CFG["decp_bb_alpha"] == 0.5
        start = T.load(model, opt, sched, dos, f"{ROOT}/dcp-4", hosts)
        b = json.load(open(f"{ROOT}/b.json"))
        dg = [None] * T.WORLD; dist.all_gather_object(dg, T.digest(model, [opt, dos["dec_s"].opt, dos["dec_p"].opt]))
        sd = {k: v for k, v in sched.state_dict().items() if k != "lr_lambdas"}
        res.update(start=start, digest_equal=dg == b["digest"], sched_equal=json.dumps(sd, sort_keys=True) == json.dumps(b["sched"], sort_keys=True),
                   from_step=CFG["decp_bb_from_step"])
        assert start == 4 and dg == b["digest"] and res["sched_equal"] and CFG["decp_bb_from_step"] == 2, res
        used = one_step(model, rot, opt, sched, dos, 4, dev)
        res["step5"] = {"link_lr_used": used["decp_bb"], "trunk_lr_used": used["trunk"], "alpha": model.decp_bb_alpha}
        assert close(used["decp_bb"], 4e-6) and close(used["trunk"], lrt * 4 / 100) and model.decp_bb_alpha == 0.5, res["step5"]
        T.save(model, opt, sched, dos, 5, f"{ROOT}/dcp-5", hosts)
        SF.export_weights(model, 5, ROOT, "export", 1, "all", T.RANK, log, side={"step": 5, "decp_bb": 1, "decp_bb_alpha": T.decp_bb_alpha(5)})
        full = {}
        for n, p in model.named_parameters():
            t_ = p.full_tensor() if hasattr(p, "full_tensor") else p
            full[n] = t_.detach().to(torch.bfloat16).cpu()
        if T.RANK == 0:
            torch.save(full, f"{ROOT}/trainer_w5_bf16.pt")
    elif PHASE in ("neg1", "neg2"):
        try:
            T.load(model, opt, sched, dos, f"{ROOT}/dcp-5", hosts)
            ok, msg = False, "resume was ACCEPTED"
        except AssertionError as e:
            ok, msg = True, str(e)[:300]
        res.update(refused=ok, msg=msg)
        assert ok, msg
    if T.RANK == 0:
        json.dump(res, open(f"{ROOT}/{PHASE}.json", "w"))
        log("RESULT", json.dumps(res)[:3000])
    dist.barrier(); dist.destroy_process_group()


def infer():
    import infer_ppm as I
    c32, c4, c06 = tiny()
    ref = torch.load(f"{ROOT}/trainer_w5_bf16.pt")
    assert "decp_bb.weight" in ref and float(ref["decp_bb.weight"].float().norm()) > 0
    out = {}
    for path in (f"{ROOT}/export/w-5", f"{ROOT}/local-5"):
        ci = I.ckpt_info(path)
        assert ci["decp_bb"] and ci["decp_bb_alpha"] == 0.5, (path, ci.get("decp_bb"), ci.get("decp_bb_alpha"))
        torch.set_default_dtype(torch.bfloat16)
        with torch.device("meta"):
            m = M.PPM(c32, c4, c06, 2048, M.TOWER_VOCAB, decp_bb=True)
        torch.set_default_dtype(torch.float32)
        m.to_empty(device="cuda:0")
        m.dec_p.model.rotary_emb = type(m.dec_p.model.rotary_emb)(config=c06, device="cuda:0")
        I.load_weights(m, ci)
        m.decp_bb_alpha = ci["decp_bb_alpha"]
        got = {n: p.detach().cpu() for n, p in m.named_parameters()}
        bad = [n for n in got if not torch.equal(got[n], ref[n])]
        assert not bad, (path, bad[:4])
        # inference pose decoder with the link == the training decoder's formula (cb1 logits of one frame)
        with torch.no_grad():
            g = torch.Generator(device="cuda:0").manual_seed(5)
            hv = torch.randn(1, c4.hidden_size, device="cuda:0", generator=g).to(torch.bfloat16)
            hprev = torch.randn(1, c32.hidden_size, device="cuda:0", generator=g).to(torch.bfloat16)
            cb0 = torch.tensor([17], device="cuda:0")
            zb = m.decp_bb(hprev) * float(m.decp_bb_alpha)
            dd = I.DecoderAR(m.dec_p, c06, "cuda:0")
            x = torch.cat([dd.d.proj(hv)[:, None] + zb[:, None], dd.d.cb0_emb(cb0)[:, None]], 1)
            hl_ar = dd._run(x, 0)[:, -1]
            z_ar = (hl_ar @ dd.W[0].t()).float()
            depth = torch.zeros(1, M.DEPTH, dtype=torch.long, device="cuda:0") + torch.arange(M.DEPTH, device="cuda:0") * M.CBS
            depth[0, 0] = 17
            xf, _ = m.dec_p.inputs(hv, depth, torch.tensor([0], device="cuda:0"), zb)
            hl = m.dec_p.model(inputs_embeds=xf[:, :2].to(torch.bfloat16), use_cache=False).last_hidden_state[:, 1]
            z_full = (hl @ dd.W[0].t()).float()
            x0, _ = m.dec_p.inputs(hv, depth, torch.tensor([0], device="cuda:0"))
            hl0 = m.dec_p.model(inputs_embeds=x0[:, :2].to(torch.bfloat16), use_cache=False).last_hidden_state[:, 1]
            z_nolink = (hl0 @ dd.W[0].t()).float()
            dmax, dlink = float((z_ar - z_full).abs().max()), float((z_full - z_nolink).abs().max())
            # with the link: AR == full pass (bf16); the link visibly changes the logits
            assert dmax <= max(0.02 * float(z_full.abs().max()), 1e-3) and dlink > 10 * dmax, (dmax, dlink)
        out[os.path.basename(path)] = {"tensors": len(got), "alpha": ci["decp_bb_alpha"], "ar_vs_full_cb1_logit_maxdiff": dmax, "link_effect_cb1_logit_maxdiff": dlink}
        del m
    print("[ckpt-infer] PASS", json.dumps(out), flush=True)


if __name__ == "__main__":
    infer() if PHASE == "infer" else main()
