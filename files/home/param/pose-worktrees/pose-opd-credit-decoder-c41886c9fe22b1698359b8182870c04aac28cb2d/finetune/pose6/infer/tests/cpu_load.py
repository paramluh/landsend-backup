#!/usr/bin/env python3
"""pose6 A1 CPU checks (no GPU): the engine's LOAD PATH for checkpoints with the new keys, the real checkpoints' key sets, the inference
SLOT layout and the start()-side G0COND input checks. Results -> tests/out/cpu_load.json.

  L1  synthetic checkpoints (tiny widths: model_ppm cfg32 / cfg4b replaced by narrow configs IN THIS TEST ONLY; every loader line runs as
      deployed) in BOTH formats the engine reads, written to node /scratch (never /data):
        dcp    = the 8,1,1 trainer's export_weights format (dcp.save of a bf16 model state dict + train_cfg.json side + COMPLETE; + an
                 unused pose_head key as the 811 exports carry), side {decp_bb, decp_bb_alpha, s2p: cb0, g0cond: 1};
        local  = ppm-local-bf16-shards-v1 (dim-0 shards rank<r>.pt in rank order, decoders in rep.pt, meta.json cfg with s2p / g0cond).
      build_model(path, [cpu]) -> every tensor bit-equal to what was saved (incl. tower_p.s_emb.weight and inp.g0.weight), binfo flags;
      negative cases: PPM_G0COND=0 with the key -> AssertionError; side g0cond 1 without the key -> AssertionError; no key + env unset ->
      no module; no key + PPM_G0COND=1 -> zero (binfo g0cond_zero_init).
  L2  real key sets: the 8,1,1 export ckpt-weka/ppm-v1/w-15973 (.metadata only) and export-357 (rank0 / rep keys) vs the engine model with
      s2p cb0 + G0COND (meta); the 31101 trainer's model (pose5/code-811-s2p, PPM_S2P=cb0 + PPM_G0COND=1, meta) vs w-15973: the new export
      keys must be exactly tower_p.s_emb.weight [2048, 2560] and inp.g0.weight [5120, 8].
  S   inference SLOT: infer.system_block for the 4 prompt types (empty / voice clip / persona / both) + assemble -> what sits at P - 1;
      FastPPM._g0_inst (start()'s check) on each, plus its fail-loud cases."""
import importlib.util
import json
import os
import sys
import time
import types

import numpy as np

BR = "/data/home/amartyavarma/ppm/pose6/infer"
OUT = f"{BR}/tests/out"
SCR = os.environ.get("A1_SCRATCH", f"/scratch/amartyavarma/pose6-a1/{os.environ.get('SLURM_JOB_ID', 'x')}")
for k in ("PPM_S2P", "PPM_G0COND"):
    os.environ.pop(k, None)
sys.path.insert(0, f"{BR}/ppm")
sys.path.insert(1, BR)
import torch  # noqa: E402
import fast_ppm as FB  # noqa: E402
import infer as MIF  # noqa: E402

M = FB.M
RES = {"t_start": time.strftime("%Y-%m-%d %H:%M:%S"), "host": os.uname().nodename}
T0 = time.time()


def log(*a):
    print(f"[a1-cpu {time.time() - T0:7.1f}s]", *a, flush=True)


def save():
    json.dump(RES, open(f"{OUT}/cpu_load.json", "w"), indent=1, default=str)


def raises(fn, exc):
    try:
        fn()
    except exc as e:
        return f"{type(e).__name__}: {str(e)[:160]}"
    return None


# ================================================================================================ L1 synthetic checkpoints
_c32, _c4 = M.cfg32, M.cfg4b


def t32(layers=None):
    c = _c32(layers)
    c.hidden_size, c.intermediate_size, c.num_attention_heads, c.num_key_value_heads, c.head_dim = 128, 256, 4, 2, 32
    return c


def t4(layers=None):
    c = _c4(layers)
    c.hidden_size, c.intermediate_size, c.num_attention_heads, c.num_key_value_heads, c.head_dim = 64, 128, 4, 2, 16
    return c


def synth_sd():
    """a tiny PPM with every part the 31101 exports have (pose, D33 link, s_emb cb0, inp.g0) + seeded random bf16 weights."""
    c32, c4, c06 = t32(2), t4(2), M.cfg06(1)
    torch.set_default_dtype(FB.BF)
    m = M.PPM(c32, c4, c06, 64, M.TOWER_VOCAB, decp_bb=True)
    m.tower_p.s_emb = FB.S2PEmb(c4.hidden_size, "cb0")
    m.inp.g0 = torch.nn.Linear(FB.G0_DIM, c32.hidden_size, bias=False)
    torch.set_default_dtype(torch.float32)
    g = torch.Generator().manual_seed(20261027)
    sd = {}
    for n, p in m.state_dict().items():
        sd[n] = (torch.randn(p.shape, generator=g) * 0.02).to(FB.BF) if p.is_floating_point() else p.clone()
    return sd


def write_dcp(sd, path, side, extra=True):
    import torch.distributed.checkpoint as dcp
    os.makedirs(path, exist_ok=True)
    sd2 = dict(sd)
    if extra:
        sd2["pose_head.lin.weight"] = torch.zeros(4, 4, dtype=FB.BF)            # the 811 exports carry the (unused) D32 pose head
    dcp.save(sd2, checkpoint_id=path, no_dist=True)
    json.dump(side, open(f"{path}/train_cfg.json", "w"))
    open(f"{path}/COMPLETE", "w").write(str(side["step"]))


def write_local(sd, path, cfg, world=4, step=77):
    os.makedirs(path, exist_ok=True)
    rep = {k: v for k, v in sd.items() if k.startswith(("dec_s.", "dec_p."))}
    sh = [dict() for _ in range(world)]
    for k, v in sd.items():
        if k in rep:
            continue
        for r, part in enumerate(torch.tensor_split(v, world, 0)):              # FSDP2 dim-0 shards in rank order (trailing ones may be empty)
            sh[r][k] = part.clone()
    for r in range(world):
        torch.save(sh[r], f"{path}/rank{r}.pt")
    torch.save(rep, f"{path}/rep.pt")
    json.dump({"format": "ppm-local-bf16-shards-v1", "step": step, "world": world, "cfg": cfg}, open(f"{path}/meta.json", "w"))
    open(f"{path}/COMPLETE", "w").write(str(step))


def check_loaded(m, sd):
    sdm = m.state_dict()
    bad = [k for k, v in sd.items() if k not in sdm or not torch.equal(sdm[k].to(v.dtype), v)]
    miss = sorted(set(sdm) - set(sd))
    return bad, miss


def L1():
    M.cfg32, M.cfg4b = t32, t4
    try:
        sd = synth_sd()
        r = {"n_tensors": len(sd), "g0_shape": list(sd[FB.G0_KEY].shape), "s_emb_shape": list(sd[FB.S2P_KEY].shape)}
        cpu = [torch.device("cpu")]
        side = {"step": 123, "decp_bb": 1, "decp_bb_alpha": 0.5, "s2p": "cb0", "g0cond": 1}
        pd = f"{SCR}/synth-dcp/w-123"
        write_dcp(sd, pd, side)
        cfg = {"layers": 2, "tower_layers": 2, "dec_layers": 1, "proj_r": 64, "tower_vocab": M.TOWER_VOCAB, "decp_bb": 1, "decp_bb_alpha": 0.5,
               "s2p": "cb0", "g0cond": 1}
        pl = f"{SCR}/synth-local/export-77"
        write_local(sd, pl, cfg)
        for name, path in (("dcp", pd), ("local", pl)):
            m, ldev, cfgs, info = FB.build_model(path, cpu)
            bad, miss = check_loaded(m, sd)
            rr = {"binfo": {k: info[k] for k in ("g0cond", "g0cond_zero_init", "s2p", "s2p_zero_init", "decp_bb", "decp_bb_alpha", "step")},
                  "has_inp_g0": hasattr(m.inp, "g0"), "g0_bias_none": getattr(getattr(m.inp, "g0", None), "bias", 1) is None,
                  "tensors_not_equal": bad, "model_keys_not_in_ckpt": miss,
                  "g0_equal": bool(torch.equal(m.inp.g0.weight, sd[FB.G0_KEY])), "s_emb_equal": bool(torch.equal(m.tower_p.s_emb.weight, sd[FB.S2P_KEY]))}
            rr["pass"] = (not bad and not miss and rr["has_inp_g0"] and rr["g0_bias_none"] and rr["g0_equal"] and rr["s_emb_equal"]
                          and info["g0cond"] is True and not info["g0cond_zero_init"] and info["s2p"] == "cb0" and not info["s2p_zero_init"]
                          and abs(info["decp_bb_alpha"] - 0.5) < 1e-12)
            r[f"load_{name}"] = rr
            log(f"L1 {name}: {json.dumps(rr)}")
            del m
        # negative / flag cases
        os.environ["PPM_G0COND"] = "0"
        r["env0_with_key_dcp"] = raises(lambda: FB.build_model(pd, cpu), AssertionError)
        r["env0_with_key_local"] = raises(lambda: FB.build_model(pl, cpu), AssertionError)
        os.environ.pop("PPM_G0COND")
        sd_no = {k: v for k, v in sd.items() if k not in (FB.G0_KEY, FB.S2P_KEY)}
        pn = f"{SCR}/synth-dcp-nog0/w-124"
        write_dcp(sd_no, pn, {"step": 124, "decp_bb": 1, "decp_bb_alpha": 0.5})
        pbad = f"{SCR}/synth-dcp-badside/w-125"
        write_dcp(sd_no, pbad, {"step": 125, "decp_bb": 1, "decp_bb_alpha": 0.5, "g0cond": 1})
        r["side_g0cond_without_key"] = raises(lambda: FB.ckpt_info(pbad), AssertionError)
        cl = dict(cfg); cl.pop("g0cond"); cl.pop("s2p")
        pln = f"{SCR}/synth-local-nog0/export-78"
        write_local(sd_no, pln, cl, step=78)
        for name, path in (("dcp", pn), ("local", pln)):
            m, _, _, info = FB.build_model(path, cpu)
            bad, miss = check_loaded(m, sd_no)
            r[f"nokey_envunset_{name}"] = {"has_inp_g0": hasattr(m.inp, "g0"), "has_s_emb": hasattr(m.tower_p, "s_emb"), "g0cond": info["g0cond"],
                                           "s2p": info["s2p"], "tensors_not_equal": bad, "model_keys_not_in_ckpt": miss}
            del m
            os.environ["PPM_G0COND"] = "1"
            m, _, _, info = FB.build_model(path, cpu)
            os.environ.pop("PPM_G0COND")
            bad, miss = check_loaded(m, sd_no)
            r[f"nokey_env1_{name}"] = {"has_inp_g0": hasattr(m.inp, "g0"), "g0cond": info["g0cond"], "g0cond_zero_init": info["g0cond_zero_init"],
                                       "g0_all_zero": bool((m.inp.g0.weight == 0).all()), "tensors_not_equal": bad, "model_keys_not_in_ckpt": miss}
            del m
        r["pass"] = (r["load_dcp"]["pass"] and r["load_local"]["pass"] and bool(r["env0_with_key_dcp"]) and bool(r["env0_with_key_local"])
                     and bool(r["side_g0cond_without_key"])
                     and all(not r[f"nokey_envunset_{n}"]["has_inp_g0"] and not r[f"nokey_envunset_{n}"]["tensors_not_equal"]
                             and not r[f"nokey_envunset_{n}"]["model_keys_not_in_ckpt"] for n in ("dcp", "local"))
                     and all(r[f"nokey_env1_{n}"]["g0cond_zero_init"] and r[f"nokey_env1_{n}"]["g0_all_zero"]
                             and not r[f"nokey_env1_{n}"]["tensors_not_equal"] and r[f"nokey_env1_{n}"]["model_keys_not_in_ckpt"] == [FB.G0_KEY]
                             for n in ("dcp", "local")))
        RES["L1"] = r
        log(f"L1 pass {r['pass']}")
    finally:
        M.cfg32, M.cfg4b = _c32, _c4
    save()



# ================================================================================================ LK D33 link OFF (31101: PPM_DECP_BB_OFF=1)
def LK():
    """the 31101 export side records decp_bb_alpha 0 + decp_bb_off 1; local copies of its saves (and an FT run with the link off) carry the
    training cfg (decp_bb_alpha = the knob, e.g. 1.0, + decp_bb_off 1): the engine must use alpha 0 for all of them, and alpha 0 must make the
    pose decoder's slot 0 exactly proj(hv_p) (= the trainer's zb=None graph): no stale link, no NaN."""
    M.cfg32, M.cfg4b = t32, t4
    try:
        sd = synth_sd()
        cpu = [torch.device("cpu")]
        base = {"layers": 2, "tower_layers": 2, "dec_layers": 1, "proj_r": 64, "tower_vocab": M.TOWER_VOCAB, "decp_bb": 1, "s2p": "cb0", "g0cond": 1}
        cases = {
            "dcp_31101_side (alpha 0 + off 1)": ("dcp", {"decp_bb": 1, "decp_bb_alpha": 0.0, "decp_bb_off": 1, "s2p": "cb0", "g0cond": 1}, 0.0),
            "dcp_side_alpha1_off1 (FT export side as written today + off)": ("dcp", {"decp_bb": 1, "decp_bb_alpha": 1.0, "decp_bb_off": 1, "s2p": "cb0", "g0cond": 1}, 0.0),
            "dcp_side_alpha0.7_no_off (unchanged path)": ("dcp", {"decp_bb": 1, "decp_bb_alpha": 0.7, "s2p": "cb0", "g0cond": 1}, 0.7),
            "local_cfg_alpha1_off1 (local copy of a 31101 save / FT local export, link off)": ("local", dict(base, decp_bb_alpha=1.0, decp_bb_off=1), 0.0),
            "local_cfg_alpha1_no_off (unchanged path)": ("local", dict(base, decp_bb_alpha=1.0), 1.0),
        }
        r = {}
        m0 = None
        for i, (name, (fmt, rec, want)) in enumerate(cases.items()):
            path = f"{SCR}/link/{i}"
            if fmt == "dcp":
                write_dcp(sd, f"{path}/w-{130 + i}", dict(rec, step=130 + i))
                path = f"{path}/w-{130 + i}"
            else:
                write_local(sd, f"{path}/export-{130 + i}", rec, step=130 + i)
                path = f"{path}/export-{130 + i}"
            m, _, _, info = FB.build_model(path, cpu)
            r[name] = {"alpha": info["decp_bb_alpha"], "model_alpha": float(m.decp_bb_alpha), "want": want,
                       "pass": info["decp_bb_alpha"] == want and float(m.decp_bb_alpha) == want}
            log(f"LK {name}: {json.dumps(r[name])}")
            if i == 0:
                m0 = m
            else:
                del m
        # slot 0 with alpha 0: the engine's zb expression (FastPPM._pose) + FusedDec.slot0 vs slot0 without a link (= the trainer's zb=None)
        g = torch.Generator().manual_seed(7)
        with torch.no_grad():
            hn = torch.randn(16, m0.decp_bb.in_features, generator=g).to(FB.BF) * 3
            hv = torch.randn(16, m0.dec_p.proj.in_features, generator=g).to(FB.BF)
            zb = m0.decp_bb(hn) * float(m0.decp_bb_alpha)                     # = fast_ppm FastPPM._pose's zb (alpha 0 here)
            fake = types.SimpleNamespace(dec=m0.dec_p)
            s_link = FB.FusedDec.slot0(fake, hv, zb)
            s_none = FB.FusedDec.slot0(fake, hv, None)
            s_tr = m0.dec_p.proj(hv).float()                                     # the trainer's slot 0 with zb=None (Decoder.inputs: proj(hv))
            z1 = m0.decp_bb(hn) * 1.0
        r["slot0"] = {"alpha": float(m0.decp_bb_alpha), "zb_all_zero": bool((zb == 0).all()), "zb_finite": bool(torch.isfinite(zb).all()),
                      "slot0_equal_no_link": bool(torch.equal(s_link, s_none)), "slot0_equal_trainer_proj": bool(torch.equal(s_link, s_tr)),
                      "slot0_bit_patterns_differing (signed zeros only)": int((s_link.view(torch.int32) != s_none.view(torch.int32)).sum()),
                      "slot0_finite": bool(torch.isfinite(s_link).all()), "link_alpha1_would_add_max_abs": float(z1.float().abs().max())}
        r["slot0"]["pass"] = (r["slot0"]["zb_all_zero"] and r["slot0"]["zb_finite"] and r["slot0"]["slot0_equal_no_link"]
                              and r["slot0"]["slot0_equal_trainer_proj"] and r["slot0"]["slot0_finite"])
        log(f"LK slot0: {json.dumps(r['slot0'])}")
        r["pass"] = all(v["pass"] for v in r.values())
        RES["LK"] = r
        log(f"LK pass {r['pass']}")
    finally:
        M.cfg32, M.cfg4b = _c32, _c4
    save()

# ================================================================================================ L2 real key sets
W15973 = "/data/home/amartyavarma/ppm/ckpt-weka/ppm-v1/w-15973"
E357 = "/data/home/amartyavarma/ppm/aloe-pose-mono-8020/exports/export-357"


def engine_meta_shapes(ci, s2p="cb0", g0=True):
    """= build_model's module tree on meta (pose / decp_bb / vadapter as the checkpoint) + the two new terms -> {name: shape}."""
    c32, c4, c06 = M.cfg32(ci["layers"]), M.cfg4b(ci["tower_layers"]), M.cfg06(ci["dec_layers"])
    with torch.device("meta"):
        m = M.PPM(c32, c4, c06, ci["proj_r"], ci["tower_vocab"], vadapter=FB.VADAPT if ci["vad"] else None, nopose=not ci["pose"],
                  decp_bb=ci["decp_bb"])
        if s2p:
            m.tower_p.s_emb = FB.S2PEmb(c4.hidden_size, s2p)
        if g0:
            m.inp.g0 = torch.nn.Linear(FB.G0_DIM, c32.hidden_size, bias=False)
    return {n: tuple(p.shape) for n, p in m.named_parameters()}


def L2():
    r = {}
    ci = FB.ckpt_info(W15973)
    want = engine_meta_shapes(ci)
    have = ci["shapes"]
    r["w15973"] = {"step": ci["step"], "side": ci["side"], "g0cond": ci["g0cond"], "s2p": ci["s2p"], "n_ckpt_keys": len(have),
                   "engine_keys_missing": sorted(set(want) - set(have)),
                   "shape_mismatch": [(k, want[k], have[k]) for k in want if k in have and tuple(have[k]) != want[k]],
                   "ckpt_extra_not_used": sorted({k.split(".")[0] for k in set(have) - set(want)})}
    log(f"L2 w-15973: {json.dumps(r['w15973'])}")
    ce = FB.ckpt_info(E357)
    want_e = engine_meta_shapes(ce)
    keys_e = set(torch.load(f"{E357}/rep.pt", map_location="cpu", weights_only=True, mmap=True).keys())
    keys_e |= set(torch.load(f"{E357}/rank0.pt", map_location="cpu", weights_only=True, mmap=True).keys())
    r["export357"] = {"step": ce["step"], "g0cond": ce["g0cond"], "s2p": ce["s2p"], "world": ce["meta"]["world"],
                      "engine_keys_missing": sorted(set(want_e) - keys_e), "ckpt_extra_not_used": sorted({k.split(".")[0] for k in keys_e - set(want_e)}),
                      "cfg_has_s2p_or_g0cond": [k for k in ("s2p", "g0cond") if k in ce["meta"]["cfg"]]}
    log(f"L2 export-357: {json.dumps(r['export357'])}")
    # the 31101 trainer's model (code-811-s2p) with its flags: the export = get_model_state_dict(model) (safety_ppm.export_weights)
    C8 = "/data/home/amartyavarma/ppm/pose5/code-811-s2p"
    spec = importlib.util.spec_from_file_location("model_ppm_811", f"{C8}/model_ppm.py")
    T8 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(T8)
    ph = any(k.startswith("pose_head.") for k in have)
    c32, c4, c06 = T8.cfg32(ci["layers"]), T8.cfg4b(ci["tower_layers"]), T8.cfg06(ci["dec_layers"])
    with torch.device("meta"):
        t8 = T8.PPM(c32, c4, c06, ci["proj_r"], ci["tower_vocab"], pose_head=ph, decp_bb=True, s2p="cb0", g0cond=True)
    s8 = {n: tuple(p.shape) for n, p in t8.state_dict().items()}
    new = sorted(set(s8) - set(have))
    r["export_31101_predicted"] = {"pose_head_built": ph, "new_keys": {k: s8[k] for k in new}, "gone_keys": sorted(set(have) - set(s8)),
                                   "shape_changes": [(k, have[k], s8[k]) for k in s8 if k in have and tuple(have[k]) != s8[k]],
                                   "engine_keys_missing_from_it": sorted(set(want) - set(s8)),
                                   "engine_shape_mismatch": [(k, want[k], s8[k]) for k in want if k in s8 and s8[k] != want[k]]}
    p = r["export_31101_predicted"]
    r["pass"] = (r["w15973"]["engine_keys_missing"] == sorted([FB.G0_KEY, FB.S2P_KEY]) and not r["w15973"]["shape_mismatch"]
                 and r["export357"]["engine_keys_missing"] == sorted([FB.G0_KEY, FB.S2P_KEY])
                 and p["new_keys"] == {FB.G0_KEY: (5120, 8), FB.S2P_KEY: (2048, 2560)} and not p["gone_keys"] and not p["shape_changes"]
                 and not p["engine_keys_missing_from_it"] and not p["engine_shape_mismatch"])
    log(f"L2 31101 export (predicted from code-811-s2p): {json.dumps(p, default=str)}")
    RES["L2"] = r
    log(f"L2 pass {r['pass']}")
    save()


# ================================================================================================ S inference SLOT
def S():
    r = {}
    rng = np.random.default_rng(5)
    clip = rng.integers(0, 2048, (32, 125)).astype(np.int64)
    pers = "You are Maya, a warm and curious podcast host who loves asking follow-up questions."
    fake = types.SimpleNamespace(g0=object())
    F = np.array([0.5765, 0.5305, -0.2010, -0.7049, 0.5178, -0.6486, 0.5740, 0.5754], np.float32)
    for name, (c, p) in {"empty": (None, None), "clip": (clip, None), "persona": (None, pers), "clip+persona": (clip, pers)}.items():
        sp = MIF.system_block(c, p)
        n = 20
        x = MIF.assemble(sp, np.full(n + FB.CD, FB.PAD, np.int64), np.zeros((32, 0), np.int64), np.zeros((32, 0), np.int64), n, 0)
        P = int(x["P"])
        s = P - 1
        it = dict(x, g0f=F)
        got = FB.FastPPM._g0_inst(fake, it, P, x["ids"], x["depth"], x["present"], x["pdepth"])
        rr = {"P": P, "ids_B[P-2:P]": x["ids"][1, P - 2:P].tolist(), "ids_A[P-1]": int(x["ids"][0, s]), "present[:, P-1]": x["present"][:, s].tolist(),
              "depth_codes_at_P-1": int((x["depth"][:, s] >= 0).sum()), "pdepth_codes_at_P-1": int((x["pdepth"][s] >= 0).sum()),
              "ids_B[P] (conv frame 0)": int(x["ids"][1, P]), "g0_inst": [got[0], got[1].tolist()],
              "voice_clip_code_positions": int((x["depth"][1, :P, 0] >= 0).sum())}
        rr["pass"] = (rr["ids_B[P-2:P]"] == [FB.SP_IM_END, FB.SP_NL] and rr["present[:, P-1]"] == [0.0, 1.0] and rr["depth_codes_at_P-1"] == 0
                      and rr["pdepth_codes_at_P-1"] == 0 and got[0] == s and np.array_equal(got[1], F) and got[1].dtype == np.float32)
        r[name] = rr
        log(f"S {name}: {json.dumps(rr)}")
    # start()'s fail-loud cases
    sp = MIF.system_block(clip, None)
    x = MIF.assemble(sp, np.full(22, FB.PAD, np.int64), np.zeros((32, 0), np.int64), np.zeros((32, 0), np.int64), 20, 0)
    P = int(x["P"])
    off = types.SimpleNamespace(g0=None)
    a = (P, x["ids"], x["depth"], x["present"], x["pdepth"])
    bad_ids = x["ids"].copy(); bad_ids[1, P - 1] = FB.PAD
    r["fail_loud"] = {
        "g0_engine_without_g0f": raises(lambda: FB.FastPPM._g0_inst(fake, dict(x), *a), ValueError),
        "g0_engine_g0f_None": raises(lambda: FB.FastPPM._g0_inst(fake, dict(x, g0f=None), *a), ValueError),
        "g0_engine_wrong_shape": raises(lambda: FB.FastPPM._g0_inst(fake, dict(x, g0f=F[:7]), *a), ValueError),
        "g0_engine_nan": raises(lambda: FB.FastPPM._g0_inst(fake, dict(x, g0f=np.where(np.arange(8) == 3, np.nan, F)), *a), ValueError),
        "g0_engine_int_vector": raises(lambda: FB.FastPPM._g0_inst(fake, dict(x, g0f=np.arange(8)), *a), ValueError),
        "g0_engine_slot_not_newline": raises(lambda: FB.FastPPM._g0_inst(fake, dict(x, g0f=F), P, bad_ids, x["depth"], x["present"], x["pdepth"]),
                                             AssertionError),
        "off_engine_with_g0f": raises(lambda: FB.FastPPM._g0_inst(off, dict(x, g0f=F), *a), ValueError)}
    r["off_engine_without_g0f_returns_None"] = FB.FastPPM._g0_inst(off, dict(x), *a) is None
    r["float64_f_cast_to_float32"] = FB.FastPPM._g0_inst(fake, dict(x, g0f=F.astype(np.float64)), *a)[1].dtype == np.float32
    r["pass"] = (all(r[k]["pass"] for k in ("empty", "clip", "persona", "clip+persona")) and all(v for v in r["fail_loud"].values())
                 and r["off_engine_without_g0f_returns_None"] and r["float64_f_cast_to_float32"])
    RES["S"] = r
    log(f"S pass {r['pass']}")
    save()


if __name__ == "__main__":
    st = (sys.argv[1] if len(sys.argv) > 1 else "S,L1,LK,L2").split(",")
    for k in st:
        try:
            {"S": S, "L1": L1, "L2": L2, "LK": LK}[k]()
        except Exception as e:                                          # record and continue with the other parts (fail visible in json)
            import traceback
            traceback.print_exc()
            RES[k] = {"pass": False, "error": f"{type(e).__name__}: {e}"}
            save()
    RES["t_end"] = time.strftime("%Y-%m-%d %H:%M:%S")
    RES["all_pass"] = all(RES.get(k, {}).get("pass") for k in st)
    save()
    log(f"done all_pass {RES['all_pass']}")
