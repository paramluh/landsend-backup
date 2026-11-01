"""speaker LoRA FT CPU DRY RUN + unit tests (10-24, no GPU, no launch). The LoRA trainer module resolved from an env exactly as rank 0 of the
world-8 job resolves it (COTRAIN_CFG_DRY=1). Two modes (argv[3]):
  run       (env = lora/run_brianna.env): cfg diff vs 26232 (abort rule) + 29345 (report); plan == env; trainable / frozen parameter counts per
            group on the real PPM 32B module tree (meta device) with the LoRA injected; LoRA math on CPU (B = 0 -> base output exactly; forward ==
            the merged weight; init world-invariant); EVERY her micro (her on B, B codes == her lane, A == partner FLAC, light == cb0 labels, a
            clip from ANOTHER call of hers on every item) and every replay micro of the first steps == 29345's data path (data_aloe_v2vox) bit
            for bit; the trainer's same-as-v2 check for her + replay items.
  identity  (env = lora/run_identity_eval.env): CFG diff vs 29345 (only layout / run dirs / init may differ), the 29345 plan digest, the
            standard eval micros and the step-0 train micros of every logical rank through the LoRA trainer's data module == data_aloe_v2vox's
            (29345's module) bit for bit, model_lora.build(rank 0) is model_ppm.build (same function call).
usage (compute node, env sourced): python -I dry_lora.py <code dir> <out dir> run|identity"""
import hashlib
import json
import os
import sys

sys.path.insert(0, sys.argv[1])
OUT, MODE = sys.argv[2], sys.argv[3]
os.makedirs(OUT, exist_ok=True)
os.environ.update(RANK="0", WORLD_SIZE="8", LOCAL_RANK="0", LOCAL_WORLD_SIZE="8", COTRAIN_CFG_DRY="1")
os.environ["QQA_RUN"] = os.environ["QQA_RUN"] + "-jDRY"
os.environ["ALOE_FLAC_ROOT"] = os.environ["ALOE_SHARED_FLAC"]
import numpy as np  # noqa: E402
import torch  # noqa: E402
import train_spklora_vox as T  # noqa: E402
import data_aloe_v2vox as V  # noqa: E402

T.LOGDIR = OUT
A, SPL, D, M, ML = T.A, T.SPL, T.D, T.M, T.ML
facts = {"mode": MODE, "PROD": T.PROD, "SPK": T.SPK, "LWORLD": T.LWORLD, "slots": T.GMAX, "lora": T.CFG.get("lora"), "freeze": list(T.FREEZE),
         "train_groups": list(T.TRAIN_GROUPS), "export_mode": T.EXPORT_MODE}
print(json.dumps(facts), flush=True)
n, bad = T.cfg_diff_26232()
r29 = T.cfg_diff_report(os.environ["ALOE_REF_CFG_29345"], "29345")
facts["cfg_diff_26232"] = {"n": n, "bad": bad}
facts["cfg_diff_29345"] = r29
PL = A.make_plan(T.LWORLD)
dg = hashlib.sha256(json.dumps(PL["items"]).encode()).hexdigest()[:16]
assert dg == os.environ["QQA_REF_PLAN_DIGEST"] and PL["steps"] == int(os.environ["QQA_STEPS"]), (dg, PL["steps"])
facts["plan"] = {"digest": dg, "steps": PL["steps"], "items": len(PL["items"]), "eval": len(PL["eval"])}
cfg, ecfg = T.CFG["data"], T.CFG["eval_data"]
OFF = np.arange(32, dtype=np.int64) * 2048


def eq_micro(a, b, keys=("ids", "depth", "present", "pid", "sel", "y", "word", "pad", "eob", "nw", "cb0", "light")):
    return all(np.array_equal(np.asarray(a[k]), np.asarray(b[k])) for k in keys) and a["P"] == b["P"] and a["prov"] == b["prov"] and \
        a["voice_clip"] == b["voice_clip"] and a["aref"] == b["aref"] and (("wav" not in a and "wav" not in b) or np.array_equal(a["wav"], b["wav"]))


# ---- parameter counts per group on the real module tree (meta device), LoRA injected as the trainer's build does
c32, c4, c06 = M.cfg32(), M.cfg4b(), M.cfg06()
with torch.device("meta"):
    pm = M.PPM(c32, c4, c06, 2048, M.TOWER_VOCAB, pose_head=False, decp_bb=False, vadapter=T.CFG["vadapter"], nopose=True)
    if T.LORA_RANK:
        conv = ML.lora_inject(pm, T.LORA_RANK, T.CFG["lora"]["alpha"], tuple(T.CFG["lora"]["targets"]))
cnt = {}
for nm, p in pm.named_parameters():
    g = T.group_of(nm)
    key = "FROZEN:" + M.group_of(nm) if g is None else g
    cnt[key] = cnt.get(key, 0) + p.numel()
facts["param_counts"] = {k: int(v) for k, v in sorted(cnt.items())}
facts["trainable_total"] = int(sum(v for k, v in cnt.items() if not k.startswith("FROZEN")))
facts["frozen_total"] = int(sum(v for k, v in cnt.items() if k.startswith("FROZEN")))
print("PARAMS", json.dumps(facts["param_counts"]), "trainable", facts["trainable_total"], "frozen", facts["frozen_total"], flush=True)
del pm
if MODE == "identity":
    assert not T.SPK and not T.LORA_RANK and not T.FREEZE and T.EXPORT_MODE == "full"
    import inspect  # noqa: E402
    src = inspect.getsource(ML.build).split("\n")
    facts["build_rank0_is_model_ppm_build"] = any("return M.build(c32, c4, c06, dev, log, **kw)" in s for s in src[:12])
    assert dg == "d1a3e3721328da0a" and len(PL["eval"]) == 316
    st, vs = A.Stores(), V.Stores()
    ok_e = [eq_micro(st.conv(it, ecfg, L, audio=True), vs.conv(it, ecfg, L, audio=True)) for L in range(64) for it in PL["eval"][L::64][:1]]
    ok_t = [eq_micro(st.conv(PL["items"][L], cfg, L, audio=True), vs.conv(PL["items"][L], cfg, L, audio=True)) for L in range(64)]
    facts["identity"] = {"eval_first_micro_per_logical_rank_equal": f"{sum(ok_e)}/{len(ok_e)}", "step0_train_micro_per_logical_rank_equal": f"{sum(ok_t)}/{len(ok_t)}",
                         "cfg_29345_diff_keys": [r.split(":")[0].strip() for r in r29]}
    ok = (not bad and all(ok_e) and all(ok_t) and facts["build_rank0_is_model_ppm_build"]
          and all(k.split(".")[0] in ("accum", "control_dir", "log_root", "run_tag", "export", "local_export_root", "resume", "init", "vadapter", "flac_root") for k in facts["identity"]["cfg_29345_diff_keys"]))   # flac_root: the DRY override only (the run keeps 29345's)
    json.dump(facts, open(f"{OUT}/dry_facts.json", "w"), indent=1, default=str)
    print("IDENTITY", json.dumps(facts["identity"]), flush=True)
    print("DRY", "PASS" if ok else "FAIL", flush=True)
    sys.exit(0)
# ---- run mode: LoRA math on CPU (no FSDP): conversion keeps the base FQN, B = 0 -> base output, forward == merged weight, init invariance
torch.manual_seed(0)
lin = torch.nn.Linear(64, 96, bias=False)
base = torch.nn.Linear(64, 96, bias=False); base.weight.data.copy_(lin.weight.data)
holder = torch.nn.Module(); holder.q_proj = lin
lin.__class__ = ML.LoRALinear
lin.lora_A = torch.nn.Parameter(torch.empty(4, 64)); lin.lora_B = torch.nn.Parameter(torch.empty(96, 4)); lin.lora_scale = 8.0 / 4
names = sorted(n_ for n_, _ in holder.named_parameters())
g = torch.Generator().manual_seed(1)
lin.lora_A.data.copy_(torch.rand(4, 64, generator=g) - 0.5); lin.lora_B.data.zero_()
x = torch.randn(5, 64)
t1 = torch.equal(lin(x), base(x))
lin.lora_B.data.copy_(torch.randn(96, 4, generator=g))
mw = ML.merged_weight(base.weight.data, lin.lora_A.data, lin.lora_B.data, lin.lora_scale, torch.float32)
t2 = torch.allclose(lin(x), x @ mw.T, rtol=1e-5, atol=1e-5)
rows_ok = torch.equal(ML.merged_weight(base.weight.data[32:64], lin.lora_A.data, lin.lora_B.data[32:64], 2.0, torch.float32), mw[32:64])
facts["lora_unit"] = {"fqn_kept": names == ["q_proj.lora_A", "q_proj.lora_B", "q_proj.weight"], "B0_equals_base": t1, "forward_eq_merged": t2,
                      "row_shard_merge_eq": rows_ok}
print("LORA-UNIT", json.dumps(facts["lora_unit"]), flush=True)
# ---- her micros (all) + replay micros (first 2 steps) through the trainer's data path
man = SPL.manifest()
side = A.her_side(PL)
RK2KR = {x["row_key"]: (x["k"], x["r"]) for x in SPL.speaker_rows(A.SPK)}
st, vs = A.Stores(), V.Stores()
her = [(t, it) for t, it in enumerate(PL["items"]) if it is not None and it[3] == 0]
rep = [(t, it) for t, it in enumerate(PL["items"]) if it is not None and it[3] == 1]


def acct(rk, s_dir):
    return man["/".join(rk.split("/")[2:4])]["acc"][s_dir[-2:]]


res = {"her_micros": 0, "B_is_her": 0, "B_codes_eq_her_lane": 0, "A_is_partner": 0, "light_eq_cb0": 0, "voice_drawn": 0, "voice_clip": 0,
       "clip_other_call_her": 0, "flipped": 0}
seen = set()
for t, it in her:
    if tuple(it[:3]) in seen:                                     # her rows repeat every epoch: check each row once (same micro builder)
        continue
    seen.add(tuple(it[:3]))
    k, r, sw = it[:3]
    m = st.conv(it, cfg, t % T.LWORLD, audio=False)
    row = st.open(k)[1][r]
    bs, as_ = ("user", "assistant") if sw else ("assistant", "user")
    P, n_ = int(m["P"]), int(row["n_frames"])
    cb = m["depth"][1][P + D.CD:P + D.CD + n_] - OFF
    pv = json.loads(m["prov"])
    res["her_micros"] += 1
    res["flipped"] += int(sw)
    res["B_is_her"] += int(acct(row["row_key"], row[f"{bs}_speaker_dir"]) == A.SPK)
    res["B_codes_eq_her_lane"] += int(np.array_equal(cb.T, np.asarray(row[f"codes_{bs}"])[:, :n_].astype(np.int64)))
    res["A_is_partner"] += int(m["aref"]["side"] == as_ and acct(row["row_key"], row[f"{as_}_speaker_dir"]) != A.SPK)
    res["light_eq_cb0"] += int(len(m["light"]) == int((np.asarray(m["cb0"]) >= 0).sum()))
    res["voice_drawn"] += int(m["voice_drawn"]); res["voice_clip"] += int(m["voice_clip"])
    if m["voice_clip"]:
        sk = pv["clips"][0]["src_key"]
        k2, r2 = RK2KR[sk]
        srow = st.open(k2)[1][r2]
        res["clip_other_call_her"] += int("/".join(sk.split("/")[2:4]) != "/".join(row["row_key"].split("/")[2:4])
                                          and acct(sk, srow[f"{side[(k2, r2)]}_speaker_dir"]) == A.SPK)
rp_ok = []
for t, it in rep[:8]:
    rp_ok.append(eq_micro(st.conv(it, cfg, t % T.LWORLD, audio=True), vs.conv(tuple(it[:3]), cfg, t % T.LWORLD, audio=True)))
res["replay_micros_eq_29345_path"] = f"{sum(rp_ok)}/{len(rp_ok)}"
res["her_rows_per_step"] = sum(1 for x in PL["items"][:T.LWORLD] if x is not None and x[3] == 0)
res["replay_rows_per_step"] = sum(1 for x in PL["items"][:T.LWORLD] if x is not None and x[3] == 1)
res["her_rows_unique"] = len(seen)
res["B_frames_her_train"] = int(sum(st.open(k)[1][r]["n_frames"] for (k, r, s) in seen))
facts["micros"] = res
print("MICROS", json.dumps(res), flush=True)
json.dump(facts, open(f"{OUT}/dry_facts.json", "w"), indent=1, default=str)
u = facts["lora_unit"]
ok = (not bad and all(u.values()) and res["B_is_her"] == res["B_codes_eq_her_lane"] == res["A_is_partner"] == res["light_eq_cb0"] == res["her_micros"]
      and res["voice_drawn"] == res["voice_clip"] == res["clip_other_call_her"] == res["her_micros"] and all(rp_ok))
print("DRY", "PASS" if ok else "FAIL", flush=True)
