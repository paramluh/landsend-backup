"""speaker FT CPU DRY RUN (10-24, no GPU, no launch): the speaker trainer module resolved from run_speaker.env exactly as rank 0 of the world-16
job resolves it (COTRAIN_CFG_DRY=1), then:
  CFG       its startup diff vs 26232 (abort rule) + the report-only diffs vs 29345 / 28511;
  PLAN      plan digest / counts / steps == env; layout (slots, dummies);
  ALL       every train + speaker-eval micro through the trainer's data path (data_aloe_spk.Stores.conv, spk=True): her on B (manifest account of
            the B data side), B codes == HER lane's sam-sep codes in the store (all 32 codebooks, every frame), A = the partner's raw FLAC
            (index entry of the other side; lane account != hers), light frames == B cb0 labels (light_frac 1.0), voice clip draws / clips
            (clip from ANOTHER call of hers, codes == the source row's codes, lane blanked);
  FLIP      for a flipped item (swap 1) vs 29345's builder (data_aloe_v2.speech_micro_v2, swap 0 = the original orientation, same row): the
            flipped B channel conversation (ids + codes) == the original A channel conversation; A's Voxtral source == the original B side's FLAC;
  SAME-AS-V2 the trainer's startup check for a few speaker items (conversation part == the reference v2 micro of the same row, old stores);
  RENDER    3 rows (>= 1 flipped, >= 1 later chunk) decoded: B lane text (Qwen3 tokenizer), her store words, partner words, clip, A FLAC + wav.
usage (compute node, env sourced from run_speaker.env): python -I dry_spk.py <code dir> <out dir>"""
import hashlib
import json
import os
import sys

sys.path.insert(0, sys.argv[1])
OUT = sys.argv[2]
os.makedirs(OUT, exist_ok=True)
os.environ.update(RANK="0", WORLD_SIZE="16", LOCAL_RANK="0", LOCAL_WORLD_SIZE="8", COTRAIN_CFG_DRY="1")
os.environ["QQA_RUN"] = os.environ["QQA_RUN"] + "-jDRY"
os.environ["ALOE_FLAC_ROOT"] = os.environ["ALOE_SHARED_FLAC"]          # dry node: the shared copy the per-host symlink points to
import numpy as np  # noqa: E402
import train_speaker_vox as T  # noqa: E402

T.LOGDIR = OUT
A, SPL, D = T.A, T.SPL, T.D
ACC = A.SPK
print("PROD", T.PROD, "SPK", T.SPK, "LWORLD", T.LWORLD, "slots", T.GMAX, "dec_k", T.CFG["dec_k"], "dec units", T.DEC_UNITS, "offload", T.OFFLOAD,
      "sched", T.CFG["sched"], "dec_sched", T.CFG["dec_sched"], flush=True)
n, bad = T.cfg_diff_26232()
r29 = T.cfg_diff_report(os.environ["ALOE_REF_CFG_29345"], "29345")
r28 = T.cfg_diff_report(os.environ["ALOE_REF_CFG_28511"], "28511")
facts = {"cfg_diff_26232": {"n": n, "bad": bad}, "cfg_diff_29345": r29}
PL = A.make_plan(T.LWORLD)
dg = hashlib.sha256(json.dumps(PL["items"]).encode()).hexdigest()[:16]
assert dg == os.environ["QQA_REF_PLAN_DIGEST"] and PL["steps"] == int(os.environ["QQA_STEPS"]) and len(PL["items"]) == int(os.environ["ALOE_EXPECT_TRAIN_MICROS"])
assert len(PL["eval"]) == int(os.environ["ALOE_EXPECT_EVAL_MICROS"]) and len(PL["eval_spk"]) == int(os.environ["ALOE_SPK_EXPECT_EVAL_MICROS"])
PF = json.load(open(os.environ["ALOE_SPK_PLAN"]))
assert PF["plan_digest"] == dg and [tuple(x) for x in PF["items"]] == PL["items"]
side = A.her_side(PL)
RK2KR = {x["row_key"]: (x["k"], x["r"]) for x in SPL.speaker_rows(ACC)}
facts["plan"] = {"digest": dg, "steps": PL["steps"], "items": len(PL["items"]), "dummies_last_step": PL["steps"] * T.LWORLD - len(PL["items"]),
                 "eval": len(PL["eval"]), "eval_spk": len(PL["eval_spk"]), "convs_heldout": PL["convs_heldout"], "stats": PL["stats"]}
man = SPL.manifest()
cfg, ecfg = T.CFG["data"], T.CFG["eval_data"]
st = A.Stores()
OFF = np.arange(32, dtype=np.int64) * 2048


def acct_of(row, s):
    return man["/".join(row["row_key"].split("/")[2:4])]["acc"][row[f"{s}_speaker_dir"][-2:]]


def conv_of(rk):
    return "/".join(rk.split("/")[2:4])


def check(item, L, c_, audio):
    k, r, sw = item
    m = st.conv(item, c_, L, audio=audio, spk=True)
    _, stk, _ = st.open(k)
    row = stk[r]
    P, n_ = int(m["P"]), int(row["n_frames"])
    bside = "user" if sw else "assistant"
    aside = "assistant" if sw else "user"
    cd = D.CD
    codesB = m["depth"][1][P + cd:P + cd + n_] - OFF                    # [n, 32]: B codes per conversation frame (code delay CD)
    pv = json.loads(m["prov"])
    out = {"item": list(item), "row_key": row["row_key"], "chunk": int(row["chunk_index"]), "swap": sw, "flip": bool(sw), "n": n_, "P": P,
           "B_is_her": acct_of(row, bside) == ACC and side[(k, r)] == bside,
           "B_codes_eq_her_lane": bool(np.array_equal(codesB.T, np.asarray(row[f"codes_{bside}"])[:, :n_].astype(np.int64))),
           "B_codes_ne_partner": not np.array_equal(codesB.T, np.asarray(row[f"codes_{aside}"])[:, :n_].astype(np.int64)),
           "A_empty": bool((m["depth"][0] < 0).all() and (m["ids"][0][P:] == D.PAD_ID).all()),
           "A_src_side": m["aref"]["side"], "A_flac": m["aref"]["flac"],
           "A_is_partner": m["aref"]["side"] == aside and m["aref"]["flac"].endswith(f"{conv_of(row['row_key'])}/{row[f'{aside}_speaker_dir']}.flac")
           and acct_of(row, aside) != ACC,
           "light": int(len(m["light"])), "cb0_labels": int((np.asarray(m["cb0"]) >= 0).sum()),
           "voice_drawn": int(m["voice_drawn"]), "voice_clip": int(m["voice_clip"])}
    out["light_ratio"] = out["light"] / max(out["cb0_labels"], 1)
    if m["voice_clip"]:
        cl = pv["clips"][0]
        sk = cl["src_key"]
        k2, r2 = RK2KR[sk]
        srow = st.open(k2)[1][r2]
        s2 = side[(k2, r2)]
        w0, w1 = cl["frames"]
        pc = m["depth"][1][:P]
        clip_codes = pc[(pc >= 0).all(-1)] - OFF                    # the prefix's coded positions = the clip frames in order
        out.update(clip_src=sk, clip_other_call=conv_of(sk) != conv_of(row["row_key"]), clip_src_is_her=acct_of(srow, s2) == ACC,
                   clip_len=w1 - w0, clip_codes_eq_src=bool(np.array_equal(clip_codes.T, np.asarray(srow[f"codes_{s2}"])[:, w0:w1].astype(np.int64))),
                   clip_lane_blank=bool(set(np.unique(m["ids"][1][3:P - 2]).tolist()) <= {D.PAD_ID, 151678, 151679}))
    if audio:
        out["wav_s"] = round(len(m["wav"]) / 24000.0, 2)
        out["wav_rms"] = float(np.sqrt(np.mean(m["wav"].astype(np.float64) ** 2)))
    return m, row, out


# ---- ALL speaker micros (train + speaker eval), no audio (A audio checked on the rendered rows)
allc = []
for t, it in enumerate(PL["items"]):
    L = t % T.LWORLD
    _, _, o = check(it, L, cfg, False)
    allc.append(o)
for j, it in enumerate(PL["eval_spk"]):
    _, _, o = check(it, j % T.LWORLD, ecfg, False)
    o["eval"] = True
    allc.append(o)
keys = ("B_is_her", "B_codes_eq_her_lane", "B_codes_ne_partner", "A_empty", "A_is_partner")
facts["all"] = {"micros": len(allc), **{k: sum(bool(o[k]) for o in allc) for k in keys},
                "flipped": sum(o["flip"] for o in allc), "later_chunks": sum(o["chunk"] > 0 for o in allc),
                "light_ratio_min": min(o["light_ratio"] for o in allc), "light_ratio_max": max(o["light_ratio"] for o in allc),
                "light_frames_train": sum(o["light"] for o in allc if not o.get("eval")), "voice_drawn": sum(o["voice_drawn"] for o in allc),
                "voice_clip": sum(o["voice_clip"] for o in allc),
                **{k: sum(bool(o.get(k)) for o in allc if o["voice_clip"]) for k in ("clip_other_call", "clip_src_is_her", "clip_codes_eq_src", "clip_lane_blank")}}
print("ALL", json.dumps(facts["all"]), flush=True)
# ---- per-step frame totals (logical world) -> decoder updates
fr = [o["light"] for o in allc if not o.get("eval")]
items_fr = {tuple(o["item"]): o["light"] for o in allc if not o.get("eval")}
per_step = [sum(items_fr[tuple(x)] for x in PL["items"][s * T.LWORLD:(s + 1) * T.LWORLD]) for s in range(PL["steps"])]
facts["frames"] = {"B_frames_train": sum(fr), "per_row_mean": sum(fr) / len(fr), "per_row_max": max(fr), "per_step": per_step,
                   "dec_frames_per_update_per_gpu_max": -(-max(fr) // T.CFG["dec_k"]), "dec_updates_per_step": T.DEC_UNITS,
                   "dec_updates_per_epoch": T.DEC_UNITS * PL["steps"]}
# ---- FLIP check + SAME-AS-V2 vs 29345's builder on the same rows
import importlib.util  # noqa: E402
spec = importlib.util.spec_from_file_location("data_aloe_v2_22383", "/data/home/amartyavarma/ppm/aloe-v2-ft/code/data_aloe_v2.py")
R = importlib.util.module_from_spec(spec); spec.loader.exec_module(R)
rm = json.load(open(os.environ["ALOE_SAMSEP_ROW_MAP"]))
OON = {k: {int(v): int(o) for o, v in rm["map"][f"m1n{k}"]["new_of_old"].items()} for k in range(4)}
RST = R.Stores(roots=R.SRC_STORES)
flips = [it for it in PL["items"] if it[2] == 1]
plain = [it for it in PL["items"] if it[2] == 0]
later = [it for it in PL["items"] if allc[PL["items"].index(it)]["chunk"] > 0]
fl = []
for it in flips[:3] + [x for x in later if x[2] == 1][:1]:
    k, r, sw = it
    m, row, o = check(it, 0, cfg, True)
    node, stk, sib = st.open(k)
    mo = A.A2.speech_micro_v2(row, r, sib, cfg, 0, 0)               # 29345's builder, ORIGINAL orientation (B = data assistant, A = data user)
    P1, P0 = int(m["P"]), int(mo["P"])
    a0 = A.vox_index()[k][r]["A"]["user"]["flac"]                        # the original orientation's A (Voxtral) source = the data user lane
    fl.append({"item": list(it), "row_key": row["row_key"], "chunk": o["chunk"],
               "B_ids_eq_orig_A": bool(np.array_equal(m["ids"][1][P1:], mo["ids"][0][P0:])),
               "B_codes_eq_orig_A": bool(np.array_equal(m["depth"][1][P1:], mo["depth"][0][P0:])),
               "B_ne_orig_B": not np.array_equal(m["depth"][1][P1:], mo["depth"][1][P0:]),
               "A_src_eq_orig_B_flac": m["aref"]["flac"] == A.vox_index()[k][r]["A"]["assistant"]["flac"] and m["aref"]["flac"].endswith(row["assistant_speaker_dir"] + ".flac"),
               "orig_A_src": a0, "A_src": m["aref"]["flac"], "wav_s": o["wav_s"], "wav_rms": o["wav_rms"]})
facts["flip"] = fl
print("FLIP", json.dumps(fl), flush=True)
sa = []
for it, L, c_ in [(PL["items"][0], 0, cfg), (flips[0], 1, cfg), (later[0], 2, cfg), (tuple(PL["eval_spk"][0]), 0, ecfg)]:
    k, r, sw = it
    ro = OON[k][r]
    _, sto, sibo = RST.open(k)
    mr = R.speech_micro_v2(sto[ro], ro, sibo, c_, L, sw)
    mn = st.conv(it, c_, L, audio=False, spk=True)
    Pr, Pn = int(mr["P"]), int(mn["P"])
    mk = lambda x: np.asarray(x) >= 0  # noqa: E731
    sa.append({"item": list(it), "ids_B": bool(np.array_equal(mr["ids"][1][Pr:], mn["ids"][1][Pn:])),
               "depth_mask_B": bool(np.array_equal(mk(mr["depth"][1][Pr:]), mk(mn["depth"][1][Pn:]))),
               "targets": all(np.array_equal(np.asarray(mr[x]), np.asarray(mn[x])) for x in ("y", "word", "pad", "eob", "nw"))
               and bool(np.array_equal(np.asarray(mr["sel"]) - Pr, np.asarray(mn["sel"]) - Pn)) and bool(np.array_equal(np.asarray(mr["light"]) - Pr, np.asarray(mn["light"]) - Pn)),
               "voice_draw": mr["voice_drawn"] == mn["voice_drawn"], "codes_B_differ_from_old": not np.array_equal(mr["depth"][1][Pr:], mn["depth"][1][Pn:])})
facts["same_as_v2"] = sa
print("SAME-AS-V2", json.dumps(sa), flush=True)
# ---- standard eval micro through the speaker module == 29345's module (data_aloe_v2vox) exactly
import data_aloe_v2vox as V  # noqa: E402
ev0 = PL["eval"][0]
ma, mb = st.conv(ev0, ecfg, 0, audio=True, spk=False), V.Stores().conv(ev0, ecfg, 0, audio=True)
facts["std_eval_identical"] = all(np.array_equal(np.asarray(ma[x]), np.asarray(mb[x])) for x in ("ids", "depth", "present", "sel", "y", "light", "cb0", "wav")) and ma["prov"] == mb["prov"]
# ---- RENDER 3 rows: plain first chunk, flipped, later chunk (flipped if possible)
from transformers import AutoTokenizer  # noqa: E402
tok = AutoTokenizer.from_pretrained("/data/home/amartyavarma/models/Qwen3-32B")
pick = [plain[0], flips[0], ([x for x in later if x[2] == 1] or later)[0]]
ren = []
for it in pick:
    m, row, o = check(it, 0, cfg, True)
    P = int(m["P"])
    bside = "user" if it[2] else "assistant"
    aside = "assistant" if it[2] else "user"
    lane = [int(t) for t in m["ids"][1][P:] if int(t) < 151643]
    ren.append({**{k: o[k] for k in ("item", "row_key", "chunk", "swap", "n", "P", "voice_clip", "A_flac", "wav_s", "wav_rms", "light_ratio") if k in o},
                "clip_src": o.get("clip_src"), "B_data_side": bside, "B_account": acct_of(row, bside), "A_account": acct_of(row, aside),
                "B_lane_text_decoded": tok.decode(lane)[:400], "her_store_words": " ".join(w.get("text", "") for w in row[f"words_{bside}"][:60])[:400],
                "partner_store_words": " ".join(w.get("text", "") for w in row[f"words_{aside}"][:40])[:300]})
facts["render"] = ren
json.dump(facts, open(f"{OUT}/dry_facts.json", "w"), indent=1, default=str)
for x in ren:
    print("RENDER", json.dumps(x), flush=True)
a = facts["all"]
ok = (not bad and a["B_is_her"] == a["B_codes_eq_her_lane"] == a["A_empty"] == a["A_is_partner"] == a["micros"]
      and a["light_ratio_min"] == a["light_ratio_max"] == 1.0 and a["clip_other_call"] == a["clip_src_is_her"] == a["clip_codes_eq_src"] == a["voice_clip"]
      and all(all(v for k, v in f.items() if k.startswith(("B_", "A_src_eq"))) for f in fl)
      and all(s["ids_B"] and s["depth_mask_B"] and s["targets"] and s["voice_draw"] for s in sa) and facts["std_eval_identical"])
print("DRY", "PASS" if ok else "FAIL", flush=True)
