"""PPM run ppm-v1: reproduce every rank's podcast (speech) row stream exactly as train_ppm.main builds it, split it at the stop step into
consumed / unseen rows, and write the si20k fine-tune's podcast plan (PODCAST_PLAN_ppm.json). CPU only, read-only on every store.

The plan functions are the TRAINER'S OWN (prod code /data/home/amartyavarma/ppm/code imported, nothing re-implemented):
  data_ppm.speech_plan (stage-3 DS.plan, FULL rows, ppm-v1 eval-episode holdout), data_ppm.d2_stream (D2 switch-over at step 170, s3eval-v1
  holdout, shuffle (seed, rank, 0xD2, 170)), data_ppm.load_extra_plan / code_of / split_code, s3rows.load_node, data_s3q.load_index.
Arguments exactly as train_ppm.main (lines ~846-866): speech_plan(snode, sidx, 20261011, 8, local, 0.0005, 'ppm-v1', E_=2);
  d2_stream(rows, sidx, hplan, local, RANK, 170, nsm=2, ['s3eval-v1'], 0.0005, 20261011); RANK = 8 x (host index in the SORTED host list) + local.

modes
  host  --host H --src live|meta --out DIR [--build N]
        live: the host's node-local stores (/scratch/amartyavarma/s3-rows + /scratch/amartyavarma/s3-rows-extra/<slice>); must run ON H.
              Also checks every file the host's rows need (size vs the HF-backup MANIFEST / the extra-slice LISTING) and, with --build N,
              builds the first N unseen micros of every slot with the trainer's MixedSpeechDS (+ one eval micro) and checks their row keys.
              A missing store -> DIR/H.live.json {"status": "missing", ...} (no plan).
        meta: the Weka copies of the same metadata (ckpt-restore/.../rank_plan/meta.tar extracted to plan/_ppmv1_meta), any host.
        -> DIR/H.<src>.npz (per local rank: codes, key / episode hashes, n_frames, eval codes) + DIR/H.<src>.json (stats, digests, checks)
  merge --dir DIR --out JSON
        verification (live == meta; vs the Oct-15 regen bundle rank_plan.npz / rank_plan_digest.json / rank_plan_meta.json; vs the
        run's logged D2 numbers and native counts) + consumed / unseen split at STOP_STEP + cross-rank row-key checks -> JSON (+ .npz with the
        full unseen code lists per rank)."""
import argparse
import hashlib
import json
import os
import re
import socket
import sys
import time

import numpy as np

os.environ.setdefault("PPM_CD", "2")
CODE = "/data/home/amartyavarma/ppm/code"
sys.path.insert(0, CODE)
import data_ppm as D  # noqa: E402

RUN = "ppm-v1"
STOP_STEP, PLAN_STEPS = 12829, 17366
SEQ = ("pose", "speech", "speech", "text")                  # PPM_GA_SEQ (logged ga_seq of every ppm-v1 job)
NSM = SEQ.count("speech")
LW = 8
HOSTS = ["slinky-0", "slinky-1", "slinky-2", "slinky-3", "slinky-5", "slinky-6", "slinky-8", "slinky-9", "slinky-10", "slinky-11", "slinky-12",
         "slinky-13", "slinky-14", "slinky-16", "slinky-17", "slinky-19"]        # sorted -V = rank order (every ppm-v1 job's PPM_HOSTS)
WORLD = LW * len(HOSTS)
SPEECH_SEED, EVAL_FRAC, EVAL_SALT, E_ = 20261011, 0.0005, "ppm-v1", 2
S0, SALTS = 170, ["s3eval-v1"]
EXTRA_PLAN = "/data/home/amartyavarma/ppm/configs/extra_plan_d2.json"
LIVE_ROOT, LIVE_EXTRA = "/scratch/amartyavarma/s3-rows", "/scratch/amartyavarma/s3-rows-extra"
META = "/data/home/amartyavarma/ppm/si20k-vox-ppm/plan/_ppmv1_meta/meta"
BUNDLE = "/data/home/amartyavarma/ppm/ckpt-restore/ppm-v1-w-8869/run_state/data_manifest"
LOGS = "/data/home/amartyavarma/ppm/logs"
LAST_JOB = "22613"
FULL = 7500
FPS = 12.5
# the exact data cfg every ppm-v1 job from step 399 on passed to data_ppm.speech_micro (logged cfg["data"] of job 22613; asserted in merge)
CFG = D.default_cfg(seed=20261011, sp_cap=2000, persona_sides="both", light_frac=1.0, pose_light_frac=1.0, txt_cap=7500, crop_conv=True,
                    chat_side="assistant", chat_eob=True, pack_attn="causal")


def h64(s):
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "little")


def sha(*parts):                                             # = regen_rank_plan.sha (the bundle's digest rule)
    h = hashlib.sha256()
    for x in parts:
        h.update(x if isinstance(x, (bytes, bytearray)) else (x.tobytes() if isinstance(x, np.ndarray) else json.dumps(x, sort_keys=True).encode()))
        h.update(b"\x00")
    return h.hexdigest()


def num(node):
    return int(str(node).split("-")[1])


def micro_digest(m):
    h = hashlib.sha256()
    for k in sorted(m):
        v = m[k]
        h.update(k.encode())
        if isinstance(v, np.ndarray):
            h.update(f"{v.dtype}|{v.shape}".encode()); h.update(np.ascontiguousarray(v).tobytes())
        else:
            h.update(json.dumps(v, sort_keys=True, default=str).encode())
    return h.hexdigest()[:32]


# ------------------------------------------------------------------------------------------------ host mode
def slot_tables(root):
    """-> node, keys (list), key hash u64, episode hash u64, n_frames int32 of the slice at root (rows.parquet + index.npz)."""
    import pyarrow.parquet as pq
    node = D.R3.load_node(root)
    t = pq.read_table(f"{root}/rows.parquet", columns=["row", "row_key", "dataset", "podcast_id", "episode_id", "n_frames"]).to_pydict()
    assert t["row"] == list(range(len(t["row"]))), root
    assert np.array_equal(np.asarray(t["n_frames"], np.int32), node["length"]), ("rows.parquet n_frames != index.npz", root)
    kh = np.fromiter((h64(k) for k in t["row_key"]), np.uint64, len(t["row_key"]))
    eh = np.fromiter((h64(D.DS.episode_key(a, b, c)) for a, b, c in zip(t["dataset"], t["podcast_id"], t["episode_id"])), np.uint64, len(kh))
    return node, t["row_key"], kh, eh, node["length"]


def file_checks(host, sroot, hp, slots_codes):
    """every file the host's rows need exists with the recorded size. native: all shards (the native stream spans every shard) x
    {DONE, index.npz, codes_user, codes_assistant, words, persona} vs the HF-backup MANIFEST (/scratch/amartyavarma/hfb/ppm-s3-rows-H);
    extra: the plan's file list of each slot vs the slice LISTING (Weka copy) + every code's shard in shards_present."""
    out = {"native": {}, "extra": []}
    node = D.R3.load_node(sroot)
    man = {}
    mp = f"/scratch/amartyavarma/hfb/ppm-s3-rows-{host}/_backup/MANIFEST.tsv"
    if os.path.exists(mp):
        for ln in open(mp):
            p = ln.rstrip("\n").split("\t")
            if len(p) >= 2 and p[0] != "path":
                man[p[0]] = int(p[1])
    need = ["NODE.json", "rows.parquet"] + [f"{s}.{e}" for s in node["shards"] for e in ("DONE", "index.npz", "codes_user.npy", "codes_assistant.npy",
                                                                                          "words.arrow", "persona.arrow")]
    miss, bad, nb = [], [], 0
    for f in need:
        p = f"{sroot}/{f}"
        if not os.path.exists(p):
            miss.append(f); continue
        sz = os.path.getsize(p); nb += sz
        if f in man and man[f] != sz:
            bad.append((f, sz, man[f]))
    out["native"] = {"root": sroot, "node": node["node"], "files": len(need), "missing": miss[:20], "n_missing": len(miss), "size_mismatch": bad[:20],
                     "n_size_mismatch": len(bad), "manifest": mp if man else None, "manifest_files_compared": int(sum(f in man for f in need)), "bytes": nb}
    for k, sl in enumerate(hp["slots"]):
        lst = {}
        for ln in open(f"{BUNDLE}/extra_slice_listings/{sl['node']}.LISTING.txt"):
            b, f = ln.split()
            lst[f] = int(b)
        miss, bad, nb = [], [], 0
        for f in sl["files"]:
            p = f"{sl['root']}/{f}"
            if not os.path.exists(p):
                miss.append(f); continue
            sz = os.path.getsize(p); nb += sz
            if lst.get(f) != sz:
                bad.append((f, sz, lst.get(f)))
        xn = D.R3.load_node(sl["root"])
        present = set(sl["shards_present"])
        rows = np.asarray([D.split_code(c)[1] for c in slots_codes.get(k + 1, [])], np.int64)
        sh_need = sorted({xn["shards"][int(i)] for i in xn["shard_of"][rows]}) if len(rows) else []
        out["extra"].append({"slot": k + 1, "node": sl["node"], "root": sl["root"], "files": len(sl["files"]), "n_missing": len(miss), "missing": miss[:20],
                             "n_size_mismatch": len(bad), "size_mismatch": bad[:20], "bytes": nb, "plan_bytes": sl["bytes"],
                             "shards_needed_by_codes": len(sh_need), "shards_needed_not_present": [s for s in sh_need if s not in present]})
    ok = out["native"]["n_missing"] == 0 and out["native"]["n_size_mismatch"] == 0 and all(
        x["n_missing"] == 0 and x["n_size_mismatch"] == 0 and not x["shards_needed_not_present"] for x in out["extra"])
    out["ok"] = bool(ok)
    return out


def host_mode(a):
    H = a.host
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    if a.src == "live":
        me = socket.gethostname().split(".")[0]
        assert me == H, ("live mode must run on the host itself", me, H)
        sroot = LIVE_ROOT
        if not os.path.exists(f"{sroot}/NODE.json"):
            ex = sorted(os.listdir(LIVE_EXTRA)) if os.path.isdir(LIVE_EXTRA) else []
            json.dump({"host": H, "src": "live", "status": "missing", "native_root": sroot, "native_present": False, "extra_present": ex,
                       "time": time.strftime("%F %T")}, open(f"{a.out}/{H}.live.json", "w"), indent=1)
            print(f"{H}: native store {sroot} MISSING (extra dirs present: {ex})")
            return
    else:
        sroot = f"{META}/{H}/s3-rows"
    XP = D.load_extra_plan(EXTRA_PLAN)
    assert XP["world"] == WORLD and XP["lw"] == LW and XP["eval_salt"] == EVAL_SALT and XP["eval_frac"] == EVAL_FRAC
    hp = json.loads(json.dumps(XP["hosts"][H]))
    if a.src == "meta":
        for sl in hp["slots"]:
            sl["root"] = f"{META}/extra/{sl['node']}"
    else:
        for sl in hp["slots"]:
            assert sl["root"] == f"{LIVE_EXTRA}/{sl['node']}", sl["root"]
    IDX = HOSTS.index(H)
    snode = D.R3.load_node(sroot)
    assert snode["node"] == H, ("speech store of another node", snode["node"], H)
    sidx = D.DS.load_index(sroot)
    tabs = {0: slot_tables(sroot)}
    for k, sl in enumerate(hp["slots"]):
        tabs[k + 1] = slot_tables(sl["root"])
        assert tabs[k + 1][0]["node"] == sl["node"]
    slot_nodes = [snode["node"]] + [sl["node"] for sl in hp["slots"]]
    stores = {"speech_store": [snode["node"], snode["digest"]], "extra_stores": [[sl["node"], tabs[k + 1][0]["digest"]] for k, sl in enumerate(hp["slots"])]}
    arrays, ranks = {}, {}
    slots_codes = {}
    for l in range(LW):
        rank = LW * IDX + l
        sp = D.speech_plan(snode, sidx, SPEECH_SEED, LW, l, EVAL_FRAC, EVAL_SALT, E_)
        native = int(len(sp["rows"]))
        codes, xs = D.d2_stream(np.asarray(sp["rows"], np.int64), sidx, hp, l, rank, S0, NSM, SALTS, EVAL_FRAC, SPEECH_SEED)
        codes = np.asarray(codes, np.int64)
        slot = codes >> D.SLOT_SHIFT
        row = codes & ((1 << D.SLOT_SHIFT) - 1)
        kh = np.zeros(len(codes), np.uint64); eh = np.zeros(len(codes), np.uint64); nf = np.zeros(len(codes), np.int32)
        gid = np.zeros(len(codes), np.int64)
        for k in np.unique(slot).tolist():
            m = slot == k
            kh[m] = tabs[k][2][row[m]]; eh[m] = tabs[k][3][row[m]]; nf[m] = tabs[k][4][row[m]]
            gid[m] = (np.int64(num(slot_nodes[k])) << np.int64(D.SLOT_SHIFT)) | row[m]
        assert len(set(codes.tolist())) == len(codes), "a code repeats in one rank's stream"
        ev = np.asarray(sp["eval_rows"], np.int64)                # eval rows: native (slot 0) row ids = their codes
        assert (snode["length"][ev] == FULL).all()
        arrays.update({f"r{l}_codes": codes, f"r{l}_kh": kh, f"r{l}_eh": eh, f"r{l}_nf": nf, f"r{l}_eval": ev, f"r{l}_evkh": tabs[0][2][ev],
                       f"r{l}_evkey": np.asarray([sidx["row_key"][int(i)] for i in ev])})
        nu = codes[STOP_STEP * NSM:]
        for k in np.unique(nu >> D.SLOT_SHIFT).tolist():
            slots_codes.setdefault(int(k), []).extend(nu[(nu >> D.SLOT_SHIFT) == k].tolist())
        ranks[l] = {"rank": rank, "native_full": native, "xs": xs, "n_stream": int(len(codes)), "speech_digest_lib": sp["digest"],
                    "eval_rows": ev.tolist(), "eval_keys": [sidx["row_key"][int(i)] for i in ev],   # (DS.plan eval_keys are pre-FULL-filter: not used)
                    "digest_speech_bundle": sha("speech", stores["speech_store"], stores["extra_stores"], gid[:PLAN_STEPS * NSM]),
                    "gid_sha_full": sha("gid", gid), "n_frames_not_full": int((nf != FULL).sum()),
                    "slot_counts_unseen": {slot_nodes[int(k)]: int(((nu >> D.SLOT_SHIFT) == k).sum()) for k in np.unique(nu >> D.SLOT_SHIFT).tolist()}}
    res = {"host": H, "src": a.src, "status": "ok", "prod_node_index": IDX, "native_root": sroot, "slots": slot_nodes[1:],
           "slot_roots": [sl["root"] for sl in hp["slots"]], "stores": stores, "ranks": ranks, "plan_s": round(time.time() - t0, 1)}
    if a.src == "live":
        res["files"] = file_checks(H, sroot, hp, slots_codes)
        if a.build:
            res["build"] = build_checks(sroot, hp, ranks, arrays, IDX, a.build, tabs)
    np.savez(f"{a.out}/{H}.{a.src}.npz", **arrays)
    json.dump(res, open(f"{a.out}/{H}.{a.src}.json", "w"), indent=1, default=int)
    print(f"{H} {a.src}: {sum(r['n_stream'] for r in ranks.values()):,} stream rows over 8 ranks; {time.time() - t0:.0f}s"
          + (f"; files ok {res['files']['ok']}" if "files" in res else "") + (f"; build {res['build']['ok']}" if "build" in res else ""))


def build_checks(sroot, hp, ranks, arrays, IDX, n, tabs):
    """build the first n unseen micros of every slot (over the host's ranks, stream order) with the trainer's MixedSpeechDS and the run's
    cfg / rank argument; check row key, length, light frames = every conversation frame (light_frac 1.0), and that the rank argument does not
    change the micro (speech rows seed light frames from row_key, data track: data_s3q.row_arrays frame_light path). + 1 eval micro."""
    out, ok = {"micros": []}, True
    seen = {}
    for l in range(LW):
        codes = arrays[f"r{l}_codes"][STOP_STEP * NSM:]
        rank = ranks[l]["rank"]
        ds = D.MixedSpeechDS(sroot, hp, [], CFG, rank)
        ds0 = D.MixedSpeechDS(sroot, hp, [], CFG, 0)
        for c in codes.tolist():
            k = c >> D.SLOT_SHIFT
            if seen.get(k, 0) >= n:
                continue
            seen[k] = seen.get(k, 0) + 1
            m = ds.get(c)
            pv = json.loads(m["prov"])
            key = tabs[k][1][c & ((1 << D.SLOT_SHIFT) - 1)]
            d_r, d_0 = micro_digest(m), micro_digest(ds0.get(c))
            nl = int(len(m["light"]))
            # light frames = every conversation frame f < nc (nc = 7500 - P after the 7,500 crop): P + CD + f
            nc = FULL - int(m["P"])
            good = pv["row_key"] == key and m["ids"].shape[1] == FULL + D.CD and d_r == d_0 and nl == nc
            ok &= bool(good)
            out["micros"].append({"local": l, "rank": rank, "code": int(c), "slot": int(k), "row_key": pv["row_key"], "key_ok": pv["row_key"] == key,
                                  "T": int(m["ids"].shape[1]), "P": int(m["P"]), "swap": int(m["swap"]), "voice_drawn": pv.get("voice_drawn"),
                                  "clips": len(pv.get("clips", [])), "persona_blocks": len(pv.get("persona", [])), "n_light": nl, "nc": nc,
                                  "digest_rank": d_r, "digest_rank0": d_0, "rank_independent": d_r == d_0, "ok": bool(good)})
        if all(seen.get(k, 0) >= n for k in range(len(hp["slots"]) + 1)):
            break
    l = 0
    ev = D.SpeechDS(sroot, ranks[l]["eval_rows"], CFG, ranks[l]["rank"])[0]
    pv = json.loads(ev["prov"])
    eok = pv["row_key"] == ranks[l]["eval_keys"][0]
    ok &= bool(eok)
    out["eval_micro"] = {"local": l, "rank": ranks[l]["rank"], "row": ranks[l]["eval_rows"][0], "row_key": pv["row_key"], "key_ok": bool(eok),
                         "T": int(ev["ids"].shape[1]), "swap": int(ev["swap"])}
    out["ok"] = bool(ok)
    return out


# ------------------------------------------------------------------------------------------------ merge mode
def logged_d2():
    """the run's own [data] D2 line + rank-0 native line (job LAST_JOB; every D2 job logged the same numbers) and the max step reached."""
    txt = open(f"{LOGS}/ppm-{LAST_JOB}.log").read()
    m = re.search(r"rows consumed before \(kept, in order\) ([\d,]+); native left \+ extra ([\d,]+) \+ ([\d,]+) shuffled per rank; holdout salts .*?"
                  r"dropped native ([\d,]+) \+ extra ([\d,]+) rows, ALREADY SEEN before step 170: ([\d,]+) rows; speech rows/rank ([\d,]+)\.\.([\d,]+) "
                  r"\(plan target ([\d,]+)\); this host's slots (.*)", txt)
    g = [x.replace(",", "") for x in m.groups()]
    r0 = re.search(r"speech node slinky-0: FULL train rows/rank ([\d,]+);.*steps available ([\d,]+); max_steps (\d+)", txt)
    cfg = json.loads(re.search(r"cfg (\{.*\})\n", txt).group(1))
    steps = []
    for f in os.listdir(LOGS):
        if re.fullmatch(r"ppm-\d+\.log", f):
            t = open(f"{LOGS}/{f}", errors="replace").read()
            if "run ppm-v1-j" in t:
                steps += [int(x) for x in re.findall(r"\] step (\d+)/17366", t)]
    stopped = re.search(r"STOPPED at step (\d+) \(clean exit\)", txt)
    return {"head": int(g[0]), "native_left": int(g[1]), "extra_kept": int(g[2]), "drop_native": int(g[3]), "drop_extra": int(g[4]),
            "seen_holdout": int(g[5]), "rows_min": int(g[6]), "rows_max": int(g[7]), "target": int(g[8]), "slinky0_slots": g[9].strip(),
            "rank0_rows": int(r0.group(1).replace(",", "")), "steps_avail": int(r0.group(2).replace(",", "")), "max_steps": int(r0.group(3)),
            "max_step_logged_any_job": max(steps), "stopped_at": int(stopped.group(1)) if stopped else None, "cfg": cfg}


def merge_mode(a):
    V, fails = {}, []

    def chk(name, cond, detail=None):
        V[name] = {"ok": bool(cond), **({"detail": detail} if detail is not None else {})}
        if not cond:
            fails.append(name)

    meta = {h: json.load(open(f"{a.dir}/{h}.meta.json")) for h in HOSTS}
    marr = {h: np.load(f"{a.dir}/{h}.meta.npz") for h in HOSTS}
    live, larr = {}, {}
    for h in HOSTS:
        p = f"{a.dir}/{h}.live.json"
        if os.path.exists(p):
            live[h] = json.load(open(p))
            if live[h]["status"] == "ok":
                larr[h] = np.load(f"{a.dir}/{h}.live.npz")
    # ---- live (node-local /scratch) == meta (Weka copies) for every live host
    lm = {}
    for h, L in live.items():
        if L["status"] != "ok":
            continue
        same = L["stores"] == meta[h]["stores"] and all(np.array_equal(larr[h][f"r{l}_{k}"], marr[h][f"r{l}_{k}"]) for l in range(LW)
                                                          for k in ("codes", "kh", "eh", "nf", "eval"))
        lm[h] = bool(same)
    chk("live_equals_meta", all(lm.values()) and len(lm) > 0, lm)
    # ---- run cfg + logged numbers
    LG = logged_d2()
    lc = LG.pop("cfg")
    chk("data_cfg_equals_logged", lc["data"] == CFG, {"logged": lc["data"]})
    chk("plan_keys_equal_logged", lc["speech_seed"] == SPEECH_SEED and lc["speech_eval_frac"] == EVAL_FRAC and lc["speech_eval_salt"] == EVAL_SALT
        and lc["eval_per_task"] == E_ and tuple(lc["ga_seq"]) == SEQ and lc["extra_plan"] == EXTRA_PLAN and lc["extra_from_step"] == S0
        and lc["train_exclude_salts"] == SALTS and lc["speech_root"] == LIVE_ROOT)
    chk("stop_step", LG["stopped_at"] == STOP_STEP and LG["max_step_logged_any_job"] == STOP_STEP and LG["max_steps"] == PLAN_STEPS,
        {k: LG[k] for k in ("stopped_at", "max_step_logged_any_job", "max_steps")})
    R = {}
    for h in HOSTS:
        for l in range(LW):
            x = meta[h]["ranks"][str(l)]
            R[x["rank"]] = (h, l, x)
    assert sorted(R) == list(range(WORLD))
    agg = {k: sum(R[r][2]["xs"][k] for r in R) for k in ("head", "native", "extra_kept", "drop_native", "drop_extra", "seen_holdout")}
    lens = [R[r][2]["n_stream"] for r in R]
    mine = {"head": agg["head"], "native_left": agg["native"] - agg["head"] - agg["drop_native"], "extra_kept": agg["extra_kept"],
            "drop_native": agg["drop_native"], "drop_extra": agg["drop_extra"], "seen_holdout": agg["seen_holdout"], "rows_min": min(lens),
            "rows_max": max(lens), "rank0_rows": R[0][2]["n_stream"], "steps_avail": min(lens) // NSM}
    chk("d2_numbers_equal_logged", all(mine[k] == LG[k] for k in mine), {"repro": mine, "logged": {k: LG[k] for k in mine}})
    XP = D.load_extra_plan(EXTRA_PLAN)
    chk("slinky0_slot_rows_equal_logged", LG["slinky0_slots"] == " ".join(f"{x['node']} {len(x['rows'])}" for x in XP["hosts"]["slinky-0"]["slots"])
        and LG["target"] == XP["target_rows_per_rank"])
    nc = {json.loads(ln)["host"]: json.loads(ln)["counts"] for ln in open(EXTRA_PLAN.replace(".json", ".native_counts.jsonl"))}
    chk("native_counts_equal_plan_input", all(meta[h]["ranks"][str(l)]["native_full"] == nc[h][l] for h in HOSTS for l in range(LW)))
    # ---- vs the Oct-15 regen bundle (rank_plan.npz / digest / meta), produced independently (crosschecked there against the trainer)
    RP = np.load(f"{BUNDLE}/rank_plan/rank_plan.npz")
    RD = json.load(open(f"{BUNDLE}/rank_plan/rank_plan_digest.json"))
    RM = json.load(open(f"{BUNDLE}/rank_plan/rank_plan_meta.json"))
    bad_g, bad_d, bad_m, bad_e = [], [], [], []
    for r in range(WORLD):
        h, l, x = R[r]
        codes = marr[h][f"r{l}_codes"]
        slot_nodes = [h] + meta[h]["slots"]
        sl = codes >> D.SLOT_SHIFT
        gid = np.asarray([(num(slot_nodes[int(s)]) << D.SLOT_SHIFT) | int(c & ((1 << D.SLOT_SHIFT) - 1)) for s, c in zip(sl, codes)], np.int64)
        if not np.array_equal(gid[:PLAN_STEPS * NSM], RP[f"r{r:03d}_speech"].reshape(-1)):
            bad_g.append(r)
        if x["digest_speech_bundle"] != RD["ranks"][str(r)]["speech"]:
            bad_d.append(r)
        rm = RM["ranks"][str(r)]
        if not (rm["host"] == h and rm["d2"] == x["xs"] and rm["speech_rows_total"] == x["n_stream"] and rm["speech_native_full"] == x["native_full"]
                and rm["speech_digest_lib"] == x["speech_digest_lib"]):
            bad_m.append(r)
        evg = (np.int64(num(h)) << np.int64(D.SLOT_SHIFT)) | marr[h][f"r{l}_eval"]
        if not np.array_equal(evg, RP[f"r{r:03d}_eval_speech"]):
            bad_e.append(r)
    chk("bundle_speech_gids_equal", not bad_g, {"ranks_compared": WORLD, "positions_per_rank": PLAN_STEPS * NSM, "mismatch_ranks": bad_g})
    chk("bundle_speech_digests_equal", not bad_d, {"mismatch_ranks": bad_d, "overall_bundle": RD["overall"]})
    chk("bundle_rank_meta_equal", not bad_m, {"mismatch_ranks": bad_m})
    chk("bundle_eval_speech_equal", not bad_e, {"mismatch_ranks": bad_e})
    # ---- consumed / unseen + row-key checks over ALL ranks
    nC = STOP_STEP * NSM
    ckh, ukh, ueh, ceh, evh = [], [], [], [], []
    per = {}
    for r in range(WORLD):
        h, l, x = R[r]
        A = marr[h]
        codes, kh, eh, nf = A[f"r{l}_codes"], A[f"r{l}_kh"], A[f"r{l}_eh"], A[f"r{l}_nf"]
        ckh.append(kh[:nC]); ceh.append(eh[:nC]); ukh.append(kh[nC:]); ueh.append(eh[nC:]); evh.append(A[f"r{l}_evkh"])
        per[r] = {"h": h, "l": l, "codes": codes, "nf": nf}
    ckh, ukh, ceh, ueh, evh = (np.concatenate(v) for v in (ckh, ukh, ceh, ueh, evh))
    cset = np.unique(ckh)
    hit = np.isin(ukh, cset)
    chk("unseen_keys_not_in_consumed_any_rank", not hit.any(), {"unseen_rows": int(len(ukh)), "consumed_rows": int(len(ckh)),
                                                               "consumed_unique_keys": int(len(cset)), "unseen_hits": int(hit.sum())})
    uu = np.unique(ukh)
    chk("unseen_keys_unique_across_ranks", len(uu) == len(ukh), {"unseen": int(len(ukh)), "unique": int(len(uu))})
    chk("eval_keys_not_trained_or_unseen", not np.isin(evh, ckh).any() and not np.isin(evh, ukh).any(), {"eval_rows": int(len(evh))})
    chk("all_rows_full_7500", all((per[r]["nf"] == FULL).all() for r in per))
    ep_overlap = float(np.isin(ueh, np.unique(ceh)).mean())
    V["note_unseen_rows_whose_episode_also_has_consumed_rows"] = round(ep_overlap, 4)
    V["note_consumed_keys_repeated"] = int(len(ckh) - len(cset))
    # ---- per-host output
    hosts_out, tot = {}, {"unseen_rows_all": 0, "unseen_rows_available": 0, "consumed_rows": 0, "unseen_frames_all": 0, "unseen_frames_available": 0}
    full_unseen = {}
    for h in HOSTS:
        L = live.get(h)
        avail = bool(L and L["status"] == "ok" and L["files"]["ok"] and L.get("build", {"ok": True})["ok"])
        ho = {"prod_node_index": HOSTS.index(h), "native_root": LIVE_ROOT, "extra_plan": EXTRA_PLAN, "host_plan_key": h,
              "extra_slots": [{"slot": k + 1, "node": sl["node"], "root": sl["root"]} for k, sl in enumerate(XP["hosts"][h]["slots"])],
              "available_now": avail, "live_status": (L or {}).get("status", "not_run"),
              "files_ok": (L or {}).get("files", {}).get("ok") if L and L["status"] == "ok" else None,
              "build_ok": (L or {}).get("build", {}).get("ok") if L and L["status"] == "ok" else None, "ranks": {}}
        if L and L["status"] != "ok":
            ho["missing_detail"] = {k: L[k] for k in ("native_root", "native_present", "extra_present")}
        nu_h = 0
        for l in range(LW):
            x = meta[h]["ranks"][str(l)]
            r = x["rank"]
            codes, nf = per[r]["codes"], per[r]["nf"]
            un = codes[nC:]
            full_unseen[f"r{r:03d}"] = un
            ho["ranks"][str(l)] = {"prod_rank": r, "n_stream": int(len(codes)), "n_consumed": int(nC), "n_unseen": int(len(un)),
                                   "n_unseen_within_plan": int(max(0, PLAN_STEPS * NSM - nC)), "n_unseen_past_plan": int(len(codes) - PLAN_STEPS * NSM),
                                   "unseen_by_slot": x["slot_counts_unseen"], "unseen_codes": un[:1500].tolist(),
                                   "eval_codes": [int(c) for c in x["eval_rows"]], "eval_row_keys": x["eval_keys"]}
            nu_h += len(un)
            tot["consumed_rows"] += int(nC)
            tot["unseen_rows_all"] += int(len(un)); tot["unseen_frames_all"] += int(nf[nC:].sum())
            if avail:
                tot["unseen_rows_available"] += int(len(un)); tot["unseen_frames_available"] += int(nf[nC:].sum())
        ho["n_unseen"] = int(nu_h)
        ho["unseen_hours"] = round(nu_h * FULL / FPS / 3600, 1)
        hosts_out[h] = ho
    tot["unseen_hours_all"] = round(tot.pop("unseen_frames_all") / FPS / 3600, 1)
    tot["unseen_hours_available"] = round(tot.pop("unseen_frames_available") / FPS / 3600, 1)
    tot["unseen_rows"] = tot["unseen_rows_available"]
    tot["unseen_hours"] = tot["unseen_hours_available"]
    tot["hosts_available"] = [h for h in HOSTS if hosts_out[h]["available_now"]]
    tot["hosts_unavailable"] = [h for h in HOSTS if not hosts_out[h]["available_now"]]
    tot["eval_rows"] = int(len(evh))
    V["live_host_checks"] = {h: ({"status": "ok", "files_ok": L["files"]["ok"], "native_files": L["files"]["native"]["files"],
                                  "native_manifest_compared": L["files"]["native"]["manifest_files_compared"],
                                  "extra": [{k: e[k] for k in ("node", "files", "n_missing", "n_size_mismatch", "shards_needed_not_present")} for e in L["files"]["extra"]],
                                  "build_ok": L.get("build", {}).get("ok"), "micros_built": len(L.get("build", {}).get("micros", [])),
                                  "rank_independent": all(m["rank_independent"] for m in L.get("build", {}).get("micros", []))}
                                 if L["status"] == "ok" else {"status": L["status"], "extra_present": L.get("extra_present")}) for h, L in live.items()}
    V["failed"] = fails
    loss = {k: lc[k] for k in ("w_task", "w_text", "w_cat_word", "w_tim", "w_nw", "w_eob", "w_cb0", "w_cb", "w_pcb0", "w_pcb", "dec_k", "dec_lr", "dec_floor",
                               "dec_sched", "dec_warmup_updates", "dec_clip", "dec_seed", "dec_micro_frames", "dec_mb_max", "dec_even", "eval_light_max")}
    loss["light_frac"] = lc["data"]["light_frac"]
    loss["history"] = "steps 0..398: dec_k 4, light_frac 0.1 (D30); from step 399 (D31) dec_k 8 disjoint chunks over ALL frames, light_frac 1.0"
    out = {"format": "si20k-podcast-plan-v1", "model": "ppm", "pretrain_run": RUN, "pretrain_jobs_last": f"ppm-v1-j{LAST_JOB}", "stop_step": STOP_STEP,
           "plan_steps": PLAN_STEPS, "prod_world": WORLD, "prod_hosts": HOSTS, "nsm": NSM, "ga_seq": list(SEQ), "d2_switch_step": S0,
           "eval_salt": EVAL_SALT, "eval_frac": EVAL_FRAC, "train_exclude_salts": SALTS, "speech_seed": SPEECH_SEED,
           "code_rule": "code = slot << 40 | node row; slot 0 = the host's own slice (native_root), slot k>=1 = extra_plan['hosts'][host]['slots'][k-1]",
           "cfg": CFG,
           "eval_cfg": {**CFG, "_note": "eval_speech = the same cfg and rank argument (data_ppm.SpeechDS(native_root, eval_rows, cfg, RANK)), random B side "
                                       "(side_of); eval_speech_A = the same rows with swap = 1 - side; decoder eval on <= 2048 light frames (randperm seed 1234)"},
           "loss": loss, "hosts": hosts_out, "totals": tot, "verification": V,
           "micro_recipe": ("import os, sys, json; os.environ['PPM_CD'] = '2'; sys.path.insert(0, '/data/home/amartyavarma/ppm/code'); import data_ppm as D\n"
                            "P = json.load(open(PLAN)); h = P['hosts'][host]; r = h['ranks'][str(j)]\n"
                            "hp = D.load_extra_plan(h['extra_plan'])['hosts'][h['host_plan_key']]\n"
                            "ds = D.MixedSpeechDS(h['native_root'], hp, [], P['cfg'], r['prod_rank'])\n"
                            "micro = ds.get(code)   # code from r['unseen_codes']; pretraining passed rank = the global prod RANK (8 x host index + local)."
                            " With crop_conv + a system prompt, data_s3q.row_arrays takes the frame_light path: light frames are seeded by "
                            "(light_seed, h64(row_key), data track), NOT by rank (verified: rank R and rank 0 give bit-identical micros). "
                            "For a light_frac 0.1 fine-tune pass {**P['cfg'], 'light_frac': 0.1} (same per-frame uniform draws, threshold 0.1)."),
           "files": {"npz_full_unseen": a.out.replace(".json", ".npz"), "tool": os.path.abspath(__file__), "host_outputs": os.path.abspath(a.dir)}}
    np.savez(a.out.replace(".json", ".npz"), **full_unseen)
    json.dump(out, open(a.out, "w"), indent=1, default=int)
    print(json.dumps({"totals": tot, "failed": fails, "per_host_unseen": {h: hosts_out[h]["n_unseen"] for h in HOSTS},
                      "available": {h: hosts_out[h]["available_now"] for h in HOSTS}, "episode_overlap": ep_overlap}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="mode", required=True)
    a1 = sp.add_parser("host"); a1.add_argument("--host", required=True); a1.add_argument("--src", choices=("live", "meta"), required=True)
    a1.add_argument("--out", required=True); a1.add_argument("--build", type=int, default=0)
    a2 = sp.add_parser("merge"); a2.add_argument("--dir", required=True); a2.add_argument("--out", required=True)
    a = ap.parse_args()
    host_mode(a) if a.mode == "host" else merge_mode(a)


if __name__ == "__main__":
    main()
