#!/usr/bin/env python
"""PODCAST PLAN for the INTL 14B next-arch fine-tune (si20k-vox-intl): the UNSEEN pretraining podcast rows of ppm-next (init w-6242).

Reproduces ppm-next's per-rank speech row stream EXACTLY as next-arch/code/train_ilp.py built it (the code ppm-next ran, imported read-only
from /data/home/amartyavarma/ppm/next-arch/code):
  splan  = data_ppm.speech_plan(node, index, PPM_SPEECH_SEED, LW 8, local, eval_frac 0.0005, salt ppm-v1, n_eval 2)      (native slice)
  stream = data_ppm.d2_stream(splan rows, index, extra_plan_d2[host], local, rank, 170, nsm 2, ['s3eval-v1'], 0.0005, seed)  (D2 codes)
  step t (0-based loop step) consumes stream[t * nsm : (t + 1) * nsm]; dcp-6242 = loop steps 0..6241 -> consumed = stream[:12484].
  eval_speech rows = splan['eval_rows'] (2 / rank, native slot 0), eval_speech_A = the same rows with the side flipped.

usage (CPU only, never on the login node; every python call goes through srun into a running job on a compute node):
  host   run ON a training host (srun --overlap into a job there): this host's 8 local ranks -> WORK/<host>.json.gz
         (+ /scratch file checks for every slice the stream reads + one real micro built per rank, il_check'd)
  build  run on any compute node after all 16 hosts: text plan + per-rank plan digests vs every logged digest / plan dump, overlap
         checks, -> PODCAST_PLAN_intl.json"""
import argparse
import gzip
import hashlib
import json
import os
import socket
import sys
import time

CODE = "/data/home/amartyavarma/ppm/next-arch/code"            # the code ppm-next ran (data_ppm / lib / chat_ppm), read-only
os.environ["PPM_CD"] = "2"
os.environ.pop("S3_MIN_FRAMES", None)
sys.path.insert(0, CODE)
import numpy as np  # noqa: E402
import data_ppm as D  # noqa: E402

HOSTS = "slinky-0,slinky-1,slinky-2,slinky-3,slinky-5,slinky-6,slinky-8,slinky-9,slinky-10,slinky-11,slinky-12,slinky-13,slinky-14,slinky-16,slinky-17,slinky-19".split(",")
NODE_SWAP = {"slinky-4": "slinky-14", "slinky-7": "slinky-5", "slinky-14": "slinky-19"}
SEED, FRAC, SALT, LW, WORLD, NSM, NTM, E_ = 20261011, 0.0005, "ppm-v1", 8, 128, 2, 1, 2
S0, EXCL, STOP, MAX_STEPS, PLAN_STEPS = 170, ["s3eval-v1"], 6242, 17366, 17366
CHAT_SEED, CHAT_SPLITS, TXT_CAP, TEXT_OFFSET, CHAT_OFFSET = 20261011, [2, 3], 7500, 0, 0
SPEECH_ROOT = "/scratch/amartyavarma/s3-rows"
EXTRA_PLAN = "/data/home/amartyavarma/ppm/configs/extra_plan_d2.json"
PLAN_DIR = "/data/home/amartyavarma/ppm/si20k-vox-intl/plan"
WORK = f"{PLAN_DIR}/podcast_intl_work"
OUT = f"{PLAN_DIR}/PODCAST_PLAN_intl.json"
LOGS = "/data/home/amartyavarma/ppm/next-arch/logs"
# the resolved data cfg every ppm-next job logged (cfg['data'], identical in jobs 22904 / 22905 / 22956 / 23277 / 23497)
CFG = D.default_cfg(seed=20261011, sp_cap=2000, persona_sides="both", light_frac=1.0, pose_light_frac=1.0, txt_cap=7500, crop_conv=True,
                    chat_side="assistant", chat_eob=True, pack_attn="causal")
# logged [plan] lines: start -> (combined digest[:16], rank-0 digest[:16], rank-0 first-step speech codes)
LOGGED = {0: ("7964360b90a9d30c", "bcb255f149a9fb5d", [225284, 227887]), 58: ("3e3f08b27ae41b80", "d319f1a5b9e9722b", [66825, 20846]),
          384: ("f1d0ae72a8395468", "34d77eebe33b0ac1", [53268, 214794]), 3565: ("c25e087d8dd67401", "ad3c207ad6eacab1", [166606, 57460]),
          6241: ("df9985f50581cc14", "e4a6482c34f69cbe", [259455, 1099511685810])}
DUMPS = {0: f"{LOGS}/aborted/plan_dump_first4-22899.json", 6241: f"{LOGS}/plan_dump_first4.json"}   # per-rank digests (22899 = same plan as 22904)
SHARD_FILES = ("codes_user.npy", "codes_assistant.npy", "words.arrow", "persona.arrow")


def il_check(m):
    """= train_ilp.il_check (interleave, read_pos v2) on one speech micro."""
    ids, pr = np.asarray(m["ids"]), np.asarray(m["present"])
    for c in (0, 1):
        assert (ids[c][pr[c] == 0] == D.PAD_ID).all(), ("interleave: a non-PAD id where channel", "AB"[c], "is absent")
    assert (np.asarray(m["depth"])[1][pr[1] == 0] < 0).all(), "interleave: channel-B codes where B is absent"
    sel, T_ = np.asarray(m["sel"]), ids.shape[1]
    st_ = np.cumsum([0] + [int(x) for x in m["segs"]])[:-1]
    assert (sel + 1 < T_).all() and not np.isin(sel + 1, st_).any(), "read_pos v2: a text read would cross a packed-segment boundary"


def stat_files(root, shards):
    miss, nbytes = [], 0
    for s in shards:
        for f in SHARD_FILES:
            p = f"{root}/{s}.{f}"
            try:
                st = os.stat(p)
                nbytes += st.st_size
                if st.st_size == 0:
                    miss.append(p + " (empty)")
            except FileNotFoundError:
                miss.append(p)
    return miss, nbytes


CENSUS = "/data/home/amartyavarma/scratch-probe/rowcount-1011/keys/s3-rows"   # 10-11 read-only census of every node's rows.parquet (row order)
XMETA = "/data/home/amartyavarma/ppm/xfer/meta"                                 # Weka copies of the 4 extra slices' NODE.json / rows.parquet / index


def census_index(node):
    """(node dict, index dict) of a slice from the census keys parquet (rows.parquet order): enough for speech_plan / d2_stream (length,
    node name for the crc32 seed, row_key, dataset, episode). Used only where the node-local store no longer exists."""
    import pyarrow.parquet as pq
    path = f"{CENSUS}/{node}.parquet"
    if not os.path.exists(path):                                  # slinky-14 was rebuilt (T3): its census lives under ppm/scan14
        path = f"/data/home/amartyavarma/ppm/scan14/keys/s3-rows/{node}.parquet"
    t = pq.read_table(path).to_pydict()
    pod, epi = [], []
    for ds, ep in zip(t["dataset"], t["ep"]):
        a, rest = ep.split("/", 1)
        b, c = rest.rsplit("/", 1)
        assert a == ds and f"{a}/{b}/{c}" == ep
        pod.append(b); epi.append(c)
    n = len(t["row_key"])
    idx = {"row": list(range(n)), "row_key": t["row_key"], "dataset": t["dataset"], "podcast_id": pod, "episode_id": epi, "n_frames": t["n_frames"]}
    return {"length": np.asarray(t["n_frames"], np.int32), "node": node, "digest": "census-keys"}, idx


def host_main(host=None):
    t0 = time.time()
    on_host = socket.gethostname().split(".")[0]
    host = host or on_host
    ni = HOSTS.index(host)
    import pyarrow.parquet as pq
    native_ok = host == on_host and os.path.exists(f"{SPEECH_ROOT}/NODE.json")
    XP = D.load_extra_plan(EXTRA_PLAN)
    assert XP["world"] == WORLD and XP["lw"] == LW and XP["eval_salt"] == SALT and XP["eval_frac"] == FRAC
    hplan = json.loads(json.dumps(XP["hosts"][host]))
    if native_ok:
        snode = D.R3.load_node(SPEECH_ROOT)
        assert snode["node"] == host or NODE_SWAP.get(snode["node"]) == host, ("speech store of another node", snode["node"], host)
        sidx = D.DS.load_index(SPEECH_ROOT)
        cn, ci = census_index(snode["node"])                     # census == live index (validates the census path used where a store is gone)
        census_match = ci["row_key"] == sidx["row_key"] and list(ci["n_frames"]) == list(sidx["n_frames"])
        del cn, ci
    else:
        nc = {json.loads(ln)["host"]: json.loads(ln)["node"] for ln in open("/data/home/amartyavarma/ppm/xfer/native_counts.jsonl")}
        snode, sidx = census_index(nc[host])
        census_match = None
    # slot tables (slot 0 = native, k >= 1 = extra slot k - 1): row_key / n_frames per node row; shard of each row
    keys, nfs, shard_name, present = {0: sidx["row_key"]}, {0: np.asarray(sidx["n_frames"], np.int64)}, {}, {}
    fcheck = {"native": {"root": SPEECH_ROOT, "node": snode["node"], "node_digest": snode["digest"], "store_present": native_ok,
                         "census_matches_live_index": census_match}}
    if native_ok:
        shard_name[0] = [snode["shards"][int(k)] for k in snode["shard_of"]]
        present[0] = set(snode["shards"])
        miss, nb = stat_files(SPEECH_ROOT, snode["shards"])
        fcheck["native"].update(shards=len(snode["shards"]), missing=miss[:20], n_missing=len(miss), bytes=nb)
    else:
        fcheck["native"].update(n_missing=-1, note=f"{SPEECH_ROOT} absent on {on_host}; plan from the census keys {CENSUS}/{snode['node']}.parquet")
    slot_ok = {}
    for k, sl in enumerate(hplan["slots"]):
        ok_ = host == on_host and os.path.exists(f"{sl['root']}/NODE.json")
        slot_ok[k + 1] = ok_
        root = sl["root"] if ok_ else f"{XMETA}/{sl['node']}"
        sl["root"] = root                                         # d2_stream reads the slot's rows.parquet (episode holdout) from here
        xn = D.R3.load_node(root)
        assert xn["node"] == sl["node"], ("extra slot holds another slice", root, xn["node"], sl["node"])
        t = pq.read_table(f"{root}/rows.parquet", columns=["row", "row_key", "n_frames"]).to_pydict()
        assert t["row"] == list(range(len(t["row"])))
        keys[k + 1], nfs[k + 1] = t["row_key"], np.asarray(t["n_frames"], np.int64)
        shard_name[k + 1] = [xn["shards"][int(j)] for j in xn["shard_of"]]
        present[k + 1] = set(sl["shards_present"])
        if ok_:
            miss, nb = stat_files(root, sl["shards_present"])
            lst = [f for f in sl["files"] if not os.path.exists(f"{root}/{f}")]
            fcheck[f"slot{k + 1}"] = {"root": root, "node": sl["node"], "store_present": True, "shards_present": len(sl["shards_present"]),
                                      "missing": miss[:20], "n_missing": len(miss), "bytes": nb, "plan_files": len(sl["files"]),
                                      "plan_files_missing": lst[:20], "n_plan_files_missing": len(lst)}
        else:
            fcheck[f"slot{k + 1}"] = {"root": XP["hosts"][host]["slots"][k]["root"], "node": sl["node"], "store_present": False, "n_missing": -1,
                                      "note": f"absent on {on_host}; keys from {root}"}
    out = {"host": host, "run_on": on_host, "prod_node_index": ni, "native_node": snode["node"], "native_root": SPEECH_ROOT, "extra_plan": EXTRA_PLAN,
           "store_present": native_ok and all(slot_ok.values()),
           "slots": [{"slot": k + 1, "node": sl["node"], "root": XP["hosts"][host]["slots"][k]["root"]} for k, sl in enumerate(hplan["slots"])],
           "files": fcheck, "ranks": {}}
    ds_chk = None
    for local in range(LW):
        rank = ni * LW + local
        splan = D.speech_plan(snode, sidx, SEED, LW, local, FRAC, SALT, E_)
        codes, xs = D.d2_stream(np.asarray(splan["rows"], np.int64), sidx, hplan, local, rank, S0, NSM, EXCL, FRAC, SEED)
        codes = [int(c) for c in codes]
        sl_, rr_ = [c >> D.SLOT_SHIFT for c in codes], [c & ((1 << D.SLOT_SHIFT) - 1) for c in codes]
        ks = [keys[s][r] for s, r in zip(sl_, rr_)]
        nf = [int(nfs[s][r]) for s, r in zip(sl_, rr_)]
        not_present = sum(1 for s, r in zip(sl_[STOP * NSM:], rr_[STOP * NSM:]) if shard_name[s][r] not in present[s]) if out["store_present"] else -1
        ev = [int(r) for r in splan["eval_rows"]]
        rec = {"prod_rank": rank, "codes": codes, "keys": ks, "n_frames": nf, "xs": xs, "n_native_plan": int(len(splan["rows"])),
               "eval_codes": [D.code_of(0, r) for r in ev], "eval_keys": [sidx["row_key"][r] for r in ev],
               "eval_n_frames": [int(sidx["n_frames"][r]) for r in ev], "eval_node_sha": splan["eval_node_sha"], "splan_digest": splan["digest"],
               "rows_shard_not_present": not_present}
        # one real micro per rank (first unseen code), exactly as the trainer's loader builds it (MixedSpeechDS, rank arg = prod rank)
        c0 = codes[STOP * NSM]
        try:
            assert out["store_present"], "store absent on this host"
            if ds_chk is None:
                ds_chk = D.MixedSpeechDS(SPEECH_ROOT, hplan, [], CFG, rank)
            ds_chk.rank = rank
            m = ds_chk.get(c0)
            il_check(m)
            ids, dp, P = np.asarray(m["ids"]), np.asarray(m["depth"]), int(m["P"])
            pv = json.loads(m["prov"])
            rec["micro_check"] = {"code": c0, "ok": True, "T": int(ids.shape[1]), "P": P, "swap": int(m["swap"]), "voice_drawn": pv.get("voice_drawn"),
                                  "clips": pv.get("clips"), "persona": pv.get("persona"), "A_clip_frames": int((dp[0, :P, 0] >= 0).sum()),
                                  "B_clip_frames": int((dp[1, :P, 0] >= 0).sum()), "A_lane_nonpad_prefix": int((ids[0, :P] != D.PAD_ID).sum()),
                                  "B_lane_nonpad_prefix": int((ids[1, :P] != D.PAD_ID).sum()), "n_sel": int(len(m["sel"])),
                                  "n_light": int(len(m["light"])), "n_cb0": int((np.asarray(m["cb0"]) >= 0).sum())}
        except Exception as e:  # noqa: BLE001
            rec["micro_check"] = {"code": c0, "ok": False, "error": repr(e)[:500]}
        out["ranks"][str(local)] = rec
        print(f"[{host}] local {local} rank {rank}: stream {len(codes):,} codes, xs {xs}, micro {rec['micro_check'].get('ok')} "
              f"({time.time() - t0:.0f}s)", flush=True)
    os.makedirs(WORK, exist_ok=True)
    tmp = f"{WORK}/{host}.json.gz.tmp"
    with gzip.open(tmp, "wt") as f:
        json.dump(out, f)
    os.replace(tmp, f"{WORK}/{host}.json.gz")
    print(f"[{host}] wrote {WORK}/{host}.json.gz in {time.time() - t0:.0f}s", flush=True)


# ------------------------------------------------------------------------------------------------ build
def text_plan_rank(rank, cpacks):
    """per optimizer step t of `rank`: (src, gemma pack or None, chat pack or None) = train_ilp's loaders (ntm 1, from step 0)."""
    mine = cpacks[CHAT_OFFSET + rank::WORLD]
    kc, out = {"gemma": 0, "chat": 0}, []
    for t in range(MAX_STEPS):
        s = "gemma" if D.h64(f"ppm-text-src|{CHAT_SEED}|{t}|{rank}") & 1 == 0 else "chat"
        if s == "gemma":
            out.append((s, TEXT_OFFSET + (kc[s] * NTM) * WORLD + rank, None))
        else:
            a, b = mine[(kc[s] * NTM) % len(mine)]
            out.append((s, None, [int(a), [int(z) for z in b]]))
        kc[s] += 1
    return out


def plan_digest(codes, tp, start):
    srows = codes[start * NSM:MAX_STEPS * NSM]
    gp = [g for s, g, _ in tp[start:] if s == "gemma"]
    cp = [c for s, _, c in tp[start:] if s == "chat"]
    src = [s for s, _, _ in tp[start:]]
    return hashlib.sha256(json.dumps({"speech": srows, "gemma": gp, "chat": cp, "src": src}).encode()).hexdigest()


def build_main():
    t0 = time.time()
    import chat_ppm as CH
    H = {}
    for h in HOSTS:
        with gzip.open(f"{WORK}/{h}.json.gz", "rt") as f:
            H[h] = json.load(f)
    R = {}
    for h in HOSTS:
        for lk, r in H[h]["ranks"].items():
            R[r["prod_rank"]] = dict(r, host=h, local=int(lk))
    assert sorted(R) == list(range(WORLD))
    ver = {}
    # ---- plan dumps: host per rank, n_speech_rows, first 4 steps' speech codes, per-rank digests
    dumps = {s: {d["rank"]: d for d in json.load(open(p))} for s, p in DUMPS.items()}
    cpacks = CH.multi_split_packs(CHAT_SPLITS, TXT_CAP, CHAT_SEED)
    print(f"[build] chat packs {len(cpacks):,} ({time.time() - t0:.0f}s)", flush=True)
    dg = {s: [None] * WORLD for s in LOGGED}
    bad = []
    for rk in range(WORLD):
        r = R[rk]
        tp = text_plan_rank(rk, cpacks)
        for s in LOGGED:
            dg[s][rk] = plan_digest(r["codes"], tp, s)
        for s, dd in dumps.items():
            d = dd[rk]
            if d["host"] != r["host"]:
                bad.append(("dump host", s, rk, d["host"], r["host"]))
            if d["n_speech_rows"] != len(r["codes"]):
                bad.append(("dump n_speech_rows", s, rk, d["n_speech_rows"], len(r["codes"])))
            if d["speech"] != r["codes"][s * NSM:(s + 4) * NSM]:
                bad.append(("dump speech first 4 steps", s, rk))
            if d["digest"] != dg[s][rk]:
                bad.append(("dump per-rank digest", s, rk, d["digest"][:16], dg[s][rk][:16]))
    for s, (comb, r0, sp0) in LOGGED.items():
        c_ = hashlib.sha256("".join(dg[s]).encode()).hexdigest()[:16]
        ver[f"logged_plan_start_{s}"] = {"combined_logged": comb, "combined_ours": c_, "rank0_logged": r0, "rank0_ours": dg[s][0][:16],
                                         "rank0_first_step_logged": sp0, "rank0_first_step_ours": R[0]["codes"][s * NSM:(s + 1) * NSM],
                                         "match": c_ == comb and dg[s][0][:16] == r0 and R[0]["codes"][s * NSM:(s + 1) * NSM] == sp0}
    ver["plan_dumps"] = {f"start_{s}": {"path": p, "ranks": WORLD} for s, p in DUMPS.items()}
    ver["plan_dump_mismatches"] = bad[:50]
    ver["n_plan_dump_mismatches"] = len(bad)
    # ---- D2 stream stats vs the logged [data] line (summed over ranks)
    xs_tot = {k: sum(R[rk]["xs"][k] for rk in range(WORLD)) for k in ("seen_holdout", "drop_native", "drop_extra", "extra_kept", "head", "native")}
    lens = [len(R[rk]["codes"]) for rk in range(WORLD)]
    ver["d2_stats"] = {"ours": xs_tot, "rows_per_rank_min_max": [min(lens), max(lens)],
                       "logged": {"head": 43520, "native_left": 3515305, "extra_kept": 888743, "drop_native": 1640, "drop_extra": 456,
                                  "seen_holdout": 19, "rows_per_rank_min_max": [34733, 34759]},
                       "match": xs_tot["head"] == 43520 and xs_tot["native"] - xs_tot["head"] - xs_tot["drop_native"] == 3515305
                       and xs_tot["extra_kept"] == 888743 and xs_tot["drop_native"] == 1640 and xs_tot["drop_extra"] == 456
                       and xs_tot["seen_holdout"] == 19 and [min(lens), max(lens)] == [34733, 34759]}
    ver["max_steps_from_streams"] = min(lens) // NSM
    # ---- consumed / unseen, overlap checks
    cons_keys, uns_keys, uns_dup = set(), {}, 0
    for rk in range(WORLD):
        for k in R[rk]["keys"][:STOP * NSM]:
            cons_keys.add(k)
    for rk in range(WORLD):
        for k in R[rk]["keys"][STOP * NSM:]:
            if k in uns_keys:
                uns_dup += 1
            uns_keys[k] = rk
    inter = [k for k in uns_keys if k in cons_keys]
    ev_keys = {k for rk in range(WORLD) for k in R[rk]["eval_keys"]}
    s3 = json.load(open("/data/home/amartyavarma/ppm/evalsets/speech_s3/MANIFEST.json"))
    s3k = set(s3["row_keys"])
    # episode-level holdouts: no unseen row from a ppm-v1 eval episode or an s3eval-v1 episode
    def ep(k):                                                    # row_key = dataset/podcast_id/episode_id/chunk -> episode key
        return k.rsplit("/", 1)[0]
    uns_ev = sum(1 for k in uns_keys if D.DS.is_eval_episode(ep(k), FRAC, SALT) or D.DS.is_eval_episode(ep(k), FRAC, "s3eval-v1"))
    uns_full = sum(1 for rk in range(WORLD) for n in R[rk]["n_frames"][STOP * NSM:] if n != 7500)
    ver["overlap"] = {"consumed_unique_keys": len(cons_keys), "unseen_unique_keys": len(uns_keys), "unseen_key_in_consumed_any_rank": len(inter),
                      "examples": inter[:5], "unseen_keys_repeated_across_positions": uns_dup, "unseen_in_eval_speech": len(ev_keys & set(uns_keys)),
                      "consumed_in_eval_speech": len(ev_keys & cons_keys), "unseen_in_speech_s3_103": len(s3k & set(uns_keys)),
                      "unseen_from_holdout_episodes(ppm-v1|s3eval-v1)": uns_ev, "unseen_not_full_7500": uns_full,
                      "ok": not inter and not (ev_keys & set(uns_keys)) and not (s3k & set(uns_keys)) and uns_ev == 0 and uns_dup == 0}
    # ---- files / micros on the hosts
    fchk, mchk = {}, []
    for h in HOSTS:
        fchk[h] = {k: {kk: v[kk] for kk in ("root", "node", "n_missing", "bytes") if kk in v} | ({"n_plan_files_missing": v["n_plan_files_missing"]} if "n_plan_files_missing" in v else {})
                   for k, v in H[h]["files"].items()}
        fchk[h]["rows_in_absent_shards"] = sum(r["rows_shard_not_present"] for r in H[h]["ranks"].values())
        mchk += [(r["prod_rank"], r["micro_check"]) for r in H[h]["ranks"].values()]
    ver["scratch_files"] = fchk
    avail = [h for h in HOSTS if H[h]["store_present"]]
    ver["hosts_store_present"] = avail
    ver["hosts_store_absent"] = [h for h in HOSTS if not H[h]["store_present"]]
    ver["scratch_ok_present_hosts"] = all(all(v.get("n_missing", 0) == 0 and v.get("n_plan_files_missing", 0) == 0 for k, v in fchk[h].items()
                                              if isinstance(v, dict)) and fchk[h]["rows_in_absent_shards"] == 0 for h in avail)
    ver["census_matches_live_index"] = {h: H[h]["files"]["native"].get("census_matches_live_index") for h in HOSTS}
    ver["micro_checks_ok"] = sum(1 for _, m in mchk if m.get("ok"))
    ver["micro_checks_expected"] = LW * len(avail)
    ver["micro_checks_failed"] = [(rk, m) for rk, m in mchk if not m.get("ok") and R[rk]["host"] in avail][:10]
    ver["micro_check_examples"] = [m for _, m in mchk if m.get("ok")][:3]
    ver["plan_reproduced_ok"] = (all(v["match"] for k, v in ver.items() if k.startswith("logged_plan_start_")) and not bad and ver["d2_stats"]["match"]
                                 and ver["overlap"]["ok"] and ver["max_steps_from_streams"] == PLAN_STEPS
                                 and all(x is not False for x in ver["census_matches_live_index"].values()))
    ver["available_ok"] = ver["scratch_ok_present_hosts"] and ver["micro_checks_ok"] == ver["micro_checks_expected"]
    # ---- the plan json
    hosts, tot_u, tot_c, tot_uf, tot_up, av_u, av_uf = {}, 0, 0, 0, 0, 0, 0
    for h in HOSTS:
        hh = H[h]
        ranks = {}
        for lk in sorted(hh["ranks"], key=int):
            r = hh["ranks"][lk]
            cu = r["codes"][STOP * NSM:]
            nf = r["n_frames"][STOP * NSM:]
            ranks[lk] = {"prod_rank": r["prod_rank"], "n_consumed": STOP * NSM, "n_unseen": len(cu),
                         "n_unseen_within_plan": PLAN_STEPS * NSM - STOP * NSM, "stream_len": len(r["codes"]),
                         "unseen_frames": int(sum(nf)), "unseen_codes": cu[:1500], "unseen_keys_head": r["keys"][STOP * NSM:STOP * NSM + 3],
                         "eval_codes": r["eval_codes"], "eval_keys": r["eval_keys"]}
            tot_u += len(cu); tot_c += STOP * NSM; tot_uf += int(sum(nf)); tot_up += PLAN_STEPS * NSM - STOP * NSM
            if hh["store_present"]:
                av_u += len(cu); av_uf += int(sum(nf))
        hosts[h] = {"prod_node_index": hh["prod_node_index"], "native_root": hh["native_root"], "native_node": hh["native_node"],
                    "extra_plan": hh["extra_plan"], "host_plan_key": h, "extra_slots": hh["slots"], "store_present": hh["store_present"],
                    "ranks": ranks,
                    "n_unseen": sum(v["n_unseen"] for v in ranks.values()), "unseen_hours": round(sum(v["unseen_frames"] for v in ranks.values()) / 12.5 / 3600, 1)}
    loss = {"w_text": 5.0, "w_cat_word": 1.0, "w_tim": 1.0, "w_nw": 1.0, "w_eob": 1.0, "w_cb0": 1.0, "w_cb": 1.0, "w_task_speech": 1.0,
            "ga_seq": ["speech", "speech", "text"], "dec_k": 8, "light_frac": 1.0, "dec_lr": 4e-4, "dec_floor": 1e-4, "dec_warmup_updates": 400,
            "dec_clip": 0.5, "dec_seed": 20261011, "dec_sched": "constant", "dec_layers": 14, "cb0_head": 1, "read_pos": "v2", "cd": 2,
            "formula": "per micro: loss = text + W_CB0 * cb0 (cb0 head on asst_text slot 3p+1, mean over B's coded frames); text = sum_S "
                       "(W_CAT_WORD*word + W_TIM*pad + W_EOB*eob + W_NW*nw) * CE / S + W_TEXT * word * normCE / S (S = scored B positions, read "
                       "user_text slot 3(sel+1)); backward (w_task * loss / NM); 0.6B decoder: DEC_K disjoint chunks of ALL light frames "
                       "(light_frac 1.0) per micro, own AdamW, its input grad x (w_task * W_CB / NM)"}
    plan = {"format": "si20k-podcast-plan-v1", "model": "intl", "pretrain_run": "ppm-next (jobs 22904 -> 22905 -> 22956 -> 23277 -> 23497; ckpt w-6242)",
            "stop_step": STOP, "plan_steps": PLAN_STEPS, "prod_world": WORLD, "prod_hosts": HOSTS, "nsm": NSM,
            "consumed_positions": f"stream[:{STOP * NSM}] per rank (loop steps 0..{STOP - 1}); unseen = stream[{STOP * NSM}:]",
            "code_format": "code = slot << 40 | node_row; slot 0 = native /scratch/amartyavarma/s3-rows, slot k>=1 = extra_plan hosts[host].slots[k-1].root",
            "cfg": CFG,
            "eval_cfg": dict(CFG, note="eval_speech = each rank's 2 splan eval rows (native slot 0) built with cfg (ILP_HIST=0 -> d31_from 0 -> no 0.1 "
                             "variant), random side side_of(row_key, seed); eval_speech_A = same rows, swap flipped; decoder eval CE on <= 2048 light "
                             "frames (torch.randperm seed 1234); every 30 min; speech_s3 = 103 pre-built Demeter rows (separate)"),
            "plan_cfg": {"speech_seed": SEED, "eval_frac": FRAC, "eval_salt": SALT, "lw": LW, "n_eval_per_rank": E_, "extra_plan": EXTRA_PLAN,
                         "extra_from_step": S0, "train_exclude_salts": EXCL, "node_swap": NODE_SWAP, "crop_full_rows_only": True},
            "loss": loss, "hosts": hosts,
            "eval_speech": {"rows_per_rank": E_, "total_rows": E_ * WORLD, "unique_keys": len(ev_keys)},
            "totals": {"unseen_rows": tot_u, "unseen_hours": round(tot_uf / 12.5 / 3600, 1), "consumed_rows": tot_c,
                       "unseen_rows_on_disk_now": av_u, "unseen_hours_on_disk_now": round(av_uf / 12.5 / 3600, 1),
                       "hosts_store_absent": ver["hosts_store_absent"],
                       "unseen_rows_within_17366_plan": tot_up, "unseen_hours_within_plan": round(tot_up * 7500 / 12.5 / 3600, 1)},
            "verification": ver,
            "full_streams": {h: f"{WORK}/{h}.json.gz" for h in HOSTS},
            "micro_recipe": MICRO_RECIPE}
    tmp = OUT + ".tmp"
    json.dump(plan, open(tmp, "w"))
    os.replace(tmp, OUT)
    print(json.dumps({"totals": plan["totals"], "per_host": {h: (hosts[h]["n_unseen"], hosts[h]["store_present"]) for h in HOSTS},
                      "plan_reproduced_ok": ver["plan_reproduced_ok"], "available_ok": ver["available_ok"],
                      "logged": {k: v["match"] for k, v in ver.items() if k.startswith("logged_plan_start_")}, "dump_mismatch": len(bad),
                      "d2": ver["d2_stats"]["match"], "overlap": ver["overlap"], "scratch_ok": ver["scratch_ok_present_hosts"],
                      "micros_ok": [ver["micro_checks_ok"], ver["micro_checks_expected"]], "census": ver["census_matches_live_index"]}, indent=1))
    print(f"[build] wrote {OUT} ({time.time() - t0:.0f}s)")


MICRO_RECIPE = r'''
# one podcast micro exactly as ppm-next's loader built it (train_ilp: StepDS(post = MixedSpeechDS(SPEECH_ROOT, hplan, rows, dcfg, RANK)) since
# ILP_HIST=0 -> d31_from 0 -> every step reads the env cfg). Run on the host that holds the code (its /scratch); CPU only.
import os, sys, json
os.environ["PPM_CD"] = "2"                                   # speech_q.CD (code delay 2), set before importing data_ppm
sys.path.insert(0, "/data/home/amartyavarma/ppm/si20k-vox-intl/code")   # data_ppm.py + lib/* byte-identical to next-arch/code (ppm-next)
import numpy as np, data_ppm as D
plan = json.load(open("/data/home/amartyavarma/ppm/si20k-vox-intl/plan/PODCAST_PLAN_intl.json"))
host, local = "slinky-0", "0"                                # this host + one of its 8 local ranks
H = plan["hosts"][host]; R = H["ranks"][local]
cfg = dict(plan["cfg"])                                      # = D.default_cfg(seed=20261011, sp_cap=2000, persona_sides="both", light_frac=1.0,
                                                             #   pose_light_frac=1.0, txt_cap=7500, crop_conv=True, chat_side="assistant",
                                                             #   chat_eob=True, pack_attn="causal")
hplan = D.load_extra_plan(H["extra_plan"])["hosts"][H["host_plan_key"]]
ds = D.MixedSpeechDS(H["native_root"], hplan, R["unseen_codes"], cfg, R["prod_rank"])   # rank arg = the PRETRAINING global rank
m = ds[0]                    # = D.speech_micro(row, node_row, sib, persona, cfg, rank): random side side_of(row_key, 20261011), 1/3 rows get
                             #   voice clips (later 2nd-half chunk of the same recording, lane blanked), personas of both sides, P <= 2000,
                             #   conversation cropped so T = 7500 + CD; light frames = per-frame rule seeded (light_seed, row_key, data track)
                             #   -> the rank argument does NOT change a speech micro (frame_light path); it is passed for fidelity only.
# train_ilp.il_check (interleave, read_pos v2), then prep():
ids, pr = m["ids"], m["present"]
for c in (0, 1): assert (ids[c][pr[c] == 0] == D.PAD_ID).all()
assert (m["depth"][1][pr[1] == 0] < 0).all()
sel = m["sel"]; assert (sel + 1 < ids.shape[1]).all()        # one segment [T]: no cross-segment read
# INTERLEAVED 14B INPUT (model_ilp.InpIL, ILP_MODE=interleave): row position p -> backbone positions 3p, 3p+1, 3p+2 =
#   [emb_A(ids[0, p]) (A = partner word lane incl. A's persona text; PAD where A absent / A clip lane blanked),
#    emb_B(ids[1, p]) (B word lane incl. B persona), aud_B(depth[1, p]) (sum of B's 32 Mimi code embeddings at CD 2; 0 where no codes)]
#   depth[0] (A's Mimi: conversation AND A's voice clip) has NO input slot (inp.aud.0 frozen, unused) -> A's voice clip is invisible except
#   as emb_A(PAD) positions; B's voice clip codes enter via aud_B. RoPE: positions (3p + k) / 3 (PPM_ROPE_POS_DIV=3).
# loss weights (prep): wt = (W_CAT_WORD*word + W_TIM*pad + W_EOB*eob + W_NW*nw) / n_text ; wn = W_TEXT*word / n_text ; cb0_w = mask / sum
'''

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("host", "build"))
    ap.add_argument("--host", default=None, help="host mode: plan this host (default: the host it runs on)")
    a = ap.parse_args()
    host_main(a.host) if a.mode == "host" else build_main()
