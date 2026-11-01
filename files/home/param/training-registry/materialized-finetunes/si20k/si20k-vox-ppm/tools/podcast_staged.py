#!/usr/bin/env python
"""si20k fine-tunes: STAGED unseen podcast rows for the 3 hosts whose pretraining slices are gone (slinky-0/1/2).

Each target T gets a PARTIAL copy of one donor slice host's native s3-rows slice (NODE.json / rows.parquet / every shard's DONE + index.npz /
the chosen shards' codes_user / codes_assistant / words / persona) at /scratch/amartyavarma/si20k-pod/<donor>-slice, read as extra slot 1
(code = 1 << 40 | donor node row; data_ppm.PartSibIndex keeps the full slice's row numbering). Rows = the donor ranks' PPM-unseen stream
(ppm-v1, after step 12,829) at unseen positions >= 400 (positions < 400 = the slice host's own fine-tune rows), slot 0 only, in the chosen
shards; target local rank j <- donor local rank j, first 420 in stream order. CPU only; never on the login node.

modes
  select   --donors a,b,c,...            per candidate donor: greedy minimal shard set; global key checks -> WORK/select.json
  listing  --donor D --shards-json F --out FILE   (ON the donor) bytes + name of every file to stage -> FILE
  verify   --target T                     (ON T) sizes vs listing, 8 real micros (MixedSpeechDS, light_frac 0.1), il checks -> WORK/verify-T.json
  extra    --map T:D,...                  extra_plan_si20k_staged.json (one slot per target) from select.json + LISTING-<donor>.txt
  write    --map T:D,...                  PODCAST_STAGED.json (with verification) from select.json + verify-*.json
"""
import argparse
import gzip
import json
import os
import socket
import sys
import time

os.environ["PPM_CD"] = "2"
CODE = "/data/home/amartyavarma/ppm/si20k-vox-ppm/code"
for p in (CODE, f"{CODE}/lib"):
    if p not in sys.path:
        sys.path.insert(0, p)
import numpy as np  # noqa: E402
import data_ppm as D  # noqa: E402

PLAN_DIR = "/data/home/amartyavarma/ppm/si20k-vox-ppm/plan"
PPM_PLAN = f"{PLAN_DIR}/PODCAST_PLAN_ppm.json"
PPM_HOSTS_DIR = f"{PLAN_DIR}/podcast_hosts"
META = f"{PLAN_DIR}/_ppmv1_meta/meta"
INTL_PLAN = "/data/home/amartyavarma/ppm/si20k-vox-intl/plan/PODCAST_PLAN_intl.json"
INTL_WORK = "/data/home/amartyavarma/ppm/si20k-vox-intl/plan/podcast_intl_work"
S3_MANIFEST = "/data/home/amartyavarma/ppm/evalsets/speech_s3/MANIFEST.json"
WORK = f"{PLAN_DIR}/staged_work"
EXTRA_OUT = f"{PLAN_DIR}/extra_plan_si20k_staged.json"
OVERLAY_OUT = f"{PLAN_DIR}/PODCAST_STAGED.json"
STAGE_ROOT = "/scratch/amartyavarma/si20k-pod"
LIVE_ROOT = "/scratch/amartyavarma/s3-rows"
TARGETS = ["slinky-0", "slinky-1", "slinky-2"]
SLICE_HOSTS = ["slinky-3", "slinky-5", "slinky-6", "slinky-8", "slinky-9", "slinky-10", "slinky-11", "slinky-12", "slinky-13", "slinky-14",
               "slinky-16", "slinky-17", "slinky-19"]
HOSTS = ["slinky-0", "slinky-1", "slinky-2"] + SLICE_HOSTS
HOSTS = sorted(HOSTS, key=lambda h: int(h.split("-")[1]))
LW, NSM = 8, 2
PPM_STOP, INTL_STOP = 12829, 6242
NC_PPM, NC_INTL = PPM_STOP * NSM, INTL_STOP * NSM
SKIP, PER_RANK = 400, 420
FULL = 7500
EVAL_FRAC, SALTS = 0.0005, ["ppm-v1", "s3eval-v1"]
SHARD_FILES = ("codes_user.npy", "codes_assistant.npy", "words.arrow", "persona.arrow")
SMALL = ("NODE.json", "NODE.set1.json", "rows.parquet")


def h64(s):
    import hashlib
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "little")


def il_check(m):
    """= train_ilp.il_check (interleave, read_pos v2) on one speech micro (copied from podcast_plan_intl.py)."""
    ids, pr = np.asarray(m["ids"]), np.asarray(m["present"])
    for c in (0, 1):
        assert (ids[c][pr[c] == 0] == D.PAD_ID).all(), ("interleave: a non-PAD id where channel", "AB"[c], "is absent")
    assert (np.asarray(m["depth"])[1][pr[1] == 0] < 0).all(), "interleave: channel-B codes where B is absent"
    sel, T_ = np.asarray(m["sel"]), ids.shape[1]
    st_ = np.cumsum([0] + [int(x) for x in m["segs"]])[:-1]
    assert (sel + 1 < T_).all() and not np.isin(sel + 1, st_).any(), "read_pos v2: a text read would cross a packed-segment boundary"


def slice_tab(root):
    import pyarrow.parquet as pq
    node = D.R3.load_node(root)
    t = pq.read_table(f"{root}/rows.parquet", columns=["row", "row_key", "dataset", "podcast_id", "episode_id", "n_frames"]).to_pydict()
    assert t["row"] == list(range(len(t["row"]))), root
    assert np.array_equal(np.asarray(t["n_frames"], np.int32), node["length"]), root
    return node, t


# ------------------------------------------------------------------------------------------------ select
def select_mode(a):
    t0 = time.time()
    os.makedirs(WORK, exist_ok=True)
    P = json.load(open(PPM_PLAN))
    assert P["stop_step"] == PPM_STOP and P["nsm"] == NSM and P["prod_hosts"] == HOSTS
    meta = {h: np.load(f"{PPM_HOSTS_DIR}/{h}.meta.npz") for h in HOSTS}
    mj = {h: json.load(open(f"{PPM_HOSTS_DIR}/{h}.meta.json")) for h in HOSTS}
    # PPM consumed key hashes (all ranks), PPM eval key hashes, PPM first-400 unseen key hashes of every rank
    c_ppm = np.unique(np.concatenate([meta[h][f"r{l}_kh"][:NC_PPM] for h in HOSTS for l in range(LW)]))
    ev_ppm = np.unique(np.concatenate([meta[h][f"r{l}_evkh"] for h in HOSTS for l in range(LW)]))
    first400 = np.unique(np.concatenate([meta[h][f"r{l}_kh"][NC_PPM:NC_PPM + SKIP] for h in SLICE_HOSTS for l in range(LW)]))
    print(f"[select] ppm consumed {len(c_ppm):,} unique keys, eval {len(ev_ppm)}, first-400 {len(first400):,} ({time.time() - t0:.0f}s)", flush=True)
    # INTL: stream == PPM stream per rank (=> INTL consumed = prefix of PPM consumed), consumed key strings, eval keys, first-400 unseen keys
    intl_same, c_intl, ev_intl, f400_intl = {}, set(), set(), set()
    for h in HOSTS:
        with gzip.open(f"{INTL_WORK}/{h}.json.gz", "rt") as f:
            X = json.load(f)
        for lk, r in X["ranks"].items():
            l = int(lk)
            assert r["prod_rank"] == mj[h]["ranks"][lk]["rank"]
            intl_same[r["prod_rank"]] = bool(np.array_equal(np.asarray(r["codes"], np.int64), meta[h][f"r{l}_codes"]))
            c_intl.update(r["keys"][:NC_INTL])
            ev_intl.update(r["eval_keys"])
            if h in SLICE_HOSTS:
                f400_intl.update(r["keys"][NC_INTL:NC_INTL + SKIP])
        del X
    print(f"[select] intl streams == ppm streams on {sum(intl_same.values())}/{len(intl_same)} ranks; intl consumed {len(c_intl):,} "
          f"({time.time() - t0:.0f}s)", flush=True)
    c_intl_h = np.unique(np.fromiter((h64(k) for k in c_intl), np.uint64, len(c_intl)))
    intl_prefix_of_ppm = bool(np.isin(c_intl_h, c_ppm).all())
    s3k = set(json.load(open(S3_MANIFEST))["row_keys"])
    out = {"time": time.strftime("%F %T"), "global": {"ppm_consumed_unique": int(len(c_ppm)), "intl_consumed_unique": int(len(c_intl)),
                                                       "intl_stream_equals_ppm_stream_ranks": int(sum(intl_same.values())),
                                                       "intl_stream_mismatch_ranks": [r for r, v in intl_same.items() if not v],
                                                       "intl_consumed_subset_of_ppm_consumed": intl_prefix_of_ppm,
                                                       "ppm_eval_keys": int(len(ev_ppm)), "intl_eval_keys": len(ev_intl), "speech_s3_keys": len(s3k)},
           "donors": {}}
    for dn in a.donors.split(","):
        node, t = slice_tab(f"{META}/{dn}/s3-rows")
        assert node["node"] == dn
        shard_of = node["shard_of"]
        nS = len(node["shards"])
        elig = {}
        cnt = np.zeros((nS, LW), np.int64)
        for l in range(LW):
            codes = meta[dn][f"r{l}_codes"]
            un = codes[NC_PPM:]
            pos = np.arange(len(un))
            m = (pos >= SKIP) & ((un >> D.SLOT_SHIFT) == 0)
            rows = un[m].astype(np.int64)
            elig[l] = (pos[m] + NC_PPM, rows)
            np.add.at(cnt[:, l], shard_of[rows], 1)
        # greedy: shard with the largest useful coverage toward 420 per rank (tie: more rows total, then fewer rows in the shard = fewer bytes)
        cov = np.zeros(LW, np.int64)
        sel = []
        while (cov < PER_RANK).any():
            need = np.maximum(0, PER_RANK - cov)
            gain = np.minimum(cnt, need[None, :]).sum(1)
            gain[sel] = -1
            best = max(range(nS), key=lambda s: (gain[s], cnt[s].sum(), -node["meta"]["rows"][s]))
            assert gain[best] > 0, (dn, "cannot reach 420 per rank")
            sel.append(best); cov += cnt[best]
        for s in sorted(sel, key=lambda s: cnt[s].sum()):          # prune redundant shards
            rest = [x for x in sel if x != s]
            if (cnt[rest].sum(0) >= PER_RANK).all():
                sel = rest
        sel = sorted(sel)
        # pooled lower bound (any rank's rows to any target rank)
        tot_sorted = np.sort(cnt.sum(1))[::-1]
        pooled = int(np.searchsorted(tot_sorted.cumsum(), LW * PER_RANK) + 1)
        selset = set(sel)
        ranks = {}
        allkeys = []
        for l in range(LW):
            pos, rows = elig[l]
            keep = np.array([int(shard_of[r]) in selset for r in rows], bool)
            pos, rows = pos[keep][:PER_RANK], rows[keep][:PER_RANK]
            assert len(rows) == PER_RANK
            keys = [t["row_key"][int(r)] for r in rows]
            allkeys += keys
            ranks[str(l)] = {"donor_prod_rank": mj[dn]["ranks"][str(l)]["rank"], "stream_pos": pos.tolist(), "rows": rows.tolist(), "row_keys": keys,
                             "n_frames_full": bool(all(t["n_frames"][int(r)] == FULL for r in rows)),
                             "stream_pos_min": int(pos.min()), "unseen_pos_min": int(pos.min() - NC_PPM)}
        kh = np.fromiter((h64(k) for k in allkeys), np.uint64, len(allkeys))
        eps = [D.DS.episode_key(t["dataset"][int(r)], t["podcast_id"][int(r)], t["episode_id"][int(r)])
               for l in range(LW) for r in ranks[str(l)]["rows"]]
        # recordings never span shards: every row of each staged row's recording (episode) lies in a chosen shard
        rk_all = {}
        for i, k in enumerate(t["row_key"]):
            rk_all.setdefault(D.SPR.rec_key(k), set()).add(int(shard_of[i]))
        rec_ok = all(rk_all[D.SPR.rec_key(k)] <= selset for k in allkeys)
        chk = {"keys_unique": len(set(allkeys)) == len(allkeys),
               "not_in_ppm_consumed": int(np.isin(kh, c_ppm).sum()) == 0,
               "not_in_intl_consumed": not (set(allkeys) & c_intl),
               "not_in_first400_any_slice_host_ppm": int(np.isin(kh, first400).sum()) == 0,
               "not_in_first400_any_slice_host_intl": not (set(allkeys) & f400_intl),
               "not_in_ppm_eval": int(np.isin(kh, ev_ppm).sum()) == 0, "not_in_intl_eval": not (set(allkeys) & ev_intl),
               "not_in_speech_s3": not (set(allkeys) & s3k),
               "no_holdout_episode": not any(D.DS.is_eval_episode(e, EVAL_FRAC, s) for e in eps for s in SALTS),
               "all_full_7500": all(ranks[str(l)]["n_frames_full"] for l in range(LW)),
               "recordings_inside_chosen_shards": bool(rec_ok),
               "donor_ranks_intl_stream_equal": all(intl_same[mj[dn]["ranks"][str(l)]["rank"]] for l in range(LW))}
        out["donors"][dn] = {"shards": [node["shards"][s] for s in sel], "n_shards": len(sel), "pooled_lower_bound_shards": pooled,
                             "shard_rows": [int(node["meta"]["rows"][s]) for s in sel], "shard_eligible": [int(cnt[s].sum()) for s in sel],
                             "eligible_in_shards_per_rank": cnt[sel].sum(0).tolist(), "eligible_total_per_rank": cnt.sum(0).tolist(),
                             "node_digest": node["digest"], "checks": chk, "checks_ok": all(chk.values()), "ranks": ranks}
        print(f"[select] {dn}: {len(sel)} shards (pooled LB {pooled}) rows {sum(out['donors'][dn]['shard_rows']):,} checks {all(chk.values())} "
              f"{ {k: v for k, v in chk.items() if not v} } ({time.time() - t0:.0f}s)", flush=True)
    json.dump(out, open(f"{WORK}/select.json", "w"), indent=1)
    print(f"[select] wrote {WORK}/select.json")


# ------------------------------------------------------------------------------------------------ listing (on the donor)
def listing_mode(a):
    me = socket.gethostname().split(".")[0]
    assert me == a.donor, (me, a.donor)
    node = D.R3.load_node(LIVE_ROOT)
    assert node["node"] == a.donor
    shards = json.load(open(a.shards_json))["donors"][a.donor]["shards"]
    assert set(shards) <= set(node["shards"])
    files = [f for f in SMALL if os.path.exists(f"{LIVE_ROOT}/{f}")] + [f"{s}.{e}" for s in node["shards"] for e in ("DONE", "index.npz")] + \
        [f"{s}.{e}" for s in shards for e in SHARD_FILES]
    with open(a.out, "w") as f:
        for x in files:
            f.write(f"{os.path.getsize(f'{LIVE_ROOT}/{x}')} {x}\n")
    print(f"{len(files)} files, {sum(os.path.getsize(f'{LIVE_ROOT}/{x}') for x in files) / 1e9:.2f} GB")


# ------------------------------------------------------------------------------------------------ verify (on the target)
def verify_mode(a):
    T = a.target
    me = socket.gethostname().split(".")[0]
    assert me == T, (me, T)
    S = json.load(open(f"{WORK}/select.json"))
    dn = a.donor
    sd = S["donors"][dn]
    root = f"{STAGE_ROOT}/{dn}-slice"
    lst = [ln.split() for ln in open(f"{WORK}/LISTING-{dn}.txt")]
    miss = [f for _, f in lst if not os.path.exists(f"{root}/{f}")]
    bad = [(f, int(b), os.path.getsize(f"{root}/{f}")) for b, f in lst if os.path.exists(f"{root}/{f}") and os.path.getsize(f"{root}/{f}") != int(b)]
    extra_files = sorted(set(os.listdir(root)) - {f for _, f in lst})
    nbytes = sum(int(b) for b, _ in lst)
    XP = D.load_extra_plan(EXTRA_OUT)
    hp = XP["hosts"][T]
    assert len(hp["slots"]) == 1 and hp["slots"][0]["node"] == dn and hp["slots"][0]["root"] == root and hp["slots"][0]["shards_present"] == sd["shards"]
    P = json.load(open(PPM_PLAN))
    cfg = {**P["cfg"], "light_frac": 0.1}
    node, t = slice_tab(root)
    assert node["node"] == dn and node["digest"] == sd["node_digest"]
    ds = D.MixedSpeechDS(root, hp, [], cfg, 0)
    micros, ok = [], not miss and not bad
    for j in range(LW):
        r = sd["ranks"][str(j)]
        for which in (0, PER_RANK - 1):                       # first + last staged row of each rank
            row = int(r["rows"][which])
            code = D.code_of(1, row)
            rec = {"rank": j, "idx": which, "code": code, "row": row}
            try:
                m = ds.get(code)
                il_check(m)
                pv = json.loads(m["prov"])
                ids, dp = np.asarray(m["ids"]), np.asarray(m["depth"])
                P_ = int(m["P"])
                good = int(ids.shape[1]) == FULL + D.CD and pv["row_key"] == r["row_keys"][which] and m["kind"] == "speech"
                rec.update(ok=bool(good), T=int(ids.shape[1]), P=P_, swap=int(m["swap"]), row_key=pv["row_key"], key_ok=pv["row_key"] == r["row_keys"][which],
                           voice_drawn=pv.get("voice_drawn"), clips=len(pv.get("clips", []) or []), persona_blocks=len(pv.get("persona", []) or []),
                           n_sel=int(len(m["sel"])), n_light=int(len(m["light"])), n_conv=FULL - P_,
                           light_frac_obs=round(len(m["light"]) / max(1, FULL - P_), 3), n_cb0=int((np.asarray(m["cb0"]) >= 0).sum()),
                           absent_pad_ok=True)
            except Exception as e:  # noqa: BLE001
                rec.update(ok=False, error=repr(e)[:400])
            ok &= bool(rec["ok"])
            micros.append(rec)
    # every staged code resolves to its recorded row key through the partial store's rows
    key_mis = 0
    for j in range(LW):
        r = sd["ranks"][str(j)]
        for row, k in zip(r["rows"], r["row_keys"]):
            if t["row_key"][int(row)] != k or str(node["shards"][int(node["shard_of"][int(row)])]) not in sd["shards"]:
                key_mis += 1
    ok &= key_mis == 0
    res = {"target": T, "donor": dn, "root": root, "files": len(lst), "missing": miss[:20], "size_mismatch": bad[:20], "unlisted_files": extra_files[:20],
           "bytes": nbytes, "micros": micros, "n_micros_ok": sum(m["ok"] for m in micros), "codes_key_or_shard_mismatch": key_mis,
           "cfg_light_frac": cfg["light_frac"], "ok": bool(ok), "time": time.strftime("%F %T")}
    json.dump(res, open(f"{WORK}/verify-{T}.json", "w"), indent=1, default=int)
    print(json.dumps({k: v for k, v in res.items() if k != "micros"}, default=int))
    print([(m["rank"], m["idx"], m.get("T"), m.get("P"), m.get("light_frac_obs"), m["ok"]) for m in micros])


# ------------------------------------------------------------------------------------------------ write
def extra_mode(a):
    """the extra plan (one slot per target = the staged partial slice); needs select.json + the donors' LISTING files."""
    S = json.load(open(f"{WORK}/select.json"))
    mp = dict(x.split(":") for x in a.map.split(","))
    xp = {"format": "ppm-extra-plan-v1", "purpose": "si20k fine-tunes: staged PPM-unseen podcast rows for hosts without a slice (slinky-0/1/2)",
          "eval_frac": EVAL_FRAC, "eval_salt": "ppm-v1", "lw": LW, "tool": os.path.abspath(__file__), "built": time.strftime("%F %T"), "hosts": {}}
    for T, dn in mp.items():
        sd = S["donors"][dn]
        lst = [ln.split() for ln in open(f"{WORK}/LISTING-{dn}.txt")]
        xp["hosts"][T] = {"slots": [{"node": dn, "root": f"{STAGE_ROOT}/{dn}-slice", "src": f"{dn}:{LIVE_ROOT}", "shards_present": sd["shards"],
                                     "files": [f for _, f in lst], "bytes": int(sum(int(b) for b, _ in lst)),
                                     "rows": sorted(int(x) for j in range(LW) for x in sd["ranks"][str(j)]["rows"])}],
                          "ranks": [[] for _ in range(LW)], "native": [0] * LW, "bytes": int(sum(int(b) for b, _ in lst))}
    json.dump(xp, open(EXTRA_OUT, "w"))
    print(f"wrote {EXTRA_OUT}")


def write_mode(a):
    S = json.load(open(f"{WORK}/select.json"))
    mp = dict(x.split(":") for x in a.map.split(","))
    xp = D.load_extra_plan(EXTRA_OUT)
    ov = {"format": "si20k-podcast-staged-v1", "built": time.strftime("%F %T"), "tool": os.path.abspath(__file__),
          "source_plans": {"ppm": PPM_PLAN, "intl": INTL_PLAN}, "code_rule": "code = 1 << 40 | donor node row (extra slot 1 = the staged partial slice)",
          "row_rule": (f"target local rank j <- donor local rank j's PPM-unseen stream (ppm-v1 stream[{NC_PPM}:]) at unseen positions >= {SKIP}, "
                       f"slot 0 only, rows in the chosen shards, first {PER_RANK} in stream order"),
          "cfg_note": "use the source plan's cfg with light_frac 0.1 (as the slice hosts' fine-tune ranks); micros are rank-independent",
          "hosts": {}, "verification": {"global": S["global"]}}
    for T, dn in mp.items():
        sd = S["donors"][dn]
        root = f"{STAGE_ROOT}/{dn}-slice"
        lst = [ln.split() for ln in open(f"{WORK}/LISTING-{dn}.txt")]
        V = json.load(open(f"{WORK}/verify-{T}.json"))
        assert V["ok"] and V["donor"] == dn and V["root"] == root
        sl = xp["hosts"][T]["slots"][0]
        assert sl["root"] == root and sl["node"] == dn and sl["shards_present"] == sd["shards"] and sl["files"] == [f for _, f in lst]
        idx = TARGETS.index(T)
        ranks = {}
        for j in range(LW):
            r = sd["ranks"][str(j)]
            ranks[str(j)] = {"prod_rank": 1000 + 8 * idx + j, "unseen_codes": [D.code_of(1, x) for x in r["rows"]], "eval_codes": [],
                             "row_keys_head": r["row_keys"][:3], "donor_prod_rank": r["donor_prod_rank"],
                             "donor_stream_pos_range": [min(r["stream_pos"]), max(r["stream_pos"])]}
        ov["hosts"][T] = {"native_root": root, "extra_plan": EXTRA_OUT, "host_plan_key": T, "staged_from": dn, "shards": sd["shards"],
                          "bytes": xp["hosts"][T]["slots"][0]["bytes"], "n_unseen": LW * PER_RANK, "ranks": ranks}
        ov["verification"][T] = {"select_checks": sd["checks"], "select_ok": sd["checks_ok"],
                                 "files": {"n": V["files"], "missing": len(V["missing"]), "size_mismatch": len(V["size_mismatch"]), "bytes": V["bytes"]},
                                 "micros_ok": f"{V['n_micros_ok']}/{len(V['micros'])}", "micro_T": sorted({m.get('T') for m in V["micros"]}),
                                 "codes_key_or_shard_mismatch": V["codes_key_or_shard_mismatch"], "time": V["time"]}
        m_src, m_dst = open(f"{WORK}/md5-src-{dn}.txt").read().split("\n"), open(f"{WORK}/md5-dst-{T}.txt").read().split("\n")
        ov["verification"][T]["md5_all_files_equal_donor"] = m_src == m_dst and len([x for x in m_src if x]) == len(lst)
        ov["verification"][T]["ok"] = bool(V["ok"] and ov["verification"][T]["md5_all_files_equal_donor"])
    allk = [k for T, dn in mp.items() for j in range(LW) for k in S["donors"][dn]["ranks"][str(j)]["row_keys"]]
    ov["verification"]["disjoint_across_targets"] = len(set(allk)) == len(allk)
    ov["verification"]["ok"] = all(ov["verification"][T]["ok"] and ov["verification"][T]["select_ok"] for T in mp) and \
        ov["verification"]["disjoint_across_targets"] and S["global"]["intl_consumed_subset_of_ppm_consumed"]
    json.dump(ov, open(OVERLAY_OUT, "w"), indent=1)
    print(f"wrote {OVERLAY_OUT}; ok {ov['verification']['ok']}")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="mode", required=True)
    a = sp.add_parser("select"); a.add_argument("--donors", required=True)
    a = sp.add_parser("listing"); a.add_argument("--donor", required=True); a.add_argument("--shards-json", required=True); a.add_argument("--out", required=True)
    a = sp.add_parser("verify"); a.add_argument("--target", required=True); a.add_argument("--donor", required=True)
    a = sp.add_parser("extra"); a.add_argument("--map", required=True)
    a = sp.add_parser("write"); a.add_argument("--map", required=True)
    x = ap.parse_args()
    {"select": select_mode, "listing": listing_mode, "verify": verify_mode, "extra": extra_mode, "write": write_mode}[x.mode](x)


if __name__ == "__main__":
    main()
