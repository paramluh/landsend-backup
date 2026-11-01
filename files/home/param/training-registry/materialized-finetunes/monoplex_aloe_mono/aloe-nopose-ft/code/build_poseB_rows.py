"""aloe-pose-ft DATA BUILD (10-16): Aloe stage-2 handover (local copy /data/home/amartyavarma/data/aloe-handover-full/DATA, ONLY this
dataset) -> s3-duplex-rows-v1 stores (= Model 1's s2h_to_s3rows format read by 21980's data_aloe: codes_user / codes_assistant, words.arrow,
persona.arrow sidecar, index.npz, rows.parquet, NODE.json) + POSE add-on per store:
  rows-00000-of-00001.pose_user.npy / pose_assistant.npy  int16 [R, 32, 7500]  that speaker's pose tokens on the row's frames (-1 elsewhere
                                                          and for a speaker that does not qualify as B)
  pose_index.npz   b_ok [R, 2] uint8 (user, assistant): 1 = this side QUALIFIES as channel B; split [R] (0 train / 1 eval)
  POSE.json        tokenizer / recipe / counts

QUALIFYING B TRACK (user spec + agreed filters, = count_poseB_rows.py): pose_manifest pose_status == 'ok' (flagged / error dropped) AND the
  manifest's combined_crossbleed_side NOT in (that speaker, 'both', 'unknown') (bleed INTO B dropped; bleed on A fine). A conversation is
  built iff >= 1 of its tracks qualifies. Rows = consecutive 10-min chunks k = 0 .. ceil(T / 7500) - 1 (n = min(T - 7500 k, 7500)); rows
  under 125 frames (the trainer's s3rows floor S3_MIN_FRAMES = 125, as 21980) cannot be read and are dropped (counted).
ROLES (Model 1 / 21980): SPEAKER_00 = data 'assistant' (swap 0 puts it on B), SPEAKER_01 = data 'user' (swap 1); asserted from the prompt
  chunk json persona_speaker. Words = w00.words.json by _separated_speaker, Model 1's rules (s2h_to_s3rows.side_words). Persona sidecar per
  role = that speaker's system_prompts/SPEAKER_0N/chunk_KK.json (Model 1's rules).
POSE: w00.SPEAKER_0N.pose_tokens.npy [32, T] (pose.json: tokenizer InternalCan/pose-tokenizer-32cb-200k-noshift@1d6ff4f0.., the PPM / SAM
  pose-s3 tokenizer revision; normalisation frame0-root-frame0-horizontal-shoulder-width-v1 = PPM's; sha256 + shape asserted; T == Mimi T).
SPLIT: eval iff blake2b8('aloe-pose-ft-v1|' + session_id) % 10000 < 200 (2% of sessions; every seg + both orientations together).
usage: python -I build_poseB_rows.py plan --out DIR --stores 8        (writes DIR/PLAN.json, prints the counts)
       python -I build_poseB_rows.py build --out DIR --store K --workers 64"""
import argparse
import collections
import csv
import glob
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, "/data/home/amartyavarma/qwen-1ch-data/s2h-m1")
import s2h_to_s3rows as B  # noqa: E402   (Model 1's converter: s3rows F, word / persona rules, write_node)

F = B.F
DATA = "/data/home/amartyavarma/data/aloe-handover-full/DATA"
TOKENIZER_POSE = "InternalCan/pose-tokenizer-32cb-200k-noshift@1d6ff4f0"
NORM = "frame0-root-frame0-horizontal-shoulder-width-v1"
MIN_N = 125
EVAL_MOD, EVAL_LT, SALT = 10000, 200, "aloe-pose-ft-v1"
SPK = ("SPEAKER_00", "SPEAKER_01")
SIDE_OF = {"SPEAKER_00": "assistant", "SPEAKER_01": "user"}
B.DATASET, B.PODCAST = "ALOE", "s2handover-pose"
B.VERSION = "aloe-pose-ft-v1"


def h64(s):
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "little")


def is_eval(session):
    return h64(f"{SALT}|{session}") % EVAL_MOD < EVAL_LT


def qualifying():
    man = {r["relative_dir"]: r for r in csv.DictReader(open(f"{DATA}/manifest.csv"))}
    pose = collections.defaultdict(dict)
    for r in csv.DictReader(open(f"{DATA}/pose_manifest.csv")):
        pose[r["relative_dir"]][r["speaker"]] = r
    convs, st = [], collections.Counter()
    for rd, m in sorted(man.items()):
        T, side = int(m["mimi_T"]), m["combined_crossbleed_side"]
        q = []
        for spk in SPK:
            p = pose.get(rd, {}).get(spk)
            if p and p["pose_status"] == "ok" and side not in (spk, "both", "unknown"):
                assert int(p["pose_T"]) == int(p["mimi_T"]) == T, (rd, spk, p["pose_T"], p["mimi_T"], T)
                q.append(spk)
        if not q:
            continue
        ses = m["session_id"]
        nrow = math.ceil(T / F.ROW_FRAMES)
        short = sum(1 for k in range(nrow) if min(T - F.ROW_FRAMES * k, F.ROW_FRAMES) < MIN_N)
        for spk in q:
            st["B tracks"] += 1; st["B rows (ceil T/7500)"] += nrow; st["B hours"] += T / F.FPS / 3600
            st["B rows dropped <125 frames"] += short; st["B rows used"] += nrow - short
            st[f"B rows used {'eval' if is_eval(ses) else 'train'}"] += nrow - short
        convs.append({"conv": f"{ses}/{m['seg']}", "session": ses, "seg": m["seg"], "dir": f"{DATA}/{rd}", "T": T, "b_ok": q,
                      "xbleed": side, "split": "eval" if is_eval(ses) else "train", "nrow": nrow - short})
    st["conversations"] = len(convs); st["conversations both B"] = sum(len(c["b_ok"]) == 2 for c in convs)
    st["sessions"] = len({c["session"] for c in convs}); st["eval sessions"] = len({c["session"] for c in convs if c["split"] == "eval"})
    return convs, st


def plan(a):
    convs, st = qualifying()
    S = a.stores
    load = {"eval": [0] * S, "train": [0] * S}
    store_of = {}
    for sp in ("eval", "train"):
        for c in sorted([c for c in convs if c["split"] == sp], key=lambda c: (-c["nrow"], c["conv"])):
            k = min(range(S), key=lambda k: (load[sp][k], k)); store_of[c["conv"]] = k; load[sp][k] += c["nrow"]
    for c in convs:
        c["store"] = store_of[c["conv"]]
    os.makedirs(a.out, exist_ok=True)
    json.dump({"convs": convs, "stats": {k: (round(v, 1) if isinstance(v, float) else v) for k, v in st.items()}, "stores": S,
               "rows_per_store": load, "salt": SALT, "eval_frac": EVAL_LT / EVAL_MOD, "built": time.strftime("%Y-%m-%dT%H:%M:%S")},
              open(f"{a.out}/PLAN.json", "w"), indent=0)
    print("[plan]", json.dumps({k: (round(v, 1) if isinstance(v, float) else v) for k, v in sorted(st.items())}), flush=True)
    print("[plan] rows per store", load, flush=True)


def build_conv(c):
    """one conversation -> (rows (s3rows dicts + pose arrays + b_ok), sidecar rows, stats)."""
    from speech_q import word_pieces
    st = collections.Counter()
    d, T = c["dir"], int(c["T"])
    codes = {spk: np.load(f"{d}/w00.{spk}.codes.npy") for spk in SPK}
    for spk, x in codes.items():
        assert x.ndim == 2 and x.shape == (F.DEPTH, T) and x.dtype == np.int16, (c["conv"], spk, x.shape, x.dtype)
    pose = {}
    for spk in c["b_ok"]:
        pj = json.load(open(f"{d}/w00.{spk}.pose.json"))
        assert pj["status"] == "ok" and pj["speaker"] == spk and pj["tokenizer"].startswith(TOKENIZER_POSE), (c["conv"], spk, pj["tokenizer"])
        assert pj["normalization"]["recipe"].startswith(NORM) and float(pj["token_rate_hz"]) == 12.5, (c["conv"], spk, pj["normalization"]["recipe"])
        raw = open(f"{d}/{pj['pose_tokens_file']}", "rb").read()
        assert hashlib.sha256(raw).hexdigest() == pj["pose_tokens_sha256"], (c["conv"], spk, "pose sha256")
        p = np.load(f"{d}/{pj['pose_tokens_file']}")
        assert p.shape == (F.DEPTH, T) == tuple(pj["pose_tokens_shape"]) and int(p.min()) >= 0 and int(p.max()) < F.CBS, (c["conv"], spk, p.shape)
        pose[spk] = p.astype(np.int16)
    W = json.load(open(f"{d}/w00.words.json"))
    clip = (W.get("aloe") or {}).get("clip") or {}
    assert (W.get("aloe") or {}).get("clip_relative_words", clip.get("clip_relative_words", True)), (c["conv"], "words not clip-relative")
    by_spk = {spk: [w for w in W["words"] if w.get("_separated_speaker") == spk] for spk in SPK}
    mj = json.load(open(f"{d}/w00.mimi.json"))
    assert float(mj["clip"]["start_sec"]) == 0.0, (c["conv"], "mimi clip start != 0")
    chunks = {}
    for spk in SPK:
        for f in sorted(glob.glob(f"{d}/system_prompts/{spk}/chunk_*.json")):
            chunks.setdefault(int(os.path.basename(f)[6:-5]), {})[spk] = f
    rows, side_rows = [], []
    ses, seg = c["session"], c["seg"]
    nk = math.ceil(T / F.ROW_FRAMES)
    for k in range(nk):
        n = min(T - F.ROW_FRAMES * k, F.ROW_FRAMES)
        if n < MIN_N:
            st["rows_dropped_lt125"] += 1
            continue
        dur, t0 = n / F.FPS, 600.0 * k
        chf = chunks.get(k, {})
        base = {"dataset": B.DATASET, "dataset_folder": "gs://aloe-processed-datasets/stage2-handover-20261008 (local aloe-handover-full, pose ok + clean B)",
                "podcast_id": B.PODCAST, "episode_id": f"{ses}_{seg}", "chunk_index": k, "frame_start": F.ROW_FRAMES * k,
                "frame_end": F.ROW_FRAMES * k + n, "n_frames": n}
        cj = {spk: json.load(open(f)) for spk, f in chf.items()}
        for spk, j in cj.items():
            assert int(j["chunk_index"]) == k and abs(float(j["start_sec"]) - t0) < 1e-6, (c["conv"], spk, k, j["start_sec"])
            if abs((float(j["end_sec"]) - float(j["start_sec"])) * F.FPS - n) > 2.0:
                st["prompt_window_ne_row"] += 1
            assert j["persona_speaker"]["dir"] == spk and j["persona_speaker"]["role"] == SIDE_OF[spk], (c["conv"], spk, j["persona_speaker"])
        pz = {}
        for spk in SPK:
            side = SIDE_OF[spk]
            base[f"{side}_speaker_dir"] = spk; base[f"{side}_speaker"] = f"{ses}/{seg}/{spk}"
            cc = np.full((F.DEPTH, F.ROW_FRAMES), F.PAD_CODE, np.int16); cc[:, :n] = codes[spk][:, F.ROW_FRAMES * k:F.ROW_FRAMES * k + n]
            base[f"codes_{side}"] = cc
            pp_ = np.full((F.DEPTH, F.ROW_FRAMES), -1, np.int16)
            if spk in pose:
                pp_[:, :n] = pose[spk][:, F.ROW_FRAMES * k:F.ROW_FRAMES * k + n]
            pz[side] = pp_
            ws = B.side_words(by_spk[spk], t0, dur, st)
            pp = word_pieces(B.tok(), ws) if ws else []
            off = np.zeros(len(ws) + 1, np.int32)
            for jj, p in enumerate(pp):
                off[jj + 1] = off[jj] + len(p)
            base[f"words_{side}"] = ws; base[f"piece_ids_{side}"] = np.array([t for p in pp for t in p], np.int32); base[f"piece_offsets_{side}"] = off
        row = dict(base, row_key=f"{B.DATASET}/{B.PODCAST}/{ses}/{seg}/c{k:02d}")
        row["prov"] = {"src_conversation": c["conv"], "src_dir": d, "src_frames": T, "chunk_start_sec": t0, "split": c["split"],
                       "n_chunks_conv": nk, "xbleed": c["xbleed"], "b_ok": ",".join(SIDE_OF[s] for s in c["b_ok"])}
        F.check_row(row, strict=True)
        row["_pose"] = pz
        row["_b_ok"] = [int("SPEAKER_01" in c["b_ok"]), int("SPEAKER_00" in c["b_ok"])]      # (user, assistant)
        sc = {"row_key": row["row_key"]}
        for side in F.SIDES:
            spk = B.SIDE_SPK[side]; j = cj.get(spk)
            ok = j is not None and bool((j.get("system_prompt") or "").strip())
            ids = []
            if ok:
                sec = {kk: (j["sections"].get(kk) or "") for kk, _ in B.HEADER}
                assert B.join_sections(sec) == j["system_prompt"], (c["conv"], spk, k, "system_prompt != joined sections")
                ids = B.tok()(j["system_prompt"], add_special_tokens=False)["input_ids"]
                assert ids and max(ids) < F.CTRL_LO
                st[f"prompt_{side}_ok"] += 1
            else:
                st[f"prompt_{side}_missing"] += 1
            sc.update({f"{side}_status": "ok" if ok else "missing", f"{side}_note": "" if ok else ("no_chunk_json" if j is None else "empty_prompt"),
                       f"{side}_system_prompt": j["system_prompt"] if ok else "", f"{side}_sections_json": json.dumps(sec) if ok else "",
                       f"{side}_tokens_json": json.dumps(j.get("tokens", {})) if ok else "", f"{side}_flags_json": json.dumps(j.get("flags", {})) if ok else "",
                       f"{side}_version": (j.get("version") or "") if ok else "", f"{side}_prompt_tokens": len(ids) if ok else -1,
                       f"{side}_start_sec": float(j["start_sec"]) if j else t0, f"{side}_end_sec": float(j["end_sec"]) if j else t0 + dur})
        st["rows_" + c["split"]] += 1; st["frames_" + c["split"]] += n
        for side, ok_ in zip(F.SIDES, row["_b_ok"]):
            st[f"brows_{c['split']}_{side}"] += ok_
        rows.append(row); side_rows.append(sc)
    return rows, side_rows, st


def build_conv_file(job):
    import pickle
    i, c, tmp = job
    out = f"{tmp}/c{i:05d}.pkl"
    res = build_conv(c)
    with open(out + ".tmp", "wb") as f:
        pickle.dump(res, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(out + ".tmp", out)
    return out


def build(a):
    P = json.load(open(f"{a.out}/PLAN.json"))
    convs = [c for c in P["convs"] if c["store"] == a.store]
    out = f"{a.out}/p{a.store}"
    import multiprocessing as mp
    import pickle
    t0 = time.time()
    tmp = f"/scratch/amartyavarma/aloe-pose-build/p{a.store}"
    os.makedirs(tmp, exist_ok=True)
    jobs = [(i, c, tmp) for i, c in enumerate(convs)]
    rows, sides, st = [], [], collections.Counter()
    # spawn (no fork of a process holding locks / tokenizer threads) + results via node-local files (only the path crosses the pipe)
    with mp.get_context("spawn").Pool(a.workers) as pool:
        paths = []
        for j, pth in enumerate(pool.imap_unordered(build_conv_file, jobs, chunksize=1)):
            paths.append(pth)
            if (j + 1) % 100 == 0 or j + 1 == len(jobs):
                print(f"[p{a.store}] {j + 1}/{len(jobs)} conversations built ({time.time() - t0:.0f}s, "
                      f"{(j + 1) / max(time.time() - t0, 1e-9) * 60:.0f} conv/min)", flush=True)
    for i in range(len(jobs)):                                      # deterministic order = the plan's conversation order
        r, s, x = pickle.load(open(f"{tmp}/c{i:05d}.pkl", "rb"))
        rows += r; sides += s; st.update(x)
    print(f"[p{a.store}] {len(convs)} conversations -> {len(rows)} rows ({time.time() - t0:.0f}s) stats {dict(st)}", flush=True)
    meta = B.write_node(out, f"p{a.store}", rows, sides, {"split_rows": {"eval": st["rows_eval"], "train": st["rows_train"]}, "stats": dict(st)})
    name = meta["shards"][0]
    R = len(rows)
    for side in F.SIDES:
        mm = np.lib.format.open_memmap(f"{out}/{name}.pose_{side}.npy", mode="w+", dtype=np.int16, shape=(R, F.DEPTH, F.ROW_FRAMES))
        for i, r in enumerate(rows):
            mm[i] = r["_pose"][side]
        mm.flush(); del mm
    b_ok = np.array([r["_b_ok"] for r in rows], np.uint8).reshape(R, 2)
    split = np.array([1 if r["prov"]["split"] == "eval" else 0 for r in rows], np.int8)
    for i, r in enumerate(rows):                                   # a qualifying side has pose on every row frame, a non-qualifying none
        for j, side in enumerate(F.SIDES):
            pv = r["_pose"][side][:, :r["n_frames"]]
            assert (pv >= 0).all() if b_ok[i, j] else (pv < 0).all(), (r["row_key"], side)
    np.savez(f"{out}/pose_index.npz", b_ok=b_ok, split=split)
    keys = [r["row_key"] for r in rows]
    pdg = hashlib.blake2b(digest_size=16)
    for side in F.SIDES:
        pdg.update(open(f"{out}/{name}.pose_{side}.npy", "rb").read())
    pdg.update(b_ok.tobytes()); pdg.update(split.tobytes())
    json.dump({"format": "aloe-pose-addon-v1", "store": f"p{a.store}", "rows": R, "tokenizer": TOKENIZER_POSE, "normalization": NORM,
               "pose_digest": pdg.hexdigest(), "node_digest": B.F.shard_digest(keys, np.array([r["n_frames"] for r in rows], np.int32)),
               "b_ok_user": int(b_ok[:, 0].sum()), "b_ok_assistant": int(b_ok[:, 1].sum()),
               "b_items_train": int(b_ok[split == 0].sum()), "b_items_eval": int(b_ok[split == 1].sum()), "stats": dict(st),
               "built": time.strftime("%Y-%m-%dT%H:%M:%S")}, open(f"{out}/POSE.json", "w"), indent=1)
    open(f"{out}/BUILD_DONE", "w").write(str(R))
    print(f"[p{a.store}] wrote {out}: {R} rows, B items train {int(b_ok[split == 0].sum())} eval {int(b_ok[split == 1].sum())} "
          f"({time.time() - t0:.0f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("plan", "build"))
    ap.add_argument("--out", required=True); ap.add_argument("--stores", type=int, default=8)
    ap.add_argument("--store", type=int, default=0); ap.add_argument("--workers", type=int, default=64)
    a = ap.parse_args()
    plan(a) if a.cmd == "plan" else build(a)


if __name__ == "__main__":
    main()
