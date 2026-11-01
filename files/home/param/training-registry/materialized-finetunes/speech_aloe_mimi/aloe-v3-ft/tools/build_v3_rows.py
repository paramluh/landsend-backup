"""aloe-v3-ft ROWS (10-17, user spec "for finetuning data our row limit is much longer - it should be up to 18 minutes"): ONE ROW PER CALL,
frames 0 .. min(T, 13,500) of the call (18 min at 12.5 Hz), built from the CONTINUOUS per-call data, in exactly Model 1's row format and rules.

CALL SET / SPLIT / STORES: Model 1's 8,229 calls (the calls behind /data/home/amartyavarma/qwen-1ch-data/s2h-m1/stores m1n0..m1n3, = the
  calls of v2 / 21980), each call's split = its Model-1 rows' split (train 8,067 / eval 162 = v2's held-out calls), call -> store v3n<k> =
  its Model-1 store m1n<k> (same call order).
SOURCE per call: /data/home/amartyavarma/data/aloe-handover-full/DATA/<relative_dir>/ (manifest.csv): w00.SPEAKER_0N.codes.npy (Mimi
  [32, T] of the whole call, mimi-stream-v5 = Model 1's codes source), w00.words.json (= Model 1's wlt words.json), w00.mimi.json.
  T asserted == manifest mimi_T == Model 1's src_frames.
  PURGED calls (in Model 1 but removed from the handover afterwards: noface exclusion, no continuous source left anywhere): rebuilt from
  Model 1's OWN chunk rows of the call (codes concatenated = the same continuous codes; words re-timed by +600 s x chunk; a word that crossed
  a 600 s chunk boundary keeps Model 1's clamped end; frames of a dropped short final chunk (< 120 s) are missing). Counted + listed.
RULES (= Model 1's converter s2h_to_s3rows.py, imported and reused, NOT re-implemented): roles SPEAKER_00 = data 'assistant', SPEAKER_01 =
  'user' (asserted per call against Model 1's rows); words per speaker from words.json via s2h_to_s3rows.side_words (verbatim text, empty /
  CJK dropped, a word belongs to the row holding its START, end clamped to the row, frame rule floor(t x 12.5 + 1e-6)); pieces =
  stage-2 speech_q.word_pieces on Qwen3-32B (whole row word list, split_special_tokens); codes verbatim (no loudness change: Model 1 stored the
  mimi-stream-v5 codes as they are); NEW_WORD / PAD / EOB targets and the Mimi code delay are made by the TRAINER from these (unchanged).
ROW LAYOUT (s3-duplex-rows-v1, row width 13,500 = NODE.json row_frames):
  chunk 0  = the TRAINING row: frames [0, n0), n0 = min(T, 13,500); row_key ALOE/s2handover-none/<ses>/<seg>/c00 (= Model 1's chunk-0 key, so
             the trainer's per-row seeds (light frames, v2 voice draw) are the same as Model 1 / v2 for the same call).
  chunk k>=1 (only calls longer than 18 min) = the EXCLUDED TAIL [13,500 k, min(T, 13,500 (k + 1))), kept ONLY as a voice-clip source (the
             trainer's plan takes chunk_index 0 only; never trained / evaluated); tails shorter than S3_MIN_FRAMES (125) are dropped.
  Calls longer than 18 min therefore contribute exactly their first 18 min to training; the rest is discarded from training.
VOX INDEX (per store vox_index.json, per row and data side): channel-A audio reference for the trainer = aloe-m1vox-ft's rule: the speaker's
  WLT FLAC (gs://aloe-processed-datasets/stage2-handover-20261008/wlt/<ses>/<seg>/SPEAKER_0N.flac; bytes / generation from DATA/_meta/
  manifest.jsonl) + the recorded Mimi LUFS gain (w00.mimi.json mimi[spk].lufs_gain_db), frame0 0 (clip-relative = Mimi frame 0); purged calls:
  kind 'mimidec' (the row's own A codes Mimi-decoded at staging, gain 0 dB). + actor id per side (manifest.csv speaker_0N_actor_id).
usage: python -I build_v3_rows.py --out DIR [--only-store K] [--workers 32] [--limit-calls N]"""
import argparse
import csv
import json
import os
import sys
import time
from collections import Counter

import numpy as np

os.environ.setdefault("S3_MIN_FRAMES", "125")     # = the trainer env (aloe_v3.env S3_MIN_FRAMES=125): check_row's row-length floor (tails >= 125 frames)
assert os.environ["S3_MIN_FRAMES"] == "125", os.environ["S3_MIN_FRAMES"]
HERE = os.path.dirname(os.path.abspath(__file__))
V3LIB = os.path.join(os.path.dirname(HERE), "code", "lib")
sys.path.insert(0, V3LIB)
import s3rows as F  # noqa: E402   (v3 lib copy: check_row against the row's store width; registered first so Model 1's converter uses it)

M1DIR = "/data/home/amartyavarma/qwen-1ch-data/s2h-m1"
sys.path.insert(0, M1DIR)
import s2h_to_s3rows as M1B  # noqa: E402   (Model 1's converter: side_words / tok / constants reused verbatim)

assert sys.modules["s3rows"] is F, "Model 1's converter must see the v3 s3rows copy"
M1_STORES = [f"{M1DIR}/stores/m1n{k}" for k in range(4)]
M1_DIGESTS = ["c1c3b2517152741137cc4bb72dd6bdbb", "09bdd43fef84574175caacf4abdddd1e", "ecb1c731e8989b3f2e1b5266b51ea0fd",
              "3b16fb8dd2660a2e8c1fe32c0d361d7b"]
DATA = "/data/home/amartyavarma/data/aloe-handover-full/DATA"
FLAC_PREFIX = "gs://aloe-processed-datasets/stage2-handover-20261008/wlt/"
RF = 13500                                        # 18 min x 60 x 12.5
TAIL_MIN = int(os.environ.get("S3_MIN_FRAMES", "125"))
VERSION = "aloe-v3-wholecall-v1"
SPK_SIDE = M1B.SPK_SIDE                           # SPEAKER_00 assistant, SPEAKER_01 user


def m1_calls():
    """-> {store k: [call dict in Model 1's call order]}: call = conv, ses, seg, split, T (src_frames), m1 rows [(chunk, node row, n)]."""
    import pyarrow.parquet as pq
    out = {}
    for k, root in enumerate(M1_STORES):
        node = F.load_node(root)
        assert node["digest"] == M1_DIGESTS[k], (root, node["digest"])
        t = pq.read_table(f"{root}/rows.parquet", columns=["row", "row_key", "src_conversation", "split", "chunk_index", "n_frames", "src_frames",
                                                           "user_speaker_dir", "assistant_speaker_dir"]).to_pydict()
        cs = {}
        for i in range(len(t["row"])):
            c = t["src_conversation"][i]
            if c not in cs:
                ses, seg = c.split("/")
                cs[c] = {"conv": c, "ses": ses, "seg": seg, "split": t["split"][i], "T": int(t["src_frames"][i]), "m1": [], "store": k}
            d = cs[c]
            assert d["split"] == t["split"][i] and d["T"] == int(t["src_frames"][i]), c
            assert t["assistant_speaker_dir"][i] == "SPEAKER_00" and t["user_speaker_dir"][i] == "SPEAKER_01", (c, "Model 1 roles")
            assert t["row_key"][i] == f"ALOE/s2handover-none/{c}/c{int(t['chunk_index'][i]):02d}", t["row_key"][i]
            d["m1"].append((int(t["chunk_index"][i]), int(t["row"][i]), int(t["n_frames"][i])))
        for d in cs.values():
            d["m1"].sort()
            assert [x[0] for x in d["m1"]] == list(range(len(d["m1"]))), (d["conv"], d["m1"])
        out[k] = list(cs.values())
    return out


def load_meta():
    man = {f"{r['session_id']}/{r['seg']}": r for r in csv.DictReader(open(f"{DATA}/manifest.csv"))}
    fm = {}
    for line in open(f"{DATA}/_meta/manifest.jsonl"):
        r = json.loads(line)
        for spk in SPK_SIDE:
            k = spk.lower()
            if r.get(f"wlt_{k}_flac_uri"):
                fm[(f"{r['session_id']}/{r['seg']}", spk)] = {"uri": r[f"wlt_{k}_flac_uri"], "bytes": int(r[f"wlt_{k}_flac_bytes"]),
                                                              "generation": int(r[f"wlt_{k}_flac_generation"])}
    return man, fm


_G = {}


def _rows_m1(k):
    if k not in _G:
        _G[k] = F.Rows(F.load_node(M1_STORES[k]))
    return _G[k]


def source_full(c, m, st):
    """continuous source -> (codes {spk: [32, T]}, words {spk: [abs-time word dicts]}, T, src dict)."""
    d = f"{DATA}/{m['relative_dir']}"
    codes = {spk: np.load(f"{d}/w00.{spk}.codes.npy") for spk in SPK_SIDE}
    T = int(codes["SPEAKER_00"].shape[1])
    for spk, x in codes.items():
        assert x.ndim == 2 and x.shape == (F.DEPTH, T), (c["conv"], spk, x.shape)
    assert T == int(m["mimi_T"]) == c["T"], (c["conv"], "T", T, m["mimi_T"], c["T"])
    W = json.load(open(f"{d}/w00.words.json"))
    clip = (W.get("aloe") or {}).get("clip") or {}
    assert clip.get("clip_relative_words", True), (c["conv"], "words not clip-relative")
    mj = json.load(open(f"{d}/w00.mimi.json"))
    assert float(mj["clip"]["start_sec"]) == 0.0, (c["conv"], "mimi clip start != 0")
    st.update({f"words_spk_{k}": v for k, v in Counter(w.get("_separated_speaker") for w in W["words"]).items()})
    words = {spk: [w for w in W["words"] if w.get("_separated_speaker") == spk] for spk in SPK_SIDE}
    gain = {spk: float(mj["mimi"][spk]["lufs_gain_db"]) for spk in SPK_SIDE}
    for spk in SPK_SIDE:
        assert int(mj["mimi"][spk]["n_frames"]) == T and int(mj["mimi"][spk]["audio_sr"]) == 24000, (c["conv"], spk)
    return codes, words, T, {"kind": "handover-full", "dir": d, "gain_db": gain}


def source_m1(c, st):
    """purged call: Model 1's chunk rows concatenated -> same outputs as source_full (words with absolute call times)."""
    rows = _rows_m1(c["store"])
    codes = {spk: [] for spk in SPK_SIDE}
    words = {spk: [] for spk in SPK_SIDE}
    for ci, r, n in c["m1"]:
        row = rows[r]
        assert int(row["frame_start"]) == 7500 * ci and int(row["n_frames"]) == n
        for spk, side in SPK_SIDE.items():
            codes[spk].append(np.asarray(row[f"codes_{side}"])[:, :n])
            for w in row[f"words_{side}"]:
                if abs(float(w["end"]) - n / 12.5) < 1e-9 and ci + 1 < len(c["m1"]):
                    st["purged_words_end_at_chunk_boundary"] += 1
                words[spk].append({"text": w["text"], "start": float(w["start"]) + 600.0 * ci, "end": float(w["end"]) + 600.0 * ci,
                                   "in_overlap": bool(w["in_overlap"]), "_separated_speaker": spk})
    codes = {spk: np.concatenate(v, 1) for spk, v in codes.items()}
    T = int(codes["SPEAKER_00"].shape[1])
    st["purged_frames_missing_vs_src"] += c["T"] - T
    return codes, words, T, {"kind": "m1chunks", "m1_rows": [[ci, r] for ci, r, _ in c["m1"]], "frames_missing": c["T"] - T}


def build_call(job):
    """one call -> (rows [chunk 0 + tails], vox entries, stats)."""
    from speech_q import word_pieces
    c, m, fm = job
    st = Counter()
    if m is not None:
        codes, words, T, src = source_full(c, m, st)
    else:
        codes, words, T, src = source_m1(c, st)
        st["calls_purged"] += 1
    rows, vox = [], []
    chunks = [(0, min(T, RF))] + [(k, min(T - RF * k, RF)) for k in range(1, -(-T // RF)) if min(T - RF * k, RF) >= TAIL_MIN]
    st["tails_dropped_short"] += sum(1 for k in range(1, -(-T // RF)) if min(T - RF * k, RF) < TAIL_MIN)
    for k, n in chunks:
        t0, dur = RF * k / F.FPS, n / F.FPS
        base = {"dataset": M1B.DATASET, "dataset_folder": "gs://aloe-processed-datasets/stage2-handover-20261008 (none bucket)",
                "podcast_id": M1B.PODCAST, "episode_id": f"{c['ses']}_{c['seg']}", "chunk_index": k, "frame_start": RF * k,
                "frame_end": RF * k + n, "n_frames": n, "_row_frames": RF}
        for spk, side in SPK_SIDE.items():
            base[f"{side}_speaker_dir"] = spk; base[f"{side}_speaker"] = f"{c['ses']}/{c['seg']}/{spk}"
            cc = np.full((F.DEPTH, RF), F.PAD_CODE, np.int16); cc[:, :n] = codes[spk][:, RF * k:RF * k + n]
            base[f"codes_{side}"] = cc
            ws = M1B.side_words(words[spk], t0, dur, st)
            pp = word_pieces(M1B.tok(), ws) if ws else []
            off = np.zeros(len(ws) + 1, np.int32)
            for jj, p in enumerate(pp):
                off[jj + 1] = off[jj] + len(p)
            base[f"words_{side}"] = ws; base[f"piece_ids_{side}"] = np.array([t for p in pp for t in p], np.int32); base[f"piece_offsets_{side}"] = off
        row = dict(base, row_key=f"{M1B.DATASET}/{M1B.PODCAST}/{c['ses']}/{c['seg']}/c{k:02d}")
        row["prov"] = {"src_conversation": c["conv"], "src_kind": src["kind"], "src_dir": src.get("dir", ""), "src_frames": int(c["T"]),
                       "call_frames": int(T), "split": c["split"], "n_chunks_conv": len(chunks), "role": "train_row" if k == 0 else "tail_clip_source",
                       "chunk_start_sec": t0, "m1_store": c["store"], "m1_rows": json.dumps([[ci, r] for ci, r, _ in c["m1"]])}
        F.check_row(row, strict=True)
        rows.append(row)
        st[f"rows_k{min(k, 1)}_{c['split']}"] += 1; st[f"frames_k{min(k, 1)}_{c['split']}"] += n
    if T > RF:
        st["calls_over_18min"] += 1; st["frames_discarded_over_18min"] += T - RF
    ent = {}
    for spk, side in SPK_SIDE.items():
        actor = (m or {}).get(f"speaker_{spk[-2:]}_actor_id") or None
        if src["kind"] == "handover-full":
            f = fm[(c["conv"], spk)]
            rel = f"{c['ses']}/{c['seg']}/{spk}.flac"
            assert f["uri"] == FLAC_PREFIX + rel, f["uri"]
            ent[side] = {"kind": "gcs", "flac": rel, "uri": f["uri"], "flac_bytes": f["bytes"], "generation": f["generation"],
                         "gain_db": src["gain_db"][spk], "frame0": 0, "audio_frame0": 0, "n_frames": int(rows[0]["n_frames"]), "a_spk": spk, "actor": actor}
        else:
            ent[side] = {"kind": "mimidec", "flac": f"_mimidec/{c['ses']}/{c['seg']}/c00.{spk}.flac", "gain_db": 0.0, "frame0": 0, "audio_frame0": 0,
                         "n_frames": int(rows[0]["n_frames"]), "a_spk": spk, "actor": actor}
    vox.append({"row_key": rows[0]["row_key"], "A": ent, "purged": int(src["kind"] != "handover-full")})
    return rows, vox, st


def write_store(out, name_node, rows, vox, extra):
    import pyarrow as pa
    import pyarrow.ipc as ipc
    import pyarrow.parquet as pq
    import build_rows as BR
    os.makedirs(out, exist_ok=True)
    name = "rows-00000-of-00001"
    R = len(rows)
    for s in F.SIDES:
        mm = np.lib.format.open_memmap(f"{out}/{name}.codes_{s}.npy", mode="w+", dtype=np.int16, shape=(R, F.DEPTH, RF))
        for i, r in enumerate(rows):
            mm[i] = r[f"codes_{s}"]
        mm.flush(); del mm
    sch = BR.arrow_schema()
    cols = {f.name: [r[f.name] for r in rows] for f in sch}
    for s in F.SIDES:
        cols[f"piece_ids_{s}"] = [np.asarray(x).tolist() for x in cols[f"piece_ids_{s}"]]
        cols[f"piece_offsets_{s}"] = [np.asarray(x).tolist() for x in cols[f"piece_offsets_{s}"]]
    tbl = pa.table({k: pa.array(v, type=sch.field(k).type) for k, v in cols.items()}, schema=sch)
    with pa.OSFile(f"{out}/{name}.words.arrow", "wb") as sink, ipc.new_file(sink, sch) as w:
        w.write_table(tbl, max_chunksize=256)
    nf = np.array([r["n_frames"] for r in rows], np.int32)
    ix = {"length": nf, "chunk_index": np.array([r["chunk_index"] for r in rows], np.int32),
          "frame_start": np.array([r["frame_start"] for r in rows], np.int32),
          "dataset": np.array([F.DATASETS.index(r["dataset"]) for r in rows], np.uint8)}
    for s in F.SIDES:
        ix[f"n_words_{s}"] = np.array([len(r[f"words_{s}"]) for r in rows], np.int32)
        ix[f"n_pieces_{s}"] = np.array([len(r[f"piece_ids_{s}"]) for r in rows], np.int32)
    np.savez(f"{out}/{name}.index.npz.tmp.npz", **ix); os.replace(f"{out}/{name}.index.npz.tmp.npz", f"{out}/{name}.index.npz")
    keys = [r["row_key"] for r in rows]
    done = {"shard": name, "rows": R, "frames": int(nf.sum()), "ids_blake2b": F.shard_digest(keys, nf)}
    json.dump(done, open(f"{out}/{name}.DONE", "w"))
    prov = [{k: r[k] for k in BR.PROV} | {"row": i, "shard": name, "row_in_shard": i, **r["prov"]} for i, r in enumerate(rows)]
    with open(f"{out}/rows.jsonl", "w") as f:
        for p in prov:
            f.write(json.dumps(p) + "\n")
    pq.write_table(pa.Table.from_pylist(prov), f"{out}/rows.parquet")
    vi = {e["row_key"]: e for e in vox}
    vrows = [dict(vi[r["row_key"]], row=i) if r["chunk_index"] == 0 else None for i, r in enumerate(rows)]
    json.dump({"format": "aloe-v3-vox-index-v1", "rows": vrows}, open(f"{out}/vox_index.json", "w"))
    tr = [i for i, r in enumerate(rows) if r["chunk_index"] == 0]
    meta = {"node": name_node, "format": F.FORMAT, "shards": [name], "rows": [R], "frames": [int(nf.sum())], "ids_blake2b": [done["ids_blake2b"]],
            "total_rows": R, "total_frames": int(nf.sum()), "train_rows_chunk0": len(tr), "frames_chunk0": int(nf[tr].sum()),
            "hours_chunk0": round(float(nf[tr].sum()) / F.FPS / 3600, 2),
            "by_dataset": {M1B.DATASET: {"rows": R, "frames": int(nf.sum()), "hours": round(nf.sum() / F.FPS / 3600, 2)}},
            "row_frames": RF, "fps": F.FPS, "depth": F.DEPTH, "codebook": F.CBS, "pad_code": F.PAD_CODE, "channels": list(F.SIDES),
            "tokenizer": M1B.TOKENIZER, "ctrl_lo": F.CTRL_LO, "split_special_tokens": True, "frame_rule": "floor(t * 12.5 + 1e-6)",
            "source": "aloe-handover-full DATA (stage2-handover-20261008 local copy) + Model 1 chunk rows for purged calls",
            "converter": os.path.abspath(__file__), "rules_from": os.path.abspath(M1B.__file__), "version": VERSION,
            "built": time.strftime("%Y-%m-%dT%H:%M:%S"), **extra}
    json.dump(meta, open(f"{out}/NODE.json", "w"), indent=1)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--only-store", type=int, default=-1)
    ap.add_argument("--workers", type=int, default=32); ap.add_argument("--limit-calls", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()
    calls = m1_calls()
    man, fm = load_meta()
    n_all = sum(len(v) for v in calls.values())
    print(f"[plan] Model 1 calls {n_all} ({ {k: len(v) for k, v in calls.items()} }); in handover-full {sum(c['conv'] in man for v in calls.values() for c in v)}, "
          f"purged {sum(c['conv'] not in man for v in calls.values() for c in v)} ({time.time() - t0:.0f}s)", flush=True)
    from multiprocessing import Pool
    for k in ([a.only_store] if a.only_store >= 0 else range(4)):
        cs = calls[k][:a.limit_calls] if a.limit_calls else calls[k]
        jobs = [(c, man.get(c["conv"]), {(c["conv"], s): fm[(c["conv"], s)] for s in SPK_SIDE if (c["conv"], s) in fm}) for c in cs]
        rows, vox, st = [], [], Counter()
        with Pool(a.workers) as pool:
            for j, (r, v, x) in enumerate(pool.imap(build_call, jobs, chunksize=4)):
                rows += r; vox += v; st.update(x)
                if (j + 1) % 500 == 0:
                    print(f"[v3n{k}] {j + 1}/{len(jobs)} calls ({time.time() - t0:.0f}s)", flush=True)
        sp = Counter(c["split"] for c in cs)
        meta = write_store(f"{a.out}/v3n{k}", f"v3n{k}", rows, vox, {"calls": len(cs), "calls_split": dict(sp), "stats": dict(st),
                                                                    "m1_store": M1_STORES[k], "m1_digest": M1_DIGESTS[k]})
        print(f"[v3n{k}] calls {len(cs)} {dict(sp)} rows {len(rows)} (chunk0 {meta['train_rows_chunk0']}, tails {len(rows) - meta['train_rows_chunk0']}) "
              f"hours chunk0 {meta['hours_chunk0']} ({time.time() - t0:.0f}s) stats {dict(st)}", flush=True)


if __name__ == "__main__":
    main()
