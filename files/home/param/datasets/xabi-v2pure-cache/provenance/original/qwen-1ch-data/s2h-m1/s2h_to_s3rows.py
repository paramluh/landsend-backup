"""MODEL 1 data: stage-2 handover "none" crossbleed bucket (aloe app recordings; WLT words.json + mimi-stream-v5 codes + per-speaker
10-min prompt chunks) -> per-logical-node stage-3 duplex row stores (s3rows FORMAT s3-duplex-rows-v1) + persona sidecars
(s3-persona-sidecar-v1), read unchanged by stage3/data_s3q.py (S3_DATA_ROOT = one node's dir, staged to /scratch at launch).

SOURCE per conversation (session/seg):  <wlt>/<session>/<seg>/words.json (both speakers' words interleaved, speaker in `_separated_speaker`,
  times clip-relative), <wlt>/<session>/<seg>/system_prompts/SPEAKER_0N/chunk_KK.json (start_sec, end_sec, system_prompt, sections),
  <mimi>/<session>/<seg>/w00.SPEAKER_0N.codes.npy ([32, T] Mimi, clip.start_sec 0 = the same clip origin as the words and chunk windows).
MAPPING (decisions):
  roles      SPEAKER_00 = the prompt's 'assistant' persona -> data role 'assistant' (channel B); SPEAKER_01 = 'user' (channel A). Asserted
             from chunk json persona_speaker.role. (The trainer swaps data role -> model channel with S3_SWAP prob:0.5 as before.)
  identity   dataset 'ALOE' (aloe app recordings), podcast_id 's2handover-none', episode_id '<session>_<seg>' (= the prompt json's episode_id),
             chunk_index k = the producer's chunk index, frame_start = 7500 k (asserted == round(start_sec x 12.5): the producer's chunks are
             exactly [600 k, min(600 (k + 1), dur)), so our row builder's windows and theirs coincide), row_key ALOE/s2handover-none/<session>/<seg>/c<k>.
  frames     n = min(T - 7500 k, 7500); final chunks with n < 1500 (120 s) are dropped (counted). |n - (end_sec - start_sec) x 12.5| <= 2 asserted.
  words      per speaker from words.json, verbatim text (stripped; empty and CJK-script words dropped as in the aloe converter), start/end made
             row-relative (t - 600 k), a word belongs to the chunk holding its START; end clamped to the row; frame rule s3rows.frame_of;
             pieces = stage-2 speech_q.word_pieces (Qwen3-32B tokenizer). NEW_WORD targets are made by the trainer from these (data_s3q).
  prompts    sidecar per data role = that speaker's chunk_KK.json: status ok, system_prompt, sections_json = the 3 prose sections
             (persona / starting_context / memory; asserted: system_prompt == the 3 non-empty sections joined with their headers), tokens_json,
             flags_json, version, start_sec / end_sec. Missing chunk json -> status missing.
  split      eval episodes = the shared split (--split split.json): every seg of a held-out conversation. Eval conversations are spread over
             the nodes first (largest first, fewest eval rows), then train conversations (largest first, fewest train rows). Writes
             <out>/eval_episodes.txt (S3_EVAL_LIST) + <out>/SPLIT.json.
OUTPUT: <out>/<node>/ {NODE.json, rows-00000-of-00001.{codes_user,codes_assistant}.npy, .index.npz, .words.arrow, .persona.arrow, .DONE, rows.jsonl, rows.parquet}
usage: python -I s2h_to_s3rows.py --out DIR --nodes 4 [--convs conversations.jsonl] [--split split.json] [--only-node m1n0] [--limit N] [--workers 16]"""
import argparse
import glob
import json
import os
import re
import sys
import time
from collections import Counter

import numpy as np

Q = "/data/home/amartyavarma/qwen-1ch"
for p in (f"{Q}/stage3/data/rows_build/scripts", f"{Q}/stage2", f"{Q}/stage3"):
    if p not in sys.path:
        sys.path.insert(0, p)
import s3rows as F  # noqa: E402

DATASET, PODCAST = "ALOE", "s2handover-none"
SRC_ROOT = "/data/home/amartyavarma/qwen-1ch-data/aloe-s2handover"
TOKENIZER = "/data/home/amartyavarma/models/Qwen3-32B"
SPK_SIDE = {"SPEAKER_00": "assistant", "SPEAKER_01": "user"}
SIDE_SPK = {v: k for k, v in SPK_SIDE.items()}
MIN_ROW_FRAMES = 1500                        # drop final chunks under 120 s
BATCH = 64
VERSION = "s2h-m1-v1"
CJK = re.compile(r"[　-鿿가-힯]")
HEADER = (("persona", "Persona"), ("starting_context", "Starting context"), ("memory", "Earlier conversation memory"))
_TOK = [None]


def tok():
    if _TOK[0] is None:
        from transformers import AutoTokenizer
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        _TOK[0] = AutoTokenizer.from_pretrained(TOKENIZER)
    return _TOK[0]


# ------------------------------------------------------------------------------------------------ source listing
def list_convs(a):
    """-> list of {conv, session, seg, wlt, mimi} (wlt / mimi = the conversation's dirs)."""
    out = []
    if a.convs:
        for ln in open(a.convs):
            if not ln.strip():
                continue
            r = json.loads(ln)
            s, g = r["id"].split("/")                              # the data agent's conversations.jsonl (MIMI_READY stage)
            assert s == r["session_id"] and g == r["seg"], r["id"]
            wl = os.path.dirname(r["words_json"]); mm = os.path.dirname(r["mimi"]["SPEAKER_00"])
            assert os.path.dirname(r["mimi"]["SPEAKER_01"]) == mm and os.path.dirname(r["mimi_json"]) == mm, r["id"]
            for spk, pr in r["prompts"].items():                   # the chunk jsons we glob are exactly the listed ones
                assert all(os.path.dirname(c) == f"{wl}/system_prompts/{spk}" for c in pr["chunks"]), (r["id"], spk)
            out.append({"conv": f"{s}/{g}", "session": s, "seg": g, "wlt": wl, "mimi": mm, "src_split": r.get("split")})
    else:
        for d in sorted(glob.glob(f"{a.src}/wlt/*/seg*")):
            s, g = d.split("/")[-2:]
            out.append({"conv": f"{s}/{g}", "session": s, "seg": g, "wlt": d, "mimi": f"{a.src}/mimi-stream-v5/{s}/{g}", "src_split": None})
    return out


def load_split(path):
    """split.json -> set of held-out SESSION ids (a conversation id 'session/seg' or 'session_seg' or 'session' -> its session)."""
    d = json.load(open(path))
    ev = None
    for k in ("eval", "heldout", "held_out", "test", "val", "eval_conversations"):
        if isinstance(d, dict) and k in d:
            ev = d[k]; break
    if ev is None and isinstance(d, dict) and "splits" in d:
        ev = [c for c, sp in d["splits"].items() if sp in ("eval", "test", "heldout", "val")]
    if ev is None and isinstance(d, dict):                     # {conv: 'train'|'eval'}
        vals = set(map(str, d.values()))
        if vals <= {"train", "eval", "test", "val", "heldout"}:
            ev = [c for c, sp in d.items() if sp != "train"]
    assert ev is not None, ("unrecognised split.json layout", list(d)[:10] if isinstance(d, dict) else type(d))
    if isinstance(ev, dict):
        ev = list(ev)
    return {str(x).replace("_seg", "/seg").split("/")[0] for x in ev}


# ------------------------------------------------------------------------------------------------ one conversation -> rows
def plan_conv(c):
    """cheap pass: chunk windows + T (npy header) -> list of (k, n) kept, + drop stats."""
    st = Counter()
    try:
        T = int(np.load(f"{c['mimi']}/w00.SPEAKER_00.codes.npy", mmap_mode="r").shape[-1])
        T1 = int(np.load(f"{c['mimi']}/w00.SPEAKER_01.codes.npy", mmap_mode="r").shape[-1])
    except FileNotFoundError:
        st["conv_no_mimi"] += 1
        return [], st
    assert T == T1, (c["conv"], T, T1)
    chunks = {}
    for spk in SPK_SIDE:
        for f in sorted(glob.glob(f"{c['wlt']}/system_prompts/{spk}/chunk_*.json")):
            k = int(os.path.basename(f)[6:-5]); chunks.setdefault(k, {})[spk] = f
    if not chunks:
        st["conv_no_prompt"] += 1
    nk = max(chunks) + 1 if chunks else (T + F.ROW_FRAMES - 1) // F.ROW_FRAMES
    assert not chunks or sorted(chunks) == list(range(nk)), (c["conv"], sorted(chunks))
    nk_codes = (T + F.ROW_FRAMES - 1) // F.ROW_FRAMES
    if nk != nk_codes:
        st["chunks_ne_codes_windows"] += 1
    rows = []
    for k in range(min(nk, nk_codes)):
        n = min(T - F.ROW_FRAMES * k, F.ROW_FRAMES)
        if n < MIN_ROW_FRAMES:
            st["rows_dropped_short"] += 1; st["frames_dropped_short"] += max(n, 0); continue
        rows.append((k, n, chunks.get(k, {})))
    return rows, st


def side_words(ws_in, t0, dur, st):
    ws = []
    for w in ws_in:
        txt = str(w.get("text") or "").strip()
        s_abs = float(w["start"])
        if not (t0 <= s_abs < t0 + dur):
            continue
        if not txt:
            st["w_empty"] += 1; continue
        if CJK.search(txt):
            st["w_cjk_dropped"] += 1; continue
        s, e = s_abs - t0, float(w["end"]) - t0
        if e > dur:
            st["w_end_clamped"] += 1
        e = min(max(e, s), dur)
        ws.append({"text": txt, "start": s, "end": e, "start_frame": F.frame_of(s), "end_frame": min(F.frame_of(e), int(round(dur * F.FPS))),
                   "in_overlap": bool(w.get("in_overlap", False)), "truncated": False})
    if any(b["start"] < a["start"] for a, b in zip(ws, ws[1:])):
        st["rows_resorted"] += 1; ws.sort(key=lambda w: (w["start"], w["end"]))
    return ws


def join_sections(sec):
    return "\n\n".join(h + "\n" + sec[k] for k, h in HEADER if isinstance(sec.get(k), str) and sec[k].strip())


def build_conv(job):
    """(conv dict, [(k, n, chunk files)], split) -> (rows, sidecar rows, stats)."""
    from speech_q import word_pieces
    c, plan, split = job
    st = Counter()
    codes = {spk: np.load(f"{c['mimi']}/w00.{spk}.codes.npy") for spk in SPK_SIDE}
    for spk, x in codes.items():
        assert x.ndim == 2 and x.shape[0] == F.DEPTH, (c["conv"], spk, x.shape)
    W = json.load(open(f"{c['wlt']}/words.json"))
    clip = (W.get("aloe") or {}).get("clip") or {}
    assert clip.get("clip_relative_words", True), (c["conv"], "words not clip-relative")
    spk_counts = Counter(w.get("_separated_speaker") for w in W["words"])
    st.update({f"words_spk_{k}": v for k, v in spk_counts.items()})
    by_spk = {spk: [w for w in W["words"] if w.get("_separated_speaker") == spk] for spk in SPK_SIDE}
    mj = json.load(open(f"{c['mimi']}/w00.mimi.json"))
    assert float(mj["clip"]["start_sec"]) == 0.0, (c["conv"], "mimi clip start != 0")
    rows, side_rows = [], []
    for k, n, chf in plan:
        dur = n / F.FPS
        t0 = 600.0 * k
        base = {"dataset": DATASET, "dataset_folder": "gs://aloe-processed-datasets/stage2-handover-20261008 (none bucket)", "podcast_id": PODCAST,
                "episode_id": f"{c['session']}_{c['seg']}", "chunk_index": k, "frame_start": F.ROW_FRAMES * k, "frame_end": F.ROW_FRAMES * k + n, "n_frames": n}
        cj = {spk: json.load(open(f)) for spk, f in chf.items()}
        for spk, j in cj.items():
            assert int(j["chunk_index"]) == k and abs(float(j["start_sec"]) - t0) < 1e-6, (c["conv"], spk, k, j["start_sec"])
            assert abs((float(j["end_sec"]) - float(j["start_sec"])) * F.FPS - n) <= 2.0, (c["conv"], spk, k, j["end_sec"], n)
            assert j["persona_speaker"]["dir"] == spk and j["persona_speaker"]["role"] == SPK_SIDE[spk], (c["conv"], spk, j["persona_speaker"])
        for spk, side in SPK_SIDE.items():
            base[f"{side}_speaker_dir"] = spk; base[f"{side}_speaker"] = f"{c['session']}/{c['seg']}/{spk}"
            cc = np.full((F.DEPTH, F.ROW_FRAMES), F.PAD_CODE, np.int16); cc[:, :n] = codes[spk][:, F.ROW_FRAMES * k:F.ROW_FRAMES * k + n]
            base[f"codes_{side}"] = cc
            ws = side_words(by_spk[spk], t0, dur, st)
            pp = word_pieces(tok(), ws) if ws else []
            off = np.zeros(len(ws) + 1, np.int32)
            for jj, p in enumerate(pp):
                off[jj + 1] = off[jj] + len(p)
            base[f"words_{side}"] = ws; base[f"piece_ids_{side}"] = np.array([t for p in pp for t in p], np.int32); base[f"piece_offsets_{side}"] = off
        row = dict(base, row_key=f"{DATASET}/{PODCAST}/{c['session']}/{c['seg']}/c{k:02d}")
        row["prov"] = {"src_conversation": c["conv"], "src_wlt": c["wlt"], "src_mimi": c["mimi"], "src_frames": int(codes["SPEAKER_00"].shape[1]),
                       "chunk_start_sec": t0, "split": split, "n_chunks_conv": len(plan)}
        F.check_row(row, strict=True)
        sc = {"row_key": row["row_key"]}
        for side in F.SIDES:
            spk = SIDE_SPK[side]; j = cj.get(spk)
            ok = j is not None and bool((j.get("system_prompt") or "").strip())
            if ok:
                sec = {kk: (j["sections"].get(kk) or "") for kk, _ in HEADER}
                assert join_sections(sec) == j["system_prompt"], (c["conv"], spk, k, "system_prompt != joined sections")
                ids = tok()(j["system_prompt"], add_special_tokens=False)["input_ids"]
                assert ids and max(ids) < F.CTRL_LO
                st[f"prompt_{side}_ok"] += 1
                for kk, _ in HEADER:
                    st[f"sec_{side}_{kk}"] += bool(sec[kk].strip())
            else:
                st[f"prompt_{side}_missing"] += 1
            sc.update({f"{side}_status": "ok" if ok else "missing", f"{side}_note": "" if ok else ("no_chunk_json" if j is None else "empty_prompt"),
                       f"{side}_system_prompt": j["system_prompt"] if ok else "",
                       f"{side}_sections_json": json.dumps(sec) if ok else "",
                       f"{side}_tokens_json": json.dumps(j.get("tokens", {})) if ok else "",
                       f"{side}_flags_json": json.dumps(j.get("flags", {})) if ok else "",
                       f"{side}_version": (j.get("version") or "") if ok else "", f"{side}_prompt_tokens": len(ids) if ok else -1,
                       f"{side}_start_sec": float(j["start_sec"]) if j else t0, f"{side}_end_sec": float(j["end_sec"]) if j else t0 + dur})
        st["rows_" + split] += 1; st["frames_" + split] += n
        rows.append(row); side_rows.append(sc)
    return rows, side_rows, st


# ------------------------------------------------------------------------------------------------ write
def sidecar_schema():
    import pyarrow as pa
    f = [("row_key", pa.string())]
    for s in F.SIDES:
        f += [(f"{s}_status", pa.string()), (f"{s}_note", pa.string()), (f"{s}_system_prompt", pa.string()), (f"{s}_sections_json", pa.string()),
              (f"{s}_tokens_json", pa.string()), (f"{s}_flags_json", pa.string()), (f"{s}_version", pa.string()), (f"{s}_prompt_tokens", pa.int32()),
              (f"{s}_start_sec", pa.float64()), (f"{s}_end_sec", pa.float64())]
    return pa.schema(f, metadata={"format": "s3-persona-sidecar-v1", "batch_rows": str(BATCH)})


def write_node(out, node, rows, side_rows, extra):
    import pyarrow as pa
    import pyarrow.ipc as ipc
    import pyarrow.parquet as pq
    import build_rows as BR
    os.makedirs(out, exist_ok=True)
    name = "rows-00000-of-00001"
    R = len(rows)
    for s in F.SIDES:
        mm = np.lib.format.open_memmap(f"{out}/{name}.codes_{s}.npy", mode="w+", dtype=np.int16, shape=(R, F.DEPTH, F.ROW_FRAMES))
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
    ssch = sidecar_schema()
    st = pa.Table.from_pylist(side_rows, schema=ssch).combine_chunks()
    with pa.OSFile(f"{out}/{name}.persona.arrow", "wb") as sink, ipc.new_file(sink, ssch, options=ipc.IpcWriteOptions(compression="zstd")) as w:
        w.write_table(st, max_chunksize=BATCH)
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
    meta = {"node": node, "format": F.FORMAT, "shards": [name], "rows": [R], "frames": [int(nf.sum())], "ids_blake2b": [done["ids_blake2b"]],
            "total_rows": R, "total_frames": int(nf.sum()), "by_dataset": {DATASET: {"rows": R, "frames": int(nf.sum()), "hours": round(nf.sum() / F.FPS / 3600, 2)}},
            "row_frames": F.ROW_FRAMES, "fps": F.FPS, "depth": F.DEPTH, "codebook": F.CBS, "pad_code": F.PAD_CODE, "channels": list(F.SIDES),
            "tokenizer": TOKENIZER, "ctrl_lo": F.CTRL_LO, "split_special_tokens": True, "frame_rule": "floor(t * 12.5 + 1e-6)",
            "source": "stage2-handover-20261008 none bucket", "converter": os.path.abspath(__file__), "version": VERSION,
            "built": time.strftime("%Y-%m-%dT%H:%M:%S"), **extra}
    json.dump(meta, open(f"{out}/NODE.json", "w"), indent=1)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC_ROOT); ap.add_argument("--convs", default=""); ap.add_argument("--split", default="")
    ap.add_argument("--out", required=True); ap.add_argument("--nodes", type=int, default=4); ap.add_argument("--only-node", default="")
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--fake-eval-frac", type=float, default=0.0, help="dev only (no split.json): hold out this fraction of sessions by hash")
    a = ap.parse_args()
    nodes = [f"m1n{i}" for i in range(a.nodes)]
    convs = list_convs(a)
    if a.limit:
        convs = convs[:a.limit]
    if a.split:
        ev_sess = load_split(a.split)
        bad = [c["conv"] for c in convs if c["src_split"] is not None and (c["src_split"] == "eval") != (c["session"] in ev_sess)]
        assert not bad, ("conversations.jsonl split != split.json", bad[:5])
    else:
        import data_s3q as DS
        ev_sess = {c["session"] for c in convs if DS.h64(f"m1fake|{c['session']}") % 1000 < a.fake_eval_frac * 1000}
    from multiprocessing import Pool
    t0 = time.time()
    with Pool(a.workers) as pool:
        plans = pool.map(plan_conv, convs, chunksize=16)
    st = Counter()
    for _, s in plans:
        st.update(s)
    load = {"eval": {h: 0 for h in nodes}, "train": {h: 0 for h in nodes}}
    node_of = {}
    items = sorted([(c, p) for c, (p, _) in zip(convs, plans) if p], key=lambda x: (-len(x[1]), x[0]["conv"]))
    for sp in ("eval", "train"):
        for c, p in items:
            if (c["session"] in ev_sess) != (sp == "eval"):
                continue
            h = min(nodes, key=lambda h: (load[sp][h], nodes.index(h))); node_of[c["conv"]] = h; load[sp][h] += len(p)
    ev_eps = sorted({f"{DATASET}/{PODCAST}/{c['session']}_{c['seg']}" for c, p in items if c["session"] in ev_sess})
    print(f"[plan] {len(convs)} conversations listed, {len(items)} with rows; eval sessions in split {len(ev_sess)}, eval episodes with rows {len(ev_eps)}; "
          f"rows per node {load} ({time.time() - t0:.0f}s) stats {dict(st)}", flush=True)
    os.makedirs(a.out, exist_ok=True)
    if a.only_node in ("", nodes[0]):
        with open(f"{a.out}/eval_episodes.txt", "w") as f:
            f.write("# MODEL 1 eval episodes (S3_EVAL_LIST): the shared split's held-out conversations, key dataset/podcast_id/episode_id\n")
            for k in ev_eps:
                f.write(k + "\n")
    todo = [a.only_node] if a.only_node else nodes
    for h in todo:
        jobs = [(c, p, "eval" if c["session"] in ev_sess else "train") for c, p in items if node_of[c["conv"]] == h]
        rows, sides, nst = [], [], Counter()
        with Pool(a.workers) as pool:
            for r, s, x in pool.imap(build_conv, jobs, chunksize=4):
                rows += r; sides += s; nst.update(x)
        meta = write_node(f"{a.out}/{h}", h, rows, sides, {"split_rows": {"eval": nst["rows_eval"], "train": nst["rows_train"]}, "stats": dict(nst)})
        print(f"[{h}] rows {len(rows)} (eval {nst['rows_eval']}, train {nst['rows_train']}) hours {meta['by_dataset'][DATASET]['hours']} "
              f"({time.time() - t0:.0f}s) stats {dict(nst)}", flush=True)
        st.update(nst)
    if True:
        json.dump({"convs_listed": len(convs), "convs_with_rows": len(items), "eval_sessions": sorted(ev_sess), "eval_episodes": ev_eps,
                   "rows_per_node": load, "nodes": nodes, "node_of": node_of, "split_file": a.split or None, "convs_file": a.convs or None,
                   "stats": dict(st), "built": time.strftime("%Y-%m-%dT%H:%M:%S")},
                  open(f"{a.out}/SPLIT{'-' + a.only_node if a.only_node else ''}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
