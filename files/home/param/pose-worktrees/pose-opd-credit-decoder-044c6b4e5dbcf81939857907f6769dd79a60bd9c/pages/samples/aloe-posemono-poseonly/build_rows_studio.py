"""STUDIO rows for the final eval (10-19, user: prefill from the 4 picked studio prompt packages). Per card's package (video/demo-data/prompts/<id>):
  1. find the package's window in OUR Aloe pose data: the v2pure cache shard holding its recording (plan.json); the speaker's item containing
     the package's first 5 words; package frame 0 = item frame (that word's start - the package's first_word_s) x 12.5, refined by the best
     Mimi cb0 match of the package's prefill_audio within +-2 frames (both reported);
  2. the matching store row of OUR v2pure stores (aloe-v2pure-ft/stores; row_key ALOE/.../<rid>/c<k>, k = the item's ordinal by (start,
     speaker, segment_id) = build_rows.py's rule): the trainer-parity micro (data_aloe_pose.Stores.conv, the v2pure code + env; channel A empty)
     with B's voice clip forced on (voice_mod 1: drawn; the clip itself = the trained data_ppm.voice_clips rule) and NO persona (persona_sides
     none); conversation cut to start at the package's frame 0 (= item frame offset);
  3. LLM prefill = the package's own tokens: depth[1] codes of frames 0..24 := prefill_audio, pdepth frames 0..24 := prefill_pose; the lane
     (words with timing) is the store row's own (frames 0..26 used by the engine prefill). Arrays padded to P + 750 + CD (frames >= 25 never read:
     everything after the prefill is generated).
   python build_rows_studio.py <out dir>      (source aloe-v2pure-ft/aloe_v2pure.env; compute node, CPU)"""
import json
import os
import pickle
import sys

import numpy as np
import pyarrow.parquet as pq

OUT = sys.argv[1]
sys.path.insert(0, "/data/home/amartyavarma/ppm/aloe-v2pure-ft/code")
import data_aloe_pose as A  # noqa: E402
from safetensors.numpy import load_file  # noqa: E402

D = A.D
CACHE = "/data/home/amartyavarma/aloe-pose/sam-cache-active-v2pure"
PROMPTS = "/data/home/amartyavarma/video/demo-data/prompts"
CARDS = [(5, "10_27286bdd"), (13, "06_74807fe0"), (15, "10_27286bdd"), (20, "19_47473a46")]
NPF, NW = 25, 750
os.makedirs(OUT, exist_ok=True)
plan = json.load(open(f"{CACHE}/plan.json"))
cfg = dict(A.aloe_cfg(), voice_mod=1, persona_sides="none")
sp = A.splits_v2(A.STORES)
bykey = {e[4]: e for s in ("train", "eval") for e in sp[s]}
st = A.Stores(A.STORES)
off = np.arange(D.DEPTH, dtype=np.int64) * D.CBS
out, found = [], {}
for card, idn in CARDS:
    pm = json.load(open(f"{PROMPTS}/{idn}/meta.json"))
    li = load_file(f"{PROMPTS}/{idn}/llm_inputs.safetensors")
    pa, pp = li["prefill_audio"].astype(np.int64), li["prefill_pose"].astype(np.int64)
    assert pa.shape == (32, NPF) and pp.shape == (32, NPF)
    rid, spk = pm["recording_id"], f"speaker_{pm['speaker']}"
    if idn not in found:
        shard = [s for s, l in plan["shards"].items() if rid in l]
        assert len(shard) == 1, (idn, shard)
        sd = f"{CACHE}/{shard[0]}"
        ci = pq.read_table(f"{sd}/items.parquet").to_pandas()
        coff = np.load(f"{sd}/codes_off.npy")
        codes = np.load(f"{sd}/codes.npy", mmap_mode="r")
        pose = np.load(f"{sd}/pose.npy", mmap_mode="r")
        import re
        nz = lambda t: re.sub(r"[^a-z' ]", "", t.lower()).split()  # noqa: E731
        pw = nz(pm["text"])
        it = ci[(ci["rid"] == rid) & (ci["speaker"] == spk)].sort_values("start")
        hit = None
        for i, r_ in it.iterrows():                                    # the package's first 5 words in one of the speaker's items
            ws = json.loads(r_["words"])
            flat = [(x, w) for w in ws for x in nz(w[2])]
            for j in range(len(flat)):
                if [x for x, _ in flat[j:j + 5]] == pw[:5]:
                    hit = (int(i), flat[j][1])
                    break
            if hit:
                break
        assert hit is not None, ("package words not found in our store data", idn)
        i, w0 = hit
        c = ci.loc[i]
        o_text = (float(w0[0]) - float(pm["first_word_s"])) * 12.5           # package frame 0 in item frames (first word anchor)
        a0, a1 = int(coff[i]), int(coff[i + 1])
        cc = np.asarray(codes[a0:a1] if codes.shape[1] == D.DEPTH else codes[:, a0:a1].T)
        pc = np.asarray(pose[a0:a1] if pose.shape[1] == D.DEPTH else pose[:, a0:a1].T)

        def match(o, arr, ref):
            lo, hi = max(0, o), min(len(arr), o + NPF)
            if hi <= lo:
                return 0.0
            return float((arr[lo:hi, 0] == ref[0, lo - o:hi - o]).mean())
        cand = [(match(o, cc, pa), o) for o in range(int(round(o_text)) - 2, int(round(o_text)) + 3)]
        m, o = max(cand)
        pmatch = match(o, pc, pp)
        print(json.dumps({"card": card, "identity": idn, "rid": rid[:8], "speaker": spk, "item": c["key"], "item_start_s": float(c["start"]),
                          "first_word": w0, "offset_frames_text": round(o_text, 2), "offset_frames": o, "cb0_match": round(m, 3),
                          "cb0_match_by_offset": {oo: round(mm, 3) for mm, oo in cand}, "pose_cb0_match": round(pmatch, 3), "item_n": int(c["n"])}), flush=True)
        # store row of this cache item: ordinal of the item in its recording by (start, speaker, segment_id)
        rec = ci[ci["rid"] == rid].sort_values(["start", "speaker", "segment_id"])
        k = list(rec.index).index(i)
        key = f"ALOE/aloe-v2pure/{rid}/c{k:05d}"
        assert key in bykey, (idn, key)
        e = bykey[key]
        assert int(e[3]) == int(c["n"]), ("store row n != cache item n", key, e[3], c["n"])
        words = [[round(float(w[0]) - o / 12.5, 2), round(float(w[1]) - o / 12.5, 2), w[2]] for w in json.loads(c["words"]) if o / 12.5 - 1e-6 <= float(w[0]) < (o + NPF + 2) / 12.5]
        found[idn] = dict(entry=e, item=c["key"], offset=o, cb0_match=m, pose_cb0_match=pmatch, words_0_2s=words, item_n=int(c["n"]))
    f = found[idn]
    k_, r_, sw, n_, key = f["entry"]
    m = st.conv((k_, r_, sw), cfg, 0)
    assert m["row_key"] == key and int(m["ids"].shape[1]) == int(m["P"]) + int(n_) + D.CD, (key, m["ids"].shape)
    pv = json.loads(m["prov"])
    P, o = int(m["P"]), f["offset"]
    assert pv["chan_A"] == "empty" and len(pv.get("clips", [])) == 1 and not pv.get("persona"), pv
    T = P + NW + D.CD
    ids = np.full((2, T), D.PAD_ID, np.int64)
    depth = np.full((2, T, D.DEPTH), -1, np.int64)
    present = np.zeros((2, T), np.float32)
    pdepth = np.full((T, D.DEPTH), -1, np.int64)
    ids[:, :P], depth[:, :P], present[:, :P] = m["ids"][:, :P], m["depth"][:, :P], m["present"][:, :P]
    src0, dst0 = (o, 0) if o >= 0 else (0, -o)                               # conversation from item frame o (lane; o < 0: PAD before the item)
    L = min(T - P - dst0, m["ids"].shape[1] - P - src0)
    ids[1, P + dst0:P + dst0 + L] = m["ids"][1, P + src0:P + src0 + L]
    present[1, P:] = 1.0
    depth[1, P + D.CD:P + D.CD + NPF] = pa.T + off                           # ... with the PACKAGE's Mimi + pose for frames 0..24
    pdepth[P + D.CD:P + D.CD + NPF] = pp.T + off
    assert (m["present"][0] == 0).all()
    row = f"S-{card:02d}"
    x = {"P": P, "ids": ids, "depth": depth, "present": present, "pdepth": pdepth, "pid": np.arange(T), "row_key": key, "item": m["item"],
         "prov": m["prov"], "swap": m["swap"], "kind": m["kind"]}
    pickle.dump(x, open(f"{OUT}/{row}.pkl", "wb"))
    cpos = np.flatnonzero(depth[1, :P, 0] >= 0)
    lane_words = I_words = [w for w in f["words_0_2s"]]
    out.append(dict(row=row, card=card, identity=idn, recording_id=pm["recording_id"], speaker=spk, prompt_chunk=pm["chunk"], ref_seed_key=pm["ref_seed_key"],
                    store_row_key=key, cache_item=f["item"], offset_frames=o, cb0_match=round(f["cb0_match"], 3), pose_cb0_match=round(f["pose_cb0_match"], 3),
                    words_0_2s=lane_words, P=P, T=T, n=NW, dur_s=NW / 12.5, prompt="voice clip only (no persona)", with_persona=False,
                    clip=dict(pv["clips"][0], pos=[int(cpos[0]), int(cpos[-1]) + 1]),
                    norm={"s1": {"frame0_root": [float(v) for v in li["global_state"][:2]], "frame0_width": float(li["global_state"][2]), "orig_w": 256, "orig_h": 480}}))
    print(row, json.dumps({k: v for k, v in out[-1].items() if k not in ("norm",)}), flush=True)
json.dump(out, open(f"{OUT}/rows.json", "w"), indent=1)
print("done", len(out))
