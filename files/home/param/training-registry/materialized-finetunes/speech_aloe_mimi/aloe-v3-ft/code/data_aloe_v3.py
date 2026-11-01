"""aloe-v3-ft DATA (10-17, user spec: "for finetuning data our row limit is much longer - it should be up to 18 minutes, and we shouldn't
apply the mid-call personas"; coordinator: channel A = Voxtral as aloe-m1vox-ft) = aloe-v2-ft's data_aloe_v2.py (job 22383) with ONLY:
  ROWS      WHOLE CALLS up to 18 min: the v3 stores v3n0..v3n3 (tools/build_v3_rows.py: Model 1's 8,229 calls / split / call -> store, built
            from the continuous per-call data with Model 1's converter rules, row width 13,500): ONE training row per call = its chunk 0 =
            frames 0..min(T, 13,500) (v2: Model 1's 10-min chunk 0 = frames 0..min(T, 7,500)). Rows with chunk_index >= 1 are the excluded tail
            beyond 18 min, never trained / evaluated (voice-clip source only). Both speakers as B (each call twice, swap 0 / 1), loss on B only,
            held-out = the 162 eval calls x 2 (= v2's). PLAN = v2's rule unchanged (items (store, row, swap) in store order, permuted with
            seed 20261011, item t x WORLD + R for rank R at step t, dummies on the last step); store call order = Model 1's, so the plan visits
            the SAME (call, orientation) items in the SAME order as v2 (asserted by the trainer against v2's plan).
  CHANNEL A VOXTRAL (= aloe-m1vox-ft / job 22335's data_aloe_vox.speech_micro_vox edit, applied to v2's micro): A text PAD, A Mimi none,
            present_A = 0 in the conversation (A never had a prefix clip in v2); vin = (P, n): the trainer adds s x vadapter(Voxtral(A audio
            of row frames 0..n-1)) at positions P..P+n-1. A waveform = the partner's WLT FLAC (vox_index.json; node-local ALOE_FLAC_ROOT),
            frames 0..n-1 of the call (clip-relative = Mimi frame 0), recorded Mimi LUFS gain, decoded in the DataLoader workers
            (audio_s4q.ffmpeg_row_audio via rowcache_vox, bit-identical either way); purged calls: Mimi decode of the row's own A codes.
  SYSPROMPT = v2: NO text at all (no persona / starting context / memory); B's own voice clip on the SAME seeded 30 % draw as v2
            (h64('aloe-v2-voice|20261011|<row_key>|<swap>') % 10 < 3; v3 chunk-0 row keys = Model 1's chunk-0 keys -> the same draws as v2),
            EMPTY system block otherwise. Clip SOURCE (env ALOE_V3_CLIP_SRC, recorded in the cfg):
              'v2'    (DEFAULT) PPM's voice_clips() unchanged = a LATER chunk of the same call in its SECOND half, nearest first, blank lane.
                      In the v3 store the only later chunk of a call is its excluded tail beyond 18 min (160 calls longer than 18 min, 146
                      tails >= 125 frames) -> the realised clip rate collapses (measured: tests/voice_clip_rates.py).
              'actor' (OFF; for the user to choose) a clip of the SAME ACTOR (manifest speaker_0N_actor_id) from one of their OTHER train calls
                      (candidates ordered by h64('aloe-v3-actor|seed|row_key|cand_key'), first with a window wins), window = sysprompt.window
                      over the whole other call's chunk-0 row with voice_clips' own (L0, u) draws, blank lane (audio only, = v2's clip form).
                      Never a held-out call as source; never the item's own call.
  LIGHT     = v2 (light_frac 0.1 training, 1.0 eval: PPM's per-frame seeded rule; row_frames = P + n, so frames 0..7,499 of a call get the same
            u as v2's chunk-0 row).
  BUDGET    sp_cap 2,000 + 13,500 + CD (v2: 2,000 + 7,500 + CD).
Everything else (micro maths, CD 2, crop_conv False, segment / budget asserts, dummy micros) = data_aloe_v2 / data_aloe_m1vox."""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.join(HERE, "lib"), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
import data_qqa as Q  # noqa: E402

D = Q.D
STORE_ROOT = os.environ.get("ALOE_STORE_ROOT", "/scratch/amartyavarma/aloe-v3-ft/stores")
SRC_ROOT = "/data/home/amartyavarma/ppm/aloe-v3-ft/stores"
NST = 4
STORES = [f"{STORE_ROOT}/v3n{k}" for k in range(NST)]
SRC_STORES = [f"{SRC_ROOT}/v3n{k}" for k in range(NST)]
FLAC_ROOT = os.environ.get("ALOE_FLAC_ROOT", "/scratch/amartyavarma/aloe-v3-ft/flac")
DIGESTS_FILE = os.environ.get("ALOE_V3_STORES_JSON", "/data/home/amartyavarma/ppm/aloe-v3-ft/plan/STORES.json")
CLIP_SRC = os.environ.get("ALOE_V3_CLIP_SRC", "v2")
assert CLIP_SRC in ("v2", "actor"), ("ALOE_V3_CLIP_SRC must be v2 (default) or actor", CLIP_SRC)
PLAN_SEED = 20261011
VOICE_SEED = 20261011
VOICE_NUM, VOICE_DEN = 3, 10                      # 30 % of (row, orientation) items draw B's own voice clip (= v2)
ROW_FRAMES = 13500                                # 18 min
BUDGET = 2000 + ROW_FRAMES + D.CD                 # sp_cap + the longest row + CD


def aloe_cfg():
    """training data cfg = v2's (data_aloe_v2.aloe_cfg: qqa_cfg with persona_sides none, light_frac 0.1, v2 voice keys) + clip_src."""
    return Q.qqa_cfg(persona_sides="none", light_frac=0.1, sysprompt="v2_voiceB_only", voice_frac=VOICE_NUM / VOICE_DEN, voice_seed=VOICE_SEED,
                     clip_src=CLIP_SRC)


def eval_cfg():
    """eval micros: = aloe_cfg() but light_frac 1.0 (= v2 / 21980's eval light definition)."""
    return dict(aloe_cfg(), light_frac=1.0)


def digests():
    return json.load(open(DIGESTS_FILE))


def splits(roots=STORES):
    """-> {'train': [(k, r)], 'eval': [(k, r)], counts, calls}: chunk-0 rows only (one per call); row == node row asserted."""
    import pyarrow.parquet as pq
    out = {"train": [], "eval": [], "n_all": {"train": 0, "eval": 0}, "n_tail": 0, "calls": {"train": [], "eval": []}}
    for k, root in enumerate(roots):
        t = pq.read_table(f"{root}/rows.parquet", columns=["row", "split", "n_frames", "chunk_index", "src_conversation"]).to_pydict()
        assert t["row"] == list(range(len(t["row"]))), root
        for r, s, ci, c in zip(t["row"], t["split"], t["chunk_index"], t["src_conversation"]):
            out["n_all"][s] += 1
            if int(ci) == 0:
                out[s].append((k, int(r))); out["calls"][s].append(c)
            else:
                out["n_tail"] += 1
    return out


def make_plan(world, roots=STORES):
    sp = splits(roots)
    items = [(k, r, s) for (k, r) in sp["train"] for s in (0, 1)]
    calls = [(c, s) for c in sp["calls"]["train"] for s in (0, 1)]
    perm = np.random.default_rng(PLAN_SEED).permutation(len(items))
    items = [items[i] for i in perm]
    calls = [calls[i] for i in perm]
    steps = -(-len(items) // world)
    ev = [(k, r, s) for (k, r) in sp["eval"] for s in (0, 1)]
    ev_calls = [(c, s) for c in sp["calls"]["eval"] for s in (0, 1)]
    return {"items": items, "steps": steps, "eval": ev, "n_train_rows": len(sp["train"]), "n_eval_rows": len(sp["eval"]), "n_all": sp["n_all"],
            "n_tail": sp["n_tail"], "item_calls": calls, "eval_calls": ev_calls}


def rank_items(plan, rank, world):
    """-> (train: [item or None (dummy)] per step, eval: [item] of this rank) (= v2)."""
    it = plan["items"]
    tr = [it[t * world + rank] if t * world + rank < len(it) else None for t in range(plan["steps"])]
    return tr, plan["eval"][rank::world]


def want_voice(row_key, swap, seed=VOICE_SEED):
    return D.h64(f"aloe-v2-voice|{int(seed)}|{row_key}|{int(swap)}") % VOICE_DEN < VOICE_NUM     # = v2 (same key string)


# ------------------------------------------------------------------------------------------------ voice clip sources
def clip_draws(row, cfg):
    """voice_clips' own (L0, u) draws per data track (data_ppm.voice_clips: rng from the row key)."""
    rng = np.random.default_rng(np.random.SeedSequence([int(cfg["seed"]), D.h64(row["row_key"]) & 0x7FFFFFFFFFFF, 0x5643]))
    return [(int(rng.integers(int(cfg["clip_min"]), int(cfg["clip_max"]) + 1)), float(rng.random())) for _ in range(2)]


class ActorIndex:
    """actor id -> [(store k, row r, data side, call)] over the chunk-0 TRAIN rows of all 4 stores (vox_index actor ids)."""

    def __init__(self, stores):
        self.by = {}
        self.actor_of = {}
        for k in range(NST):
            vi = stores.vox(k)
            sp = stores.split_of(k)
            for r, e in enumerate(vi):
                if e is None:
                    continue
                for side in ("user", "assistant"):
                    a = e["A"][side].get("actor")
                    self.actor_of[(k, r, side)] = a
                    if a and sp[r] == "train":
                        self.by.setdefault(a, []).append((k, r, side, e["row_key"]))

    def candidates(self, actor, row_key, seed):
        rk0 = D.SPR.rec_key(row_key)
        cs = [c for c in self.by.get(actor, []) if D.SPR.rec_key(c[3]) != rk0]
        return sorted(cs, key=lambda c: D.h64(f"aloe-v3-actor|{int(seed)}|{row_key}|{c[3]}"))


def clip_v2(row, d, sib, cfg):
    """the v2 rule: data_ppm.voice_clips unchanged (later chunk of the same call in its 2nd half; v3: the excluded tail)."""
    return D.voice_clips(row, sib, cfg)[d]


def clip_actor(row, d, k, r, stores, cfg):
    """the 'actor' alternative (module doc) for data track d of row (k, r) -> clip dict or None."""
    side = ("user", "assistant")[d]
    actor = stores.actors().actor_of.get((int(k), int(r), side))
    if not actor:
        return None
    L0, u = clip_draws(row, cfg)[d]
    wcfg = {"clip_min": cfg["clip_min"], "stride": cfg["stride"], "min_cov": cfg["min_cov"], "top_frac": cfg["top_frac"]}
    for k2, r2, ss, key2 in stores.actors().candidates(actor, row["row_key"], cfg["seed"]):
        sr = stores.open(k2)[1][r2]
        w = D.SPR.window(sr[f"words_{ss}"], int(sr["n_frames"]), L0, u, wcfg, lo=0)
        if w is not None:
            return dict(w, src="actor_other_call", src_row=sr, src_side=ss, src_key=sr["row_key"], src_chunk=int(sr["chunk_index"]), blank_lane=True,
                        side=side, actor=actor)
    return None


def prefix_v3(row, swap, sib, cfg, k=None, r=None, stores=None):
    """-> (sysprompt.build_prefix dict, provenance) = data_aloe_v2.prefix_v2 with the clip source of cfg['clip_src']."""
    sw = int(swap)
    wv = want_voice(row["row_key"], sw, cfg["voice_seed"])
    pv = {"voice_drawn": int(wv), "clip_src": cfg["clip_src"]}
    segs = []
    if wv:
        d = 1 ^ sw                                                   # model channel B (c = 1) holds data track 1 ^ swap (podcast_prefix rule)
        cl = clip_v2(row, d, sib, cfg) if cfg["clip_src"] == "v2" else clip_actor(row, d, k, r, stores, cfg)
        if cl is not None:
            segs.append({"kind": "speech", "track": 1, **cl})
    pv["clips"] = [{"track": "AB"[s["track"]], "data_side": s["side"], "src_key": s["src_key"], "src_chunk": s["src_chunk"],
                    "frames": [s["w0"], s["w0"] + s["L"]], "cov": round(s["cov"], 3), "src": s.get("src")} for s in segs]
    pv["voice_clip"] = int(bool(segs))
    fixed = 5 + sum(int(s["L"]) + 5 for s in segs)
    assert fixed <= int(cfg["sp_cap"]), ("voice clip exceeds the system-prompt cap", fixed)
    if not segs:                                                     # EMPTY system block: header + footer only (podcast_prefix)
        segs = [{"kind": "text", "track": 1, "ids": []}]
    spx = D.SPR.build_prefix(row, segs, {"new_word": cfg["new_word"]})
    assert 5 <= spx["P"] <= int(cfg["sp_cap"]), ("system prompt length", spx["P"])
    assert pv["voice_clip"] or spx["P"] == 5, ("empty system block must be 5 positions", spx["P"])
    assert (spx["ids"][0] == D.PAD_ID).all() and not (np.asarray(spx["depth"])[0] >= 0).any(), "channel A must carry nothing in the prefix"
    pv["P"] = int(spx["P"])
    return spx, pv


def speech_micro_v3(row, node_row, sib, cfg, rank, swap, k=None, stores=None, vox=True):
    """= data_aloe_v2.speech_micro_v2 (prefix_v3) + (vox) aloe-m1vox-ft's channel-A edit (A emptied in the conversation, vin = (P, n))."""
    sw = int(swap)
    spx, pv = prefix_v3(row, sw, sib, cfg, k=k, r=node_row, stores=stores)
    P, n = int(spx["P"]), int(row["n_frames"])
    rf = P + n if not cfg["crop_conv"] else 7500
    dc = {"new_word": cfg["new_word"], "eob_gap_s": cfg["eob_gap_s"], "wpid": cfg["wpid"], "light_seed": cfg["light_seed"], "row_frames": rf,
          "frame_light": True, "no_crop": not cfg["crop_conv"]}
    a = D.DS.row_arrays(row, dc, rank, int(node_row), "ppm", float(cfg["light_frac"]), sp=spx, swap=sw)
    sel, y, word, pad, eob, nw, cb0, light = D._b_only(a)
    T = len(a["pid"])
    out = {"kind": "speech", "ids": a["ids"], "depth": a["depth"], "present": a["present"], "pid": a["pid"], "segs": [T],
           "sel": sel, "y": y, "word": word, "pad": pad, "eob": eob, "nw": nw, "n_text": float(len(sel)), "cb0": cb0, "light": light,
           "P": P, "swap": sw}
    assert (cb0[:P + D.CD] < 0).all() and (light >= P + D.CD).all() and (sel >= P).all()
    assert T == (7500 if cfg["crop_conv"] else P + n) + D.CD, ("row length", T, P, n)
    pv.update(row_key=row["row_key"], swap=sw, chan_B_data_side=("user" if sw else "assistant"), n_frames=n, T=T, n_light=int(len(light)))
    if vox:                                                          # aloe-m1vox-ft channel-A edit (data_aloe_vox.speech_micro_vox)
        assert (out["depth"][0, :P] < 0).all(), "an A code in the prefix"
        out["ids"] = out["ids"].copy(); out["depth"] = out["depth"].copy(); out["present"] = out["present"].copy()
        out["ids"][0, P:] = D.PAD_ID
        out["depth"][0] = -1
        out["present"][0, P:] = 0.0
        out["vin"] = (P, n)
        pv["chan_A"] = "voxtral_latents_only"
    out["prov"] = json.dumps(pv)
    out["voice_clip"] = pv["voice_clip"]
    out["voice_drawn"] = pv["voice_drawn"]
    return out


class Stores:
    """the 4 node-local v3 stores, opened lazily (per DataLoader worker); digests asserted against plan/STORES.json. No persona sidecar."""

    def __init__(self, roots=STORES):
        self.roots, self.s, self.vi, self.sp, self._act = list(roots), {}, {}, {}, None
        self.dg = digests()

    def open(self, k):
        k = int(k)
        if k not in self.s:
            node = D.R3.load_node(self.roots[k])
            assert node["digest"] == self.dg["stores"][k]["digest"], ("staged store differs from the build", self.roots[k], node["digest"])
            assert int(node["meta"]["row_frames"]) == ROW_FRAMES, node["meta"]["row_frames"]
            st = D.R3.Rows(node)
            self.s[k] = (node, st, D.SPR.SibIndex(node, st))
        return self.s[k]

    def vox(self, k):
        if k not in self.vi:
            j = json.load(open(f"{self.roots[k]}/vox_index.json"))
            assert j["format"] == "aloe-v3-vox-index-v1"
            self.vi[k] = j["rows"]
        return self.vi[k]

    def split_of(self, k):
        if k not in self.sp:
            import pyarrow.parquet as pq
            self.sp[k] = pq.read_table(f"{self.roots[k]}/rows.parquet", columns=["split"]).to_pydict()["split"]
        return self.sp[k]

    def actors(self):
        if self._act is None:
            self._act = ActorIndex(self)
        return self._act

    def conv(self, item, cfg, rank, audio=True, vox=True):
        k, r, sw = item
        node, st, sib = self.open(int(k))
        row = st[int(r)]
        D.R3.check_row(row)
        assert int(row["chunk_index"]) == 0, ("v3: chunk-0 (whole-call) rows only", row["row_key"], row["chunk_index"])
        m = speech_micro_v3(row, int(r), sib, cfg, rank, int(sw), k=int(k), stores=self, vox=vox)
        T = len(m["pid"])
        assert m["segs"] == [T] and np.array_equal(m["pid"], np.arange(T)) and m["swap"] == int(sw)
        assert T == int(m["P"]) + int(row["n_frames"]) + D.CD <= BUDGET, ("segment length", T, m["P"], row["n_frames"])
        if vox:
            import audio_s4q as AU
            import rowcache_vox as RC                               # node-local decoded-row cache (SPEED FIX 10-16; bit-identical either way)
            aside = "assistant" if int(sw) else "user"               # swap 0: B = data assistant, A = data user
            e = self.vox(int(k))[int(r)]
            assert e is not None and e["row_key"] == row["row_key"] and e["row"] == int(r), ("vox index row", item, row["row_key"])
            a = e["A"][aside]
            assert int(a["n_frames"]) == int(row["n_frames"]) and int(a["frame0"]) == int(row["frame_start"]) == 0, (item, a)
            assert a["a_spk"] == row[f"{aside}_speaker_dir"], ("A audio speaker != the row's A side", a["a_spk"], row[f"{aside}_speaker_dir"])
            m["aref"] = {"row_key": row["row_key"], "side": aside, "flac": a["flac"], "gain_db": float(a["gain_db"]), "frame0": int(a["frame0"]),
                         "n": int(a["n_frames"]), "kind": a["kind"]}
            if audio:
                path = f"{FLAC_ROOT}/{a['flac']}"
                if a["kind"] == "gcs":
                    assert os.path.getsize(path) == int(a["flac_bytes"]), ("A FLAC missing / wrong size on this node", path)
                else:
                    assert a["kind"] == "mimidec" and int(a["audio_frame0"]) == 0 and os.path.getsize(path) > 0, ("A mimidec file", path)
                m["wav"] = RC.row_audio({"flac": path, "frame0": int(a["audio_frame0"]), "n_frames": int(a["n_frames"]), "row_key": row["row_key"]},
                                        float(a["gain_db"]))
                assert m["wav"].dtype == np.float32 and m["wav"].shape == (int(a["n_frames"]) * AU.SAMPLES_PER_FRAME_24K,)
        m["row_key"] = row["row_key"]
        m["item"] = [int(k), int(r), int(sw)]
        return m


def as_micro(m, dummy=False):
    """one conversation -> the trainer's micro (= data_aloe_v2.as_micro + the vox fields of data_aloe_m1vox.as_micro)."""
    out = Q.pack([m], budget=BUDGET)
    out["swap"] = int(m["swap"])
    out["item"] = json.dumps(m["item"])
    out["voice_clip"] = int(m["voice_clip"]) if not dummy else 0
    out["voice_drawn"] = int(m["voice_drawn"]) if not dummy else 0
    if dummy:
        out["dummy"] = 1
        out["own"] = np.zeros_like(out["own"])
    assert out["segs"] == [len(m["pid"])]
    if "vin" in m:
        out["present"] = m["present"]
        out["vin"] = np.asarray(m["vin"], np.int64)
        out["aref"] = json.dumps(m["aref"])
    if "wav" in m:
        out["wav"] = m["wav"]
    return out


class VoxDS:
    """one rank's training stream (= data_aloe_m1vox.VoxDS): item t = the micro of step t (dummy where the plan is exhausted)."""

    def __init__(self, items, cfg, rank, fallback, singles_at=0):
        self.items, self.cfg, self.rank, self.fb, self.singles_at = list(items), dict(cfg), int(rank), fallback, singles_at
        self.st = None

    def __len__(self):
        return len(self.items)

    def __getitem__(self, t):
        if self.st is None:
            self.st = Stores()
        it = self.items[int(t)]
        m = self.st.conv(it if it is not None else self.fb, self.cfg, self.rank)
        out = as_micro(m, dummy=it is None)
        out["step"] = int(t)
        if int(t) == self.singles_at:
            s1 = as_micro(m)
            s1.pop("wav", None)                                      # the loss check reuses the micro's own wav / latents
            out["singles"] = [s1]
        return out


def eval_micros(items, cfg, rank):
    st = Stores()
    return [as_micro(st.conv(it, cfg, rank)) for it in items]
