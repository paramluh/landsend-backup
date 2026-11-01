"""aloe SINGLE-SPEAKER FT DATA (10-24, user via main: "finetune our Aloe FT model on one speaker, all acoustic codes"; pick Ashley
TyypZEov3LSf5Gp8N6kSnBTZYXy2; "still use a voice prompt"). = 29345's data path (data_aloe_v2vox over the sam-sep stores: v2 speech micro,
channel A = Voxtral latents of the partner's raw-gain FLAC computed in the loop, no A text / Mimi, B = lane + NEW_WORD + Mimi codes, all
codebooks, no persona) with ONLY these changes, ALL gated on ALOE_SPK_ACCOUNT (unset = data_aloe_v2vox exactly, every name re-exported):
  PLAN      spk_plan.build(account, logical world, epochs / steps knobs): every chunk of every call of hers, her ALWAYS on B (swap 1 = FLIP when
            she was the data 'user' side), ~4 % of her calls held out (speaker eval), the standard Aloe eval list kept (316 items).
  CHUNKS    later chunks allowed for the speaker items (29345: first chunks only); the standard eval items keep 29345's path EXACTLY
            (data_aloe_v2vox.Stores.conv, first chunks, reference voice rule).
  VOICE     B voice clip on the SAME seeded draw rule (h64('aloe-v2-voice|<seed>|<row_key>|<swap>') % DEN < NUM; default 3 / 10 = 29345's
            30 %, ALOE_SPK_VOICE_NUM / _DEN: USER TO CONFIRM), but the clip is ALWAYS one of HER clips from a DIFFERENT call: source rows =
            her rows of her TRAIN calls other than this item's call, in a seeded order (SeedSequence(seed, h64(row_key), 0x5350, swap));
            first source row whose window (sysprompt.window with 29345's draws L0 ~ U[38, 125] frames, u; min_cov 0.5, top 0.9, stride 4,
            searched over the whole source row) is feasible. Format = 29345's B clip exactly (speech segment on track B, lane BLANKED: codes
            only, <sosb> ... <eosb>, P = 5 + L + 5). Every drawn item gets a clip (29345's later-chunk rule found one for ~43 % of draws).
  LIGHT     training light_frac = ALOE_SPK_LIGHT_FRAC (main 10-24: 1.0 = EVERY B coded frame trains the depth decoder; 29345 0.1).
            Eval cfg unchanged (light_frac 1.0, <= 2,048 seeded frames per micro in the trainer).
Everything else (v2 speech-row maths, CD 2, light seeds, A gain raw, RAWGAIN / M1VOX index asserts, budget) = 29345's code, called as is."""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.join(HERE, "lib"), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
import data_aloe_v2vox as V  # noqa: E402   (29345's module, byte-identical copy)
import spk_plan as SPL  # noqa: E402

A2, D, Q = V.A2, V.D, V.Q
STORES, SRC_STORES, DIGESTS, BUDGET = V.STORES, V.SRC_STORES, V.DIGESTS, V.BUDGET
FLAC_ROOT, INDEX, SAMSEP, RAWGAIN_PLAN_DIGEST, GAIN_MODE, RAWGAIN = V.FLAC_ROOT, V.INDEX, V.SAMSEP, V.RAWGAIN_PLAN_DIGEST, V.GAIN_MODE, V.RAWGAIN
splits, rawgain, a_gain, vox_index, a_entry, speech_micro_v2vox, as_micro, node_items = (V.splits, V.rawgain, V.a_gain, V.vox_index, V.a_entry,
                                                                                        V.speech_micro_v2vox, V.as_micro, V.node_items)
SPK = os.environ.get("ALOE_SPK_ACCOUNT", "")
SPK_EPOCHS = int(os.environ.get("ALOE_SPK_EPOCHS", "1"))
SPK_STEPS = os.environ.get("ALOE_SPK_STEPS", "auto")
SPK_EVAL_FRAC = float(os.environ.get("ALOE_SPK_EVAL_FRAC", "0.04"))
SPK_LIGHT_FRAC = float(os.environ.get("ALOE_SPK_LIGHT_FRAC", "0.1"))
VOICE_NUM = int(os.environ.get("ALOE_SPK_VOICE_NUM", str(A2.VOICE_NUM)))
VOICE_DEN = int(os.environ.get("ALOE_SPK_VOICE_DEN", str(A2.VOICE_DEN)))
SPK_SEED = 0x5350
_PLAN = {}


def aloe_cfg():
    c = V.aloe_cfg()
    if SPK:
        c = dict(c, light_frac=SPK_LIGHT_FRAC, spk_voice=f"her clip from another call (train calls), draw h64 % {VOICE_DEN} < {VOICE_NUM}, lane blanked",
                 spk_voice_num=VOICE_NUM, spk_voice_den=VOICE_DEN)
    return c


def eval_cfg():
    return dict(aloe_cfg(), light_frac=1.0)


def make_plan(world, roots=None):
    if not SPK:
        return V.make_plan(world) if roots is None else V.make_plan(world, roots=roots)
    key = (int(world), SPK, SPK_EPOCHS, SPK_STEPS, SPK_EVAL_FRAC)
    if key not in _PLAN:
        root = (roots or SRC_STORES)[0].rsplit("/", 1)[0]
        _PLAN[key] = SPL.build(SPK, int(world), epochs=SPK_EPOCHS, steps=None if SPK_STEPS == "auto" else int(SPK_STEPS), eval_frac=SPK_EVAL_FRAC,
                               root=root)
    return _PLAN[key]


def rank_items(plan, rank, world):
    """-> (train [item | None], standard eval items) of logical rank `rank` (= data_aloe_v2.rank_items); speaker eval: rank_items_spk."""
    return A2.rank_items(plan, rank, world)


def rank_items_spk(plan, rank, world):
    return [tuple(x) for x in plan["eval_spk"][rank::world]]


def her_side(plan):
    """(k, r) -> her data side, every speaker row (train + held-out)."""
    if "_side" not in plan:
        rows = SPL.speaker_rows(plan["account"], plan["stores"][0].rsplit("/", 1)[0])
        plan["_side"] = {(x["k"], x["r"]): x["side"] for x in rows}
    return plan["_side"]


def want_voice(row_key, swap, seed):
    return D.h64(f"aloe-v2-voice|{int(seed)}|{row_key}|{int(swap)}") % VOICE_DEN < VOICE_NUM


def her_clip(row, swap, cfg, plan, stores):
    """one of HER clips from a DIFFERENT call (module doc VOICE) -> (sysprompt speech segment for track B, provenance) or (None, why)."""
    sw = int(swap)
    rng = np.random.default_rng(np.random.SeedSequence([int(cfg["seed"]), D.h64(row["row_key"]) & 0x7FFFFFFFFFFF, 0x5643]))
    draws = [(int(rng.integers(int(cfg["clip_min"]), int(cfg["clip_max"]) + 1)), float(rng.random())) for _ in range(2)]
    L0, u = draws[1 ^ sw]                                           # 29345's draw for the data track on model channel B
    conv = "/".join(row["row_key"].split("/")[2:4])
    pool = [p for p in plan["clip_pool"] if p[3] != conv]
    order = np.random.default_rng(np.random.SeedSequence([int(cfg["seed"]), D.h64(row["row_key"]) & 0x7FFFFFFFFFFF, SPK_SEED, sw])).permutation(len(pool))
    wcfg = {"clip_min": cfg["clip_min"], "stride": cfg["stride"], "min_cov": cfg["min_cov"], "top_frac": cfg["top_frac"]}
    for j, i in enumerate(order):
        k2, r2, s2, c2 = pool[int(i)]
        _, st2, _ = stores.open(int(k2))
        sr = st2[int(r2)]
        assert "/".join(sr["row_key"].split("/")[2:4]) == c2 != conv, ("clip source must be another call of hers", sr["row_key"], conv)
        w = D.SPR.window(sr[f"words_{s2}"], int(sr["n_frames"]), L0, u, wcfg, lo=0)
        if w is not None:
            seg = dict(w, src="spk_other_call", src_row=sr, src_side=s2, src_key=sr["row_key"], src_chunk=int(sr["chunk_index"]), blank_lane=True,
                       side=("user", "assistant")[1 ^ sw], n_chunks=-1)
            return seg, {"src_conv": c2, "tries": j + 1, "L0": L0}
    return None, {"src_conv": None, "tries": len(order), "L0": L0}


def prefix_spk(row, swap, cfg, plan, stores):
    """= data_aloe_v2.prefix_v2 with the clip source replaced by her_clip (module doc VOICE); same asserts, same EMPTY block when not drawn."""
    sw = int(swap)
    wv = want_voice(row["row_key"], sw, cfg["voice_seed"])
    pv = {"voice_drawn": int(wv), "voice_rule": "spk_other_call"}
    segs = []
    if wv:
        cl, info = her_clip(row, sw, cfg, plan, stores)
        pv["clip_info"] = info
        if cl is not None:
            segs.append({"kind": "speech", "track": 1, **cl})
    pv["clips"] = [{"track": "AB"[s["track"]], "data_side": s["side"], "src_key": s["src_key"], "src_chunk": s["src_chunk"],
                    "frames": [s["w0"], s["w0"] + s["L"]], "cov": round(s["cov"], 3)} for s in segs]
    pv["voice_clip"] = int(bool(segs))
    fixed = 5 + sum(int(s["L"]) + 5 for s in segs)
    assert fixed <= int(cfg["sp_cap"]), ("voice clip exceeds the system-prompt cap", fixed)
    if not segs:
        segs = [{"kind": "text", "track": 1, "ids": []}]
    spx = D.SPR.build_prefix(row, segs, {"new_word": cfg["new_word"]})
    assert 5 <= spx["P"] <= int(cfg["sp_cap"]), ("system prompt length", spx["P"])
    assert pv["voice_clip"] or spx["P"] == 5, ("empty system block must be 5 positions", spx["P"])
    assert (spx["ids"][0] == D.PAD_ID).all() and not (np.asarray(spx["depth"])[0] >= 0).any(), "channel A must carry nothing in the prefix"
    pv["P"] = int(spx["P"])
    return spx, pv


def speech_micro_spk(row, node_row, cfg, rank, swap, plan, stores):
    """= data_aloe_v2.speech_micro_v2 with prefix_spk in place of prefix_v2 (same row_arrays call, same asserts), then channel A emptied in the
    conversation exactly as data_aloe_v2vox.speech_micro_v2vox."""
    sw = int(swap)
    spx, pv = prefix_spk(row, sw, cfg, plan, stores)
    P, n = int(spx["P"]), int(row["n_frames"])
    rf = P + n if not cfg["crop_conv"] else 7500
    dc = {"new_word": cfg["new_word"], "eob_gap_s": cfg["eob_gap_s"], "wpid": cfg["wpid"], "light_seed": cfg["light_seed"], "row_frames": rf,
          "frame_light": True, "no_crop": not cfg["crop_conv"]}
    a = D.DS.row_arrays(row, dc, rank, int(node_row), "ppm", float(cfg["light_frac"]), sp=spx, swap=sw)
    sel, y, word, pad, eob, nw, cb0, light = D._b_only(a)
    T = len(a["pid"])
    m = {"kind": "speech", "ids": a["ids"], "depth": a["depth"], "present": a["present"], "pid": a["pid"], "segs": [T],
         "sel": sel, "y": y, "word": word, "pad": pad, "eob": eob, "nw": nw, "n_text": float(len(sel)), "cb0": cb0, "light": light,
         "P": P, "swap": sw}
    assert (cb0[:P + D.CD] < 0).all() and (light >= P + D.CD).all() and (sel >= P).all()
    assert T == (7500 if cfg["crop_conv"] else P + n) + D.CD, ("row length", T, P, n)
    pv.update(row_key=row["row_key"], swap=sw, chan_B_data_side=("user" if sw else "assistant"), n_frames=n, T=T, n_light=int(len(light)))
    m["voice_clip"] = pv["voice_clip"]
    m["voice_drawn"] = pv["voice_drawn"]
    assert (m["depth"][0, :P] < 0).all() and (m["ids"][0, :P] == D.PAD_ID).all(), "the prefix must carry nothing on channel A"
    m["ids"] = m["ids"].copy(); m["depth"] = m["depth"].copy(); m["present"] = m["present"].copy()
    m["ids"][0, P:] = D.PAD_ID
    m["depth"][0] = -1
    m["present"][0, P:] = 0.0
    m["vin"] = (P, n)
    pv["chan_A"] = "voxtral_latents_only"
    m["prov"] = json.dumps(pv)
    return m


class Stores(V.Stores):
    """= data_aloe_v2vox.Stores; conv(..., spk=True) = the speaker micro (any chunk, her on B asserted, her-other-call voice clip), spk=False =
    29345's micro exactly (data_aloe_v2vox.Stores.conv: first chunks, reference voice rule)."""

    def conv(self, item, cfg, rank, audio=True, spk=None):
        if spk is None:
            spk = bool(SPK)
        if not spk:
            return super().conv(item, cfg, rank, audio=audio)
        import audio_s4q as AU
        import rowcache_vox as RC
        plan = make_plan(_WORLD[0])
        k, r, sw = item
        node, st, sib = self.open(int(k))
        row = st[int(r)]
        D.R3.check_row(row)
        side = her_side(plan)[(int(k), int(r))]
        assert int(sw) == (0 if side == "assistant" else 1), ("the speaker must be on B", row["row_key"], side, sw)
        m = speech_micro_spk(row, int(r), cfg, rank, int(sw), plan, self)
        T = len(m["pid"])
        assert m["segs"] == [T] and np.array_equal(m["pid"], np.arange(T)) and m["swap"] == int(sw)
        assert T == int(m["P"]) + int(row["n_frames"]) + D.CD <= BUDGET, ("segment length", T, m["P"], row["n_frames"])
        aside, a = a_entry(item, row)
        assert aside != side, ("A must be the partner", row["row_key"])
        g_db = a_gain(a)
        m["aref"] = {"row_key": row["row_key"], "side": aside, "flac": a["flac"], "gain_db": g_db, "frame0": int(a["frame0"]),
                     "n": int(a["n_frames"]), "kind": a["kind"], "gain_mode": GAIN_MODE, "gain_mimi_db": float(a["gain_db"])}
        if audio:
            path = f"{FLAC_ROOT}/{a['flac']}"
            assert a["kind"] == "gcs" and os.path.getsize(path) == int(a["flac_bytes"]), ("A FLAC missing / wrong size on this node", path)
            m["wav"] = RC.row_audio({"flac": path, "frame0": int(a["audio_frame0"]), "n_frames": int(a["n_frames"]), "row_key": row["row_key"]},
                                    g_db)
            assert m["wav"].dtype == np.float32 and m["wav"].shape == (int(a["n_frames"]) * AU.SAMPLES_PER_FRAME_24K,)
        m["row_key"] = row["row_key"]
        m["item"] = [int(k), int(r), int(sw)]
        return m


_WORLD = [64]


def set_world(w):
    """the logical world the speaker plan is built for (the trainer calls this before any Stores / AccDS use; DataLoader workers inherit it)."""
    _WORLD[0] = int(w)


class AccDS(V.AccDS):
    """= data_aloe_v2vox.AccDS with the speaker Stores (spk=True for every training micro when ALOE_SPK_ACCOUNT is set)."""

    def __getitem__(self, i):
        if self.st is None:
            self.st = Stores()
        return super().__getitem__(i)


def eval_micros(items, cfg, rank, spk=False):
    st = Stores()
    return [as_micro(st.conv(it, cfg, rank, spk=spk)) for it in items]
