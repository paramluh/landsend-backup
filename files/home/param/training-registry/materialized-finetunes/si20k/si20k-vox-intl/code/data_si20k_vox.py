"""si20k-vox DATA (10-19, user): the Aloe Voxtral recipe's data maths (data_aloe_v2vox = data_aloe_v2 + m1vox's channel-A edit, jobs 23081 /
24193) on the spoken-instruct 20k TTS rows (qwen-1ch-data/si20k-vox, build_si20k_rows.py), with ONLY the data and the new prompt rule changed:
  ROWS      every row of the si20k stores s20n0..3 (all chunks: a conversation over 7,500 frames gives c00 + a c01 tail row of >= 1,500 frames),
            swap 0 ONLY: B = data 'assistant', A = data 'user' (standing rule: no A / B swap; the TTS rendered the roles). Held-out = the
            builder's stratified eval conversations (split == 'eval'), swap 0. Plan = data_aloe_v2.make_plan's rule (seed 20261011, items
            permuted, item t x WORLD + R, steps = ceil(items / WORLD), dummies on the last step).
  SYSPROMPT the NEW standard prompt rule (user 10-19, all future runs): two INDEPENDENT seeded draws per row:
              voice clip for B with p 0.8: u_v = h64('si20k-voice|<voice_seed>|<row_key>') / 2^64 < 0.8
              persona text for B with p 0.2: u_p = h64('si20k-persona|<persona_seed>|<row_key>') / 2^64 < 0.2
            CLIP = the exact tt-v3 cloning reference that voiced B in this conversation (manifest voice.B -> its bank Mimi codes, staged in
            the store's voiceclip_B.npz, sha checked): the generator's bank_segments form (sysprompt.bank_sources): whole clip (w0 0, L =
            64..190 frames), blank lane (PAD, no NEW_WORD), codes at the code delay inside <sosb> .. <eosb>, on model channel B.
            PERSONA = B's persona sidecar sections (persona_text_s3q 'all' block <sop> .. <eop> on channel B, cropped to the sp cap) when the
            sidecar has one; spoken-instruct carries NO assistant persona / system text, so every row's B sidecar is 'missing' and the
            realised persona fraction is 0 (drawn ~0.2, used 0; logged). A never gets a clip or text. Order [clip B] [persona B]; neither ->
            the EMPTY system block (P = 5) = data_ppm.podcast_prefix's empty case.
  CHANNEL A = data_aloe_v2vox.speech_micro_v2vox's edit: no text (PAD), no Mimi (depth -1 everywhere on A), present_A = 0 at p >= P, vin = (P, n):
            the trainer adds s x vadapter(Voxtral(A audio of row frames 0..n-1)) at positions P..P+n-1.
  A AUDIO   RAW (VOX_A_GAIN=raw: no loudness normalisation): channel 0 (user) of the conversation's stereo PCM16 24 kHz TTS FLAC (the primary
            render, read in place from /data), samples [1920 frame0, 1920 (frame0 + n)), float32 = PCM16 / 32768 (libsndfile), gain 0 dB.
            Per-row ledger A_LEDGER.json in each store (path, byte size checked on every read; mtime checked at preflight).
  LIGHT / EVAL / everything else = data_aloe_v2 (light_frac 0.1 train, 1.0 eval; PPM speech-row maths, CD 2, crop_conv False, budget asserts).
AccDS = data_aloe_v2vox.AccDS (one physical rank emulating logical ranks of the 64; with 8 nodes it reduces to one slot)."""
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
NST = 4
SRC_ROOT = os.environ.get("ALOE_SI20K_SRC", "/data/home/amartyavarma/qwen-1ch-data/si20k-vox/stores")
STORE_ROOT = os.environ.get("ALOE_SI20K_STORE_ROOT", "/scratch/amartyavarma/si20k-vox/stores")
STORES = [f"{STORE_ROOT}/s20n{k}" for k in range(NST)]
SRC_STORES = [f"{SRC_ROOT}/s20n{k}" for k in range(NST)]
STORES_JSON = os.environ.get("ALOE_SI20K_STORES_JSON", f"{SRC_ROOT}/STORES.json")
PLAN_SEED = 20261011                              # = data_aloe_v2
VOICE_SEED = 20261011
PERSONA_SEED = 20261011                           # independent of the voice draw by its salt ('si20k-persona' vs 'si20k-voice')
VOICE_P, PERSONA_P = 0.8, 0.2                     # the 10-19 standard prompt rule
BUDGET = 2000 + 7500 + D.CD                       # = data_aloe_v2
GAIN_MODE = os.environ.get("VOX_A_GAIN", "raw")
assert GAIN_MODE == "raw", ("si20k: the reference's VOX_A_GAIN is raw (no loudness normalisation of channel A)", GAIN_MODE)
# SI20K_A_GAIN (user 10-18 PT): the switchable gain step on channel A's waveform before Voxtral, computed IN THE DATA COLLATOR (DataLoader
# worker) from the row's own waveform + word timings; never precomputed (no gain files, no stored levels, no rewritten audio):
#   raw          0 dB, the TTS render's own level
#   fixed:<dB>   a constant gain
#   lufs:<LUFS>  per row: voxtral_front.lufs_gain_db(row waveform, target) (the -23 LUFS / peak-limited Mimi rule)
#   normal       mimic user mics at different volumes (user 10-18 PT): per CONVERSATION a target active-speech level
#                ~ N(SI20K_A_LEVEL_MEAN, SI20K_A_LEVEL_SD) clipped to +-2.5 SD (fitted offline on REAL user mics: Aloe handover s2h-m1
#                channel A, tools/fit_user_levels.py; the only offline numbers), seeded by (seed, conversation key, epoch): a fresh draw
#                per pass, reproducible; gain = target - the row's own active-speech level (constant over the conversation: the filter's
#                fade cut stays relative); peak-limited so max |x| <= -1 dBFS (the gain is lowered, counted). Train only; eval uses
#                level:<mean> (the median gain) and a low-mic eval split level:<mean - 2 SD>.
#   level:<dB>   a fixed target active-speech level (eval)
# ACTIVE-SPEECH LEVEL (= the filter's user-span dBFS idea, from word timings): user turns = runs of user word onsets with gaps <= 2.0 s,
#   span = [first onset, last onset + 0.6 s]; level = 10 log10(mean x^2) over the concatenated spans (onsets only, so TTS lane words and
#   real WLT words are measured the same way).
A_GAIN = os.environ.get("SI20K_A_GAIN", "raw")
#   gauss:<SD>   THE CHOSEN MODE (user 10-18 PT, delegated): per CONVERSATION gain ~ N(0, SD dB) around its own level, clipped to
#                +-2.5 SD, seeded (seed, conversation key, epoch), peak-limited; train only (eval: own level + a low-mic split at -2 SD).
#   spread:<dB>  the alternative the user may pick (10-18 PT): per CONVERSATION a gain ~ U(-dB, +dB) around the conversation's own level
#                (seeded like normal, peak-limited the same way); no population fit.
assert A_GAIN in ("raw", "normal") or A_GAIN.startswith(("fixed:", "lufs:", "level:", "spread:", "gauss:")), \
    ("SI20K_A_GAIN: raw | fixed:<dB> | lufs:<LUFS> | normal | gauss:<SD> | spread:<dB> | level:<dB>", A_GAIN)
A_LEVEL_MEAN = float(os.environ.get("SI20K_A_LEVEL_MEAN", "nan"))
A_LEVEL_SD = float(os.environ.get("SI20K_A_LEVEL_SD", "nan"))
assert A_GAIN != "normal" or (A_LEVEL_MEAN == A_LEVEL_MEAN and A_LEVEL_SD > 0), ("normal gain needs SI20K_A_LEVEL_MEAN / _SD", A_LEVEL_MEAN, A_LEVEL_SD)
GAIN_SEED, CLIP_SD, PEAK_DBFS, TURN_GAP_S, TURN_TAIL_S = 20261018, 2.5, -1.0, 2.0, 0.6
SPF = 1920
# SI20K_A_CORRUPT (user 10-18 PT): qqa-corrupt-v2 on the Voxtral-side waveform in the collator (acorrupt.py), train only, BEFORE the mic gain
A_CORRUPT = os.environ.get("SI20K_A_CORRUPT", "none")
assert A_CORRUPT in ("none", "qqa-corrupt-v2"), A_CORRUPT


def active_level_db(w, onsets_s, sr=24000):
    """active-speech level (module comment) of waveform w given the user word onsets (s, row-relative) -> dBFS or None."""
    on = sorted(float(t) for t in onsets_s)
    if not on:
        return None
    spans, a, b = [], on[0], on[0]
    for t in on[1:]:
        if t - b > TURN_GAP_S:
            spans.append((a, b + TURN_TAIL_S)); a = t
        b = t
    spans.append((a, b + TURN_TAIL_S))
    seg = [w[int(x * sr):min(len(w), int(y * sr))] for x, y in spans]
    seg = [x for x in seg if x.size]
    if not seg:
        return None
    x = np.concatenate(seg).astype(np.float64)
    return float(10.0 * np.log10(np.mean(x * x) + 1e-12))


def conv_key(row_key):
    return D.SPR.rec_key(row_key)


def gain_db(w, rule, row_key, onsets_s, cfg):
    """the SI20K_A_GAIN rule (cfg['a_gain']) for one row -> (gain dB, info)."""
    info = {"rule": rule, "level_db": None, "target_db": None, "limited": 0, "no_level": 0}
    if rule == "raw":
        g = 0.0
    elif rule.startswith("fixed:"):
        g = float(rule.split(":", 1)[1])
    elif rule.startswith("gauss:"):
        rng = np.random.default_rng(np.random.SeedSequence([int(cfg["a_gain_seed"]), D.h64(conv_key(row_key)) & 0x7FFFFFFFFFFF, int(cfg["a_epoch"])]))
        sd_ = float(rule.split(":", 1)[1])
        g = sd_ * float(np.clip(rng.standard_normal(), -float(cfg["a_gain_clip_sd"]), float(cfg["a_gain_clip_sd"])))
    elif rule.startswith("spread:"):
        rng = np.random.default_rng(np.random.SeedSequence([int(cfg["a_gain_seed"]), D.h64(conv_key(row_key)) & 0x7FFFFFFFFFFF, int(cfg["a_epoch"])]))
        x_ = float(rule.split(":", 1)[1])
        g = float(rng.uniform(-x_, x_))
        info["level_db"] = active_level_db(w, onsets_s)
    elif rule.startswith("lufs:"):
        import voxtral_front as VF
        try:
            g, _l, _s = VF.lufs_gain_db(w, target=float(rule.split(":", 1)[1])); g = float(g)
        except (ValueError, AssertionError):
            g = 0.0
    else:
        lv = active_level_db(w, onsets_s)
        info["level_db"] = lv
        if rule == "normal":
            rng = np.random.default_rng(np.random.SeedSequence([int(cfg["a_gain_seed"]), D.h64(conv_key(row_key)) & 0x7FFFFFFFFFFF, int(cfg["a_epoch"])]))
            z = float(np.clip(rng.standard_normal(), -float(cfg["a_gain_clip_sd"]), float(cfg["a_gain_clip_sd"])))
            tgt = float(cfg["a_level_mean"]) + float(cfg["a_level_sd"]) * z
        else:
            assert rule.startswith("level:"), rule
            tgt = float(rule.split(":", 1)[1])
        info["target_db"] = tgt
        if lv is None:
            g = 0.0; info["no_level"] = 1
        else:
            g = tgt - lv
    pk = float(np.max(np.abs(w))) if w.size else 0.0
    if pk > 0 and 20.0 * np.log10(pk) + g > float(cfg.get("a_peak_dbfs", PEAK_DBFS)):
        g = float(cfg.get("a_peak_dbfs", PEAK_DBFS)) - 20.0 * np.log10(pk); info["limited"] = int(info["rule"] not in ("raw",))
    info["gain_db"] = float(g)
    return float(g), info


def a_gain_cfg(rule):
    return {"a_gain": rule, "a_level_mean": A_LEVEL_MEAN, "a_level_sd": A_LEVEL_SD, "a_gain_clip_sd": CLIP_SD, "a_peak_dbfs": PEAK_DBFS,
            "a_gain_seed": GAIN_SEED, "a_epoch": 0, "a_turn_gap_s": TURN_GAP_S, "a_turn_tail_s": TURN_TAIL_S}


def aloe_cfg():
    """training data cfg = data_aloe_v2.aloe_cfg() with the prompt keys of the new rule (persona_sides B, voice 0.8 / persona 0.2) + the
    channel-A gain keys (SI20K_A_GAIN)."""
    return Q.qqa_cfg(persona_sides="B", light_frac=0.1, sysprompt="si20k_bankclipB_p0.8_personaB_p0.2", voice_frac=VOICE_P, voice_seed=VOICE_SEED,
                     persona_frac=PERSONA_P, persona_seed=PERSONA_SEED, a_corrupt=A_CORRUPT, **a_gain_cfg(A_GAIN))


def eval_cfg():
    """eval: light 1.0; normal gain -> the fixed median target (level:<mean>), else the train rule."""
    rule = f"level:{A_LEVEL_MEAN:.2f}" if A_GAIN == "normal" else ("raw" if A_GAIN.startswith(("spread:", "gauss:")) else A_GAIN)   # own level
    return dict(aloe_cfg(), light_frac=1.0, a_corrupt="none", **a_gain_cfg(rule))


def eval_low_cfg():
    """the low-mic eval split (normal gain only): target level mean - 2 SD (quiet-mic robustness)."""
    if A_GAIN == "normal":
        return dict(eval_cfg(), **a_gain_cfg(f"level:{A_LEVEL_MEAN - 2.0 * A_LEVEL_SD:.2f}"))
    if A_GAIN.startswith("gauss:"):                                     # -2 SD
        return dict(eval_cfg(), **a_gain_cfg(f"fixed:{-2.0 * float(A_GAIN.split(':', 1)[1]):.2f}"))
    if A_GAIN.startswith("spread:"):                                    # the low end of the spread
        return dict(eval_cfg(), **a_gain_cfg(f"fixed:{-float(A_GAIN.split(':', 1)[1]):.2f}"))
    return None


def stores_meta():
    j = json.load(open(STORES_JSON))
    assert j["format"] == "si20k-vox-stores-v1" and len(j["nodes"]) == NST, (STORES_JSON, j.get("format"))
    return j


def splits(roots=STORES):
    """-> {'train': [(k, r)], 'eval': [(k, r)]} over every row (row == node row asserted)."""
    import pyarrow.parquet as pq
    out = {"train": [], "eval": [], "n_all": {"train": 0, "eval": 0}}
    for k, root in enumerate(roots):
        t = pq.read_table(f"{root}/rows.parquet", columns=["row", "split"]).to_pydict()
        assert t["row"] == list(range(len(t["row"]))), root
        for r, s in zip(t["row"], t["split"]):
            out[s].append((k, int(r))); out["n_all"][s] += 1
    return out


def make_plan(world, roots=STORES):
    """= data_aloe_v2.make_plan with swap 0 only (no orientation doubling)."""
    sp = splits(roots)
    items = [(k, r, 0) for (k, r) in sp["train"]]
    perm = np.random.default_rng(PLAN_SEED).permutation(len(items))
    items = [items[i] for i in perm]
    steps = -(-len(items) // world)
    ev = [(k, r, 0) for (k, r) in sp["eval"]]
    return {"items": items, "steps": steps, "eval": ev, "n_train_rows": len(sp["train"]), "n_eval_rows": len(sp["eval"]), "n_all": sp["n_all"]}


def rank_items(plan, rank, world):
    it = plan["items"]
    tr = [it[t * world + rank] if t * world + rank < len(it) else None for t in range(plan["steps"])]
    return tr, plan["eval"][rank::world]


def _u(s):
    return D.h64(s) / 2.0 ** 64


def want_voice(row_key, seed=VOICE_SEED):
    return _u(f"si20k-voice|{int(seed)}|{row_key}") < VOICE_P


def want_persona(row_key, seed=PERSONA_SEED):
    return _u(f"si20k-persona|{int(seed)}|{row_key}") < PERSONA_P


def prefix_si20k(row, clip, pers, cfg):
    """-> (sysprompt.build_prefix dict, provenance). clip = (voice_id, codes int [32, L], sha) of B's bank reference; pers = sidecar or None."""
    assert float(cfg["voice_frac"]) == VOICE_P and float(cfg["persona_frac"]) == PERSONA_P and cfg["persona_sides"] == "B", cfg
    wv, wp = want_voice(row["row_key"], cfg["voice_seed"]), want_persona(row["row_key"], cfg["persona_seed"])
    pv = {"voice_drawn": int(wv), "persona_drawn": int(wp)}
    segs = []
    if wv:
        vid, cd, sha = clip
        cd = np.asarray(cd, np.int64)
        assert cd.ndim == 2 and cd.shape[0] == 32 and int(cd.min()) >= 0 and int(cd.max()) < 2048, (row["row_key"], cd.shape)
        segs.append({"kind": "speech", "track": 1, "side": "assistant", "w0": 0, "L": int(cd.shape[1]), "cov": 1.0, "src": "bank",
                     "src_row": {"codes_assistant": cd}, "src_side": "assistant", "src_key": f"bank:{vid}", "src_chunk": None, "blank_lane": True})
    pv["clips"] = [{"track": "B", "data_side": "assistant", "src_key": s["src_key"], "frames": [0, s["L"]], "sha": clip[2]} for s in segs]
    pv["voice_clip"] = int(bool(segs))
    fixed = 5 + sum(int(s["L"]) + 5 for s in segs)
    assert fixed <= int(cfg["sp_cap"]), ("voice clip exceeds the system-prompt cap", fixed)
    pv["persona_used"] = 0
    if wp and pers is not None:                                       # B persona block (data_ppm.podcast_prefix's text rule, channel B only)
        av = D.PT.avail(pers.get("assistant"))
        keys = [k for k in D.PT.SECTIONS if k in av]
        if keys:
            D._tok_init()
            ids = D.PT.encode(D.PT.block_text(av, keys))
            b2, cut = D.crop_blocks([[D.PT.SOP] + ids + [D.PT.EOP]], int(cfg["sp_cap"]) - fixed)
            if b2[0] is not None:
                segs.append({"kind": "text", "track": 1, "ids": b2[0]})
                pv["persona_used"] = 1; pv["persona_tokens"] = len(b2[0]); pv["persona_cut"] = int(cut)
    if not segs:                                                     # EMPTY system block: header + footer only (podcast_prefix)
        segs = [{"kind": "text", "track": 1, "ids": []}]
    spx = D.SPR.build_prefix(row, segs, {"new_word": cfg["new_word"]})
    assert 5 <= spx["P"] <= int(cfg["sp_cap"]), ("system prompt length", spx["P"])
    assert pv["voice_clip"] or pv["persona_used"] or spx["P"] == 5, ("empty system block must be 5 positions", spx["P"])
    assert (spx["ids"][0] == D.PAD_ID).all() and not (np.asarray(spx["depth"])[0] >= 0).any(), "channel A must carry nothing in the prefix"
    if pv["voice_clip"]:                                             # the clip block holds exactly B's bank codes (code delay inside the block)
        lay = spx["layout"][0]
        assert lay["kind"] == "speech" and lay["track"] == 1 and lay["L"] == segs[0]["L"] and lay["n_words"] == 0
    pv["P"] = int(spx["P"])
    return spx, pv


def speech_micro_si20k(row, node_row, clip, pers, cfg, rank):
    """= data_aloe_v2.speech_micro_v2 (swap 0) with prefix_si20k, then channel A emptied (= data_aloe_v2vox.speech_micro_v2vox)."""
    sw = 0
    spx, pv = prefix_si20k(row, clip, pers, cfg)
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
    pv.update(row_key=row["row_key"], swap=sw, chan_B_data_side="assistant", n_frames=n, T=T, n_light=int(len(light)))
    m["voice_clip"], m["voice_drawn"] = pv["voice_clip"], pv["voice_drawn"]
    m["persona_used"], m["persona_drawn"] = pv["persona_used"], pv["persona_drawn"]
    assert (m["depth"][0, :P] < 0).all() and (m["ids"][0, :P] == D.PAD_ID).all(), "the prefix must carry nothing on channel A"
    m["ids"] = m["ids"].copy(); m["depth"] = m["depth"].copy(); m["present"] = m["present"].copy()
    m["ids"][0, P:] = D.PAD_ID
    m["depth"][0] = -1
    m["present"][0, P:] = 0.0
    m["vin"] = (P, n)
    pv["chan_A"] = "voxtral_latents_only"
    m["prov"] = json.dumps(pv)
    return m


def a_audio(e, far=False):
    """channel 0 (user) of the stereo TTS FLAC, row frames [frame0, frame0 + n) -> float32 [n x 1920], gain 0 dB (RAW); far=True: + channel 1."""
    import soundfile as sf
    s0, L = int(e["frame0"]) * SPF, int(e["n_frames"]) * SPF
    x, sr = sf.read(e["flac"], start=s0, frames=L, dtype="float32", always_2d=True)
    assert sr == 24000 and x.shape[1] == 2, ("TTS FLAC must be stereo 24 kHz", e["flac"], sr, x.shape)
    assert x.shape[0] > L - 6000, ("FLAC shorter than the row by > 0.25 s", e["flac"], x.shape[0], L)
    w = np.zeros(L, np.float32)
    w[:x.shape[0]] = x[:, 0]
    assert float(e.get("gain_db", 0.0)) == 0.0                         # the ledger records the render level (no gain baked in)
    if far:
        b = np.zeros(L, np.float32); b[:x.shape[0]] = x[:, 1]
        return w, b
    return w


class Stores:
    """the 4 node-local si20k stores (+ persona sidecar, B voice-clip sidecar, A ledger), opened lazily (per DataLoader worker)."""

    def __init__(self, roots=STORES):
        self.roots, self.s = list(roots), {}
        self.meta = stores_meta()

    def open(self, k):
        if k not in self.s:
            node = D.R3.load_node(self.roots[k])
            m = self.meta["nodes"][f"s20n{k}"]
            assert node["digest"] == m["node_digest"], ("staged store differs from the build", self.roots[k], node["digest"], m["node_digest"])
            st = D.R3.Rows(node)
            ps = D.PER.PersonaSidecar(node)
            z = np.load(f"{self.roots[k]}/voiceclip_B.npz")
            vc = {"codes": z["codes"], "off": z["off"], "row_key": z["row_key"], "voice_id": z["voice_id"], "sha": z["sha"]}
            assert len(vc["row_key"]) == len(node["length"]) and len(vc["off"]) == len(node["length"]) + 1
            led = json.load(open(f"{self.roots[k]}/A_LEDGER.json"))
            assert led["format"] == "si20k-a-ledger-v1"
            self.s[k] = (node, st, ps, vc, led["rows"])
        return self.s[k]

    def clip(self, k, r, row_key):
        vc = self.open(k)[3]
        assert str(vc["row_key"][r]) == row_key, ("voice-clip sidecar misaligned", k, r, vc["row_key"][r], row_key)
        a, b = int(vc["off"][r]), int(vc["off"][r + 1])
        return str(vc["voice_id"][r]), vc["codes"][:, a:b], str(vc["sha"][r])

    def conv(self, item, cfg, rank, audio=True):
        k, r, sw = item
        assert int(sw) == 0, ("si20k: swap 0 only (B = data assistant)", item)
        node, st, ps, vc, led = self.open(int(k))
        row = st[int(r)]
        D.R3.check_row(row)
        try:
            pers = ps.get(int(r), row["row_key"])
        except (FileNotFoundError, OSError):
            pers = None
        m = speech_micro_si20k(row, int(r), self.clip(int(k), int(r), row["row_key"]), pers, cfg, rank)
        T = len(m["pid"])
        assert m["segs"] == [T] and np.array_equal(m["pid"], np.arange(T)) and m["swap"] == 0
        assert T == int(m["P"]) + int(row["n_frames"]) + D.CD <= BUDGET, ("segment length", T, m["P"], row["n_frames"])
        e = led[row["row_key"]]
        assert int(e["frame0"]) == int(row["frame_start"]) and int(e["n_frames"]) == int(row["n_frames"]) and int(e["channel"]) == 0, (item, e)
        m["aref"] = {"row_key": row["row_key"], "side": "user", "flac": e["flac"], "gain_db": 0.0, "frame0": int(e["frame0"]),
                     "n": int(e["n_frames"]), "kind": "tts_ch0", "gain_mode": GAIN_MODE, "si20k_a_gain": cfg.get("a_gain", "raw")}
        if audio:
            assert os.path.getsize(e["flac"]) == int(e["flac_bytes"]), ("A FLAC changed / missing since the build", e["flac"])
            w, fe = a_audio(e, far=True)
            ci = {"mode": "none"}
            if cfg.get("a_corrupt", "none") == "qqa-corrupt-v2":            # training corruption first (in the collator, per row + epoch)
                import acorrupt as AC
                w, ci = AC.apply(w, row["row_key"], int(cfg.get("a_epoch", 0)), far=fe)
            g, gi = gain_db(w, cfg.get("a_gain", "raw"), row["row_key"], [x["start"] for x in row["words_user"]], cfg)   # then the mic gain
            gi["corrupt"] = ci
            if g != 0.0:
                w = (w * np.float32(10.0 ** (g / 20.0))).astype(np.float32)
            m["wav"] = w
            m["a_gain"] = gi
            m["aref"]["gain_db"] = float(g); m["aref"]["gain_rule"] = gi["rule"]
            assert m["wav"].dtype == np.float32 and m["wav"].shape == (int(e["n_frames"]) * SPF,)
        m["row_key"] = row["row_key"]
        m["item"] = [int(k), int(r), 0]
        return m


def as_micro(m, dummy=False):
    """= data_aloe_v2vox.as_micro (data_aloe_v2.as_micro + present / vin / aref / wav) + the persona flags."""
    out = Q.pack([m], budget=BUDGET)
    out["swap"] = int(m["swap"])
    out["item"] = json.dumps(m["item"])
    for f_ in ("voice_clip", "voice_drawn", "persona_used", "persona_drawn"):
        out[f_] = int(m[f_]) if not dummy else 0
    if dummy:
        out["dummy"] = 1
        out["own"] = np.zeros_like(out["own"])
    assert out["segs"] == [len(m["pid"])]
    out["present"] = m["present"]
    out["vin"] = np.asarray(m["vin"], np.int64)
    out["aref"] = json.dumps(m["aref"])
    if "wav" in m:
        out["wav"] = m["wav"]
    gi = m.get("a_gain") or {}
    out["a_gain_db"] = float(gi.get("gain_db", 0.0)) if not dummy else 0.0
    out["a_gain_limited"] = int(gi.get("limited", 0)) if not dummy else 0
    out["a_gain_no_level"] = int(gi.get("no_level", 0)) if not dummy else 0
    return out


class AccDS:
    """= data_aloe_v2vox.AccDS (one PHYSICAL rank's stream emulating logical ranks `lranks`; None = pad slot -> dummy micro)."""

    def __init__(self, rows_by_slot, lranks, cfg, fallback, singles_at=0):
        assert len(rows_by_slot) == len(lranks) and lranks[0] is not None
        self.rows, self.lr, self.cfg, self.fb, self.singles_at = [list(r) if r is not None else None for r in rows_by_slot], list(lranks), dict(cfg), fallback, singles_at
        self.G = len(lranks)
        self.T = len(self.rows[0])
        assert all(r is None or len(r) == self.T for r in self.rows)
        self.st = None

    def __len__(self):
        return self.T * self.G

    def __getitem__(self, i):
        if self.st is None:
            self.st = Stores()
        t, g = divmod(int(i), self.G)
        L = self.lr[g]
        pad = L is None
        it = None if pad else self.rows[g][t]
        rk = self.lr[0] if pad else L
        m = self.st.conv(it if it is not None else self.fb, self.cfg, rk)
        out = as_micro(m, dummy=it is None)
        out["step"], out["slot"], out["lrank"], out["padslot"] = t, g, (-1 if pad else int(L)), int(pad)
        if t == self.singles_at:
            s1 = as_micro(m)
            s1.pop("wav", None)
            out["singles"] = [s1]
        return out


def eval_micros(items, cfg, rank):
    st = Stores()
    return [as_micro(st.conv(it, cfg, rank)) for it in items]


def prompt_fractions(plan, roots=STORES):
    """dry run of the prompt rule over the plan (cheap: hashes + sidecar status only) -> counts per split."""
    st = Stores(roots)
    out = {}
    for name, its in (("train", plan["items"]), ("eval", plan["eval"])):
        c = {"n": 0, "voice_drawn": 0, "voice_clip": 0, "persona_drawn": 0, "persona_avail": 0, "persona_used": 0, "both": 0, "neither": 0}
        for k, r, _ in its:
            node, rows_, ps, vc, _ = st.open(int(k))
            rk = str(vc["row_key"][int(r)])
            v, p = want_voice(rk), want_persona(rk)
            av = bool(D.PT.avail(ps.get(int(r), rk).get("assistant")))
            c["n"] += 1; c["voice_drawn"] += v; c["voice_clip"] += v; c["persona_drawn"] += p; c["persona_avail"] += av
            c["persona_used"] += p and av; c["both"] += v and p and av; c["neither"] += (not v) and not (p and av)
        out[name] = c
    return out


def node_items(plan, node, world, lw=8):
    """= data_aloe_v2vox.node_items: every item node `node`'s ranks read under this plan (train incl. the dummy fallback, eval) -> sorted list."""
    out = set()
    for R in range(node * lw, (node + 1) * lw):
        tr, ev = rank_items(plan, R, world)
        out |= {tuple(x) for x in tr if x is not None} | {tuple(x) for x in ev}
        if any(x is None for x in tr):
            out.add(tuple(plan["items"][0]))
    return sorted(out)
