"""si20k-txt (10-19, text-out rerun): = the si20k-vox podcast rows below with the B reference voice clip DROPPED (user decision 10-19,
SI20K_B_CLIP=drop asserted): podcast_prefix_bonly still makes the pretraining voice draw (logged b_clip_dropped) but builds no clip segment;
B's persona block (same draw, same text) is unchanged. B's Mimi codes stay in the micro arrays (never read by the text-out model).
si20k-vox CO-TRAINING podcast rows (10-18 PT, user: "co-train on the si20k rows plus UNSEEN pretraining podcast data at 1:1, to maintain
pretraining ability"). CPU only (DataLoader workers).

PLAN = plan/PODCAST_PLAN_<model>.json (tools/podcast_plan_<model>.py): the model's OWN pretraining speech-row stream reproduced per pretraining
  rank; the rows each rank would have read AFTER the stop step (PPM ppm-v1: after step 12,829; INTL ppm-next: after 6,242), verified unseen
  (no row key consumed by any rank through the stop step). Rows are node-local on the pretraining hosts (/scratch/amartyavarma/s3-rows +
  D2 extra slots): a fine-tune rank (host H, local rank j) reads the unseen continuation of the pretraining rank that ran on (H, j), in
  stream order: step t -> unseen_codes[t]. Eval = that pretraining rank's own eval_speech rows (what pretraining logged as eval_speech).
ROWS (coordinator 10-18 PT, unless the user overrides): BOTH fine-tunes use the SAME rows = the 1,163,344 rows NEITHER model trained on (PPM
  ppm-v1 stream positions >= 25,658 = 12,829 x 2 per rank; the 14B's per-rank streams are identical and it consumed only positions < 12,484,
  so its positions 12,484..25,657, which PPM trained on, are skipped): both runs read plan/PODCAST_PLAN_ppm.json.
MICRO = the model's pretraining speech-micro maths EXACTLY (data_ppm.MixedSpeechDS.get / speech_micro with the pretraining data cfg and the
  pretraining rank argument: random side, voice clips of both speakers from the recording's later chunks, both personas, 7,500-frame crop),
  i.e. the NATIVE A input: PPM A = partner words + Mimi (channel A tables, present 1); INTL A = partner words in the user_text slot (model
  path 'base' = model_ilp.ILP.forward). NOT Voxtral: the si20k rows keep Voxtral-A; podcast rows use the native A path (the alternative,
  Voxtral-encoding the podcast A channel too, is not what pretraining saw). Same loss terms / weights as the si20k micros (= pretraining's:
  factorised text CE word x5, cb0 1, light 1). ONE knob differs from pretraining and follows the fine-tune recipe: light_frac 0.1 in training
  (pretraining 1.0 with decoder K 8; the recipe's decoder is K 1 on ~10 % of B's frames, the light-ratio band check covers every micro);
  eval micros keep the eval light definition (light_frac 1.0, <= 2,048 seeded frames), = pretraining's eval."""
import hashlib
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
BUDGET = 2000 + 7500 + D.CD
TRAIN_LIGHT_FRAC = 0.1
B_CLIP = os.environ.get("SI20K_B_CLIP", "drop")                  # si20k-txt: the B voice clip is dropped (user 10-19)
assert B_CLIP == "drop", ("si20k-txt: SI20K_B_CLIP must be drop", B_CLIP)


STAGED = os.environ.get("SI20K_PODCAST_STAGED", "/data/home/amartyavarma/ppm/si20k-vox-ppm/plan/PODCAST_STAGED.json")


def load_plan(path):
    """the model's plan + the STAGED overlay (hosts whose own slice is gone, slinky-0 / 1 / 2: whole shards of another host's slice staged to
    their /scratch, PPM-unseen rows = also unseen by the 14B, whose consumed rows are a prefix of PPM's; no pretraining eval rows there)."""
    j = json.load(open(path))
    assert j["format"] == "si20k-podcast-plan-v1", (path, j.get("format"))
    if STAGED and os.path.exists(STAGED):
        o = json.load(open(STAGED))
        assert o["format"] == "si20k-podcast-staged-v1", (STAGED, o.get("format"))
        j["staged"] = {h: dict(e, staged=True) for h, e in o["hosts"].items()}
        j["staged_overlay"] = {"path": STAGED, "sha": plan_sha(STAGED), "hosts": sorted(o["hosts"])}
    return j


def native_ok(hp):
    """this host's own pretraining slice (and its D2 extra slot) complete on its /scratch: NODE.json, every shard's .DONE and codes files."""
    root = hp["native_root"]
    try:
        node = json.load(open(f"{root}/NODE.json"))
        for sh in node["shards"]:
            if not (os.path.isfile(f"{root}/{sh}.DONE") and os.path.getsize(f"{root}/{sh}.codes_user.npy") > 0
                    and os.path.getsize(f"{root}/{sh}.codes_assistant.npy") > 0 and os.path.isfile(f"{root}/{sh}.words.arrow")):
                return False
        for sl in hp.get("extra_slots") or []:
            if not os.path.isfile(f"{sl['root']}/NODE.json"):
                return False
    except (OSError, KeyError, ValueError):
        return False
    return True


def host_entry(plan, host):
    """-> (entry, mode): the host's own slice when it is complete on this node ('native'), else the staged overlay ('staged': slinky-0 / 1 / 2
    lost their slices on 10-18; another agent restores them from HF; whichever is on disk at launch is used, logged per rank)."""
    hp = plan["hosts"][host]
    if native_ok(hp):
        return hp, "native"
    st = (plan.get("staged") or {}).get(host)
    assert st is not None, ("pretraining slice missing / incomplete on this host and no staged overlay", host, hp["native_root"])
    return st, "staged"


def plan_sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]


def cfgs(plan):
    """(train cfg, eval cfg): the pretraining speech cfg with the fine-tune's train light_frac 0.1 + the training corruption
    (SI20K_A_CORRUPT, as the si20k user audio); eval = pretraining eval cfg, clean A."""
    tr = dict(plan["cfg"], light_frac=TRAIN_LIGHT_FRAC, a_corrupt=os.environ.get("SI20K_A_CORRUPT", "none"), b_clip=B_CLIP)
    ev = dict(plan.get("eval_cfg") or plan["cfg"], a_corrupt="none", b_clip=B_CLIP)
    ev.pop("_note", None); ev.pop("note", None)
    return tr, ev


FLAC_MANIFEST = os.environ.get("SI20K_PODCAST_FLAC_MANIFEST", "/data/home/amartyavarma/ppm/si20k-vox-ppm/plan/podcast_flac_manifest.jsonl")
_MAN = [None]


def manifest():
    """row_key -> entry of the podcast FLAC manifest (another agent stages mono FLACs of the Voxtral (A) side per host)."""
    if _MAN[0] is None:
        _MAN[0] = {}
        for ln in open(FLAC_MANIFEST):
            if ln.strip():
                e = json.loads(ln); _MAN[0][e["row_key"]] = e
    return _MAN[0]


def _ent(e, *keys, default=None):
    for k in keys:
        if k in e and e[k] is not None:
            return e[k]
    return default


def a_wave(e, n, data_side):
    """the row's A (partner) waveform: mono FLAC of the manifest, row frames [0, n) -> float32 [n x 1920] (zero-padded past the end)."""
    import soundfile as sf
    side = _ent(e, "side", "chan_A_data_side", "a_side", "a_data_side", "data_side", "channel_side")
    if side is not None:
        assert str(side) == data_side, ("manifest FLAC holds another side than this micro's A", e["row_key"], side, data_side)
    s0 = int(_ent(e, "start_sample", "sample0", "offset_samples", "flac_start_sample", default=-1))
    if s0 < 0:
        f0 = _ent(e, "frame0", "flac_frame0", "row_frame0", "frame_start", "offset_frames")
        s0 = int(f0) * 1920 if f0 is not None else int(round(float(_ent(e, "offset_s", "start_s", default=0.0)) * 24000))
    L = int(n) * 1920
    x, sr = sf.read(_ent(e, "path", "flac"), start=s0, frames=L, dtype="float32", always_2d=True)
    assert sr == 24000, ("podcast FLAC must be 24 kHz", e["row_key"], sr)
    ch = int(_ent(e, "channel", "ch", default=0)) if x.shape[1] > 1 else 0
    assert x.shape[0] > L - 6000, ("podcast FLAC shorter than the row by > 0.25 s", e["row_key"], x.shape[0], L)
    w = np.zeros(L, np.float32); w[:x.shape[0]] = x[:, ch]
    return w


def podcast_prefix_bonly(row, swap, sib, pers, cfg):
    """= data_ppm.podcast_prefix (same draws) with channel A's clip and persona dropped: A carries nothing in the prefix (A = Voxtral only)."""
    D_ = D
    sw = int(swap)
    want_voice = D_.h64(f"ppm-voice|{int(cfg['seed'])}|{row['row_key']}") % int(cfg["voice_mod"]) == 0
    segs, pv = [], {"voice_drawn": int(want_voice)}
    assert cfg.get("b_clip") == "drop", ("si20k-txt: the B voice clip must be dropped", cfg.get("b_clip"))
    pv["b_clip_dropped"] = int(want_voice)                          # si20k-txt (user 10-19): drawn, not built (neither A's nor B's clip)
    pv["a_clip_dropped"] = int(want_voice)
    pv["clips"] = [{"track": "B", "data_side": s_["side"], "src_key": s_["src_key"], "frames": [s_["w0"], s_["w0"] + s_["L"]]} for s_ in segs]
    fixed = 5 + sum(int(s_["L"]) + 5 for s_ in segs)
    blocks = []
    if pers is not None and cfg["persona_sides"] in ("both", "B"):
        D_._tok_init()
        av = D_.PT.avail(pers.get(D_.PER.SIDES[1 ^ sw]))
        keys = [k for k in D_.PT.SECTIONS if k in av]
        if keys:
            try:
                blocks.append([D_.PT.SOP] + D_.PT.encode(D_.PT.block_text(av, keys)) + [D_.PT.EOP])
            except AssertionError:
                pv["persona_skipped"] = 1
    b2, cut = D_.crop_blocks(blocks, int(cfg["sp_cap"]) - fixed)
    segs += [{"kind": "text", "track": 1, "ids": x} for x in b2 if x is not None]
    pv.update(persona=[len(x) for x in b2 if x is not None], persona_cut=int(cut), a_persona_dropped=1)
    if not segs:
        segs = [{"kind": "text", "track": 1, "ids": []}]
    spx = D_.SPR.build_prefix(row, segs, {"new_word": cfg["new_word"]})
    pv["P"] = int(spx["P"])
    return spx, pv


class Source:
    """one fine-tune rank's podcast stream: (host, local rank j) -> pretraining rank plan['hosts'][host]['ranks'][j]."""

    def __init__(self, plan, host, j, cfg):
        hp, self.mode = host_entry(plan, host)
        rk = hp["ranks"][str(int(j))]
        self.prod_rank, self.codes, self.eval_codes = int(rk["prod_rank"]), [int(c) for c in rk["unseen_codes"]], [int(c) for c in rk["eval_codes"]]
        self.root = hp["native_root"]
        self.hp = D.load_extra_plan(hp["extra_plan"])["hosts"][hp["host_plan_key"]] if hp.get("extra_plan") else None
        self.cfg = dict(cfg)
        self.ds = None

    def micro(self, code, audio=True, epoch=0):
        """the pretraining speech micro of this code (random side, B as pretraining) with channel A = Voxtral (user 10-18 PT): A's prefix clip
        and persona dropped, A emptied in the conversation (no text / Mimi / present), vin = (P, n); A waveform = the manifest's mono FLAC,
        corrupted in the collator exactly like the si20k user audio (acorrupt, train cfg only); no mic gain on podcast rows."""
        if self.ds is None:
            self.ds = D.MixedSpeechDS(self.root, self.hp, [], self.cfg, self.prod_rank)
        orig = D.podcast_prefix
        D.podcast_prefix = podcast_prefix_bonly
        try:
            m = self.ds.get(int(code))                                # swap None = pretraining's random side (side_of(row_key, seed))
        finally:
            D.podcast_prefix = orig
        T = len(m["pid"])
        assert m["segs"] == [T] and T <= BUDGET, ("podcast micro length", T)
        P_ = int(m["P"]); n = T - P_ - D.CD
        assert (m["ids"][0, :P_] == D.PAD_ID).all() and not (m["depth"][0, :P_] >= 0).any(), "A must carry nothing in the prefix"
        m["ids"] = m["ids"].copy(); m["depth"] = m["depth"].copy(); m["present"] = m["present"].copy()
        m["ids"][0, P_:] = D.PAD_ID; m["depth"][0] = -1; m["present"][0, P_:] = 0.0
        m["vin"] = (P_, n)
        rk = json.loads(m["prov"])["row_key"]
        aside = "assistant" if int(m["swap"]) else "user"             # data_ppm: swap 1 -> data 'user' on B, so A = the data assistant
        m["aref"] = {"row_key": rk, "side": aside, "n": n, "kind": "podcast_flac", "gain_db": 0.0}
        if audio:
            e = manifest()[rk]
            w = a_wave(e, n, aside)
            ci = {"mode": "none"}
            if self.cfg.get("a_corrupt", "none") == "qqa-corrupt-v2":
                import acorrupt as AC
                w, ci = AC.apply(w, rk, int(epoch), far=None)
            m["wav"] = w; m["aref"].update(flac=_ent(e, "path", "flac"), corrupt=ci.get("mode"))
        return m


def as_micro(m, code, dummy=False):
    out = Q.pack([dict(m, row_key=json.loads(m["prov"])["row_key"])], budget=BUDGET)
    out["src"] = "podcast"
    out["swap"] = int(m["swap"])
    out["item"] = json.dumps(int(code))
    out["present"] = m["present"]
    out["vin"] = np.asarray(m["vin"], np.int64)
    out["aref"] = json.dumps(m["aref"])
    if "wav" in m:
        out["wav"] = m["wav"]
    if dummy:
        out["dummy"] = 1
        out["own"] = np.zeros_like(out["own"])
    return out


class PodDS:
    """step t -> the podcast micro of unseen_codes[t] (training cfg)."""

    def __init__(self, plan_path, host, j, steps, cfg):
        self.args = (plan_path, host, int(j), dict(cfg))
        self.steps = int(steps)
        self.src = None

    def __len__(self):
        return self.steps

    def __getitem__(self, t):
        if self.src is None:
            p, h, j, c = self.args
            self.src = Source(load_plan(p), h, j, c)
            assert len(self.src.codes) >= self.steps, ("not enough unseen podcast rows for this rank", h, j, len(self.src.codes), self.steps)
        code = self.src.codes[int(t)]
        out = as_micro(self.src.micro(code), code)
        out["step"] = int(t)
        return out


def eval_micros(plan, host, j, cfg):
    src = Source(plan, host, j, cfg)
    return [as_micro(src.micro(c), c) for c in src.eval_codes]
