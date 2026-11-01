"""aloe-zoom-cotrain ZOOM DATA (10-21): InternalCan/zoom-strangers-5k as PPM CHAT TEXT ROWS (the PPM 32B pretraining's text half).

SOURCE  $ALOE_ROOT/zoom-charisma/full-5k/{train,eval}.jsonl = the files of the PRIVATE HF dataset repo InternalCan/zoom-strangers-5k
        (pushed from this directory by zoom-charisma/code/push_hf.py, job 24552: train.jsonl 60,115,866 B, eval.jsonl 1,228,694 B; the
        sha256 below is asserted at load, so nothing else can be read). 4,900 train / 100 eval conversations (eval = 2 % stratified by
        setting). TEXT ONLY: {"id", "messages": [{"role": user|assistant, "content"}], "meta"}; no audio, no word timings, no system
        prompt; the user always opens, every row ends on the assistant. Synthetic (planner z-ai/glm-5.3-flash MIT, speakers
        xiaomi/mimo-v2.6-flash MIT + z-ai/glm-5.3-flash MIT, judge openai/gpt-oss-120b Apache-2.0): commercially usable.
MAPPING = PPM's corpus15m chat rows EXACTLY (chat_ppm.py, the same code PPM pretraining ran on half of its text micros):
        render  qwen-1ch render_q.build_row_q(messages, render_q.cfg_from({}), tok, data_q.make_rng(seed)) = the stage-1/2 two-lane renderer
                (E28 build_row4 frame layout, timing4: within-turn gap capped at 6, turn gap U{8..12}, guard frame 7, NO NEW_WORD; the plain
                Qwen3 chat template asserted per row) -> sys_ids [P] (P = 0 here: no system message), a_ids / b_ids [F] (user / assistant
                lanes on one frame timeline).
        micro   chat_ppm.chat_micro(side 'assistant', keep_eob True) = PPM_CHAT_SIDE=assistant (D19: the assistant is ALWAYS model channel
                B; zoom's 'assistant' = the friendly person) + PPM_CHAT_EOB=1 (D24: the assistant's <|im_end|> kept as B's EOB target);
                channel A = the user lane (input only); every conversation position predicts B's next token (word / PAD / EOB); B text
                CE only (PPM's factorised weights in the trainer); no audio, no NEW_WORD (text rows never carry it).
        seed    the render rng of conversation j = data_q.ROW_SEED + SEED_OFF[split] + j (j = line index in the split's jsonl): corpus
                rows use ROW_SEED + corpus row; the offsets keep zoom's draws disjoint from every corpus split's.
PACKS   = chat_ppm.pack_plan's rule: a seeded permutation of the split's conversations packed greedily into rows of <= CAP = 7,500 frames
        (length P + 1 + F per conversation from the scan table, cropped to CAP), then data_ppm.packed = ONE plain-causal segment over
        the whole pack (pack_attn causal, PPM user decision 10-11: positions 0..N-1, conversations attend across their boundaries).
EPOCHS  epoch e packs its own permutation (seed PLAN_SEED + 1000 e); the training stream = epoch 0's packs, then epoch 1's, ...;
        zoom logical rank z reads stream pack t x ZL + z at step t (ZL = zoom logical world, 64 in production).
EVAL    the 100 eval conversations in file order, packed by the same rule WITHOUT shuffling -> eval packs round-robin over the zoom
        logical ranks.
TABLE   plan/zoom_table.npz (tools/scan_zoom.py, run on a compute node): per conversation [ok, reason, F, P, n_b_word, n_b_eob, n_b_pad,
        n_a_word] (columns 2 / 3 = chat_ppm's table layout, so chat_ppm.pack_plan is used unchanged) + the render cfg key.
TRAINER FORM (as_noop_speech): the chat micro re-expressed as a ZERO-WEIGHT speech micro so every FSDP unit runs on zoom ranks in the
        Aloe ranks' order (train_cotrain.py doc): depth = -1 everywhere ([2, N, 32]; code_feedback of -1 codes is exactly 0, so the
        trunk input is bit-identical to the depth-None chat micro), cb0 = -100 everywhere (no tower labels), light = [] (no decoder
        frames), zoom = 1. CPU only (DataLoader workers)."""
_ALOE_ROOT = __import__("os").environ["ALOE_ROOT"]        # release: every path comes from aloe.env (ALOE_ROOT)
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.join(HERE, "lib"), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
import chat_ppm as CH  # noqa: E402

SRC = os.environ.get("ZOOM_SRC", f"{_ALOE_ROOT}/zoom-charisma/full-5k")
HF_REPO = "InternalCan/zoom-strangers-5k"
SHA256 = {"train": "6c40f7707dd629fd17b3", "eval": "afc014d9315e68f94ea3"}       # sha256 prefixes of the pushed files (asserted)
BYTES = {"train": 60115866, "eval": 1228694}                                      # = the HF push log's sizes
N_CONV = {"train": 4900, "eval": 100}
SEED_OFF = {"train": 7_000_000_000, "eval": 7_100_000_000}                      # render rng = ROW_SEED + SEED_OFF[split] + j
CAP = 7500
PLAN_SEED = 20261021
TABLE = os.environ.get("ZOOM_TABLE", f"{_ALOE_ROOT}/ppm/aloe-zoom-cotrain/plan/zoom_table.npz")
CTRL_LO, PAD, EOB = CH.CTRL_LO, CH.PAD, CH.EOB
DEPTH = 32


def file_sha(split):
    h = hashlib.sha256()
    with open(f"{SRC}/{split}.jsonl", "rb") as f:
        for blk in iter(lambda: f.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


def load_split(split, check=True):
    """-> list of conversations (dicts: id, messages, meta) in file order; the file is asserted = the pushed HF file."""
    path = f"{SRC}/{split}.jsonl"
    if check:
        assert os.path.getsize(path) == BYTES[split], ("zoom file size != the HF push", path, os.path.getsize(path))
        assert file_sha(split).startswith(SHA256[split]), ("zoom file sha256 != the HF push", path)
    rows = [json.loads(ln) for ln in open(path)]
    assert len(rows) == N_CONV[split], (split, len(rows))
    for r in rows:
        roles = [m["role"] for m in r["messages"]]
        assert roles[0] == "user" and roles[-1] == "assistant" and all(x in ("user", "assistant") for x in roles), (r["id"], roles[:3])
    return rows


def render_seed(split, j):
    import data_q as Dq
    return Dq.ROW_SEED + SEED_OFF[split] + int(j)


class Renderer:
    """lazy (per DataLoader worker): qwen-1ch's stage-1/2 renderer + tokenizer (= chat_ppm.Renderer's, messages from the zoom jsonl)."""

    def __init__(self, split):
        for p in (CH.Q1, f"{CH.Q1}/stage2"):
            if p not in sys.path:
                sys.path.append(p)
        import data_q as Dq
        import render_q as RM
        self.Dq, self.RM, self.split = Dq, RM, split
        self.rows = load_split(split)
        self.tok, self.cfg = RM.load_tok(), RM.cfg_from({})

    def __call__(self, j):
        return self.RM.build_row_q(self.rows[int(j)]["messages"], self.cfg, self.tok, self.Dq.make_rng(render_seed(self.split, j)))


def cfg_key():
    """render identity: qwen-1ch's data_q.cfg_key (render cfg, template, PAD, tokenizer rev) + zoom's source / seeds / cap."""
    for p in (CH.Q1, f"{CH.Q1}/stage2"):
        if p not in sys.path:
            sys.path.append(p)
    import data_q as Dq
    k, d = Dq.cfg_key()
    z = {"q1": k, "src_sha": SHA256, "seed_off": SEED_OFF, "row_seed": Dq.ROW_SEED, "cap": CAP, "chat_side": "assistant", "chat_eob": True}
    return hashlib.sha256(json.dumps(z, sort_keys=True).encode()).hexdigest()[:16], z


def load_table(path=TABLE):
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    k, _ = cfg_key()
    assert meta["cfg_key"] == k, ("zoom table built for another render cfg", meta["cfg_key"], k)
    tb = {s: z[s] for s in ("train", "eval")}
    for s in tb:
        assert tb[s].shape == (N_CONV[s], 8) and (tb[s][:, 0] == 1).all(), ("unrenderable zoom conversations", s, int((tb[s][:, 0] != 1).sum()))
    return tb, meta


def epoch_packs(tb_train, e, cap=CAP):
    """epoch e's packs (lists of train conversation indices) = chat_ppm.pack_plan(all 4,900, seed PLAN_SEED + 1000 e)."""
    return CH.pack_plan(np.arange(len(tb_train)), tb_train, cap, PLAN_SEED + 1000 * int(e), shuffle=True)


def eval_packs(tb_eval, cap=CAP):
    return CH.pack_plan(np.arange(len(tb_eval)), tb_eval, cap, PLAN_SEED, shuffle=False)


class Stream:
    """the training pack stream: pack index s -> (epoch, packs[k]) over consecutive epochs (each epoch's plan built once, cached)."""

    def __init__(self, tb_train, cap=CAP):
        self.tb, self.cap, self.ep = tb_train, cap, {}

    def epoch(self, e):
        if e not in self.ep:
            self.ep[e] = epoch_packs(self.tb, e, self.cap)
        return self.ep[e]

    def get(self, s):
        e, s = 0, int(s)
        while s >= len(self.epoch(e)):
            s -= len(self.epoch(e)); e += 1
        return e, s, self.epoch(e)[s]


def chat_pack_micro(rend, conv_ids, cap=CAP):
    """rendered conversations of one pack -> PPM's chat micro (chat_micro + data_ppm.packed's causal flattening)."""
    rows = [rend(j) for j in conv_ids]
    m = CH.chat_micro(rows, [f"zoom-{rend.split}:{rend.rows[j]['id']}" for j in conv_ids], cap, "assistant", 0, True)
    N = int(sum(m["segs"]))
    m["segs"], m["pid"] = [N], np.arange(N, dtype=np.int64)            # = data_ppm.flatten_segments (pack_attn causal)
    return m


def as_noop_speech(m, dummy=False):
    """chat micro -> the trainer's zoom micro (module doc): a ZERO-WEIGHT speech-shaped micro whose text part is the chat micro unchanged.
    own_zoom = B text targets (the zoom group's tokens_step normaliser); own (Aloe counts) = 0. dummy: zero counts (trainer zeroes weights)."""
    N = int(m["ids"].shape[1])
    assert m["kind"] == "text" and m["src"] == "chat" and m["ids"].shape == (2, N) and m["segs"] == [N]
    assert ((m["ids"] < CTRL_LO) | (m["ids"] == PAD) | (m["ids"] == EOB)).all() and not m["nw"].any()
    out = {"kind": "speech", "zoom": 1, "ids": m["ids"], "present": m["present"].astype(np.float32), "pid": m["pid"], "segs": [N],
           "depth": np.full((2, N, DEPTH), -1, np.int64), "sel": m["sel"], "y": m["y"], "word": m["word"], "pad": m["pad"], "eob": m["eob"],
           "nw": m["nw"], "cb0": np.full(N, -100, np.int64), "light": np.zeros(0, np.int64), "n_text": float(len(m["sel"])),
           "own": np.zeros((1, 3), np.float64), "own_zoom": float(len(m["sel"])), "P": np.zeros(1, np.int64), "prov": m["prov"],
           "swap": 0, "voice_clip": 0, "voice_drawn": 0}
    if dummy:
        out["dummy"] = 1
        out["own_zoom"] = 0.0
    return out


class ZoomAccDS:
    """one PHYSICAL zoom rank's stream (= data_aloe_v2vox.AccDS's shape): element i = (step t, slot g) -> the pack zoom logical rank
    zl[g] reads at step t (stream pack t x ZL + zl[g]); zl[g] None = a pad slot (dummy micro). start = the first step (resume)."""

    def __init__(self, tb_train, zlranks, ZL, steps, start=0, cap=CAP, max_packs=None):
        assert zlranks[0] is not None
        self.tb, self.zl, self.ZL, self.T, self.start, self.cap = tb_train, list(zlranks), int(ZL), int(steps), int(start), cap
        self.max_packs = None if max_packs is None else int(max_packs)     # zoom_epoch mode: stream packs >= max_packs are dummies
        self.G = len(self.zl)
        self.st, self.rend = None, None

    def __len__(self):
        return (self.T - self.start) * self.G

    def __getitem__(self, i):
        if self.st is None:
            self.st, self.rend = Stream(self.tb, self.cap), Renderer("train")
        t, g = divmod(int(i), self.G)
        t += self.start
        z = self.zl[g]
        pad = z is None
        s = t * self.ZL + (self.zl[0] if pad else int(z))
        over = self.max_packs is not None and s >= self.max_packs
        e, k, pk = self.st.get(s if not over else 0)
        out = as_noop_speech(chat_pack_micro(self.rend, pk, self.cap), dummy=pad or over)
        out.update(step=t, slot=g, lrank=(-1 if pad else int(z)), padslot=int(pad), zoom_epoch=e, zoom_pack=k, zoom_convs=json.dumps([int(x) for x in pk]))
        return out


def eval_micros(tb_eval, zrank, ZL, cap=CAP):
    """zoom logical rank zrank's eval micros (eval packs zrank, zrank + ZL, ...)."""
    rend = Renderer("eval")
    return [dict(as_noop_speech(chat_pack_micro(rend, pk, cap)), zoom_convs=json.dumps([int(x) for x in pk]))
            for pk in eval_packs(tb_eval, cap)[int(zrank)::int(ZL)]]
