"""STAGE-3 PERSONA TEXT in the system prompt (S3_PERSONA_TEXT=1; default OFF; needs S3_SYSPROMPT=1). CPU only (DataLoader workers, tests).

SOURCE: the persona sidecars (persona_s3q, <shard>.persona.arrow): per data role side, <side>_status and <side>_sections_json. ONLY the
  prose strings sections['persona'] / ['starting_context'] / ['memory'] are used (never the *_items lists). A side with status != ok (or a
  section whose prose is empty) has nothing to include. Text-KL rows are unaffected (speech rows only).
SAMPLING (deterministic per row, its OWN rng stream: SeedSequence([S3_PT_SEED, blake2b(row_key) & 2^47-1, 0x5054]); the voice-clip
  sampler's stream (sysprompt.sample, S3_SP_SEED) is untouched, so every clip draw is identical with the flag on or off). Fixed draw
  count per row, in this order:
    u [2 data tracks (user, assistant), 3 sections (persona, starting_context, memory)] ~ U[0,1)  -> section k of track d is INCLUDED iff
        it exists (non-empty prose, status ok) and u[d, k] < p_k; p = persona S3_PT_P_PERSONA 0.2, starting_context S3_PT_P_CONTEXT 0.7,
        memory S3_PT_P_MEMORY 0.8: independent across sections and speakers.
    perm  = a uniform permutation of the 4 segment LABELS (text_user, text_assistant, clip_user, clip_assistant) (data-track terms).
    sperm = per data track a uniform permutation of the 3 sections (used only with S3_PT_SHUFFLE_SECTIONS=1).
  Draws are per DATA track and mapped to MODEL channels by the row's swap (model channel c holds data track c ^ swap), so a speaker's
  text follows the speaker: <sop> on channel A always carries the persona of whoever is on channel A after the swap.
TEXT BLOCK of a speaker (one wrapper per speaker; omitted entirely when no section is included):
    <sop> tok("\\n\\n".join(HEADER_k + "\\n" + prose_k for the included k)) <eop>
  HEADER = Persona / Starting context / Earlier conversation memory: exactly the source system prompt's layout (the sidecar's
  system_prompt == that join over every non-empty section, verified on 16,838 sides). Section order: persona, starting context, memory
  (S3_PT_SHUFFLE_SECTIONS=1: the row's sperm order instead). Qwen3 tokenizer (render_q.load_tok), add_special_tokens=False, every id
  asserted < 151643 (plain text, no special token). One token per position; the block runs ONLY on its own model channel: channel A's
  block through channel A's tables, B's through B's, the other channel contributes ZERO (present 0) during the block.
NEW IDS 2: <sop> 151680, <eop> 151681 (the SAME ids for both speakers: the channel identity comes from which table embeds the block).
  Free: the tokenizer emits 0..151668; the lineage uses 151670..151675 (+ skips 151669) and the voice-clip system prompt 151676..151679
  (sysprompt.NEW_IDS); grep of qwen-1ch / sam-graft finds no other use of 151680/151681. Input rows: sop <- SOB row, eop <- EOB
  (<|im_end|>) row, in emb_A, emb_B and voice.embed (model_s3q.persona_text_rows; exactly like sysprompt_rows). Never a target.
ORDER inside the prefix (header and footer fixed):
    <|im_start|> system \\n  [segments]  <|im_end|> \\n
  With at least one text block: ALL present segments (text_user, text_assistant, clip_user if drawn, clip_assistant if drawn) in the
  row's uniformly random order `perm` (restricted to the present ones: still uniform over their orders). With NO text block (flag off,
  mode 'none', or nothing drawn): the segment list is EXACTLY sysprompt.segments() (clips in model channel A-then-B order: today's
  prefixes, bit-identical; the existing eval stays comparable).
MODES (eval): 'rand' = the sampled draws (training and the eval_pt_rand variant), 'none' = no text (the existing eval variants),
  'all' = every existing section of both speakers (eval_pt_all; same perm / sperm as the row's draws)."""
import json

import numpy as np

SOP, EOP = 151680, 151681
PT_IDS = (SOP, EOP)
SECTIONS = ("persona", "starting_context", "memory")
HEADER = {"persona": "Persona", "starting_context": "Starting context", "memory": "Earlier conversation memory"}
LABELS = ("text_user", "text_assistant", "clip_user", "clip_assistant")
SIDES = ("user", "assistant")
MODES = ("rand", "none", "all")
CTRL_LO = 151643
STREAM = 0x5054                                                  # 'PT': the persona-text rng stream tag


def default_cfg(**kw):
    c = {"seed": 20261007, "p": {"persona": 0.2, "starting_context": 0.7, "memory": 0.8}, "shuffle_sections": False, "joint": False}
    c.update(kw)
    return c


def h64(s):
    import sysprompt as SPR
    return SPR.h64(s)


def avail(pside):
    """{section: prose} of one sidecar side (persona_s3q.get()[side]); {} unless status == ok."""
    if not pside or pside.get("status") != "ok" or not pside.get("sections_json"):
        return {}
    d = json.loads(pside["sections_json"])
    return {k: d[k] for k in SECTIONS if isinstance(d.get(k), str) and d[k].strip()}


def sample(row_key, pers, cfg):
    """pers = persona_s3q.PersonaSidecar.get(...) ({side: fields}) or None -> per DATA track d (0 user, 1 assistant): avail keys, drawn
    (u < p regardless of existence), perm (labels), sperm. Fixed draw count."""
    rng = np.random.default_rng(np.random.SeedSequence([int(cfg["seed"]), h64(row_key) & 0x7FFFFFFFFFFF, STREAM]))
    u = rng.random((2, len(SECTIONS)))
    perm = [LABELS[i] for i in rng.permutation(len(LABELS))]
    sperm = [[SECTIONS[i] for i in rng.permutation(len(SECTIONS))] for _ in range(2)]
    av = [avail((pers or {}).get(s)) for s in SIDES]
    p = cfg["p"]
    drawn = [[k for j, k in enumerate(SECTIONS) if u[d, j] < float(p[k])] for d in range(2)]
    return {"avail": av, "drawn": drawn, "perm": perm, "sperm": sperm, "u": u.tolist()}


def included(pts, d, mode, cfg):
    """sections of data track d included under `mode`, in block order."""
    av = pts["avail"][d]
    if mode == "none":
        return []
    keys = [k for k in SECTIONS if k in av and (mode == "all" or k in pts["drawn"][d])]
    if cfg.get("shuffle_sections"):
        keys = [k for k in pts["sperm"][d] if k in keys]
    return keys


def block_text(av, keys):
    return "\n\n".join(HEADER[k] + "\n" + av[k] for k in keys)


_TOK = [None]


def tok():
    if _TOK[0] is None:
        import os
        import sys
        q1 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))     # qwen-1ch (render_q.py: the Qwen3 tokenizer, as the text paths)
        if q1 not in sys.path:
            sys.path.append(q1)
        import render_q as RM
        _TOK[0] = RM.load_tok()
    return _TOK[0]


def encode(text):
    ids = tok()(text, add_special_tokens=False)["input_ids"]
    assert ids and max(ids) < CTRL_LO, ("persona text produced a special / control id", max(ids) if ids else None)
    return [int(t) for t in ids]


def segments(clip_segs, pts, swap, mode, cfg, state=None):
    """clip_segs = sysprompt.segments(state, smp, swap) -> (the full segment list, info). No text block -> clip_segs unchanged.
    cfg['joint'] (S3_PT_JOINT, 10-09) with state given (the row's MODEL-channel prompt state NONE / A / B / BOTH, training rows only):
    model channel c's text block is kept only if c is in the state, so the S3_SP_P draw decides text and clip jointly (NONE = no prompt)."""
    assert mode in MODES, mode
    sw = int(swap)
    allowed = (0, 1)
    if cfg.get("joint") and state is not None:
        allowed = {"NONE": (), "A": (0,), "B": (1,), "BOTH": (0, 1)}[state]
    txt, info = {}, {"A": None, "B": None}
    for d in range(2):
        keys = included(pts, d, mode, cfg)
        if not keys:
            continue
        c = d ^ sw                                                # model channel of data track d
        if c not in allowed:
            continue
        ids = [SOP] + encode(block_text(pts["avail"][d], keys)) + [EOP]
        txt[d] = {"kind": "text", "track": c, "ids": ids, "side": SIDES[d], "sections": keys}
        info["AB"[c]] = {"role": SIDES[d], "sections": keys, "n_tokens": len(ids)}
    if not txt:
        return list(clip_segs), dict(info, order=[f"clip_{'AB'[s['track']]}" for s in clip_segs])
    clip_by_d = {}
    for s in clip_segs:
        assert s["kind"] == "speech"
        clip_by_d[int(s["track"]) ^ sw] = s                       # data track of model channel s['track']
    segs, order = [], []
    for lab in pts["perm"]:
        kind, side = lab.split("_")
        d = SIDES.index(side)
        s = txt.get(d) if kind == "text" else clip_by_d.get(d)
        if s is not None:
            segs.append(s); order.append(f"{kind}_{'AB'[d ^ sw]}")
    assert len(segs) == len(txt) + len(clip_segs)
    return segs, dict(info, order=order)


def prov(info):
    """compact provenance for prov / eval_rows.jsonl."""
    out = {"pt_order": info["order"]}
    for c in "AB":
        x = info[c]
        out[f"pt_{c}_sections"] = x["sections"] if x else []
        out[f"pt_{c}_tokens"] = x["n_tokens"] if x else 0
    return out


def decode_prefix(ids, present, P, tk=None):
    """human-readable prefix of one row: ids / present [2, >=P] (numpy) -> text. Runs of positions owned by one channel are shown as
    [ch A] / [ch B] spans; text tokens detokenised, <sop>/<eop>/<sos*>/<eos*> named, clip lanes shown as words + PAD/NW counts."""
    import sysprompt as SPR
    tk = tk or tok()
    names = {SOP: "<sop>", EOP: "<eop>", SPR.SOSA: "<sosa>", SPR.EOSA: "<eosa>", SPR.SOSB: "<sosb>", SPR.EOSB: "<eosb>", SPR.IM_START: "<|im_start|>", SPR.IM_END: "<|im_end|>"}
    own = [int(np.argmax(present[:, p])) for p in range(P)]
    out, p = [], 0
    while p < P:
        c, q = own[p], p
        while q < P and own[q] == c:
            q += 1
        span = [int(t) for t in ids[c, p:q]]
        parts, buf, npad = [], [], 0
        for t in span + [-1]:
            if 0 <= t < CTRL_LO:
                buf.append(t); continue
            if buf:
                parts.append(tk.decode(buf)); buf = []
            if t in names:
                parts.append(names[t])
            elif t >= 0:
                npad += 1
        s = "".join(parts)
        out.append(f"[ch {'AB'[c]} pos {p}..{q - 1}{f', {npad} PAD/NW lane slots' if npad else ''}] {s}")
        p = q
    return "\n".join(out)
