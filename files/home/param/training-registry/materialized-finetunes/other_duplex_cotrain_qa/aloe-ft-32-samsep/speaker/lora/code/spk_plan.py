"""SINGLE-SPEAKER Aloe FT plan (10-24, user via main: "pick a speaker to finetune our Aloe FT model on one speaker, all acoustic codes"; pick
Ashley = account TyypZEov3LSf5Gp8N6kSnBTZYXy2). Pure python + numpy (no torch): used by tools/build_spk.py (plan file, A-FLAC ledgers,
RAWGAIN, report) AND by data_aloe_spk.make_plan (the trainer rebuilds the plan in process and asserts its digest == the env's).
ROWS      every row (EVERY chunk, both store splits) of the sam-sep s2h-m1 stores (= 29345's row universe: handover none-crossbleed calls,
          sam-sep Mimi codes, all 32 codebooks) of a conversation with the speaker on a lane. Speaker id = manifest speaker_<NN>_account_uid,
          NN = the row's <side>_speaker_dir lane, joined on rows.jsonl src_conversation == manifest recording_key.
SIDE      the speaker is ALWAYS model channel B: item (k, r, swap), swap 0 when she is the row's data 'assistant' side, swap 1 (= FLIP) when she
          is the data 'user' side (data_aloe_v2 / v2vox convention: swap 0 -> B = data assistant, A = data user; swap 1 -> B = data user, A =
          data assistant, whose raw FLAC feeds Voxtral). ONE item per row (29345 trains both orientations of every row).
HOLD-OUT  by CONVERSATION, deterministic: her conversations in the standard Aloe EVAL split (held out from 29345 too: never seen by the init;
          kept out of training so the standard eval stays held out) + her train-split conversations ranked by h64('aloe-spk-eval|<acct>|<conv>')
          until n = max(1, round(EVAL_FRAC x her conversations)) (EVAL_FRAC 0.04); every chunk of a held-out call -> the speaker eval.
ORDER     epoch e: the train items (sorted by (k, r)) permuted with np.random.default_rng(PLAN_SEED) for e = 0 (= the v2 builder's call) and
          np.random.default_rng([PLAN_SEED, e]) for e >= 1, epochs concatenated. steps = ceil(EPOCHS x n / LWORLD) when STEPS is not given
          (the last step's free slots = zero-weight dummies, as in 29345); STEPS given -> the concatenation (as many epochs as needed)
          truncated to STEPS x LWORLD. Item t x LWORLD + R -> logical rank R at step t (= data_aloe_v2.rank_items).
EVAL      'eval' = the STANDARD Aloe eval list, exactly 29345's (v2 builder over the sam-sep stores: the 158 first-chunk eval rows x both
          orientations = 316 items; eval digest dab655b138da40b1 asserted); 'eval_spk' = every row of her held-out calls, her on B (sorted).
CLIPS     voice-clip source pool = her rows (every chunk) of her TRAIN conversations, (k, r, her data side), sorted; the B voice clip of an item
          is drawn from this pool restricted to OTHER conversations (data_aloe_spk.prefix_spk)."""
import csv
import hashlib
import json
import math

import numpy as np

MANIFEST = "/data/home/amartyavarma/data/aloe-handover-full/DATA/manifest.csv"
STORES_SRC = "/data/home/amartyavarma/qwen-1ch-data/s2h-m1-samsep/stores"
SAMSEP_DIGESTS = ["7318ffb3ea75cbcc3ff48b63b6cf8b88", "31ecb2273542310096936c0baf49efa8", "1759200f7818c6017dfb6a46e37fa5df",
                  "611b2e7edb1503cc600a6489e49dcf40"]
STD_EVAL_DIGEST = "dab655b138da40b1"           # 29345's eval list (data/plan/PLAN_samsep.json eval_digest)
PLAN_SEED = 20261011
FPS = 12.5
_CACHE = {}


def h64(s):
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "little")


def digest(x):
    return hashlib.sha256(json.dumps(x).encode()).hexdigest()[:16]


def manifest():
    if "man" not in _CACHE:
        csv.field_size_limit(1 << 30)
        _CACHE["man"] = {m["recording_key"]: {"acc": {"00": m["speaker_00_account_uid"], "01": m["speaker_01_account_uid"]},
                                             "actor": {"00": m["speaker_00_actor_id"], "01": m["speaker_01_actor_id"]},
                                             "bucket": m["relative_dir"].split("/")[1], "dur": float(m["clip_duration_sec"])}
                         for m in csv.DictReader(open(MANIFEST))}
    return _CACHE["man"]


def store_rows(root=STORES_SRC):
    """-> [k] = list of rows.jsonl dicts (row == line index asserted)."""
    if ("rows", root) not in _CACHE:
        out = []
        for k in range(4):
            rs = [json.loads(ln) for ln in open(f"{root}/m1n{k}/rows.jsonl")]
            assert [int(r["row"]) for r in rs] == list(range(len(rs))), (root, k)
            out.append(rs)
        _CACHE[("rows", root)] = out
    return _CACHE[("rows", root)]


def side_account(r, side):
    return manifest()[r["src_conversation"]]["acc"][r[f"{side}_speaker_dir"][-2:]]


def speaker_rows(acct, root=STORES_SRC):
    """every store row with `acct` on a lane -> [dict(k, r, side, swap, conv, split, chunk, n_frames, row_key)] sorted by (k, r)."""
    out = []
    for k, rs in enumerate(store_rows(root)):
        for r in rs:
            sides = [s for s in ("assistant", "user") if side_account(r, s) == acct]
            if not sides:
                continue
            assert len(sides) == 1, ("speaker on both lanes of a row", r["row_key"])
            assert manifest()[r["src_conversation"]]["bucket"] == "none", ("row outside the none-crossbleed bucket", r["row_key"])
            s = sides[0]
            out.append({"k": k, "r": int(r["row"]), "side": s, "swap": 0 if s == "assistant" else 1, "conv": r["src_conversation"],
                        "split": r["split"], "chunk": int(r["chunk_index"]), "n_frames": int(r["n_frames"]), "row_key": r["row_key"],
                        "partner": side_account(r, "user" if s == "assistant" else "assistant")})
    return out


def std_eval(root=STORES_SRC):
    """= data_aloe_v2.make_plan(...)['eval'] over the sam-sep stores: first-chunk eval rows in (k, row) order x swap 0 / 1."""
    ev = [(k, int(r["row"]), s) for k, rs in enumerate(store_rows(root)) for r in rs if r["split"] == "eval" and int(r["chunk_index"]) == 0
          for s in (0, 1)]
    assert digest(ev) == STD_EVAL_DIGEST, ("standard eval list != 29345's", digest(ev))
    return ev


def schedule_defaults(steps):
    """the 29345-style PLACEHOLDERS (user's call): warmup round(10 x steps / 306) (26232's rule), linear decay to 0 at the last step,
    exports at 29345's fractions of its 249 steps (41, 83, 124, 166, 208)."""
    ex = sorted(e for e in {int(round(x * steps / 249)) for x in (41, 83, 124, 166, 208)} if 0 < e < steps)
    return {"warmup": int(round(10 * steps / 306)), "export_steps": ex}


def general_items(root=STORES_SRC):
    """= data_aloe_v2.make_plan(64, roots)['items'] over the sam-sep stores (29345's plan d1a3e3721328da0a): first-chunk train rows in (k, row)
    order x swap 0 / 1, permuted with default_rng(PLAN_SEED)."""
    base = [(k, int(r["row"]), s) for k, rs in enumerate(store_rows(root)) for r in rs if r["split"] == "train" and int(r["chunk_index"]) == 0
            for s in (0, 1)]
    perm = np.random.default_rng(PLAN_SEED).permutation(len(base))
    out = [base[i] for i in perm]
    assert digest(out) == GENERAL_DIGEST, ("general plan != 29345's", digest(out))
    return out


GENERAL_DIGEST = "d1a3e3721328da0a"


def build(acct, lworld, epochs=1, steps=None, eval_frac=0.04, root=STORES_SRC, eval_calls=0, replay_frac=0.0):
    """module doc; LoRA-run additions (both default off = the plan above exactly):
    eval_calls > 0   hold out EXACTLY this many of her calls (standard-eval-split calls first, then the h64 rank), not EVAL_FRAC;
    replay_frac > 0  each step = n_her = LW - round(LW x replay_frac) of HER rows (her epochs, as above; None = dummy when her stream ends on
                     the last step) on logical ranks 0..n_her-1 + n_rep = round(LW x replay_frac) GENERAL rows on the rest: 29345's plan order
                     (general_items) minus every row of her calls (both orientations), consumed in order; items are then (k, r, swap, src)
                     with src 0 = her (speaker path), 1 = general (29345's data path exactly)."""
    steps_arg = steps
    rows = speaker_rows(acct, root)
    assert rows, ("no rows for account", acct)
    convs = sorted({x["conv"] for x in rows})
    ev_convs = sorted({x["conv"] for x in rows if x["split"] == "eval"})
    assert all(x["split"] == "eval" for x in rows if x["conv"] in set(ev_convs)), "a conversation split across store splits"
    key = lambda c: (h64(f"aloe-spk-eval|{acct}|{c}"), c)  # noqa: E731
    if eval_calls:
        assert 0 < eval_calls < len(convs), (eval_calls, len(convs))
        cand = sorted(ev_convs, key=key) + sorted((c for c in convs if c not in set(ev_convs)), key=key)
        ho = sorted(cand[:eval_calls])
    else:
        n_ho = max(1, int(round(eval_frac * len(convs))))
        rest = sorted((c for c in convs if c not in set(ev_convs)), key=key)
        ho = sorted(set(ev_convs) | set(rest[:max(0, n_ho - len(ev_convs))]))
    tr_rows = [x for x in rows if x["conv"] not in set(ho)]
    ev_rows = [x for x in rows if x["conv"] in set(ho)]
    train = [(x["k"], x["r"], x["swap"]) for x in tr_rows]
    n = len(train)
    n_rep = int(round(lworld * float(replay_frac)))
    n_her = lworld - n_rep
    assert n_her >= 1 and 0 <= n_rep < lworld, (lworld, replay_frac)
    if steps is None:
        n_ep = int(epochs)
        assert n_ep >= 1
        steps = -(-n_ep * n // n_her)
        n_items = n_ep * n
    else:
        steps = int(steps)
        n_items = steps * n_her
        n_ep = -(-n_items // n)
    her = []
    for e in range(n_ep):
        perm = np.random.default_rng(PLAN_SEED if e == 0 else [PLAN_SEED, e]).permutation(n)
        her += [train[i] for i in perm]
    her = her[:n_items]
    rep_info = None
    if n_rep == 0:
        items = her
    else:
        mine = set(convs)
        gen = [it for it in general_items(root) if store_rows(root)[it[0]][it[1]]["src_conversation"] not in mine]
        need = steps * n_rep
        assert need <= len(gen), ("replay longer than the general plan", need, len(gen))
        items = []
        for t in range(steps):
            hs = her[t * n_her:(t + 1) * n_her]
            items += [(k, r, s, 0) for k, r, s in hs] + [None] * (n_her - len(hs))
            items += [(k, r, s, 1) for k, r, s in gen[t * n_rep:(t + 1) * n_rep]]
        rep_info = {"frac": float(replay_frac), "rows_per_step": n_rep, "her_rows_per_step": n_her, "general_digest": GENERAL_DIGEST,
                    "general_minus_her_calls": len(gen), "replay_items_used": need, "excluded_her_calls": len(mine),
                    "rule": "29345's plan order minus her calls, consumed in order; her rows on logical ranks 0..n_her-1"}
    pool = sorted((x["k"], x["r"], x["side"], x["conv"]) for x in tr_rows)
    plan = {"format": "aloe-spk-plan-v1" if n_rep == 0 else "aloe-spk-plan-v2-replay", "account": acct, "lworld": int(lworld), "epochs_arg": epochs,
            "steps_arg": steps_arg, "items": items, "steps": int(steps), "epochs": n_items / n, "eval": std_eval(root),
            "eval_spk": [(x["k"], x["r"], x["swap"]) for x in ev_rows], "n_train_rows": n, "n_eval_spk_rows": len(ev_rows), "convs": len(convs),
            "convs_heldout": ho, "convs_heldout_from_std_eval": ev_convs, "eval_frac": eval_frac, "clip_pool": pool,
            "stores": [f"{root}/m1n{k}" for k in range(4)], "plan_seed": PLAN_SEED}
    if eval_calls:
        plan["eval_calls"] = int(eval_calls)
    if rep_info:
        plan["replay"] = rep_info
    plan["n_train_micros"] = sum(x is not None for x in items)
    plan["plan_digest"] = digest(plan["items"])
    plan["eval_digest"] = digest(plan["eval"])
    plan["eval_spk_digest"] = digest(plan["eval_spk"])
    plan["pool_digest"] = digest(plan["clip_pool"])
    plan["stats"] = stats(rows, tr_rows, ev_rows)
    return plan


def stats(rows, tr_rows, ev_rows):
    def one(xs):
        return {"rows": len(xs), "convs": len({x["conv"] for x in xs}), "flipped_rows": sum(x["swap"] for x in xs),
                "first_chunk_rows": sum(x["chunk"] == 0 for x in xs), "later_chunk_rows": sum(x["chunk"] > 0 for x in xs),
                "frames": sum(x["n_frames"] for x in xs), "hours": round(sum(x["n_frames"] for x in xs) / FPS / 3600, 3),
                "partners": len({x["partner"] for x in xs}), "std_eval_split_rows": sum(x["split"] == "eval" for x in xs)}
    return {"all": one(rows), "train": one(tr_rows), "eval_spk": one(ev_rows)}
