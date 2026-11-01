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


def build(acct, lworld, epochs=1, steps=None, eval_frac=0.04, root=STORES_SRC):
    steps_arg = steps
    rows = speaker_rows(acct, root)
    assert rows, ("no rows for account", acct)
    convs = sorted({x["conv"] for x in rows})
    ev_convs = sorted({x["conv"] for x in rows if x["split"] == "eval"})
    assert all(x["split"] == "eval" for x in rows if x["conv"] in set(ev_convs)), "a conversation split across store splits"
    n_ho = max(1, int(round(eval_frac * len(convs))))
    rest = sorted((c for c in convs if c not in set(ev_convs)), key=lambda c: (h64(f"aloe-spk-eval|{acct}|{c}"), c))
    ho = sorted(set(ev_convs) | set(rest[:max(0, n_ho - len(ev_convs))]))
    tr_rows = [x for x in rows if x["conv"] not in set(ho)]
    ev_rows = [x for x in rows if x["conv"] in set(ho)]
    train = [(x["k"], x["r"], x["swap"]) for x in tr_rows]
    n = len(train)
    if steps is None:
        n_ep = int(epochs)
        assert n_ep >= 1
        steps = -(-n_ep * n // lworld)
        n_items = n_ep * n
    else:
        steps = int(steps)
        n_items = steps * lworld
        n_ep = -(-n_items // n)
    items = []
    for e in range(n_ep):
        perm = np.random.default_rng(PLAN_SEED if e == 0 else [PLAN_SEED, e]).permutation(n)
        items += [train[i] for i in perm]
    items = items[:n_items]
    pool = sorted((x["k"], x["r"], x["side"], x["conv"]) for x in tr_rows)
    plan = {"format": "aloe-spk-plan-v1", "account": acct, "lworld": int(lworld), "epochs_arg": epochs, "steps_arg": steps_arg,
            "items": items, "steps": int(steps), "epochs": n_items / n, "eval": std_eval(root), "eval_spk": [(x["k"], x["r"], x["swap"]) for x in ev_rows],
            "n_train_rows": n, "n_eval_spk_rows": len(ev_rows), "convs": len(convs), "convs_heldout": ho, "convs_heldout_from_std_eval": ev_convs,
            "eval_frac": eval_frac, "clip_pool": pool, "stores": [f"{root}/m1n{k}" for k in range(4)], "plan_seed": PLAN_SEED}
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
