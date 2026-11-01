"""compare two identity.py outputs: every digest of every logical rank's step-0 train element and first eval micro.
keys only in the second file must be the persona bookkeeping keys and must be 0 (p = 0 / eval)."""
import json
import sys

a, b = (json.load(open(x)) for x in sys.argv[1:3])
mode = sys.argv[3] if len(sys.argv) > 3 else "identity"          # identity | eval_only
EXTRA = {"persona_drawn", "persona_block", "persona_tokens"}
n_arr, bad, extra_bad = 0, [], []
assert a["plan_digest"] == b["plan_digest"] == "d1a3e3721328da0a", (a["plan_digest"], b["plan_digest"])
assert (a["n_items"], a["n_eval"], a["steps"]) == (b["n_items"], b["n_eval"], b["steps"]) == (15898, 316, 249)


def cmp(x, y, path, extra_ok):
    global n_arr
    for k in sorted(set(x) | set(y)):
        if k not in x:
            if k in EXTRA and (k in ("persona_drawn",) if extra_ok else False) or (k in EXTRA and y[k] == "0"):
                continue
            extra_bad.append((path, k, y[k])); continue
        if k not in y:
            bad.append((path, k, "missing")); continue
        if isinstance(x[k], list) and x[k] and isinstance(x[k][0], dict):
            for i, (u, v) in enumerate(zip(x[k], y[k])):
                cmp(u, v, f"{path}.{k}[{i}]", extra_ok)
            continue
        n_arr += 1
        if x[k] != y[k]:
            bad.append((path, k))


for L in map(str, range(64)):
    ra, rb = a["ranks"][L], b["ranks"][L]
    assert ra["item0"] == rb["item0"] and ra["eval_item0"] == rb["eval_item0"]
    if mode == "identity" or not rb["persona"][0]:
        cmp(ra["train"], rb["train"], f"L{L}.train", mode != "identity")
    cmp(ra["eval"], rb["eval"], f"L{L}.eval", mode != "identity")
print(f"{mode}: {n_arr} digests compared over 64 logical ranks; mismatches {len(bad)}; extra-key problems {len(extra_bad)}")
print("bad", bad[:10], "extra", extra_bad[:10])
print("PASS" if not bad and not extra_bad else "FAIL")
