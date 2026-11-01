"""compare two identity.py outputs (29345 ref vs posedup): every digest of every logical rank's step-0 train element and first eval micro.
mode identity (POSEDUP_POSE=0): every key identical, no extra key.
mode pose (POSEDUP_POSE=1): every 29345 key identical except 'kind' (speech -> pose) and 'own' (compared as its first 3 columns, 'own3');
  the only extra keys are the pose fields (pdepth, pcb0, plight, pose_ok, pose_src, own3)."""
import json
import sys

a, b = (json.load(open(x)) for x in sys.argv[1:3])
mode = sys.argv[3] if len(sys.argv) > 3 else "identity"
POSE_EXTRA = {"pdepth", "pcb0", "plight", "pose_ok", "pose_src", "own3"}
n_arr, bad, extra_bad = 0, [], []
assert a["plan_digest"] == b["plan_digest"] == "d1a3e3721328da0a", (a["plan_digest"], b["plan_digest"])
assert (a["n_items"], a["n_eval"], a["steps"]) == (b["n_items"], b["n_eval"], b["steps"]) == (15898, 316, 249), \
    ((a["n_items"], a["n_eval"], a["steps"]), (b["n_items"], b["n_eval"], b["steps"]))


def cmp(x, y, path):
    global n_arr
    for k in sorted(set(x) | set(y)):
        if k not in x:
            if mode == "pose" and k in POSE_EXTRA:
                continue
            extra_bad.append((path, k)); continue
        if k not in y:
            bad.append((path, k, "missing")); continue
        if isinstance(x[k], list) and x[k] and isinstance(x[k][0], dict):
            for i, (u, v) in enumerate(zip(x[k], y[k])):
                cmp(u, v, f"{path}.{k}[{i}]")
            continue
        n_arr += 1
        if mode == "pose" and k == "kind":
            if not (x[k] == "'speech'" and y[k] == "'pose'"):
                bad.append((path, k, x[k], y[k]))
            continue
        if mode == "pose" and k == "own":
            if x[k] != y.get("own3"):
                bad.append((path, "own[:, :3]"))
            continue
        if x[k] != y[k]:
            bad.append((path, k))


for L in map(str, range(64)):
    ra, rb = a["ranks"][L], b["ranks"][L]
    assert ra["item0"] == rb["item0"] and ra["eval_item0"] == rb["eval_item0"]
    cmp(ra["train"], rb["train"], f"L{L}.train")
    cmp(ra["eval"], rb["eval"], f"L{L}.eval")
print(f"{mode}: {n_arr} digests compared over 64 logical ranks; mismatches {len(bad)}; extra-key problems {len(extra_bad)}")
print("bad", bad[:10], "extra", extra_bad[:10])
print("PASS" if not bad and not extra_bad else "FAIL")
