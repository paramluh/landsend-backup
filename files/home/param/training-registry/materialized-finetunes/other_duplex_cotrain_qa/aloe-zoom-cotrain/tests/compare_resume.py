"""compare the resume test's metrics.jsonl: A (uninterrupted), B (stopped after 3), C (resumed from B's save). usage: compare_resume.py <runs dir> <A> <B> <C>"""
import json, sys
R = sys.argv[1]
def rd(tag):
    out = {"train": {}, "eval": {}}
    for ln in open(f"{R}/{tag}/metrics.jsonl"):
        d = json.loads(ln)
        k = "eval" if any(x.startswith("eval") for x in d if x not in ("step", "time_pt")) else "train"
        out[k][d["step"]] = d
    return out
A, B, C = (rd(t) for t in sys.argv[2:5])
ok = True
e_b, e_c = B["eval"][3], C["eval"][3]
keys = [k for k in e_b if k.startswith("eval") and k != "eval/time_s"]
dif = {k: (e_b[k], e_c[k]) for k in keys if e_b[k] != e_c[k]}
print(f"[1] EVAL step 3: B (before its save) vs C (after restoring the save): {len(keys) - len(dif)} / {len(keys)} values bit-identical", "" if not dif else dif)
ok &= not dif
cnt = ("train/text_targets", "train/cb0_labels", "train/light_frames", "train/zoom_text_targets") + tuple(k for k in A["train"][4] if k.startswith("train/lr_")) + ("train/dec_lr_first",)
for s in (4, 5, 6):
    d = {k: (A["train"][s][k], C["train"][s][k]) for k in cnt if A["train"][s][k] != C["train"][s][k]}
    print(f"[2] step {s}: data counts + LRs C == A:", "identical" if not d else d)
    ok &= not d
for s in (1, 2, 3):
    d = {k: (A["train"][s][k], B["train"][s][k]) for k in cnt if A["train"][s][k] != B["train"][s][k]}
    ok &= not d
L = ("train_speech/word_ce_norm", "train_speech/cb0_ce", "train_speech/light_ce", "train_zoom/word_ce_norm", "train/loss_aloe", "train/loss_zoom", "train/gn_trunk", "train/vox_a_scale")
for s in (1, 2, 3, 4, 5, 6):
    src = B if s <= 3 else C
    print(f"[3] step {s}: A vs {'B' if s <= 3 else 'C'}: " + " ".join(f"{k.split('/')[-1]} {A['train'][s][k]:.6f}/{src['train'][s][k]:.6f} (rel {abs(A['train'][s][k] - src['train'][s][k]) / max(abs(A['train'][s][k]), 1e-12):.1e})" for k in L))
mx = max(abs(A["train"][s][k] - C["train"][s][k]) / max(abs(A["train"][s][k]), 1e-12) for s in (4, 5, 6) for k in L)
mxb = max(abs(A["train"][s][k] - B["train"][s][k]) / max(abs(A["train"][s][k]), 1e-12) for s in (1, 2, 3) for k in L)
print(f"[3] max rel diff A vs C (steps 4-6) {mx:.2e}; A vs B before the stop (steps 1-3, pure backward nondeterminism) {mxb:.2e}")
e_a6, e_c6 = A["eval"][6], C["eval"][6]
print("[4] EVAL step 6 A vs C: " + " ".join(f"{k} {e_a6[k]:.5f}/{e_c6[k]:.5f}" for k in ("eval_speech/word_ce_norm", "eval_speech/cb0_ce", "eval_speech/light_ce", "eval_zoom/word_ce_norm")))
ok &= mx < max(10 * mxb, 1e-3)
print("RESUME TEST", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
