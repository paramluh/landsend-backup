#!/usr/bin/env python3
"""posefix scorer baselines -> posefix/score/out/{gt_actors,gt_aloe,sessions,offline}.json + SUMMARY.md (CPU decode).

  gt_actors   video/demo-data/prompts/*/gt/pose_tokens.safetensors (12 s), each denormalised with its OWN frame0_pose, filters none
  gt_aloe     held-out (POSE_INDEX split 'eval', first chunk c00) handover tracks, both sides, the 8 sample convs c0..c7 first, then
              the next eval convs (sorted) up to N_ALOE; first 7500 tokens (10 min, the training row); person 01; none + align+orient
  sessions    the live sessions (person 01); none + align+orient
  sim         posefix/out/{parity,conv}/*.npz simulator outputs (gen part after the actor-GT prefill, whole, conv matched GT) ->
              SUMMARY_SIM.md
  offline     samples/runs/B-00*/pose_s{1,2}.npy (generated after a 10 s = 125-token real prefill; measured from token 125) and the
              matched gt_pose_s{1,2}.npy over the same window; person 01; none + align+orient
usage: run_baselines.py [--groups gt_actors,gt_aloe,sessions,offline] [--summary-only]"""
import argparse
import glob
import json
import os
import sys
import time

H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, H)
import numpy as np  # noqa: E402
import torch  # noqa: E402
import score as S  # noqa: E402

OUT = os.path.join(H, "out")
PERSON = S.DEFAULT_PERSON
ALOE = "/data/home/amartyavarma/data/aloe-handover-full/DATA"
POSE_INDEX = "/data/home/amartyavarma/ppm/aloe-ft-32-samsep/poseduplex/data/POSE_INDEX.json"
CONVS = "/data/home/amartyavarma/ppm/aloe-ft-32-samsep/poseduplex/samples/convs"
RUNS = "/data/home/amartyavarma/ppm/aloe-ft-32-samsep/poseduplex/samples/runs"
LIVE = "/data/home/amartyavarma/ppm/aloe-posedup-live"
SESSIONS = [("20261025-142135-d545", "CFG 3"), ("20261025-142453-1086", "CFG off"), ("20261025-140015-5d7e", "CFG 3")]
N_ALOE = 12
FILTERS2 = ("none", "align+orient")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def run(codes, person, filters_list, name, info, start_token=0, max_tokens=None):
    f0 = S.frame0_of(person)
    c, _, _ = S.prep_codes(codes, start_token, max_tokens)
    dec = S.decode(c, "cpu")
    out = []
    for flt in filters_list:
        r = S.score_stream(codes, f0, flt, start_token, max_tokens, name=name, info=dict(info, person=person), dec=dec)
        w = r["whole"]
        log(f"{name:40s} {flt:13s} {w['seconds']:7.1f}s conf b/h {w['conf_body']}/{w['conf_hand']} speed p50 {w.get('speed_p50')} "
            f"jumps/s {w.get('jumps_per_s')} ps p50 {w.get('ps_speed_p50')} centre {w['centre_actor_p50']} width {w['width_actor_p50']} "
            f"oof {w['out_of_frame']} cb0chg {w.get('cb0_change')}")
        out.append(r)
    return out


def gt_actors():
    res = []
    for d in sorted(glob.glob(os.path.join(S.PROMPTS, "*", "gt", "pose_tokens.safetensors"))):
        person = d.split("/")[-3]
        codes, info = S.load_codes(d)
        res += run(codes, person, ("none",), person, info)
    return res


def aloe_tracks():
    pix = json.load(open(POSE_INDEX))["rows"]
    ev = {}
    for k, d in pix.items():
        for r, v in d.items():
            if v["split"] == "eval" and v["row_key"].endswith("/c00"):
                ev[v["conv"]] = v
    first = []
    for i in range(8):
        cj = json.load(open(os.path.join(CONVS, f"c{i}.json")))
        first.append(cj["row_key"].split("/", 2)[2].rsplit("/", 1)[0])
    order = [c for c in first if c in ev] + sorted(c for c in ev if c not in first)
    tracks = []
    for conv in order:
        v = ev[conv]
        sides = [s for s in ("assistant", "user") if v["sides"].get(s, {}).get("ok") and v["sides"][s]["T"] >= 1500]
        if len(sides) < 2:
            continue
        for s in sides:
            tracks.append((conv, s, os.path.join(ALOE, v["sides"][s]["path"])))
        if len(tracks) >= 2 * N_ALOE:
            break
    return tracks


def gt_aloe():
    res = []
    for conv, side, path in aloe_tracks():
        codes, info = S.load_codes(path)
        info.update(conv=conv, side=side)
        res += run(codes, PERSON, FILTERS2, f"{conv.split('-')[0]}:{side}", info, max_tokens=7500)
    return res


def sessions():
    res = []
    for sid, lab in SESSIONS:
        d = os.path.join(LIVE, "sessions", sid)
        if not os.path.isdir(d):
            d = next(iter(sorted(glob.glob(os.path.join(LIVE, "stage-lb", "*", "sessions", sid)))))
        codes, info = S.load_codes(d)
        info.update(label=lab)
        res += run(codes, PERSON, FILTERS2, f"{sid} ({lab})", info)
    return res


def offline():
    res = []
    for d in sorted(glob.glob(os.path.join(RUNS, "B-00*"))):
        if not os.path.isdir(d):
            continue
        for s in ("s1", "s2"):
            for kind, fn in (("gen", f"pose_{s}.npy"), ("gt", f"gt_pose_{s}.npy")):
                p = os.path.join(d, fn)
                if not os.path.exists(p):
                    continue
                codes, info = S.load_codes(p)
                info.update(kind=kind, run=os.path.basename(d), side=s)
                res += run(codes, PERSON, FILTERS2, f"{os.path.basename(d)}:{s}:{kind}", info, start_token=125)
    return res


SIM = os.path.join(LIVE, "posefix", "out")
SIM_GROUPS = ["_pf0", "_pf2", "_pf5", "_pf1168"]
PARITY = ["1086_repro", "1086_pf2", "1086_pf5", "1086_pf1168"]


def sim():
    """posefix simulator outputs out/{parity,conv}/<name>.npz (key 'pose' [F, 32] from code frame 0; <name>.json pose_prefill_frames =
    leading frames that are the actor's real GT tokens). Per file: part 'gen' (frames >= prefill), 'whole' (prefill > 0 only) and, for
    conv/cNsK, 'gtmatch' = convs/cN.npz gtpose_K over the same frame window."""
    res = []
    for p in sorted(glob.glob(os.path.join(SIM, "*", "*.npz"))):
        jp = p[:-4] + ".json"
        if not os.path.exists(jp):
            continue
        j = json.load(open(jp))
        d, base = p.split("/")[-2], os.path.basename(p)[:-4]
        codes, info = S.load_codes(p, "pose")
        pf = int(j.get("pose_prefill_frames") or 0)
        info.update(sim_dir=d, base=base, pf=pf, suffix="_" + base.rsplit("_", 1)[1], seed=j.get("seed"),
                    sim_gen_cb0_change=j.get("gen_cb0_change"), sim_gen_cb0_distinct=j.get("gen_cb0_distinct"))
        res += run(codes, PERSON, FILTERS2, f"{d}/{base}:gen", dict(info, part="gen"), start_token=pf)
        if pf:
            res += run(codes, PERSON, FILTERS2, f"{d}/{base}:whole", dict(info, part="whole"))
        if d == "conv" and base[0] == "c":
            ci, si = base[1:].split("_")[0].split("s")
            gt = np.load(os.path.join(CONVS, f"c{int(ci)}.npz"))[f"gtpose_{int(si)}"]
            gc = S._fix_codes(gt)
            res += run(gc[:len(codes)], PERSON, FILTERS2, f"{d}/{base}:gtmatch", dict(info, part="gtmatch", src=f"c{int(ci)}.npz gtpose_{int(si)}"),
                       start_token=pf)
    return res


def sim_summary(R):
    rs = R.get("sim", [])
    sel = lambda g, flt, pred=lambda r: True: [r for r in R.get(g, []) if r["filters"] == flt and pred(r)]  # noqa: E731
    rows = []
    for flt in FILTERS2:
        rows.append((f"GT Aloe held-out B tracks (10 min) [{flt}]", sel("gt_aloe", flt)))
    for flt in FILTERS2:
        rows.append((f"live 1086 CFG off [{flt}]", sel("sessions", flt, lambda r: r["name"].startswith("20261025-142453-1086"))))
    for flt in FILTERS2:
        rows.append((f"conv matched GT, all windows [{flt}]", [r for r in rs if r["filters"] == flt and r["info"]["part"] == "gtmatch"]))
        for suf in SIM_GROUPS:
            rows.append((f"sim conv *{suf} gen [{flt}]", [r for r in rs if r["filters"] == flt and r["info"]["sim_dir"] == "conv"
                                                         and r["info"]["suffix"] == suf and r["info"]["part"] == "gen"]))
    for flt in FILTERS2:
        for b in PARITY:
            rows.append((f"sim {b} gen [{flt}]", [r for r in rs if r["filters"] == flt and r["info"]["base"] == b and r["info"]["part"] == "gen"]))
    rows = [(n, x) for n, x in rows if x]
    spread = (f"GT Aloe held-out B tracks (10 min) [none]", sel("gt_aloe", "none"))
    md = ["# posefix simulator outputs scored (score.py; person 01_ab63d745; 'gen' = frames after the actor-GT pose prefill)", "",
          "Same units / caveats as SUMMARY.md. conv = sim on held-out convs c0..c7 (side s0/s1), grouped by prefill suffix; matched GT = "
          "the same conv side's real B pose over the same frames (from the pf0 window start). Group rows = mean over streams. "
          f"Files scored: {len({r['info']['base'] + r['info']['sim_dir'] for r in rs})}. Per-stream numbers: out/sim.json.", "",
          "## Confidence, motion, tokens", "", table(rows, COLS1, spread), "",
          "## Framing, head (held angles)", "", table(rows, COLS2, spread), "",
          "## Body proportions", "", table(rows, COLS3, spread), ""]
    chk = [(r["info"]["base"], r["whole"].get("cb0_change"), r["info"].get("sim_gen_cb0_change")) for r in rs
           if r["filters"] == "none" and r["info"]["part"] == "gen"]
    bad = [c for c in chk if c[1] is not None and c[2] is not None and abs(c[1] - c[2]) > 0.02]
    md += [f"Cross-check: scorer cb0 change on 'gen' vs the simulator's own gen_cb0_change: {len(chk) - len(bad)}/{len(chk)} within 0.02"
           + (f"; off: {bad[:5]}" if bad else ""), ""]
    open(os.path.join(OUT, "SUMMARY_SIM.md"), "w").write("\n".join(md))
    log("wrote", os.path.join(OUT, "SUMMARY_SIM.md"))


# ------------------------------------------------------------------------------------------------ summary
COLS1 = [("conf_body", "conf body", 2), ("conf_hand", "conf hand", 2), ("lowconf_frac", "conf<0.3", 2), ("conf_flicker", "flicker", 3),
         ("speed_p50", "speed p50", 4), ("speed_p95", "speed p95", 3), ("jumps_per_s", "jumps/s", 2),
         ("ps_speed_p50", "ps speed p50", 4), ("ps_jumps_per_s", "ps jumps/s", 2),
         ("cb0_change", "cb0 chg", 2), ("cb0_distinct_per30s", "cb0 dist/30s", 0), ("light_change", "cb1-31 chg", 2)]
COLS2 = [("centre_own_p50", "ctr own p50", 2), ("centre_actor_p50", "ctr actor p50", 2), ("centre_actor_p95", "ctr actor p95", 2),
         ("centre_actor_end", "ctr actor end", 2), ("width_actor_p50", "w/actor p50", 2), ("width_actor_p05", "w p05", 2),
         ("width_actor_p95", "w p95", 2), ("width_actor_end", "w end", 2), ("width_collapse_frac", "w<0.5", 3), ("out_of_frame", "oof", 3),
         ("out_of_frame_body", "oof body", 3), ("out_of_frame_hand", "oof hand", 3), ("out_of_frame_bottom_share", "oof @bottom", 2),
         ("head_valid_frac", "head ok", 2), ("pitch_p50", "pitch p50", 2), ("pitch_sd_slow", "pitch sd(slow)", 2), ("pitch_drift", "pitch drift", 2),
         ("yaw_p50", "yaw p50", 2), ("yaw_sd_slow", "yaw sd(slow)", 2), ("roll_p50", "roll p50", 1), ("roll_sd_slow", "roll sd(slow)", 1)]
COLS3 = [("prop_eye", "eye/sh", 3), ("prop_ear", "ear/sh", 3), ("prop_upper_arm", "upper arm/sh", 3), ("prop_forearm", "forearm/sh", 3),
         ("seconds", "s", 0)]


def _fmt(v, n):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "-"
    return f"{v:.{n}f}" if n else f"{v:.0f}"


def _mean(rs, k):
    v = [r["whole"].get(k) for r in rs]
    v = [x for x in v if x is not None]
    return float(np.mean(v)) if v else None


def _pct(rs, k, q):
    v = [r["whole"].get(k) for r in rs]
    v = [x for x in v if x is not None]
    return float(np.percentile(v, q)) if v else None


def table(rows, cols, spread=None, ref=None):
    """spread: (label, results) -> an extra row of p10-p90 across those streams; ref: (label, dict) -> a row of reference values."""
    head = "| group | n | " + " | ".join(c[1] for c in cols) + " |"
    sep = "|---|---|" + "---|" * len(cols)
    lines = [head, sep]
    for name, rs in rows:
        lines.append(f"| {name} | {len(rs)} | " + " | ".join(_fmt(_mean(rs, k), n) for k, _, n in cols) + " |")
        if spread and name == spread[0]:
            lines.append(f"| ↳ same, p10-p90 across tracks | | " + " | ".join(
                f"{_fmt(_pct(rs, k, 10), n)}-{_fmt(_pct(rs, k, 90), n)}" for k, _, n in cols) + " |")
    if ref:
        lines.append(f"| {ref[0]} | | " + " | ".join(_fmt(ref[1].get(k), n) if ref[1].get(k) is not None else "" for k, _, n in cols) + " |")
    return "\n".join(lines)


def summary(R):
    rows = []
    sel = lambda g, flt, pred=lambda r: True: [r for r in R.get(g, []) if r["filters"] == flt and pred(r)]  # noqa: E731
    rows.append(("GT actors (20 x 12 s, own frame 0)", sel("gt_actors", "none")))
    for flt in FILTERS2:
        rows.append((f"GT Aloe held-out B tracks (10 min) [{flt}]", sel("gt_aloe", flt)))
    for sid, lab in SESSIONS:
        for flt in FILTERS2:
            rows.append((f"live {sid[-4:]} {lab} [{flt}]", sel("sessions", flt, lambda r, s=sid: r["name"].startswith(s))))
    for flt in FILTERS2:
        rows.append((f"offline gen B-00x (after 10 s prefill) [{flt}]", sel("offline", flt, lambda r: r["info"]["kind"] == "gen")))
        rows.append((f"offline matched GT (same window) [{flt}]", sel("offline", flt, lambda r: r["info"]["kind"] == "gt")))
    rows = [(n, rs) for n, rs in rows if rs]
    spread = (f"GT Aloe held-out B tracks (10 min) [none]", sel("gt_aloe", "none"))
    p01 = next((r for r in R.get("gt_actors", []) if r["name"] == PERSON), None)
    ref2 = ref3 = None
    if p01:
        f0 = p01["ref_actor_frame0"]
        ref2 = (f"{PERSON} frame-0 reference", {f"{a}_p50": v for a, v in f0["head"].items()})
        ref3 = (f"{PERSON} frame-0 reference", f0)
    md = ["# posefix pose-quality scores (score.py; CPU decode = pose.decode to 7.6e-6; person 01_ab63d745 unless noted)", "",
          "Units: speed / centre in actor frame-0 shoulder widths (sw) per 25 fps frame; width = horizontal shoulder width / actor "
          "frame-0 width; ps_* = video/dmd-rt/rt/pose_stats.py definitions; oof = confident joints outside 256x480; head angles in "
          "eye-distance units (roll deg), slow = trailing 2 s mean; group rows = mean over streams of per-stream values. Full "
          "definitions: score.py docstring; per-stream + per-30 s-window numbers: out/*.json.", "",
          "Caveats: (1) oof depends on the actor's crop: REAL Aloe tracks placed in person 01's frame already put ~21 % of confident "
          "joints outside (p10-p90 1-43 %), so compare oof to that row, not to 0. (2) jumps > 0.08 sw/frame are natural fast gestures "
          "in real data (Aloe 0.7/s, actor clips 1.7/s), not glitches; ps_* also count low-confidence joints and are inflated by "
          "off-frame hand flailing (d545 ps jumps 0.53/s -> 0.02/s once align clamps them in). (3) forearm/sh is posture "
          "(foreshortening) dependent; eye/ear/upper-arm vs sh are the shape signals. (4) with orient on, head angles equal the "
          "actor frame-0 reference by construction (pitch -0.40, roll -5.8 deg); read the [none] rows for the model's held angles.", "",
          "## Confidence, motion, tokens", "", table(rows, COLS1, spread), "",
          "## Framing, head (held angles)", "", table(rows, COLS2, spread, ref2), "",
          "## Body proportions (segment / Euclidean shoulder distance, median)", "", table(rows, COLS3, spread, ref3), ""]
    act = sel("gt_actors", "none")
    if act:
        md += ["## Actors: frame-0 reference proportions (from frame0_pose) vs decoded GT clip medians", "",
               "| person | eye/sh f0 | ear/sh f0 | upper arm/sh f0 | forearm/sh f0 | eye/sh clip | upper arm/sh clip | forearm/sh clip | conf body | speed p50 |",
               "|---|---|---|---|---|---|---|---|---|---|"]
        for r in act:
            f0, w = r["ref_actor_frame0"], r["whole"]
            md.append(f"| {r['name']}{' (default avatar)' if r['name'] == PERSON else ''} | {_fmt(f0['prop_eye'], 3)} | {_fmt(f0['prop_ear'], 3)} | "
                      f"{_fmt(f0['prop_upper_arm'], 3)} | {_fmt(f0['prop_forearm'], 3)} | {_fmt(w['prop_eye'], 3)} | "
                      f"{_fmt(w['prop_upper_arm'], 3)} | {_fmt(w['prop_forearm'], 3)} | {_fmt(w['conf_body'], 2)} | {_fmt(w['speed_p50'], 4)} |")
        md.append("")
    ses = [r for r in R.get("sessions", [])]
    if ses:
        md += ["## Live sessions per 30 s window (centre vs actor p50 / width ratio p50 / pitch p50 / cb0 change / speed p50)", ""]
        for r in ses:
            ws = " · ".join(f"{w['t0_s']:.0f}s: {_fmt(w['centre_actor_p50'], 2)}/{_fmt(w['width_actor_p50'], 2)}/{_fmt(w['pitch_p50'], 2)}/"
                            f"{_fmt(w.get('cb0_change'), 2)}/{_fmt(w.get('speed_p50'), 4)}" for w in r["windows"])
            md.append(f"- {r['name']} [{r['filters']}]: {ws}")
        md.append("")
    open(os.path.join(OUT, "SUMMARY.md"), "w").write("\n".join(md))
    log("wrote", os.path.join(OUT, "SUMMARY.md"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", default="gt_actors,gt_aloe,sessions,offline,sim")
    ap.add_argument("--summary-only", action="store_true")
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "16")))
    os.makedirs(OUT, exist_ok=True)
    fns = {"gt_actors": gt_actors, "gt_aloe": gt_aloe, "sessions": sessions, "offline": offline, "sim": sim}
    R = {}
    for g in fns:
        p = os.path.join(OUT, f"{g}.json")
        if g in a.groups.split(",") and not a.summary_only:
            t = time.time()
            R[g] = fns[g]()
            json.dump(R[g], open(p, "w"), indent=1)
            log(f"group {g}: {len(R[g])} results in {time.time() - t:.0f}s -> {p}")
        elif os.path.exists(p):
            R[g] = json.load(open(p))
    summary(R)
    if R.get("sim"):
        sim_summary(R)


if __name__ == "__main__":
    main()
