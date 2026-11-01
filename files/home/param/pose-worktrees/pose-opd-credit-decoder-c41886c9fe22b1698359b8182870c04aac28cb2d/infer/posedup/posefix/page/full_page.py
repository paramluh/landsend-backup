#!/usr/bin/env python3
"""posefix FULL-PREFILL media for main's plain page: for every run in out/{fullD,fullM} whose avatar render exists in clips/{fullD,fullM}:
a/full-<kind>-<name>.mp4 = the render + the run's own B voice + white flash / beep at the prefill -> generated switch (mux.mux_one), and the
scorer's headline on the GENERATED 30 s (start token = prefill frames). GT per person = clips/gt/<person>_gt.mp4 (her own real 12 s) muxed
with her real audio (gt/mimi_local decoded). Writes full_manifest.json (groups: person, then prefill length, then arm; per clip GT /
Monoplex / Duplex cells with label "pose + words + audio for N s") and prints the files map to publish.
usage: python full_page.py [workers]"""
import glob
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

P = "/data/home/amartyavarma/ppm/aloe-posedup-live/posefix"
sys.path.insert(0, f"{P}/page")
import mux as MX  # noqa: E402

W = int(sys.argv[1]) if len(sys.argv) > 1 else 8
RS = os.environ.get("RENDER_SET", "")                                 # 10-26: videort re-render sets: "contract" (wavefront contract maths)
CT = RS in ("contract", "seq", "seq2")                                       # or "seq" (Vansh's sequential sampler, the user's choice)
SFX, PFX = {"contract": ("_contract", "c-"), "seq": ("_seq", "s-"), "seq2": ("_seq2", "s2-")}.get(RS, ("", ""))
SCORE_PY = "/data/home/amartyavarma/video/venv-distilled/bin/python"
PERSON_LABEL = {"10_27286bdd": "Person 10", "01_ab63d745": "Person 01"}
INPUT_LABEL = {"1086": "live call 1086 replay (duplex hears the user's mic)", "c0s0": "held-out Aloe c0 (duplex hears the partner)",
               "c1s0": "held-out Aloe c1 (duplex hears the partner)"}
runs = []
for kind in ("fullD", "fullM", "openD", "openM"):
    for j in sorted(glob.glob(f"{P}/out/{kind}/*.json")):
        name = os.path.basename(j)[:-5]
        if not os.path.isfile(f"{P}/out/{kind}/{name}.npz"):
            continue
        if kind.startswith("open"):                                   # cv<i>_open<s>_<arm>: held-out conv i's real opening, avatar Person 01
            cv, op, arm = name.split("_")
            runs.append(dict(kind=kind, name=name, person="01_ab63d745", input=f"c{cv[2:]}s0", s=op[4:], arm=arm, open=True, meta=json.load(open(j))))
            continue
        pfx, inp, full, arm = name.split("_")
        runs.append(dict(kind=kind, name=name, person={"p10": "10_27286bdd", "p01": "01_ab63d745"}[pfx], input=inp, s=full[4:], arm=arm,
                         meta=json.load(open(j))))


def media(r):
    src = f"{P}/clips/{r['kind']}{SFX}/{r['name']}.mp4"
    out = f"{P}/page/a/{PFX}full-{r['kind']}-{r['name']}.mp4"
    if not os.path.isfile(src):
        return None
    if not os.path.isfile(out) or os.path.getmtime(out) < os.path.getmtime(src):
        z = np.load(f"{P}/out/{r['kind']}/{r['name']}.npz")
        N = int(r["meta"]["pose_prefill_frames"])
        st = MX.mux_one(src, z["b"], len(z["pose"]), N * 0.08, out)
        if not st.startswith("ok"):
            return None
    return f"a/{PFX}full-{r['kind']}-{r['name']}.mp4"


def score(group):
    """runs with the same prefill frames N -> {name: whole} (generated 30 s)."""
    N = int(group[0]["meta"]["pose_prefill_frames"])
    outj = f"{P}/page/scores/full_N{N}.json"
    ins = [f"{P}/out/{r['kind']}/{r['name']}.npz" for r in group]
    by = {}
    for person in sorted({r["person"] for r in group}):
        sel = [f"{P}/out/{r['kind']}/{r['name']}.npz" for r in group if r["person"] == person]
        oj = outj.replace(".json", f"_{person}.json")
        subprocess.run([SCORE_PY, f"{P}/score/score.py", *sel, "--person", person, "--filters", "none", "--start-token", str(N),
                        "--max-tokens", str(N + 375), "--out", oj], capture_output=True)
        if os.path.isfile(oj):
            for it in json.load(open(oj)):
                by[os.path.basename(it["info"]["src"])[:-4] + "|" + it["info"]["src"].split("/")[-2]] = it["whole"]
    return by


os.makedirs(f"{P}/page/scores", exist_ok=True)
with ThreadPoolExecutor(W) as ex:
    vids = list(ex.map(media, runs))
groups = {}
for r in runs:
    groups.setdefault(int(r["meta"]["pose_prefill_frames"]), []).append(r)
sc = {}
for g in groups.values():
    sc.update(score(g))
gt = {}
for person in sorted({r["person"] for r in runs}):
    src = f"{P}/clips/gt{SFX}/{person}_gt.mp4"
    out = f"{P}/page/a/{PFX}full-gt-{person}.mp4"
    if os.path.isfile(src):
        if not os.path.isfile(out):
            import torch  # noqa: F401
            sys.path.insert(0, "/data/home/amartyavarma/ppm/aloe-pose-monoplex-infer/ppm")
            import ppm_io as IO
            from safetensors.numpy import load_file
            c = np.asarray(load_file(f"/data/home/amartyavarma/video/demo-data/prompts/{person}/gt/mimi_local.safetensors")["codes"], np.int64)
            wav = np.asarray(IO.Mimi("cpu").decode(c), np.float32).reshape(-1)
            MX.mux_one(src, wav, c.shape[1], None, out)
        gt[person] = f"a/{PFX}full-gt-{person}.mp4" if os.path.isfile(out) else None


def head(ws):
    ws = [w for w in ws if w]
    if not ws:
        return None
    m = lambda k: float(np.mean([w[k] for w in ws if w.get(k) is not None]))  # noqa: E731
    return {"n": len(ws), "cb0_repeat_pct": round(100 * (1 - m("cb0_change")), 1), "out_of_frame_pct": round(100 * m("out_of_frame"), 1),
            "conf": round(m("conf_all"), 3)}


man = {"title": "Full prefill (pose + words + audio): monoplex vs duplex", "groups": []}
for s_ in ("2", "5", "10", "1168"):                                   # CONVERSATION OPENINGS first (main: prefill from the conversation start)
    for arm in ("lk151", "lk30"):
        rs = [r for r, v in zip(runs, vids) if r.get("open") and r["s"] == s_ and r["arm"] == arm]
        if not rs:
            continue
        N = int(rs[0]["meta"]["pose_prefill_frames"])
        lab = f"Conversation opening 0-{N * 0.08:.2f} s prefilled (pose + words + audio) · pose light top-k {arm[2:]}"
        clips = []
        for i in range(4):
            inp = f"c{i}s0"
            gv = f"a/{PFX}gtc-{inp}-{N + 375}.mp4"
            if CT and not os.path.isfile(f"{P}/page/{gv}") and os.path.isfile(f"{P}/clips/gtc{SFX}/{inp}.mp4"):
                st_ = np.load(f"/data/home/amartyavarma/ppm/aloe-ft-32-samsep/poseduplex/samples/convs/{inp[:2]}.npz")[f"stem_{inp[3]}"].astype(np.float32)
                MX.mux_one(f"{P}/clips/gtc{SFX}/{inp}.mp4", st_, N + 375, None, f"{P}/page/{gv}")
            cells = [{"kind": "gt", "label": "Real (this conversation's B, real pose + voice)", "video": gv if os.path.isfile(f"{P}/page/{gv}") else None}]
            for kind, ml in (("openM", "Monoplex (export-357)"), ("openD", "Duplex (export-498)")):
                r = next((x for x in rs if x["input"] == inp and x["kind"] == kind), None)
                if r is None:
                    continue
                w = sc.get(f"{r['name']}|{kind}")
                cells.append({"kind": "fullM" if kind == "openM" else "fullD", "label": ml, "video": vids[runs.index(r)], "seed": r["meta"]["seed"],
                              "prefilled": f"conversation opening 0-{N * 0.08:.2f} s (pose + words + audio)", "score": head([w])})
            clips.append({"input": f"held-out Aloe conversation c{i} (B = side 0; the duplex hears A's real audio from t = 0; avatar Person 01)", "cells": cells})
        man["groups"].append({"label": lab, "person": "01_ab63d745", "prefill_frames": N, "arm": arm, "open": True,
                              "headline": {"monoplex": head([sc.get(f"{r['name']}|openM") for r in rs if r["kind"] == "openM"]),
                                           "duplex": head([sc.get(f"{r['name']}|openD") for r in rs if r["kind"] == "openD"])},
                              "clips": clips})
for person in ("10_27286bdd", "01_ab63d745"):
    for s in ("2", "5", "10", "1168"):
        for arm in ("lk151", "lk30"):
            rs = [r for r, v in zip(runs, vids) if not r.get("open") and r["person"] == person and r["s"] == s and r["arm"] == arm]
            if not rs:
                continue
            N = int(rs[0]["meta"]["pose_prefill_frames"])
            lab = f"{PERSON_LABEL[person]} · pose + words + audio for {N * 0.08:.2f} s · pose light top-k {arm[2:]}"
            clips = []
            for inp in ("1086", "c0s0", "c1s0"):
                cells = [{"kind": "gt", "label": f"Real ({PERSON_LABEL[person]}'s own 12 s)", "video": gt.get(person)}]
                for kind, ml in (("fullM", "Monoplex (export-357)"), ("fullD", "Duplex (export-498)")):
                    r = next((x for x in rs if x["input"] == inp and x["kind"] == kind), None)
                    if r is None:
                        continue
                    v = vids[runs.index(r)]
                    w = sc.get(f"{r['name']}|{kind}")
                    cells.append({"kind": kind, "label": ml, "video": v, "seed": r["meta"]["seed"], "prefilled": f"pose + words + audio for {N * 0.08:.2f} s",
                                  "score": head([w]), "words": r["meta"].get("words", "")[:300]})
                clips.append({"input": INPUT_LABEL[inp], "cells": cells})
            man["groups"].append({"label": lab, "person": person, "prefill_frames": N, "arm": arm,
                                  "headline": {"monoplex": head([sc.get(f"{r['name']}|fullM") for r in rs if r["kind"] == "fullM"]),
                                               "duplex": head([sc.get(f"{r['name']}|fullD") for r in rs if r["kind"] == "fullD"])},
                                  "clips": clips})
json.dump(man, open(f"{P}/page/{f'full_manifest{SFX}.json' if CT else 'full_manifest.json'}", "w"), indent=1)
# ---- data.json for main's plain builder (build_plain_main.py): data_base.json (the earlier groups) + the full-prefill groups first
D = json.load(open(f"{P}/page/data_base.json"))
D["convs"] = [c for c in D["convs"] if not c.get("full")]
if os.environ.get("FULL_ONLY"):                                       # page size cap: only the full-prefill / opening groups
    D["meta"]["groups"], D["convs"] = [], []
if os.environ.get("ACTOR_LK151_ONLY"):
    man["groups"] = [g for g in man["groups"] if g.get("open") or g["arm"] == "lk151"]
SHORT = {"10_27286bdd": "Person 10", "01_ab63d745": "Person 01"}
for g in man["groups"]:
    N = g["prefill_frames"]
    sec = N * 0.08
    hd = {k2: ({"repeat": v["cb0_repeat_pct"], "oof": v["out_of_frame_pct"], "conf": v["conf"]} if v else None)
          for k2, v in (("mono", g["headline"]["monoplex"]), ("dupM", g["headline"]["duplex"]))}
    glab = (f"Conversation opening 0-{sec:.2f} s prefilled (pose + words + audio) · light k{g['arm'][2:]}" if g.get("open") else
            f"FULL prefill (pose + words + audio) {sec:.2f} s · light k{g['arm'][2:]} · {SHORT[g['person']]}")
    D["meta"]["groups"].append({"label": glab, "head": {k2: v for k2, v in hd.items() if v}, "full": True})
    gi = len(D["meta"]["groups"]) - 1
    for c in g["clips"]:
        vers = []
        for cell in c["cells"]:
            if cell["kind"] == "gt":
                vers.append({"kind": "gt", "label": cell["label"], "tracks": [], "np_s": 0, "video": {"B": {"src": cell["video"]}} if cell["video"] else None})
            else:
                sc_ = cell.get("score")
                vers.append({"kind": "mono" if cell["kind"] == "fullM" else "dupM", "label": cell["label"], "tracks": [], "np_s": round(sec, 2),
                             "video": {"B": {"src": cell["video"]}} if cell["video"] else None, "seed": cell.get("seed"),
                             "score": {"repeat": sc_["cb0_repeat_pct"], "oof": sc_["out_of_frame_pct"], "conf": sc_["conf"]} if sc_ else None})
        D["convs"].append({"group": gi, "full": True, "title": (c["input"] if g.get("open") else f"{SHORT[g['person']]} · {c['input']}"),
                           "arm": f"pose light top-k {g['arm'][2:]}",
                           "prefill_label": (f"prefill = conversation opening 0-{sec:.2f} s (pose + words + audio), then generated" if g.get("open") else
                                             f"pose + words + audio for {sec:.2f} s of {SHORT[g['person']]}'s real recording ({N} frames), then generated"),
                           "np_frames": N, "ck": {"x": {"versions": vers}}})
json.dump(D, open(f"{P}/page/{f'data{SFX}.json' if CT else 'data.json'}", "w"))
files = {"full_manifest.json": "full_manifest.json"}
files.update({v: v for v in vids if v})
files.update({v: v for v in gt.values() if v})
json.dump(files, open(f"{P}/page/{f'full_files{SFX}.json' if CT else 'full_files.json'}", "w"))
print(f"{sum(1 for v in vids if v)}/{len(runs)} clips muxed; GT {gt}; {len(man['groups'])} groups; files {len(files)}")
