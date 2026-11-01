#!/usr/bin/env python3
"""pose6 A3: CPU-only test of pose6/worker/worker_s2pg0.py. No model load, no GPU; run with python -B (no .pyc anywhere).
Writes ONLY under pose6/worker/tests/out (every worker jwrite is checked against that root).

T1 constants: the DEPLOYED genweb worker.py (imported read-only in a subprocess) and the variant give identical DEF, LABEL, GROUP, PEOPLE,
   MAX_CLIP_S / MAX_PF_S, CD / PAD / DEPTH / SPF, OFF, persons() and schema_rows(persons()).
T2 imports: the variant's engine modules (fast_ppm, infer, ppm_io, g0src) come from pose6/infer; nothing from aloe-pose-monoplex-infer is
   loaded; stick resolves to genweb/stick.py; GENWEB_WEIGHTS defaults to s2pg0.
T3 call sites (fake engine + recording g0src stub): run() for every opening type the playground offers calls g0_from_opening(dir, kind) as
   briefs/CONTRACT.md says, and g0f reaches eng.start's instance dict. Also: the request override; bad overrides; g0src raising, which
   fails the run with a clear ValueError BEFORE eng.start.
T4 (only if pose6/infer/g0src.py exists) the REAL g0src through run() for every person x source: a finite f[8] or a loud error (listed).
T5 check_mods on fake binfo / models.
usage: python -B test_worker_cpu.py   (env: aloe-posedup-live/env_live.sh; exit 0 = all PASS)"""
import importlib.util
import json
import os
import subprocess
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
WDIR = os.path.dirname(HERE)
OUT = f"{HERE}/out"
VAR = f"{WDIR}/worker_s2pg0.py"
DEP = "/data/home/amartyavarma/ppm/aloe-posedup-live/genweb/worker.py"
GW = "/data/home/amartyavarma/ppm/aloe-posedup-live/genweb"
MI6 = "/data/home/amartyavarma/ppm/pose6/infer"
REAL_G0 = os.path.isfile(f"{MI6}/g0src.py")
os.makedirs(OUT, exist_ok=True)
os.environ.pop("GENWEB_WEIGHTS", None)
os.environ.pop("GENWEB_CK", None)
RES = {"real_g0src": REAL_G0, "tests": {}}
import hashlib  # noqa: E402
RES["md5"] = {p: hashlib.md5(open(p, "rb").read()).hexdigest() for p in (VAR, DEP, f"{MI6}/ppm/fast_ppm.py", f"{MI6}/g0src.py") if os.path.isfile(p)}
print("md5 " + json.dumps(RES["md5"]), flush=True)

DUMP = r"""
import importlib.util, json, sys
spec = importlib.util.spec_from_file_location(sys.argv[2], sys.argv[1])
W = importlib.util.module_from_spec(spec); spec.loader.exec_module(W)
pp = W.persons()
print("JSON" + json.dumps({"DEF": W.DEF, "LABEL": W.LABEL, "GROUP": W.GROUP, "PEOPLE": W.PEOPLE, "MAX": [W.MAX_CLIP_S, W.MAX_PF_S],
      "consts": [W.CD, W.PAD, W.DEPTH, W.SPF], "OFF": [int(v) for v in W.OFF], "persons": pp, "schema": W.schema_rows(pp),
      "PARAMS": {k: [t.__name__, lo, hi] for k, (t, lo, hi) in W.FP.PARAMS.items()}, "fp": W.FP.__file__}, default=str))
"""


def dump(path, name, env_extra=None):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", **(env_extra or {}))
    r = subprocess.run([sys.executable, "-B", "-c", DUMP, path, name], capture_output=True, text=True, env=env, timeout=600, cwd=OUT)
    line = [ln for ln in r.stdout.splitlines() if ln.startswith("JSON")]
    assert r.returncode == 0 and line, (path, r.returncode, r.stderr[-3000:])
    return json.loads(line[-1][4:])


def check(name, ok, detail=None):
    RES["tests"][name] = {"pass": bool(ok), "detail": detail}
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f": {json.dumps(detail, default=str)[:600]}" if detail is not None else ""), flush=True)


# ---------------------------------------------------------------- T1 constants (subprocesses: the two workers' engines share module names)
try:
    dep = dump(DEP, "genweb_worker_deployed")
    STUBDIR = f"{OUT}/stub_g0src"                                       # the variant needs g0src at import: the real one, else a stub
    if not REAL_G0:
        os.makedirs(STUBDIR, exist_ok=True)
        with open(f"{STUBDIR}/g0src.py", "w") as f:
            f.write('KINDS = ("openings", "openings-sam", "clip")\n'
                    'def g0_from_opening(opening_dir, kind):\n    raise NotImplementedError("stub")\n')
    var = dump(VAR, "genweb_worker_s2pg0", None if REAL_G0 else {"PYTHONPATH": STUBDIR})
    diffs = {k: (dep[k] == var[k]) for k in ("DEF", "LABEL", "GROUP", "PEOPLE", "MAX", "consts", "OFF", "persons", "schema", "PARAMS")}
    check("T1 constants deployed == variant", all(diffs.values()), {"equal": diffs, "deployed_fp": dep["fp"], "variant_fp": var["fp"],
                                                                    "n_persons": len(dep["persons"]), "n_schema_rows": len(dep["schema"])})
except Exception:  # noqa: BLE001
    check("T1 constants deployed == variant", False, traceback.format_exc()[-1500:])

# ---------------------------------------------------------------- import the variant in-process
if not REAL_G0:
    sys.path.insert(0, f"{OUT}/stub_g0src")
spec = importlib.util.spec_from_file_location("genweb_worker_s2pg0", VAR)
W = importlib.util.module_from_spec(spec)
spec.loader.exec_module(W)
W.WID = "cputest"
REAL_MOD = W.G0S

# ---------------------------------------------------------------- T2 imports
files = {k: os.path.abspath(getattr(W, k).__file__) for k in ("FP", "MIF", "IO", "G0S")}
want = {"FP": f"{MI6}/ppm/fast_ppm.py", "MIF": f"{MI6}/infer.py", "IO": f"{MI6}/ppm/ppm_io.py",
        "G0S": f"{MI6}/g0src.py" if REAL_G0 else f"{OUT}/stub_g0src/g0src.py"}
old = sorted({os.path.abspath(m.__file__) for m in list(sys.modules.values())
              if getattr(m, "__file__", None) and "aloe-pose-monoplex-infer" in m.__file__})
st = importlib.util.find_spec("stick")
check("T2 imports from pose6/infer, none from the deployed engine dir", files == want and not old and st is not None
      and os.path.abspath(st.origin) == f"{GW}/stick.py" and W.G == GW and W.WEIGHTS == "s2pg0" and W.CK == "",
      {"files": files, "deployed_engine_modules_loaded": old, "stick": st.origin if st else None, "G": W.G, "WEIGHTS": W.WEIGHTS,
       "label": W.WEIGHTS_LABEL[W.WEIGHTS]})

# ---------------------------------------------------------------- fakes
_jw = W.jwrite


def jwrite_guard(path, obj):
    assert os.path.abspath(path).startswith(OUT + "/"), ("test would write outside tests/out", path)
    return _jw(path, obj)


W.jwrite = jwrite_guard


class Stop(Exception):
    pass


class FakeEng:
    def __init__(self):
        self.calls = []

    def start(self, insts, seed, params, f0=0):
        self.calls.append({"insts": insts, "seed": seed, "f0": f0, "params": params})
        raise Stop()


class StubG0:
    KINDS = ("openings", "openings-sam", "clip")

    def __init__(self, fail=False):
        self.calls, self.fail = [], fail

    def g0_from_opening(self, d, kind):
        self.calls.append((d, kind))
        if self.fail:
            raise KeyError("orig_h")
        return [0.5, 0.5, -0.2, -1.0, 1.0, -1.0, 1.0, float(len(d) % 7)]


def make_worker():
    wk = object.__new__(W.Worker)
    wk.pp, wk.eng, wk.step, wk.mimi = W.persons(), FakeEng(), 0, None
    return wk


def req_of(person, src="start", s=1.92, pose=True, po=False, **kw):
    r = {"run_id": f"t-{person}-{src}-{s}-{int(pose)}-{int(po)}", "seed": 1234, "clip_s": 4.0,
         "prefill": {"person": person, "s": s, "src": src, "pose": pose, "audio": True, "words": True},
         "prompt": {"voice": "none", "persona": ""}, "params": {"pose_only": True} if po else {}, "hidden": True}
    r.update(kw)
    return r


def go(wk, req):
    rd = f"{OUT}/runs/{req['run_id']}"
    os.makedirs(rd, exist_ok=True)
    try:
        wk.run(rd, req)
    except Stop:
        return wk.eng.calls[-1], None
    except Exception as e:  # noqa: BLE001
        return None, e
    raise AssertionError("run() returned without reaching eng.start")


def fields(root, p):
    try:
        return "fields" in json.load(open(f"{GW}/{root}/{p}/meta.json"))["sources"]["pose_tokens"]["frame0"]
    except (OSError, KeyError, ValueError):
        return False


def has_opening(root, p, need_meta=False):
    return all(os.path.isfile(f"{GW}/{root}/{p}/{x}") for x in ("pose_tokens.safetensors", "mimi_local.safetensors")
               + (("meta.json",) if need_meta else ()))


def pkg(p):
    return W.package_dir(p)


def expect(p, src, s, pose, po, noav):
    """independent of g0_source: the frame each deployed code path decodes / renders against (worker.py f0gs / sgs)."""
    if po:
        for root in ("openings-sam", "openings"):
            if has_opening(root, p, need_meta=True):
                return f"{GW}/{root}/{p}", root
        return pkg(p), "clip"
    start = src == "start" and has_opening("openings", p)
    if start and fields("openings", p) and (s > 0 and pose or noav):
        return f"{GW}/openings/{p}", "openings"
    return pkg(p), "clip"


# ---------------------------------------------------------------- T3 call sites with the recording stub
stub = StubG0()
W.G0S = stub
wk = make_worker()
PP = [q["id"] for q in wk.pp]
cases = []
for p in PP:
    noav = W.no_avatar(p)
    for src, s, pose, po in (("start", 1.92, True, False), ("clip", 1.92, True, False), ("start", 0.0, True, False),
                             ("start", 1.92, False, False), ("start", 1.92, True, True), ("clip", 0.0, True, True)):
        cases.append((p, src, s, pose, po, noav))
bad = []
for (p, src, s, pose, po, noav) in cases:
    n0 = len(stub.calls)
    call, err = go(wk, req_of(p, src, s, pose, po))
    exp = expect(p, src, s, pose, po, noav)
    got = stub.calls[n0] if len(stub.calls) > n0 else None
    g = call["insts"][0].get("g0f") if call else None
    ok = (err is None and got == exp and len(stub.calls) == n0 + 1 and g is not None and g.dtype.name == "float32" and g.shape == (8,)
          and g.tolist() == [0.5, 0.5, float(__import__("numpy").float32(-0.2)), -1.0, 1.0, -1.0, 1.0, float(len(exp[0]) % 7)]
          and call["insts"][0]["a_mode"] == "empty" and call["f0"] == 0)
    if not ok:
        bad.append({"case": [p, src, s, pose, po, noav], "expected": exp, "got": got, "err": repr(err)})
kinds = sorted({expect(*c)[1] for c in cases})
check("T3a every person x opening type -> g0_from_opening(dir, kind) = the decode frame; g0f float32[8] in eng.start's instance dict",
      not bad and kinds == ["clip", "openings", "openings-sam"], {"cases": len(cases), "kinds_seen": kinds, "failures": bad[:6]})

call, err = go(wk, req_of(PP[0], g0f=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]))
n_before = len(stub.calls)
check("T3b request g0f override reaches eng.start (g0src not called)", err is None and
      call["insts"][0]["g0f"].tolist() == [float(v) for v in __import__("numpy").float32([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])]
      and len(stub.calls) == n_before, {"g0f": call["insts"][0]["g0f"].tolist() if call else None, "err": repr(err)})

outs = []
for g in ([0.1] * 7, [0.1] * 7 + [float("nan")], ["a"] * 8, 5, [[0.1] * 8]):
    n_eng = len(wk.eng.calls)
    call, err = go(wk, req_of(PP[0], g0f=g))
    outs.append({"g0f": repr(g)[:40], "err": f"{type(err).__name__}: {err}" if err else None, "engine_started": len(wk.eng.calls) > n_eng})
check("T3c bad request g0f -> ValueError before eng.start", all(o["err"] and o["err"].startswith("ValueError: G0COND") and not o["engine_started"]
                                                              for o in outs), outs)

W.G0S = StubG0(fail=True)
n_eng = len(wk.eng.calls)
call, err = go(wk, req_of(PP[0], "clip"))
msg = f"{type(err).__name__}: {str(err)[:400]}" if err else None                     # = loop()'s status.json "error" string
check("T3d g0src raising -> the run fails loudly (clear error, engine never started)", err is not None and isinstance(err, ValueError)
      and "G0COND: no frame-0 features for clip" in str(err) and len(wk.eng.calls) == n_eng, {"status_error": msg})

# ---------------------------------------------------------------- T4 the real g0src (when A2's file exists)
if REAL_G0:
    W.G0S = REAL_MOD
    import time as _t
    t_ = _t.time()
    wok, wbad = W.g0_warm(W.persons())                                  # T6: the startup warm over every pickable source (first call: cold)
    check("T6 g0_warm at worker start: every playground source warms", wok > 0 and not wbad,
          {"sources_ok": wok, "failing": wbad[:8], "seconds": round(_t.time() - t_, 2)})
    rows, errs = [], []
    for (p, src, s, pose, po, noav) in cases:
        call, err = go(wk, req_of(p, src, s, pose, po))
        d, kind = expect(p, src, s, pose, po, noav)
        r = {"person": p, "src": src, "s": s, "pose": pose, "pose_only": po, "kind": kind, "dir": d}
        if err is None:
            r["g0f"] = [round(float(v), 5) for v in call["insts"][0]["g0f"]]
        else:
            r["error"] = f"{type(err).__name__}: {str(err)[:300]}"
            errs.append(r)
        rows.append(r)
    json.dump(rows, open(f"{OUT}/real_g0src_cases.json", "w"), indent=1)
    import numpy as np
    finite = all(np.isfinite(r["g0f"]).all() for r in rows if "g0f" in r)
    check("T4 real g0src: every person x opening type -> finite f[8] (else a loud per-run error)", finite and not errs,
          {"cases": len(rows), "ok": len(rows) - len(errs), "errors": [(e["person"], e["kind"], e["error"][:160]) for e in errs][:10]})
else:
    check("T4 real g0src", True, "SKIPPED: pose6/infer/g0src.py not written yet (A2); T1-T3 used a stub with the contract's KINDS")


# ---------------------------------------------------------------- T5 check_mods
class _I:
    pass


def fake_m(g0):
    m = _I()
    m.inp = _I()
    if g0:
        m.inp.g0 = object()
    return m


t5 = []
for b, g0, want_ok in (({"s2p": "cb0", "g0cond": True}, True, True), ({"s2p": "cb0"}, True, True), ({"s2p": "full", "g0cond": True}, True, False),
                       ({"s2p": None}, True, False), ({"s2p": "cb0", "g0cond": True}, False, False),
                       ({"s2p": "cb0", "g0cond": True, "s2p_zero_init": True}, True, False),
                       ({"s2p": "cb0", "g0cond": True, "g0cond_zero_init": True}, True, False), ({"s2p": "cb0", "g0cond": False}, True, False)):
    try:
        W.check_mods(fake_m(g0), b)
        got = True
    except AssertionError:
        got = False
    t5.append({"binfo": b, "inp.g0": g0, "accepted": got, "expected": want_ok})
check("T5 check_mods accepts only s2p cb0 + inp.g0 loaded from the checkpoint", all(r["accepted"] == r["expected"] for r in t5), t5)

json.dump(RES, open(f"{OUT}/test_worker_cpu.json", "w"), indent=1, default=str)
n_fail = sum(not v["pass"] for v in RES["tests"].values())
print(f"SUMMARY {len(RES['tests']) - n_fail}/{len(RES['tests'])} PASS (real g0src: {REAL_G0})", flush=True)
sys.exit(1 if n_fail else 0)
