"""CPU unit test of the 10-22 FRESH POSE APPEND (PPM_FRESH_POSE_ROOT2 / _TAIL; fresh_pose.append_rank_rows; train_ppm.fresh_append_guard +
s811_init). No GPU, no data reads beyond the plan-1 audit copies (fresh/scan/plans/PLAN-<host>.json: host row counts) and one saved
train_state.json. Run on a compute node (imports torch via train_ppm):
  python fresh/append/test_append_cpu.py map            step -> row mapping (real plan-1 row counts, synthetic plan 2), resume at 17352
  python fresh/append/test_append_cpu.py guard <env> <train_state.json> <mode>   one trainer import under <env> (mode: unset | append)
  python fresh/append/test_append_cpu.py dedup          appended-plan dedup against earlier plans (synthetic scans)
  python fresh/append/test_append_cpu.py all            map + dedup + both guard modes (subprocesses)
  python fresh/append/test_append_cpu.py map3           10-23 delivery 3: plans [1, npl2, npl3] (real audit row counts; npl3 synthetic until its
                                                         audit copies exist), resume at 17641 (job 27204's final save)
  python fresh/append/test_append_cpu.py all3 <dcp-17641 train_state.json> <append3 env>   all (27114 stand-in) + map3 + guard[append3]
FP_DIR (fresh_pose.py dir) and TRAIN_PY (the trainer file) default to the installed ones."""
import glob
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

C = "/data/home/amartyavarma/ppm/code-811"
A = f"{C}/fresh/append"
FP_DIR = os.environ.get("FP_DIR", f"{C}/fresh")
TRAIN_PY = os.environ.get("TRAIN_PY", f"{C}/train_ppm.py")
SF, NPM, LW, END1 = 16861, 8, 8, 17352
END2 = 17641                                                      # 10-23: job 27204's final save = the end over [plan 1, npl2]
NPL2 = "/scratch/amartyavarma/ppm-fresh-pose-npl2"
PY = "/data/home/amartyavarma/venvs/qwen-s2/bin/python"


def fp():
    sp_ = list(sys.path); sys.path.insert(0, FP_DIR)
    try:
        import fresh_pose as FP_  # noqa: E402
    finally:
        sys.path[:] = sp_
    return FP_


def test_map():
    FP = fp()
    n1 = {json.load(open(p))["host"]: int(json.load(open(p))["n_rows"]) for p in sorted(glob.glob(f"{C}/fresh/scan/plans/PLAN-*.json"))}
    assert len(n1) == 16, n1
    rng = np.random.default_rng(7)
    n2 = {h: int(rng.integers(25000, 36000)) for h in n1}            # synthetic plan-2 / plan-3 host row counts (uneven on purpose)
    n3 = {h: int(rng.integers(3000, 9000)) for h in n1}
    ok = 0
    # 27114's mapping: rank stream = rank_rows(n1); its end
    old = {(h, l): FP.rank_rows(n1[h], l, LW) for h in n1 for l in range(LW)}
    end1 = SF + min(len(v) // NPM for v in old.values())
    assert end1 == END1, ("plan-1 end from the audit row counts", end1)
    ok += 1
    for tail in (1, 0):
        ends, new = FP.append_ends({h: [n1[h], n2[h]] for h in n1}, LW, NPM, SF, tail)
        end = ends[-1]
        assert ends[0] == END1, ends
        for (h, l), s in new.items():
            assert np.array_equal(s, FP.append_stream([n1[h], n2[h]], l, LW, NPM, SF, tail, ends)), "append_ends != the trainer's per-rank call"
            o = old[(h, l)]
            # (1) history: every step < END1 reads the same plan-1 row as job 27114 (step t micro m = stream[(t - SF) x 8 + m])
            assert np.array_equal(s[:(END1 - SF) * NPM], o[:(END1 - SF) * NPM]), (h, l, "history")
            # (2) resume at 17352 == the uninterrupted stream from 17352 on (the trainer's pstream slice); and at a later step (17400)
            full = s[:(end - SF) * NPM]
            for st in (END1, END1 + 48):
                ps = s[(st - SF) * NPM:(end - SF) * NPM]
                assert len(ps) == (end - st) * NPM and np.array_equal(ps, full[(st - SF) * NPM:]), (h, l, st)
            # (3) ids decode to (plan, row) inside range; plan-1 ids are plain (= rank_rows)
            dec = np.array([FP.split_row(x) for x in full])
            assert set(dec[:, 0]) <= {0, 1} and (dec[dec[:, 0] == 0, 1] < n1[h]).all() and (dec[dec[:, 0] == 1, 1] < n2[h]).all()
            assert all(FP.split_row(x) == (0, int(x)) for x in s[:16])
            # (4) where plan 2 starts for this rank: TAIL=0 exactly at END1 for every rank; TAIL=1 after ALL its plan-1 rows
            i2 = int(np.flatnonzero(dec[:, 0] == 1)[0]) if (dec[:, 0] == 1).any() else len(dec)
            assert i2 == ((END1 - SF) * NPM if not tail else len(o)), (h, l, tail, i2, len(o))
        # (5) never repeated: per host, the rows its 8 ranks read over the WHOLE fresh run (27114 + the append) are distinct
        for h in n1:
            u = np.concatenate([new[(h, l)][:(end - SF) * NPM] for l in range(LW)])
            assert len(np.unique(u)) == len(u), (h, "repeat")
            if tail:                                                 # TAIL=1: plan-1 rows unused by the whole run = 0 when plan 2 is long
                p1 = u[(u >> FP.PLAN_SHIFT) == 0]
                assert len(p1) == n1[h], (h, "every plan-1 row used once with tail", len(p1), n1[h])
        # (6) a THIRD plan appended later (at a checkpoint <= the plans-1..2 end): every step before that end unchanged
        ends3, new3 = FP.append_ends({h: [n1[h], n2[h], n3[h]] for h in n1}, LW, NPM, SF, tail)
        assert ends3[:2] == ends and ends3[2] > ends[1], (ends3, ends)
        for k_, s3 in new3.items():
            assert np.array_equal(s3[:(ends[1] - SF) * NPM], new[k_][:(ends[1] - SF) * NPM]), (k_, "plan-3 append changed history")
            d3 = s3[:(ends3[2] - SF) * NPM]
            assert len(np.unique(d3)) == len(d3)
        print(f"[map] tail {tail}: ends {ends} (+ plan 3 synthetic: {ends3}); plan-2 rows {sum(n2.values()):,} synthetic; unused plan-1 rows "
              f"{sum(len(old[k]) for k in old) - 128 * (END1 - SF) * NPM if not tail else 0:,} dropped", flush=True)
        ok += 1
    # (7) split_row round trip
    for k, r in ((0, 0), (1, 0), (1, 33432), (3, (1 << 40) - 1)):
        x = (k << FP.PLAN_SHIFT) | r
        assert FP.split_row(x) == (k, r) and FP.split_row(np.int64(x)) == (k, r)
    ok += 1
    # (8) the multi-plan dataset refuses plans with the same row-key prefix
    try:
        FP.FreshPoseMultiDS([{"n_rows": 1}, {"n_rows": 1}], [0], {}, 0); raise SystemExit("FreshPoseMultiDS accepted duplicate key prefixes")
    except AssertionError:
        ok += 1
    print(f"test_append_cpu map: {ok}/{ok} OK", flush=True)


def test_dedup():
    """appended-plan dedup against earlier plans (fresh_pose.prior_index / prior_drop) on synthetic scans: segment id, clip id, overlapping /
    disjoint / touching source spans of an earlier video, unparseable clip ids, modes overlap / video / off."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    FP = fp()
    d = tempfile.mkdtemp(prefix="deduptest-", dir=f"{A}/testdata")
    v1, v2, v3 = "7000000000000000001", "7000000000000000002", "7000000000000000003"
    prior = {"segment_id": [f"{v1}_00-10.00_to_00-20.00__seg0000", f"{v1}_00-30.00_to_00-40.00__seg0000", f"{v2}_00-00.00_to_00-05.00__seg0000",
                            f"{v3}_weird__seg0000"],
             "clip_id": [f"{v1}_00-10.00_to_00-20.00", f"{v1}_00-30.00_to_00-40.00", f"{v2}_00-00.00_to_00-05.00", f"{v3}_weird"],
             "video_id": [v1, v1, v2, v3]}
    pq.write_table(pa.table(prior), f"{d}/slinky-0.parquet")
    new = {"segment_id": [f"{v1}_00-10.00_to_00-20.00__seg0000",      # 1 same segment
                          f"{v1}_00-10.00_to_00-20.00__seg0001",      # 2 same clip, another segment id
                          f"{v1}_00-15.00_to_00-25.00__seg0000",      # 3 overlaps [10, 20]
                          f"{v1}_00-20.00_to_00-30.00__seg0000",      # touches both ends -> new span (kept)
                          f"{v1}_00-45.00_to_01-05.00__seg0000",      # after [30, 40] -> kept
                          f"{v1}_00-35.00_to_00-36.00__seg0000",      # inside [30, 40] -> 3
                          f"{v1}_bad__seg0000",                       # unparseable, earlier video -> 5
                          f"{v3}_00-00.00_to_00-01.00__seg0000",      # earlier clip of v3 unparseable -> whole video -> 5
                          "7999_00-00.00_to_00-09.00__seg0000"],      # unrelated video -> kept
           "clip_id": [f"{v1}_00-10.00_to_00-20.00", f"{v1}_00-10.00_to_00-20.00", f"{v1}_00-15.00_to_00-25.00", f"{v1}_00-20.00_to_00-30.00",
                       f"{v1}_00-45.00_to_01-05.00", f"{v1}_00-35.00_to_00-36.00", f"{v1}_bad", f"{v3}_00-00.00_to_00-01.00",
                       "7999_00-00.00_to_00-09.00"],
           "video_id": [v1, v1, v1, v1, v1, v1, v1, v3, "7999"]}
    FP.PRIOR_SCANS = [d]
    pri = FP.prior_index(FP.h64s(new["video_id"]))
    want = {"overlap": [1, 2, 3, 0, 0, 3, 5, 5, 0], "video": [1, 2, 4, 4, 4, 4, 4, 4, 0], "off": [1, 2, 0, 0, 0, 0, 0, 0, 0]}
    for mode, w in want.items():
        FP.PRIOR_VIDEO = mode
        r, nsp = FP.prior_drop(pri, new)
        assert list(r) == w, (mode, list(r), w)
        if mode == "overlap":
            assert list(np.flatnonzero(nsp)) == [3, 4], list(nsp)
    assert FP.clip_span(f"{v1}_01-05.50_to_02-00.00", v1) == (65.5, 120.0) and FP.clip_span(f"{v1}_00-05.00_to_00-05.00", v1) is None
    assert FP.clip_span(f"{v2}_00-01.00_to_00-02.00", v1) is None
    import shutil
    shutil.rmtree(d)
    print("test_append_cpu dedup: 5/5 OK", flush=True)


def env_of(path, extra):
    e = {k: v for k, v in os.environ.items() if not k.startswith("PPM_")}
    for ln in open(path):
        ln = ln.strip()
        if ln and not ln.startswith("#") and "=" in ln:
            k, v = ln.split("=", 1); e[k] = v
    e.update(extra)
    e.update(RANK="0", WORLD_SIZE="128", LOCAL_RANK="0", LOCAL_WORLD_SIZE="8", PPM_RUN="ppm-v1-jTEST", PPM_RUN_BASE="ppm-v1",
             HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    return e


def test_guard(mode, ts_path):
    """one trainer import (env already set by the parent): fresh_append_guard cases + s811_init on stand-in checkpoints."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("train_ppm", TRAIN_PY)
    T = importlib.util.module_from_spec(spec); sys.modules["train_ppm"] = T; spec.loader.exec_module(T)
    ts = json.load(open(ts_path))
    fo = ts["cfg"]["fresh"]
    assert "pose_root2" not in fo and "pose_digests" not in fo, fo
    ok = 0
    tmp = tempfile.mkdtemp(prefix="appendtest-", dir=f"{A}/testdata")

    def ckpt(cfg_fresh, step):
        d = f"{tmp}/dcp-{step}-{len(os.listdir(tmp))}"; os.makedirs(d)
        c = json.loads(json.dumps(ts)); c["step"] = step; c["cfg"]["fresh"] = cfg_fresh
        json.dump(c, open(f"{d}/train_state.json", "w")); return d

    def s811(d):
        T.CFG["s811"]["from_step"] = T.CFG["s811"]["prev_ga_seq"] = None
        for k in T.FRESH_APPEND_LATE:
            T.CFG.get("fresh", {}).pop(k, None)
        T.FRESH_SAVED_DIG[0] = None
        T.s811_init(d)
        return dict(T.CFG["fresh"])

    def raises(f):
        try:
            f()
        except AssertionError:
            return True
        return False

    if mode == "unset":                                              # v4 env: byte-identical cfg, the old guard path
        assert "pose_root2" not in T.CFG["fresh"] and "pose_tail" not in T.CFG["fresh"] and T.FRESH_POSE2 is None, T.CFG["fresh"]
        assert T.CFG["fresh"] == fo, ("v4 env fresh block != 27114's", T.CFG["fresh"], fo)
        f_ = s811(ckpt(fo, END1)); assert f_ == fo and T.CFG["s811"]["from_step"] == SF, f_; ok += 1            # resume 27114's final save
        assert raises(lambda: s811(ckpt(dict(fo, pose_from=SF + 1), END1))); ok += 1                         # old guard still strict
        later = dict(fo, pose_root2=["/scratch/x"], pose_tail=1, pose_append_at=[END1], pose_digests={"slinky-0": ["a", "b"]})
        assert raises(lambda: s811(ckpt(later, END1 + 54))); ok += 1                                           # cannot drop appended plans
    else:                                                            # append env
        fn = T.CFG["fresh"]
        R2 = list(fn["pose_root2"])
        assert R2 and fn["pose_tail"] in (0, 1) and {k: v for k, v in fn.items() if k not in T.FRESH_APPEND_KEYS} == fo, fn
        f_ = s811(ckpt(fo, END1)); assert f_["pose_append_at"] == [END1] and T.FRESH_SAVED_DIG[0] is None, f_; ok += 1  # 27114's final save
        assert T.fresh_append_guard(fo, dict(fn), END1) == ([END1], None); ok += 1
        assert T.fresh_append_guard(None, dict(fn), SF) == ([SF], None); ok += 1                                    # from PPM's end (no block)
        dg = {"slinky-0": ["f6390f6785e18ef24e2d48a93c0143de", "0" * 32]}
        later = dict(fn, pose_append_at=[END1], pose_digests=dg)
        f_ = s811(ckpt(later, END1 + 54)); assert f_["pose_append_at"] == [END1] and T.FRESH_SAVED_DIG[0] == dg; ok += 1  # a later save
        assert raises(lambda: s811(ckpt(dict(later, pose_tail=1 - fn["pose_tail"]), END1 + 54))); ok += 1          # tail changed
        assert raises(lambda: s811(ckpt(dict(later, pose_root2=["/x"] + R2), END1 + 54))); ok += 1                # not a prefix
        assert raises(lambda: s811(ckpt(dict(fo, speech_flip_from=SF + 1), END1))); ok += 1                         # other key changed
        assert raises(lambda: s811(ckpt({k: v for k, v in later.items() if k != "pose_digests"}, END1 + 54))); ok += 1  # digests missing
        # a third plan appended at a later checkpoint: the checkpoint's list is a prefix -> the new plan is added at that step
        assert T.fresh_append_guard(dict(later), dict(fn, pose_root2=R2 + ["/scratch/p3"]), END1 + 300) == ([END1, END1 + 300], dg); ok += 1
        # the main() check: added at a checkpoint past the plan-1 end is refused (append_at <= FRESH_END1), shown on the stand-in:
        f_ = s811(ckpt(fo, END1 + 10)); assert f_["pose_append_at"] == [END1 + 10]; ok += 1   # (main() asserts ap_[0] <= FRESH_END1)
    import shutil
    shutil.rmtree(tmp)
    print(f"test_append_cpu guard[{mode}]: {ok}/{ok} OK", flush=True)


def audit_rows(tag):
    d = f"{C}/fresh/scan/plans" if tag == "npl" else f"{A}/{tag}/scan/plans"
    return {json.load(open(p))["host"]: int(json.load(open(p))["n_rows"]) for p in sorted(glob.glob(f"{d}/PLAN-*.json"))}


def test_map3():
    """10-23 delivery 3: the stream over [plan 1, npl2, npl3] with TAIL=1 (the env's) and TAIL=0 (coverage), resumed from dcp-17641."""
    FP = fp()
    n1, n2 = audit_rows("npl"), audit_rows("npl2")
    assert len(n1) == 16 and set(n2) == set(n1), (sorted(n1), sorted(n2))
    n3 = audit_rows("npl3")
    real3 = len(n3) == 16
    if not real3:                                                    # (before the npl3 plan exists: synthetic, uneven on purpose)
        rng = np.random.default_rng(11)
        n3 = {h: int(rng.integers(20000, 30000)) for h in n1}
    ok = 0
    for tail in (1, 0):
        ends2, s2 = FP.append_ends({h: [n1[h], n2[h]] for h in n1}, LW, NPM, SF, tail)
        if tail:
            assert ends2 == [END1, END2], ("the [plan 1, npl2] ends from the audit copies != job 27204's", ends2)
        ends3, s3 = FP.append_ends({h: [n1[h], n2[h], n3[h]] for h in n1}, LW, NPM, SF, tail)
        assert ends3[:2] == ends2 and ends3[2] > ends2[1], (ends3, ends2)
        E2, end = ends2[1], ends3[2]
        for (h, l), s in s3.items():
            assert np.array_equal(s, FP.append_stream([n1[h], n2[h], n3[h]], l, LW, NPM, SF, tail, ends3[:2])), "append_ends != the trainer's call"
            # (1) history: every step < 17641 reads exactly the row of the [plan 1, npl2] run (job 27204), hence of 27114 before 17352
            assert np.array_equal(s[:(E2 - SF) * NPM], s2[(h, l)][:(E2 - SF) * NPM]), (h, l, "history")
            # (2) resume at 17641 (and at 17641 + 48) == the uninterrupted stream
            full = s[:(end - SF) * NPM]
            for st in (E2, E2 + 48):
                ps = s[(st - SF) * NPM:(end - SF) * NPM]
                assert len(ps) == (end - st) * NPM and np.array_equal(ps, full[(st - SF) * NPM:]), (h, l, st)
            dec = np.array([FP.split_row(x) for x in full])
            # (3) ids in range, plan order 0 <= 1 <= 2 along the stream (TAIL=1: the rank's npl2 leftovers before any npl3 row)
            assert (np.diff(dec[:, 0]) >= 0).all(), (h, l, "plans out of order")
            for k_, n_ in ((0, n1[h]), (1, n2[h]), (2, n3[h])):
                assert (dec[dec[:, 0] == k_, 1] < n_).all(), (h, l, k_)
            # (4) npl3 rows: this rank's rows l, l + 8, .. in order, a prefix (each at most once, in order)
            r3 = dec[dec[:, 0] == 2, 1]
            assert np.array_equal(r3, FP.rank_rows(n3[h], l, LW)[:len(r3)]), (h, l, "npl3 rows not the rank's prefix in order")
            if tail:                                                 # TAIL=1: npl3 starts right after ALL the rank's plan-1 + npl2 rows
                i3 = int(np.flatnonzero(dec[:, 0] == 2)[0]) if len(r3) else len(dec)
                assert i3 == len(FP.rank_rows(n1[h], l, LW)) + len(FP.rank_rows(n2[h], l, LW)), (h, l, "npl2 leftovers not first")
            else:                                                    # TAIL=0: npl3 starts at 17641 on every rank
                assert int(np.flatnonzero(dec[:, 0] == 2)[0]) == (E2 - SF) * NPM, (h, l)
        # (5) never repeated per host over the WHOLE fresh run (27114 + 27204 + the npl3 leg); TAIL=1: every npl2 row used once
        for h in n1:
            u = np.concatenate([s3[(h, l)][:(end - SF) * NPM] for l in range(LW)])
            assert len(np.unique(u)) == len(u), (h, "repeat")
            if tail:
                assert int(((u >> FP.PLAN_SHIFT) == 1).sum()) == n2[h], (h, "every npl2 row used once with tail")
        print(f"[map3] tail {tail}: ends {ends3} ({'real' if real3 else 'synthetic'} npl3 rows {sum(n3.values()):,}); resume {E2} exact; "
              f"npl2 leftovers first; npl3 rows in order, once", flush=True)
        ok += 1
    print(f"test_append_cpu map3: {ok}/{ok} OK", flush=True)


def test_guard3(ts_path):
    """one trainer import under the append3 env (PPM_FRESH_POSE_ROOT2 = npl2,npl3): the real dcp-17641 train_state (fresh block with
    pose_root2 [npl2], pose_append_at [17352], pose_digests) -> npl3 added at 17641; refusals."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("train_ppm", TRAIN_PY)
    T = importlib.util.module_from_spec(spec); sys.modules["train_ppm"] = T; spec.loader.exec_module(T)
    ts = json.load(open(ts_path))
    fo = ts["cfg"]["fresh"]
    assert int(ts["step"]) == END2 and fo["pose_root2"] == [NPL2] and fo["pose_append_at"] == [END1] and len(fo["pose_digests"]) == 16, \
        ("not job 27204's final save", ts["step"], fo.get("pose_root2"), fo.get("pose_append_at"))
    fn = T.CFG["fresh"]
    R2 = list(fn["pose_root2"])
    assert len(R2) == 2 and R2[0] == NPL2 and fn["pose_tail"] == fo["pose_tail"] == 1, fn
    assert {k: v for k, v in fn.items() if k not in T.FRESH_APPEND_KEYS + T.FRESH_APPEND_LATE} == \
        {k: v for k, v in fo.items() if k not in T.FRESH_APPEND_KEYS + T.FRESH_APPEND_LATE}, ("other fresh keys differ", fn, fo)
    ok = 0
    tmp = tempfile.mkdtemp(prefix="append3test-", dir=f"{A}/testdata")

    def ckpt(cfg_fresh, step):
        d = f"{tmp}/dcp-{step}-{len(os.listdir(tmp))}"; os.makedirs(d)
        c = json.loads(json.dumps(ts)); c["step"] = step; c["cfg"]["fresh"] = cfg_fresh
        json.dump(c, open(f"{d}/train_state.json", "w")); return d

    def s811(d):
        T.CFG["s811"]["from_step"] = T.CFG["s811"]["prev_ga_seq"] = None
        for k in T.FRESH_APPEND_LATE:
            T.CFG.get("fresh", {}).pop(k, None)
        T.FRESH_SAVED_DIG[0] = None
        T.s811_init(d)
        return dict(T.CFG["fresh"])

    def raises(f):
        try:
            f()
        except AssertionError:
            return True
        return False

    dg = fo["pose_digests"]
    f_ = s811(ckpt(fo, END2))                                        # (1) the real resume: npl3 added at 17641, saved digests carried
    assert f_["pose_append_at"] == [END1, END2] and T.FRESH_SAVED_DIG[0] == dg and T.CFG["s811"]["from_step"] == SF, f_; ok += 1
    assert T.fresh_append_guard(fo, dict(fn), END2) == ([END1, END2], dg); ok += 1
    assert raises(lambda: s811(ckpt(dict(fo, pose_tail=0), END2))); ok += 1                                       # tail changed
    assert raises(lambda: T.fresh_append_guard(fo, dict(fn, pose_root2=list(reversed(R2))), END2)); ok += 1     # reordered: not a prefix
    assert raises(lambda: T.fresh_append_guard(fo, dict(fn, pose_root2=[R2[1]]), END2)); ok += 1                 # npl2 dropped
    assert raises(lambda: s811(ckpt(dict(fo, speech_flip_from=SF + 1), END2))); ok += 1                           # another fresh key changed
    assert raises(lambda: s811(ckpt({k: v for k, v in fo.items() if k != "pose_digests"}, END2))); ok += 1        # digests missing
    assert raises(lambda: s811(ckpt(dict(fo, pose_append_at=[END1, END2]), END2))); ok += 1                       # append_at / list length mismatch
    # (2) a later save of the npl3 leg (3 plans recorded) resumes under the same env with the same append steps
    dg3 = {h: list(v) + ["0" * 32] for h, v in dg.items()}
    later = dict(fo, pose_root2=R2, pose_append_at=[END1, END2], pose_digests=dg3)
    f_ = s811(ckpt(later, END2 + 60)); assert f_["pose_append_at"] == [END1, END2] and T.FRESH_SAVED_DIG[0] == dg3, f_; ok += 1
    # (3) ... and the 27204 env (npl2 only) can no longer resume it (dropping an appended plan is refused)
    assert raises(lambda: T.fresh_append_guard(later, dict(fn, pose_root2=[NPL2]), END2 + 60)); ok += 1
    # (4) the main() boundary check (append step <= the end over the plans before it): 17641 <= 17641 passes, 17642 would not
    assert END2 <= END2 and not (END2 + 1 <= END2); ok += 1
    import shutil
    shutil.rmtree(tmp)
    print(f"test_append_cpu guard[append3]: {ok}/{ok} OK", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd in ("map", "all"):
        test_map()
    if cmd in ("dedup", "all"):
        test_dedup()
    if cmd == "guard":
        test_guard(sys.argv[4], sys.argv[3])
    if cmd == "guard3":
        test_guard3(sys.argv[3])
    if cmd in ("map3", "all3"):
        test_map3()
    if cmd == "all3":                                               # all (27114 stand-in, 2-plan env) + map3 (above) + guard[append3]
        for args, env_, extra in (([os.path.abspath(__file__), "all"], None, None),
                                  ([os.path.abspath(__file__), "guard3", sys.argv[3], sys.argv[2]], sys.argv[3], {"PPM_MAX_STEPS": "18999"})):
            r = subprocess.run([PY] + args, env=(env_of(env_, extra) if env_ else None), capture_output=True, text=True)
            print(r.stdout[-4000:], r.stderr[-3000:] if r.returncode else "", flush=True)
            assert r.returncode == 0, (args[1], r.returncode)
        print("test_append_cpu all3: OK")
    if cmd == "all":
        ts_ = sys.argv[2] if len(sys.argv) > 2 else f"{A}/testdata/train_state_dcp17281_slinky0.json"
        app_env = sys.argv[3] if len(sys.argv) > 3 else f"{C}/prod_811_fresh_append.env"
        for env_, mode, extra in ((f"{C}/prod_811_fresh16861_v4.env", "unset", {}),
                                  (app_env, "append", {"PPM_MAX_STEPS": "17999"})):
            r = subprocess.run([PY, os.path.abspath(__file__), "guard", env_, ts_, mode], env=env_of(env_, extra), capture_output=True, text=True)
            print(r.stdout[-3000:], r.stderr[-3000:] if r.returncode else "", flush=True)
            assert r.returncode == 0, (mode, r.returncode)
        print("test_append_cpu all: OK")
