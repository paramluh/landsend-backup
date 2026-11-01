"""gesturemetric: extract strokes + the POSE-OWNER's (B's) word-lane words for GENERATED pose -> out/gen/<name>.npz
(same fields as real_extract.py: st, sv, ws, L, sw, own [(t0, t1, w)], oth; plus lo_s, kind, model, hidden, run, person, wav, sr, n_drop).

TIME ORIGIN (checked 10-26): every source is on ONE clock = conversation code frame k at k x 0.08 s. Pose: genweb workers save
tokens.npz pose = pdep[P+CD : P+CD+n] (worker.py / worker_dup.py), i.e. code frame k of the conversation (the first N =
pose_prefill_frames are the forced actor / real prefill, asserted equal in the workers); decode gives 2 pose frames per token, so pose
frame j is at j x 0.04 s. Words: words.json t0 = lane frame / 12.5 from infer_ppm.lane_words on the B lane indexed from conversation
frame 0 (lane frame f sits at sequence step P+f, its audio / pose at P+CD+f, so lane f and code frame f are the same 80 ms slot - the
training convention); 'prefill' flags agree (prefill words have t0 < N x 0.08 in monoplex runs). b.wav = Mimi decode of the same code
frames from sample 0 (1920 samples / frame), so b.wav time = the same clock. Live sessions: session.npz tok has n+CD frames and pose / mimi
n frames, all indexed from conversation frame 0 (livedup_import.py builds the livedup runs' words.json from tok with the same t = f / 12.5);
session.npz b = the per-frame streaming Mimi decode (1920 samples / frame). No -1 pose frames were found in any source (n_drop saved;
leading drops would shift the stroke clock by n_drop x 0.08 and are corrected; interior drops are refused). Empirical check (main()
prints it): mean b-audio RMS around generated B word onsets rises across t0 (see out/gen/_origin_check.json) - a lane / audio offset of
CD = 2 frames would put the rise at +0.16 s instead.
Sources: (a) genweb/runs/* (monoplex: B pose + B words; duplex / livedup: B pose, B words, A words as 'oth'), (b) live sessions
(sessions/*, stage-lb/*/sessions/*) not already imported as a livedup run, words from session.npz tok via livedup_import.lane_words;
hidden = request.json hidden (runs) or session cfg client == 'test' (automated test client) (sessions).
SESSION PREFILL (prefill_check.py -> out/gen/_prefill_check.json): lo_s = session.json pose_prefill_frames x 0.08 (feed start line as
fallback; the two always agree). 124 sessions record 26: 121 of them start with exactly 26 frames equal to the person's
genweb/openings (116) or GT clip (5) tokens (3 match only 5-22 frames, opening file likely changed since; metadata kept). The 52 early
10-25 sessions (pre "full prefill" server) record nothing and match neither source at any offset -> no forced pose prefill, lo_s = 0.
usage: gen_extract.py [SHARD/NSHARD]"""
import glob
import importlib.util
import json
import os
import sys
import time

H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, H)
import numpy as np  # noqa: E402
import gm  # noqa: E402
import torch  # noqa: E402
import score as S  # noqa: E402

LIVE = "/data/home/amartyavarma/ppm/aloe-posedup-live"
RUNS = os.path.join(LIVE, "genweb", "runs")
OUT = os.path.join(H, "out", "gen")
SR = 24000
_spec = importlib.util.spec_from_file_location("livedup_import", os.path.join(LIVE, "genweb", "livedup_import.py"))
LI = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(LI)


def jload(p, d=None):
    try:
        return json.load(open(p))
    except (OSError, ValueError):
        return d


def person_of(summ, req):
    p = summ.get("person") or (req.get("prefill") or {}).get("person") or (req.get("prompt") or {}).get("voice") or S.DEFAULT_PERSON
    return p if p in os.listdir(S.PROMPTS) else S.DEFAULT_PERSON


def strokes_of(codes_raw, person):
    """-> (st, sv, ws, L, sw, n_drop) on the conversation clock (leading -1 frames shift the clock; interior ones are refused)."""
    codes = S._fix_codes(codes_raw)
    keep = (codes >= 0).all(1)
    n_drop = int((~keep).sum())
    lead = int(np.argmax(keep)) if keep.any() else len(keep)
    assert n_drop == lead, ("interior -1 pose frames", n_drop, lead)
    k55, sw = gm.k55_of(codes, person)
    ws = gm.wrist_speed(k55, sw)
    st, sv = gm.strokes(ws)
    return st + lead * 0.08, sv, ws, (k55.shape[0] / gm.FPS) + lead * 0.08, sw, n_drop


def env_around(wav, onsets, span=0.6, hop=0.01):
    """mean RMS dB envelope (track-max relative) at offsets -span..+span around onsets (for the origin check)."""
    if wav is None or not len(onsets):
        return None, 0
    env = gm.loudness_env(wav, SR, hop)
    env = env - env.max()
    offs = np.arange(-span, span + 1e-9, hop)
    rows = []
    for t in onsets:
        i = np.round((t + offs - 0.025) / hop).astype(int)
        if i[0] >= 0 and i[-1] < len(env):
            rows.append(env[i])
    return (np.mean(rows, 0) if rows else None), len(rows)


def save(name, rec):
    fo = os.path.join(OUT, name + ".npz")
    np.savez(fo + ".tmp.npz", **rec)
    os.replace(fo + ".tmp.npz", fo)


def runs(sh, nsh, origin):
    seen_sessions, n = set(), 0
    for i, r in enumerate(sorted(os.listdir(RUNS))):
        d = os.path.join(RUNS, r)
        summ, req = jload(os.path.join(d, "summary.json"), {}), jload(os.path.join(d, "request.json"), {})
        if req.get("session"):
            seen_sessions.add(req["session"])
        if summ.get("session"):
            seen_sessions.add(summ["session"])
        if i % nsh != sh:
            continue
        fo = os.path.join(OUT, r + ".npz")
        if os.path.exists(fo) or not os.path.exists(os.path.join(d, "tokens.npz")):
            continue
        t0 = time.time()
        try:
            pose = np.load(os.path.join(d, "tokens.npz"))["pose"]
            words = jload(os.path.join(d, "words.json"), []) or []
            person = person_of(summ, req)
            st, sv, ws, L, sw, n_drop = strokes_of(pose, person)
        except Exception as e:  # noqa: BLE001
            print(f"SKIP {r}: {type(e).__name__} {e}", flush=True)
            continue
        own = [(float(w["t0"]), float(w["t1"]), str(w["w"])) for w in words if w.get("ch", "B") == "B"]
        oth = [(float(w["t0"]), float(w["t1"]), str(w["w"])) for w in words if w.get("ch") == "A"]
        N = int(summ.get("pose_prefill_frames") or summ.get("prefill_frames") or 0)
        kind = req.get("model") or ("posefix" if r.startswith("posefix") else "?")
        wavp = os.path.join(d, "b.wav")
        rec = dict(st=st, sv=sv, ws=ws.astype(np.float32), L=L, sw=sw, own=np.array(own, dtype=object), oth=np.array(oth, dtype=object),
                   lo_s=N * 0.08, kind=kind, model=str(summ.get("model", "?")), hidden=bool(req.get("hidden")) or r.startswith("posefix"),
                   run=r, person=person, wav=wavp if os.path.exists(wavp) else "", sr=SR, n_drop=n_drop, src="genweb-run")
        save(r, rec)
        n += 1
        if os.path.exists(wavp) and origin is not None:
            import soundfile as sf
            wv, _ = sf.read(wavp, dtype="float32")
            e, k = env_around(wv if wv.ndim == 1 else wv.mean(1), [w[0] for w in own if w[0] >= N * 0.08])
            if e is not None:
                origin.append((kind, e * k, k))
        print(f"{r} {kind} hidden={rec['hidden']} L={L:.0f}s lo={N * 0.08:.2f}s strokes={len(st)} own={len(own)} oth={len(oth)} "
              f"{time.time() - t0:.1f}s", flush=True)
    return seen_sessions, n


def sessions(sh, nsh, seen, origin):
    ds = sorted(glob.glob(os.path.join(LIVE, "sessions", "*"))) + sorted(glob.glob(os.path.join(LIVE, "stage-lb", "*", "sessions", "*")))
    n, skipped = 0, []
    for i, d in enumerate(ds):
        sid = os.path.basename(d)
        if i % nsh != sh:
            continue
        if sid in seen:
            skipped.append((sid, "imported as a livedup run"))
            continue
        fo = os.path.join(OUT, "sess_" + sid + ".npz")
        if os.path.exists(fo):
            continue
        sj = jload(os.path.join(d, "session.json"), {}) or {}
        if not sj.get("pose_frames"):
            skipped.append((sid, "no pose frames recorded"))
            continue
        t0 = time.time()
        try:
            z = np.load(os.path.join(d, "session.npz"))
            pose = np.asarray(z["pose"]).reshape(-1, 32)
            tok = np.asarray(z["tok"], np.int64)
            b = np.asarray(z["b"], np.float32).reshape(-1)
            mimi = np.asarray(z["mimi"]).reshape(-1, 32).astype(np.int16)
            person = sj.get("person") if sj.get("person") in os.listdir(S.PROMPTS) else S.DEFAULT_PERSON
            st, sv, ws, L, sw, n_drop = strokes_of(pose, person)
        except Exception as e:  # noqa: BLE001
            skipped.append((sid, f"{type(e).__name__} {e}"))
            continue
        bw = LI.lane_words(tok)
        ts = [q["t"] for q in bw]
        own = [(float(ts[j]), float(round(min(ts[j + 1] if j + 1 < len(ts) else ts[j] + 0.4, ts[j] + 1.5), 2)), q["text"])
               for j, q in enumerate(bw)]
        N = sj.get("pose_prefill_frames")
        if N is None:                    # fallback: the feed's start line (always equal to session.json where both exist)
            try:
                N = json.loads(open(os.path.join(d, "feed.jsonl")).readline()).get("pose_prefill_frames")
            except (OSError, ValueError):
                N = None
        N = int(N or 0)
        client = (sj.get("cfg") or {}).get("client")
        rec = dict(st=st, sv=sv, ws=ws.astype(np.float32), L=L, sw=sw, own=np.array(own, dtype=object), oth=np.array([], dtype=object),
                   lo_s=N * 0.08, kind="session", model=str(sj.get("label", "?")), hidden=client == "test", client=str(client), run=sid,
                   person=person, wav=os.path.join(d, "session.npz") + ":b", sr=SR, mimi_b=mimi, n_drop=n_drop, src=d)
        save("sess_" + sid, rec)
        n += 1
        if origin is not None:
            e, k = env_around(b, [w[0] for w in own if w[0] >= N * 0.08])
            if e is not None:
                origin.append(("session", e * k, k))
        print(f"sess_{sid} client={client} L={L:.0f}s lo={N * 0.08:.2f}s strokes={len(st)} own={len(own)} {time.time() - t0:.1f}s",
              flush=True)
    return n, skipped


def main():
    sh, nsh = (map(int, sys.argv[1].split("/")) if len(sys.argv) > 1 else (0, 1))
    torch.set_grad_enabled(False)
    os.makedirs(OUT, exist_ok=True)
    origin = []
    seen, nr = runs(sh, nsh, origin)
    ns, skipped = sessions(sh, nsh, seen, origin)
    offs = np.round(np.arange(-0.6, 0.6 + 1e-9, 0.01), 2)
    res = {}
    for kind in sorted(set(k for k, _, _ in origin)):
        tot = sum(e for k, e, _ in origin if k == kind)
        cnt = sum(c for k, _, c in origin if k == kind)
        m = tot / max(cnt, 1)
        rise = np.diff(m)
        res[kind] = {"words": int(cnt), "env_db": [round(float(x), 2) for x in m], "steepest_rise_s": float(offs[int(np.argmax(rise))]),
                     "db_at_-0.2": round(float(m[40]), 2), "db_at_0": round(float(m[60]), 2), "db_at_+0.2": round(float(m[80]), 2)}
        print("ORIGIN", kind, {k: v for k, v in res[kind].items() if k != "env_db"}, flush=True)
    json.dump({"offsets_s": offs.tolist(), "by_kind": res, "skipped": skipped, "n_runs": nr, "n_sessions": ns},
              open(os.path.join(OUT, f"_origin_check.{sh}of{nsh}.json"), "w"), indent=1)
    for s in skipped:
        print("SKIPPED", *s, flush=True)


if __name__ == "__main__":
    main()
