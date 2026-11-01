#!/usr/bin/env python3
"""livedup (10-26 dupui): import ONE finished live call of instance P (the live demo's own server.py, model "playground") into a playground
run dir, so History / replay / pose stats / the post-call renders treat it like a generated run. Plumbing only: every number is copied from
what P recorded (session.npz / session.json / feed.jsonl = LIVE/server.py's session end), nothing is re-sampled or re-decoded.
  usage: livedup_import.py <session dir> <run dir>          (run by livedup_api.py in a subprocess, CPU only)
Outputs (the workers' file contract):
  a.wav      channel A = the caller's mic exactly as P recorded it (session.npz a, 24 kHz)
  b.wav      channel B = the model's voice exactly as P streamed it (session.npz b = the per-frame streaming Mimi decode)
  mix.wav    a + b on the call's frame clock (as the model heard / spoke; = the duplex worker's mix)
  words.json [{t0, t1, w, ch "B", prefill}] = B's own word lane (session.npz tok), grouped into words exactly as infer_ppm.lane_words;
             no A words (no ASR ever)
  tokens.npz pose / mimi (code frames, local ids, int16) + lane (B's lane tokens) = session.npz pose / mimi / tok
  feed.jsonl the session's own feed (P's feed.jsonl: start + one 'f' per code frame with pose, Mimi codes, the frame's int16 audio) with the
             run id as session, f = code frame k, t = k x 0.08 (the workers' feed form) -> the render queue (Param seq / Vansh)
  summary.json + status.json (done)"""
import base64
import json
import os
CANOPY_ROOT = os.environ.get("CANOPY_ROOT") or os.path.expanduser("~")   # cluster home holding ppm/, venvs/, models/ (see README: paths)
HOME_DIR = os.path.expanduser("~")
import sys
import time

import numpy as np
import soundfile as sf

QWEN32 = os.environ.get("QWEN32_DIR", CANOPY_ROOT + "/models/Qwen3-32B")
TE = 151643                                          # = infer_ppm.TE: lane ids < TE are word pieces
FPS, SPF, CD = 12.5, 1920, 2


def jwrite(path, obj):
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=str)
    os.replace(tmp, path)


def lane_words(lane):
    """= infer_ppm.lane_words (same grouping: a new word = a piece starting with 'Ġ' or following a non-word frame; t = frame / 12.5), with the
    tokenizers library on the same tokenizer.json (Qwen2Tokenizer, clean_up_tokenization_spaces false: decode = the ByteLevel decoder)."""
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(f"{QWEN32}/tokenizer.json")
    words, cur = [], None
    for f in range(len(lane)):
        t = int(lane[f])
        if t < TE:
            sp = (tk.id_to_token(t) or "").startswith("Ġ")
            if cur is None or sp:
                cur = {"pieces": [t], "frame": f}
                words.append(cur)
            else:
                cur["pieces"].append(t)
        else:
            cur = None
    return [{"text": tk.decode(w["pieces"], skip_special_tokens=False).strip(), "frame": int(w["frame"]), "t": round(w["frame"] / FPS, 2)}
            for w in words]


def wav(path, x):
    sf.write(path + ".part.wav", np.clip(np.asarray(x, np.float32), -1, 1), 24000, subtype="PCM_16")
    os.replace(path + ".part.wav", path)


def main():
    sd, rd = sys.argv[1], sys.argv[2]
    rid = os.path.basename(rd.rstrip("/"))
    t0 = time.time()
    # 10-26 aloedemo rule: agent-made calls never show to the user (request "hidden": true). A call the page never registered (empty
    # url_query: the non-preempting 'test' clients) or a page call made by an automated browser (label "agent test") is hidden.
    try:
        rq = json.load(open(f"{rd}/request.json"))
        if not rq.get("hidden") and (not rq.get("url_query") or str(rq.get("label") or "").startswith("agent test")):
            rq.update(hidden=True, hidden_why="agent test call (never registered by a user's page, or an automated browser)")
            json.dump(rq, open(f"{rd}/request.json.tmp", "w"), indent=1)
            os.replace(f"{rd}/request.json.tmp", f"{rd}/request.json")
    except (OSError, ValueError):
        pass
    z = np.load(f"{sd}/session.npz")
    sj = json.load(open(f"{sd}/session.json"))
    a = np.asarray(z["a"], np.float32)
    b = np.asarray(z["b"], np.float32)
    tok = np.asarray(z["tok"], np.int64)
    pose = np.asarray(z["pose"], np.int16).reshape(-1, 32)
    mimi = np.asarray(z["mimi"], np.int16).reshape(-1, 32)
    n = min(len(a), len(b)) if len(a) and len(b) else max(len(a), len(b))
    wav(f"{rd}/a.wav", a)
    wav(f"{rd}/b.wav", b)
    aa, bb = np.zeros(n, np.float32), np.zeros(n, np.float32)
    aa[:min(n, len(a))] = a[:n]
    bb[:min(n, len(b))] = b[:n]
    wav(f"{rd}/mix.wav", aa + bb)
    pf = sj.get("prefill") if isinstance(sj.get("prefill"), dict) else {}
    N = int(pf.get("frames") or 0)                                     # prefill code frames (lane frames 0..N+1 forced)
    pfl = N + CD if N else 0
    bw = lane_words(tok)
    ts = [q["t"] for q in bw]
    wl = [{"t0": ts[i], "t1": round(min(ts[i + 1] if i + 1 < len(ts) else ts[i] + 0.4, ts[i] + 1.5), 2), "w": q["text"], "ch": "B",
           "prefill": bool(q["frame"] < pfl)} for i, q in enumerate(bw)]
    jwrite(f"{rd}/words.json", wl)
    np.savez_compressed(f"{rd}/tokens.npz", pose=pose, mimi=mimi, lane=tok)
    # the render feed: P's own feed for this call (its start + every code frame), renamed to the run, workers' f / t form
    start, frames = None, []
    with open(f"{sd}/feed.jsonl") as fi:
        for ln in fi:
            try:
                m = json.loads(ln)
            except ValueError:
                continue
            if m.get("type") == "start" and start is None:
                start = m
            elif m.get("type") == "f":
                frames.append(m)
    frames.sort(key=lambda m: int(m.get("f", 0)))
    person = sj.get("person") if sj.get("person") not in (None, "None", "") else (start or {}).get("person")
    st = dict(start or {"type": "start", "fps": 12.5, "prefill_s": 0.0})
    st.update(type="start", session=rid)
    st.pop("record", None)
    if person:
        st["person"] = person
    with open(f"{rd}/feed.jsonl.tmp", "w") as fo:
        fo.write(json.dumps(st) + "\n")
        for k, m in enumerate(frames):
            assert int(m.get("f", k)) == k, ("feed frame gap", k, m.get("f"))
            fo.write(json.dumps({"type": "f", "session": rid, "f": k, "t": round(k * 0.08, 3), "pose": m["pose"], "mimi": m["mimi"],
                                 "pcm_b64": m["pcm_b64"]}) + "\n")
        fo.write(json.dumps({"type": "end", "session": rid}) + "\n")
    os.replace(f"{rd}/feed.jsonl.tmp", f"{rd}/feed.jsonl")
    if len(frames) != len(pose):
        print(f"[livedup-import] {rid}: feed has {len(frames)} frames, session.npz pose {len(pose)}", flush=True)
    gen = pose[N:, 0] if len(pose) > N else pose[:0, 0]
    chg = float((gen[1:] != gen[:-1]).mean()) if len(gen) > 1 else None
    nbg = sum(1 for q in wl if not q["prefill"])
    lat = sj.get("latency_ms") if isinstance(sj.get("latency_ms"), dict) else {}
    secs = round(len(pose) / FPS, 2)
    # POSE CFG ACTUALLY APPLIED (posefix 10-26: per-frame 'pcg' = pose CFG applied after the speaking-only gate; session.json cfg_guided_frames /
    # cfg_frames_total). Never from 'pcf' (that only says pose CFG is switched on). Older engines: pending.
    pcg_pct = None
    if sj.get("cfg_guided_frames") is not None and sj.get("cfg_frames_total"):
        pcg_pct = round(100.0 * float(sj["cfg_guided_frames"]) / float(sj["cfg_frames_total"]), 1)
    elif "pcg" in z.files and len(z["pcg"]):
        pcg_pct = round(100.0 * float(np.asarray(z["pcg"], np.float32).mean()), 1)
    if not (sj.get("params") or {}).get("pose_cfg_on"):
        pcg_txt = "pose CFG off"
    elif pcg_pct is not None:
        pcg_txt = f"pose CFG active on {pcg_pct:g}% of frames (speaking-only)" if (sj.get("params") or {}).get("pose_cfg_speaking_only") else f"pose CFG active on {pcg_pct:g}% of frames"
    else:
        pcg_txt = "pose CFG active: pending (engine update)"
    summ = {"run_id": rid, "model": f"{sj.get('label') or 'pose duplex'} (step {sj.get('step')}) LIVE call (instance P)", "dup_mode": "live",
            "session": sj.get("session"), "session_dir": sd, "ck": sj.get("ck"), "step": sj.get("step"), "seed": sj.get("seed"),
            "person": person, "voice": sj.get("voice"), "frames": len(pose), "code_frames": len(pose), "call_s": secs,
            "pose_prefill_frames": sj.get("pose_prefill_frames"), "prefill_frames": N, "prefill": pf, "params": sj.get("params"),
            "frame0_global_state": sj.get("frame0_global_state"),   # 10-26 posefix: the opening's frame 0 (the feed start copied below carries it too)
            "param_log": sj.get("param_log"), "cfg": sj.get("cfg"), "prompt": sj.get("prompt"), "latency_ms": lat, "end_reason": sj.get("end_reason"),
            "feed": sj.get("feed"), "gen_cb0_change": chg, "gen_cb0_distinct": int(len(np.unique(gen))) if len(gen) else 0,
            "n_words_b": len(wl), "n_words_b_generated": nbg, "words_b": " ".join(q["w"] for q in wl)[:600],
            "pose_cfg_active_pct": pcg_pct, "pose_cfg_active": pcg_txt, "cfg_guided_frames": sj.get("cfg_guided_frames"),
            "cfg_frames_total": sj.get("cfg_frames_total"), "a_seconds": round(len(a) / 24000, 2), "b_seconds": round(len(b) / 24000, 2), "imported": time.time(), "import_s": round(time.time() - t0, 1)}
    jwrite(f"{rd}/summary.json", summ)
    head = (f"live call {secs:.0f} s · B {nbg} words · pose cb0 changes on {100 * chg:.0f}% of generated frames · "
            f"{pcg_txt} · {lat.get('p50', '?')} ms/frame p50" if chg is not None else f"live call {secs:.0f} s · B {nbg} words · {pcg_txt}")
    jwrite(f"{rd}/status.json", {"state": "done", "progress": 1.0, "msg": "call recorded; avatar rendering", "headline": head, "worker": "livedup",
                                 "model": "livedup", "t_end": time.time(), "seed": sj.get("seed"), "frames_done": len(pose), "frames_total": len(pose),
                                 "ms_per_frame": lat.get("mean")})
    print(f"[livedup-import] {rid}: {head} ({summ['import_s']} s)", flush=True)


if __name__ == "__main__":
    main()
