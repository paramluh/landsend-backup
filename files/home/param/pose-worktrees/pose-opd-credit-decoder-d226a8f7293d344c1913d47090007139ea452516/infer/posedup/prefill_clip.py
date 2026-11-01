"""10-27 user (via main): "add a toggle called prefill which determines if to prefill the assistant side and a way to supply an amount and it
always prefills from the same clip selected as the voice". SHARED by the pose demo (server.py) and the speech twin (aloe-m1vox-live
twin_hook.py) so both prefill identically.

prefill_from_clip(voice_id, pf_s, openings_root, want_pose, LPmod, tok, clip_codes=None, want_words=True)
  -> (codes int64 [32, N] | None, lane int64 [N + LPmod.CD] | None, pose int64 [N, 32] | None, info dict)

The window is ALWAYS the first N frames of the SAME clip B gets as its voice prompt (never the conversation opening):
  - SAM actor (openings_root != genweb/openings and its package there is ready = meta.json; the whole package from that root, never mixed):
      codes = <root>/<id>/voice_clip.safetensors (the voice clip itself), words = clipwin_words.json, pose = clipwin_pose_tokens.safetensors
      (samsep: byte slices of the training store / POSE_INDEX at the clip's conversation frames; ready = clipwin_meta.json). The pose tokens
      are normalised by the CONVERSATION's frame 0, so info["frame0_global_state"] = clipwin_meta.json's frame0 fields [root_y, root_x, width].
  - video person (demo-data/prompts/<id>/gt): codes = gt/mimi_local[:, :125] (= the voice clip add_video_people builds), pose = gt/pose_tokens,
      words = genweb/words/<id>/words.json (the avatar clip's timings; window starts at 0). No frame0 field (the package's own frame 0).
  - bank voice: codes = clip_codes (the preset's clip codes as the server holds them); no words (PAD lane), no pose.
10-27 user ("can we pick a prefill clip the conversation starts with"): clip_src = "start" | "rule" (default "rule" here; the servers
default to "start"). "start" = the voice clip IS the first 10 s (125 frames) of that voice's conversation OPENING package (whole package
from one root: openings_root when its package is ready, else genweb/openings, = server.py opening_root), with that window's words.json /
pose_tokens and frame0 = meta.json's (server.py opening_frame0_gs); unavailable (None) when the voice has no opening package. "rule" =
the training-rule clip above. clip_for() gives the voice clip itself; prefill_from_clip() prefills the first N frames of the same clip.
pf_s snaps like the servers' prefill_s: pf = round(round(pf_s / 0.16) * 0.16, 2); N = min(clip frames, pose frames when the clip has pose,
2 * round(pf / 0.16) + 2) (2.0 s -> 1.92 -> 26 frames). N does not depend on want_pose (both demos get the same window). pf <= 0 = no prefill.
The lane is the training placement of the timed words over frames 0..N+CD-1, built exactly as server.py prefill_inputs (word_pieces ->
lane over N + CD + 25 frames -> insert_new_word -> first N + CD). info["sha256_codes"] = sha256 of the int64 [32, N] codes (C order)."""
import hashlib
import json
import os

import numpy as np

GENWEB = "/data/home/amartyavarma/ppm/aloe-posedup-live/genweb"
OPENINGS_DEFAULT = f"{GENWEB}/openings"
VIDEO_PEOPLE = os.environ.get("PPMLIVE_VIDEO_PEOPLE", "/data/home/amartyavarma/video/demo-data/prompts")
PKG_FILES = ("mimi_local.safetensors", "pose_tokens.safetensors", "meta.json")
CLIP_MAX = 125                                   # training clip_max (frames) = the video-person voice clip length


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a, np.int64)).tobytes()).hexdigest()


def snap(pf_s):
    try:
        pf_s = float(pf_s or 0.0)
    except (TypeError, ValueError):
        pf_s = 0.0
    return round(round(pf_s / 0.16) * 0.16, 2)


def n_rule(pf):
    return 2 * int(round(pf / 0.16)) + 2 if pf > 0 else 0


def _pkg_ready(root, vid):
    return bool(vid) and all(os.path.isfile(f"{root}/{vid}/{f}") for f in PKG_FILES)


def _load(f):
    from safetensors.numpy import load_file
    return np.asarray(load_file(f)["codes"], np.int64)


def opening_root(voice_id, openings_root):
    """the root serving voice_id's whole opening package (server.py opening_root) | None when it has none."""
    root = os.path.abspath(openings_root) if openings_root and os.path.isdir(openings_root) else OPENINGS_DEFAULT
    if _pkg_ready(root, voice_id):
        return root
    return OPENINGS_DEFAULT if _pkg_ready(OPENINGS_DEFAULT, voice_id) else None


def start_available(voice_id, openings_root):
    return opening_root(voice_id, openings_root) is not None


def _frame0(meta_file):
    try:
        f = json.load(open(meta_file))["sources"]["pose_tokens"]["frame0"]["fields"]
        return [float(f["frame0_root"][0]), float(f["frame0_root"][1]), float(f["frame0_width"])]
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        return None


def clip_source(voice_id, openings_root, clip_codes=None, clip_src="rule"):
    """-> dict(kind, codes [32, L] int64, words file | None, pose file | None, frame0 [y, x, w] | None, key, notes) for the voice clip."""
    notes = []
    if clip_src == "start":
        r = opening_root(voice_id, openings_root)
        if r is None:
            return None
        d = f"{r}/{voice_id}"
        codes = np.ascontiguousarray(_load(f"{d}/mimi_local.safetensors")[:, :CLIP_MAX])
        wf = f"{d}/words.json"
        return {"kind": "conversation start", "codes": codes, "words": wf if os.path.isfile(wf) else None,
                "pose": f"{d}/pose_tokens.safetensors", "frame0": _frame0(f"{d}/meta.json"),
                "key": f"OPENING:{os.path.basename(r)}/{voice_id}/mimi_local[:, :{codes.shape[1]}]",
                "notes": notes + ([] if os.path.isfile(wf) else ["no word timings for this opening: PAD word lane"])}
    root = os.path.abspath(openings_root) if openings_root and os.path.isdir(openings_root) else OPENINGS_DEFAULT
    if voice_id and root != os.path.abspath(OPENINGS_DEFAULT) and _pkg_ready(root, voice_id) and os.path.isfile(f"{root}/{voice_id}/voice_clip.safetensors"):
        d = f"{root}/{voice_id}"
        codes = _load(f"{d}/voice_clip.safetensors")
        wf = pf_ = gs = None
        if os.path.isfile(f"{d}/clipwin_meta.json"):
            try:
                m = json.load(open(f"{d}/clipwin_meta.json"))
                fl = ((m.get("sources") or {}).get("pose_tokens") or {}).get("frame0", {}).get("fields") or {}
                if fl.get("frame0_root") is not None and fl.get("frame0_width") is not None:
                    gs = [float(fl["frame0_root"][0]), float(fl["frame0_root"][1]), float(fl["frame0_width"])]
            except (OSError, ValueError):
                notes.append("clipwin_meta.json unreadable")
            wf = f"{d}/clipwin_words.json" if os.path.isfile(f"{d}/clipwin_words.json") else None
            pf_ = f"{d}/clipwin_pose_tokens.safetensors" if os.path.isfile(f"{d}/clipwin_pose_tokens.safetensors") else None
        else:
            notes.append("clip-window words / pose not ready (no clipwin_meta.json): codes only")
        return {"kind": "sam clip window", "codes": codes, "words": wf, "pose": pf_, "frame0": gs if pf_ else None,
                "key": f"OPENINGS:{os.path.basename(root)}/{voice_id}/voice_clip", "notes": notes}
    if voice_id and os.path.isdir(f"{VIDEO_PEOPLE}/{voice_id}/gt"):
        g = f"{VIDEO_PEOPLE}/{voice_id}/gt"
        codes = _load(f"{g}/mimi_local.safetensors")[:, :CLIP_MAX]
        wf = f"{GENWEB}/words/{voice_id}/words.json"
        return {"kind": "video clip window", "codes": codes, "words": wf if os.path.isfile(wf) else None,
                "pose": f"{g}/pose_tokens.safetensors" if os.path.isfile(f"{g}/pose_tokens.safetensors") else None, "frame0": None,
                "key": f"VIDEO/{voice_id}", "notes": notes + ([] if os.path.isfile(wf) else ["no word timings for this clip: PAD word lane"])}
    if clip_codes is not None:
        c = np.asarray(clip_codes, np.int64)
        if c.ndim == 2 and c.shape[0] != 32 and c.shape[1] == 32:
            c = c.T
        return {"kind": "bank clip", "codes": np.ascontiguousarray(c), "words": None, "pose": None, "frame0": None, "key": f"BANK/{voice_id}",
                "notes": ["bank clip: no words / pose for this clip (audio-only prefill)"]}
    return None


def clip_seconds(voice_id, openings_root, clip_codes=None, clip_src="rule"):
    s = clip_source(voice_id, openings_root, clip_codes, clip_src)
    return round(s["codes"].shape[1] / 12.5, 2) if s else 0.0


def clip_for(voice_id, clip_src, openings_root, clip_codes=None):
    """-> (codes int64 [32, L] | None, info{kind, key, sha256, frames}) : the voice clip B's prompt gets for this clip source."""
    s = clip_source(voice_id, openings_root, clip_codes, clip_src)
    if s is None:
        return None, {"kind": None, "clip_src": clip_src, "notes": [f"no {clip_src!r} clip for voice {voice_id!r}"]}
    return s["codes"], {"kind": s["kind"], "clip_src": clip_src, "key": s["key"], "sha256": sha(s["codes"]), "frames": int(s["codes"].shape[1])}


def _lane(LPmod, tok, wf, N):
    w = json.load(open(wf)); w = w["words"] if isinstance(w, dict) else w
    ws = []
    for x in w:
        if isinstance(x, dict):
            ws.append({"text": str(x.get("text", x.get("w"))), "start": float(x.get("start", x.get("t0"))), "end": float(x.get("end", x.get("t1", 0)))})
        else:
            ws.append({"text": str(x[2]), "start": float(x[0]), "end": float(x[1])})
    ws = [x for x in sorted(ws, key=lambda x: x["start"]) if x["start"] >= 0]
    pieces = LPmod.SP.word_pieces(tok, ws)
    pid, offs = [], [0]
    for pc in pieces:
        pid += pc; offs.append(len(pid))
    nl = N + LPmod.CD + 25                                    # longer than the window: a word starting right after it puts its NEW_WORD
    wif = np.zeros(nl, np.int64)                              # on the window's last frame (= the training lane)
    lane = np.asarray(LPmod.SP.lane(ws, pid, offs, nl, wif), np.int64)
    LPmod.SP.insert_new_word(lane, wif)
    return lane[:N + LPmod.CD]


def prefill_from_clip(voice_id, pf_s, openings_root, want_pose, LPmod, tok, clip_codes=None, want_words=True, clip_src="rule"):
    info = {"on": False, "src": None, "voice": voice_id, "clip_src": clip_src, "frames": 0, "prefill_s": 0.0, "streams": [], "notes": []}
    pf = snap(pf_s)
    info["prefill_s"] = pf
    if pf <= 0:
        info["notes"].append("prefill off")
        return None, None, None, info
    s = clip_source(voice_id, openings_root, clip_codes, clip_src)
    if s is None:
        info["notes"].append(f"no {clip_src!r} clip for voice {voice_id!r}: no prefill")
        return None, None, None, info
    codes_full = s["codes"]
    pose_full = _load(s["pose"]) if s["pose"] else None
    N = min(codes_full.shape[1], n_rule(pf), *( [pose_full.shape[1]] if pose_full is not None else [] ))
    codes = np.ascontiguousarray(codes_full[:, :N])
    info.update(on=True, src=s["kind"], clip_key=s["key"], frames=N, clip_frames=int(codes_full.shape[1]), notes=info["notes"] + s["notes"])
    lane = np.full(N + LPmod.CD, LPmod.PAD, np.int64)
    if want_words and s["words"]:
        lane = _lane(LPmod, tok, s["words"], N)
        info["streams"].append("words")
    pose = None
    if want_pose:
        if pose_full is not None:
            pose = np.ascontiguousarray(pose_full[:, :N].T)
            info["streams"].insert(0, "pose")
            if s["frame0"] is not None:
                info["frame0_global_state"] = s["frame0"]
        else:
            info["notes"].append("no pose tokens for this clip: no pose prefill")
    info["streams"].append("audio")
    info["sha256_codes"], info["sha256_lane"] = sha(codes), sha(lane)
    if pose is not None:
        info["sha256_pose"] = sha(pose)
    return codes, lane, pose, info
