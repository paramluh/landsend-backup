"""gesturemetric 10-26: GESTURE-SPEECH COUPLING metrics on posev3 pose tokens + the same speaker's word timings (+ optional audio).
User ask (via main): a gesture-speech correlation score that penalises random flailing, timing/emphasis AND meaning.

Strokes (from posefix/score/hands.py, same decode as run_stats.py): per hand, the 0.2 s smoothed wrist speed (sw / frame); a stroke = a
local peak of it >= PEAK_THR with prominence >= PROM and >= MIN_SEP_S from the previous one; left + right peaks within MERGE_S merged.
Stroke time = the peak's transition time (frame i -> i+1 = (i + 0.5) / 25 s on the stream's own clock).

(1) TIMING  events = own-speaker word onsets ('words': all; 'content': non-function words = stress proxy) or loudness peaks ('loud').
    hit     = share of strokes with an event within +-W s of (stroke - LAG)
    chance  = mean hit with the events circularly shifted by K uniform offsets in [SHIFT_MIN, L - SHIFT_MIN] (kills all coupling)
    local   = mean hit with every stroke jittered by U(+-JIT) (keeps speaking-vs-silence coupling, kills fine timing)
    timing_pp = 100 (hit - chance)   (all coupling; flailing / mismatched speech -> 0)
    fine_pp   = 100 (hit - local)    (sub-second alignment only)
    kappa     = (hit - chance) / (1 - chance)   HEADLINE (speech-density-free; 0 = flailing / unrelated)
(3) MEANING-CHEAP  per own-speaker word: does a stroke fall in [t0 - PRE, t0 + POST]? rate for each category (deictic / self / size /
    negation) vs the rate for the other content words; lift = rate_cat / rate_other (1 = no semantic preference).
    + substance lift = stroke rate on substantive words (not FILLER, not FUNC) / rate on FILLER words (backchannels / hesitations).
No Whisper anywhere: words come from the data's own word files (real) or the model's word lane (generated)."""
import os
import re
import sys

import numpy as np

SCORE = "/data/home/amartyavarma/ppm/aloe-posedup-live/posefix/score"
sys.path.insert(0, SCORE)

FPS = 25
PEAK_THR = 0.015        # = hands.MOVE_THR (sw / frame): a stroke must reach the gesture-motion threshold
PROM = 0.0075           # peak prominence (half the threshold): ignore ripples on one sustained movement
MIN_SEP_S = 0.3
MERGE_S = 0.2
W = 0.2                 # +- s hit window (proposal; re-checked by calib)
LAG = 0.0               # stroke - event lag (s); calib.py fits it on real Aloe and this is overwritten from out/calib.json if present
K_SHIFT = 200
SHIFT_MIN = 3.0
JIT = 1.0
PRE, POST = 0.5, 0.3    # (3) word window [onset - PRE, onset + POST] (fixed width: no word-length bias)

FUNC = set("""a an the and or but if so of to in on at by for with from as is am are was were be been being do does did have has had
it its it's this's i'm i'd i'll i've you you're you'd you'll you've he she they we him her them us his hers their our your my me
that's there's what's who's let's can could would should will shall may might must just also too very really like um uh umm uhh er
erm hmm mm mhm uh-huh uhhuh yeah yes yep ya okay ok oh ah well right gonna wanna kinda sorta than then because cause 'cause which who
whom whose what when where how why not no""".split())
FILLER = set("""yeah yes yep ya yup okay ok oh ah ahh aha um uh umm uhh er erm hmm mm mhm uh-huh uhhuh huh wow right cool nice sure
true really totally exactly absolutely ha haha""".split())
CATS = {
    "deictic": set("this that these those here there over look see".split()),
    "self": set("i me my mine myself i'm i've i'd i'll we us our ours ourselves we're we've".split()),
    "size": set("""big bigger biggest huge massive enormous giant large small smaller tiny little long short tall wide narrow high low
                   whole all everything every lot lots much many more most loads tons full entire""".split()),
    "negation": set("""no not never nothing nobody none nope nah don't doesn't didn't can't cannot won't wouldn't isn't wasn't aren't
                       weren't haven't hasn't hadn't shouldn't couldn't neither nor""".split()),
}


def norm_word(w):
    return re.sub(r"[^a-z'\-]", "", w.lower().replace("’", "'")).strip("'-")


# ------------------------------------------------------------------------------------------------ pose -> strokes
def k55_of(codes, person, max_tokens=None, device="cpu"):
    """[F, 32] local ids (no -1) -> (k55 [2F, 55, 3] px, actor frame-0 shoulder width px). Same decode as run_stats.py (filters none).
    device "cuda" decodes on the GPU (score.decode's CUDA path = pose.decode, TF32 off); keypoints stay on CPU."""
    import score as S
    c, _, _ = S.prep_codes(codes, 0, max_tokens)
    kp, kp0 = S.to_keypoints(S.decode(c, device), S.frame0_of(person))
    k0 = kp0.numpy()
    return kp.numpy()[:, S.P.SEL].astype(np.float64), max(abs(float(k0[5, 0] - k0[6, 0])), 1e-3)


def wrist_speed(k55, sw):
    import hands as HD
    return HD.series(k55, sw)[1]          # [T-1, 2] smoothed wrist speed (sw / frame), left / right


def strokes(ws, lo_s=0.0):
    """ws [T-1, 2] -> (stroke times s [n], stroke peak speeds [n]); only strokes at t >= lo_s."""
    from scipy.signal import find_peaks
    ts, vs = [], []
    for h in range(ws.shape[1]):
        p, pr = find_peaks(ws[:, h], height=PEAK_THR, prominence=PROM, distance=max(1, int(MIN_SEP_S * FPS)))
        ts.append((p + 0.5) / FPS)
        vs.append(pr["peak_heights"])
    t, v = np.concatenate(ts), np.concatenate(vs)
    o = np.argsort(t)
    t, v = t[o], v[o]
    keep_t, keep_v = [], []
    for a, b in zip(t, v):
        if keep_t and a - keep_t[-1] < MERGE_S:
            if b > keep_v[-1]:
                keep_t[-1], keep_v[-1] = a, b
            continue
        keep_t.append(a); keep_v.append(b)
    t, v = np.array(keep_t), np.array(keep_v)
    m = t >= lo_s
    return t[m], v[m]


# ------------------------------------------------------------------------------------------------ speech events
def word_events(words, kind="words", lo_s=0.0, hi_s=None):
    """words [(t0, t1, w)] -> onset times for kind 'words' (all) / 'content' (non-function words)."""
    out = []
    for t0, t1, w in words:
        n = norm_word(w)
        if not n:
            continue
        if kind == "content" and n in FUNC:
            continue
        out.append(t0)
    out = np.array(sorted(out), dtype=np.float64)
    m = out >= lo_s - 1.0
    if hi_s is not None:
        m &= out <= hi_s + 1.0
    return out[m]


def loudness_env(wav, sr, hop_s=0.01):
    """mono float wav -> RMS dB envelope at 1 / hop_s Hz (50 ms window)."""
    hop, win = int(round(sr * hop_s)), int(round(sr * 0.05))
    n = max(0, (len(wav) - win) // hop + 1)
    if n == 0:
        return np.zeros(0)
    idx = np.arange(win)[None] + hop * np.arange(n)[:, None]
    rms = np.sqrt((wav[idx].astype(np.float64) ** 2).mean(1) + 1e-12)
    return 20 * np.log10(rms + 1e-9)


def loud_events(env_db, hop_s=0.01, prom_db=6.0, floor_rel_db=-30.0):
    """loudness peaks: local maxima with >= prom_db prominence, >= 0.15 s apart, louder than (track p99 + floor_rel_db)."""
    from scipy.signal import find_peaks
    if not len(env_db):
        return np.zeros(0)
    floor = np.percentile(env_db, 99) + floor_rel_db
    p, _ = find_peaks(env_db, height=floor, prominence=prom_db, distance=int(0.15 / hop_s))
    return p * hop_s + 0.025


# ------------------------------------------------------------------------------------------------ (1) timing
def _hit(st, ev, w=W, lag=LAG):
    if not len(st) or not len(ev):
        return np.zeros(len(st), bool)
    x = st - lag
    i = np.clip(np.searchsorted(ev, x), 1, len(ev) - 1) if len(ev) > 1 else np.zeros(len(x), int)
    d = np.minimum(np.abs(x - ev[i - 1 if len(ev) > 1 else i]), np.abs(x - ev[i]))
    return d <= w


def timing(st, ev, lo_s, hi_s, w=W, lag=LAG, k=K_SHIFT, seed=0):
    """strokes st, events ev on [lo_s, hi_s] -> dict(n, hit, chance, local, timing_pp, fine_pp) (+ per-null samples for pooling)."""
    rng = np.random.default_rng(seed)
    st = st[(st >= lo_s) & (st <= hi_s)]
    ev = np.sort(ev[(ev >= lo_s) & (ev <= hi_s)])     # the shift null wraps inside [lo_s, hi_s]: only in-range events
    L = hi_s - lo_s
    n = len(st)
    if n == 0 or L <= 2 * SHIFT_MIN:
        return {"n": n, "hit": None, "chance": None, "local": None, "timing_pp": None, "fine_pp": None, "kappa": None, "hits": 0}
    h = _hit(st, ev, w, lag)
    ch = []
    for d in rng.uniform(SHIFT_MIN, L - SHIFT_MIN, k):
        e2 = np.sort(lo_s + (ev - lo_s + d) % L)
        ch.append(_hit(st, e2, w, lag).mean())
    lc = [_hit(st + rng.uniform(-JIT, JIT, n), ev, w, lag).mean() for _ in range(k // 4)]
    hit, chance, local = float(h.mean()), float(np.mean(ch)), float(np.mean(lc))
    return {"n": n, "hits": int(h.sum()), "hit": hit, "chance": chance, "local": local,
            "timing_pp": 100 * (hit - chance), "fine_pp": 100 * (hit - local), "kappa": float(np.clip((hit - chance) / max(1 - chance, 0.05), -1, 1))}   # clipped: chance ~1 (dense speech) is unstable


def lag_hist(st, ev, lo_s, hi_s, span=1.5, bin_s=0.04):
    """histogram of (stroke - event) for all pairs within +-span (events in range) -> counts per bin (for the lag calibration)."""
    st = st[(st >= lo_s) & (st <= hi_s)]
    bins = np.arange(-span, span + 1e-9, bin_s)
    if not len(st) or not len(ev):
        return np.zeros(len(bins) - 1)
    d = (st[:, None] - ev[None]).ravel()
    return np.histogram(d[np.abs(d) <= span], bins)[0]


# ------------------------------------------------------------------------------------------------ (3) cheap meaning
def word_stroke_table(st, words, lo_s, hi_s, pre=PRE, post=POST):
    """-> list of (category or 'other' or None, has_stroke) for own-speaker words fully inside [lo_s, hi_s].
    category: first matching CATS key; 'other' = remaining content words; None = remaining function words (not used)."""
    rows = []
    st = np.sort(st)
    for t0, t1, w in words:
        if t0 < lo_s + pre or t0 > hi_s - post:
            continue
        n = norm_word(w)
        if not n:
            continue
        cat = next((c for c, s in CATS.items() if n in s), None)
        if cat is None:
            cat = None if n in FUNC else "other"
        a, b = np.searchsorted(st, [t0 - pre, t0 + post])
        rows.append((cat, bool(b > a)))
    return rows


def lifts(rows):
    """pooled rows -> {cat: {n, rate, lift}} + 'any' (all 4 categories pooled) vs 'other'."""
    import collections
    c = collections.defaultdict(lambda: [0, 0])
    for cat, s in rows:
        if cat is None:
            continue
        c[cat][0] += 1; c[cat][1] += s
        if cat != "other":
            c["any"][0] += 1; c["any"][1] += s
    oth = c["other"][1] / max(c["other"][0], 1)
    out = {}
    for k, (n, s) in c.items():
        r = s / max(n, 1)
        out[k] = {"n": n, "rate": round(r, 4), "lift": None if k == "other" or oth == 0 else round(r / oth, 3)}
    return out


def substance_rows(st, words, lo_s, hi_s, pre=PRE, post=POST):
    """-> list of ('sub' | 'fill', has_stroke) for own words with onset window inside [lo_s, hi_s] (other function words skipped)."""
    rows = []
    st = np.sort(st)
    for t0, t1, w in words:
        if t0 < lo_s + pre or t0 > hi_s - post:
            continue
        n = norm_word(w)
        if not n or (n in FUNC and n not in FILLER):
            continue
        a, b = np.searchsorted(st, [t0 - pre, t0 + post])
        rows.append(("fill" if n in FILLER else "sub", bool(b > a)))
    return rows


def substance_lift(rows):
    s = [x for c, x in rows if c == "sub"]
    f = [x for c, x in rows if c == "fill"]
    if not s or not f or not np.mean(f):
        return None
    return {"n_sub": len(s), "n_fill": len(f), "rate_sub": round(float(np.mean(s)), 4), "rate_fill": round(float(np.mean(f)), 4),
            "lift": round(float(np.mean(s) / np.mean(f)), 3)}
