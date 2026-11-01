"""A-INPUT LEVEL for the live demo (10-15 level fix): a drop-in for vox_stream.RunningLoudness (same interface: __init__(g0_db=...),
attributes target / slew / g_db, push(frame [1920] raw) -> (frame heard by the model, gain_db)).

TRAINING (what is matched): ONE pure gain per conversation on A = -23 LUFS integrated over the whole conversation's A (pyloudnorm
BS.1770-4: K-weighting, 400 ms blocks / 75 % overlap, gates -70 LUFS and -10 LU), capped so peak x gain <= 0.995
(qqa_rows.py / voxtral_front.lufs_gain_db). The model therefore never heard A above 0.995 and heard the user's FIRST words already at the
conversation's -23 LUFS gain.

The old live rule (RunningLoudness) held the gain at g0 (0 dB = the raw mic level) until 3 s of gated speech existed, then slewed at
3 dB/s, and capped the gain only by the peak of frames BEFORE the current one: the opening question reached the model 5-17 dB off and
frames could exceed full scale (heard peaks 1.4-2.9).

This rule (causal at frame granularity: a frame's gain uses only that frame and earlier samples, no added latency):
  1. before any speech: gain = the prior g0 (auto: the median training gain of the demo's recorded browser-mic sessions, PRIOR_DB);
  2. speech ONSET per 80 ms frame: the first VOICED frame (autocorrelation pitch peak >= 0.5 at 62-400 Hz) louder (K-weighted) than
     max(-70, noise floor + 20 dB) (noise floor:
     fast-fall / slow-rise tracker initialised from the first non-zero frame; digital silence ignored); low-level clicks / breaths / room noise never start it;
  3. ACQUISITION (fast attack): L = the BS.1770 gating rule on the voiced speech frames since the onset (absolute -70, relative
     -10 LU, power mean; incl. the current frame) + a bias by the number k of gated frames (calibrated offline on held-out training A: median over
     conversations of [whole-conversation BS.1770 integrated - this estimate at k frames]; the few onset frames read low), blended with the prior by K0 frames of pseudo-count;
     gain = target - L (clamped to [-20, +24] dB) applied at once (no slew) until ACQ_S s of gated speech frames exist, so the opening
     words are at the training level from the first speech frames; then slewed;
  4. TRACKING: once acquisition is done and >= WARM_S of gated 400 ms blocks exist, the target is the BS.1770-4 integrated loudness of everything heard so far
     (= the training quantity, converging to the per-conversation gain), approached at <= `slew` dB/s;
  5. PEAK SAFETY: the gain is ramped linearly across each frame (no steps), then a lookahead-in-frame peak limiter (attack 2 ms, release
     50 ms) and a final hard ceiling keep every sample <= 0.995 (training's peak guard). Unlike training's whole-episode peak cap, a
     single loud transient does not lower the level of the rest of the session (it is limited where it happens).
"""
import json
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PRIOR_FILE = os.path.join(HERE, "levelfix", "prior.json")
SR, SPF = 24000, 1920
TARGET, PEAK = -23.0, 0.995
BIAS_DB = 0.0          # overwritten from levelfix/prior.json (calibration)
BIAS_BY_K = []         # bias by the number k of gated speech frames (1, 2, ...; the last one holds for larger k); from prior.json
K0 = 1.0               # prior pseudo-count in speech frames
ACQ_S = 1.0            # seconds of speech with an instant (unslewed) gain
MIN_K = 1              # gated voiced frames needed before the first acquired gain (from prior.json)
WARM_S = 3.0           # gated 400 ms blocks (s) before the BS.1770 integrated estimate takes over
VAD_ABS, VAD_SNR = -70.0, 20.0
NF_RISE_DB_S = 3.0
DIGITAL_ZERO = -100.0
NF_INIT_MAX = -65.0
VOICED = 0.5            # voicing() threshold: only voiced speech frames start / feed the acquisition estimate
PRIOR_DB = 2.5         # overwritten from levelfix/prior.json
LIM_ATTACK, LIM_RELEASE_S = 48, 0.05
LIVE = False           # set True by live_engine (the demo server process): only then are traces saved
TRACE_DIR = os.path.join(os.environ.get("PPMLIVE_STATE_DIR") or HERE, "sessions", "_alevel")   # live sessions: per-frame gain trace saved when the session's estimator is freed
HINT_ON = ("A loudness match on: your mic reaches the model at the -23 LUFS training level from your first voiced words (fast acquisition), "
           "then tracks the BS.1770 integrated level; peaks limited to 0.995 (training never exceeded it)")
HINT_G0 = ("A gain before your first voiced speech (dB). 0 = automatic (the median level of earlier browser mics). From the first voiced "
           "frame on, the gain is the estimated -23 LUFS training gain")
HINT_SLEW = "max gain change speed (dB/s) after the first ~1 s of voiced speech (which is acquired instantly)"
HINT_GAIN = "manual gain on your mic audio (dB), used only when the loudness match is off"
G_MIN, G_MAX = -20.0, 24.0  # plausible acquisition gains (recorded browser mics needed -8 ... +20 dB)


def _load_prior():
    global PRIOR_DB, BIAS_DB, BIAS_BY_K, K0, ACQ_S, MIN_K
    try:
        d = json.load(open(PRIOR_FILE))
        BIAS_BY_K = [float(v) for v in d.get("bias_by_k", [])]
        PRIOR_DB = float(d.get("prior_db", PRIOR_DB))
        BIAS_DB = float(d.get("bias_db", BIAS_DB))
        K0 = float(d.get("k0", K0))
        ACQ_S = float(d.get("acq_s", ACQ_S))
        MIN_K = int(d.get("min_k", MIN_K))
    except Exception:  # noqa: BLE001  (no file: built-in defaults)
        pass


_load_prior()


def _db(e):
    return -0.691 + 10.0 * np.log10(e + 1e-20)


def voicing(x):
    """periodicity of an 80 ms frame: peak of the normalised autocorrelation at pitch lags 2.5-16 ms (62-400 Hz), on the frame decimated
    to 8 kHz (mean of 3). Voiced speech ~0.6-0.95; breath / noise / clicks ~0-0.35."""
    n = len(x) // 3 * 3
    if n < 192:
        return 0.0
    y = np.asarray(x[:n], np.float64).reshape(-1, 3).mean(1)
    y = y - y.mean()
    m = len(y)
    F = np.fft.rfft(y, 2 * m)
    r = np.fft.irfft(F * np.conj(F))[:m]
    if r[0] <= 1e-18:
        return 0.0
    lo, hi = 20, min(128, m // 2)
    rr = r[lo:hi] / (r[0] * (m - np.arange(lo, hi)) / m)     # unbiased normalisation
    return float(np.clip(rr.max(), 0.0, 1.0))


class StreamLevel:
    def __init__(self, sr=SR, target=TARGET, g0_db=0.0, warm_s=WARM_S, slew_db_s=3.0, peak=PEAK, prior_db=None, bias_db=None, k0=None,
                 acq_s=None, trace=None, min_k=None):
        import pyloudnorm as pyln
        from scipy.signal import lfilter_zi
        assert sr == SR, sr
        self.sr, self.target, self.peak, self.slew, self.warm_s = sr, float(target), float(peak), float(slew_db_s), float(warm_s)
        self.filt = [(f.b, f.a) for f in pyln.Meter(sr)._filters.values()]
        self.zi = [lfilter_zi(b, a) * 0.0 for b, a in self.filt]
        g0 = float(g0_db)
        self.g0_auto = (g0 == 0.0)                               # 0 = automatic start gain (the knob's default)
        self.g0 = float(PRIOR_DB if prior_db is None else prior_db) if self.g0_auto else g0
        self.bias = float(BIAS_DB if bias_db is None else bias_db)
        self.bias_k = (list(BIAS_BY_K) if BIAS_BY_K else None) if bias_db is None else None   # per gated-frame-count bias (calibrated)
        self.k0 = float(K0 if k0 is None else k0)
        self.acq_s = float(ACQ_S if acq_s is None else acq_s)
        self.min_k = int(MIN_K if min_k is None else min_k)
        self.g_cur = self.g0                                     # gain (dB) at the end of the last frame (before the limiter)
        self.env = 1.0                                           # limiter envelope at the end of the last frame
        self.rel = float(np.exp(-1.0 / (LIM_RELEASE_S * sr)))
        self.nf = None                                           # noise floor (dB), initialised from the first non-zero frame
        self.zero_lead = False                                   # the stream opened with digital silence
        self.sp_e = []                                           # K-weighted mean squares of the frames since the speech onset
        self.onset = None                                        # sample index of the speech onset
        self.buf = np.zeros(0)
        self.e100, self.blocks = [], []
        self.n = 0
        self.trace = [] if (trace or (trace is None and LIVE and os.path.isdir(TRACE_DIR))) else None
        self.save_trace = trace is None and self.trace is not None
        self.t_start = time.strftime("%Y%m%d-%H%M%S")
        self.phase = "prior"
        self.acq_started = False

    @property
    def g_db(self):
        """gain for samples right after the last frame (the server's lookahead write): last gain x limiter state."""
        return self.g_cur + 20.0 * np.log10(max(self.env, 1e-6))

    def _integrated(self):
        z = np.asarray(self.blocks)
        if not len(z):
            return None
        z = z[_db(z) > -70.0]
        if len(z) * 0.1 < self.warm_s:
            return None
        lr = _db(z.mean()) - 10.0
        z2 = z[_db(z) > lr]
        return float(_db(z2.mean())) if len(z2) else None

    def _fast(self, count=False):
        """BS.1770 gating on the 80 ms frames since the onset (absolute -70, relative -10 LU) -> power-mean loudness + bias (and the number
        of gated frames = the speech evidence: a lone click / breath that started the onset is gated out once real speech arrives)."""
        e = np.asarray(self.sp_e)
        e = e[_db(e) > -70.0]
        if not len(e):
            return (None, 0) if count else None
        m = e.mean()
        e2 = e[_db(e) > _db(m) - 10.0]
        k = len(e2) if len(e2) else len(e)
        b = self.bias if self.bias_k is None else self.bias_k[min(k, len(self.bias_k)) - 1]
        L = float(_db(e2.mean() if len(e2) else m)) + b
        return (L, k) if count else L

    def push(self, frame):
        from scipy.signal import lfilter
        x = np.asarray(frame, np.float64)
        n = len(x)
        y = x
        for i, (b, a) in enumerate(self.filt):
            y, self.zi[i] = lfilter(b, a, y, zi=self.zi[i])
        ef = float(np.mean(y ** 2)) if n else 0.0
        lf = _db(ef)
        # BS.1770 block history (the training quantity)
        self.buf = np.concatenate([self.buf, y])
        sub = int(0.1 * self.sr)
        while len(self.buf) >= sub:
            self.e100.append(float(np.mean(self.buf[:sub] ** 2))); self.buf = self.buf[sub:]
            if len(self.e100) >= 4:
                self.blocks.append(float(np.mean(self.e100[-4:])))
        # VAD + noise floor
        speech = False
        if lf <= DIGITAL_ZERO and self.nf is None:
            self.zero_lead = True
        if lf > DIGITAL_ZERO:                                    # digital silence (stream start, muted mic) never sets the noise floor
            if self.nf is None:                                  # from the first non-zero frame; capped at -50 (a live mic opens on its
                # noise floor), or at -65 after leading digital silence (file / generated audio: the first sound is usually speech)
                self.nf = float(np.clip(lf, -90.0, NF_INIT_MAX if self.zero_lead else -50.0))
            speech = lf > max(VAD_ABS, self.nf + VAD_SNR)
            if lf < self.nf:
                self.nf = max(-90.0, 0.5 * self.nf + 0.5 * lf)
            else:
                self.nf = min(self.nf + NF_RISE_DB_S * n / self.sr, lf)
        v = voicing(x) if speech or self.onset is not None else 0.0
        voiced = speech and v >= VOICED
        if voiced and self.onset is None:
            self.onset = self.n
        if self.onset is not None and voiced:
            self.sp_e.append(ef)                                 # voiced speech frames from the onset on (gated in _fast)
        g_prev = self.g_cur
        Lf, k = self._fast(count=True) if self.sp_e else (None, 0)
        acq_done = k * SPF / self.sr >= self.acq_s               # >= ACQ_S of gated speech frames
        L_int = self._integrated() if acq_done else None
        step = self.slew * n / self.sr
        if L_int is not None:
            self.phase = "track"                                 # the training quantity (integrated over everything heard), slewed
            g_new = float(np.clip(self.target - L_int, g_prev - step, g_prev + step))
        elif Lf is not None and k >= self.min_k:
            L = (k * Lf + self.k0 * (self.target - self.g0)) / (k + self.k0)
            want = float(np.clip(self.target - L, G_MIN, G_MAX))
            if not acq_done:
                self.phase = "acquire"                           # instant: the opening words get the estimated gain at once
                g_new = want
            else:
                self.phase = "settle"                            # fast estimate, slewed, until the integrated estimate is warm
                g_new = float(np.clip(want, g_prev - step, g_prev + step))
        else:
            g_new = g_prev
        self.g_cur = g_new
        if self.phase == "acquire" and not self.acq_started:
            self.acq_started = True
            g_prev = g_new                                       # the first acquired frame gets its gain whole (the step lands at the
                                                                 # speech onset, after the noise floor at the prior gain)
        # linear amplitude ramp across the frame (no gain steps at frame boundaries within speech)
        a0, a1 = 10.0 ** (g_prev / 20.0), 10.0 ** (g_new / 20.0)
        ramp = a0 + (a1 - a0) * (np.arange(1, n + 1) / max(n, 1))
        out = x * ramp
        # peak limiter: per-sample need r = min(1, peak / |out|), 2 ms lookahead min inside the frame, exponential release from the state
        ab = np.abs(out)
        if ab.max(initial=0.0) > self.peak or self.env < 1.0:
            r = np.minimum(1.0, self.peak / np.maximum(ab, 1e-12))
            if LIM_ATTACK > 1:
                from numpy.lib.stride_tricks import sliding_window_view
                rp = np.concatenate([r, np.ones(LIM_ATTACK - 1)])
                r = sliding_window_view(rp, LIM_ATTACK).min(1)
            # env[i] = 1 - rel^(i+1) * max(d_prev, max_{k<=i} (1 - r[k]) rel^-(k+1))   (release toward 1, attack instant)
            idx = np.arange(1, n + 1, dtype=np.float64)
            w = np.maximum.accumulate((1.0 - r) * np.exp(idx / (LIM_RELEASE_S * self.sr)))
            d = np.exp(-idx / (LIM_RELEASE_S * self.sr)) * np.maximum(1.0 - self.env, w)
            env = 1.0 - d
            out = out * env
            self.env = float(env[-1]) if env[-1] < 1.0 - 1e-5 else 1.0
        out = np.clip(out, -self.peak, self.peak)
        self.n += n
        if self.trace is not None:
            self.trace.append((self.n / self.sr, g_new, self.env, lf, bool(voiced), self.phase))
        return out.astype(np.float32), g_new

    def __del__(self):
        """live sessions: save the per-frame trace (t_end_s, gain_db, limiter env, frame loudness, voiced, phase) next to the sessions."""
        try:
            if self.save_trace and self.trace and self.n > 0:
                tr = self.trace
                np.savez_compressed(os.path.join(TRACE_DIR, f"{self.t_start}-{time.strftime('%H%M%S')}.npz"),
                                    t=np.asarray([x[0] for x in tr], np.float32), g=np.asarray([x[1] for x in tr], np.float32),
                                    env=np.asarray([x[2] for x in tr], np.float32), lf=np.asarray([x[3] for x in tr], np.float32),
                                    voiced=np.asarray([x[4] for x in tr]), phase=np.asarray([x[5] for x in tr]),
                                    g0=self.g0, onset=-1 if self.onset is None else self.onset)
        except Exception:  # noqa: BLE001
            pass
