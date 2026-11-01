"""gesturemetric: loudness envelopes for real Aloe tracks (no local FLAC: CPU Mimi decode of the own side's mimi-stream-v5 codes).
Decoder = genweb/worker.py's (ppm_io.Mimi: transformers MimiModel kyutai/mimi @ 89091b3e, local HF cache, offline), run on CPU in 30 s
chunks (375 frames) with 2 s (25 frames) of left context decoded and discarded (the worker decodes a whole clip in one call; chunking
only bounds memory). -> out/loud/<conv>__<side>.npy = gm.loudness_env at 100 Hz (float16 dB), sample 0 = code frame 0 = pose token 0.
CPU cost: ~0.5 s per s of audio on 2 threads (60 s in 28-32 s, chunked == one-call to 1e-5 dB), so the default covers only the first
MAX_TOK = 3000 tokens (240 s, = calib_hands.py's 4 min) of each track: loudness events exist on [0, len(env) / 100 s] only.
usage: loud_extract.py [N_CONVS] [MAX_TOK] [SHARD/NSHARD]
       loud_extract.py --check RUN_DIR [RUN_DIR ...]   (decode a genweb run's B codes, compare the env with its b.wav's)
       loud_extract.py --rate [N]                       (loud_events per second of own speech on N real tracks)"""
import json
import os
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, H)
import numpy as np  # noqa: E402
import torch  # noqa: E402
import gm  # noqa: E402

ALOE = "/data/home/amartyavarma/data/aloe-handover-full/DATA"
OUT = os.path.join(H, "out", "loud")
SPF, SR = 1920, 24000
CHUNK, CTX = 375, 25
_M = {}


def mimi():
    if "m" not in _M:
        from transformers import MimiModel
        _M["m"] = MimiModel.from_pretrained("kyutai/mimi", revision="89091b3e466eb6a9d11e537bf26b144f194978f7").eval()
    return _M["m"]


@torch.no_grad()
def decode(codes):
    """codes [32, F] local -> float32 wav [F * 1920] at 24 kHz (chunked, see module doc)."""
    codes = np.asarray(codes, np.int64)
    F = codes.shape[1]
    out = []
    for s in range(0, F, CHUNK):
        a0 = max(0, s - CTX)
        c = torch.as_tensor(codes[:, a0:min(F, s + CHUNK)])[None]
        w = mimi().decode(c).audio_values[0, 0].float().numpy()
        out.append(w[(s - a0) * SPF:(min(F, s + CHUNK) - a0) * SPF])
    return np.concatenate(out)[:F * SPF]


def tracks(n_conv):
    import real_extract as RE
    for v in RE.eval_convs()[:n_conv]:
        for side in ("assistant", "user"):
            s = v["sides"][side]
            d = os.path.join(ALOE, os.path.dirname(s["path"]))
            yield v["conv"].replace("/", "_") + "__" + side, os.path.join(d, f"w00.{s['spk']}.codes.npy"), d, s["spk"]


def main():
    n_conv = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    max_tok = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
    sh, nsh = (map(int, sys.argv[3].split("/")) if len(sys.argv) > 3 else (0, 1))
    os.makedirs(OUT, exist_ok=True)
    for i, (name, cp, _, _) in enumerate(tracks(n_conv)):
        if i % nsh != sh:
            continue
        fo = os.path.join(OUT, name + ".npy")
        if os.path.exists(fo):
            continue
        t0 = time.time()
        c = np.load(cp)
        assert c.shape[0] == 32, c.shape
        w = decode(c[:, :max_tok])
        env = gm.loudness_env(w, SR).astype(np.float16)
        np.save(fo + ".part.npy", env)
        os.replace(fo + ".part.npy", fo)
        print(f"{name} {c.shape[1]} frames -> {len(w) / SR:.0f}s env {len(env)} {time.time() - t0:.1f}s", flush=True)


def check(runs):
    import soundfile as sf
    for rd in runs:
        z = np.load(os.path.join(rd, "tokens.npz"))
        c = z["mimi"].astype(np.int64)
        c = c.T if c.shape[1] == 32 else c
        t0 = time.time()
        w = decode(c)
        el = time.time() - t0
        b, sr = sf.read(os.path.join(rd, "b.wav"), dtype="float32")
        e1, e2 = gm.loudness_env(w, SR), gm.loudness_env(b, sr)
        n = min(len(e1), len(e2))
        e1, e2 = e1[:n], e2[:n]
        best = max(range(-20, 21), key=lambda k: np.corrcoef(e1[max(0, k):n + min(0, k)], e2[max(0, -k):n - max(0, k)])[0, 1])
        r0 = np.corrcoef(e1, e2)[0, 1]
        m = (e2 > np.percentile(e2, 99) - 30)
        p1, p2 = gm.loud_events(e1), gm.loud_events(e2)
        hit = np.mean([np.min(np.abs(p2 - t)) <= 0.03 for t in p1]) if len(p1) and len(p2) else float("nan")
        print(f"{os.path.basename(rd)} {c.shape[1]} fr decode {el:.1f}s | env corr {r0:.4f} best lag {10 * best:+d} ms | "
              f"abs dB diff (speech) med {np.median(np.abs(e1 - e2)[m]):.2f} | peaks {len(p1)} vs {len(p2)}, matched<=30ms {hit:.3f}",
              flush=True)


def rate(n):
    import real_extract as RE
    by_name = {}
    for v in RE.eval_convs()[:60]:
        by_name[v["conv"].replace("/", "_")] = v
    done = sorted(f for f in os.listdir(OUT) if f.endswith(".npy") and not f.endswith(".part.npy"))[:n]
    tot_p, tot_s = 0, 0.0
    for f in done:
        name = f[:-4]
        conv, side = name.rsplit("__", 1)
        v = by_name[conv]
        env = np.load(os.path.join(OUT, f)).astype(np.float64)
        L = len(env) / 100
        ws = [w for w in RE.words_of(v).get(v["sides"][side]["spk"], []) if w[1] <= L]
        sp = np.zeros(len(env), bool)
        for a, b, _ in ws:
            sp[int(a * 100):int(b * 100) + 1] = True
        p = gm.loud_events(env)
        pin = sp[np.clip((p * 100).astype(int), 0, len(sp) - 1)]
        ss = sp.sum() / 100
        tot_p += pin.sum(); tot_s += ss
        print(f"{name} speech {ss:.0f}s peaks in speech {pin.sum()} ({pin.sum() / max(ss, 1e-9):.2f}/s) outside {(~pin).sum()} "
              f"({(~pin).sum() / max(L - ss, 1e-9):.2f}/s)", flush=True)
    print(f"POOLED {tot_p / max(tot_s, 1e-9):.2f} peaks per s of own speech over {len(done)} tracks")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    if len(sys.argv) > 1 and sys.argv[1] == "--check":
        check(sys.argv[2:])
    elif len(sys.argv) > 1 and sys.argv[1] == "--rate":
        rate(int(sys.argv[2]) if len(sys.argv) > 2 else 10)
    else:
        main()
