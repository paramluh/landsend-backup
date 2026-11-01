"""aloe-v2vox-raw UNIT TEST (CPU): channel A's waveform with VOX_A_GAIN=raw.
  1. FLAC ('gcs') entries, train AND eval micros: the loader's wav (data_aloe_v2vox.Stores.conv -> rowcache_vox.row_audio ->
     audio_s4q.ffmpeg_row_audio) == the RAW FLAC samples, bit for bit: an independent ffmpeg decode of the same file (same binary, -f f32le
     -ac 1 -ar 24000), samples [1920 frame0, 1920 (frame0 + n)) zero-padded, NO scaling.  And the old path (VOX_A_GAIN=mimi) == raw x
     float32(10^(recorded gain / 20)) bit for bit (so the gain is the ONLY difference).
  2. 'mimidec' entries: wav == the decoded file x float32(10^(-recorded Mimi LUFS gain / 20)) bit for bit, i.e. 20 log10(rms raw / rms old)
     == -recorded gain.
usage: VOX_A_GAIN=raw ALOE_FLAC_ROOT=<dir holding the A tracks> python -I test_raw_gain.py [--n 6]"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

R = "/data/home/amartyavarma/ppm/aloe-v2vox-raw"
sys.path.insert(0, f"{R}/code"); sys.path.insert(0, f"{R}/code/lib")


def raw_decode(path, frame0, n):
    import audio_s4q as AU
    r = subprocess.run([AU.FFMPEG_EXE, "-nostdin", "-hide_banner", "-loglevel", "error", "-i", path, "-f", "f32le", "-ac", "1", "-ar", "24000", "-"],
                       capture_output=True, check=True)
    x = np.frombuffer(r.stdout, dtype=np.float32)
    s0, L = 1920 * frame0, 1920 * n
    w = np.zeros(L, np.float32)
    seg = x[s0:s0 + L]
    w[:len(seg)] = seg
    return w


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=6)
    a = ap.parse_args()
    assert os.environ.get("VOX_A_GAIN") == "raw"
    import data_aloe_v2vox as A
    assert A.GAIN_MODE == "raw"
    PL = A.make_plan(64)
    idx = A.vox_index()
    rg = A.rawgain()
    st = A.Stores()

    def entry(it):
        k, r, sw = it
        return idx[int(k)][int(r)]["A"]["assistant" if int(sw) else "user"]

    present = lambda e: os.path.exists(f"{A.FLAC_ROOT}/{e['flac']}") and (e["kind"] != "gcs" or os.path.getsize(f"{A.FLAC_ROOT}/{e['flac']}") == int(e["flac_bytes"]))  # noqa: E731
    rng = np.random.default_rng(20261017)
    cases = []
    for name, items, cfg in (("train", PL["items"], A.aloe_cfg()), ("eval", PL["eval"], A.eval_cfg())):
        perm = rng.permutation(len(items))
        g = [items[i] for i in perm if entry(items[i])["kind"] == "gcs" and present(entry(items[i]))][:a.n]
        m = [items[i] for i in perm if entry(items[i])["kind"] == "mimidec" and present(entry(items[i]))][:max(1, a.n // 3)]
        cases += [(name, it, cfg) for it in g + m]
    assert sum(1 for c in cases if entry(c[1])["kind"] == "gcs") >= 2 and sum(1 for c in cases if entry(c[1])["kind"] == "mimidec") >= 1, "too few staged cases"
    ok = 0
    for name, it, cfg in cases:
        e = entry(it)
        L = 0                                                       # rank argument (does not touch the A waveform)
        A.GAIN_MODE = "raw"
        m_raw = st.conv(tuple(it), cfg, L)
        A.GAIN_MODE = "mimi"
        m_old = st.conv(tuple(it), cfg, L)
        A.GAIN_MODE = "raw"
        w, wo = m_raw["wav"], m_old["wav"]
        ref = raw_decode(f"{A.FLAC_ROOT}/{e['flac']}", int(e["audio_frame0"]), int(e["n_frames"]))
        assert w.dtype == np.float32 and w.shape == ref.shape, (it, w.shape, ref.shape)
        if e["kind"] == "gcs":
            g_old = float(e["gain_db"])
            assert m_raw["aref"]["gain_db"] == 0.0 and rg[e["flac"]]["gain_raw_db"] == 0.0
            assert w.tobytes() == ref.tobytes(), ("raw-A wav != raw FLAC samples", it)
            assert wo.tobytes() == (ref * np.float32(10.0 ** (g_old / 20.0))).astype(np.float32).tobytes(), ("old path != raw x recorded gain", it)
            msg = f"wav == raw FLAC samples (bit-exact); old path == raw x 10^({g_old:+.3f}/20) (bit-exact)"
        else:
            g_rec = float(rg[e["flac"]]["recorded_lufs_gain_db"])
            assert m_raw["aref"]["gain_db"] == -g_rec and m_old["aref"]["gain_db"] == 0.0
            assert wo.tobytes() == ref.tobytes(), ("old mimidec path != the decoded file", it)
            assert w.tobytes() == (ref * np.float32(10.0 ** (-g_rec / 20.0))).astype(np.float32).tobytes(), ("raw mimidec != file x 10^(-g/20)", it)
            d_db = 20 * np.log10(np.sqrt(np.mean(w.astype(np.float64) ** 2)) / np.sqrt(np.mean(wo.astype(np.float64) ** 2)))
            assert abs(d_db + g_rec) < 1e-3, (it, d_db, g_rec)
            msg = f"wav == mimidec file x 10^({-g_rec:+.3f}/20) (bit-exact; level change {d_db:+.3f} dB = -recorded gain)"
        ok += 1
        print(f"PASS {name} {e['kind']:7s} {e['flac']} n={int(e['n_frames'])}: {msg}", flush=True)
    print(f"ALL PASS: {ok} cases ({sum(1 for c in cases if c[0] == 'train')} train, {sum(1 for c in cases if c[0] == 'eval')} eval)", flush=True)


if __name__ == "__main__":
    main()
