"""aloe-v2vox-raw (10-17, user: = aloe-v2vox-ft job 22420 with NO loudness normalisation of the channel-A audio going into Voxtral).
Builds, once, for EVERY channel-A track the v2 plan reads (64 logical ranks: train items incl. the dummy fallback item + eval items):
  plan/RAWGAIN.json   flac rel -> the gain the raw-A loader applies (VOX_A_GAIN=raw):
                        'gcs'     (WLT FLAC, the recording)                -> 0 dB (raw recording level)
                        'mimidec' (Mimi decode of a purged conversation's  -> MINUS the track's recorded Mimi LUFS gain (w00.mimi.json
                                   codes, made from the NORMALISED track)      mimi[spk].lufs_gain_db, read from the no-face quarantine copy
                                                                              gs://aloe-processed-datasets/_quarantine/stage2-handover-20261008-
                                                                              noface-20261016/stage2-handover-20261008/mimi-stream-v5/<ses>/<seg>/)
                      + the old (VOX_A_GAIN=mimi) gain for reference (gcs: the index's recorded gain; mimidec: 0 dB).
                      Each mimidec track's quarantined codes are checked == the m1 store row's own A codes (so the gain belongs to the codes
                      the decode is made from).
  mimidec/_mimidec/... the mimidec FLACs, decoded ONCE here exactly as tools/stage_v2vox.py (job 22407) decodes them on a node (transformers
                      MimiModel /data/home/amartyavarma/models/mimi, fp32, the row's A codes[:, :n] -> audio[:n*1920] -> clip [-1, 1-2^-23]
                      -> 24 kHz PCM_24 FLAC); MANIFEST.json (rel -> bytes, md5). tools/stage_raw.py copies them to every node (md5 verified).
usage (compute node, 1 GPU; GOOGLE_APPLICATION_CREDENTIALS = key file path): python -I build_rawgain_mimidec.py"""
import base64
import hashlib
import io
import json
import os
import sys
import time

import numpy as np

R = "/data/home/amartyavarma/ppm/aloe-v2vox-raw"
sys.path.insert(0, f"{R}/code"); sys.path.insert(0, f"{R}/code/lib")
BUCKET = "aloe-processed-datasets"
QPRE = "_quarantine/stage2-handover-20261008-noface-20261016/stage2-handover-20261008/mimi-stream-v5/"
WORLD, LW = 64, 8
OUT_GAIN = f"{R}/plan/RAWGAIN.json"
OUT_DEC = f"{R}/mimidec"


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return base64.b64encode(h.digest()).decode()


def main():
    t0 = time.time()
    import data_aloe_v2vox as A
    import s3rows as F
    PL = A.make_plan(WORLD, roots=A.SRC_STORES)
    dg = hashlib.sha256(json.dumps(PL["items"]).encode()).hexdigest()[:16]
    assert dg == "ba2d70729ea437ab" and PL["steps"] == 253 and len(PL["eval"]) == 324, (dg, PL["steps"], len(PL["eval"]))
    items = set()
    for node in range(WORLD // LW):
        items |= set(A.node_items(PL, node, WORLD, LW))
    idx = A.vox_index()
    tracks = {}
    for k, r, sw in sorted(items):
        aside = "assistant" if int(sw) else "user"
        e = idx[int(k)][int(r)]
        a = e["A"][aside]
        t = {"kind": a["kind"], "gain_mimi_db": float(a["gain_db"]), "a_spk": a["a_spk"], "row_key": e["row_key"], "store": int(k), "row": int(r),
             "side": aside, "n_frames": int(a["n_frames"]), "frame0": int(a["frame0"]), "audio_frame0": int(a["audio_frame0"])}
        if a["flac"] in tracks:
            assert {x: tracks[a["flac"]][x] for x in ("kind", "gain_mimi_db", "a_spk")} == {x: t[x] for x in ("kind", "gain_mimi_db", "a_spk")}
            continue
        tracks[a["flac"]] = t
    gcs = {f: t for f, t in tracks.items() if t["kind"] == "gcs"}
    mdec = {f: t for f, t in tracks.items() if t["kind"] == "mimidec"}
    assert len(gcs) + len(mdec) == len(tracks)
    print(f"[plan] {dg}: {len(items)} items (train + fallback + eval over 64 ranks) -> {len(tracks)} A tracks: {len(gcs)} gcs, {len(mdec)} mimidec", flush=True)
    for f, t in gcs.items():
        t["gain_raw_db"] = 0.0
    # ---- mimidec: recorded gain from the quarantined w00.mimi.json, codes check against the store row
    from google.cloud import storage
    cli = storage.Client.from_service_account_json(os.environ["GOOGLE_APPLICATION_CREDENTIALS"])
    bkt = cli.bucket(BUCKET)
    rows_ = {}
    mj_cache = {}
    for f, t in sorted(mdec.items()):
        _, ses, seg, base = f.split("/")
        assert base.startswith("c00.") and base.endswith(f".{t['a_spk']}.flac") and t["audio_frame0"] == 0 and t["gain_mimi_db"] == 0.0, (f, t)
        if (ses, seg) not in mj_cache:
            bl = bkt.blob(f"{QPRE}{ses}/{seg}/w00.mimi.json"); bl.reload()
            mj_cache[(ses, seg)] = (json.loads(bl.download_as_bytes()), int(bl.generation))
        mj, gen = mj_cache[(ses, seg)]
        assert mj["session"] == ses and mj["seg"] == seg, (f, mj["session"], mj["seg"])
        m = mj["mimi"][t["a_spk"]]
        cb = bkt.blob(f"{QPRE}{ses}/{seg}/{m['codes_file']}"); cb.reload()
        qc = np.load(io.BytesIO(cb.download_as_bytes()))
        k, r, side, n = t["store"], t["row"], t["side"], t["n_frames"]
        if k not in rows_:
            rows_[k] = F.Rows(F.load_node(A.SRC_STORES[k]))
        row = rows_[k][r]
        assert row["row_key"] == t["row_key"] and int(row["frame_start"]) == t["frame0"] == 0
        sc = np.asarray(row[f"codes_{side}"])[:, :n].astype(np.int64)
        qa = np.asarray(qc).astype(np.int64)
        qa = qa if qa.shape[0] == 32 else qa.T
        assert sc.shape == (32, n) and np.array_equal(qa[:, t["frame0"]:t["frame0"] + n], sc), ("quarantined codes != the store row's A codes", f, qa.shape)
        t.update(gain_raw_db=-float(m["lufs_gain_db"]), recorded_lufs_gain_db=float(m["lufs_gain_db"]),
                 lufs_gain_db_intended=m.get("lufs_gain_db_intended"), lufs_status=m.get("lufs_status"), lufs_in=m.get("lufs_in"),
                 src=f"gs://{BUCKET}/{QPRE}{ses}/{seg}/w00.mimi.json#{gen}", codes_checked=True)
    print(f"[mimidec] {len(mdec)} tracks: recorded gain read from the quarantine + codes == store rows; raw gains dB "
          f"min {min(t['gain_raw_db'] for t in mdec.values()):.3f} max {max(t['gain_raw_db'] for t in mdec.values()):.3f}", flush=True)
    # ---- decode the mimidec FLACs once (= stage_v2vox.py's decode)
    import soundfile as sf
    import torch
    from transformers import MimiModel
    dev = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    mimi = MimiModel.from_pretrained("/data/home/amartyavarma/models/mimi").to(dev).eval()
    man, clipped = {}, 0
    for f, t in sorted(mdec.items()):
        row = rows_[t["store"]][t["row"]]
        n = t["n_frames"]
        c = np.asarray(row[f"codes_{t['side']}"])[:, :n].astype(np.int64)
        assert c.shape == (32, n) and int(c.min()) >= 0, (f, c.shape)
        with torch.no_grad():
            w = mimi.decode(torch.as_tensor(c, device=dev)[None]).audio_values[0, 0].float().cpu().numpy()[:n * 1920]
        assert w.shape == (n * 1920,) and np.isfinite(w).all(), (f, w.shape)
        clipped += int((np.abs(w) >= 1.0).sum())
        dst = f"{OUT_DEC}/{f}"
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        sf.write(dst + ".part.flac", np.clip(w, -1.0, 1.0 - 2 ** -23), 24000, subtype="PCM_24", format="FLAC")
        os.replace(dst + ".part.flac", dst)
        man[f] = {"bytes": os.path.getsize(dst), "md5": md5(dst), "store": t["store"], "row": t["row"], "side": t["side"]}
    json.dump({"format": "aloe-v2vox-raw-mimidec-v1", "device": str(dev), "decoder": "transformers MimiModel /data/home/amartyavarma/models/mimi fp32",
               "clipped_samples": clipped, "files": man}, open(f"{OUT_DEC}/MANIFEST.json", "w"), indent=1)
    json.dump({"format": "aloe-v2vox-rawgain-v1", "plan_digest": dg, "rule": "VOX_A_GAIN=raw: gcs 0 dB; mimidec -recorded lufs_gain_db",
               "n_gcs": len(gcs), "n_mimidec": len(mdec), "rows": tracks}, open(OUT_GAIN, "w"), indent=1)
    print(f"[done] {OUT_GAIN} ({len(tracks)} tracks) + {len(man)} mimidec FLACs ({clipped} clipped samples, {dev}) in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
