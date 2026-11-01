"""si20k fp32 vgain-theta FIX-UP (coordinator 10-19: "the exports save the vgain theta (log s) in bf16, so the exported s is off by -1.3 %
(PPM) / +0.9 % (INTL) from the trained fp32 value"; no maths change). Puts the EXACT fp32 theta of the export's step into an existing export:
  EXACT VALUE  the trainer logs s = exp(theta) every step as train/vox_a_scale = math.exp(float(theta_fp32)) (a python double, JSON repr exact)
               after that step's update, i.e. the value at the export; theta_fp32 = float32(log(s)) recovers it exactly (asserted:
               exp(float(theta_fp32)) == s bit for bit, and bf16(theta_fp32) == the bf16 theta stored in the export).
  DCP export   (.metadata + __R_0.distcp, consolidated): the fp32 tensor is torch.save'd and APPENDED to the distcp file that holds vgain.theta
               (every other byte of every file unchanged), the .metadata entry points at it (offset / length) with dtype float32. The
               original .metadata is kept outside the export dir (<run>/hf_backup/thetafix/). Verified by dcp.load: theta == theta_fp32 exactly
               and a reference tensor bit-identical before / after.
  local export (rank<R>.pt + rep.pt + meta.json): the rank file holding the non-empty vgain.theta shard is re-saved with that one entry
               in fp32 (every other entry asserted bit-identical after the rewrite).
  RECORD       train_cfg.json (DCP) / meta.json (local) gain "vgain_theta_fp32", "vgain_s_fp32", "theta_fixup" {source, bf16 before, time};
               + THETA_FP32.json next to them.
usage: python -I theta_fixup.py <export dir> --metrics <metrics.jsonl> --step N [--run-dir <run dir>] [--dry]"""
import argparse
import io
import json
import math
import os
import pickle
import shutil
import sys
import time

import numpy as np
import torch

KEY = "vgain.theta"


def exact_theta(metrics, step):
    s = None
    for ln in open(metrics):
        try:
            d = json.loads(ln)
        except ValueError:
            continue
        if int(d.get("step", -1)) == int(step) and d.get("train/vox_a_scale") is not None:
            s = float(d["train/vox_a_scale"])
    assert s is not None, ("no train/vox_a_scale at this step", metrics, step)
    th = np.float32(math.log(s))
    assert math.exp(float(th)) == s, ("theta_fp32 does not reproduce the logged s exactly", float(th), s)
    return float(th), s


def bf16_of(x):
    return float(torch.tensor([x], dtype=torch.float32).to(torch.bfloat16).float()[0])


def fix_dcp(d, th, s, src, run_dir, dry):
    import torch.distributed.checkpoint as dcp
    md_path = f"{d}/.metadata"
    md = pickle.load(open(md_path, "rb"))
    sdm = md.state_dict_metadata
    assert KEY in sdm, ("export has no vgain.theta", d)
    tm = sdm[KEY]
    nz = [c for c in tm.chunks if int(np.prod(list(c.sizes))) > 0]      # the one non-empty chunk (FSDP: rank 0's shard; the rest are empty)
    assert len(nz) == 1 and list(nz[0].offsets) == [0] and list(nz[0].sizes) == [1], ("vgain.theta: expected one 1-element chunk", tm.chunks)
    idx = [k for k in md.storage_data if k.fqn == KEY and list(k.offset) == [0]]
    assert len(idx) == 1, ("vgain.theta must have one storage item at offset 0", [k for k in md.storage_data if k.fqn == KEY])
    info = md.storage_data[idx[0]]
    f = f"{d}/{info.relative_path}"
    raw = open(f, "rb").read()[info.offset:info.offset + info.length] if os.path.getsize(f) < 4e9 else None
    if raw is None:
        with open(f, "rb") as fh:
            fh.seek(info.offset); raw = fh.read(info.length)
    old = torch.load(io.BytesIO(raw), weights_only=True)
    if tm.properties.dtype == torch.float32:
        assert float(old.reshape(-1)[0]) == np.float32(th), ("already fp32 but a different value", float(old.reshape(-1)[0]), th)
        return {"status": "already fp32", "theta": th}
    assert tm.properties.dtype == torch.bfloat16 and old.dtype == torch.bfloat16 and tuple(old.shape) == (1,), (tm.properties.dtype, old.dtype, old.shape)
    before = float(old.float()[0])
    assert before == bf16_of(th), ("stored bf16 theta != bf16(exact theta): wrong step / run?", before, bf16_of(th), th)
    ref_key = "head.spec_out" if "head.spec_out" in sdm else sorted(k for k in sdm if k != KEY)[0]
    ref0 = torch.empty(tuple(sdm[ref_key].size), dtype=sdm[ref_key].properties.dtype)
    dcp.load({ref_key: ref0}, checkpoint_id=d, no_dist=True)
    rec = {"status": "dry" if dry else "fixed", "format": "dcp", "file": info.relative_path, "old_offset": info.offset, "old_length": info.length,
           "file_size_before": os.path.getsize(f), "bf16_before": before, "theta_fp32": th, "s_fp32": s, "source": src}
    if dry:
        return rec
    bak = f"{run_dir}/hf_backup/thetafix"
    os.makedirs(bak, exist_ok=True)
    shutil.copy2(md_path, f"{bak}/{os.path.basename(d)}.metadata.orig-bf16theta")
    buf = io.BytesIO(); torch.save(torch.tensor([th], dtype=torch.float32), buf); blob = buf.getvalue()
    off = os.path.getsize(f)
    with open(f, "ab") as fh:
        fh.write(blob); fh.flush(); os.fsync(fh.fileno())
    md.storage_data[idx[0]] = type(info)(relative_path=info.relative_path, offset=off, length=len(blob),
                                         **{k_: getattr(info, k_) for k_ in getattr(info, "__dataclass_fields__", {}) if k_ not in ("relative_path", "offset", "length")})
    tm.properties.dtype = torch.float32
    tmp = md_path + ".tmp"
    with open(tmp, "wb") as fh:
        pickle.dump(md, fh); fh.flush(); os.fsync(fh.fileno())
    os.replace(tmp, md_path)
    t = torch.empty(1, dtype=torch.float32)
    dcp.load({KEY: t}, checkpoint_id=d, no_dist=True)
    assert float(t[0]) == np.float32(th), ("dcp.load after the fix-up != theta_fp32", float(t[0]), th)
    ref1 = torch.empty_like(ref0)
    dcp.load({ref_key: ref1}, checkpoint_id=d, no_dist=True)
    assert torch.equal(ref0, ref1), ("reference tensor changed by the fix-up", ref_key)
    rec.update(new_offset=off, new_length=len(blob), file_size_after=os.path.getsize(f), verified_dcp_load=True, ref_tensor_unchanged=ref_key)
    meta_f = f"{d}/train_cfg.json"
    j = json.load(open(meta_f)) if os.path.isfile(meta_f) else {}
    j.update(vgain_theta_fp32=th, vgain_s_fp32=s, theta_fixup=dict(rec, time=time.strftime("%F %T %Z")))
    json.dump(j, open(meta_f + ".tmp", "w"), indent=1, default=str); os.replace(meta_f + ".tmp", meta_f)
    return rec


def fix_local(d, th, s, src, run_dir, dry):
    hit = None
    for fn in sorted(os.listdir(d)):
        if fn.startswith("rank") and fn.endswith(".pt"):
            sd = torch.load(f"{d}/{fn}", map_location="cpu", mmap=True, weights_only=True)
            if KEY in sd and sd[KEY].numel() == 1:
                assert hit is None, ("two rank files hold vgain.theta", hit, fn)
                hit = fn
    assert hit is not None, ("no rank file holds the vgain.theta element", d)
    sd = torch.load(f"{d}/{hit}", map_location="cpu", weights_only=True)
    old = sd[KEY]
    if old.dtype == torch.float32:
        assert float(old.reshape(-1)[0]) == np.float32(th), ("already fp32 but a different value", float(old.reshape(-1)[0]), th)
        return {"status": "already fp32", "theta": th}
    before = float(old.float().reshape(-1)[0])
    assert old.dtype == torch.bfloat16 and before == bf16_of(th), ("stored bf16 theta != bf16(exact theta)", old.dtype, before, bf16_of(th), th)
    rec = {"status": "dry" if dry else "fixed", "format": "local", "file": hit, "bf16_before": before, "theta_fp32": th, "s_fp32": s, "source": src}
    if dry:
        return rec
    new = dict(sd); new[KEY] = torch.tensor([th], dtype=torch.float32).reshape(old.shape)
    torch.save(new, f"{d}/{hit}.tmp"); os.replace(f"{d}/{hit}.tmp", f"{d}/{hit}")
    chk = torch.load(f"{d}/{hit}", map_location="cpu", weights_only=True)
    assert set(chk) == set(sd) and float(chk[KEY].reshape(-1)[0]) == np.float32(th) and chk[KEY].dtype == torch.float32
    assert all(torch.equal(chk[k], sd[k]) for k in sd if k != KEY), "a non-theta entry changed"
    rec.update(verified_reload=True, other_entries_unchanged=len(sd) - 1)
    meta_f = f"{d}/meta.json"
    j = json.load(open(meta_f))
    j.update(vgain_theta_fp32=th, vgain_s_fp32=s, theta_fixup=dict(rec, time=time.strftime("%F %T %Z")))
    json.dump(j, open(meta_f + ".tmp", "w"), indent=1, default=str); os.replace(meta_f + ".tmp", meta_f)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("export"); ap.add_argument("--metrics", required=True); ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--run-dir", default=""); ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    d = a.export.rstrip("/")
    run_dir = a.run_dir or os.path.dirname(os.path.dirname(d)) if "/exports/" in d else (a.run_dir or os.path.dirname(d))
    th, s = exact_theta(a.metrics, a.step)
    src = f"{a.metrics} step {a.step} train/vox_a_scale"
    rec = fix_dcp(d, th, s, src, run_dir, a.dry) if os.path.isfile(f"{d}/.metadata") else fix_local(d, th, s, src, run_dir, a.dry)
    if not a.dry and rec.get("status") == "fixed":
        json.dump(dict(rec, export=d, time=time.strftime("%F %T %Z")), open(f"{d}/THETA_FP32.json", "w"), indent=1, default=str)
    print("THETA_FIXUP", json.dumps(dict(rec, export=d), default=str), flush=True)


if __name__ == "__main__":
    sys.exit(main())
