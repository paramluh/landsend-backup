#!/usr/bin/env python3
"""pose6 kit (B3, 10-27): BASE + SIDECAR preflight for the s2p+G0COND monoplex FT (CPU only, no model build, nothing written except --json).

BASE (any w-/export path): format = consolidated DCP (.metadata, e.g. ckpt-weka/ppm-v1/w-N of run 31101) or ppm-local-bf16-shards-v1
(meta.json). Its key -> full shape table must equal EXACTLY the FT model's parameter table = job 24272's export-357 (the mono8020 model, read
from its rank / rep files: kit/ref/model_keys_export357.json, built by --make-ref) PLUS the two new keys with the right shapes:
  tower_p.s_emb.weight  [n_cb x 2048, H_tower]   cb0 -> n_cb = 1 (PPM_S2P=cb0; a 'full' base, 32 x 2048 rows, FAILS)
  inp.g0.weight         [H_trunk, 8]             (Linear(G0_DIM 8 -> 5120, no bias))
No extra key (e.g. pose_head.*), no missing key, every dtype bf16 (DCP), both new tensors finite (norms printed; all-zero = WARNING: an
untrained base). H_tower / H_trunk are read from the reference table (layer-norm widths), not typed.
SIDECAR: every train + eval item of the plan (data_aloe_pose.splits_v2 under the kit env) has a usable frame-0 record
(data_aloe_pose.g0_coverage of pose6/code = the trainer's own check), printed with the f = 0 count.

usage: preflight.py --base <path> --code <pose6/code> [--json out.json] [--skip-sidecar]
       preflight.py --make-ref <export-357 dir> --out kit/ref/model_keys_export357.json"""
import argparse
import json
import os
import sys
import time

REF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ref", "model_keys_export357.json")
REP_PREFIX = ("dec_s.", "dec_p.")


def table_local(path):
    """ppm-local-bf16-shards-v1 -> {key: [full shape]}, dtypes, rep keys (rank files read with mmap: headers only)."""
    import torch
    meta = json.load(open(f"{path}/meta.json"))
    assert meta.get("format") == "ppm-local-bf16-shards-v1" and (os.path.isfile(f"{path}/COMPLETE") or os.environ.get("PREFLIGHT_ALLOW_INCOMPLETE") == "1"), \
        ("not a COMPLETE local-shards dir (PREFLIGHT_ALLOW_INCOMPLETE=1: verify before COMPLETE is written)", path)
    W0 = int(meta["world"])
    rep = torch.load(f"{path}/rep.pt", map_location="cpu", mmap=True, weights_only=True)
    tab, dt = {k: list(v.shape) for k, v in rep.items()}, {k: str(v.dtype) for k, v in rep.items()}
    rows = {}
    for r in range(W0):
        f = f"{path}/rank{r}.pt"
        if not os.path.isfile(f):
            raise AssertionError(("local-shards rank file missing (node-local stage dirs hold only their node's ranks)", f))
        sd = torch.load(f, map_location="cpu", mmap=True, weights_only=True)
        for k, v in sd.items():
            if k not in rows:
                rows[k] = [0] + list(v.shape[1:])
                dt[k] = str(v.dtype)
            assert list(v.shape[1:]) == rows[k][1:], ("inconsistent trailing dims", k, r)
            rows[k][0] += v.shape[0]
        del sd
    tab.update(rows)
    return tab, dt, sorted(rep), {"format": "ppm-local-bf16-shards-v1", "world": W0, "step": meta.get("step"), "run_tag": meta.get("run_tag")}


def table_dcp(path):
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.metadata import TensorStorageMetadata
    assert os.path.isfile(f"{path}/COMPLETE") and os.path.isfile(f"{path}/.metadata"), ("not a COMPLETE consolidated DCP", path)
    md = dcp.FileSystemReader(path).read_metadata().state_dict_metadata
    bad = [k for k, v in md.items() if not isinstance(v, TensorStorageMetadata)]
    assert not bad, ("non-tensor DCP entries", bad[:6])
    side = {}
    for f in ("train_cfg.json", "side.json"):
        if os.path.isfile(f"{path}/{f}"):
            side = json.load(open(f"{path}/{f}")); break
    return ({k: list(v.size) for k, v in md.items()}, {k: str(v.properties.dtype) for k, v in md.items()},
            sorted(k for k in md if k.startswith(REP_PREFIX)),
            {"format": "dcp", "step": open(f"{path}/COMPLETE").read().strip(), "side": side})


def table(path):
    return table_local(path) if os.path.isfile(f"{path}/meta.json") else table_dcp(path)


def new_tensors(path, fmt, keys):
    """the two new tensors' values (small): finite + norm."""
    import torch
    out = {}
    if fmt == "dcp":
        import torch.distributed.checkpoint as dcp
        md = dcp.FileSystemReader(path).read_metadata().state_dict_metadata
        sd = {k: torch.empty(tuple(md[k].size), dtype=md[k].properties.dtype) for k in keys}
        dcp.load(sd, checkpoint_id=path, no_dist=True)
    else:
        W0 = int(json.load(open(f"{path}/meta.json"))["world"])
        parts = {k: [] for k in keys}
        for r in range(W0):
            sd_r = torch.load(f"{path}/rank{r}.pt", map_location="cpu", mmap=True, weights_only=True)
            for k in keys:
                parts[k].append(sd_r[k].clone())
        sd = {k: torch.cat(v, 0) for k, v in parts.items()}
    for k, t in sd.items():
        x = t.float()
        out[k] = {"finite": bool(torch.isfinite(x).all()), "norm": float(x.norm()), "absmax": float(x.abs().max()), "nonzero_frac": float((x != 0).float().mean())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base")
    ap.add_argument("--code", default="/data/home/amartyavarma/ppm/pose6/code")
    ap.add_argument("--json")
    ap.add_argument("--skip-sidecar", action="store_true")
    ap.add_argument("--make-ref")
    ap.add_argument("--mods", default="s2pg0", choices=("s2pg0", "none"), help="none = the mono17850 kit: the base must be the pre-mods key set EXACTLY (no s_emb / g0)")
    ap.add_argument("--out", default=REF)
    a = ap.parse_args()
    t0 = time.time()
    if a.make_ref:
        tab, dt, rep, info = table_local(a.make_ref)
        json.dump({"src": os.path.realpath(a.make_ref), "info": info, "shapes": tab, "dtypes": dt, "rep": rep}, open(a.out, "w"), indent=0)
        print(f"[preflight] ref table {a.out}: {len(tab)} keys ({len(rep)} replicated) from {a.make_ref}; {time.time() - t0:.0f}s")
        return 0
    ref = json.load(open(REF))
    rs = ref["shapes"]
    assert "tower_p.s_emb.weight" not in rs and "inp.g0.weight" not in rs, "reference table must be the pre-mods model (export-357)"
    h_tower = rs["tower_p.norm.weight"][0] if "tower_p.norm.weight" in rs else \
        [v[0] for k, v in sorted(rs.items()) if k.startswith("tower_p.layers.0.") and k.endswith("input_layernorm.weight")][0]
    h_trunk = [v[0] for k, v in sorted(rs.items()) if k.startswith("layers.0.") and k.endswith("input_layernorm.weight")][0]
    want = dict(rs)
    if a.mods == "s2pg0":
        want["tower_p.s_emb.weight"] = [1 * 2048, h_tower]              # PPM_S2P=cb0 (S2PEmb: (32 if full else 1) x 2048 rows)
        want["inp.g0.weight"] = [h_trunk, 8]                             # G0_DIM 8 -> H_trunk, no bias
    NEWK = ["tower_p.s_emb.weight", "inp.g0.weight"] if a.mods == "s2pg0" else []
    res = {"base": os.path.realpath(a.base), "mods": a.mods, "h_tower": h_tower, "h_trunk": h_trunk, "ok": True, "fail": []}
    tab, dt, rep, info = table(a.base)
    res["base_info"] = info
    miss = sorted(set(want) - set(tab))
    extra = sorted(set(tab) - set(want))
    shp = sorted(k for k in set(want) & set(tab) if list(want[k]) != list(tab[k]))
    nbf = sorted(k for k, v in dt.items() if v != "torch.bfloat16")
    res.update({"n_keys_base": len(tab), "n_keys_want": len(want), "missing": miss[:20], "n_missing": len(miss), "extra": extra[:20],
                "n_extra": len(extra), "shape_mismatch": [(k, want[k], tab[k]) for k in shp[:20]], "n_shape_mismatch": len(shp),
                "non_bf16": nbf[:20], "n_non_bf16": len(nbf)})
    for name, cond in (("missing keys (base lacks FT-model params, e.g. the 2 new keys)", miss), ("extra keys (not in the FT model)", extra),
                       ("shape mismatches", shp), ("non-bf16 tensors", nbf)):
        if cond:
            res["fail"].append(f"{name}: {cond[:6]}{' ...' if len(cond) > 6 else ''}")
    for k in NEWK:
        res[k] = {"want": want[k], "base": tab.get(k, "<absent>")}
    if NEWK and not miss and not shp:
        nv = new_tensors(a.base, info["format"], NEWK)
        res["new_values"] = nv
        for k, v in nv.items():
            if not v["finite"]:
                res["fail"].append(f"{k}: non-finite values")
            if v["norm"] == 0.0:
                res.setdefault("warn", []).append(f"{k}: ALL ZERO in the base (an untrained / pre-mods base: s2p / g0 would start from zero, not from the base's training)")
    if not a.skip_sidecar and a.mods == "s2pg0":
        sys.path[:0] = [f"{a.code}/vendor", f"{a.code}/lib", a.code]
        assert os.environ.get("PPM_G0COND") == "1" and os.environ.get("ALOE_G0_SIDECAR"), "sidecar check needs the kit env sourced (PPM_G0COND=1, ALOE_G0_SIDECAR)"
        import data_aloe_pose as A
        sp = A.splits_v2()
        cov = {}
        for s in ("train", "eval"):
            try:
                n, nz = A.g0_coverage(sp[s])
                cov[s] = {"items": len(sp[s]), "covered": int(n), "f_zero": int(nz)}
            except AssertionError as e:                                  # the trainer's own fail-loud check
                cov[s] = {"items": len(sp[s]), "error": repr(e)[:400]}
                res["fail"].append(f"sidecar {s}: {repr(e)[:200]}")
        res["sidecar"] = {"path": os.environ["ALOE_G0_SIDECAR"], **cov}
        if cov.get("train", {}).get("items") != 27517 or cov.get("eval", {}).get("items") != 531:
            res["fail"].append(f"item counts != mono8020's 27,517 / 531: {cov}")
    res["ok"] = not res["fail"]
    res["secs"] = round(time.time() - t0, 1)
    print("[preflight] " + json.dumps(res, indent=1, default=str))
    print(f"[preflight] RESULT {'PASS' if res['ok'] else 'FAIL'}: base {info['format']} step {info.get('step')}, {len(tab)} keys vs {len(want)} wanted"
          f" (s_emb {tab.get('tower_p.s_emb.weight')}, g0 {tab.get('inp.g0.weight')})" + ("" if res["ok"] else f"; FAIL: {res['fail']}"), flush=True)
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1, default=str)
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
