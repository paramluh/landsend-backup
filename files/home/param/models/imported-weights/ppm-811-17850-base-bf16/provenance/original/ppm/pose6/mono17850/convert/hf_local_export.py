"""pose6 C1 (10-27): PPM training save dcp-N on our private HF repo (InternalCan/ppm-v1-c <tag>/<physical host dir>/rank<R>.pt, rep.pt,
train_state.json) -> the INFERENCE / FT-init checkpoint format ppm-local-bf16-shards-v1 (= ckpt_local_export.py's output), in ONE process,
reading ONLY the 'model' section of every file by HTTP range reads (the AdamW states, 2/3 of each file, are never downloaded).

The conversion maths and file writing are ckpt_local_export.py's (md5 87c639ee, pose5/code-811-s2p = ppm/code) VERBATIM:
    out = {n: t.to(torch.bfloat16).contiguous().clone() for n, t in sd["model"].items()}
    torch.save(out, f"{OUT}/rank{r}.pt.tmp"); os.replace(f"{OUT}/rank{r}.pt.tmp", f"{OUT}/rank{r}.pt")
    rep.pt: same on rep.pt's 'model' section (torch.save straight to OUT/rep.pt); meta.json: the same dict, keys, order and json.dump call.
Only how `sd["model"]` is obtained differs: instead of torch.load(mmap=True) on a local file, each tensor is rebuilt from its zip record as
torch's own _rebuild_tensor_v2 does (storage record bytes -> dtype view -> (offset, size, stride)), which yields the same values bit for bit:
  * the zip central directory + data.pkl are read by range reads; data.pkl is unpickled with a stub unpickler that records, for every tensor,
    its storage key / dtype / numel / offset / size / stride and never touches tensor data;
  * every needed storage record is range-read, its CRC32 checked against the central directory (fail loud), length == numel x itemsize;
  * the rank -> host map is train_state.json 'hosts' (LOGICAL hosts, as the stock tool); a host's HF dir = its physical host (--alias:
    the 8,1,1 run's logical slinky-10 slot was physical slinky-7); every rank file must exist in exactly that dir (fail loud).
Optional checks: --ref <existing local-N dir> (keys / order / dtype / shape / torch.equal per tensor + file sha256), --manifest-dir (the 10-22
monoft per-host manifests of the same conversion: file sha256 + per-tensor rows), every output tensor finite.
HF token: huggingface_hub's default (never printed). Read-only on HF."""
import argparse
import collections
import hashlib
import io
import json
import os
import pickle
import re
import shutil
import threading
import time
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
import torch
from huggingface_hub import HfApi, get_token, hf_hub_url

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="InternalCan/ppm-v1-c")
ap.add_argument("--revision", required=True, help="pinned commit sha of the HF repo")
ap.add_argument("--tag", required=True, help="e.g. dcp-17850")
ap.add_argument("--out", required=True)
ap.add_argument("--ranks", default="all", help="'all' or a comma list")
ap.add_argument("--rep", type=int, default=1)
ap.add_argument("--meta", type=int, default=1)
ap.add_argument("--alias", default="slinky-10:slinky-7", help="logical:hf-dir pairs, comma separated")
ap.add_argument("--threads", type=int, default=16)
ap.add_argument("--chunk-mb", type=int, default=32)
ap.add_argument("--ref", default="", help="existing local-N dir to compare against (verify 1)")
ap.add_argument("--manifest-dir", default="", help="dir of manifest-<host>.json (sha256 of the same conversion, 10-22)")
ap.add_argument("--manifest-must-match", type=int, default=0)
ap.add_argument("--stats", required=True)
ap.add_argument("--min-free-gb", type=float, default=300.0)
a = ap.parse_args()
REPO, REV, TAG = a.repo, a.revision, a.tag
ALIAS = dict(x.split(":") for x in a.alias.split(",") if x)
TOK = get_token()
assert TOK, "no HF token found (huggingface_hub default path)"
S = requests.Session()
S.mount("https://", requests.adapters.HTTPAdapter(pool_connections=8, pool_maxsize=max(64, a.threads * 2)))
LOCK = threading.Lock()


def log(*x):
    print(f"[{time.strftime('%H:%M:%S')} c1]", *x, flush=True)


class Src:
    """one HF file: resolved once to its signed CDN url (no hub round trip per range); refresh on 401/403."""

    def __init__(self, path, size=-1):
        self.path, self.known = path, size
        self.refresh()

    def refresh(self):
        url = hf_hub_url(REPO, self.path, revision=REV)
        for _ in range(6):
            r = S.head(url, headers={"authorization": f"Bearer {TOK}", "accept-encoding": "identity"}, allow_redirects=False, timeout=60)
            if r.status_code in (301, 302, 303, 307, 308):
                loc = r.headers["location"]
                if loc.startswith("/"):
                    url = "https://huggingface.co" + loc
                    continue
                self.url, self.auth = loc, False
                self.size = int(r.headers.get("x-linked-size") or self.known)
                assert self.known < 0 or self.size == self.known, ("size != listing", self.path, self.size, self.known)
                return
            if r.status_code == 200:
                self.url, self.auth, self.size = url, True, int(r.headers.get("content-length") or self.known)
                assert self.known < 0 or self.size == self.known, ("size != listing", self.path, self.size, self.known)
                return
            r.raise_for_status()
        raise RuntimeError(("too many redirects", self.path))

    def get(self, lo, hi, mv):
        """bytes [lo, hi) -> mv (len hi - lo), exact length + content-range checked, retried."""
        last = None
        for att in range(7):
            try:
                h = {"range": f"bytes={lo}-{hi - 1}", "accept-encoding": "identity"}
                if self.auth:
                    h["authorization"] = f"Bearer {TOK}"
                with S.get(self.url, headers=h, stream=True, timeout=(30, 300)) as r:
                    if r.status_code in (401, 403) and not self.auth:
                        with LOCK:
                            self.refresh()
                        raise RuntimeError(f"signed url refused {r.status_code}: re-resolved")
                    if r.status_code != 206:
                        raise RuntimeError(f"HTTP {r.status_code} for range {lo}-{hi}")
                    cr = r.headers.get("content-range", "")
                    assert cr.startswith(f"bytes {lo}-{hi - 1}/"), ("content-range", cr, lo, hi)
                    pos, n = 0, hi - lo
                    while pos < n:
                        k = r.raw.readinto(mv[pos:n])
                        if not k:
                            break
                        pos += k
                    if pos != n:
                        raise RuntimeError(f"short read {pos}/{n}")
                    return n
            except Exception as e:  # noqa: BLE001 (retried, then raised)
                last = e
                time.sleep(min(30, 2 ** att))
        raise RuntimeError(("range read failed", self.path, lo, hi, repr(last)))

    def read(self, lo, hi):
        b = bytearray(hi - lo)
        self.get(lo, hi, memoryview(b))
        return bytes(b)


class RangeFile:
    """minimal seekable read-only file over Src for zipfile (EOCD, central directory, data.pkl), 4 MB read-ahead blocks."""
    BS = 4 << 20

    def __init__(self, src):
        self.s, self.pos, self.cache = src, 0, {}
        assert src.size > 0, ("unknown size", src.path)

    def seekable(self):
        return True

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.s.size + off)
        return self.pos

    def tell(self):
        return self.pos

    def _blk(self, i):
        if i not in self.cache:
            self.cache[i] = self.s.read(i * self.BS, min(self.s.size, (i + 1) * self.BS))
        return self.cache[i]

    def read(self, n=-1):
        end = self.s.size if n is None or n < 0 else min(self.s.size, self.pos + n)
        out = bytearray()
        while self.pos < end:
            i = self.pos // self.BS
            b = self._blk(i)
            o = self.pos - i * self.BS
            take = b[o:o + (end - self.pos)]
            out += take
            self.pos += len(take)
        return bytes(out)


class TDesc:
    __slots__ = ("key", "dtype", "numel", "offset", "size", "stride")

    def __init__(self, key, dtype, numel, offset, size, stride):
        self.key, self.dtype, self.numel, self.offset, self.size, self.stride = key, dtype, numel, offset, size, stride


class _Opaque:
    def __init__(self, *a, **k):
        self.args = (a, k)

    def __setstate__(self, st):
        self.state = st


class StubUnpickler(pickle.Unpickler):
    """data.pkl -> the same nesting, every tensor a TDesc (no tensor bytes read)."""

    def find_class(self, mod, name):
        if (mod, name) == ("torch._utils", "_rebuild_tensor_v2"):
            def rb(storage, storage_offset, size, stride, requires_grad=False, backward_hooks=None, metadata=None):
                key, dtype, numel = storage
                return TDesc(key, dtype, numel, int(storage_offset), tuple(size), tuple(stride))
            return rb
        if (mod, name) == ("collections", "OrderedDict"):
            return collections.OrderedDict
        if mod == "torch" and name.endswith("Storage"):
            return ("STYPE", getattr(torch, name).dtype)
        return type(f"{mod}.{name}", (_Opaque,), {})

    def persistent_load(self, pid):
        typename, stype, key, location, numel = pid
        assert typename == "storage" and isinstance(stype, tuple) and stype[0] == "STYPE", pid
        return (str(key), stype[1], int(numel))


def model_section(path, size):
    """-> (sd['model'] rebuilt as {name: tensor} in pickle order, keep-alive buffers, stats). Only the model records are downloaded."""
    t0 = time.time()
    src = Src(path, size)
    z = zipfile.ZipFile(RangeFile(src))
    infos = z.infolist()
    names = [i.filename for i in infos]
    prefix = names[0].split("/")[0]
    pk = f"{prefix}/data.pkl"
    sd = StubUnpickler(io.BytesIO(z.read(pk))).load()          # zipfile checks data.pkl's CRC
    assert isinstance(sd, dict) and "model" in sd, ("no model section", path, type(sd))
    model = sd["model"]
    assert all(isinstance(v, TDesc) for v in model.values()), "non-tensor entry in model section"
    byname = {i.filename: i for i in infos}
    order = sorted(infos, key=lambda i: i.header_offset)
    nxt = {}
    for i, j in zip(order, order[1:]):
        nxt[i.filename] = j.header_offset
    cd_start = z.start_dir
    nxt[order[-1].filename] = cd_start
    keys = []
    for d in model.values():
        if d.key not in keys:
            keys.append(d.key)
    recs = [byname[f"{prefix}/data/{k}"] for k in keys]
    for i in recs:
        assert i.compress_type == 0, ("compressed record", i.filename)
    # spans [header_offset, next header_offset) merged when adjacent (the model storages are written first, back to back)
    sp = []
    for i in sorted(recs, key=lambda i: i.header_offset):
        lo, hi = i.header_offset, nxt[i.filename]
        if sp and lo - sp[-1][1] <= (1 << 20):
            sp[-1][1] = hi
            sp[-1][2].append(i)
        else:
            sp.append([lo, hi, [i]])
    bufs, nbytes = [], 0
    CH = a.chunk_mb << 20
    with ThreadPoolExecutor(a.threads) as ex:
        futs = []
        for lo, hi, _ in sp:
            lo64 = lo - (lo % 64)                     # buffer start == file offset mod 64 -> the 64-aligned records stay aligned in RAM
            buf = torch.empty(hi - lo64, dtype=torch.uint8)
            mv = memoryview(buf.numpy()).cast("B")
            bufs.append((lo64, hi, buf))
            for c in range(lo64, hi, CH):
                e = min(hi, c + CH)
                futs.append(ex.submit(src.get, c, e, mv[c - lo64:e - lo64]))
        for f in futs:
            nbytes += f.result()
    t_fetch = time.time() - t0
    recview = {}
    crc_ok = 0
    for (lo64, hi, buf), (_, _, rl) in zip(bufs, sp):
        npb = buf.numpy()
        for i in rl:
            o = i.header_offset - lo64
            hdr = bytes(npb[o:o + 30])
            assert hdr[:4] == b"PK\x03\x04", ("bad local header", i.filename)
            nl, xl = int.from_bytes(hdr[26:28], "little"), int.from_bytes(hdr[28:30], "little")
            assert bytes(npb[o + 30:o + 30 + nl]).decode() == i.filename, ("local header name", i.filename)
            d = o + 30 + nl + xl
            assert d + i.file_size <= hi - lo64, ("record overruns span", i.filename)
            if zlib.crc32(npb[d:d + i.file_size]) != i.CRC:
                raise AssertionError(("CRC32 MISMATCH", path, i.filename))
            crc_ok += 1
            recview[i.filename] = (buf, d, i.file_size)
    out = {}
    for n, dsc in model.items():
        buf, d, sz = recview[f"{prefix}/data/{dsc.key}"]
        isz = torch.empty((), dtype=dsc.dtype).element_size()
        assert sz == dsc.numel * isz, ("record size != numel x itemsize", n, sz, dsc.numel, isz)
        if dsc.numel:
            assert d % isz == 0 and (buf.data_ptr() + d) % isz == 0, ("misaligned record", n)
            base = buf[d:d + sz].view(dsc.dtype)
            if all(s > 0 for s in dsc.size):
                assert dsc.offset + sum((s - 1) * st for s, st in zip(dsc.size, dsc.stride)) < dsc.numel, ("view out of bounds", n)
            t = torch.empty((0,), dtype=dsc.dtype)
            t.set_(base.untyped_storage(), base.storage_offset() + dsc.offset, dsc.size, dsc.stride)   # = torch._utils._rebuild_tensor
        else:
            t = torch.empty((0,), dtype=dsc.dtype)
            t.set_(t.untyped_storage(), dsc.offset, dsc.size, dsc.stride)
        out[n] = t
    st = {"file": path, "file_size": src.size, "zip_records": len(infos), "model_tensors": len(model), "model_records": len(recs),
          "spans": len(sp), "fetched_bytes": nbytes, "fetch_s": round(t_fetch, 1), "fetch_MBps": round(nbytes / max(t_fetch, 1e-6) / 1e6, 1),
          "crc_ok": crc_ok, "other_sections": [k for k in sd if k != "model"]}
    return out, bufs, st


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(1 << 24)
            if not b:
                return h.hexdigest()
            h.update(b)


def finite_bad(out):
    return [n for n, t in out.items() if t.numel() and not bool(torch.isfinite(t).all())]


def compare_ref(out, refp):
    ref = torch.load(refp, map_location="cpu", mmap=True, weights_only=True)
    r = {"keys_same_order": list(ref) == list(out), "n_ref": len(ref), "n_new": len(out)}
    bad = []
    for n, t in out.items():
        u = ref.get(n)
        if u is None or u.dtype != t.dtype or tuple(u.shape) != tuple(t.shape) or not torch.equal(u, t):
            bad.append(n)
    r["not_equal"] = bad[:10]
    r["n_not_equal"] = len(bad)
    r["equal_all"] = r["keys_same_order"] and not bad and len(ref) == len(out)
    del ref
    return r


def main():
    t00 = time.time()
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "2")))
    api = HfApi()
    files = {}
    for e in api.list_repo_tree(REPO, path_in_repo=TAG, repo_type="model", revision=REV, recursive=True):
        if getattr(e, "size", None) is not None:
            files[e.path] = int(e.size)
    rk = re.compile(rf"^{re.escape(TAG)}/([^/]+)/rank(\d+)\.pt$")
    dirs = sorted({rk.match(p).group(1) for p in files if rk.match(p)})           # host dirs = dirs holding rank files (not run_state/ etc.)
    d0 = [rk.match(p).group(1) for p in files if rk.match(p) and rk.match(p).group(2) == "0"]
    assert len(d0) == 1, ("rank0.pt location", d0)
    stats = {"repo": REPO, "revision": REV, "tag": TAG, "out": a.out, "alias": ALIAS, "hf_dirs": dirs, "items": {}, "job": os.environ.get("SLURM_JOB_ID"),
             "host": os.uname().nodename, "torch": torch.__version__}
    # train_state.json of rank 0's host dir (the stock tool reads its own host's copy; rank 0's host writes meta.json)
    ts_dir = d0[0]
    ts = json.loads(Src(f"{TAG}/{ts_dir}/train_state.json").read(0, files[f"{TAG}/{ts_dir}/train_state.json"]))
    hdir = [ALIAS.get(h, h) for h in ts["hosts"]]
    assert hdir[0] == ts_dir, ("rank 0's logical host does not map to the dir holding rank0.pt", ts["hosts"][0], ts_dir)
    hosts, step, world = ts["hosts"], int(ts["step"]), int(ts["world"])
    assert world == len(hosts) == 128 and TAG == f"dcp-{step}", (world, len(hosts), step, TAG)
    for r in range(world):                                           # every rank file in exactly its (aliased) host dir
        where = [d for d in dirs if f"{TAG}/{d}/rank{r}.pt" in files]
        assert where == [hdir[r]], ("rank file location", r, hosts[r], hdir[r], where)
    for d in sorted(set(hdir)):                                      # source save COMPLETE on every dir we read (stock: per node)
        c = Src(f"{TAG}/{d}/COMPLETE").read(0, files[f"{TAG}/{d}/COMPLETE"]).decode().strip()
        assert c == str(step), ("source dir not COMPLETE for this step", d, c)
    assert f"{TAG}/{ts_dir}/rep.pt" in files, "rep.pt missing in rank 0's dir"
    stats.update({"step": step, "world": world, "train_state_dir": ts_dir, "rank_dirs": dict(enumerate(hdir))})
    log(f"{REPO}@{REV[:10]} {TAG}: step {step} world {world}, {len(dirs)} dirs, rank->dir verified (alias {ALIAS}); out {a.out}")
    mans = {}
    if a.manifest_dir:
        for f in sorted(os.listdir(a.manifest_dir)):
            if f.startswith("manifest-") and f.endswith(".json"):
                m = json.load(open(f"{a.manifest_dir}/{f}"))
                assert int(m["step"]) == step, ("manifest step", f)
                for fn, v in m["files"].items():
                    mans[fn] = v
        log(f"manifests: {len(mans)} file records")
    os.makedirs(a.out, exist_ok=True)
    ranks = list(range(world)) if a.ranks == "all" else [int(x) for x in a.ranks.split(",")]
    free = shutil.disk_usage(a.out).free / 1e9
    need = len(ranks) * 0.68 + (2.4 if a.rep else 0) + 3
    assert free - need >= a.min_free_gb, (f"/data free {free:.0f} GB: writing ~{need:.0f} GB would go below the {a.min_free_gb} GB floor")
    items = [(f"rank{r}.pt", f"{TAG}/{hdir[r]}/rank{r}.pt") for r in ranks] + ([("rep.pt", f"{TAG}/{ts_dir}/rep.pt")] if a.rep else [])
    tot_fetch = tot_out = 0
    pre = ThreadPoolExecutor(1)                                     # prefetch item i+1 while item i is cast / saved
    fut = pre.submit(model_section, items[0][1], files[items[0][1]])
    for k, (fn, hp) in enumerate(items):
        sdm, bufs, st = fut.result()
        if k + 1 < len(items):
            fut = pre.submit(model_section, items[k + 1][1], files[items[k + 1][1]])
        t1 = time.time()
        out = {n: t.to(torch.bfloat16).contiguous().clone() for n, t in sdm.items()}
        del sdm, bufs
        bad = finite_bad(out)
        if bad:
            raise AssertionError(("NON-FINITE tensors", fn, bad[:8]))
        nb = sum(t.numel() * 2 for t in out.values())
        if fn == "rep.pt":
            torch.save(out, f"{a.out}/rep.pt")
        else:
            torch.save(out, f"{a.out}/{fn}.tmp")
            os.replace(f"{a.out}/{fn}.tmp", f"{a.out}/{fn}")
        st.update({"bf16_bytes": nb, "n_keys": len(out), "non_bf16": [n for n, t in out.items() if t.dtype != torch.bfloat16][:5],
                   "finite": True, "out_size": os.path.getsize(f"{a.out}/{fn}"), "sha256": sha256(f"{a.out}/{fn}"),
                   "rows": {n: int(t.shape[0]) for n, t in out.items()} if mans else None})
        if mans:
            mv = mans.get(fn)
            st["manifest_sha_equal"] = bool(mv and mv["sha256"] == st["sha256"])
            st["manifest_size_equal"] = bool(mv and mv["size"] == st["out_size"])
            if mv and "rows" in mv:
                st["manifest_rows_equal"] = mv["rows"] == st["rows"]
            if a.manifest_must_match and not st["manifest_sha_equal"]:
                raise AssertionError(("sha256 != the 10-22 conversion's manifest", fn, st["sha256"], mv))
        st.pop("rows", None)
        if a.ref:
            st["ref"] = compare_ref(out, f"{a.ref}/{fn}")
            st["ref"]["sha_equal"] = sha256(f"{a.ref}/{fn}") == st["sha256"]
            st["ref"]["size_equal"] = os.path.getsize(f"{a.ref}/{fn}") == st["out_size"]
        st["cast_save_s"] = round(time.time() - t1, 1)
        tot_fetch += st["fetched_bytes"]
        tot_out += st["out_size"]
        stats["items"][fn] = st
        el = time.time() - t00
        log(f"{fn}: {st['model_tensors']} tensors, fetched {st['fetched_bytes'] / 1e9:.2f} GB of {st['file_size'] / 1e9:.2f} GB "
            f"({st['fetch_MBps']:.0f} MB/s, crc ok {st['crc_ok']}), cast+save {st['cast_save_s']}s, sha {st['sha256'][:12]}"
            + (f", manifest sha {'==' if st.get('manifest_sha_equal') else '!='}" if mans else "")
            + (f", ref equal {st['ref']['equal_all']} sha {st['ref']['sha_equal']}" if a.ref else "")
            + f" | {k + 1}/{len(items)} done, {tot_fetch / 1e9:.1f} GB fetched, avg {tot_fetch / el / 1e6:.0f} MB/s, {el:.0f}s")
        del out
        json.dump(stats, open(a.stats + ".tmp", "w"), indent=1)
        os.replace(a.stats + ".tmp", a.stats)
    if a.meta:                                                      # = ckpt_local_export.py's meta.json, src = the HF path
        json.dump({"format": "ppm-local-bf16-shards-v1", "src": f"hf://{REPO}@{REV}/{TAG}", "step": step, "world": world, "hosts": hosts,
                   "run_tag": ts["cfg"].get("run_tag"), "run_base": ts["cfg"].get("run_base"), "cfg": ts["cfg"], "save_time": ts.get("time"),
                   "made": time.strftime("%Y-%m-%d %H:%M:%S %Z")}, open(f"{a.out}/meta.json", "w"), indent=1)
        if a.ref:
            mr, mn = json.load(open(f"{a.ref}/meta.json")), json.load(open(f"{a.out}/meta.json"))
            stats["meta_ref"] = {"keys_same_order": list(mr) == list(mn),
                                 "differ": [k for k in mr if mr.get(k) != mn.get(k)]}
    stats["total_fetched_bytes"] = tot_fetch
    stats["total_out_bytes"] = tot_out
    stats["seconds"] = round(time.time() - t00, 1)
    stats["avg_MBps"] = round(tot_fetch / max(stats["seconds"], 1e-6) / 1e6, 1)
    json.dump(stats, open(a.stats + ".tmp", "w"), indent=1)
    os.replace(a.stats + ".tmp", a.stats)
    log(f"DONE {len(items)} files, fetched {tot_fetch / 1e9:.1f} GB, wrote {tot_out / 1e9:.1f} GB, {stats['seconds']:.0f}s ({stats['avg_MBps']:.0f} MB/s)")


if __name__ == "__main__":
    main()
