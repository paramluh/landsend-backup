"""C1 probe (read-only on HF): list dcp-17641 / dcp-17850 in InternalCan/ppm-v1-c, open one rank file by range reads, parse its zip
central directory + data.pkl, time a range fetch. Never prints the token."""
import io, json, sys, time, zipfile, zlib
from huggingface_hub import HfApi, HfFileSystem
REPO = "InternalCan/ppm-v1-c"
api = HfApi()
t0 = time.time()
info = api.repo_info(REPO, repo_type="model")
print("repo sha", info.sha, "private", info.private, flush=True)
for tag in ("dcp-17641", "dcp-17850"):
    try:
        dirs = [e.path for e in api.list_repo_tree(REPO, path_in_repo=tag, repo_type="model")]
        print(tag, len(dirs), dirs[:20], flush=True)
    except Exception as e:
        print(tag, "LIST FAIL", repr(e)[:200], flush=True)
top = [e.path for e in api.list_repo_tree(REPO, repo_type="model")]
print("top", top, flush=True)
fs = HfFileSystem()
p = f"{REPO}/dcp-17850/slinky-0/rank0.pt"
t1 = time.time()
f = fs.open(p, "rb", block_size=1 << 20)
z = zipfile.ZipFile(f)
il = z.infolist()
print("zip entries", len(il), "open+cd %.1fs" % (time.time() - t1), flush=True)
for i in il[:6] + il[-4:]:
    print("  ", i.filename, i.compress_type, i.file_size, i.header_offset, flush=True)
pk = [i for i in il if i.filename.endswith("data.pkl")][0]
t1 = time.time(); b = z.read(pk.filename); print("data.pkl", len(b), "%.1fs" % (time.time() - t1), flush=True)
# range fetch timing: the 3 largest records, direct cat_file
big = sorted(il, key=lambda i: -i.file_size)[:3]
for i in big:
    t1 = time.time()
    lh = fs.cat_file(p, start=i.header_offset, end=i.header_offset + 30)
    nlen = int.from_bytes(lh[26:28], "little"); xlen = int.from_bytes(lh[28:30], "little")
    d0 = i.header_offset + 30 + nlen + xlen
    data = fs.cat_file(p, start=d0, end=d0 + i.compress_size)
    dt = time.time() - t1
    print("  rec", i.filename, len(data), "crc_ok", zlib.crc32(data) == i.CRC, "%.1f MB/s" % (len(data) / dt / 1e6), flush=True)
print("total %.1fs" % (time.time() - t0))
