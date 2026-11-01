"""verify_store.py ROOT SLOT [PERSONA_ROOT]: s3rows load check (NODE.json node == SLOT, check_row on first + last row); with PERSONA_ROOT
also: every sidecar row_key column == that shard's words.arrow row_key column (in order) + PersonaSidecar.get on first + last row."""
import sys
sys.path.insert(0, '/data/home/amartyavarma/ppm/code-811/lib')
import s3rows as R3  # noqa: E402

root, slot = sys.argv[1], sys.argv[2]
n = R3.load_node(root)
assert n["node"] == slot, (n["node"], slot)
r = R3.Rows(n)
N = len(r)
R3.check_row(r[0]); R3.check_row(r[N - 1])
msg = f"load ok node={n['node']} rows={N} shards={len(n['shards'])}"
if len(sys.argv) > 3:
    import pyarrow as pa
    import pyarrow.ipc as ipc
    import persona_s3q as PER
    pr = sys.argv[3]
    for s in n["shards"]:
        wk = ipc.open_file(pa.memory_map(f"{root}/{s}.words.arrow", "r")).read_all().column("row_key").to_pylist()
        pk = ipc.open_file(pa.memory_map(PER.sidecar_path(pr, s), "r")).read_all().column("row_key").to_pylist()
        assert wk == pk, ("persona row_key != words.arrow row_key", s, len(wk), len(pk))
    ps = PER.PersonaSidecar(n, root=pr)
    g0, g1 = ps.get(0, r[0]["row_key"]), ps.get(N - 1, r[N - 1]["row_key"])
    msg += f"; persona row_key == words.arrow for {len(n['shards'])} shards, reader ok (row0 status {g0['user']['status']}/{g0['assistant']['status']})"
print(msg)
