"""crosscheck_b.py OUR_ROOT SHARD...: our slinky-11 sidecars vs conv-pool-v2 'b-' sidecars on slinky-11 (/scratch/amartyavarma/conv-pool-v2/rows/
b-<shard>.persona.arrow, read-only): row_key + *_status must be identical; also reports which other columns differ (row counts)."""
import sys
from collections import Counter

import pyarrow as pa
import pyarrow.ipc as ipc

our, shards = sys.argv[1], sys.argv[2:]
rd = lambda p: ipc.open_file(pa.memory_map(p, "r")).read_all()  # noqa: E731
same = lambda x, y: x == y or (isinstance(x, float) and isinstance(y, float) and x != x and y != y)  # noqa: E731
key_diff, other_diff, rows = Counter(), Counter(), 0
for s in shards:
    A, Bt = rd(f"{our}/{s}.persona.arrow"), rd(f"/scratch/amartyavarma/conv-pool-v2/rows/b-{s}.persona.arrow")
    assert A.schema.equals(Bt.schema), ("schema differs", s)
    assert A.num_rows == Bt.num_rows, ("row count differs", s, A.num_rows, Bt.num_rows)
    rows += A.num_rows
    for c in A.column_names:
        nd = sum(1 for x, y in zip(A[c].to_pylist(), Bt[c].to_pylist()) if not same(x, y))
        if nd:
            (key_diff if (c == "row_key" or c.endswith("_status")) else other_diff)[c] += nd
print(f"crosscheck_b {len(shards)} shards {rows} rows: row_key/*_status diffs: {dict(key_diff) or 'NONE'}; other column diffs: {dict(other_diff) or 'NONE'}")
