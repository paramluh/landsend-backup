import pyarrow.parquet as pq, sys
t = pq.read_table(sys.argv[1]); print(t.schema); print(t.num_rows); print(t.slice(0, 2).to_pylist())
