import pyarrow as pa, pyarrow.ipc as ipc, json
r="/data/home/amartyavarma/ppm/aloe-pose-ft/stores/p0/rows-00000-of-00001.words.arrow"
try:
    t=ipc.open_file(r).read_all()
except Exception:
    t=ipc.open_stream(open(r,"rb")).read_all()
print(t.schema); d=t.slice(0,1).to_pylist()[0]; print({k:(str(v)[:300]) for k,v in d.items()})
