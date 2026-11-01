import concurrent.futures,json,time,os
from pathlib import Path
from huggingface_hub import hf_hub_download
BASE=Path('/home/param/datasets/blackband-repro-20261031')
spec=json.loads((BASE/'source-files.json').read_text());start=time.time()
def fetch(f):
    p=Path(hf_hub_download('InternalCan/stage2-handover-pose-mimi-sam-v1',f['path'],repo_type='dataset',revision=spec['revision'],local_dir=BASE/'source'))
    assert p.stat().st_size==f['size']
    return f['size']
done=0;size=0
with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
    for future in concurrent.futures.as_completed([pool.submit(fetch,f) for f in spec['files']]):
        size+=future.result();done+=1
        row={'state':'downloading' if done<len(spec['files']) else 'done','pid':os.getpid(),'completed_shards':done,'total_shards':len(spec['files']),'completed_bytes':size,'elapsed_seconds':time.time()-start,'updated':time.time()}
        (BASE/'fetch-status.json').write_text(json.dumps(row)+'\n')
        if done%20==0 or done==len(spec['files']):print(json.dumps(row),flush=True)
