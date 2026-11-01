import pyarrow.parquet as pq, json, random
f="/data/home/amartyavarma/qwen-1ch-data/persona/processed/r1/s0000/slinky-0.parquet"
t=pq.read_table(f)
print(t.schema); print(t.num_rows)
d=t.filter(__import__('pyarrow.compute',fromlist=['x']).equal(t['status'],'ok')).slice(0,400).to_pylist()
import collections
print(collections.Counter(r['ek'].split('/')[0] for r in d))
for r in d[:1]+[x for x in d if x['chunk_index']>0][:1]+[x for x in d if x['ek'].startswith('B')][:1]:
    print('=====',r['ek'],r['side'],r['chunk_index'],r['version'],r['prompt_tokens'])
    print(r['system_prompt'][:3000])
    print('--sections keys', list(json.loads(r['sections_json']).keys()))
    print('--tokens', r['tokens_json'])
    print('--flags', r['flags_json'][:500])
