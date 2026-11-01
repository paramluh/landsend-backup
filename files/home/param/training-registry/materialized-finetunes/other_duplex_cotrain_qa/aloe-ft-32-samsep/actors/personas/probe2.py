import pyarrow.parquet as pq, pyarrow.compute as pc, json, random, collections, glob
f="/data/home/amartyavarma/qwen-1ch-data/persona/processed/r1/s0003/slinky-5.parquet"
t=pq.read_table(f,columns=['ek','side','chunk_index','status','flags_json','sections_json'])
t=t.filter(pc.equal(t['status'],'ok'))
random.seed(1)
idx=random.sample(range(t.num_rows),40)
for i in idx:
    fl=json.loads(t['flags_json'][i].as_py() or '{}'); s=json.loads(t['sections_json'][i].as_py())
    print(t['ek'][i].as_py()[:1], t['side'][i].as_py(), '|', (fl.get('channel_check') or {}).get('a_speaker'), '|', s.get('starting_context','')[:160].replace('\n',' '))
print('item keys', s['persona_items'][:2])
A="/data/home/amartyavarma/data/aloe-handover-full/DATA/files/none/00024c62-133e-4115-817e-d3f3a5416e47/seg00/system_prompts/SPEAKER_01/episode.json"
E=json.load(open(A))
print('EP keys', list(E.keys()))
for k,v in E.items():
    if k!='profile': print(k, '=>', json.dumps(v)[:400])
print('profile keys', list(E['profile'].keys()))
