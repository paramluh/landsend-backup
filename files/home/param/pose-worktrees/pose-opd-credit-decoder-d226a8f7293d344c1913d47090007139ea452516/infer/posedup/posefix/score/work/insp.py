import json, numpy as np, glob, os
from safetensors.numpy import load_file
pix = json.load(open('/data/home/amartyavarma/ppm/aloe-ft-32-samsep/poseduplex/data/POSE_INDEX.json'))
rows = pix['rows']
k0 = next(iter(rows)); r0 = next(iter(rows[k0]))
print(json.dumps(rows[k0][r0])[:1500])
ev = [(k, r, v) for k, d in rows.items() for r, v in d.items() if v['split'] == 'eval']
print('eval rows', len(ev), 'convs', len({v['conv'] for _, _, v in ev}))
print(sorted({v['row_key'].split('/')[-1] for _,_,v in ev}))
z = np.load('/data/home/amartyavarma/ppm/aloe-posedup-live/sessions/20261025-142135-d545/session.npz')
for k in z.files: print('sess', k, z[k].shape, z[k].dtype)
p = z['pose']; print('pose -1 rows', (p < 0).any(1).sum(), 'first rows', p[:3, :4])
c = np.load('/data/home/amartyavarma/ppm/aloe-ft-32-samsep/poseduplex/samples/convs/c0.npz')
for k in c.files:
    if 'pdep' in k or 'pose' in k: print('conv', k, c[k].shape, c[k].dtype, (c[k][:,0] >= 0).sum())
g = load_file('/data/home/amartyavarma/video/demo-data/prompts/01_ab63d745/gt/pose_tokens.safetensors')
for k, v in g.items(): print('gt', k, v.shape, v.dtype, v.min(), v.max())
l = load_file('/data/home/amartyavarma/video/demo-data/prompts/01_ab63d745/llm_inputs.safetensors')
for k, v in l.items(): print('llm', k, v.shape, v.dtype)
print(l['frame0_pose'][:13])
print(open('/data/home/amartyavarma/video/demo-data/prompts/01_ab63d745/meta.json').read()[:800])
for d in sorted(glob.glob('/data/home/amartyavarma/ppm/aloe-ft-32-samsep/poseduplex/samples/runs/B-00*'))[:2]:
    for f in sorted(os.listdir(d)): print(d, f)
