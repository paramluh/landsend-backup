import json, numpy as np, glob, os
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
d = '/data/home/amartyavarma/ppm/aloe-ft-32-samsep/poseduplex/samples/runs/B-000'
print(open(d + '/meta.json').read()[:3000])
for f in ('pose_s1.npy', 'gt_pose_s1.npy', 'pose_s2.npy'):
    a = np.load(d + '/' + f); print(f, a.shape, a.dtype, a.min(), a.max())
l = np.load(d + '/lanes.npz')
for k in l.files: print('lanes', k, l[k].shape)
print(open(d + '/pose_render.json').read()[:1500])
