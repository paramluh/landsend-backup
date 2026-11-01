import numpy as np, glob, json
from safetensors.numpy import load_file
g = load_file('/data/home/amartyavarma/video/demo-data/prompts/01_ab63d745/gt/pose_tokens.safetensors')['codes'].T.astype(np.int64)
for p in sorted(glob.glob('/data/home/amartyavarma/ppm/aloe-posedup-live/posefix/out/*/*.npz'))[:20]:
    z = np.load(p); a = z['pose']; j = json.load(open(p[:-4] + '.json'))
    neg = np.where((a < 0).any(1))[0]
    pf = j['pose_prefill_frames']
    m = None
    if pf:
        # where do the prefill tokens match the actor GT?
        for off in range(0, 4):
            seg = a[off:off + min(pf, 150)]
            if len(seg) and np.array_equal(seg, g[:len(seg)]): m = off
    print(p.split('/')[-2:], z.files, a.shape, a.dtype, 'neg rows', neg[:6], len(neg), 'pf', pf, 'gt-match offset', m, a.max())
c = np.load('/data/home/amartyavarma/ppm/aloe-ft-32-samsep/poseduplex/samples/convs/c0.npz')
gp = c['gtpose_0'].T.astype(np.int64); s = np.load('/data/home/amartyavarma/ppm/aloe-posedup-live/posefix/out/conv/c0s0_pf0.npz')['pose']
print('c0 gtpose_0', gp.shape, 'sim', s.shape, 'first rows equal', [np.array_equal(gp[i], s[i]) for i in range(4)])
