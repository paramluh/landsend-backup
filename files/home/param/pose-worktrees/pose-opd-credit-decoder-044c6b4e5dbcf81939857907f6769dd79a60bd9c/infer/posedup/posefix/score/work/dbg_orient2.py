import sys, numpy as np
sys.path.insert(0, '/data/home/amartyavarma/ppm/aloe-posedup-live/posefix/score')
import score as S
for n in ('c2s0_pf0', 'c3s1_pf0', 'c1s0_pf0'):
    codes, _ = S.load_codes(f'/data/home/amartyavarma/ppm/aloe-posedup-live/posefix/out/conv/{n}.npz', 'pose')
    f0 = S.frame0_of(S.DEFAULT_PERSON)
    kp, kp0 = S.to_keypoints(S.decode(codes), f0)
    for flt in ('none', 'align', 'align+orient'):
        k = S.apply_filters(kp, f0, flt).numpy()[:, :5]
        ed = np.linalg.norm(k[:, 1, :2] - k[:, 2, :2], axis=-1)
        b = np.where(ed < 1.0)[0]
        print(n, flt, 'frames eye dist < 1 px:', len(b), b[:6], 'eyes there', k[b[:2], 1:3, :2].round(1).tolist() if len(b) else '', 'nose', k[b[:2], 0, :2].round(1).tolist() if len(b) else '')
