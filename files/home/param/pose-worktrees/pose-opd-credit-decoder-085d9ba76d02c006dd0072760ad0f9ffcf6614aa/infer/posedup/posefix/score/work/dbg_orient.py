import json, sys, numpy as np, torch
sys.path.insert(0, '/data/home/amartyavarma/ppm/aloe-posedup-live/posefix/score')
import score as S
R = json.load(open('/data/home/amartyavarma/ppm/aloe-posedup-live/posefix/score/out/sim.json'))
bad = [r for r in R if r['filters'] == 'align+orient' and any(abs(r['whole'].get(k) or 0) > 10 for k in ('pitch_sd_slow', 'yaw_sd_slow', 'pitch_p50'))]
print('bad', [(r['name'], r['whole']['pitch_sd_slow'], r['whole']['yaw_sd_slow']) for r in bad])
for r in bad[:2]:
    p = r['info']['src']
    codes, _ = S.load_codes(p, 'pose')
    f0 = S.frame0_of(S.DEFAULT_PERSON)
    dec = S.decode(codes)
    kp, kp0 = S.to_keypoints(dec, f0)
    xy = kp.numpy()[:, :5, :2]
    ed = np.linalg.norm(xy[:, 1] - xy[:, 2], axis=-1)
    conf = kp.numpy()[:, :5, 2]
    i = np.argsort(ed)[:5]
    print(p, 'raw eye dist px min', ed.min(), 'p1', np.percentile(ed, 1), 'median', np.median(ed), 'frames', i, 'conf there', conf[i].round(2).tolist())
    ko = S.apply_filters(kp, f0, 'orient')
    xo = ko.numpy()[:, :5, :2]
    big = np.where(np.abs(xo).max((1, 2)) > 2000)[0]
    print('  orient-only: frames with head coords > 2000 px:', len(big), big[:10], 'max', np.abs(xo).max())
    if len(big):
        b = big[0]
        print('  raw head at first bad frame', xy[b].round(1).tolist(), 'raw ed', ed[max(0, b - 50):b + 1].min())
