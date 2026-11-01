import os, sys, time, subprocess
sys.path.insert(0, '/data/home/amartyavarma/ppm/aloe-posedup-live/posefix/score')
import numpy as np, torch
torch.set_num_threads(int(os.environ.get('NT', '16')))
import score as S
print('cuda', torch.cuda.is_available(), os.environ.get('CUDA_VISIBLE_DEVICES'), flush=True)
if torch.cuda.is_available():
    print(subprocess.run(['nvidia-smi', '--query-gpu=index,name,clocks.sm,temperature.gpu', '--format=csv'], capture_output=True, text=True).stdout)
c, info = S.load_codes('/data/home/amartyavarma/ppm/aloe-posedup-live/sessions/20261025-142135-d545'); print(info, c.shape, (c < 0).any(1).sum())
c2, info2 = S.load_codes('/data/home/amartyavarma/ppm/aloe-posedup-live/sessions/20261025-142135-d545/feed.jsonl'); print(info2, c2.shape, (c2 < 0).any(1).sum(), 'feed==npz', np.array_equal(c2[(c2>=0).all(1)], c[(c>=0).all(1)]))
c3, _ = S.load_codes('/data/home/amartyavarma/ppm/aloe-posedup-live/stage-lb/instB/sessions/20261025-140015-5d7e/feed.jsonl'); print('5d7e feed', c3.shape, (c3 < 0).any(1).sum(), np.where((c3<0).any(1))[0][:10])
a = np.load('/data/home/amartyavarma/data/aloe-handover-full/DATA/files/none/12a30b79-c174-4bc8-bb98-bdd0585b5491/seg00/w00.SPEAKER_00.pose_tokens.npy'); print('aloe', a.shape, a.dtype)
cl = S._fix_codes(a)[:7500]
for dev in ['cpu'] + (['cuda'] if torch.cuda.is_available() else []):
    for name, x in (('sess', c[(c >= 0).all(1)]), ('aloe7500', cl)):
        S.decode(x[:50], dev)
        t = time.time(); d = S.decode(x, dev); print(dev, name, x.shape, d.shape, f'{time.time() - t:.2f}s', flush=True)
        np.save(f'/data/home/amartyavarma/ppm/aloe-posedup-live/posefix/score/work/dec_{dev}_{name}.npy', d.numpy())
if torch.cuda.is_available():
    for name in ('sess', 'aloe7500'):
        a1 = np.load(f'/data/home/amartyavarma/ppm/aloe-posedup-live/posefix/score/work/dec_cpu_{name}.npy'); a2 = np.load(f'/data/home/amartyavarma/ppm/aloe-posedup-live/posefix/score/work/dec_cuda_{name}.npy')
        print(name, 'cpu vs cuda max abs', np.abs(a1 - a2).max(), 'mean abs', np.abs(a1 - a2).mean(), 'offset rms', np.sqrt((a1.reshape(-1,55,3)[...,:2]**2).mean()))
