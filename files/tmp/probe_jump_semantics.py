import json,sys
from pathlib import Path
import numpy as np
cases=[('v1',Path('/home/param/runs/pose-runs/pose-sites-20261030/video-inspection/v1-19-21/pose_overlay.json'),512),('145',Path('/home/param/runs/pose-runs/pose-jump-audit-20261031-nocfg/positive-qa/runs/rollout-000145/pose_overlay.json'),474)]
for name,p,f in cases:
 a=np.array(json.loads(p.read_text())['frames']);xy=a[:,:,:2]*[256,480];v=np.diff(xy,axis=0)
 print(name)
 print('frame nose_x,y headcentroid_x,y shouldercentroid_x,y HEADspeed SHOULDERspeed')
 for i in range(f-12,f+16):
  print(i,*np.round(np.r_[xy[i,0],xy[i,:5].mean(0),xy[i,5:7].mean(0),np.median(np.linalg.norm(v[i-1,:5],axis=1)),np.mean(np.linalg.norm(v[i-1,5:7],axis=1))],2))
 for w in [3,5,8,12,20]:
  # extrapolate independent linear trajectories to boundary half-frame
  pre=np.arange(f-w,f);post=np.arange(f,f+w)
  preB=np.c_[np.ones(w),pre-(f-.5)];postB=np.c_[np.ones(w),post-(f-.5)]
  prefit=np.linalg.lstsq(preB,xy[pre].reshape(w,-1),rcond=None)[0];postfit=np.linalg.lstsq(postB,xy[post].reshape(w,-1),rcond=None)[0]
  gap=(postfit[0]-prefit[0]).reshape(55,2);orig=v[f-1]
  print('linear',w, 'median head/gap',np.median(np.linalg.norm(gap[:5],axis=1)), 'shoulders',np.mean(np.linalg.norm(gap[5:7],axis=1)), 'gap/t0 head',np.median(np.linalg.norm(gap[:5],axis=1)/np.linalg.norm(orig[:5],axis=1)), 'project head',np.median(np.sum(gap[:5]*orig[:5],axis=1)/np.sum(orig[:5]**2,axis=1)))
