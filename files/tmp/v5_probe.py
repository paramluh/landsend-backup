import json,numpy as np
from pathlib import Path
cases=[('v1','/home/param/runs/pose-runs/pose-sites-20261030/video-inspection/v1-19-21/pose_overlay.json',512),('145','/home/param/runs/pose-runs/pose-jump-audit-20261031-nocfg/positive-qa/runs/rollout-000145/pose_overlay.json',474),('132','/home/param/runs/pose-runs/pose-jump-audit-20261031-nocfg/positive-qa/runs/rollout-000132/pose_overlay.json',170),('968','/home/param/runs/pose-runs/pose-jump-audit-20261031-xabi-nocfg/positive-qa/runs/rollout-000968/pose_overlay.json',142),('642','/home/param/runs/pose-runs/pose-jump-audit-20261031-xabi-nocfg/positive-qa/runs/rollout-000642/pose_overlay.json',284)]
for name,path,f in cases:
 p=np.array(json.loads(Path(path).read_text())['frames']);xy=p[:,:,:2]*[256,480];v=np.diff(xy,axis=0)
 print(name)
 for d in [1,2]:
  end=f+d-1;pre=v[f-9:f-1];post=v[end+2:end+12];drift=np.median(pre,axis=0)
  delta=xy[end]-xy[f-1]-d*drift;norm=np.linalg.norm(delta,axis=-1);u=delta/np.maximum(norm[:,None],1e-8)
  q=np.percentile(np.linalg.norm(pre-drift,axis=-1),90,axis=0)/np.maximum(norm,.3)
  qraw=np.percentile(np.linalg.norm(pre,axis=-1),90,axis=0)/np.maximum(norm,.3)
  r=np.percentile(np.linalg.norm(post-drift,axis=-1),90,axis=0)/np.maximum(norm,.3)
  n=np.arange(end+6,end+13);res=xy[n]-xy[f-1]-(n-f+1)[:,None,None]*drift
  proj=(res*u).sum(-1)/np.maximum(norm,.3)
  keep=np.percentile(proj,10,axis=0)
  print('duration',d,'head',np.round([np.median(norm[:5]),np.median(qraw[:5]),np.median(r[:5]),np.median(keep[:5])],2),'S5',np.round([norm[5],qraw[5],r[5],keep[5]],2),'S6',np.round([norm[6],qraw[6],r[6],keep[6]],2))
