exec(open('/tmp/v5_probe.py').read().split('for name,path,f in cases:')[0])
cases.append(('243','/home/param/runs/pose-runs/pose-jump-audit-20261031-nocfg/positive-qa/runs/rollout-000243/pose_overlay.json',292))
for name,path,f in cases:
 p=np.array(json.loads(Path(path).read_text())['frames']);xy=p[:,:,:2]*[256,480];v=np.diff(xy,axis=0)
 print(name)
 for d in [1,2]:
  end=f+d-1;pre=v[f-9:f-1];post=v[end+2:end+12];drift=np.median(pre,axis=0)
  delta=xy[end]-xy[f-1];delta_det=delta-d*drift
  n=np.arange(end+6,end+13);res=xy[n]-xy[f-1];res_det=res-(n-f+1)[:,None,None]*drift
  for g,ids in [('H',list(range(5))),('S',[5,6])]:
   norm=np.linalg.norm(delta[ids]);u=delta[ids]/norm
   proj=(res[:,ids]*u).sum((1,2))/norm;proj_det=(res_det[:,ids]*u).sum((1,2))/norm
   rms=lambda q:np.sqrt(np.mean(np.sum(q[:,ids]**2,axis=-1),axis=-1))
   speed=norm/len(ids)**.5
   print(d,g,'step/pre/post/accel',np.round([speed,np.percentile(rms(pre),90)/speed,np.percentile(rms(post),90)/speed,np.percentile(rms(np.diff(post,axis=0)),90)/speed],3),'retain_p10_med_raw_det',np.round([np.percentile(proj,10),np.median(proj),np.percentile(proj_det,10),np.median(proj_det)],3))
 print('eachshoulder onset/priorp90',np.linalg.norm(v[f-1,5:7],axis=-1)/np.maximum(np.percentile(np.linalg.norm(v[f-9:f-1,5:7],axis=-1),90,axis=0),.3))
