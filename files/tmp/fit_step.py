exec(open('/tmp/probe_jump_semantics.py').read().split('for name,p,f in cases:')[0])
for name,p,f in cases:
 a=np.array(json.loads(p.read_text())['frames']);xy=a[:,:,:2]*[256,480]
 print(name)
 for w in [4,6,8,10,12]:
  t=np.arange(-w,w)+.5; y=xy[f-w:f+w,:7].reshape(2*w,-1);t=t/w
  smooth=np.c_[t*0+1,t,t**2,t**3]
  step=np.c_[t*0+1,t,t>=0,t*(t>=0)]
  def errs(X):
   pred=X@np.linalg.lstsq(X,y,rcond=None)[0];return ((pred-y)**2).reshape(2*w,7,2).sum(-1).mean(0)**.5
  se,je=errs(smooth),errs(step)
  print(w,'head smooth/step',np.median(se[:5]),np.median(je[:5]),'ratio',np.median(se[:5])/np.median(je[:5]),'shoulders',se[5:].mean(),je[5:].mean(),'ratio',se[5:].mean()/je[5:].mean())
