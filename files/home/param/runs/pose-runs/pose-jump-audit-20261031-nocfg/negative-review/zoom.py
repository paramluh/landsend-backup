from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
r=Path(__file__).parent
choices=[]
for case in json.loads((r/'sheets/contacts.json').read_text()):
 p=json.loads((r/'public/data'/(case['case_id']+'.json')).read_text());a=np.array(p['frames']);xy=a[:,:,:2]*[256,480];v=np.linalg.norm(np.diff(xy,axis=0),axis=-1);h=np.median(v[:,:5],axis=1);s=np.median(v[:,5:7],axis=1)
 for item in case['contacts']:
  i=item['frame']-1;idx=max(range(max(0,i-1),min(len(v),i+2)),key=lambda j:h[j]*s[j]);f=idx+1
  if h[idx]<5 or s[idx]<2:continue
  fig=plt.figure(figsize=(12,6),layout='constrained');gs=fig.add_gridspec(2,6,height_ratios=[3,1.5])
  for col,off in enumerate([-4,-2,-1,0,1,4]):
   n=f+off;ax=fig.add_subplot(gs[0,col]);
   for u,w in p['edges']:
    if min(a[n,u,2],a[n,w,2])>.2:ax.plot(xy[n,[u,w],0],xy[n,[u,w],1],color='green' if u<13 else '#aaaaaa',lw=1.5)
   ax.set(xlim=(-5,261),ylim=(485,-5),aspect='equal',title=f'{n/25:.2f} ({off:+d})');ax.grid(alpha=.2)
  t=np.arange(1,len(a))/25;ax=fig.add_subplot(gs[1,:]);ax.plot(t,h,label='head');ax.plot(t,s,label='shoulders');ax.set(xlim=(f/25-1.5,f/25+1.5));ax.axvline(f/25,ls='--',color='black');ax.legend();ax.grid(alpha=.2);fig.suptitle(case['case_id']+' raw-motion transition '+str(f));fig.savefig(r/'sheets'/f'{case["case_id"]}-zoom-{f}.png',dpi=110);plt.close(fig);choices.append((case['case_id'],f,round(h[idx],2),round(s[idx],2)))
(r/'sheets/zoom-contacts.json').write_text(json.dumps(choices,indent=2));print(choices)
