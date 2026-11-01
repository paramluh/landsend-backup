from pathlib import Path
import json,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,'/home/param/pose-repo/tools/pose_jump_audit')
from detection import frame_metrics
root=Path(__file__).parent
sel=json.loads((root/'selection.json').read_text());metrics=[]
for e in sel:
 r=e['run_id'];f=e['frame'];p=json.loads((root/'runs'/r/'pose_overlay.json').read_text());a=np.array(p['frames']);m,s=frame_metrics(a);xy=a[:,:,:2]*[256,480]
 fig=plt.figure(figsize=(15,8.5),layout='constrained');gs=fig.add_gridspec(3,6,height_ratios=[3,1.4,1.4])
 for z,offset in enumerate([-4,-1,0,1,4,10]):
  ax=fig.add_subplot(gs[0,z]);n=f+offset
  for u,v in p['edges']:
   c=min(a[n,u,2],a[n,v,2]);
   if c>.05:ax.plot(xy[n,[u,v],0],xy[n,[u,v],1],color=plt.cm.RdYlGn(np.clip((c-.2)/.4,0,1)),alpha=np.clip((c-.05)/.15,0,1),lw=1.5)
  ax.scatter(xy[n,:7,0],xy[n,:7,1],s=6,c='black')
  ax.set(xlim=(-10,266),ylim=(490,-10),aspect='equal',title=f'{n/25:.2f}s ({offset:+d})');ax.set_facecolor('#ededed');ax.grid(alpha=.2)
 t=m['time_s'];ax=fig.add_subplot(gs[1,:]);
 for n in ['head','shoulders','core']:ax.plot(t,m[n+'_px'],label=n)
 ax.set(xlim=(f/25-2,f/25+2),ylabel='px/frame');ax.axvline(f/25,color='black',lw=1,ls='--');ax.legend();ax.grid(alpha=.3)
 ax=fig.add_subplot(gs[2,:]);ax.plot(t,m['head_px'],alpha=.65,label='head');ax.plot(t,m['core_px'],label='core');ax.axvline(f/25,color='black',lw=1,ls='--');ax.set(xlim=(5.12,len(a)/25),ylabel='px/frame full run',xlabel='seconds');ax.grid(alpha=.3);ax.legend()
 fig.suptitle(f'{r} @ {f/25:.2f}s · pose snapshots and motion context',fontsize=16)
 fig.savefig(root/f'{r}-{f}.png',dpi=120);plt.close(fig)
 region=(t>=f/25-2)&(t<=f/25+2);allmask=(t>=5.12)&(np.abs(t-f/25)>.2)
 row={'run_id':r,'frame':f,'time_s':f/25,'selection_group':e['group'],'peak':{k:float(m[k+'_px'][f-1]) for k in ['head','shoulders','core']},'generated_p90':{k:float(np.percentile(m[k+'_px'][allmask],90)) for k in ['head','shoulders','core']},'core_fraction_elsewhere_above_third_peak':float(np.mean(m['core_px'][allmask]>m['core_px'][f-1]/3)),'core_nearby':[(int(m['frame'][i]),round(float(m['core_px'][i]),3)) for i in range(f-7,f+10)]}
 metrics.append(row)
(root/'independent-metrics.json').write_text(json.dumps(metrics,indent=2));print(json.dumps(metrics,indent=2))
