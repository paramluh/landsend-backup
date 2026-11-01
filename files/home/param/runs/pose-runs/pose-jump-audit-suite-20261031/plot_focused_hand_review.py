import json,numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
root=Path('/home/param/runs/pose-runs/pose-jump-audit-suite-20261031/hand-focused-review')
rank=json.loads((root/'ranking.json').read_text())
cmap=LinearSegmentedColormap.from_list('confidence',['#ef4444','#facc15','#22c55e'])
offsets=[-4,-2,-1,0,1,4,8]
for r in rank['selected']:
 w=json.loads(Path(r['window_json']).read_text());p=np.array(w['frames']);f=r['frame']-w['start_frame'];ids=np.array(r['anchor_indices'])
 start=13 if r['group']=='left_hand' else 34;hi=np.arange(start,start+21);xy=p[:,:,:2]*[256,480]
 edges=[(start+a,start+b) for k in [1,5,9,13,17] for a,b in [(0,k),(k,k+1),(k+1,k+2),(k+2,k+3)]]
 fig=plt.figure(figsize=(15,7),facecolor='#10141a');gs=fig.add_gridspec(2,7,height_ratios=[1,1.2],hspace=.35,wspace=.12)
 ax=fig.add_subplot(gs[0,:5]);ax.set_facecolor('#171d27');tt=(np.arange(len(p))-f)/25;step=xy[f,ids]-xy[f-1,ids]
 projection=np.sum((xy[:,ids]-xy[f-1,ids])*step,axis=(1,2))/np.sum(step**2)
 ax.plot(tt,projection,color='#63c5fa',marker='.',ms=4);ax.axvline(0,color='white',lw=1);ax.axhline(1,color='#4ade80',ls='--',alpha=.6)
 ax.set_xlim(-.6,.6);ax.set_ylim(min(-.2,projection[(tt>=-.6)&(tt<=.6)].min()-.1),max(1.3,projection[(tt>=-.6)&(tt<=.6)].max()+.1))
 ax.tick_params(colors='white');ax.set_xlabel('Seconds relative to candidate frame',color='white');ax.set_ylabel('Displacement / initial jump',color='white');ax.grid(alpha=.15)
 ax2=fig.add_subplot(gs[0,5:]);ax2.set_facecolor('#171d27');ax2.set_xlim(0,256);ax2.set_ylim(480,0);ax2.set_aspect('equal');ax2.tick_params(colors='white',labelsize=7)
 for of,col in [(-1,'#f59e0b'),(0,'#22c55e'),(8,'#60a5fa')]:
  q=xy[f+of];ax2.scatter(q[:13,0],q[:13,1],s=8,color='#9ca3af',alpha=.4);ax2.plot(q[hi,0],q[hi,1],'.',color=col,ms=3,label=str(of));ax2.scatter(q[ids,0],q[ids,1],s=15,color=col)
 ax2.set_title('Whole-canvas location',color='white',fontsize=9);ax2.legend(fontsize=6,loc='upper left')
 vals=np.clip(xy[f-4:f+9,hi].reshape(-1,2),[0,0],[256,480]);mn=vals.min(0)-10;mx=vals.max(0)+10;span=np.maximum(mx-mn,[45,65]);center=(mn+mx)/2
 for k,of in enumerate(offsets):
  a=fig.add_subplot(gs[1,k]);a.set_facecolor('#171d27');a.set_xlim(center[0]-span[0]/2,center[0]+span[0]/2);a.set_ylim(center[1]+span[1]/2,center[1]-span[1]/2);a.set_aspect('equal');q=xy[f+of];conf=p[f+of,:,2]
  for u,v in edges:a.plot(q[[u,v],0],q[[u,v],1],color=cmap(np.clip((min(conf[u],conf[v])-.2)/.4,0,1)),lw=1.5,alpha=.7)
  a.scatter(q[hi,0],q[hi,1],c=np.clip((conf[hi]-.2)/.4,0,1),cmap=cmap,vmin=0,vmax=1,s=14)
  a.scatter(q[ids,0],q[ids,1],s=35,facecolors='none',edgecolors='white',lw=.6)
  a.set_title(f'f{r["frame"]+of} ({of:+d})\nmedian conf {np.median(conf[ids]):.2f}',color='white',fontsize=9);a.tick_params(colors='#9ca3af',labelsize=6)
 m=r['metrics'];name=Path(r['window_json']).stem
 fig.suptitle(name+f'  {r["group"]}  jump={r["original_group_metrics"]["single_frame_step_px"]:.1f}px\nShape-prioritized review only; next3-frame max/jump={m["next3_step_ratio"]:.2f}. White rings = detector anchors. Pose review only.',color='white',fontsize=11)
 fig.savefig(root/(name+'.png'),dpi=140,bbox_inches='tight',facecolor=fig.get_facecolor());plt.close(fig)
print('plots created')
