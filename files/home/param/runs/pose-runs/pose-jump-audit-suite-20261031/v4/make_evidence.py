import json,hashlib,sys,time
from pathlib import Path
from dataclasses import asdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,'/home/param/pose-repo/tools/pose_jump_audit')
from trajectory_review import assess_event,ReviewConfig,VERSION
out=Path(__file__).parent
cases=[('Original v1: user-identified jump',Path('/home/param/runs/pose-runs/pose-sites-20261030/video-inspection/v1-19-21/pose_overlay.json'),512,True),('Rollout145: user-confirmed false positive',Path('/home/param/runs/pose-runs/pose-jump-audit-20261031-nocfg/positive-qa/runs/rollout-000145/pose_overlay.json'),474,False)]
fig,axs=plt.subplots(2,2,figsize=(13,7),layout='constrained');records=[]
for row,(name,path,f,label) in enumerate(cases):
 p=np.array(json.loads(path.read_text())['frames']);xy=p[:,:,:2]*[256,480]
 a=assess_event(p,{'frame':f});records.append({'name':name,'frame':f,'known_user_label_jump':label,'source_pose_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'assessment':a})
 for col,(group,ids) in enumerate([('head',np.arange(5)),('shoulders',np.array([5,6]))]):
  ax=axs[row,col];trajectory=xy[:,ids].mean(1);direction=trajectory[f]-trajectory[f-1];direction/=np.linalg.norm(direction)
  y=(trajectory-trajectory[f-1])@direction;visible=np.arange(f-15,f+16);ax.plot((visible-f+.5)/25,y[visible],color='#303943',lw=2,marker='o',ms=3,label='Observed pose')
  half=6;t=(np.arange(-half,half)+.5)/half;indices=np.arange(f-half,f+half)
  for kind,X,color,style in [('Continuous cubic',np.c_[t*0+1,t,t*t,t*t*t],'#8563b8','--'),('Independent lines + step',np.c_[t*0+1,t,t>=0,t*(t>=0)],'#cf7b29','-')]:
   fit=X@np.linalg.lstsq(X,y[indices],rcond=None)[0]
   if 'step' in kind:
    ax.plot((indices[:half]-f+.5)/25,fit[:half],color=color,lw=2,ls=style,label=kind);ax.plot((indices[half:]-f+.5)/25,fit[half:],color=color,lw=2,ls=style)
   else:ax.plot((indices-f+.5)/25,fit,color=color,lw=2,ls=style,label=kind)
  ax.axvspan(-.02,.02,color='#cbd3db',alpha=.5);ax.axvline(0,color='#8a97a3',lw=.7)
  ax.set(title=name+' · '+group,xlabel='Seconds relative to flagged transition',ylabel='Position along transition direction (px)');ax.grid(alpha=.2)
  if row==0 and col==0:ax.legend(loc='upper left',fontsize=9)
fig.suptitle('A speed spike can also be a continuous gesture',fontsize=18)
fig.savefig(out/'reference-vs-145-trajectories.png',dpi=150);plt.close(fig)
report={'version':VERSION,'checker_sha256':hashlib.sha256(Path('/home/param/pose-repo/tools/pose_jump_audit/trajectory_review.py').read_bytes()).hexdigest(),'created':time.time(),'config':asdict(ReviewConfig()),'validation':{'command':'/home/param/pose-repo/.venv/bin/python -m unittest discover -s tools/pose_jump_audit -p test_trajectory_review.py -v','test_methods_passed':13,'scope':'Varied direction/magnitude/confidence rigid steps; independent body-shape changes; noisy steps; two-frame step with settling; step over slow motion; continuous torso with abrupt head; fast symmetric/asymmetric smooth gestures; smooth monotonic transitions; hand entry; one-frame excursion; original positive; user-confirmed145 negative.'},'known_cases':records,'cohorts':[],'limits':['The negative user label145 motivated a general torso-onset/trajectory distinction. Thresholds are transparent and behavioral controls are synthetic; this is not a validated perceptual classifier.','Original v3 windows remain in full render pool. A v4 rejection does not prove an event is harmless; a pass does not confirm a visible avatar defect.','Both trajectory models have four parameters per coordinate; fit half-windows4/6/8frames. The two-interval variant excludes the intermediate observation from both fits.','Only head and shoulders vote; hands are excluded. Low-confidence body tracking and generated image appearance can differ from decoded pose.']}
for suffix in ['nocfg','xabi-nocfg','aloe-raw-nocfg']:
 r=Path('/home/param/runs/pose-runs/pose-jump-audit-20261031-'+suffix);v=json.loads((r/'trajectory-review-v4.json').read_text());report['cohorts'].append({'cohort':suffix,'v3_events_preserved':v['v3_events'],'v4_stricter_events':v['stricter_events'],'v4_stricter_rollouts':v['stricter_rollouts']})
(out/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
