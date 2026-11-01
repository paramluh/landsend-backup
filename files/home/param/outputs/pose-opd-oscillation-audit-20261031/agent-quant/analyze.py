import json, math, statistics as st
from pathlib import Path
OUT=Path(__file__).parent
ROOTS={'current':Path('/home/param/outputs/pose-opd-symmetric-b336-website-pad0-live-20261031/current').resolve(),'fixed':Path('/home/param/outputs/pose-opd-credit-live-attempt2/snapshots/1793465773953555733-553034'),'asym':Path('/home/param/outputs/pose-opd-symmetric-production-launch-20261031/dashboard/reference-stopped-update13')}

def corr(a,b):
 try:return st.correlation(a,b)
 except:return None

def smooth(y,n=10):return [st.mean(y[max(0,i-n+1):i+1]) for i in range(len(y))]
def extrema(y):
 return {'min':[1+y.index(min(y)),min(y)],'max':[1+y.index(max(y)),max(y)]}
def sig_extrema(y):
 out=[]
 for i in range(1,len(y)-1):
  if y[i]>y[i-1] and y[i]>y[i+1]: out.append(['max',i+1,y[i]])
  elif y[i]<y[i-1] and y[i]<y[i+1]:out.append(['min',i+1,y[i]])
 return out

def cos(c,d,r):return (c*c-d*d-r*r)/(2*d*r) if d*r else None

summary={}
for name,path in ROOTS.items():
 ds=[json.loads(s) for s in (path/'metrics.jsonl').read_text().splitlines()]
 cfg=json.loads((path/'config.json').read_text()); m=cfg['method']
 (OUT/(name+'-metrics.jsonl')).write_text('\n'.join(json.dumps(d) for d in ds)+'\n')
 dct={'source':str(path),'n':len(ds),'summary':{},'norm_samples':[],'phase_means':{},'correlations':{},'pairwise_codebooks':{}}
 keys=['loss/direct','loss/text','loss/audio','loss/pose','gradient_norm','credit/absolute_sum','learning_rate','learning_rate_pose_light','kl/lexical','kl/category','kl/text']
 keys += [k for k in ds[-1] if k.startswith(('logit_grad_credit_direct_ratio','logit_grad_alignment','conversation/','grad_norm/combined/'))]
 for key in keys:
  vals=[d.get(key) for d in ds]
  if any(v is None for v in vals):continue
  dct['summary'][key]={'first':vals[0],'last':vals[-1],'median':st.median(vals),'mean':st.mean(vals),'extrema':extrema(vals),'smooth10_extrema':sig_extrema(smooth(vals))}
 for lo,hi in [(1,5),(6,10),(11,15),(16,20),(21,25),(26,30),(31,35),(36,40),(41,45),(46,50),(51,60),(61,70),(71,80),(81,89)]:
  seg=[d for d in ds if lo<=d['update']<=hi]
  if not seg:continue
  dct['phase_means'][f'{lo}-{hi}']={k:st.mean([d[k] for d in seg]) for k in keys if all(k in d for d in seg)}
 for d in ds:
  if 'grad_norm/direct/all' not in d:continue
  row={'update':d['update']}
  for g in ['all','layers','head','tower_s','dec_s','tower_p','dec_p','inp','vadapter']:
   a=d['gradient_norm'] if g=='all' else d[f'grad_norm/combined/{g}']
   b=d[f'grad_norm/direct/{g}']; c=d[f'grad_norm/credit/{g}']
   row[g]={'direct':b,'credit':c,'combined':a,'credit_to_direct':c/b if b else None,'cosine':cos(a,b,c)}
  row['component_norms']={k:d[k] for k in d if k.startswith(('grad_norm/direct/','grad_norm/credit/')) and k.endswith('/all')}
  dct['norm_samples'].append(row)
 for mod in ['audio','pose']:
  cs=[]; diffs=[]
  for i in range(1,32):
   for j in range(i+1,32):
    a=[d[f'loss/{mod}_cb{i}'] for d in ds]; b=[d[f'loss/{mod}_cb{j}'] for d in ds]
    cs.append(corr(a,b)); diffs.append(corr([v-u for u,v in zip(a,a[1:])],[v-u for u,v in zip(b,b[1:])]))
  dct['pairwise_codebooks'][mod]={'median_corr_cb1_31':st.median(cs),'min_corr':min(cs),'median_firstdiff_corr':st.median(diffs)}
  vals=[sum(d[f'loss/{mod}_cb{i}'] for i in range(1,32)) for d in ds]
  dct['summary'][f'{mod}_depth_sum']={'first':vals[0],'last':vals[-1],'extrema':extrema(vals),'smooth10_extrema':sig_extrema(smooth(vals))}
 base=[d['loss/audio'] for d in ds]
 for k in keys:
  if any(k not in d for d in ds):continue
  y=[d[k] for d in ds]
  dct['correlations'][k]={'r_with_audio':corr(base,y),'r_firstdiff_with_audio':corr([b-a for a,b in zip(base,base[1:])],[b-a for a,b in zip(y,y[1:])])}
 dct['clip_gt1_count']=sum(d['gradient_norm']>1 for d in ds)
 summary[name]=dct
(OUT/'summary.json').write_text(json.dumps(summary,indent=2))
for name,d in summary.items():
 print('\nRUN',name,'N',d['n'],'CLIP>1',d['clip_gt1_count'])
 for k in ['loss/direct','loss/text','loss/audio','loss/pose','audio_depth_sum','pose_depth_sum','gradient_norm','learning_rate']:
  print(k,json.dumps(d['summary'].get(k)))
 print('CODEBOOKCORR',d['pairwise_codebooks'])
 print('PHASEMEANS')
 for phase,means in d['phase_means'].items():
  print(phase,' '.join(f'{k}={v:.5g}' for k,v in means.items() if k in ['loss/direct','loss/text','loss/audio','loss/pose','gradient_norm','conversation/student/sampled_fraction/word','conversation/student/sampled_fraction/pad','conversation/partner/sampled_fraction/word']))
 print('NORMS')
 for r in d['norm_samples']:
  print(r['update'],{g:{k:round(v,4) if v is not None else v for k,v in vals.items()} for g,vals in r.items() if g in ['all','layers','head','tower_s','tower_p']})
