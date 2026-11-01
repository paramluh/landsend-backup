#!/usr/bin/env python3
"""Read-only r3 attrition audit; no detector thresholds or saved scans changed."""
import concurrent.futures, collections, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, '/home/param/pose-repo/tools/pose_jump_audit')
import strict_detection_r3 as r3

BASE=Path('/home/param/runs/pose-runs')
SUITE=BASE/'pose-jump-audit-suite-20261031'
ROOTS=[BASE/'pose-jump-audit-20261031-aloe-raw-nocfg',BASE/'pose-jump-audit-20261031-aloe-raw-nocfg-cohort2']
STAGES=['eligible','necessary_amplitude','destination_tracks','head_10x_motion','head_and_each_shoulder_4x_innovation','actual_subset_amplitude','raw_and_detrended_persistence','one_second_context','all_three_fit_scales']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def run(path):
    rd=Path(path);pose=json.loads((rd/'pose_overlay.json').read_text());cached=json.loads((rd/'strict-detection-v5-r3.json').read_text())
    assert cached['checker_sha256']==sha(r3.__file__)
    assert cached['sources']['pose_sha256']==sha(rd/'pose_overlay.json')
    d=r3.prepare(pose['frames']);c=r3.Config();v=d['velocity'];s=d['scale']
    bound=np.sqrt(np.sum(v[:,:5]**2,axis=(1,2))/c.min_visible_head)
    viable=(bound>=s*c.head_step_shoulder_widths)&(r3.rms(v[:,5:7])>=s*c.shoulder_step_shoulder_widths)
    viable&=(np.linalg.norm(v[:,5],axis=-1)>=s*c.each_shoulder_step_shoulder_widths)&(np.linalg.norm(v[:,6],axis=-1)>=s*c.each_shoulder_step_shoulder_widths)
    frames={f for f in range(cached['eligible_frame_start'],cached['eligible_frame_end']+1) if viable[f-1]}
    assert len(frames)==cached['amplitude_gate_frames']
    count=collections.Counter(eligible=cached['eligible_frames'],necessary_amplitude=len(frames));reason=collections.Counter();extra=collections.Counter();near=[];complete=[]
    for row in cached['frame_diagnostics']:
        if row['frame'] not in frames:continue
        f=row['frame'];m=row['metrics'];why=row['rejection_reasons']
        conf=d['p'][f,:7,2];numerical=(sum(conf[:5]>=.4)>=3 and min(conf[5:7])>=.4)
        if not numerical:extra['numeric_destination_confidence_reject']+=1
        if 'head_onset_innovation_ratio' not in m:
            reason['destination_tracks']+=1
            if numerical:extra['in_canvas_context_only_reject']+=1
            continue
        count['destination_tracks']+=1
        ratio=m['head_jump_to_both_sides_median']
        innov=min(m[k] for k in ['head_onset_innovation_ratio','left_shoulder_onset_innovation_ratio','right_shoulder_onset_innovation_ratio'])
        extra['confidence_pass_motion_fail' if ratio<10 else 'confidence_pass_motion_pass']+=1
        extra['confidence_pass_innovation_fail' if innov<4 else 'confidence_pass_innovation_pass']+=1
        if ratio<10 and innov<4:extra['confidence_pass_both_onset_gates_fail']+=1
        near.append({'root':str(rd.parent.parent),'run_id':rd.name,'frame':f,'time_s':f/25,'onset_margin':min(ratio/10,innov/4),'head_motion_ratio':ratio,'minimum_innovation_ratio':innov,'cached_stage':row.get('diagnostics_stage'),'rejection_reasons':why})
        if ratio<10:reason['head_10x_motion']+=1;continue
        count['head_10x_motion']+=1
        if innov<4:reason['head_and_each_shoulder_4x_innovation']+=1;continue
        count['head_and_each_shoulder_4x_innovation']+=1
        assert row.get('diagnostics_stage')=='complete'
        complete.append({'root':str(rd.parent.parent),'run_id':rd.name,**row})
        variant=row['persistence_variants'][0]
        amp_fail=any(x.endswith('_step_too_small') or x=='both_shoulders_do_not_change_configuration' for x in variant['rejection_reasons'])
        if amp_fail:reason['actual_subset_amplitude']+=1;continue
        count['actual_subset_amplitude']+=1
        if not variant['passes_persistence']:reason['raw_and_detrended_persistence']+=1;continue
        count['raw_and_detrended_persistence']+=1
        context=min(value for key,value in m.items() if key.endswith('_context_peak_ratio'))
        if context<2.5:reason['one_second_context']+=1;continue
        count['one_second_context']+=1
        if not all(fit['v5_qualifies'] for fit in row['trajectory_fits']):reason['all_three_fit_scales']+=1;continue
        count['all_three_fit_scales']+=1
    return {'root':str(rd.parent.parent),'run_id':rd.name,'counts':dict(count),'rejections':dict(reason),'conditional_onset_counts':dict(extra),'near':near,'complete_onset_survivors':complete}

def main():
    start=time.time();jobs=[p for root in ROOTS for p in sorted((root/'runs').glob('rollout-*'))]
    assert len(jobs)==2000
    rows=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=24) as pool:
        for i,row in enumerate(pool.map(run,jobs,chunksize=8),1):
            rows.append(row)
            if i%200==0:print(json.dumps({'processed':i}),flush=True)
    counts=collections.Counter();rejections=collections.Counter();extras=collections.Counter();near=[];complete=[]
    for row in rows:
        counts.update(row['counts']);rejections.update(row['rejections']);extras.update(row['conditional_onset_counts']);near.extend(row['near']);complete.extend(row['complete_onset_survivors'])
    selected=sorted(near,key=lambda x:x['onset_margin'],reverse=True)[:60]
    loaded={}
    for item in selected:
        key=(item['root'],item['run_id']);rd=Path(key[0])/'runs'/key[1]
        if key not in loaded:loaded[key]=r3.prepare(json.loads((rd/'pose_overlay.json').read_text())['frames'])
        item['full_diagnostic_replay']=r3.evaluate(loaded[key],item['frame'],full_diagnostics=True)
    stages=[{'stage':s,'surviving_frames':counts[s],'newly_rejected_frames':rejections[s]} for s in STAGES]
    report={'state':'complete','created':time.time(),'elapsed_seconds':time.time()-start,'checker_sha256':sha(r3.__file__),'audit_source_sha256':sha(__file__),'roots':[str(p) for p in ROOTS],'rollouts':2000,'method':'Read-only exact amplitude-population reconstruction from source poses; sequential counts use cached frozen r3 rows. Later gates are counted only among survivors of earlier gates, because onset-rejected cached rows omit later diagnostics. Top60 onset near-misses are replayed with full diagnostics without altering saved scans.','stage_order_note':'Audit presents motion before innovation, then true subset amplitude, persistence, context, fits. This is a clearly declared sequential ordering, not independent failure rates.','stages':stages,'conditional_onset_counts':dict(extras),'head_motion_ratio_distribution':{'maximum':max(x['head_motion_ratio'] for x in near),'percentiles':{str(q):float(np.percentile([x['head_motion_ratio'] for x in near],q)) for q in [50,90,95,99,100]},'survivors_at_ratio':{str(t):sum(x['head_motion_ratio']>=t for x in near) for t in [2,3,4,5,6,7,8,9,10]},'survivors_with_4x_innovation_at_ratio':{str(t):sum(x['head_motion_ratio']>=t and x['minimum_innovation_ratio']>=4 for x in near) for t in [2,3,4,5,6,7,8,9,10]}},'complete_onset_survivors':complete,'top_onset_near_misses':selected,'per_run_counts':[{k:v for k,v in r.items() if k not in ['near','complete_onset_survivors']} for r in rows]}
    out=SUITE/'v6/old-aloe-attrition.json';out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'output':str(out),'stages':stages,'conditional_onset_counts':dict(extras),'complete_onset_survivors':len(complete),'elapsed_seconds':report['elapsed_seconds']}),flush=True)

if __name__=='__main__':main()
