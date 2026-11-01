#!/usr/bin/env python3
"""Verify a frozen strict scan and render coverage of ONLY its accepted events.

Prior broad-pool render completeness is deliberately outside this receipt.
Human labels are read from an explicit revision overlay, never rewritten into
historical controls or inferred from detector acceptance.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

import strict_detection_r3 as checker
from verify import probe_clip, probe_poster

EXPECTED_SHA='771b696b6b3cdb3509b245bc5909d5552321ca22d79e49633d668b5b60efdd2e'
DEFAULT_SUITE=Path('/home/param/runs/pose-runs/pose-jump-audit-suite-20261031')


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def require(ok,message):
    if not ok:raise ValueError(message)
def canonical(obj):return json.dumps(obj,sort_keys=True,separators=(',',':'))


def verify_controls(path,source,labels_path,control_poses):
    controls=read(path)
    require(controls['checker_sha256']==sha(source)==EXPECTED_SHA,'Checker/control hash does not match frozen r3')
    require(controls['version']==checker.VERSION,'Control version differs')
    require(canonical(controls['config'])==canonical(asdict(checker.Config())),'Control configuration differs')
    require(controls['tests_passed'] and controls['tests_run']>0,'Behavioral controls did not pass')
    require(not any(controls.get(k) for k in ('failures','errors','skipped')),'Controls have failures/errors/skips')
    for name,digest in controls['test_source_hashes'].items():
        archived=path.parent/'source'/name
        actual=archived if archived.exists() else source.parent/name
        require(sha(actual)==digest,f'Changed test source:{name}')
    current={c['case']:c.get('user_confirmed_jump') for c in controls['known_cases']}
    label_receipt=read(labels_path)
    for update in label_receipt.get('updates',[]):
        current[update['case']]=update['current_user_confirmed_jump']
    require(current.get('v1') is True and current.get('243') is False and current.get('145') is False,
            'Current known labels must include the243 rewatch correction')
    expected={'v1':True,'243':False,'145':False}
    replay=[]
    for case in controls['known_cases']:
        name=case['case'];p=control_poses/(name+'.json')
        require(sha(p)==case['source_sha256'],f'Changed control pose:{name}')
        result=checker.evaluate(checker.prepare(read(p)['frames']),case['assessment']['frame'])
        require(result['passes']==case['assessment']['passes'],f'Control replay changed eligibility:{name}')
        if name in expected:require(result['passes']==expected[name],f'Unexpected required control status:{name}')
        replay.append({'case':name,'frame':result['frame'],'eligibility':result['passes'],
                       'historical_human_label':case.get('user_confirmed_jump'),'current_human_label':current.get(name),
                       'source_pose_sha256':case['source_sha256'],'replayed':True})
    return controls,{'controls_sha256':sha(path),'tests_run':controls['tests_run'],'test_sources_match':True,
                     'current_label_receipt':str(labels_path),'current_label_receipt_sha256':sha(labels_path),
                     'control_pose_replays':replay}


def eligible_and_required_frames(pose,summary,old,cfg):
    """Independently reconstruct the full-frame necessary amplitude gate."""
    p=np.asarray(pose['frames'],dtype=np.float64)
    fps=float(pose['fps']);start=max(math.ceil(summary['prefill_frames']/12.5*fps),cfg.prior_frames+1)
    stop=len(p)-cfg.persistence_end_frame-1
    eligible=np.arange(start,max(start,stop+1))
    xy=p[:,:7,:2]*[256,480]
    inside=((p[:,:7,:2]>=-cfg.canvas_margin).all(-1)&(p[:,:7,:2]<=1+cfg.canvas_margin).all(-1))
    reliable=(p[:,:7,2]>=cfg.scale_confidence_min)&inside
    widths=np.linalg.norm(xy[:,5]-xy[:,6],axis=-1)
    valid=reliable[:,5]&reliable[:,6]&(widths>256*.05)
    scale=float(np.median(widths[valid])) if valid.any() else 128.
    delta=np.diff(xy,axis=0)
    head_bound=np.sqrt(np.sum(delta[:,:5]**2,axis=(1,2))/cfg.min_visible_head)
    shoulder_rms=np.sqrt(np.mean(np.sum(delta[:,5:7]**2,axis=-1),axis=-1))
    mask=(head_bound>=scale*cfg.head_step_shoulder_widths)&(shoulder_rms>=scale*cfg.shoulder_step_shoulder_widths)
    mask&=(np.linalg.norm(delta[:,5],axis=-1)>=scale*cfg.each_shoulder_step_shoulder_widths)
    mask&=(np.linalg.norm(delta[:,6],axis=-1)>=scale*cfg.each_shoulder_step_shoulder_widths)
    gated={int(f) for f in eligible if mask[f-1]}
    forced={int(e['frame']) for e in old['events'] if start<=e['frame']<=stop}
    return p,scale,start,stop,len(eligible),gated,gated|forced


def check_render_coverage(root,run_id,events,pose,token_sha,rendered_roots):
    """Accept exact or larger windows, with exact marked events and source hashes."""
    if not events:return [],[]
    array=np.asarray(pose['frames'],dtype=np.float64);clips=[]
    for base in rendered_roots:
        path=base/run_id/'complete.json'
        if not path.exists():continue
        receipt=read(path)
        if receipt.get('state')!='done':continue
        for clip in receipt.get('clips',[]):clips.append((path.parent,clip))
    coverage=[];verified={}
    for event in events:
        f=event['frame'];need_start=max(0,f-50);need_stop=min(len(array),f+50)
        suitable=[]
        for directory,clip in clips:
            marks={e['frame'] for e in clip['events']}
            if f in marks and clip['start_frame']<=need_start and clip['stop_frame']>=need_stop:suitable.append((directory,clip))
        require(bool(suitable),f'Missing rendered coverage for strict event:{run_id}:{f}')
        directory,clip=min(suitable,key=lambda item:item[1]['stop_frame']-item[1]['start_frame'])
        key=str(directory/clip['name'])
        if key not in verified:
            require(clip['source_tokens_sha256']==token_sha,f'Render token mismatch:{key}')
            start,stop=clip['start_frame'],clip['stop_frame']
            require(0<=start<stop<=len(array),'Render window outside source')
            require(clip['output_frames']==stop-start,'Render frame count metadata mismatch')
            require(clip['pose_window_sha256']==hashlib.sha256(array[start:stop].tobytes()).hexdigest(),f'Render pose mismatch:{key}')
            a=probe_clip((Path(key)/'video.mp4',stop-start,False))
            b=probe_clip((Path(key)/'overlay.mp4',stop-start,True))
            require(a['audio_packets_sha256']==b['audio_packets_sha256'],'Overlay audio differs from source clip')
            poster=probe_poster(Path(key)/'overlay.jpg')
            verified[key]={'window':key,'source_tokens_sha256':token_sha,'video':a,'overlay':b,'poster':poster}
        coverage.append({'run_id':run_id,'frame':f,'window':key,'source_start_frame':clip['start_frame'],
                         'source_stop_frame':clip['stop_frame'],'strict_event_marker_verified':True})
    return coverage,list(verified.values())


def verify(root,*,count=1000,start_index=0,expected_duration=60.,controls_path=None,labels_path=None,
           control_poses=None,rendered_roots=None):
    root=Path(root).resolve();source=Path(checker.__file__).resolve()
    controls_path=Path(controls_path or DEFAULT_SUITE/'v5/r3/controls.json')
    labels_path=Path(labels_path or DEFAULT_SUITE/'v5/ground-truth-update-243-rewatch.json')
    control_poses=Path(control_poses or controls_path.parent/'control-poses')
    rendered_roots=[Path(p) for p in (rendered_roots or [root/'rendered-strict-v5-r3',root/'rendered'])]
    failures=[];run_checks=[];all_events=[];coverage=[];media=[];control_check={};preservation={'r2_present':False}
    result={'created':time.time(),'state':'failed','root':str(root),'expected_rollouts':count,
            'checker_sha256':EXPECTED_SHA,'verifier_sha256':sha(__file__),'failures':failures}
    try:
        controls,control_check=verify_controls(controls_path,source,labels_path,control_poses)
        report_path=root/'strict-review-v5-r3.json';report=read(report_path);cfg=checker.Config()
        result.update(candidate_events_reported=report.get('candidate_events'),
                      candidate_rollouts_reported=report.get('candidate_rollouts'))
        require(report['checker_sha256']==EXPECTED_SHA and report['version']==checker.VERSION,'Scan checker/version mismatch')
        require(report['controls']==controls,'Scan controls differ from pinned receipt')
        require(canonical(report['config'])==canonical(asdict(cfg)),'Scan config differs from frozen checker')
        expected={f'rollout-{i:06d}' for i in range(start_index,start_index+count)}
        actual={p.name for p in (root/'runs').glob('rollout-*') if p.is_dir()}
        require(actual==expected,f'Run IDs differ:missing={sorted(expected-actual)[:10]},extra={sorted(actual-expected)[:10]}')
        sources={s['run_id']:s for s in report['sources']}
        require(len(sources)==len(report['sources'])==count and set(sources)==expected,'Duplicate/missing source manifest IDs')
        require(report['analyzed_rollouts']==count,'Cohort scan incomplete')
        r2_path=root/'strict-review-v5-r2.json'
        if r2_path.exists():
            r2=read(r2_path)
            require(sha(r2_path)==sha(root/'strict-review-v5.json'),'Historical r2 report changed')
            require(sha(source.with_name('strict_detection.py'))==r2['checker_sha256'],'Historical r2 source changed')
            require(r2['sources']==report['sources'],'Input source files changed since frozen r2 scan')
            preservation={'r2_present':True,'r2_report_sha256':sha(r2_path),'r2_report_unchanged':True,
                          'r2_checker_unchanged':True,'r2_input_sources_unchanged':True}
        for rid in sorted(expected):
            rd=root/'runs'/rid
            try:
                s=read(rd/'summary.json');pose=read(rd/'pose_overlay.json');old=read(rd/'detection.json')
                scan=read(rd/'strict-detection-v5-r3.json')
                require(read(rd/'status.json')['state']=='done','Generation not marked complete')
                require(s['sample_index']==int(rid.split('-')[1]),'Sample index differs from run ID')
                require(float(s['duration'])==expected_duration,'Rollout duration differs')
                require(pose['fps']==25 and pose['width']==256 and pose['height']==480 and pose.get('start_time',0)==0,'Pose timeline/geometry differs')
                require(len(pose['frames'])==pose['frame_count']==round(expected_duration*25),'Pose frame count differs')
                current={'pose_sha256':sha(rd/'pose_overlay.json'),'tokens_sha256':sha(rd/'tokens.npz'),
                         'summary_sha256':sha(rd/'summary.json'),'v3_detection_sha256':sha(rd/'detection.json')}
                require(current==scan['sources'],'Per-run source hashes differ from current files')
                require({'run_id':rid,**current}==sources[rid],'Cohort source hashes differ from run')
                require(s['tokens_sha256']==current['tokens_sha256'],'Generation token hash differs')
                require(scan['checker_sha256']==EXPECTED_SHA and scan['version']==checker.VERSION,'Per-run checker differs')
                require(canonical(scan['config'])==canonical(asdict(cfg)),'Per-run configuration differs')
                array,scale,start,stop,n,gated,required=eligible_and_required_frames(pose,s,old,cfg)
                require(array.shape==(round(expected_duration*25),55,3) and np.isfinite(array).all(),'Invalid pose array')
                require(abs(scan['scale_shoulder_width_px']-scale)<1e-9,'Normalization scale differs')
                require((scan['eligible_frame_start'],scan['eligible_frame_end'],scan['eligible_frames'])==(start,stop,n),'Eligible-frame population differs')
                rows=scan['frame_diagnostics'];frames=[r['frame'] for r in rows]
                require(len(frames)==len(set(frames))==scan['evaluated_frames'],'Duplicate/missing frame diagnostics')
                require(set(frames)==required,'Necessary amplitude candidates or requested old diagnostics omitted')
                require(scan['amplitude_gate_frames']==len(gated),'Amplitude population count differs')
                accepted={r['frame'] for r in rows if r['passes']}
                represented=[f for e in scan['events'] for f in e['equivalent_frames']]
                require(set(represented)==accepted and len(represented)==len(set(represented)),'Accepted frames omitted/duplicated in merged events')
                require(scan['candidate_events']==len(scan['events']),'Per-run event count differs')
                if scan['events']:
                    prepared=checker.prepare(array)
                    for event in scan['events']:
                        require(event['run_id']==rid and event['frame'] in accepted,'Invalid event identity')
                        require(abs(event['time_s']-event['frame']/25)<1e-9,'Event timestamp differs')
                        require(checker.evaluate(prepared,event['frame'])['passes'],'Accepted event fails replay under frozen checker')
                    cov,vid=check_render_coverage(root,rid,scan['events'],pose,current['tokens_sha256'],rendered_roots)
                    coverage.extend(cov);media.extend(vid)
                all_events.extend(scan['events'])
                run_checks.append({'run_id':rid,'eligible_frames':n,'evaluated_frames':len(rows),
                                   'candidate_events':len(scan['events']),'strict_result_sha256':sha(rd/'strict-detection-v5-r3.json')})
            except Exception as exc:failures.append({'stage':'run','run_id':rid,'error':f'{type(exc).__name__}: {exc}'})
        if not failures:
            require(sum(r['eligible_frames'] for r in run_checks)==report['eligible_frames'],'Cohort eligible count differs')
            require(canonical(all_events)==canonical(report['events']),'Cohort event list differs from per-run scans')
            require(report['candidate_events']==len(all_events),'Cohort candidate count differs')
            require(report['candidate_rollouts']==len({e['run_id'] for e in all_events}),'Cohort candidate run count differs')
            require(len(coverage)==len(all_events),'Accepted strict events lack verified render coverage')
        result.update(scan_summary_sha256=sha(report_path),controls=control_check,preservation=preservation)
    except Exception as exc:failures.append({'stage':'cohort','error':f'{type(exc).__name__}: {exc}'})
    result.update(state='passed' if not failures else 'failed',verified_rollouts=len(run_checks),
                  eligible_frames_verified=sum(r['eligible_frames'] for r in run_checks),candidate_events_verified=len(all_events),
                  candidate_rollouts_verified=len({e['run_id'] for e in all_events}),strict_events_with_render_coverage=len(coverage),
                  rendered_windows_verified=len({r['window'] for r in coverage}),coverage=coverage,media_checks=media,run_checks=run_checks,
                  interpretation='Completion/provenance/media verification only. Detector candidates are not confirmed visible defects. Current human labels are metadata separate from frozen historical controls. Old broad-pool render coverage is not required.')
    path=root/'strict-completion-verification.json';tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(path)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--count',type=int,default=1000);p.add_argument('--start-index',type=int,default=0)
    p.add_argument('--expected-duration',type=float,default=60.);p.add_argument('--controls',type=Path)
    p.add_argument('--labels',type=Path);p.add_argument('--control-poses',type=Path)
    p.add_argument('--rendered-roots',type=Path,nargs='+');a=p.parse_args()
    r=verify(a.root,count=a.count,start_index=a.start_index,expected_duration=a.expected_duration,
             controls_path=a.controls,labels_path=a.labels,control_poses=a.control_poses,rendered_roots=a.rendered_roots)
    print(json.dumps({k:v for k,v in r.items() if k not in ('run_checks','coverage','media_checks')},indent=2))
    raise SystemExit(0 if r['state']=='passed' else 1)
