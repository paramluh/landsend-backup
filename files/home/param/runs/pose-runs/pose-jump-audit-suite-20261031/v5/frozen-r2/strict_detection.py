"""V5 conservative whole-pose step review; older detections are never modified.

A large speed can be an ordinary gesture. Require an abrupt departure from the
preceding trajectory in the face AND EACH shoulder, a persistent head/torso
configuration change, and a discontinuous trajectory preferred at three scales.
This is pose-only triage, not a perceptual defect classifier.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time

import numpy as np

VERSION = 'persistent-head-torso-step-v5-r2'


@dataclass(frozen=True)
class Config:
    confidence_min: float = .2
    canvas_margin: float = .02
    min_visible_head: int = 3
    prior_frames: int = 8
    post_frames: int = 12
    noise_floor_shoulder_widths: float = .0015
    head_step_shoulder_widths: float = .015
    shoulder_step_shoulder_widths: float = .005
    each_shoulder_step_shoulder_widths: float = .004
    onset_innovation_ratio_min: float = 4.
    head_single_frame_to_prior_speed_min: float = 5.
    single_frame_fraction_of_two_frame_path_min: float = 2/3
    persistence_start_frame: int = 6
    persistence_end_frame: int = 12
    retained_offset_fraction_min: float = .6
    fit_half_windows_frames: tuple = (4, 6, 8)
    continuous_over_step_rms_min: float = 2**.5
    step_fit_rms_over_offset_max: float = .2
    transition_intervals: tuple = (1, 2)
    context_frames: int = 25
    context_exclusion_frames: int = 3
    context_peak_ratio_min: float = 2.5
    min_context_frames: int = 8
    merge_frames: int = 10


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rms(a):
    """RMS displacement across joints, preserving the joint configuration."""
    return np.sqrt(np.mean(np.sum(a*a, axis=-1), axis=-1))


def prepare(frames, width=256, height=480, config=None):
    c = config or Config()
    p = np.asarray(frames, dtype=np.float64)
    if p.ndim != 3 or p.shape[1:] != (55, 3) or not np.isfinite(p).all():
        raise ValueError('Expected finite normalized pose [T,55,3]')
    xy = p[:, :7, :2]*[width, height]
    visible = ((p[:, :7, 2] >= c.confidence_min)
               & (p[:, :7, :2] >= -c.canvas_margin).all(-1)
               & (p[:, :7, :2] <= 1+c.canvas_margin).all(-1))
    widths = np.linalg.norm(xy[:, 5]-xy[:, 6], axis=-1)
    ok = visible[:, 5] & visible[:, 6] & (widths > width*.05)
    scale = float(np.median(widths[ok])) if ok.any() else width*.5
    return {'p': p, 'xy': xy, 'velocity': np.diff(xy, axis=0), 'visible': visible,
            'scale': scale, 'width': width, 'height': height}


def fit_models(xy, frame, groups, scale, config):
    """Same stable anchors, same observations, four parameters in each model."""
    fits=[]
    for duration in config.transition_intervals:
        for half in config.fit_half_windows_frames:
            offsets=np.arange(-half,half)
            offsets=offsets[(offsets<0)|(offsets>=duration-1)]
            t=(offsets+.5)/half
            continuous=np.column_stack((np.ones(len(t)),t,t*t,t*t*t))
            step=np.column_stack((np.ones(len(t)),t,t>=0,t*(t>=0)))
            fit={'transition_intervals':duration,'half_window_frames':half,'fit_observations':len(t),
                 'parameters_per_coordinate_each_model':4}
            for name,ids in groups.items():
                y=xy[frame+offsets][:,ids].reshape(len(t),-1)
                errors={}
                for model,X in [('continuous',continuous),('step',step)]:
                    residual=(X@np.linalg.lstsq(X,y,rcond=None)[0]-y).reshape(len(t),len(ids),2)
                    errors[model]=float(np.sqrt(np.mean(np.sum(residual*residual,axis=-1))))
                fit[name]={'continuous_rms_px':errors['continuous'],'step_rms_px':errors['step'],
                           'continuous_over_step_rms':errors['continuous']/max(errors['step'],scale*.0005),
                           'anchor_indices':ids.tolist()}
            fits.append(fit)
    return fits


def evaluate(data, frame, config=None, full_diagnostics=True):
    c = config or Config()
    xy, v, visible, scale = (data[k] for k in ('xy', 'velocity', 'visible', 'scale'))
    f = int(frame)
    row = {'frame': f, 'passes': False, 'score': 0., 'rejection_reasons': [], 'metrics': {}}
    reasons, m = row['rejection_reasons'], row['metrics']
    if f-c.prior_frames-1 < 0 or f+1+c.persistence_end_frame >= len(xy):
        reasons.append('insufficient_immediate_context');return row
    # Use the same confidently visible anchors throughout the immediate fit and
    # persistence windows. Hand confidence/entry cannot create a detection.
    stable = visible[f-c.prior_frames-1:f+c.persistence_end_frame+2].all(axis=0)
    head = np.flatnonzero(stable[:5]);shoulders = np.array([5, 6])
    m['visible_head_count'] = int(len(head))
    if len(head) < c.min_visible_head or not stable[5:7].all():
        reasons.append('insufficient_stable_visible_head_and_shoulders');return row
    groups = {'head': head, 'shoulders': shoulders}
    noise = scale*c.noise_floor_shoulder_widths
    prior = v[f-c.prior_frames-1:f-1]
    drift = np.median(prior, axis=0)
    innovation = v[f-1]-drift
    for name, ids in groups.items():
        amplitude = float(rms(innovation[ids]))
        background = float(np.percentile(rms(prior[:, ids]-drift[ids]), 90))
        ratio = amplitude/max(background, noise)
        m[name+'_onset_innovation_px'] = amplitude
        m[name+'_prior_innovation_p90_px'] = background
        m[name+'_onset_innovation_ratio'] = ratio
        if name == 'head' and ratio < c.onset_innovation_ratio_min:
            reasons.append('head_onset_continues_prior_trajectory')
        first=float(rms(v[f-1,ids]));second=float(rms(v[f,ids]))
        dominance=first/max(first+second,noise)
        m[name+'_single_frame_fraction_of_two_frame_path']=dominance
        if dominance<c.single_frame_fraction_of_two_frame_path_min:
            reasons.append(name+'_transition_not_dominated_by_single_frame')
        if name=='head':
            prior_speed=float(np.percentile(rms(prior[:,ids]),90))
            m['head_single_frame_to_prior_speed']=first/max(prior_speed,noise)
            if m['head_single_frame_to_prior_speed']<c.head_single_frame_to_prior_speed_min:
                reasons.append('head_jump_not_large_relative_to_immediate_track_motion')
    for j, name in ((5, 'left_shoulder'), (6, 'right_shoulder')):
        amplitude = float(np.linalg.norm(innovation[j]))
        background = float(np.percentile(np.linalg.norm(prior[:, j]-drift[j], axis=-1), 90))
        ratio = amplitude/max(background, noise)
        m[name+'_onset_innovation_ratio'] = ratio
        if ratio < c.onset_innovation_ratio_min:
            reasons.append(name+'_onset_continues_prior_trajectory')
    if reasons and not full_diagnostics:
        row['diagnostics_stage']='onset_rejected';return row

    variants = []
    for duration in c.transition_intervals:
        end = f+duration-1
        step = xy[end]-xy[f-1]
        offsets = np.arange(c.persistence_start_frame, c.persistence_end_frame+1)
        samples = end+offsets
        raw = xy[samples]-xy[f-1]
        detrended = raw-(samples-f+1)[:, None, None]*drift
        variant = {'transition_intervals': duration, 'endpoint_frame': end, 'groups': {}, 'rejection_reasons': []}
        for name, ids in groups.items():
            delta = step[ids]
            norm2 = float(np.sum(delta*delta))
            amplitude = float(np.sqrt(norm2/len(ids)))
            minimum = scale*(c.head_step_shoulder_widths if name == 'head' else c.shoulder_step_shoulder_widths)
            raw_projection = np.sum(raw[:, ids]*delta, axis=(1, 2))/max(norm2, 1e-12)
            corrected_projection = np.sum(detrended[:, ids]*delta, axis=(1, 2))/max(norm2, 1e-12)
            stats = {'step_rms_px': amplitude,
                     'raw_retained_median': float(np.median(raw_projection)),
                     'detrended_retained_median': float(np.median(corrected_projection)),
                     'raw_retained_by_frame': raw_projection.tolist(),
                     'detrended_retained_by_frame': corrected_projection.tolist()}
            variant['groups'][name] = stats
            if amplitude < minimum:variant['rejection_reasons'].append(name+'_step_too_small')
            if stats['raw_retained_median'] < c.retained_offset_fraction_min:
                variant['rejection_reasons'].append(name+'_returns_toward_prior_pose')
            if stats['detrended_retained_median'] < c.retained_offset_fraction_min:
                variant['rejection_reasons'].append(name+'_offset_explained_by_prior_motion')
        each = np.linalg.norm(step[5:7], axis=-1)
        variant['each_shoulder_step_px'] = each.tolist()
        if np.any(each < scale*c.each_shoulder_step_shoulder_widths):
            variant['rejection_reasons'].append('both_shoulders_do_not_change_configuration')
        variant['passes_persistence'] = not variant['rejection_reasons']
        variants.append(variant)
    row['persistence_variants'] = variants
    passing_variants = [x for x in variants if x['passes_persistence']]
    if not passing_variants:reasons.append('no_persistent_head_and_torso_offset')

    # Broad context rejects repeatedly erratic motion. Immediate trajectory fits
    # below permit a new smooth velocity after a genuine discontinuity; demanding
    # a still head afterward falsely rejected user-confirmed rollout243.
    for side, inds in [('before', np.arange(max(0, f-c.context_frames-1), f-c.context_exclusion_frames-1)),
                       ('after', np.arange(f+c.context_exclusion_frames, min(len(v), f+c.context_frames)))]:
        if len(inds) < c.min_context_frames:
            reasons.append(side+'_insufficient_context');continue
        for name, ids in groups.items():
            reference = max(float(rms((xy[f+d-1]-xy[f-1])[ids])) for d in c.transition_intervals)
            background = float(np.percentile(rms(v[inds][:, ids]), 90))
            ratio = reference/max(background, noise)
            m[side+'_'+name+'_context_peak_ratio'] = ratio
            if ratio < c.context_peak_ratio_min:reasons.append(side+'_'+name+'_repeated_or_continuing_motion')

    fits=fit_models(xy,f,groups,scale,c)
    for fit in fits:
        margins = []
        for name in groups:
            a = fit[name]
            variant=next(x for x in variants if x['transition_intervals']==fit['transition_intervals'])
            amplitude=variant['groups'][name]['step_rms_px']
            a['step_rms_over_offset'] = a['step_rms_px']/max(amplitude, noise)
            margins += [a['continuous_over_step_rms']/c.continuous_over_step_rms_min,
                        c.step_fit_rms_over_offset_max/max(a['step_rms_over_offset'], 1e-12)]
        fit['v5_min_margin'] = float(min(margins))
        fit['v5_qualifies'] = bool(min(margins) >= 1.)
    row['trajectory_fits'] = fits
    matching_durations=[variant['transition_intervals'] for variant in passing_variants
                        if all(any(x['transition_intervals']==variant['transition_intervals'] and x['half_window_frames']==half and x['v5_qualifies']
                                   for x in fits) for half in c.fit_half_windows_frames)]
    scale_pass=bool(matching_durations)
    row['passing_transition_intervals']=matching_durations
    if not scale_pass:reasons.append('step_not_preferred_and_smooth_on_both_sides_at_all_scales')
    margins = [m['head_onset_innovation_ratio']/c.onset_innovation_ratio_min,
               m['left_shoulder_onset_innovation_ratio']/c.onset_innovation_ratio_min,
               m['right_shoulder_onset_innovation_ratio']/c.onset_innovation_ratio_min]
    margins += [m['head_single_frame_to_prior_speed']/c.head_single_frame_to_prior_speed_min]
    margins += [m[g+'_single_frame_fraction_of_two_frame_path']/c.single_frame_fraction_of_two_frame_path_min for g in groups]
    duration_margins=[]
    for variant in variants:
        duration=variant['transition_intervals']
        values=[variant['groups'][g][key]/c.retained_offset_fraction_min for g in groups
                for key in ('raw_retained_median','detrended_retained_median')]
        values += [x['v5_min_margin'] for x in fits if x['transition_intervals']==duration]
        values += [variant['groups']['head']['step_rms_px']/(scale*c.head_step_shoulder_widths),
                   variant['groups']['shoulders']['step_rms_px']/(scale*c.shoulder_step_shoulder_widths),
                   min(variant['each_shoulder_step_px'])/(scale*c.each_shoulder_step_shoulder_widths)]
        duration_margins.append(min(values))
    margins += [max(duration_margins)]
    margins += [value/c.context_peak_ratio_min for key,value in m.items() if key.endswith('_context_peak_ratio')]
    row['score'] = float(min(margins))
    row['passes'] = not reasons
    row['diagnostics_stage']='complete'
    return row


def scan(frames, *, fps=25., prefill_seconds=5.28, run_id=None, config=None, diagnostic_frames=()):
    c = config or Config();data = prepare(frames, config=c)
    v, scale = data['velocity'], data['scale']
    start = max(int(np.ceil(prefill_seconds*fps)), c.prior_frames+1)
    stop = len(data['xy'])-c.persistence_end_frame-2
    all_frames = np.arange(start, max(start, stop+1))
    # Necessary amplitude gate for either a1- or2-interval step; applying it to
    # every eligible frame makes this a population scan, not a v3-pool filter.
    viable = np.zeros(len(v), dtype=bool)
    for duration in c.transition_intervals:
        delta = data['xy'][duration:]-data['xy'][:-duration]
        # Safe upper bound for any valid subset: inactive/occluded anchors must
        # never dilute a real movement of the minimum3 visible face anchors.
        head_upper=np.sqrt(np.sum(delta[:,:5]*delta[:,:5],axis=(1,2))/c.min_visible_head)
        ok = ((head_upper >= scale*c.head_step_shoulder_widths)
              & (rms(delta[:, 5:7]) >= scale*c.shoulder_step_shoulder_widths)
              & (np.linalg.norm(delta[:, 5], axis=-1) >= scale*c.each_shoulder_step_shoulder_widths)
              & (np.linalg.norm(delta[:, 6], axis=-1) >= scale*c.each_shoulder_step_shoulder_widths))
        viable[:len(ok)] |= ok
    selected = {int(f) for f in all_frames if viable[f-1]}
    selected.update(int(f) for f in diagnostic_frames if start <= f <= stop)
    forced=set(diagnostic_frames)
    rows = [evaluate(data, f, c, full_diagnostics=f in forced) for f in sorted(selected)]
    accepted = [r for r in rows if r['passes']]
    clusters = []
    for row in accepted:
        if not clusters or row['frame']-clusters[-1][-1]['frame'] > c.merge_frames:clusters.append([row])
        else:clusters[-1].append(row)
    events = []
    for cluster in clusters:
        r = dict(max(cluster, key=lambda x:x['score']))
        r.update(run_id=run_id, time_s=r['frame']/fps, equivalent_frames=[x['frame'] for x in cluster])
        events.append(r)
    counts = Counter(reason for row in rows for reason in row['rejection_reasons'])
    return {'version': VERSION, 'run_id': run_id, 'fps': fps, 'config': asdict(c),
            'scale_shoulder_width_px': scale, 'eligible_frames': len(all_frames),
            'eligible_frame_start': start, 'eligible_frame_end': stop,
            'amplitude_gate_frames': sum(viable[all_frames-1]) if len(all_frames) else 0,
            'evaluated_frames': len(rows), 'candidate_events': len(events), 'events': events,
            'rejection_reason_counts': dict(counts), 'frame_diagnostics': rows,
            'interpretation': 'Conservative pose-only candidates. Neither acceptance nor rejection is a confirmed perceptual label.'}


def analyze_cohort(root, output, controls_path=None):
    c = Config();events=[];old_rows=[];source_rows=[];reason_counts=Counter()
    total_eligible=total_evaluated=analyzed=0
    checker_hash=digest(__file__)
    for path in sorted((root/'runs').glob('*/pose_overlay.json')):
        rd=path.parent;s=json.loads((rd/'summary.json').read_text())
        old=json.loads((rd/'detection.json').read_text())
        pose=json.loads(path.read_text())
        forced=[e['frame'] for e in old['events']]
        result=scan(pose['frames'], fps=pose.get('fps',25.), prefill_seconds=s['prefill_frames']/12.5,
                    run_id=rd.name, config=c, diagnostic_frames=forced)
        sources={'pose_sha256':digest(path),'tokens_sha256':digest(rd/'tokens.npz'),
                 'summary_sha256':digest(rd/'summary.json'),'v3_detection_sha256':digest(rd/'detection.json')}
        result.update(checker_sha256=checker_hash,sources=sources)
        (rd/'strict-detection-v5.json').write_text(json.dumps(result,indent=2,default=lambda x:int(x))+'\n')
        analyzed+=1;total_eligible+=result['eligible_frames'];total_evaluated+=result['evaluated_frames']
        events.extend(result['events']);reason_counts.update(result['rejection_reason_counts'])
        source_rows.append({'run_id':rd.name,**sources})
        diagnostics={x['frame']:x for x in result['frame_diagnostics']}
        for e in old['events']:
            match=[x['frame'] for x in result['events'] if any(abs(q-e['frame']) <=2 for q in x['equivalent_frames'])]
            d=diagnostics.get(e['frame'])
            old_rows.append({'run_id':rd.name,'frame':e['frame'],'time_s':e['time_s'],'retained':bool(match),
                             'v5_frames':match,'rejection_reasons':d['rejection_reasons'] if d else ['outside_eligible_context'],
                             'score':d['score'] if d else 0.})
        if analyzed%100==0:print(json.dumps({'progress_runs':analyzed,'events':len(events)}),flush=True)
    controls=json.loads(controls_path.read_text()) if controls_path else None
    out={'version':VERSION,'checker_sha256':checker_hash,'config':asdict(c),
         'created':time.time(),'root':str(root),'analyzed_rollouts':analyzed,'eligible_frames':total_eligible,
         'evaluated_amplitude_plausible_frames':total_evaluated,'candidate_events':len(events),
         'candidate_rollouts':len({e['run_id'] for e in events}),'events':events,
         'old_pool_review':{'events':len(old_rows),'retained':sum(r['retained'] for r in old_rows),
                            'rejected':sum(not r['retained'] for r in old_rows),'rows':old_rows},
         'rejection_reason_counts':dict(reason_counts),'sources':source_rows,'controls':controls,
         'interpretation':'Experimental conservative full-frame pose scan. Two user-confirmed positives inform controls; user145 is negative. Other prior candidates are unconfirmed. No IDs are excluded. All earlier detections/manifests remain archived unchanged. Review actual rendered windows before claiming visible defects.'}
    output.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:out[k] for k in ('version','analyzed_rollouts','eligible_frames','candidate_events','candidate_rollouts')}),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--controls',type=Path);args=ap.parse_args();analyze_cohort(args.root,args.output,args.controls)
