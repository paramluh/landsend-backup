"""Adapt actual pose events to the retained Amu five-phase progress schema."""
from pathlib import Path
import statistics
from pose_doc import read_jsonl, read_json, timestamp, version

SPECS = [('preparation','prepare + publish','data_audio_begin'),
         ('rollout','rollout','actor_rollout_begin'),
         ('teacher','teacher score','teacher_update_begin'),
         ('learner','learner','replay_microbatch_begin'),
         ('optimizer','optimizer','optimizer_begin')]


def adapt(now, doc):
    root = Path(doc['run_dir']) if doc.get('run_dir') else None
    rows = read_jsonl(root/'events-rank0.jsonl', []) if root else []
    rows = [r for r in rows if r.get('rank', 0) == 0]
    groups = {}
    for row in rows:
        u = version(row)
        if u is None: continue
        if row.get('event') == 'phase' and row.get('phase') == 'data_audio_begin':
            groups = {k:v for k,v in groups.items() if k < u}
        groups.setdefault(u, []).append(row)
    completed = doc.get('step') or 0
    active = max(groups, default=completed or None)
    def bounds(events):
        found = {}
        for row in events:
            t = timestamp(row)
            if row.get('event') == 'phase' and t is not None:
                found.setdefault(row.get('phase'), t)
            if row.get('event') == 'update' and t is not None:
                found['update_end'] = t
        return [found.get(s[2]) for s in SPECS] + [found.get('update_end')]
    references = [[] for _ in SPECS]
    for u, events in sorted(groups.items()):
        if u > completed: continue
        b = bounds(events)
        if b[-1] is None: continue
        for i in range(5):
            if b[i] is not None and b[i+1] is not None and b[i+1] >= b[i]:
                references[i].append(b[i+1]-b[i])
    refs = [statistics.median(v[-10:]) if v else None for v in references]
    events = groups.get(active, [])
    b = bounds(events)
    phases = [dict(key=k,label=label,ref=refs[i],start=b[i],end=b[i+1],
                   state='done' if b[i] is not None and b[i+1] is not None else 'cur' if b[i] is not None else 'todo')
              for i,(k,label,_) in enumerate(SPECS)]
    current = next((i for i,p in reversed(list(enumerate(phases))) if p['state']=='cur'), None)
    failed = doc.get('state') == 'failed'
    stopped = failed or doc.get('state') in ('bounded_or_stopped','stopped')
    terminal_at = doc.get('supervisor_finished_at') or doc.get('observed_at') if stopped else None
    sub = None
    rollout = doc.get('rollout_progress')
    if current == 1 and rollout:
        config = read_json(root/'config.json', []) if root else {}
        prefill = config.get('method', {}).get('prefill_frames', 25)
        physical = rollout.get('frame_clock') == 'physical'
        if physical:
            prefill = rollout.get('generated_start_frame', prefill)
        elapsed = rollout.get('seconds_elapsed')
        rate = max(0, rollout['current_frame']-prefill)/elapsed if elapsed and elapsed > 0 else None
        if physical:
            team = rollout.get('actor_group_gpus', config.get('actor_group_gpus', 4))
            text = f'rank 0 {team}-GPU team · submitted physical frames (includes prefill; excludes media-delay slots)'
            if rate is not None: text += f' · {rate:.1f} submitted frames/s'
        else:
            text = 'rank 0 GPU pair · submitted model-frame positions (includes prefill and media delay)'
            if rate is not None: text += f' · {rate:.1f} submitted steps/s'
        sub = dict(done=rollout['current_frame'],total=rollout['total_frames'],
                   unit='frames' if physical else 'positions',text=text)
    elif current is not None:
        sub = {'text':'rank 0 phase events; no inner-phase counter is emitted'}
    last = doc.get('observed_at')
    return dict(now=now,root=doc['run'],max=doc.get('max_steps'),update=active,completed=completed,
                finished=doc.get('done',False),idle=b[0] is None,starting=bool(rows),phase=current,phases=phases,sub=sub,
                update_start=b[0],per_update_for_run_eta=doc.get('mean50_s'),
                ref_src='median of up to 10 completed updates from this run' if any(refs) else 'unavailable until an update completes',
                src='events-rank0.jsonl',log='events-rank0.jsonl',log_age=max(0,now-last) if last else None,
                failed=failed,stopped=stopped,terminal_at=terminal_at,state=doc.get('state'),
                supervisor_exit_code=doc.get('supervisor_exit_code'))


def renderer(original):
    """Keep Amu layout, CSS and cadence; add only honest terminal/unknown ETA guards."""
    edits = [
        ('not started yet${P.job?', "${P.stopped?(P.failed?'failed':'stopped'):P.starting?'initializing':'not started yet'}${P.job?"),
        ('const n=NOW(),ph=P.phases||[]', 'const n=P.stopped?(P.terminal_at||P.now):NOW(),ph=P.phases||[]'),
        ('o.nowx=P.finished?P.max:', 'if(P.stopped||!ph.length||ph.some(p=>p.ref==null)){o.eta=null;o.reta=null;o.ufr=null;}o.nowx=P.finished?P.max:'),
        ('if(P&&P.idle)return null;', 'if(P&&(P.idle||P.stopped))return null;'),
        ('update ${P.update} of ${P.max}</span>', '${P.stopped?(P.failed?"failed · ":"stopped · "):""}update ${P.update} of ${P.max}</span>'),
        ('${ft(o.eta-o.n)}', '${ft(o.eta==null?null:o.eta-o.n)}'),
        ('our last ≤10 updates, warm-up updates 1-2 excluded', 'up to 10 completed updates'),
    ]
    for old,new in edits:
        if old not in original: raise ValueError('Amu renderer anchor missing: '+old)
        original=original.replace(old,new)
    return original
