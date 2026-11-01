#!/usr/bin/env python3
"""Render the unchanged OPD monitor with deterministic, explicitly synthetic data."""
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
SRC, DIST = ROOT / 'source', ROOT / 'dist'
BANNER = '''<div id="demo-banner" style="background:#4b360f;color:#ffe0a3;border-bottom:1px solid #a97723;padding:12px 20px;font:600 14px/1.5 system-ui,sans-serif">ORIGINAL TEMPLATE PREVIEW · All plots, comparisons, progress and statuses are dummy data. No training run is connected.</div>'''


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(text)
    temp.replace(path)


def value(key, update, variant=0):
    wave = math.sin(update * .31 + variant) * .035 + math.sin(update * 1.9) * .012
    decay = math.exp(-update / 35)
    scale = 1 + variant * .12
    if key == 'train/step_time': return round((620 + 40 * math.sin(update / 7) + (800 if update < 3 else 0)) * scale, 2)
    if 'learning_rate' in key: return 3e-6
    if 'phase_seconds' in key or 'publication_seconds' in key:
        return round(({'rollout': 300, 'teacher': 80, 'critic': 55, 'learner': 170, 'publication': 15}.get(key.split('/')[1].split('_')[0], 70)) * (1 + wave), 2)
    if 'latency_p50' in key: return round(.48 + .15 * (1-decay) + wave, 4)
    if 'latency_p90' in key: return round(1.45 + .2 * (1-decay) + wave, 4)
    if 'words_q2' in key: return round(5 + 2 * (1-decay) + 4 * wave, 3)
    if 'words_q_' in key: return round(7.5 + 3 * (1-decay) + 4 * wave, 3)
    if 'talk_q2' in key: return round(74 + 4 * math.sin(update / 12), 3)
    if 'silence_pct' in key: return round(26 + 5 * math.sin(update / 12), 3)
    if 'cap_pct' in key: return round(25 + 5 * math.sin(update / 12), 3)
    if 'pct' in key: return round(max(0, 1 + 10 * wave), 3)
    if 'words_a' in key: return round(35 + 8 * (1-decay) + 5 * wave, 3)
    if 'answer_seconds' in key: return round(17 + 4 * (1-decay) + wave, 3)
    if 'tail' in key: return round(2.8 + .5 * math.sin(update / 12), 3)
    if 'ratio' in key: return round(1.4 + .2 * math.sin(update / 12), 4)
    if key.startswith('grad_norm/'): return round(0.7 + .2 * decay + wave, 5)
    if 'alerts_breached' in key: return 0
    if 'absolute' in key or key.startswith('signal/'): return round(3 + 4 * decay + wave, 5)
    return round(max(.0001, (.1 + .6 * decay + wave) * scale), 6)


def build():
    cfg = json.loads((SRC / 's3branches_opd.json').read_text())
    now = time.time()
    keys = list(dict.fromkeys([k for k, ev, _ in cfg['lead'] if not ev] +
                [c[0] for c in cfg['timing']['charts']] +
                ['kl/text', 'kl/audio', 'signal/credit_absolute_sum']))
    for chart in cfg['timing']['charts']:
        if len(chart) > 2:
            keys.extend(k for k, _ in chart[2] if k not in keys)

    def run(name, label, count, variant=0):
        return dict(run=name, label=label, short=label, jobs=[], job_runs={}, started=True,
                    step=count, max_steps=110, rows_per_step=384, epoch_rows=84319,
                    epoch_steps=84319 / 384, ppm={'speech_rows_per_step':384,'speech_epoch_rows':84319},
                    train_cols=['step','rows']+keys,
                    train=[[u,u*384]+[value(k,u,variant) for k in keys] for u in range(1,count+1)],
                    cols=['step','rows','genghis/successes'],
                    pts=[[u,u*384,round(4 + u / 22 + variant)] for u in range(0,count+1,10)],
                    t_last=now-80, t_step1=now-count*620, mean50_s=620, mean_s=620,
                    updated='synthetic preview', done=count>=110, state='DEMO',
                    last_save=[now-1800,60 if count>=60 else max(0,count-10)], save_every_min=90,
                    save_wall=True, last_eval=[now-1800,60],eval_every_min=90,
                    export_steps=[110], next_export_step=110)

    main = run('ppm/opd/nextrun','nextrun · DEMO',62)
    main['branches'] = []
    for i,branch in enumerate(cfg['branches']):
        ref = run(branch['run'],branch['label']+' · DEMO',110 if 'param' in branch['run'] else 48,i+1)
        for k in ['color','color_dark','archived','from_step','fit','rel']:
            if k in branch: ref[k] = branch[k]
        main['branches'].append(ref)
    main['lead'] = cfg['lead']
    main['xwin'] = cfg['xwin']
    main['diff'] = [[k+':t', k] for k in keys] + [['genghis/successes:e','genghis/successes']]
    main['base_line'] = {'label':'synthetic base · 4/20','vals':{'genghis/successes':4}}
    timing = cfg['timing']
    thresholds = {}
    for name,(op,limit) in timing['thresholds'].items():
        key = 'timing_rollout/' + (name[:-4] if name.endswith('_low') else name)
        thresholds.setdefault(key,[]).append([op,limit,name])
    main['timing'] = dict(charts=timing['charts'],thr=thresholds,
                          base={k:value(k,0) for k in timing['base']},base_label='synthetic base',
                          param={k:{str(u):value(k,u,1) for u in (40,80,110)} for k in timing['param']},
                          param_label='synthetic reference checkpoints',
                          latest={'update':62,'breached':[],'streak':0,'t':now,'src':'SYNTHETIC'},alert=None)
    payload = json.dumps(main,separators=(',',':'))
    write(SRC / 's3live_opd.json',payload)
    result = subprocess.run([sys.executable,str(SRC/'build_stage3.py'),str(SRC),'--no-doc'],
                            env=dict(os.environ,S3_PAGE='opd',CANOPY_ROOT=str(ROOT/'empty'),CURVES_DIR=str(SRC)),
                            capture_output=True,text=True,check=True)
    if 'replace anchor missing:' in result.stdout:
        raise RuntimeError(result.stdout)
    DIST.mkdir(exist_ok=True)
    html = (SRC/'stage3_opd.html').read_text().replace('<body>','<body>'+BANNER,1)
    for name in ['index.html','stage3_opd.html']:
        write(DIST/name,html)
    write(DIST/'s3live_opd.json',payload)
    write(DIST/'ver_s3_opd.txt',(SRC/'ver_s3_opd.txt').read_text())
    shutil.copyfile(SRC/'opd_prog.js',DIST/'opd_prog.js')
    phase_specs = [('rollout','rollout',300),('teacher','qwen / teacher score',80),('critic','critic / category score',55),('learner','learner',170),('publication','publication',15)]
    phases = []
    start = now-505
    for i,(key,label,duration) in enumerate(phase_specs):
        phases.append(dict(key=key,label=label,ref=duration,state='done' if i<3 else 'cur' if i==3 else 'todo',start=start if i<=3 else None,end=start+duration if i<3 else None))
        start += duration
    write(DIST/'opd_prog.json',json.dumps(dict(now=now,root='SYNTHETIC PREVIEW',job='',max=110,update=63,completed=62,
          finished=False,idle=False,update_start=now-505,phase=3,phases=phases,
          sub={'done':41,'total':100,'unit':'demo batches','text':'Synthetic progress example'},
          per_update_for_run_eta=620,ref_src='synthetic example',src='demo',log='dummy data',log_age=0)))
    write(DIST/'decision_opd.txt','DEMO ONLY · Synthetic curves and phase timings for inspecting the original monitor. No run is connected.\n')
    write(DIST/'opd_samples'/'index.html','''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{background:#171a21;color:#e6e8ec;font:14px/1.6 system-ui;padding:24px}section{border:1px solid #343b48;border-radius:10px;padding:20px;max-width:860px;margin:16px 0}small{color:#a7aebc}h2{font-size:18px}</style><h2>Samples · preview placeholder</h2><p>The supplied monitor links to a separate rollout viewer. That viewer and its media are not included in the handoff.</p><section><small>DUMMY SAMPLE · UPDATE 60</small><p><b>Prompt lane:</b> Example input prompt would appear here.</p><p><b>Response lane:</b> Generated text and audio would appear here.</p><p><small>No real rollout, audio, or pose data is loaded.</small></p></section></html>''')
    print('Original-template dummy preview rebuilt',flush=True)


if __name__ == '__main__':
    build()
