#!/usr/bin/env python3
"""Pose dashboard using the retained monitor's page-local extension mechanism."""
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
SRC,DIST=ROOT/'source',ROOT/'dist'
sys.path.insert(0,str(ROOT.parent))
PAGE='pose_opd_credit'
CRITICAL=[['kl/text','Text KL · full token distribution'],['kl/lexical','Lexical KL · conditional word distribution'],
 ['kl/category','Category KL · word / special-token mass'],['kl/audio/tower','Audio tower KL · codebook 0'],
 ['kl/audio/depth_mean','Audio depth KL · mean of codebooks 1–31'],['kl/pose/tower','Pose tower KL · codebook 0'],
 ['kl/pose/depth_mean','Pose depth KL · mean of codebooks 1–31']]
COMPONENTS=[['inp','Input embeddings'],['layers','Backbone'],['head','Text head'],['vadapter','Audio input adapter'],
 ['vgain','Audio gain'],['tower_s','Audio tower'],['dec_s','Audio depth decoder'],['tower_p','Pose tower'],
 ['dec_p','Pose depth decoder'],['decp_bb','Backbone → pose depth bridge']]
EXTRAS=[[f'kl/{s}/cb{i}',f'{s.title()} · codebook {i} KL'] for s in ('audio','pose') for i in range(32)]
EXTRAS += [['loss/direct','Direct objective · weighted KL'],['credit/absolute_sum','Credit coefficients · absolute sum'],
 ['policy/logprob_mean_abs_difference','Actor / replay log-probability gap · mean'],
 ['policy/logprob_max_abs_difference','Actor / replay log-probability gap · maximum'],['train/step_time','Seconds per update']]


def write(path,text):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp');temp.write_text(text);temp.replace(path)


def synthetic_row(u):
    wave=1+.05*math.sin(u*.71)+.025*math.sin(u*2.3);decay=math.exp(-u/38)
    r={'kl/lexical':(.12+.45*decay)*wave,'kl/category':(.018+.085*decay)*wave}
    r['kl/text']=r['kl/category']+.68*r['kl/lexical']
    for stream,mult in [('audio',1),('pose',1.3)]:
        for cb in range(32):
            r[f'kl/{stream}/cb{cb}']=round(mult*(.08+.5*decay)*wave*(1 if cb==0 else .35+.025*cb)*(1+.035*math.sin(u*.21+cb)),7)
        r[f'kl/{stream}/tower']=r[f'kl/{stream}/cb0']
        r[f'kl/{stream}/depth_mean']=sum(r[f'kl/{stream}/cb{i}'] for i in range(1,32))/31
    for i,(component,_) in enumerate(COMPONENTS):
        factor=[.22,1,.35,.16,.06,.43,.25,.49,.29,.12][i]
        r[f'grad_norm/direct/{component}']=factor*(.48+.8*decay)*wave
        r[f'grad_norm/credit/{component}']=factor*(.21+.35*decay)*(1+.14*math.sin(u*.42+i))
    r.update({'gradient_norm':1.05*(.7+decay)*wave,
      'loss/direct':(r['kl/text']+sum(.5*r[f'kl/{s}/tower']+.5*r[f'kl/{s}/depth_mean'] for s in ('audio','pose')))/3,
      'credit/absolute_sum':(10+40*decay)*wave,'policy/logprob_mean_abs_difference':.009*wave,
      'policy/logprob_max_abs_difference':.045*wave,'train/step_time':620+25*math.sin(u*.3)+(600 if u<3 else 0)})
    return r


def demo_doc(now):
    rows=[synthetic_row(u) for u in range(1,63)];keys=list(rows[0])
    return dict(run='pose-opd-credit · preview',label='Pose OPD credit · DEMO',short='Pose OPD credit · DEMO',
      jobs=[],job_runs={},started=True,step=62,max_steps=110,rows_per_step=160,epoch_rows=None,
      ppm={'speech_rows_per_step':160},train_cols=['step','rows']+keys,
      train=[[i,i*160]+[r[k] for k in keys] for i,r in enumerate(rows,1)],cols=['step','rows'],pts=[],
      t_last=now-80,t_step1=now-62*620,mean50_s=620,mean_s=620,updated='synthetic preview',done=False,state='DEMO',
      last_save=[now-1800,60],save_every_min=120,save_wall=True,export_steps=[110],next_export_step=110,
      branches=[],xwin=[0,110],missing_telemetry=[])


def config():
    original=json.loads((SRC/'s3branches_opd.json').read_text());template=(SRC/'ppm.tpl.html').read_text()
    old_tabs=template[template.index('function aTabs(){'):template.index('const TABSA=')]
    pairs=[]
    for old,new in original['replace']:
        if old.startswith('const PAGE='):
            new="const PAGE='pose_opd_credit',P8=false,P14=false,PA=true,PV=false,LSP='pose_opd_credit:';"
        elif old=='<title>PPM live curves</title>':new='<title>Pose OPD credit · run dashboard</title>'
        elif old.startswith('  return [[') or old=="let TAB=LS.get('ppm5_tab','critical');":continue
        elif old=='function hdrA(){':
            new='''function smpTab(){const g=document.querySelector('.grid[data-tab="samples"]');if(!g||g.querySelector('iframe'))return;const f=document.createElement('iframe');f.src='samples/index.html';f.title='Four samples per rollout: dataset audio, generated audio and pose';f.style.cssText='width:100%;height:1100px;border:1px solid var(--line);border-radius:10px;background:var(--bg)';g.appendChild(f)}\nfunction hdrA(){'''
        elif old.startswith('for(const [k,lab] of '):new="for(const [k,lab] of [['save','next checkpoint']]){"
        new=new.replace('once nextrun/metrics.jsonl has rows','once the pose run emits metrics')
        if old=='setInterval(poll,60000);setInterval(hdr,30000);':
            new+='\n'+(ROOT/'page_notes.js').read_text()
        pairs.append([old,new])
    pairs += [[old_tabs,(ROOT/'chart_layout.js').read_text()],['(50-step mean)','(recent update median)'],
              ['50-step mean step time','recent update median'],
              ['D.s3=await r.json();','D.s3=await r.json();if(window.poseMissingNotes)window.poseMissingNotes();'],
              ['(median of last ≤10 updates, warm-up 1-2 excluded)','(median of recent updates)']]
    return dict(title='Pose OPD credit',main_run='pose-opd-credit',page_tpl='ppm.tpl.html',default_xm='st',
      no_forecast=True,refs=False,branches=[],replace=pairs,strip=['gb','ib','mb','s2ref','hestia','dmour','dmtxd'])


def progress(now,demo,doc):
    if not demo:
        return {'now':now,'idle':not doc.get('latest_event'),'root':doc['run'],'max':doc.get('max_steps'),
                'completed':doc.get('step') or 0,'finished':doc.get('done',False),'phase':doc.get('phase'),
                'rollout_progress':doc.get('rollout_progress'),'latest_event':doc.get('latest_event'),
                'observed_at':doc.get('observed_at'),'mean_s':doc.get('mean50_s'),'state':doc.get('state')}
    specs=[('preparation','data + audio',40),('publication','actor build + publish',45),
           ('rollout','rollout',300),('replay','teacher + replay',220),('optimizer','optimizer',15)]
    phases=[];start=now-235
    for i,(key,label,duration) in enumerate(specs):
        phases.append(dict(key=key,label=label,ref=duration,state='done' if i<2 else 'cur' if i==2 else 'todo',
          start=start if i<=2 else None,end=start+duration if i<2 else None));start+=duration
    return dict(now=now,root='SYNTHETIC PREVIEW',max=110,update=63,completed=62,finished=False,idle=False,
      update_start=now-235,phase=2,phases=phases,
      sub={'done':3751,'total':7502,'unit':'submitted frames','text':'Demo rollout · microbatch 1 · four sample rows selected'},
      per_update_for_run_eta=620,ref_src='synthetic phase timings',src='demo',log='dummy data',log_age=0)


def refresh_progress():
    settings=json.loads((ROOT/'settings.json').read_text());demo=settings['mode']=='demo';now=time.time()
    if demo:doc=demo_doc(now)
    else:
        from pose_doc import build_doc
        doc=build_doc(Path(settings['run_dir']) if settings.get('run_dir') else None)
    write(DIST/'opd_prog.json',json.dumps(progress(now,demo,doc)))


def build():
    now=time.time();settings=json.loads((ROOT/'settings.json').read_text());demo=settings['mode']=='demo'
    if settings['mode'] not in ('demo','live'):raise ValueError('mode must be demo or live')
    if demo:doc=demo_doc(now)
    else:
        from pose_doc import build_doc
        from pose_metrics import enrich_doc
        run=Path(settings['run_dir']) if settings.get('run_dir') else None;method={}
        if run and (run/'config.json').is_file():
            saved=json.loads((run/'config.json').read_text());method=saved.get('method',{})
        elif run and (run/'preparation.json').is_file():
            saved=json.loads((run/'preparation.json').read_text());method=saved.get('identity',{}).get('method',{})
        doc=enrich_doc(build_doc(run),method);doc.update(short='Pose OPD credit',label='Pose OPD credit',branches=[])
    doc.update(critical_charts=CRITICAL,grad_components=COMPONENTS,extra_charts=EXTRAS)
    payload=json.dumps(doc,separators=(',',':'),allow_nan=False)
    write(SRC/f's3branches_{PAGE}.json',json.dumps(config(),ensure_ascii=False))
    write(SRC/f's3live_{PAGE}.json',payload)
    result=subprocess.run([sys.executable,str(SRC/'build_stage3.py'),str(SRC),'--no-doc'],
      env=dict(os.environ,S3_PAGE=PAGE,CANOPY_ROOT=str(ROOT/'empty'),CURVES_DIR=str(SRC)),capture_output=True,text=True,check=True)
    if 'replace anchor missing:' in result.stdout:raise RuntimeError(result.stdout)
    DIST.mkdir(exist_ok=True)
    banner=('POSE OPD CREDIT · PREVIEW — Curves, progress and sample media are synthetic. No training run is connected.' if demo
            else 'POSE OPD CREDIT · Read-only training telemetry')
    html=(SRC/f'stage3_{PAGE}.html').read_text().replace('<body>','<body><div id="demo-banner" style="background:#4b360f;color:#ffe0a3;padding:10px 16px;font:600 13px/1.6 system-ui">'+banner+'</div>',1)
    start=html.index('const D=')+len('const D=');end=html.index(';\nconst PAGE=',start)
    safe=json.dumps(json.loads(html[start:end]),separators=(',',':')).replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
    html=html[:start]+safe+html[end:]
    for name in ['index.html',f'stage3_{PAGE}.html','stage3_opd.html']:write(DIST/name,html)
    write(DIST/f's3live_{PAGE}.json',payload)
    write(DIST/f'ver_s3_{PAGE}.txt',(SRC/f'ver_s3_{PAGE}.txt').read_text())
    # Existing original-preview tabs can detect the new schema and reload.
    write(DIST/'s3live_opd.json',payload)
    write(DIST/'ver_s3_opd.txt',(SRC/f'ver_s3_{PAGE}.txt').read_text())
    prog_js=((SRC/'opd_prog.js') if demo else (ROOT/'live_progress.js')).read_text().replace('our last ≤10 updates, warm-up updates 1-2 excluded','recent recorded updates')
    write(DIST/'opd_prog.js',prog_js);write(DIST/'opd_prog.json',json.dumps(progress(now,demo,doc)))
    write(DIST/f'decision_{PAGE}.txt','')
    if demo and (ROOT/'samples'/'build_demo.py').is_file():
        spec=importlib.util.spec_from_file_location('sample_demo',ROOT/'samples'/'build_demo.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.build_demo(DIST/'samples')
    elif not demo:
        # A live page must never retain the previous demo's manifest or media.
        staging=DIST/'samples-next'
        if staging.exists():shutil.rmtree(staging)
        staging.mkdir()
        if settings.get('samples_dir'):
            source=Path(settings['samples_dir']).resolve()
            if source==DIST.resolve() or DIST.resolve() in source.parents:raise ValueError('samples_dir must be outside the served output')
            shutil.copytree(source,staging,dirs_exist_ok=True)
        else:
            for name in ('index.html','samples.css','samples.js'):shutil.copyfile(ROOT/'samples'/name,staging/name)
            write(staging/'manifest.json',json.dumps({'schema_version':1,'synthetic':False,'rollouts':[]}))
        previous=DIST/'samples-previous'
        if previous.exists():shutil.rmtree(previous)
        if (DIST/'samples').exists():(DIST/'samples').rename(previous)
        staging.rename(DIST/'samples')
        if previous.exists():shutil.rmtree(previous)
    print('Pose dashboard rebuilt · '+settings['mode'],flush=True)


if __name__=='__main__':build()
