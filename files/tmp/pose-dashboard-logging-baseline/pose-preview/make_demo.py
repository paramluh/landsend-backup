#!/usr/bin/env python3
"""Pose dashboard using the retained monitor's page-local extension mechanism."""
import importlib.util
import hashlib
import gzip
import json
import math
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from urllib.parse import urlsplit

ROOT=Path(__file__).resolve().parent
SRC,DIST=ROOT/'source',ROOT/'dist'
POSE_SOURCE_ROOT=Path('/home/param/datasets/aloe-duplex/pose/data/files/none')
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
 ['policy/logprob_max_abs_difference','Actor / replay log-probability gap · maximum'],['train/step_time','Seconds per update'],['learning_rate','Learning rate · applied to this update']]


def write(path,text):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp');temp.write_text(text);temp.replace(path)


def compress_sample_json(source, staging):
    """Cache lossless HTTP gzip variants outside the decoder's immutable data.

    The 120-second pose documents are roughly 11 MB each. Level-six gzip
    removes over half their transfer bytes without changing a coordinate or
    confidence value. Rebuilds only link existing variants, rather than
    recompressing every historical rollout once a minute.
    """
    cache=source.parent/'.sample-http-cache'
    for directory in (source/'assets').glob('update-*-*'):
        if not re.fullmatch(r'update-\d{6}-[0-9a-f]{16}',directory.name):continue
        for path in directory.glob('*.json'):
            stat=path.stat()
            if stat.st_size<4096:continue
            identity=hashlib.sha256(f'gzip6-v1:{path.resolve()}:{stat.st_mtime_ns}:{stat.st_size}'.encode()).hexdigest()
            cached=cache/f'{identity}.json.gz'
            if not cached.is_file():
                cache.mkdir(parents=True,exist_ok=True)
                temporary=cached.with_suffix('.tmp')
                with path.open('rb') as incoming,temporary.open('wb') as output:
                    with gzip.GzipFile(filename='',mode='wb',fileobj=output,compresslevel=6,mtime=0) as zipped:
                        shutil.copyfileobj(incoming,zipped)
                temporary.replace(cached)
            target=staging/path.relative_to(source)
            target=target.with_name(target.name+'.gzip6.gz')
            try:os.link(cached,target)
            except OSError:shutil.copy2(cached,target)


def source_pose_normalization(sample, pose_root=POSE_SOURCE_ROOT):
    """Recover the generated speaker's exact frame-zero camera mapping."""
    row=sample.get('row_id','')
    match=re.fullmatch(r'ALOE/s2handover-none/([0-9a-f-]{36})/(seg\d{2})/c00',row)
    audio=sample.get('source_audio') or {}
    if not match or audio.get('row_key')!=row or audio.get('frame0')!=0:return None
    session,segment=match.groups()
    audio_match=re.fullmatch(re.escape(session+'/'+segment)+r'/SPEAKER_0([01])\.flac',audio.get('flac',''))
    if not audio_match:return None
    input_speaker=int(audio_match.group(1))
    input_side='assistant' if input_speaker==0 else 'user'
    output_side='user' if input_speaker==0 else 'assistant'
    if audio.get('side')!=input_side or sample.get('orientation')!=f'{input_side} → {output_side}':return None
    output_speaker=f'SPEAKER_0{1-input_speaker}'
    path=pose_root/session/segment/f'w00.{output_speaker}.pose.json'
    if not path.is_file():return None
    source=json.loads(path.read_text())
    if (source.get('status')!='ok' or source.get('session_id')!=session
            or source.get('seg')!=segment or source.get('speaker')!=output_speaker):
        raise ValueError(f'Inconsistent source pose metadata: {path}')
    normal=source['normalization'];root=normal['root0_yx_normalized'];width=normal['shoulder_width0_normalized']
    if len(root)!=2 or not all(math.isfinite(v) for v in [*root,width]) or width<=0:
        raise ValueError(f'Invalid source pose normalization: {path}')
    return {'root_yx':root,'shoulder_width':width}


def prepare_motion_index(path, cache_root, chunk_frames=125, source_normalization=None):
    """Prepare exact, independently loadable five-second pose JSON chunks.

    This is a browser representation only. It neither rounds coordinates nor
    changes the decoder's pose documents. Cache publication is atomic; later
    site builds only link these already-prepared small immutable files.
    """
    stat=path.stat()
    metadata=json.dumps(source_normalization,sort_keys=True,separators=(',',':'),allow_nan=False)
    identity=hashlib.sha256(f'motion-v2-gzip6:{path.resolve()}:{stat.st_mtime_ns}:{stat.st_size}:{chunk_frames}:{metadata}'.encode()).hexdigest()
    cached=cache_root/identity
    # The metadata hash keeps camera mappings immutable even after a correction.
    suffix='-'+hashlib.sha256(metadata.encode()).hexdigest()[:12] if source_normalization else ''
    name=path.stem+'.motion-v2'+suffix
    if (cached/(name+'.json')).is_file():return cached,name+'.json'
    cache_root.mkdir(parents=True,exist_ok=True)
    document=json.loads(path.read_text())
    frames=document['frames'];confidence=document.get('confidence')
    count=len(frames)
    if not count or not document.get('joints') or len(document['parents'])!=len(document['joints']):
        raise ValueError(f'Invalid pose structure: {path}')
    if confidence is not None and len(confidence)!=count:raise ValueError(f'Invalid confidence length: {path}')
    bounds=[math.inf,-math.inf,math.inf,-math.inf]
    z_up=document.get('up_axis')=='z'
    for frame in frames:
        for point in frame:
            x=point[0]+point[1 if z_up else 2]*.25
            y=point[2 if z_up else 1]
            if not math.isfinite(x) or not math.isfinite(y):raise ValueError(f'Non-finite pose: {path}')
            bounds[0]=min(bounds[0],x);bounds[1]=max(bounds[1],x)
            bounds[2]=min(bounds[2],y);bounds[3]=max(bounds[3],y)
    index={key:value for key,value in document.items() if key not in ('frames','confidence')}
    index.update(schema_version=1,frame_count=count,chunk_frames=chunk_frames,bounds=bounds,chunks=[])
    if source_normalization is not None:index['source_normalization']=source_normalization
    with tempfile.TemporaryDirectory(prefix='.motion-',dir=cache_root) as temporary:
        work=Path(temporary)
        def emit(filename,value):
            body=json.dumps(value,separators=(',',':'),allow_nan=False).encode()
            (work/filename).write_bytes(body)
            (work/(filename+'.gzip6.gz')).write_bytes(gzip.compress(body,compresslevel=6,mtime=0))
        for number,start in enumerate(range(0,count,chunk_frames)):
            end=min(start+chunk_frames,count)
            filename=f'{name}-{number:03d}.json'
            chunk={'first_frame':start,'frames':frames[start:end]}
            if confidence is not None:chunk['confidence']=confidence[start:end]
            emit(filename,chunk)
            index['chunks'].append({'url':filename,'first_frame':start,'frame_count':end-start})
        # Write the index only after every chunk has been written successfully.
        emit(name+'.json',index)
        try:work.rename(cached)
        except FileExistsError:
            if not (cached/(name+'.json')).is_file():raise
    return cached,name+'.json'


def publish_motion_indexes(source, staging):
    """Add progressive pose URLs to the served manifest, leaving source intact."""
    manifest_path=staging/'manifest.json'
    if not manifest_path.is_file():return
    manifest=json.loads(manifest_path.read_text())
    seen={}
    for rollout in manifest.get('rollouts',[]):
        for sample in rollout.get('samples',[]):
            value=sample.get('pose_url')
            if not isinstance(value,str):continue
            parsed=urlsplit(value);relative=Path(parsed.path)
            if (parsed.scheme or parsed.netloc or relative.is_absolute() or '..' in relative.parts
                    or len(relative.parts)!=3 or relative.parts[0]!='assets'
                    or not re.fullmatch(r'update-\d{6}-[0-9a-f]{16}',relative.parts[1])
                    or relative.suffix!='.json'):
                continue
            if value not in seen:
                path=source/relative
                if not path.is_file():continue
                cached,index_name=prepare_motion_index(path,source.parent/'.sample-motion-cache',
                    source_normalization=source_pose_normalization(sample))
                destination=staging/relative.parent;destination.mkdir(parents=True,exist_ok=True)
                for asset in cached.iterdir():
                    target=destination/asset.name
                    try:os.link(asset,target)
                    except OSError:shutil.copy2(asset,target)
                seen[value]=(relative.parent/index_name).as_posix()
            sample['pose_index_url']=seen[value]
    write(manifest_path,json.dumps(manifest,separators=(',',':'),allow_nan=False))


def publish_samples(staging, destination):
    """Keep immutable media paths continuously readable during live refresh.

    Renaming the complete samples directory previously made every media URL
    briefly disappear once per minute. Publish asset folders first, then
    atomically replace only mutable files (the manifest last). Existing
    content-addressed assets remain available to already-open players.
    """
    existing_manifest=destination/'manifest.json'
    try:was_demo=json.loads(existing_manifest.read_text()).get('synthetic') is True
    except (OSError,ValueError):was_demo=False
    if was_demo:shutil.rmtree(destination)
    destination.mkdir(parents=True,exist_ok=True)
    source_assets=staging/'assets'
    if source_assets.exists():
        target_assets=destination/'assets';target_assets.mkdir(exist_ok=True)
        for directory in source_assets.iterdir():
            target=target_assets/directory.name
            if not target.exists():
                directory.rename(target)
                continue
            # A gzip variant can be added to an older immutable rollout.
            for path in directory.rglob('*'):
                relative=path.relative_to(directory);out=target/relative
                if path.is_dir():out.mkdir(parents=True,exist_ok=True)
                elif not out.exists():
                    out.parent.mkdir(parents=True,exist_ok=True)
                    path.rename(out)
    for path in staging.iterdir():
        if path.is_file() and path.name!='manifest.json':path.replace(destination/path.name)
    if (staging/'manifest.json').is_file():(staging/'manifest.json').replace(existing_manifest)
    shutil.rmtree(staging)


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
      'policy/logprob_max_abs_difference':.045*wave,'train/step_time':240+10*math.sin(u*.3)+(120 if u<3 else 0)})
    return r


def demo_doc(now):
    rows=[synthetic_row(u) for u in range(1,27)];keys=list(rows[0])
    return dict(run='pose-opd-credit · preview',label='Pose OPD credit · DEMO',short='Pose OPD credit · DEMO',
      jobs=[],job_runs={},started=True,step=26,max_steps=43,rows_per_step=320,epoch_rows=13566,
      ppm={'speech_rows_per_step':320},train_cols=['step','rows']+keys,
      train=[[i,i*320]+[r[k] for k in keys] for i,r in enumerate(rows,1)],cols=['step','rows'],pts=[],
      t_last=now-80,t_step1=now-26*240,mean50_s=240,mean_s=240,updated='synthetic preview',done=False,state='DEMO',
      last_save=[now-1800,20],save_every_min=120,save_wall=True,export_steps=[43],next_export_step=43,
      branches=[],xwin=[0,43],missing_telemetry=[])


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
            new=new.replace('setInterval(poll,60000);setInterval(hdr,30000);','setInterval(poll,5000);setInterval(hdr,5000);')
            from amu_progress import __file__ as progress_module
            progress_version=hashlib.sha256((SRC/'opd_prog.js').read_bytes()+Path(progress_module).read_bytes()).hexdigest()[:16]
            sample_version=hashlib.sha256(b''.join((ROOT/'samples'/name).read_bytes() for name in ('index.html','samples.css','samples.js'))).hexdigest()[:16]
            new+='\n'+(ROOT/'page_notes.js').read_text()+'\npoll();\n/* Amu progress asset version '+progress_version+'; sample renderer '+sample_version+' */'
        pairs.append([old,new])
    pairs += [[old_tabs,(ROOT/'chart_layout.js').read_text()],['(50-step mean)','(recent update median)'],
              ['50-step mean step time','recent update median'],
              ['D.s3=await r.json();','D.s3=await r.json();if(window.poseMissingNotes)window.poseMissingNotes();'],
              ['if(PA&&ASIGF()!==ASIG){location.reload();return}','/* Pose tabs are fixed; new metric columns do not require a page reload. */'],
              ['const LIVE=x=>!!(x&&!x.done&&x.t_last',"const LIVE=x=>!!(x&&!x.done&&!['failed','bounded_or_stopped','stopped'].includes(x.state)&&x.t_last"],
              ['run=s.step!=null&&!s.done,left=',"run=s.step!=null&&!s.done&&!['failed','bounded_or_stopped','stopped'].includes(s.state),left="],
              ["${m||'–'} s/update","${m!=null?(+m).toFixed(1):'–'} s/update"],
              ['(median of last ≤10 updates, warm-up 1-2 excluded)','(median of recent updates)']]
    return dict(title='Pose OPD credit',main_run='pose-opd-credit',page_tpl='ppm.tpl.html',default_xm='st',
      no_forecast=True,refs=False,branches=[],replace=pairs,strip=['gb','ib','mb','s2ref','hestia','dmour','dmtxd'])


def progress(now,demo,doc):
    if not demo:
        settings=json.loads((ROOT/'settings.json').read_text());collector={}
        if settings.get('collector_health'):
            try:collector=json.loads(Path(settings['collector_health']).read_text())
            except (OSError,ValueError):collector={'status':'unavailable'}
        from amu_progress import adapt
        result=adapt(now,doc);result['collector']=collector
        return result

    specs=[('preparation','data + audio',10),('publication','actor build + publish',15),
           ('rollout','rollout',130),('replay','teacher + replay',80),('optimizer','optimizer',5)]
    phases=[];start=now-95
    for i,(key,label,duration) in enumerate(specs):
        phases.append(dict(key=key,label=label,ref=duration,state='done' if i<2 else 'cur' if i==2 else 'todo',
          start=start if i<=2 else None,end=start+duration if i<2 else None));start+=duration
    return dict(now=now,root='SYNTHETIC PREVIEW',max=43,update=27,completed=26,finished=False,idle=False,
      update_start=now-95,phase=2,phases=phases,
      sub={'done':751,'total':1502,'unit':'submitted frames','text':'Demo rollout · 16 rows per GPU pair · four sample rows selected'},
      per_update_for_run_eta=240,ref_src='synthetic phase timings',src='demo',log='dummy data',log_age=0)


def live_doc(settings):
    from pose_doc import build_doc
    from pose_metrics import enrich_doc
    run=Path(settings['run_dir']).resolve() if settings.get('run_dir') else None;method={}
    if run and (run/'config.json').is_file():
        saved=json.loads((run/'config.json').read_text());method=saved.get('method',{})
    elif run and (run/'preparation.json').is_file():
        saved=json.loads((run/'preparation.json').read_text());method=saved.get('identity',{}).get('method',{})
    doc=enrich_doc(build_doc(run,{'run':settings.get('run_name','pose-opd-credit')}),method)
    doc.update(short='Pose OPD credit',label='Pose OPD credit',branches=[],
               critical_charts=CRITICAL,grad_components=COMPONENTS,extra_charts=EXTRAS)
    return doc


def refresh_progress():
    settings=json.loads((ROOT/'settings.json').read_text());demo=settings['mode']=='demo';now=time.time()
    if demo:doc=demo_doc(now)
    else:
        doc=live_doc(settings)
        # Publish curves/header and progress from the SAME immutable snapshot
        # on every5s refresh, independently of the slower HTML/media rebuild.
        payload=json.dumps(doc,separators=(',',':'),allow_nan=False)
        write(DIST/f's3live_{PAGE}.json',payload)
        write(DIST/'s3live_opd.json',payload)
    write(DIST/'opd_prog.json',json.dumps(progress(now,demo,doc)))


def build():
    now=time.time();settings=json.loads((ROOT/'settings.json').read_text());demo=settings['mode']=='demo'
    if settings['mode'] not in ('demo','live'):raise ValueError('mode must be demo or live')
    if demo:doc=demo_doc(now)
    else:
        doc=live_doc(settings)
    doc.update(critical_charts=CRITICAL,grad_components=COMPONENTS,extra_charts=EXTRAS)
    payload=json.dumps(doc,separators=(',',':'),allow_nan=False)
    write(SRC/f's3branches_{PAGE}.json',json.dumps(config(),ensure_ascii=False))
    write(SRC/f's3live_{PAGE}.json',payload)
    result=subprocess.run([sys.executable,str(SRC/'build_stage3.py'),str(SRC),'--no-doc'],
      env=dict(os.environ,S3_PAGE=PAGE,CANOPY_ROOT=str(ROOT/'empty'),CURVES_DIR=str(SRC)),capture_output=True,text=True,check=True)
    if 'replace anchor missing:' in result.stdout:raise RuntimeError(result.stdout)
    DIST.mkdir(exist_ok=True)
    banner=('POSE OPD CREDIT · PREVIEW — Curves, progress and sample media are synthetic. No training run is connected.' if demo
            else 'POSE OPD CREDIT · LIVE · Read-only production telemetry · Unavailable metrics and samples remain empty')
    if not demo and doc.get('state')=='failed':
        banner='POSE OPD CREDIT · TRAINING ATTEMPT FAILED · Metrics and samples below belong to the stopped attempt'
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
    from amu_progress import renderer
    prog_js=(SRC/'opd_prog.js').read_text()
    if not demo:prog_js=renderer(prog_js)
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
            # Decoder assets are immutable once their manifest is published.
            # Hard links avoid recopying all previous audio every refresh.
            def link_or_copy(src,dst):
                if 'assets' in Path(src).parts:
                    try:return os.link(src,dst)
                    except OSError:pass
                return shutil.copy2(src,dst)
            shutil.copytree(source,staging,dirs_exist_ok=True,copy_function=link_or_copy,
                            ignore=shutil.ignore_patterns('.*'))
            compress_sample_json(source,staging)
            publish_motion_indexes(source,staging)
        else:
            for name in ('index.html','samples.css','samples.js'):shutil.copyfile(ROOT/'samples'/name,staging/name)
            write(staging/'manifest.json',json.dumps({'schema_version':1,'synthetic':False,'rollouts':[]}))
        # Media is produced by the decoder, while the website owns its renderer.
        # Always publish the current renderer, including when samples_dir exists.
        for name in ('index.html','samples.css','samples.js'):
            shutil.copyfile(ROOT/'samples'/name,staging/name)
        publish_samples(staging,DIST/'samples')
    print('Pose dashboard rebuilt · '+settings['mode'],flush=True)


if __name__=='__main__':build()
