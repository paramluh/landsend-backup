"""Compare whole-node conversational rollout layouts with the same 48 rows.

baseline: two concurrent threads running (student pair + partner pair), 24 rows each.
fanout3: one student pair, three partner pairs, 48 = 16 + 16 + 16 rows.
fanout2: one student pair, two partner pairs, 48 = 24 + 24 rows (6-GPU diagnostic).

Example --candidates baseline:25:33:0,fanout3:25:33:0,fanout3:24:33:5.713
means layout:student_backbone_cut:partner_backbone_cut:synthetic_resident_GiB.
An optional :serial or :threaded suffix controls host submission. A further
:38 suffix selects the first38 prepared rows as a labeled throughput diagnostic.
Sampling is shortened to 375 frames; actor and audio cache capacity remains1500.
Resident reservations are synthetic allocations, never proof of training fit.
Production uses independent rank-leader processes; cross-check their timings
before attributing a same-process baseline improvement to the layout itself.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import socket
import threading
import time
import traceback

import numpy as np

from .teacher_partner_benchmark import (
    CudaPhaseRecorder, _audit_result, _build_model, _release_engines,
    _storage_identity, _sync, _write_json, memory_report,
)


@dataclass(frozen=True)
class Candidate:
    layout: str
    student_cut: int
    partner_cut: int
    reserve_gib: float
    concurrent: bool = True
    rows: int = 48

    @property
    def name(self):
        return (f'{self.layout}-s{self.student_cut}-p{self.partner_cut}-r{self.reserve_gib:g}'
                +('' if self.concurrent else '-serial')+('' if self.rows==48 else f'-n{self.rows}'))


def parse_candidate(value):
    parts=value.split(':')
    if len(parts) not in (4,5,6) or (len(parts)>=5 and parts[4] not in ('serial','threaded')):
        raise ValueError('Candidate is layout:student_cut:partner_cut:reserve_gib[:serial|threaded[:rows]]')
    layout,student,partner,reserve=parts[:4]
    candidate=Candidate(layout,int(student),int(partner),float(reserve),len(parts)==4 or parts[4]=='threaded',
                        int(parts[5]) if len(parts)==6 else 48)
    if (layout not in ('baseline','fanout3','fanout2') or min(candidate.student_cut,candidate.partner_cut)<1
            or not math.isfinite(candidate.reserve_gib) or candidate.reserve_gib<0
            or not (3 if layout=='fanout3' else 2)<=candidate.rows<=48):
        raise ValueError('Invalid layout, backbone cut or reservation')
    return candidate


def groups_for(layout,rows=48):
    """Explicit contiguous row ownership; student row order never changes."""
    if layout=='baseline':
        half=rows//2
        return [dict(student=(0,1),partners=[(2,3)],begin=0,end=half,sizes=[half]),
                dict(student=(4,5),partners=[(6,7)],begin=half,end=rows,sizes=[rows-half])]
    if layout=='fanout3':
        sizes=[rows//3+(i<rows%3) for i in range(3)]
        return [dict(student=(0,1),partners=[(2,3),(4,5),(6,7)],begin=0,end=rows,sizes=sizes)]
    if layout=='fanout2':
        return [dict(student=(0,1),partners=[(2,3),(4,5)],begin=0,end=rows,sizes=[rows//2,rows-rows//2])]
    raise ValueError(f'Unknown layout: {layout}')


def run_groups(functions):
    """Drain both workers before cleanup, aborting bootstrap rendezvous on error."""
    if len(functions)==1:
        return [functions[0](None)]
    barrier=threading.Barrier(len(functions),timeout=300)
    def call(function):
        try:
            return function(barrier)
        except BaseException:
            barrier.abort()
            raise
    with ThreadPoolExecutor(max_workers=len(functions),thread_name_prefix='rollout-team') as pool:
        futures=[pool.submit(call,function) for function in functions]
        results=[None]*len(futures);errors=[]
        positions={future:i for i,future in enumerate(futures)}
        for future in as_completed(futures):
            try:results[positions[future]]=future.result()
            except BaseException as exc:errors.append(exc)
    if errors:
        # A peer's barrier exception must not hide the actual CUDA OOM, which
        # the candidate loop can record before testing the next placement.
        raise next((exc for exc in errors if not isinstance(exc,threading.BrokenBarrierError)),errors[0])
    return results


def result_fingerprint(results):
    """Exact token, heard-waveform and input-latent parity across scheduling."""
    digest=hashlib.sha256()
    def add(value):
        if isinstance(value,dict):
            for key in sorted(value):
                digest.update(str(key).encode());add(value[key])
        elif isinstance(value,(list,tuple)):
            for item in value:add(item)
        elif hasattr(value,'detach'):
            tensor=value.detach().cpu().contiguous()
            digest.update(str((tensor.dtype,tuple(tensor.shape))).encode())
            digest.update(tensor.view(__import__('torch').uint8).numpy().tobytes())
        else:
            array=np.asarray(value)
            digest.update(str((array.dtype,array.shape)).encode());digest.update(array.tobytes())
    for result in results:
        for rollout in result['rollouts']:
            add({name:getattr(rollout,name) for name in ('text','audio','pose','sampled_masks','behavior_logprobs')})
        for name in ('latents','waveforms','partner_rollouts'):add(result[name])
    return digest.hexdigest()


def arguments(argv=None):
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--candidates',default='baseline:25:33:0,fanout3:25:33:0,fanout2:25:33:0,fanout3:25:33:5.713,fanout3:24:33:5.713')
    p.add_argument('--more-candidates-file',type=Path,help='Optional JSON list of additional candidate strings, read after each candidate')
    p.add_argument('--frames',type=int,default=375)
    p.add_argument('--capacity-frames',type=int,default=1500)
    p.add_argument('--warmup-trials',type=int,default=1)
    p.add_argument('--warm-trials',type=int,default=2)
    p.add_argument('--row-offset',type=int,default=0)
    p.add_argument('--items-json',type=Path,help='Reuse a prior rows.json membership selection with its matching identity.json')
    p.add_argument('--prepared-rows',type=Path,help='Trusted local prepared-row pickle cache, written on first use')
    p.add_argument('--disable-partner-text-history',action='store_true')
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--no-graphs',action='store_true')
    p.add_argument('--profile-index',type=int,help='Additional bounded profile after this candidate; excluded from headline')
    p.add_argument('--profile-calls',type=int,default=16)
    p.add_argument('--serial-control-index',type=int,help='One extra serial-partner trial after this candidate; excluded from headline')
    p.add_argument('--qwen32-config',default='/home/param/models/imported-weights/ppm-811-17850-base-bf16/configs/Qwen3-32B')
    p.add_argument('--qwen4-config',default='/home/param/models/imported-weights/ppm-811-17850-base-bf16/configs/Qwen3-4B')
    p.add_argument('--voxtral',default='/home/param/models/imported-weights/aloe-ft-vox17127-nopersona/assets/voxtral-realtime')
    p.add_argument('--mimi',default='/home/param/models/mimi')
    args=p.parse_args(argv)
    try:
        args.candidates=[parse_candidate(value) for value in args.candidates.split(',')]
    except (ValueError,TypeError) as exc:
        p.error(str(exc))
    if len({c.name for c in args.candidates})!=len(args.candidates):p.error('Duplicate candidates')
    if args.capacity_frames<1500 or not 1<=args.frames<=args.capacity_frames:
        p.error('Full cache capacity must be at least1500 and cover the measured prefix')
    if args.warmup_trials<1 or args.warm_trials<1 or args.row_offset<0 or args.profile_calls<1:
        p.error('Require at least one warmup and warm trial, nonnegative row offset and positive profile bound')
    if args.profile_index is not None and not 0<=args.profile_index<len(args.candidates):
        p.error('Profile index outside candidate list')
    if args.serial_control_index is not None and not 0<=args.serial_control_index<len(args.candidates):
        p.error('Serial control index outside candidate list')
    return args


def prepare_rows(cfg,args,log):
    from .data import ProductionRows,prefix_seed
    from .train import rollout_seed
    selection_bytes=args.items_json.read_bytes() if args.items_json else None
    serialized_config=json.loads(json.dumps(cfg.to_dict()))
    key={'config':serialized_config,'frames':args.frames,'row_offset':args.row_offset,
        'data_sha256':hashlib.sha256(Path(__file__).with_name('data.py').read_bytes()).hexdigest(),
        'selection_sha256':hashlib.sha256(selection_bytes).hexdigest() if selection_bytes is not None else None}
    if args.prepared_rows and args.prepared_rows.exists():
        with args.prepared_rows.open('rb') as source:cached=pickle.load(source)
        if cached.get('format')!='teacher-partner-prepared-rows-v1' or cached['key']!=key:
            raise ValueError('Prepared-row cache differs from requested data/config/selection')
        log('prepared_rows_cache_hit',path=str(args.prepared_rows))
        return cached['rows'],cached['seeds'],cached['preparation']
    data=ProductionRows(cfg)
    if selection_bytes is not None:
        original=json.loads((args.items_json.parent/'identity.json').read_text())
        if original['config']!=serialized_config:raise ValueError('Prior selection was built with a different recipe')
        admitted=json.loads(selection_bytes)
        selected=admitted['selected_items'];eligible=admitted['eligible_rows']
        log('reuse_admitted_rows',path=str(args.items_json),selected_rows=len(selected))
    else:
        items=data.items();selected=items[args.row_offset:args.row_offset+48];eligible=len(items)
    if len(selected)!=48 or len({tuple(item) for item in selected})!=48:
        raise ValueError('Exactly48 distinct eligible dataset entries required')
    rows=[prefix_seed(data.load(item,rank=0,update=0,slot=i),args.frames) for i,item in enumerate(selected)]
    if any(int(row['conversation_frames'])!=args.frames for row in rows):raise ValueError('Dataset prefix shorter than requested')
    seeds=[rollout_seed(cfg.seed,0,0,i) for i in range(48)]
    preparation={'eligible_rows':eligible,'selected_items':selected,'seeds':seeds,
                 'prefill_frames':[int(r['prefill_frames']) for r in rows]}
    if args.prepared_rows:
        args.prepared_rows.parent.mkdir(parents=True,exist_ok=True)
        temporary=args.prepared_rows.with_name(args.prepared_rows.name+f'.tmp-{os.getpid()}')
        with temporary.open('wb') as output:
            os.chmod(temporary,0o600)
            pickle.dump({'format':'teacher-partner-prepared-rows-v1','key':key,'rows':rows,
                         'seeds':seeds,'preparation':preparation},output,protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(args.prepared_rows)
        log('prepared_rows_cache_written',path=str(args.prepared_rows),bytes=args.prepared_rows.stat().st_size)
    return rows,seeds,preparation


def _place_model(model,devices,cut):
    """Change only layer placement between candidates; preserve loaded weights."""
    if not 1<=cut<len(model.layers):raise ValueError('Backbone cut must leave layers on both GPUs')
    first,last=devices
    model.inp.to(first)
    placement=[first if i<cut else last for i in range(len(model.layers))]
    for layer,device in zip(model.layers,placement):layer.to(device)
    for name in ('head','tower_s','dec_s'):getattr(model,name).to(last)
    for name in ('tower_p','dec_p','decp_bb','pose_head','vadapter','vgain'):
        if hasattr(model,name):getattr(model,name).to(first)
    model._opd_backbone_cut=cut
    return placement


class ModelPool:
    """Keep checkpoint weights, parking conflicting GPU-pair roles on CPU."""
    def __init__(self,torch,M,native,specs,cfg,args,prefixes,log):
        self.torch,self.M,self.native,self.specs=torch,M,native,specs
        self.cfg,self.args,self.prefixes,self.log=cfg,args,prefixes,log
        self.models={};self.fronts={}

    def retain_for(self,candidates):
        """Release actors with no future use instead of copying them to CPU."""
        needed=set()
        for candidate in candidates:
            for group in groups_for(candidate.layout,candidate.rows):
                needed.add(('student',tuple(group['student'])))
                needed.update(('partner',tuple(pair)) for pair in group['partners'])
        for key in list(self.models):
            if key not in needed:
                self.log('release_unused_weights',role=key[0],devices=key[1])
                self.models.pop(key)
        gc.collect()
        for index in range(self.torch.cuda.device_count()):
            with self.torch.cuda.device(index):self.torch.cuda.empty_cache()

    def get(self,role,pair,cut):
        from .rollout import make_fastppm_class
        from voxtral_front import VoxtralFront
        key=(role,tuple(pair));devices=tuple(self.torch.device(f'cuda:{i}') for i in pair)
        for other_key,other in self.models.items():
            if other_key!=key and other['pair']==tuple(pair) and other['resident']:
                self.log('park_weights_begin',role=other_key[0],devices=pair)
                other['model'].to('cpu');other['resident']=False
                other['storage_identity']=_storage_identity(other['model'])
                _sync(self.torch,devices)
                for device in devices:
                    with self.torch.cuda.device(device):self.torch.cuda.empty_cache()
                self.log('park_weights_complete',role=other_key[0],devices=pair)
        if key not in self.models:
            self.log('load_weights_begin',role=role,devices=pair,cut=cut)
            model,layer_devices,cfgs=_build_model(self.torch,self.M,self.specs[role],devices,
                backbone_cut=cut,require_pose=role=='student',log=self.log)
            self.models[key]={'model':model,'layer_devices':layer_devices,'cfgs':cfgs,'devices':devices,
                'pair':tuple(pair),'cut':cut,'resident':True,'has_pose':role=='student',
                'prefix_capacity':self.prefixes[role],
                'actor_type':make_fastppm_class(self.native,require_pose=role=='student'),
                'initial_hook_ids':set(model.inp._forward_hooks),'storage_identity':_storage_identity(model)}
            self.log('load_weights_complete',role=role,devices=pair,cut=cut)
        value=self.models[key]
        if not value['resident'] or value['cut']!=cut:
            self.log('replace_weight_placement',role=role,devices=pair,cut=cut,checkpoint_reread=False)
            value['layer_devices']=_place_model(value['model'],devices,cut)
            value.update(resident=True,cut=cut,storage_identity=_storage_identity(value['model']))
        if pair[0] not in self.fronts:
            self.fronts[pair[0]]=VoxtralFront(self.args.voxtral,device=devices[0])
        value['front']=self.fronts[pair[0]]
        return value


def main(argv=None):
    args=arguments(argv)
    from .config import TrainingConfig
    from .train import checkpoint_spec,source_digest
    cfg=TrainingConfig.from_dict(json.loads(args.config.read_text()))
    if cfg.rollout_mode!='teacher_partner':raise ValueError('Recipe must select teacher_partner')
    if args.frames<=cfg.partner_prefill_max_frames+8:raise ValueError('Measured prefix must include generated conversation')
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'identity.json').exists():raise ValueError('Use a fresh output directory')
    log_lock=threading.Lock()
    def log(event,**fields):
        line=json.dumps({'event':event,'wall_time':time.time(),**fields},default=str,allow_nan=False)
        with log_lock:
            print(line,flush=True)
            with (args.output/'events.jsonl').open('a') as f:f.write(line+'\n')
    identity={'format':'teacher-partner-layout-benchmark-v1','host':socket.gethostname(),
        'source_sha256':source_digest(),'config':cfg.to_dict(),'arguments':vars(args),
        'student_rows_per_node':48,'capacity_frames':args.capacity_frames,'physical_frames':args.frames,
        'learner_optimizer_or_kl_scorers_resident':False,
        'synthetic_resident_note':'Optional uint8 allocations approximate resident shard bytes; this is not training-fit validation',
        'timing_scope':'whole-node48 student rows; baseline teams execute concurrently',
        'baseline_execution':'two threads in one process; cross-check independent production-rank processes for GIL overhead',
        'headline_profile_calls':0,'full_training_update_memory_validated':False}
    _write_json(args.output/'identity.json',identity)
    log('load_rows_begin')
    rows,seeds,preparation=prepare_rows(cfg,args,log)
    _write_json(args.output/'rows.json',preparation)
    log('load_rows_complete',eligible_rows=preparation['eligible_rows'],selected_rows=48)
    if args.prepare_only:
        _write_json(args.output/'result.json',{'identity':identity,'preparation':preparation,'gpu_benchmark_run':False});return
    import torch
    from . import runtime as R
    from .train import load_actor_native
    from .partner_fanout import rollout_teacher_partners
    if torch.cuda.device_count()<8:raise ValueError('Eight visible GPUs are required for the whole-node comparison')
    all_devices=tuple(torch.device(f'cuda:{i}') for i in range(8))
    M=R.load_native();M.QWEN32,M.QWEN4B_CFG=args.qwen32_config,args.qwen4_config
    native=load_actor_native(M)
    specs={'student':checkpoint_spec(cfg.student,'student'),
           'partner':checkpoint_spec(cfg.resolved_partner_checkpoint,'audio_teacher',gain_policy=cfg.audio_teacher_gain_policy)}
    prefixes={'student':max(int(r['P'][0]) for r in rows),
              'partner':max(int(r['partner']['P'][0]) for r in rows)}
    if cfg.fixed_sequence_frames is not None and cfg.conversation_prefix_frames is not None:
        p=cfg.fixed_sequence_frames-cfg.conversation_prefix_frames-cfg.speech_delay
        prefixes={role:max(prefix,p) for role,prefix in prefixes.items()}
    pool=ModelPool(torch,M,native,specs,cfg,args,prefixes,log)
    candidates=[]
    for index,candidate in enumerate(args.candidates):
        groups=groups_for(candidate.layout,candidate.rows);active_ids=sorted({d for g in groups for pair in [g['student'],*g['partners']] for d in pair})
        devices=tuple(all_devices[d] for d in active_ids)
        engines={};loaded={};reservations=[];recorder=None;results=None
        pool.retain_for(args.candidates[index:])
        record={'id':candidate.name,'layout':candidate.layout,'student_cut':candidate.student_cut,
            'partner_cut':candidate.partner_cut,'synthetic_resident_gib_per_active_gpu':candidate.reserve_gib,
            'concurrent_partner_submission':candidate.concurrent,
            'student_rows':candidate.rows,'same48_row_comparison':candidate.rows==48,
            'row_count_note':'Uses the first rows of the same48-row admission; lower-count cases are throughput diagnostics, not a production allocation.' if candidate.rows!=48 else None,
            'active_devices':active_ids,'active_gpus':len(active_ids),'groups':groups,'runs':[],'status':'setup',
            'inactive_gpu_note':'Previously loaded weights/frontends may remain on unused GPU pairs to avoid rereading checkpoints; their memory is included in whole-device reports, but they perform no rollout work.'}
        candidates.append(record)
        directory=args.output/'candidates'/candidate.name;directory.mkdir(parents=True,exist_ok=True)
        def save():
            _write_json(directory/'partial.json',record)
            _write_json(args.output/'partial.json',{'identity':identity,'candidates':candidates})
        try:
            log('candidate_begin',**{k:v for k,v in record.items() if k!='runs'})
            for gi,group in enumerate(groups):
                loaded[f's{gi}']=pool.get('student',group['student'],candidate.student_cut)
                for pi,pair in enumerate(group['partners']):loaded[f'p{gi}_{pi}']=pool.get('partner',pair,candidate.partner_cut)
            _sync(torch,all_devices)
            record['bare_model_memory']=memory_report(torch,all_devices)
            # Live allocations, with an explicit scope label, reserve the space
            # before cache/graph capture so the fit measurement is conservative.
            count=math.ceil(candidate.reserve_gib*2**30)
            if count:
                for device in devices:reservations.append(torch.empty(count,dtype=torch.uint8,device=device))
            record['reservation_memory']=memory_report(torch,all_devices)
            setup=[]
            for gi,group in enumerate(groups):
                sizes={f's{gi}':group['end']-group['begin'],**{f'p{gi}_{pi}':n for pi,n in enumerate(group['sizes'])}}
                for name,instances in sizes.items():
                    role=loaded[name];started=time.perf_counter()
                    before=memory_report(torch,role['devices'])
                    engines[name]=role['actor_type'](role['model'],role['layer_devices'],role['cfgs'],role['devices'],
                        n_inst=instances,sampler_backend=cfg.actor_sampler_backend,
                        depth_cache_positions=cfg.actor_depth_cache_positions,
                        attention_precision=cfg.actor_attention_precision,linear_backend=cfg.actor_linear_backend,
                        max_s=(args.capacity_frames+8)/12.5,p_max=role['prefix_capacity']+8)
                    engine=engines[name]
                    if not role['has_pose']:
                        engine.enable_rewind(window=8)
                        if args.disable_partner_text_history:
                            from .partner_faststate import disable_partner_text_history
                            log('partner_text_history_disabled',role=name,**disable_partner_text_history(engine))
                    _sync(torch,role['devices']);cache=memory_report(torch,role['devices'])
                    cache_seconds=time.perf_counter()-started;started=time.perf_counter()
                    if not args.no_graphs:engine.capture(f=25,P=role['prefix_capacity'])
                    _sync(torch,role['devices'])
                    item={'role':name,'instances':instances,'cache_seconds':cache_seconds,
                        'capture_seconds':time.perf_counter()-started,'before':before,'after_cache':cache,
                        'after_graph_capture':memory_report(torch,role['devices'])}
                    setup.append(item);log('actor_ready',candidate=candidate.name,**item)
            record['setup']=setup;record['status']='running';engine=None
            total=args.warmup_trials+args.warm_trials
            concurrent_fingerprint=None
            extra_profile=int(index==args.profile_index)
            for repeat in range(total+extra_profile+int(index==args.serial_control_index)):
                profiled=repeat==total and bool(extra_profile)
                serial_control=repeat==total+extra_profile
                cold=repeat<args.warmup_trials
                _sync(torch,all_devices)
                for device in all_devices:torch.cuda.reset_peak_memory_stats(device)
                if profiled:
                    recorder=CudaPhaseRecorder(torch,calls=args.profile_calls)
                    for name,engine in engines.items():recorder.attach(engine,name,loaded[name]['devices'])
                    engine=None
                def make_call(gi,group):
                    def call(barrier):
                        with torch.cuda.device(group['student'][0]),torch.inference_mode():
                            return rollout_teacher_partners(engines[f's{gi}'],
                                [engines[f'p{gi}_{pi}'] for pi in range(len(group['partners']))],
                                rows[group['begin']:group['end']],seeds[group['begin']:group['end']],
                                loaded[f's{gi}']['front'],[loaded[f'p{gi}_{pi}']['front'] for pi in range(len(group['partners']))],
                                args.mimi,graphs=not args.no_graphs,teacher_replay='block',concurrent=candidate.concurrent and not serial_control,
                                capacity_frames=args.capacity_frames,profile_calls=args.profile_calls if profiled else 0,
                                delivery_delay_samples=cfg.partner_audio_delay_samples,
                                startup_barrier=barrier,
                                progress=lambda **fields:log('progress',candidate=candidate.name,repeat=repeat,team=gi,**fields))
                    return call
                log('rollout_begin',candidate=candidate.name,repeat=repeat,cold=cold,profiled=profiled,serial_control=serial_control)
                started=time.perf_counter()
                try:
                    results=run_groups([make_call(gi,group) for gi,group in enumerate(groups)])
                    _sync(torch,all_devices);elapsed=time.perf_counter()-started
                finally:
                    if recorder:recorder.detach()
                audits=[_audit_result(result,rows[group['begin']:group['end']]) for group,result in zip(groups,results)]
                generated=sum(args.frames-int(r['prefill_frames']) for r in rows[:candidate.rows])
                generation=max(float(result['timings']['generation_seconds']) for result in results)
                warm_generation=max(float(result['timings']['warm_generation_seconds']) for result in results)
                warm_frames=sum(int(result['timings']['warm_student_frames'])*(group['end']-group['begin'])
                                for group,result in zip(groups,results))
                fingerprint=result_fingerprint(results)
                if repeat==total-1:concurrent_fingerprint=fingerprint
                run={'repeat':repeat,'cold':cold,'profiled':profiled,'serial_control':serial_control,'wall_seconds':elapsed,
                    'generation_critical_seconds':generation,'student_rows':candidate.rows,
                    'warm_generation_critical_seconds':warm_generation,'warm_loop_student_frames':warm_frames,
                    'warm_loop_audio_seconds_per_wall_second':warm_frames/12.5/warm_generation,
                    'generated_audio_frames':generated,'student_rows_per_wall_second':candidate.rows/elapsed,
                    'generated_audio_seconds_per_wall_second':generated/12.5/elapsed,
                    'generation_audio_seconds_per_wall_second':generated/12.5/generation,
                    'team_timings':[result['timings'] for result in results],
                    'memory':memory_report(torch,all_devices),'audit':audits,'result_sha256':fingerprint,
                    'serial_matches_concurrent':fingerprint==concurrent_fingerprint if serial_control else None,
                    'cuda_actor_phases':recorder.report() if recorder else None}
                record['runs'].append(run);log('rollout_complete',candidate=candidate.name,**run);save()
                if serial_control and fingerprint!=concurrent_fingerprint:
                    raise RuntimeError('Serial and concurrent partner schedules changed tokens, heard PCM, or input latents')
                results=None;recorder=None;gc.collect()
            warm=[r for r in record['runs'] if not r['cold'] and not r['profiled'] and not r['serial_control']]
            record.update(status='complete',warm_wall_seconds_median=float(np.median([r['wall_seconds'] for r in warm])),
                warm_generation_critical_seconds_median=float(np.median([r['warm_generation_critical_seconds'] for r in warm])))
            log('candidate_complete',candidate=candidate.name,warm_wall_seconds_median=record['warm_wall_seconds_median'])
        except torch.cuda.OutOfMemoryError as exc:
            record.update(status='out_of_memory',error=str(exc),memory_at_failure=memory_report(torch,all_devices))
            log('candidate_out_of_memory',candidate=candidate.name,error=str(exc));save()
        except Exception as exc:
            record.update(status='failed',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
            save();raise
        finally:
            if recorder:recorder.detach()
            results=recorder=engine=None
            reservations.clear()
            _release_engines(torch,engines,loaded,all_devices)
            record['memory_after_cache_release']=memory_report(torch,all_devices)
            save();_write_json(directory/'result.json',record)
        if args.more_candidates_file and args.more_candidates_file.exists():
            requested=json.loads(args.more_candidates_file.read_text())
            if not isinstance(requested,list):raise ValueError('Additional candidates file must contain a JSON list')
            known={item.name for item in args.candidates}
            for text in requested:
                extra=parse_candidate(text)
                if extra.name not in known:
                    args.candidates.append(extra);known.add(extra.name)
                    log('candidate_appended',candidate=extra.name)
    summary={'identity':identity,'preparation':preparation,'candidates':candidates,'gpu_benchmark_run':True,
        'all_candidates_succeeded':all(c['status']=='complete' for c in candidates),
        'full_training_update_seconds':None,'full_training_update_memory_validated':False,
        'extrapolation_note':'Thirty-second prefixes retain full cache memory but do not measure two-minute attention costs or distributed learner throughput.'}
    _write_json(args.output/'result.json',summary);log('benchmark_complete',output=str(args.output/'result.json'))


if __name__=='__main__':main()
