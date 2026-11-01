#!/usr/bin/env python3
"""Forward-KL-only distillation of production pose duplex on its Aloe dataset."""
from __future__ import annotations
import argparse,datetime,gc,json,math,os,signal,socket,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
from finetune.duplex_distill.recipe import load_recipe
from finetune.storage_paths import canonical_run_dir


def args_parser(argv=None):
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--mode',choices=('train','smoke'),default='train')
    p.add_argument('--data-root',type=Path,default=Path('/home/param/datasets/aloe-duplex'))
    p.add_argument('--student',type=Path,default=Path('/home/param/models/imported-weights/ppm-aloe-poseduplex-v5-final/export-498-dcp'))
    p.add_argument('--audio-teacher',type=Path,default=Path('/home/param/models/imported-weights/ppm-aloe-samsep-vox17127-final/export-249-dcp'))
    p.add_argument('--pose-teacher',type=Path,default=Path('/home/param/models/imported-weights/transcript-pose/xabi-original/export-200'))
    p.add_argument('--smoke-depth',type=int,default=2)
    p.add_argument('--smoke-steps',type=int,default=2)
    p.add_argument('--resume',type=Path)
    p.add_argument('--wandb-project',default='param-pose-llm')
    p.add_argument('--no-wandb',action='store_true')
    p.add_argument('--profile-steps',type=int,default=0,
                   help='Bounded scratch profile: full training math; no W&B, eval, or checkpoints; exits after 2-10 steps')
    p.add_argument('--text-chunk-size','--profile-text-chunk-size',dest='text_chunk_size',type=int,default=32,
                   help='Positions per full-vocabulary text-head chunk (default: 32)')
    p.add_argument('--text-direct-addmm','--profile-text-direct-addmm',dest='text_direct_addmm',action='store_true',
                   help='Use the independently validated direct-addmm text-head variant')
    p.add_argument('--micro-batch-size','--profile-micro-batch-size',dest='micro_batch_size',type=int,choices=(1,2),default=1,
                   help='Independent conversations packed per rank/update (default: 1)')
    argv=list(sys.argv[1:] if argv is None else argv)
    args=p.parse_args(argv)
    if args.text_chunk_size<1:
        p.error('--text-chunk-size must be positive')
    return args


def identity(item):
    return item


def wandb_resume_options(root, resume, project):
    """Resume the recorded W&B run, including when the output directory changes."""
    if resume is None:
        return {}
    paths = [Path(root) / 'wandb.json', Path(resume).resolve().parents[2] / 'wandb.json']
    state = Path(resume) / 'train_state.json'
    if state.is_file():
        recorded_run = json.loads(state.read_text()).get('run_dir')
        if isinstance(recorded_run, str) and Path(recorded_run).is_absolute():
            paths.append(Path(recorded_run) / 'wandb.json')
    records = [json.loads(path.read_text()) for path in dict.fromkeys(paths) if path.is_file()]
    if not records:
        raise ValueError('Resume requires existing wandb.json; use --no-wandb explicitly to disable logging')
    if any(not isinstance(record.get('id'), str) or not record['id'] or record.get('project') != project for record in records):
        raise ValueError('W&B resume metadata is invalid or its project differs from --wandb-project')
    if len({record['id'] for record in records}) != 1:
        raise ValueError('W&B resume source and output directories refer to different runs')
    return {'id': records[0]['id'], 'resume': 'must'}


def checkpoint_event(log, event, **values):
    # A failing disk logger must not strand peers before the save collective.
    try:
        log(event, **values)
    except Exception as exc:
        try:
            print(json.dumps({'event': event, 'checkpoint_log_error': str(exc), **values}, default=str), flush=True)
        except Exception:
            pass


def checkpoint_with_retries(save_fn, error_type, positional_args, *, required, log,
                            attempts=3, retry_seconds=5):
    """Periodic IO failures return None; required saves retry then fail visibly.

    Only globally agreed local failures are recoverable. Transport failures and
    arbitrary exceptions propagate immediately, since peers may not agree.
    """
    total = attempts if required else 1
    for attempt in range(total):
        checkpoint_event(log, 'save_start', step=positional_args[3], attempt=attempt, required=required)
        try:
            saved = save_fn(*positional_args, attempt=attempt)
        except error_type as exc:
            checkpoint_event(log, 'save_failed', step=positional_args[3], attempt=attempt,
                             required=required, error=str(exc), training_can_continue=not required)
            if not required:
                return None
            if attempt + 1 == total:
                raise
            time.sleep(retry_seconds)
        else:
            checkpoint_event(log, 'save_complete', step=positional_args[3], checkpoint=saved)
            return saved


def batch_summary(bundle):
    """Describe actual consumed rows; preserve the one-row conditioning shape."""
    rows=bundle if isinstance(bundle,list) else [bundle]
    students=[row['student'] for row in rows]
    per_row=[{'timeline_frames':len(row['ids'][0]),'text_positions':len(row['sel']),
              'dummy':bool(row.get('dummy',False)),
              'speech_depth_frames':0 if row.get('dummy',False) else len(row['light']),
              'pose_depth_frames':0 if row.get('dummy',False) else len(row['plight'])}
             for row in students]
    return {'row_count':len(rows),'timeline_frames':sum(row['timeline_frames'] for row in per_row),
            'real_row_count':sum(not row['dummy'] for row in per_row),
            'dummy_row_count':sum(row['dummy'] for row in per_row),
            'per_row_frames':[row['timeline_frames'] for row in per_row],
            'per_row_counts':per_row,
            'text_positions':sum(row['text_positions'] for row in per_row),
            'speech_depth_frames':sum(row['speech_depth_frames'] for row in per_row),
            'pose_depth_frames':sum(row['pose_depth_frames'] for row in per_row),
            'conditioning':[row['conditioning'] for row in students] if isinstance(bundle,list)
                           else students[0]['conditioning']}


def main():
    args=args_parser()
    if args.profile_steps:
        assert args.mode=='train' and 2<=args.profile_steps<=10
        assert args.resume is None,'Profiles start from production weights, never resume the stopped run'
        args.no_wandb=True
    host=socket.gethostname().split('.')[0]
    assert host in [f'landsend-{i}' for i in (0,1,2,3,4,5,12)],'Unexpected training host'
    import numpy as np
    import torch
    import torch.distributed as dist
    from torch.utils.data import DataLoader
    from finetune.duplex_distill.data import ProductionData
    import runtime as R
    from finetune.duplex_distill import metrics as DM
    from finetune.duplex_distill.profiling import PhaseProfiler,NullProfile
    from checkpoints import load_checkpoint,save_checkpoint,CheckpointSaveError
    world=int(os.environ['WORLD_SIZE']);rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK'])
    micro_batch_size=args.micro_batch_size
    if args.mode=='train':assert world==56,'Training and full profiles require all seven nodes / 56 GPUs'
    dev=torch.device('cuda',local);torch.cuda.set_device(dev);torch.set_num_threads(1)
    profiler=PhaseProfiler(dev) if args.profile_steps else NullProfile()
    R.profile=profiler
    R.TEXT_CHUNK_SIZE=args.text_chunk_size
    if args.text_direct_addmm:
        from finetune.duplex_distill.benchmark_text_head import load_variants
        R.factorized_text_kl_from_hidden=load_variants(HERE/'losses.py')['direct_addmm']
    profile_records=[]
    dist.init_process_group('nccl',device_id=dev,timeout=datetime.timedelta(minutes=45))
    root=canonical_run_dir(args.run_dir).resolve();root.mkdir(parents=True,exist_ok=True);(root/'control').mkdir(exist_ok=True)
    def log(event,**kw):
        doc={'event':event,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'rank':rank,'host':host,**kw}
        print(json.dumps(doc,default=str),flush=True)
        with (root/f'events-rank{rank}.jsonl').open('a') as f:f.write(json.dumps(doc,default=str)+'\n')
    def write(name,value):
        p=root/name;tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,default=str)+'\n');tmp.replace(p)
    hosts=[None]*world;dist.all_gather_object(hosts,host)
    if args.mode=='train':assert hosts==[f'landsend-{i}' for i in (0,1,2,3,4,5,12) for _ in range(8)]
    cfg=load_recipe(args.student/'train_cfg.json',world=world,micro_batch_size=micro_batch_size)
    cfg.update(checkpoint_retry_seconds=60,checkpoint_required_attempts=3,checkpoint_required_retry_delay_seconds=5)
    cfg.update(objective='forward_kl_teacher_to_student_only',temperature=1.0,kl_lexical_weight=1.0,
       student_checkpoint=str(args.student),audio_teacher_checkpoint=str(args.audio_teacher),pose_teacher_checkpoint=str(args.pose_teacher),
       global_batch_size=world*micro_batch_size,micro_batch_size=micro_batch_size,
       rows_per_update=world*micro_batch_size,gradient_accumulation=1,hostmap=hosts,decoder_activation_checkpointing='per_layer',
       metrics={'train':'teacher KL summaries','text_kl':'teacher text KL diagnostics',
                'audio_codebooks':'teacher KL','pose_codebooks':'teacher KL',
                'data_loss':'detached dataset cross entropy summaries','data_text_ce':'detached dataset text cross entropy diagnostics',
                'data_audio_codebooks':'detached dataset cross entropy','data_pose_codebooks':'detached dataset cross entropy'},
       teacher_conditioning='same B prefix/voice/drop decision; native-empty A for monoplex; same raw A audio for student/audio teacher',
       mode=args.mode,smoke_depth=args.smoke_depth if args.mode=='smoke' else None,
       text_head={'chunk_size':R.TEXT_CHUNK_SIZE,'direct_addmm':args.text_direct_addmm,
                  'implementation':'benchmark_text_head.direct_addmm' if args.text_direct_addmm else 'losses.factorized_text_kl_from_hidden'},
       profiling_text={'enabled':bool(args.profile_steps),'chunk_size':R.TEXT_CHUNK_SIZE,
                       'direct_addmm':args.text_direct_addmm,'scope':'same resolved implementation as training'},
       profiling_batch={'enabled':bool(args.profile_steps),'micro_batch_size':micro_batch_size,
                        'global_rows_per_update':world*micro_batch_size,'schedule_steps':cfg['steps'],
                        'schedule_warmup':cfg['warmup'],
                        'schedule_policy':'same resolved two-epoch global-batch schedule as training'},
       profile_steps=args.profile_steps,purpose='bounded_scratch_profile' if args.profile_steps else 'training')
    if args.resume:
        saved_cfg=json.loads((args.resume/'train_state.json').read_text())['cfg']
        for key in ('micro_batch_size','global_batch_size','steps','warmup'):
            if saved_cfg.get(key)!=cfg[key]:
                raise ValueError(f'Resume recipe mismatch for {key}: saved={saved_cfg.get(key)}, requested={cfg[key]}')
    if local==0:write('config.json',cfg)
    torch.manual_seed(cfg['dec_seed']);np.random.seed(cfg['dec_seed'])
    data=ProductionData(args.data_root,args.student/'train_cfg.json',world=world,micro_batch_size=micro_batch_size)
    plan=data.make_plan(world=world,micro_batch_size=micro_batch_size)
    assert plan['steps']==cfg['steps'] and plan['micro_batch_size']==micro_batch_size
    plan_hashes=[None]*world;dist.all_gather_object(plan_hashes,plan['digest']);assert len(set(plan_hashes))==1
    if local==0:write('plan.json',plan)
    R.install_forwards();M=R.M
    M.QWEN32='/home/param/models/imported-weights/ppm-811-17850-base-bf16/configs/Qwen3-32B'
    M.QWEN4B_CFG='/home/param/models/imported-weights/ppm-811-17850-base-bf16/configs/Qwen3-4B'
    depth=args.smoke_depth if args.mode=='smoke' else None
    c32,c4,c06=M.cfg32(depth),M.cfg4b(depth),M.cfg06(depth)
    assert all(float(getattr(c,'attention_dropout',0))==0 for c in (c32,c4,c06))
    vadapter=cfg['vadapter']
    models={};initialization={}
    for role,path in [('student',args.student),('audio_teacher',args.audio_teacher),('pose_teacher',args.pose_teacher)]:
        log('build_start',role=role)
        model,info=M.build(c32,c4,c06,dev,init=False,dec_mode='replicate',
            decp_bb=role!='audio_teacher',vadapter=vadapter if role!='pose_teacher' else None,nopose=role=='audio_teacher',
            log=lambda msg:log('model_build',role=role,message=msg))
        initialization[role]=load_checkpoint(model,path,dev,smoke=args.mode=='smoke')
        if role!='student':model.requires_grad_(False);model.eval()
        else:model.train()
        if hasattr(model,'decp_bb'):model.decp_bb_alpha=cfg['decp_bb_alpha']
        models[role]=model
        log('model_loaded',role=role,allocated_GB=torch.cuda.memory_allocated()/1e9)
    if local==0:write('initialization.json',initialization)
    student,at,pt=(models[k] for k in ('student','audio_teacher','pose_teacher'))
    import audio_s4q as AU
    front=AU.make_front({'s4_audio':'flac','s4_vox_model':'/home/param/models/imported-weights/aloe-ft-vox17127-nopersona/assets/voxtral-realtime','s4_enc_hidden':1280},dev)
    rot=M.Rot(c32,dev)
    groups={g:[] for g in cfg['lr']}
    for name,param in student.named_parameters():
        group=M.group_of(name)
        if group not in ('dec_s','dec_p'):groups[group].append(param)
    opt=torch.optim.AdamW([{'params':ps,'name':g,'lr':cfg['lr'][g]} for g,ps in groups.items() if ps],
                         betas=(.9,cfg['adam_beta2']),eps=1e-8,weight_decay=0.,fused=True)
    decopts={'dec_s':R.DecoderOptimizer(student.dec_s,cfg['dec_lr'],cfg['dec_clip'],cfg['adam_beta2']),
             'dec_p':R.DecoderOptimizer(student.dec_p,cfg['dec_p']['lr'],cfg['dec_p']['clip'],cfg['adam_beta2'])}
    for name,value in decopts.items():
        value.name=name
        getattr(student,name)._profile_name=name
    optimizers={'main':opt,**{k:v.opt for k,v in decopts.items()}}
    start=0
    if args.resume:
        from checkpoints import resume_checkpoint
        resumed=resume_checkpoint(student,optimizers,args.resume,dev)
        start=int(resumed['step'])
        log('resume_complete',step=start,path=str(args.resume),world=world)
    stop={'requested':False}
    def request_stop(*_):stop['requested']=True
    signal.signal(signal.SIGUSR1,request_stop)
    wandb_run=None
    wandb_error=None
    if rank==0 and not args.no_wandb:
        try:
            import wandb
            resume_options=wandb_resume_options(root,args.resume,args.wandb_project)
            wandb_run=wandb.init(project=args.wandb_project,name=root.name,dir=str(root),config=cfg,
                                settings=wandb.Settings(init_timeout=120),**resume_options)
            write('wandb.json',{'url':wandb_run.url,'id':wandb_run.id,'project':args.wandb_project})
        except Exception as exc:
            wandb_error=f'{type(exc).__name__}: {exc}'
    wandb_errors=[None]*world;dist.all_gather_object(wandb_errors,wandb_error)
    if any(wandb_errors):raise RuntimeError('W&B initialization failed: '+str(wandb_errors[0]))
    def lr_scale(step):
        # Original scheduler begins at step0 and reaches zero at the end.
        if step<cfg['warmup']:return step/max(cfg['warmup'],1)
        return max(0.,(cfg['steps']-step)/max(cfg['steps']-cfg['warmup'],1))
    def run_batch(bundle,step,grad=True):
        bundles=bundle if isinstance(bundle,list) else [bundle]
        if len(bundles)>1:
            assert len(bundles)==micro_batch_size==2
        latents=[]
        with profiler.span('audio_encoding'),torch.no_grad():
            # Encoder context must never cross independent conversations.
            for item in bundles:
                raw=item['student'];ref=json.loads(raw['aref'])
                latents.append(front.encode([torch.from_numpy(np.ascontiguousarray(raw['wav']))],
                    [{'row_key':ref['row_key'],'n':int(ref['n'])}]))
                assert torch.isfinite(latents[-1]).all()
        with profiler.span('prepare_inputs'):
            b,counts=(R.prepare(bundles[0],dev,latents[0],world) if len(bundles)==1
                      else R.prepare_many(bundles,dev,latents,world))
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            with profiler.span('audio_teacher_forward'):
                audio=at(b['audio_teacher'],rot,role='audio_teacher')
            with profiler.span('pose_teacher_forward'):
                pose=pt(b['pose_teacher'],rot,role='pose_teacher')
        with profiler.span('student_forward'),torch.set_grad_enabled(grad),torch.autocast('cuda',dtype=torch.bfloat16):
            out=student(b['student'],rot,role='student',teachers={'audio':audio,'pose':pose},lexical_weight=cfg['kl_lexical_weight'])
        audio.pop('text');audio.pop('logits_s');pose.pop('logits_p')
        loss=(cfg['modality_weights']['text']*out['text']['loss']
              +cfg['modality_weights']['audio']*cfg['w_cb0']*out['loss_s']
              +cfg['modality_weights']['pose']*cfg['w_pcb0']*out['loss_p'])
        roots=[loss];grads=[None];media=[]
        for name,teacher,tout,key,code,lightkey,count_i,wcb in [
            ('dec_s',at,audio,'s',b['student']['depth'][1],'light',3,cfg['w_cb']),
            ('dec_p',pt,pose,'p',b['student']['pdepth'],'plight',4,cfg['w_pcb'])]:
            with profiler.span(name+'_depth'),torch.set_grad_enabled(grad):
                dh,dz,metric=R.decoder_kl(getattr(student,name),getattr(teacher,name),out['hv_'+key],tout['hv_'+key],code,b['student'][lightkey],
                    zb=out.get('zb_'+key),teacher_zb=tout.get('zb_'+key),global_frames=counts[count_i],world=world,
                    dummy=b['student']['dummy'],chunk_cap=cfg['dec_mb_max'],step=step,rank=rank,seed=cfg['dec_seed'],grad=grad,
                    objective_scale=wcb*cfg['modality_weights']['audio' if key=='s' else 'pose'])
            media.append(metric)
            roots.append(out['hv_'+key]);grads.append(dh)
            if dz is not None:roots.append(out['zb_'+key]);grads.append(dz)
        if grad:
            # Replicated decoder updates happen once, as in production, before
            # accumulated decoder input gradients enter the FSDP backbone graph.
            with profiler.span('decoder_optimizer'):
                decnorm={k:v.step(world,lr_scale(step)) for k,v in decopts.items()}
            with profiler.span('student_backward'):
                torch.autograd.backward(roots,grads)
        else:decnorm={}
        with profiler.span('metrics'):
            raw_metrics=DM.collect(out,b['student'],media,world,dev)
            metrics=DM.summarize(raw_metrics,cfg)
        del audio,pose,out,b,latents,loss,roots,grads
        return metrics,decnorm,raw_metrics
    # Full-width reduced-depth smoke uses real 7500-frame conditioned examples.
    smoke_items=data.smoke_items(plan,count=world*args.smoke_steps*micro_batch_size) if args.mode=='smoke' else None
    train_ds=data.rank_dataset(plan,rank,world,start=start)
    loader=DataLoader(train_ds,batch_size=2 if micro_batch_size==2 else None,num_workers=cfg['dl_workers'] if args.mode=='train' else 0,
                      pin_memory=False,persistent_workers=args.mode=='train',collate_fn=identity)
    iterator=iter(loader)
    limit=args.profile_steps or (args.smoke_steps if args.mode=='smoke' else cfg['steps'])
    log('ready',world=world,global_batch=cfg['global_batch_size'],micro_batch=micro_batch_size,
        steps=limit,full_depth=args.mode=='train',allocated_GB=torch.cuda.memory_allocated()/1e9)
    next_checkpoint_at=time.monotonic()+cfg['checkpoint_interval_seconds']
    for step in range(start,limit):
        t0=time.monotonic();torch.cuda.reset_peak_memory_stats()
        if args.profile_steps:profiler.begin(step+1)
        with profiler.span('data_wait'):
            if smoke_items:
                smoke_rows=[data.batch(smoke_items[step*world*micro_batch_size+slot*world+rank],rank,load_audio=True)
                            for slot in range(micro_batch_size)]
                bundle=smoke_rows if micro_batch_size==2 else smoke_rows[0]
            else:bundle=next(iterator)
        batch_info=batch_summary(bundle)
        if micro_batch_size==2:
            assert batch_info['row_count']==2
            log('batch_loaded',step=step+1,global_batch=cfg['global_batch_size'],**batch_info)
        with profiler.span('zero_grad'):
            opt.zero_grad(set_to_none=True)
            for value in decopts.values():value.zero()
            scale=lr_scale(step)
            for group in opt.param_groups:group['lr']=cfg['lr'][group['name']]*scale
        metrics,decnorm,_=run_batch(bundle,step)
        norms={}
        with profiler.span('main_gradient_clip'):
            for group in opt.param_groups:
                norm=torch.nn.utils.clip_grad_norm_(group['params'],cfg['clip'][group['name']],error_if_nonfinite=True)
                norms[group['name']]=float(norm)
        with profiler.span('gradient_checks'):
            assert all(p.grad is None for teacher in (at,pt) for p in teacher.parameters()),'Teacher received gradient'
            assert any(v>0 for v in norms.values()),'All student gradients are zero'
        with profiler.span('main_optimizer'):
            opt.step()
        assert all(math.isfinite(v) for v in metrics.values()),'Nonfinite metric'
        if min(metrics['train/text_loss'],metrics['train/audio_loss'],metrics['train/pose_loss']) < -1e-3:raise RuntimeError('Negative KL')
        metrics.update({f'grad_norm/{k}':v for k,v in {**norms,**decnorm}.items()})
        metrics.update({f'lr/{g["name"]}':g['lr'] for g in opt.param_groups})
        metrics.update({'step':step+1,'time/step_seconds':time.monotonic()-t0,'memory/max_allocated_GB':torch.cuda.max_memory_allocated()/1e9})
        if rank==0:
            with (root/'metrics.jsonl').open('a') as f:f.write(json.dumps(metrics)+'\n')
            if wandb_run:wandb_run.log(metrics,step=step+1)
        log('step_complete',step=step+1,seconds=metrics['time/step_seconds'],kl=metrics['train/loss'],data_ce=metrics['data_loss/loss'],
            max_memory_GB=metrics['memory/max_allocated_GB'],teacher_gradients=False,
            global_batch=cfg['global_batch_size'],**batch_info)
        write(f'progress-rank{rank}.json',{'step':step+1,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':os.getpid(),
             'host':host,'rank':rank,'world':world,'global_batch':cfg['global_batch_size'],'micro_batch':micro_batch_size,
             'kl':metrics['train/loss'],'data_loss':metrics['data_loss/loss'],'state':'running'})
        control=torch.tensor([int(stop['requested'] or (root/'control/STOP_AFTER_SAVE').exists()),
            int(not args.profile_steps and rank==0 and time.monotonic()>=next_checkpoint_at)],device=dev)
        dist.all_reduce(control,op=dist.ReduceOp.MAX)
        requested=bool(control[0])
        if args.profile_steps:
            record=profiler.finish(rank=rank,host=host,max_memory_GB=torch.cuda.max_memory_allocated()/1e9,
                global_batch=cfg['global_batch_size'],**batch_info)
            profile_records.append(record)
            with (root/f'profile-rank{rank}.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            if requested:break
            continue
        save=bool(control[1]) or (step+1==limit and cfg['save_final']) or requested
        if save and args.mode=='train':
            saved=checkpoint_with_retries(save_checkpoint,CheckpointSaveError,
                (student,optimizers,root,step+1,cfg),required=requested or step+1==limit,log=log,
                attempts=cfg['checkpoint_required_attempts'],retry_seconds=cfg['checkpoint_required_retry_delay_seconds'])
            next_checkpoint_at=time.monotonic()+(cfg['checkpoint_interval_seconds'] if saved else cfg['checkpoint_retry_seconds'])
        if requested:break
    dist.barrier()
    if args.profile_steps:
        gathered=[None]*world if rank==0 else None
        dist.gather_object(profile_records,gathered,dst=0)
        if rank==0:write('profile-all-ranks.json',{'records':gathered,'world':world,'steps':limit,
            'micro_batch_size':micro_batch_size,'global_batch_size':cfg['global_batch_size'],
            'timing_method':'CUDA events without per-stage synchronization; CPU dispatch times overlap GPU; nested stages overlap parents',
            'production_training_resumed':False,'checkpoint_written':False})
    if local==0:write('PROFILE_COMPLETE.json' if args.profile_steps else 'SMOKE_COMPLETE.json' if args.mode=='smoke' else 'TRAINING_STOPPED.json',{'step':step+1,'mode':'profile' if args.profile_steps else args.mode,'world':world,'success':True})
    if wandb_run:wandb_run.finish()
    log('complete',step=step+1,mode=args.mode)
    dist.destroy_process_group()


if __name__=='__main__':main()
