#!/usr/bin/env python3
"""Forward-KL-only distillation of production pose duplex on its Aloe dataset."""
from __future__ import annotations
import argparse,datetime,gc,json,math,os,signal,socket,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
from finetune.duplex_distill.recipe import load_recipe


def args_parser():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--mode',choices=('train','smoke'),default='train')
    p.add_argument('--data-root',type=Path,default=Path('/home/param/datasets/aloe-duplex'))
    p.add_argument('--student',type=Path,default=Path('/home/param/models/ppm-aloe-poseduplex-v5-final/export-498-dcp'))
    p.add_argument('--audio-teacher',type=Path,default=Path('/home/param/models/ppm-aloe-samsep-vox17127-final/export-249-dcp'))
    p.add_argument('--pose-teacher',type=Path,default=Path('/home/param/models/transcript-pose/xabi-original/export-200'))
    p.add_argument('--smoke-depth',type=int,default=2)
    p.add_argument('--smoke-steps',type=int,default=2)
    p.add_argument('--resume',type=Path)
    p.add_argument('--wandb-project',default='param-pose-llm')
    p.add_argument('--no-wandb',action='store_true')
    p.add_argument('--profile-steps',type=int,default=0,
                   help='Bounded scratch profile: full training math; no W&B, eval, or checkpoints; exits after 2-10 steps')
    return p.parse_args()


def identity(item):
    return item


def main():
    args=args_parser()
    if args.profile_steps:
        assert args.mode=='train' and 2<=args.profile_steps<=10
        assert args.resume is None,'Profiles start from production weights, never resume the stopped run'
        args.no_wandb=True
    host=socket.gethostname().split('.')[0]
    assert host in [f'landsend-{i}' for i in (1,2,3,4,5,12)],'No training/GPU work on landsend-0'
    import numpy as np
    import torch
    import torch.distributed as dist
    from torch.utils.data import DataLoader
    from finetune.duplex_distill.data import ProductionData
    import runtime as R
    from finetune.duplex_distill import metrics as DM
    from finetune.duplex_distill.profiling import PhaseProfiler,NullProfile
    from checkpoints import load_checkpoint,save_checkpoint
    world=int(os.environ['WORLD_SIZE']);rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK'])
    if args.mode=='train':assert world==48
    dev=torch.device('cuda',local);torch.cuda.set_device(dev);torch.set_num_threads(1)
    profiler=PhaseProfiler(dev) if args.profile_steps else NullProfile()
    R.profile=profiler
    profile_records=[]
    dist.init_process_group('nccl',device_id=dev,timeout=datetime.timedelta(minutes=45))
    root=args.run_dir.resolve();root.mkdir(parents=True,exist_ok=True);(root/'control').mkdir(exist_ok=True)
    def log(event,**kw):
        doc={'event':event,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'rank':rank,'host':host,**kw}
        print(json.dumps(doc,default=str),flush=True)
        with (root/f'events-rank{rank}.jsonl').open('a') as f:f.write(json.dumps(doc,default=str)+'\n')
    def write(name,value):
        p=root/name;tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,default=str)+'\n');tmp.replace(p)
    hosts=[None]*world;dist.all_gather_object(hosts,host)
    if args.mode=='train':assert hosts==[f'landsend-{i}' for i in (1,2,3,4,5,12) for _ in range(8)]
    cfg=load_recipe(args.student/'train_cfg.json',world=world)
    cfg.update(objective='forward_kl_teacher_to_student_only',temperature=1.0,kl_lexical_weight=1.0,
       student_checkpoint=str(args.student),audio_teacher_checkpoint=str(args.audio_teacher),pose_teacher_checkpoint=str(args.pose_teacher),
       global_batch_size=world,micro_batch_size=1,gradient_accumulation=1,hostmap=hosts,decoder_activation_checkpointing='per_layer',
       metrics={'train':'teacher KL','audio_codebooks':'teacher KL','pose_codebooks':'teacher KL',
                'data_loss':'detached dataset cross entropy','data_audio_codebooks':'detached dataset cross entropy','data_pose_codebooks':'detached dataset cross entropy'},
       teacher_conditioning='same B prefix/voice/drop decision; native-empty A for monoplex; same raw A audio for student/audio teacher',
       mode=args.mode,smoke_depth=args.smoke_depth if args.mode=='smoke' else None,
       profile_steps=args.profile_steps,purpose='bounded_scratch_profile' if args.profile_steps else 'training')
    if local==0:write('config.json',cfg)
    torch.manual_seed(cfg['dec_seed']);np.random.seed(cfg['dec_seed'])
    data=ProductionData(args.data_root,args.student/'train_cfg.json');plan=data.make_plan(world=world)
    ds=data.rank_dataset(plan,rank,world)
    plan_hashes=[None]*world;dist.all_gather_object(plan_hashes,plan['digest']);assert len(set(plan_hashes))==1
    if local==0:write('plan.json',plan)
    R.install_forwards();M=R.M
    M.QWEN32='/home/param/models/ppm-811-17850-base-bf16/configs/Qwen3-32B'
    M.QWEN4B_CFG='/home/param/models/ppm-811-17850-base-bf16/configs/Qwen3-4B'
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
    front=AU.make_front({'s4_audio':'flac','s4_vox_model':'/home/param/models/aloe-ft-vox17127-nopersona/assets/voxtral-realtime','s4_enc_hidden':1280},dev)
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
    stop={'requested':False}
    def request_stop(*_):stop['requested']=True
    signal.signal(signal.SIGUSR1,request_stop)
    wandb_run=None
    if rank==0 and not args.no_wandb:
        import wandb
        wandb_run=wandb.init(project=args.wandb_project,name=root.name,dir=str(root),config=cfg,
                            settings=wandb.Settings(init_timeout=120))
        write('wandb.json',{'url':wandb_run.url,'id':wandb_run.id,'project':args.wandb_project})
    def lr_scale(step):
        # Original scheduler begins at step0 and reaches zero at the end.
        if step<cfg['warmup']:return step/max(cfg['warmup'],1)
        return max(0.,(cfg['steps']-step)/max(cfg['steps']-cfg['warmup'],1))
    def run_batch(bundle,step,grad=True):
        raw=bundle['student'];ref=json.loads(raw['aref'])
        with profiler.span('audio_encoding'),torch.no_grad():
            lat=front.encode([torch.from_numpy(np.ascontiguousarray(raw['wav']))],[{'row_key':ref['row_key'],'n':int(ref['n'])}])
            assert torch.isfinite(lat).all()
        with profiler.span('prepare_inputs'):
            b,counts=R.prepare(bundle,dev,lat,world)
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
        del audio,pose,out,b,lat,loss,roots,grads
        return metrics,decnorm,raw_metrics
    # Full-width reduced-depth smoke uses real 7500-frame conditioned examples.
    smoke_items=data.smoke_items(plan,count=world*args.smoke_steps) if args.mode=='smoke' else None
    train_ds=data.rank_dataset(plan,rank,world,start=start)
    loader=DataLoader(train_ds,batch_size=None,num_workers=cfg['dl_workers'] if args.mode=='train' else 0,
                      pin_memory=False,persistent_workers=args.mode=='train',collate_fn=identity)
    iterator=iter(loader)
    limit=args.profile_steps or (args.smoke_steps if args.mode=='smoke' else cfg['steps'])
    log('ready',world=world,global_batch=world,steps=limit,full_depth=args.mode=='train',allocated_GB=torch.cuda.memory_allocated()/1e9)
    for step in range(start,limit):
        t0=time.monotonic();torch.cuda.reset_peak_memory_stats()
        if args.profile_steps:profiler.begin(step+1)
        with profiler.span('data_wait'):
            bundle=data.batch(smoke_items[step*world+rank],rank,load_audio=True) if smoke_items else next(iterator)
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
            max_memory_GB=metrics['memory/max_allocated_GB'],teacher_gradients=False,conditioning=bundle['student']['conditioning'])
        write(f'progress-rank{rank}.json',{'step':step+1,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':os.getpid(),
             'host':host,'rank':rank,'world':world,'kl':metrics['train/loss'],'data_loss':metrics['data_loss/loss'],'state':'running'})
        requested=torch.tensor([int(stop['requested'] or (root/'control/STOP_AFTER_SAVE').exists())],device=dev);dist.all_reduce(requested,op=dist.ReduceOp.MAX)
        if args.profile_steps:
            record=profiler.finish(rank=rank,host=host,max_memory_GB=torch.cuda.max_memory_allocated()/1e9,
                text_positions=len(bundle['student']['sel']),speech_depth_frames=len(bundle['student']['light']),
                pose_depth_frames=len(bundle['student']['plight']),conditioning=bundle['student']['conditioning'])
            profile_records.append(record)
            with (root/f'profile-rank{rank}.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            if requested:break
            continue
        save=(step+1 in cfg['export_steps'] or step+1==limit or bool(requested))
        if save and args.mode=='train':
            log('save_start',step=step+1);saved=save_checkpoint(student,optimizers,root,step+1,cfg);log('save_complete',step=step+1,checkpoint=saved)
        if requested:break
        if args.mode=='train' and (step+1)%cfg['eval_every']==0:
            student.eval();evaluation=[]
            for batch_i in range(math.ceil(len(plan['eval'])/world)):
                ix=batch_i*world+rank;is_dummy=ix>=len(plan['eval'])
                ev=data.batch(plan['eval'][ix] if not is_dummy else plan['eval'][0],rank,load_audio=True,dummy=is_dummy,evaluation=True)
                for key in ('light','plight'):
                    full=ev['student'][key]
                    perm=torch.randperm(len(full),generator=torch.Generator().manual_seed(1234))[:cfg['eval_light_max']].numpy()
                    ev['student'][key]=full[perm]
                with torch.no_grad():_,_,eraw=run_batch(ev,step,grad=False)
                evaluation.append(eraw)
            if rank==0:
                means=DM.summarize(DM.sum_raw(evaluation),cfg)
                em={('eval/'+k[6:] if k.startswith('train/') else 'eval_'+k):v for k,v in means.items()}
                em['step']=step+1
                with (root/'eval_metrics.jsonl').open('a') as f:f.write(json.dumps(em)+'\n')
                if wandb_run:wandb_run.log(em,step=step+1)
            student.train();log('eval_complete',step=step+1,batches=len(evaluation))
    dist.barrier()
    if args.profile_steps:
        gathered=[None]*world if rank==0 else None
        dist.gather_object(profile_records,gathered,dst=0)
        if rank==0:write('profile-all-ranks.json',{'records':gathered,'world':world,'steps':limit,
            'timing_method':'CUDA events without per-stage synchronization; CPU dispatch times overlap GPU; nested stages overlap parents',
            'production_training_resumed':False,'checkpoint_written':False})
    if local==0:write('PROFILE_COMPLETE.json' if args.profile_steps else 'SMOKE_COMPLETE.json' if args.mode=='smoke' else 'TRAINING_STOPPED.json',{'step':step+1,'mode':'profile' if args.profile_steps else args.mode,'world':world,'success':True})
    if wandb_run:wandb_run.finish()
    log('complete',step=step+1,mode=args.mode)
    dist.destroy_process_group()


if __name__=='__main__':main()
