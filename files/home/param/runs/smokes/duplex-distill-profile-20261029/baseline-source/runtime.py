"""Production forward paths with distribution outputs and detached decoder roots."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from losses import distribution_kl, factorized_text_kl_from_hidden
from profiling import NullProfile

profile = NullProfile()

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'production'))
import model_ppm as M


def head_forward(self,h,sel,*,role,teacher=None,y=None,weight=None,mask=None,lexical_weight=1.0):
    hn=M.FO.rms(h,self.norm.weight,float(self.norm.variance_epsilon))
    if role=='pose_teacher':return {'hn':hn}
    x=self.proj(hn.index_select(0,sel))
    rows=torch.cat([self.spec_out,self.nw_out,self.eob_out],0)
    if role=='audio_teacher':
        return {'hn':hn,'x':x.detach(),'W':self.lm_head.weight.detach().clone(),'R':rows.detach().clone()}
    loss,stats=factorized_text_kl_from_hidden(x,self.lm_head.weight,rows,teacher['x'],teacher['W'],teacher['R'],
        y,weight,M.SPEC,lexical_weight=lexical_weight,mask=mask)
    return {'hn':hn,'loss':loss,'stats':stats}


def tower_forward(self,ids,depth,hn,pid,ctx,rot):
    cnd=M.shift1(self.cond(hn),ctx,0.0)
    e=self.text_emb(ids)
    codes=self.code_emb(M.shift1(depth,ctx,-1))
    h=(e+codes.to(e.dtype)+cnd.to(e.dtype))[None]
    pe=rot(h,pid)
    for layer in self.layers:h=layer(h,pe,ctx)
    hv=M.FO.rms(h[0],self.norm.weight,float(self.norm.variance_epsilon))
    return hv,(hv@self.cb0_head.weight.to(hv.dtype).t()).float()


def ppm_forward(self,b,rot,*,role,teachers=None,lexical_weight=1.0):
    with profile.span(f'{role}/inputs'):
        e=self.inp(b['ids'],b.get('depth'),b.get('present'),b.get('pdepth'))[None]
        if b.get('vlat') is not None:
            v=self.vgain(self.vadapter(b['vlat'])[0])
            e=e+torch.zeros_like(e[0]).index_copy(0,b['vpos'],v.to(e.dtype))[None]
        ctx=M.Ctx(b['segs'],e.device);pe=rot(e,b['pid']);h=e
    with profile.span(f'{role}/trunk'):
        for layer in self.layers:h=layer(h,pe,ctx)
    with profile.span(f'{role}/text_head'):
        head=self.head(h[0],b['sel'],role=role,teacher=teachers['audio']['text'] if teachers else None,
            y=b['y'],weight=b.get('text_weight'),mask=b.get('text_mask'),lexical_weight=lexical_weight)
    hn=head.pop('hn');out={'text':head}
    if role!='pose_teacher':
        with profile.span(f'{role}/speech_tower'):
            out['hv_s'],out['logits_s']=self.tower_s(b['ids'][1],b['depth'][1],hn,b['pid'],ctx,rot)
    if role!='audio_teacher':
        with profile.span(f'{role}/pose_tower'):
            out['hv_p'],out['logits_p']=self.tower_p(b['ids'][1],b['pdepth'],hn,b['pid'],ctx,rot)
            out['zb_p']=M.shift1(self.decp_bb(hn),ctx,0.0)*float(self.decp_bb_alpha)
    if role=='student':
        with profile.span(f'{role}/cb0_kl'):
            out['loss_s'],out['stats_s']=distribution_kl(out.pop('logits_s'),teachers['audio']['logits_s'],
                b['cb0'],b['speech_weight'],mask=b['speech_mask'])
            out['loss_p'],out['stats_p']=distribution_kl(out.pop('logits_p'),teachers['pose']['logits_p'],
                b['pcb0'],b['pose_weight'],mask=b['pose_mask'])
    return out


def decoder_layer(layer,h,cos,sin):
    """One depth layer, op-for-op with production._dec_trunk."""
    B,T,_=h.shape;a=layer.self_attn
    y=layer.input_layernorm(h);shp=(B,T,-1,a.head_dim)
    q=a.q_norm(a.q_proj(y).view(shp));k=a.k_norm(a.k_proj(y).view(shp))
    v=a.v_proj(y).view(shp)
    q,k=M.Q3.apply_rotary_pos_emb(q,k,cos,sin,unsqueeze_dim=2)
    bf=torch.bfloat16 if torch.is_autocast_enabled() else q.dtype
    o=M.DA.attn32(q.to(bf),k.to(bf),v.to(bf),a.scaling)
    h=h+a.o_proj(o.reshape(B,T,-1))
    return h+layer.mlp(layer.post_attention_layernorm(h))


def decoder_trunk_checkpointed(model,x):
    # Checkpoint each layer separately: checkpointing the whole trunk still
    # materializes all 28 layers' activations together during recomputation.
    cos,sin=model.rotary_emb(x,torch.arange(x.shape[1],device=x.device)[None])
    h=x
    for layer in model.layers[:model.config.num_hidden_layers]:
        h=M.checkpoint(decoder_layer,layer,h,cos,sin,use_reentrant=False)
    return model.norm(h)


def decoder_forward(self,hv,depth,flat,zb=None):
    with torch.autocast('cuda',dtype=torch.bfloat16):
        x,labels=self.inputs(hv,depth,flat,zb=zb)
        # Recompute the depth trunk during backward to fit both frozen teachers
        # without changing frames, microbatch, conditioning, or forward math.
        hl=decoder_trunk_checkpointed(self.model,x) if torch.is_grad_enabled() else M._dec_trunk(self.model,x)
        logits=torch.einsum('ndh,dvh->ndv',hl[:,1:],self.heads.weight.view(31,2048,-1)).float()
    return logits,labels


def install_forwards():
    M.Head.forward=head_forward;M.Tower.forward=tower_forward
    M.PPM.forward=ppm_forward;M.Decoder.forward=decoder_forward


def prepare(bundle,dev,lat,world):
    """Global token normalization is identical to the original recipe's mean."""
    raw=bundle['student'];dummy=bool(raw.get('dummy',False))
    text_mask=np.asarray(raw['word'])+np.asarray(raw['pad'])+np.asarray(raw['nw'])+np.asarray(raw['eob'])>0
    counts=torch.tensor([int(text_mask.sum()),int((np.asarray(raw['cb0'])>=0).sum()),
       int((np.asarray(raw['pcb0'])>=0).sum()),len(raw['light']),len(raw['plight'])],device=dev,dtype=torch.float64)
    if dummy:counts.zero_()
    dist.all_reduce(counts)
    result={}
    for role,source in bundle.items():
        if role not in ('student','audio_teacher','pose_teacher'):continue
        b={k:torch.as_tensor(np.asarray(v),device=dev) for k,v in source.items() if isinstance(v,np.ndarray) and k!='wav'}
        b.update(kind=source['kind'],segs=[int(v) for v in source['segs']],dummy=dummy)
        b['pid']=b['pid'].reshape(1,-1)
        b['text_mask']=torch.as_tensor(text_mask,device=dev)&(not dummy)
        b['speech_mask']=(b['cb0']>=0)&(not dummy)
        if 'pcb0' in b:b['pose_mask']=(b['pcb0']>=0)&(not dummy)
        b['text_weight']=b['text_mask'].float()*(world/counts[0].clamp_min(1))
        b['speech_weight']=b['speech_mask'].float()*(world/counts[1].clamp_min(1))
        if 'pcb0' in b:b['pose_weight']=b['pose_mask'].float()*(world/counts[2].clamp_min(1))
        if role!='pose_teacher':
            prefix,n=(int(v) for v in source['vin'])
            assert tuple(lat.shape)==(1,4*n,1280),(lat.shape,n)
            b['vpos']=torch.arange(prefix,prefix+n,device=dev);b['vlat']=lat
        result[role]=b
    return result,counts


class DecoderOptimizer:
    """Replicated FP32 decoder and optimizer, exactly one update per micro/step."""
    def __init__(self,decoder,lr,clip,beta2):
        self.params=list(decoder.parameters());self.peak_lr=lr;self.clip=clip
        self.flat=torch.zeros(sum(p.numel() for p in self.params),device=self.params[0].device,dtype=torch.float32)
        offset=0
        for p in self.params:
            p.grad=self.flat[offset:offset+p.numel()].view_as(p);offset+=p.numel()
        self.opt=torch.optim.AdamW(self.params,lr=lr,betas=(.9,beta2),eps=1e-8,weight_decay=0.,fused=True)
    def zero(self):self.flat.zero_()
    def step(self,world,multiplier):
        prefix=getattr(self,'name','decoder')
        with profile.span(f'{prefix}/gradient_allreduce'):
            dist.all_reduce(self.flat);self.flat.div_(world)
        with profile.span(f'{prefix}/clip'):
            norm=torch.nn.utils.clip_grad_norm_(self.params,self.clip,error_if_nonfinite=True)
        with profile.span(f'{prefix}/adamw'):
            for group in self.opt.param_groups:group['lr']=self.peak_lr*multiplier
            self.opt.step()
        return float(norm)


def decoder_kl(student,teacher,hv,teacher_hv,depth,light,*,zb,teacher_zb,global_frames,world,dummy,chunk_cap,step,rank,seed,grad=True,objective_scale=1.0):
    """Backward each depth chunk immediately; return accumulated tower/link grads."""
    hd=hv.detach().requires_grad_(grad)
    zd=zb.detach().requires_grad_(grad) if zb is not None else None
    n=light.numel();dev=hv.device
    prefix=getattr(student,'_profile_name','decoder')
    metrics={k:torch.zeros(31,device=dev,dtype=torch.float64) for k in ('kl','ce','count')}
    if n and not dummy:
        order=torch.randperm(n,generator=torch.Generator().manual_seed(seed*1000003+step*1009+rank)).to(dev)
        batches=(n+chunk_cap-1)//chunk_cap;size=max(16,(n+batches-1)//batches)
        for start in range(0,n,size):
            flat=light.index_select(0,order[start:start+size])
            with profile.span(f'{prefix}/teacher_forward'):
                with torch.no_grad():tl,labels=teacher(teacher_hv,depth,flat,zb=teacher_zb)
            with profile.span(f'{prefix}/student_forward'):
                sl,student_labels=student(hd,depth,flat,zb=zd)
            assert torch.equal(labels,student_labels)
            weight=torch.full((len(flat),),world/(31*max(float(global_frames),1)),device=dev)
            with profile.span(f'{prefix}/kl'):
                loss,stats=distribution_kl(sl,tl,labels,weight)
            if grad:
                with profile.span(f'{prefix}/backward'):
                    (loss*float(objective_scale)).backward()
            with profile.span(f'{prefix}/stats'):
                metrics['kl']+=stats['kl_by_codebook'].double()
                metrics['ce']+=stats['data_ce_by_codebook'].double()
                metrics['count']+=stats['count_by_codebook'].double()
            del sl,tl,loss,stats
    return (torch.zeros_like(hv) if hd.grad is None else hd.grad.detach(),
            None if zd is None else torch.zeros_like(zb) if zd.grad is None else zd.grad.detach(),metrics)
