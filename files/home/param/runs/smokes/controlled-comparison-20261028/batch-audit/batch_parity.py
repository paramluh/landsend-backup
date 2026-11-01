"""Targeted B1/B2 numerical and random-stream audit; no training or checkpoint writes."""
import gc, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,'/home/param/pose-repo/infer/transcript_pose')
import infer as CLI
R=Path('/home/param/pose-runs/controlled-comparison-20261028/batch-audit')
FP,IP=CLI.engine_modules('/home/param/models/ppm-811-17850-base-bf16/configs/Qwen3-32B')
devs=[torch.device('cuda:0'),torch.device('cuda:1')]
torch.cuda.set_device(devs[-1]);F=18
insts=[]
for i in range(2):
    with np.load(R/f'prepare-{i}'/'inputs.npz') as z:
        inst={k:z[k].copy() for k in ('ids','depth','present','pdepth','text_force')}
        inst.update(P=int(z['P']),a_mode='empty')
        insts.append(inst)
args=CLI.arguments(['--recipe','compact-all','--out',str(R/'unused-output')])
params=CLI.sampling(args)
seeds=[20261028,20261029]
report={'frames':F,'prefixes':[it['P'] for it in insts], 'seeds':seeds}

def rngcheck():
    V=2048
    base=torch.softmax(torch.sin(torch.arange(V,device=devs[-1]).float()),-1)[None]
    def run(ss):
        gs=tuple(torch.Generator(device=devs[-1]).manual_seed(s) for s in ss)
        probs=base.expand(len(ss),-1).contiguous()
        return torch.stack([FP.row_multinomial(probs,gs) for _ in range(160)]).cpu().numpy()
    one=[run([s])[:,0] for s in seeds]
    two=run(seeds);reverse=run(seeds[::-1])
    assert np.array_equal(one[0],two[:,0]) and np.array_equal(one[1],two[:,1])
    assert np.array_equal(two,reverse[:,::-1])
    # Same check under graph replay and after restarting generators.
    def graph(ss):
        gs=tuple(torch.Generator(device=devs[-1]).manual_seed(s) for s in ss)
        probs=base.expand(len(ss),-1).contiguous(); out=torch.empty(len(ss),device=devs[-1],dtype=torch.long)
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            FP.row_multinomial(probs,gs)
        torch.cuda.current_stream().wait_stream(stream)
        g=torch.cuda.CUDAGraph()
        for gen in gs:g.register_generator_state(gen)
        with torch.cuda.graph(g,stream=stream):out.copy_(FP.row_multinomial(probs,gs))
        for gen,s in zip(gs,ss):gen.manual_seed(s)
        result=[]
        for _ in range(160):g.replay();result.append(out.cpu().numpy().copy())
        return np.stack(result)
    g1=[graph([s])[:,0] for s in seeds];g2=graph(seeds);gr=graph(seeds[::-1])
    assert np.array_equal(g1[0],g2[:,0]) and np.array_equal(g1[1],g2[:,1]) and np.array_equal(g2,gr[:,::-1])
    return dict(draws=160, vocabulary=V,eager_batch_and_row_invariant=True,graph_batch_and_row_invariant=True)

def collect(e,its,ss,forced=None):
    e.start(its,ss,params,f0=0)
    zz=[]; tokens=[]
    for f in range(F+2):
        force=None
        if forced is not None and f>=2:
            force={'codes':np.stack([x['audio'][f-2] for x in forced]),'pose':np.stack([x['pose'][f-2] for x in forced])}
        e.step(f,graphs=True,cold_pad=False,force=force)
        # Verify both towers receive each actual lane token, including lexical frame0.
        expected=np.array([it['text_force'][it['P']+f] for it in its])
        assert np.array_equal(e.tok_now.cpu().numpy(),expected)
        assert np.array_equal(e.tokp.cpu().numpy(),expected)
        if f>=2:zz.append((e.z0.cpu().numpy().copy(),e.zp0[:len(its)].cpu().numpy().copy()))
    out=[]
    for i,it in enumerate(its):
        P=it['P'];r=e.readback(i,P+F+2)
        assert np.array_equal(r['lane_b'][P:P+F+2],it['text_force'][P:P+F+2])
        out.append(dict(audio=r['dep_b'][P+2:P+F+2],pose=r['pdep'][P+2:P+F+2],
                        za=np.stack([z[0][i] for z in zz]),zp=np.stack([z[1][i] for z in zz])))
    return out

def compare_logit(ref,got):
    d=got-ref
    return dict(max_abs=float(np.max(np.abs(d))),relative_rms=float(np.sqrt(np.mean(d*d))/max(1e-12,np.sqrt(np.mean(ref*ref)))),
                top1_equal_frames=int(np.sum(ref.argmax(-1)==got.argmax(-1))),frames=len(ref),finite=bool(np.isfinite(got).all()))

with torch.inference_mode():
    report['sampling_rng']=rngcheck();print('RNG B1/B2+row swap eager/graph PASS',flush=True)
    m,ldev,cfgs,ck=FP.build_model('/home/param/models/transcript-pose/compact-all/export-200',devs)
    def engine(ni):
        e=FP.FastPPM(m,ldev,cfgs,devs,n_inst=ni,cfg_rows=False,max_s=F/12.5,p_max=max(it['P'] for it in insts),mimi_stream=False,per_instance_rng=True,forced_text_only=True)
        e.capture(f=2,P=max(it['P'] for it in insts));return e
    e=engine(1)
    b1=[]
    for it,s in zip(insts,seeds):b1.extend(collect(e,[it],[s]));print('B1 done',s,flush=True)
    del e;gc.collect()
    for d in devs:
        with torch.cuda.device(d):torch.cuda.empty_cache()
    e=engine(2);b2=collect(e,insts,seeds);print('B2 free done',flush=True)
    shared=collect(e,insts,seeds,forced=b1);print('B2 shared history done',flush=True)
    swapped=collect(e,insts[::-1],seeds[::-1],forced=b1[::-1]);print('B2 row swap done',flush=True)
    report['clips']=[]
    for i in range(2):
        item=dict(row=i,first_text_token=int(insts[i]['text_force'][insts[i]['P']]),forced_lane_and_both_tower_tokens_exact=True)
        for stream,key in [('audio','za'),('pose','zp')]:
            item[stream]=dict(free_running_codes_equal=int(np.sum(b1[i][stream]==b2[i][stream])),total_codes=int(b1[i][stream].size),
                              free_running_cb0_equal_frames=int(np.sum(b1[i][stream][:,0]==b2[i][stream][:,0])),
                              teacherforced_logits=compare_logit(b1[i][key],shared[i][key]),
                              row_swap_teacherforced_logits=compare_logit(shared[i][key],swapped[1-i][key]))
        report['clips'].append(item)
    report['checkpoint']=ck
    report['peak_gib']={str(d):torch.cuda.max_memory_allocated(d)/2**30 for d in devs}
    (R/'batch-parity.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
    assert all(c[s]['teacherforced_logits']['finite'] and c[s]['teacherforced_logits']['relative_rms']<0.02 for c in report['clips'] for s in ('audio','pose'))
