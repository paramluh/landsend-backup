"""Read-only GPU parity probe: full causal prefill versus cached steps on the same forced tokens."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch

p=argparse.ArgumentParser()
p.add_argument('--repo',default='/home/param/pose-repo')
p.add_argument('--ckpt',required=True)
p.add_argument('--input',nargs='+',required=True)
p.add_argument('--tokenizer',default='/home/param/models/ppm-811-17850-base-bf16/configs/Qwen3-32B')
p.add_argument('--devs',default='3,4')
p.add_argument('--out',required=True)
a=p.parse_args()
sys.path.insert(0,str(Path(a.repo)/'infer/transcript_pose'))
import infer as cli
FP,IP=cli.engine_modules(a.tokenizer)
devs=[torch.device(f'cuda:{x}') for x in a.devs.split(',')]
torch.cuda.set_device(devs[-1])
inputs=[]
for path in a.input:
    with np.load(path,allow_pickle=False) as z:
        inst={k:z[k].copy() for k in ('ids','depth','present','pdepth')}
        P=int(z['P'])
    inst.update(P=P,a_mode='empty')
    inputs.append((path,inst))
F=6
reports=[]
with torch.inference_mode():
    m,ldev,cfgs,info=FP.build_model(a.ckpt,devs)
    e=FP.FastPPM(m,ldev,cfgs,devs,n_inst=1,cfg_rows=False,max_s=1,
                 p_max=max(inst['P'] for _,inst in inputs),mimi_stream=False)
    e.capture(f=2,P=max(inst['P'] for _,inst in inputs))
    for path,inst in inputs:
        P=inst['P']
        assert inst['ids'].shape[1]>=P+F
        captured={}
        originals={}
        for name,stack in [('audio',e.ts),('pose',e.tp)]:
            original=stack.run
            originals[name]=original
            def record(*args,_name=name,_original=original,**kwargs):
                out=_original(*args,**kwargs)
                captured[_name]=out.detach().clone()
                return out
            stack.run=record
        e.start([inst],0,{'pose_cfg_on':0},f0=F)
        refs={'text':e.text_logits(e.prefill_hn[0][P-1:P+F-1]).float().cpu()}
        for name,tower in [('audio',m.tower_s),('pose',m.tower_p)]:
            hh=captured[name][0,P:P+F]
            with torch.cuda.device(hh.device):
                norm=FP.FO.rms(hh,tower.norm.weight,float(tower.norm.variance_epsilon))
                refs[name]=(norm@tower.cb0_head.weight.t()).float().cpu()
        e.ts.run=originals['audio'];e.tp.run=originals['pose']
        e.start([inst],0,{'pose_cfg_on':0},f0=0)
        rows=[]
        for f in range(F):
            T=P+f
            force={'lane':inst['ids'][1,T:T+1],'codes':inst['depth'][1,T:T+1],'pose':inst['pdepth'][T:T+1]}
            e.step(f,graphs=True,cold_pad=False,force=force)
            for name,value in [('text',e.zt[0]),('audio',e.z0[0]),('pose',e.zp0[0])]:
                if name != 'text' and f < 2:
                    continue
                got=value.float().cpu();ref=refs[name][f];diff=got-ref
                rows.append(dict(frame=f,stream=name,max_abs=float(diff.abs().max()),rms=float(diff.square().mean().sqrt()),
                                 relative_rms=float(diff.square().mean().sqrt()/ref.square().mean().sqrt().clamp_min(1e-12)),
                                 top1_equal=bool(got.argmax()==ref.argmax()),finite=bool(torch.isfinite(got).all())))
        report=dict(checkpoint=info,input=path,prefix_positions=P,frames=F,comparisons=rows,
                    method='Same teacher-forced tokens: eager full causal prefill versus CUDA-graph cached sequential steps; no optimization')
        reports.append(report)
    Path(a.out).write_text(json.dumps(reports,indent=2)+'\n')
    print(json.dumps(reports,indent=2),flush=True)
    assert all(r['finite'] and r['relative_rms'] < 0.02 for rep in reports for r in rep['comparisons']), 'Nonfinite or large (>2% RMS) cached/eager logit mismatch'
