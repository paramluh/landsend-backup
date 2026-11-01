import ast
import gc
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys

os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
os.environ.setdefault('PPM_CD','2')
os.environ.setdefault('FA3_DIR','/home/param/pose-repo/.cache/kernels/vllm-flash-attn3')
sys.path.insert(0,'/home/param/pose-repo/eval/ar_diagnostics')
sys.path.insert(0,'/home/param/pose-repo/infer/engine/ppm')
import torch
import torch.nn.functional as F
import perturbation_gain as G
import perturbation_logits as L
import model_ppm as M

torch.set_num_threads(2)
torch.set_grad_enabled(False)
device='cuda:7'
torch.cuda.set_device(device)
root=Path('/home/param/outputs/pose-ar-logits-20261030')
model_id=os.environ.get('CHECK_MODEL','duplex211')
checkpoint=next(m['checkpoint'] for m in json.loads((root/'models.json').read_text())['models'] if m['model_id']==model_id)
reader=G.CheckpointReader(checkpoint)

def native_method(path,name,globals_):
    tree=ast.parse(Path(path).read_text())
    nodes=[n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name]
    assert len(nodes)==1, (path,name,len(nodes))
    node=nodes[0]
    scope=dict(globals_)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),scope)
    return scope[name]

class ReferenceProjection:
    def __init__(self):
        self.w={name:reader.tensor('head.proj.'+name,device) for name in ('D','G','V','U')}
    def __call__(self,x):
        linear=F.linear(x,self.w['D'])
        gate=F.silu(F.linear(x,self.w['G']))
        value=F.linear(x,self.w['V'])
        nonlinear=F.linear(gate*value,self.w['U'])
        return (x+linear)+nonlinear

reports=[]
for submodel in sys.argv[1:] or ['backbone','pose_tower','pose_depth_decoder']:
    cache_path=root/'state-cache'/model_id/submodel/'c0-side0.pt'
    if not cache_path.exists():
        continue
    cache=torch.load(cache_path,map_location='cpu',weights_only=True)
    head=L.NativeHead(reader,M,submodel,device)
    if submodel=='backbone':
        source=Path('/home/param/websites/pose-duplex-engine/live_ppm_pose.py') if model_id!='mono355' else Path('/home/param/websites/pose-monoplex-engine/ppm/fast_ppm.py')
        reference_function=native_method(source,'text_logits',{})
        proj=ReferenceProjection()
        native_weight=reader.tensor('head.lm_head.weight',device)
        replacements=torch.cat([reader.tensor('head.'+name,device) for name in ('spec_out','nw_out','eob_out')])
        native=SimpleNamespace(m=SimpleNamespace(head=SimpleNamespace(proj=proj,lm_head=SimpleNamespace(weight=native_weight))),
                               spec=torch.tensor([151670,151671,151645],device=device),R_spec=replacements,R=replacements)
        configurations=[('local',0,None),('last_position',0,None)]
    elif submodel=='pose_tower':
        source=Path('/home/param/websites/pose-duplex-engine/live_ppm_pose.py') if model_id!='mono355' else Path('/home/param/websites/pose-monoplex-engine/ppm/fast_ppm.py')
        native_weight=reader.tensor('tower_p.cb0_head.weight',device)
        configurations=[('local',0,0),('last_position',0,0)]
    else:
        source=Path('/home/param/pose-repo/infer/engine/ppm/heads_s2q.py')
        reference_function=native_method(source,'logits',{'torch':torch,'DEPTH':32,'CBS':2048})
        native_weight=reader.tensor('dec_p.heads.weight',device)
        native=SimpleNamespace(heads=SimpleNamespace(weight=native_weight))
        configurations=[('local',0,1),('local',15,16),('local',30,31),('last_position',0,31)]
    with torch.inference_mode():
        for readout,input_cb,target_cb in configurations:
            i=next(i for i,v in enumerate(cache['variants']) if v['codebook']==input_cb and v['alpha']==1)
            small=torch.stack([cache[readout+'_clean'][i],cache[readout][i]])
            h=small.repeat(64,1).to(device)
            observed=head(h,target_cb)
            if submodel=='backbone':
                expected=reference_function(native,h.to(torch.bfloat16))[:,:151682]
            elif submodel=='pose_tower':
                expected=F.linear(h.to(torch.bfloat16),native_weight).float()
            else:
                full=torch.zeros((len(h),32,1024),device=device,dtype=torch.float32)
                full[:,target_cb]=h
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    expected=reference_function(native,full)[:,target_cb-1].float()
            max_abs=float((observed-expected).abs().max())
            relative_rms=float((observed-expected).square().mean().sqrt()/expected.square().mean().sqrt())
            assert max_abs==0, (submodel,readout,target_cb,max_abs,relative_rms)
            reports.append({'model_id':model_id,'submodel':submodel,'readout':readout,'input_codebook':input_cb,
                            'target_codebook':target_cb,'source_cache':str(cache_path),
                            'source_cache_sha256':hashlib.sha256(cache_path.read_bytes()).hexdigest(),
                            'native_source':str(source),'native_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
                            'reference_max_absolute_difference':max_abs,'reference_relative_rms_difference':relative_rms,
                            'logits_shape':list(observed.shape),'cached_state_dtype':str(h.dtype),'batch_size':len(h)})
            del observed,expected,h,small
            if submodel=='pose_depth_decoder': del full
    del head,native_weight,cache
    if submodel=='backbone': del native,proj,replacements
    elif submodel=='pose_depth_decoder': del native
    gc.collect();torch.cuda.empty_cache()
output=root/'native-cached-head-validation.json'
if output.exists():
    old=json.loads(output.read_text())
    checked={r['submodel'] for r in reports}
    reports=[r for r in old['checks'] if r['submodel'] not in checked]+reports
result={'passed':True,'scope':'Independent native output-head algebra on authentic cached normalized states; not full-model or vector-level forward parity.',
        'checkpoint':checkpoint,'device':device,'checks':reports}
output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
