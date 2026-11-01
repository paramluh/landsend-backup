import hashlib
import json
import re
import sys
import time
from pathlib import Path

import torch

start=time.time()
torch.set_num_threads(4)
base=Path(sys.argv[1])
root=base/'weights'
meta=json.loads((root/'meta.json').read_text())
assert meta['step']==17850 and meta['world']==128 and meta['run_tag']=='ppm-v1-j27653'
assert meta['cfg']['layers']==64 and meta['cfg']['tower_layers']==36 and meta['cfg']['dec_layers']==28
assert meta['cfg']['proj_r']==2048 and meta['cfg']['tower_vocab']==151682
assert meta['cfg']['decp_bb']==1 and meta['cfg']['decp_bb_off']==1 and meta['cfg']['pose_head']==0
assert not meta['cfg'].get('s2p') and not meta['cfg'].get('g0cond')
shapes={}
reference=None
elements=0
for rank in range(128):
    state=torch.load(root/f'rank{rank}.pt',map_location='meta',mmap=True,weights_only=True)
    assert isinstance(state,dict) and all(isinstance(t,torch.Tensor) for t in state.values())
    if reference is None:
        reference={k:tuple(t.shape[1:]) for k,t in state.items()}
        shapes={k:[] for k in state}
    assert set(state)==set(reference)
    for key,tensor in state.items():
        assert tensor.dtype==torch.bfloat16 and tuple(tensor.shape[1:])==reference[key], (rank,key,tensor.dtype,tensor.shape)
        shapes[key].append(tensor.shape[0])
        elements+=tensor.numel()
rep=torch.load(root/'rep.pt',map_location='meta',mmap=True,weights_only=True)
assert all(isinstance(t,torch.Tensor) and t.dtype==torch.bfloat16 for t in rep.values())
assert not(set(rep)&set(reference))
elements+=sum(t.numel() for t in rep.values())
keys=set(reference)|set(rep)
assert len(reference)==1522 and len(rep)==628 and len(keys)==2150
assert not any(k.startswith('pose_head.') or k.startswith('tower_p.s_emb.') or k.startswith('inp.g0.') for k in keys)
global_shapes={key:(sum(parts),)+reference[key] for key,parts in shapes.items()}
for key,parts in shapes.items():
    rows=sum(parts);chunk=(rows+127)//128
    assert parts==[max(0,min(chunk,rows-rank*chunk)) for rank in range(128)],('source sharding',key)
    target_chunk=(rows+55)//56
    for rank in range(56):
        first=rank*target_chunk;last=min(rows,first+target_chunk)
        if last<=first:continue
        covered=sum(max(0,min(last,(sr+1)*chunk,rows)-max(first,sr*chunk)) for sr in range(first//chunk,min(128,(last+chunk-1)//chunk)))
        assert covered==last-first,('world56 coverage',key,rank)
layers={
    'backbone':sorted({int(m.group(1)) for k in keys if (m:=re.match(r'^layers\.(\d+)\.',k))}),
    'speech_tower':sorted({int(m.group(1)) for k in keys if (m:=re.match(r'^tower_s\.layers\.(\d+)\.',k))}),
    'pose_tower':sorted({int(m.group(1)) for k in keys if (m:=re.match(r'^tower_p\.layers\.(\d+)\.',k))}),
    'speech_decoder':sorted({int(m.group(1)) for k in keys if (m:=re.match(r'^dec_s\.model\.layers\.(\d+)\.',k))}),
    'pose_decoder':sorted({int(m.group(1)) for k in keys if (m:=re.match(r'^dec_p\.model\.layers\.(\d+)\.',k))}),
}
assert layers['backbone']==list(range(64)) and layers['speech_tower']==list(range(36)) and layers['pose_tower']==list(range(36))
assert layers['speech_decoder']==list(range(28)) and layers['pose_decoder']==list(range(28))
d33={}
for key in ('decp_bb.weight','decp_bb.bias'):
    assert key in global_shapes
    tensors=[]
    for rank in range(128):
        s=torch.load(root/f'rank{rank}.pt',map_location='cpu',mmap=True,weights_only=True)
        tensors.append(s[key])
    tensor=torch.cat(tensors,dim=0)
    assert tuple(tensor.shape)==global_shapes[key] and torch.isfinite(tensor).all() and torch.count_nonzero(tensor)>0
    d33[key]={'shape':list(tensor.shape),'dtype':str(tensor.dtype),'nonzero':int(torch.count_nonzero(tensor)),
      'sha256_raw_bf16':hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()}
report={'ok':True,'source_step':17850,'source_run_tag':meta['run_tag'],'source_hf':meta['src'],
    'format':meta['format'],'source_world':128,'target_world':56,'target_reshard_row_coverage_passed':True,
    'unique_parameters':len(keys),'sharded_parameters':len(reference),'replicated_parameters':len(rep),
    'parameter_elements':elements,'dtype':'bfloat16','layer_counts':{k:len(v) for k,v in layers.items()},
    'd33':d33,'base_decp_bb_off':True,'xabi_finetune_decp_bb_alpha':1.0,
    'pose_head_present':False,'s2p_present':False,'g0cond_present':False,'optimizer_state_present':False,
    'validation_scope':'All tensor headers/dtypes/shard shapes and world56 row coverage; learned D33 tensor contents are finite/nonzero and hashed. Full-file SHA256 separately verifies exact source-copy identity.',
    'torch_version':torch.__version__,'elapsed_s':round(time.time()-start,2)}
(base/'provenance/weight-validation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
