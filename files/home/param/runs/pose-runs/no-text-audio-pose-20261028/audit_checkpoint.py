import json,os,socket,sys
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
root=Path(sys.argv[1]);step=int(sys.argv[2]);node=socket.gethostname().split('.')[0];idx=int(node.rsplit('-',1)[1])-1
p=root/'saves'/f'step-{step:05d}';d=json.loads((p/'NODE_COMPLETE').read_text());assert d['step']==step and d['world']==32 and len(d['digests'])==32 and len(set(d['digests']))==32
assert (root/'LATEST').read_text().strip()==f'saves/step-{step:05d}'
ranks=[]
for rank in range(idx*8,idx*8+8):
 f=p/f'rank{rank}.pt';b=torch.load(f,map_location='cpu',weights_only=False,mmap=True)
 assert (b['step'],b['world'],b['rank'],b['host'],b['plan'])==(step,32,rank,node,d['plan'])
 assert b['model'] and b['opt'] and b['sched']['last_epoch']==step and b['lrs']
 assert all(t.dtype==torch.float32 for t in b['model'].values())
 assert all({'step','exp_avg','exp_avg_sq'}<=set(s) and int(s['step'].item())==step for s in b['opt'].values())
 ranks.append({'rank':rank,'bytes':f.stat().st_size,'model_tensors':len(b['model']),'optimizer_tensors':len(b['opt']),'scheduler_epoch':b['sched']['last_epoch']});del b
nd=torch.load(p/'node.pt',map_location='cpu',weights_only=False,mmap=True);assert nd['step']==step and nd['rep']
updates={}
for name in ['dec_s','dec_p']:
 dec=nd['dec'][name];assert dec['micros']==step and dec['opt']['state']
 assert all(int(s['step'].item())==step*8 for s in dec['opt']['state'].values())
 updates[name]={'outer_steps':dec['micros'],'adamw_updates':step*8}
del nd
export=root/'exports'/f'export-{step}'
for rank in range(idx*8,idx*8+8):
 b=torch.load(export/f'rank{rank}.pt',map_location='cpu',weights_only=False,mmap=True);assert b and all(t.dtype==torch.bfloat16 for t in b.values());del b
if idx==0:
 assert (export/'COMPLETE').read_text().strip()==str(step)
 m=json.loads((export/'meta.json').read_text());assert m['step']==step and m['world']==32
 rep=torch.load(export/'rep.pt',map_location='cpu',weights_only=False,mmap=True);assert rep and all(t.dtype==torch.bfloat16 for t in rep.values());del rep
print(json.dumps({'status':'pass','node':node,'step':step,'plan':d['plan'],'ranks':ranks,'decoder_updates':updates,'bf16_export_complete':True,'latest_matches':True,'retained_saves':sorted(x.name for x in (root/'saves').glob('step-*') if (x/'NODE_COMPLETE').is_file())}))
