"""CPU-only exact first-update membership/prefill/seed preparation; never launches GPUs."""
import argparse,hashlib,json,os,pickle,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--code',type=Path,required=True);p.add_argument('--selection',type=Path,required=True);p.add_argument('--rank-start',type=int,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
sys.path.insert(0,str(a.code))
from finetune.pose_opd_credit.config import TrainingConfig
from finetune.pose_opd_credit.data import ProductionRows,prefix_seed
from finetune.pose_opd_credit.train import rollout_seed
cfg=TrainingConfig.from_dict(json.loads((a.code/'finetune/pose_opd_credit/recipe.teacher_partner.json').read_text()))
serialized=json.loads(json.dumps(cfg.to_dict()));selection_bytes=a.selection.read_bytes();selection=json.loads(selection_bytes);assert len(selection['selected_items'])==48
rows=[];seeds=[];assignments=[];data=ProductionRows(cfg)
for i,item in enumerate(selection['selected_items']):
 rank=a.rank_start+i//6;slot=i%6
 rows.append(prefix_seed(data.load(item,rank=rank,update=0,slot=slot),1500));seeds.append(rollout_seed(cfg.seed,0,rank,slot));assignments.append({'rank':rank,'slot':slot})
key={'config':serialized,'frames':1500,'row_offset':0,'data_sha256':hashlib.sha256((a.code/'finetune/pose_opd_credit/data.py').read_bytes()).hexdigest(),'selection_sha256':hashlib.sha256(selection_bytes).hexdigest()}
prepared={'eligible_rows':selection['eligible_rows'],'selected_items':selection['selected_items'],'seeds':seeds,'prefill_frames':[int(r['prefill_frames']) for r in rows],'production_assignments':assignments}
a.output.parent.mkdir(parents=True,exist_ok=True);temp=a.output.with_suffix('.tmp')
with temp.open('wb') as f:os.chmod(temp,0o600);pickle.dump({'format':'teacher-partner-prepared-rows-v1','key':key,'rows':rows,'seeds':seeds,'preparation':prepared},f,protocol=pickle.HIGHEST_PROTOCOL)
temp.replace(a.output)
print(json.dumps({'prepared':str(a.output),'rows':len(rows),'gpu_launched':False,'rank_start':a.rank_start}))
