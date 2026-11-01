"""Explicit latency proxy, not a reverse-fanout rollout or fit validation."""
import json,pathlib,sys,time,statistics
import numpy as np
import torch
sys.path.insert(0,str(pathlib.Path.cwd()))
from finetune.pose_opd_credit.config import TrainingConfig
from finetune.pose_opd_credit.data import ProductionRows,prefix_seed
from finetune.pose_opd_credit.train import checkpoint_spec,load_actor_native
from finetune.pose_opd_credit import runtime
from finetune.pose_opd_credit.rollout import make_fastppm_class,NATURAL_SAMPLING
from finetune.pose_opd_credit.partner_rollout import _instances
from finetune.pose_opd_credit.partner_block import CommittedBlockReplay
from finetune.pose_opd_credit.teacher_partner_benchmark import _build_model,_build_engine_pair

ROOT=pathlib.Path(__file__).parent
def log(event,**data):print(json.dumps({'event':event,'time':time.time(),**data},default=str),flush=True)
def sync():
    for d in (0,1):torch.cuda.synchronize(d)
def measure(fn):
    for _ in range(3):fn()
    sync();v=[]
    for _ in range(10):
        t=time.perf_counter();fn();sync();v.append(1000*(time.perf_counter()-t))
    return {'median_ms':statistics.median(v),'trials_ms':v}

@torch.inference_mode()
def main():
    torch.set_num_threads(1)
    cfg=TrainingConfig.from_dict(json.loads(pathlib.Path('finetune/pose_opd_credit/recipe.teacher_partner.json').read_text()))
    items=json.loads(pathlib.Path('/home/param/runs/pose-runs/teacher-partner-faststate-check-v1/selected.json').read_text())['selected_items']
    row=prefix_seed(ProductionRows(cfg).load(items[0],rank=0,update=0,slot=0),375)['partner']
    log('one_admitted_real_row_loaded',replicated_to=48)
    M=runtime.load_native();M.QWEN32='/home/param/models/imported-weights/ppm-811-17850-base-bf16/configs/Qwen3-32B';M.QWEN4B_CFG='/home/param/models/imported-weights/ppm-811-17850-base-bf16/configs/Qwen3-4B'
    native=load_actor_native(M)
    spec=checkpoint_spec(cfg.resolved_partner_checkpoint,'audio_teacher',gain_policy=cfg.audio_teacher_gain_policy)
    devices=tuple(torch.device(f'cuda:{i}') for i in (0,1))
    model,placement,cfgs=_build_model(torch,M,spec,devices,backbone_cut=cfg.partner_actor_backbone_cut,require_pose=False,log=lambda event,**kw:log(event,role=kw.get('role')))
    loaded={'partner':{'model':model,'layer_devices':placement,'cfgs':cfgs,'devices':devices,'has_pose':False,'prefix_capacity':134,'actor_type':make_fastppm_class(native,require_pose=False)}}
    engines,setup=_build_engine_pair(torch,cfg,loaded,instances=48,capacity=1500,graphs=True,log=log)
    e=engines['partner']
    e.start(_instances(e,[row]*48,extra=8),list(range(48)),params=dict(NATURAL_SAMPLING),f0=0)
    # The real voice prefix is prefilled. Later temporal caches remain zero:
    # this is a native graph shape/latency measurement, not generated history.
    p=int(np.asarray(row['P']).reshape(-1)[0]);n=len(row['ids'][1])
    for r in range(48):
        e.laneB[r,:n].copy_(torch.as_tensor(row['ids'][1],device=e.dl))
        e.depB[r,:n].copy_(torch.as_tensor(row['depth'][1],device=e.dl))
    block=CommittedBlockReplay(e,6,graphs=True)
    start=200
    def replay():
        e._opd_next_frame=start;e._opd_pending_frame=None
        block.replay(start,start+6)
    replay();sync()
    results={'batch':48,'frames':6,'capacity':1500,'position':p+start,
        'proxy_only':True,'history':'Real voice prefix; zero later KV and zero external audio features; repeated one admitted row',
        'six_frame_block':measure(replay),'setup':setup}
    e.set_f(start+6)
    def fresh():
        with torch.cuda.device(e.dl):
            e.graphs['text'].replay();e.graphs['speech_full'].replay()
    results['fresh_text_audio']=measure(fresh)
    def combined():replay();fresh()
    results['combined_teacher']=measure(combined)
    for key in ('six_frame_block','fresh_text_audio','combined_teacher'):log(key,**results[key])
    results['device_allocated_gib']={str(d):torch.cuda.memory_allocated(d)/2**30 for d in devices}
    (ROOT/'results.json').write_text(json.dumps(results,indent=2));log('complete')
if __name__=='__main__':main()
