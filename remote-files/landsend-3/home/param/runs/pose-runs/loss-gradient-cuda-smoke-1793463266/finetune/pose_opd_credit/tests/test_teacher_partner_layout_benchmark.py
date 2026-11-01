"""Whole-node row ownership and concurrent-worker failure handling."""
from contextlib import redirect_stderr
import io
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

from finetune.pose_opd_credit.teacher_partner_layout_benchmark import (
    arguments, groups_for, parse_candidate, prepare_rows, result_fingerprint, run_groups,
)


class LayoutTests(unittest.TestCase):
    def test_all_layouts_cover_same48_rows_without_overlapping_devices(self):
        for layout,counts,gpus in [('baseline',[24,24],8),('fanout3',[16,16,16],8),('fanout2',[24,24],6)]:
            groups=groups_for(layout)
            rows=[i for g in groups for i in range(g['begin'],g['end'])]
            devices=[d for g in groups for pair in [g['student'],*g['partners']] for d in pair]
            self.assertEqual(rows,list(range(48)))
            self.assertEqual(len(devices),len(set(devices)))
            self.assertEqual(len(devices),gpus)
            self.assertEqual([n for g in groups for n in g['sizes']],counts)
            for group in groups:self.assertEqual(sum(group['sizes']),group['end']-group['begin'])

    def test_cli_cannot_silently_shrink_full_capacity_or_skip_warm_comparison(self):
        required=['--config','/tmp/recipe.json','--output','/tmp/results']
        args=arguments(required)
        self.assertEqual((args.frames,args.capacity_frames,args.warm_trials),(375,1500,2))
        for flags in [['--capacity-frames','375'],['--warm-trials','0'],['--warmup-trials','0'],
                      ['--candidates','fanout3:25:33:nan'],['--candidates','fanout3:25:33:-1']]:
            with self.assertRaises(SystemExit),redirect_stderr(io.StringIO()):arguments(required+flags)
        self.assertEqual(parse_candidate('fanout3:24:33:5.713').reserve_gib,5.713)
        self.assertFalse(parse_candidate('fanout3:25:33:0:serial').concurrent)
        lower=parse_candidate('fanout2:25:33:0:threaded:38')
        self.assertEqual(lower.rows,38)
        self.assertEqual(groups_for(lower.layout,lower.rows)[0]['sizes'],[19,19])

    def test_independent_teams_rendezvous_and_preserve_result_order(self):
        entered=[]
        def worker(index):
            def call(barrier):
                entered.append(index)
                barrier.wait(timeout=1)
                self.assertEqual(sorted(entered),[0,1])
                return index
            return call
        self.assertEqual(run_groups([worker(0),worker(1)]),[0,1])

    def test_worker_failure_aborts_peer_and_preserves_original_error(self):
        ready=threading.Event()
        def waiting(barrier):
            ready.set()
            barrier.wait(timeout=1)
        def failing(barrier):
            self.assertTrue(ready.wait(timeout=1))
            raise MemoryError('original allocation failure')
        with self.assertRaisesRegex(MemoryError,'original allocation failure'):
            run_groups([waiting,failing])

    def test_schedule_parity_covers_tokens_heard_pcm_and_input_latents(self):
        rollout=SimpleNamespace(text=np.array([1,2]),audio=np.array([[3,4]]),pose=np.array([[5,6]]),
            sampled_masks={'text':np.array([True,True])},behavior_logprobs={'text':np.array([-.2,-.3])})
        result={'rollouts':[rollout],'latents':[np.zeros((2,3),np.float32)],
            'waveforms':[np.zeros(8,np.float32)],'partner_rollouts':[{'dep_b':np.array([7,8])}],
            'timings':{'total_seconds':10}}
        original=result_fingerprint([result])
        result['timings']['total_seconds']=20
        self.assertEqual(original,result_fingerprint([result]))
        for array in [rollout.text,result['latents'][0],result['waveforms'][0],result['partner_rollouts'][0]['dep_b']]:
            array.flat[0]+=1
            self.assertNotEqual(original,result_fingerprint([result]))
            array.flat[0]-=1

    def test_prepared_cache_preserves_selection_and_rejects_changed_recipe(self):
        from dataclasses import replace
        from finetune.pose_opd_credit.config import TrainingConfig
        cfg=TrainingConfig()
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            (root/'identity.json').write_text(json.dumps({'config':cfg.to_dict()}))
            (root/'rows.json').write_text(json.dumps({'eligible_rows':100,'selected_items':[[0,i,0] for i in range(48)]}))
            args=arguments(['--config','unused','--output','unused','--items-json',str(root/'rows.json'),
                            '--prepared-rows',str(root/'prepared.pkl')])
            class FakeRows:
                def __init__(self,config):pass
                def load(self,item,**kwargs):return {'conversation_frames':375,'prefill_frames':38,'item':item}
            with patch('finetune.pose_opd_credit.data.ProductionRows',FakeRows),patch('finetune.pose_opd_credit.data.prefix_seed',lambda row,frames:row):
                expected=prepare_rows(cfg,args,lambda *args,**kwargs:None)
            with patch('finetune.pose_opd_credit.data.ProductionRows',side_effect=AssertionError('cache must skip loader')):
                self.assertEqual(prepare_rows(cfg,args,lambda *args,**kwargs:None),expected)
                with self.assertRaisesRegex(ValueError,'differs'):
                    prepare_rows(replace(cfg,seed=cfg.seed+1),args,lambda *args,**kwargs:None)


if __name__=='__main__':unittest.main()
