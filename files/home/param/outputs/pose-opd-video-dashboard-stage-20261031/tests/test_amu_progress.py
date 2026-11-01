import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from amu_progress import adapt, renderer
from pose_doc import build_doc

ROOT=Path(__file__).resolve().parents[1]

class AmuProgressTest(unittest.TestCase):
    def test_retained_layout_css_and_original_poll_cadence(self):
        source=(ROOT/'pose-preview/source/opd_prog.js').read_text()
        self.assertEqual(hashlib.sha256(source.encode()).hexdigest(),'c2b828113ea08e3f734b244e620555ca547bf6470646c908fe9624f9c59f4b82')
        patched=renderer(source)
        self.assertEqual(source.split('st.textContent=`')[1].split('`;')[0],patched.split('st.textContent=`')[1].split('`;')[0])
        self.assertIn('poll();setInterval(poll,5000);setInterval(render,1000);setInterval(redraw,15000)',patched)
        self.assertIn('P.stopped?(P.terminal_at||P.now):NOW()',patched)

    def test_measured_phases_and_real_rollout_counts_then_failure(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'plan.json').write_text(json.dumps({'updates':[[]]*129}))
            rows=[]
            names=['data_audio_begin','actor_rollout_begin','teacher_update_begin','replay_microbatch_begin','optimizer_begin']
            for n,t in zip(names,[100,110,200,220,260]):
                rows.append(dict(event='phase',phase=n,update=1,time=t))
            rows.append(dict(event='update',update=1,time=265,seconds=165,**{'loss/direct':.2}))
            rows.extend([dict(event='phase',phase='data_audio_begin',update=2,time=270),dict(event='phase',phase='actor_rollout_begin',update=2,time=280),dict(event='rollout_progress',update=2,time=300,current_frame=275,total_frames=1502,seconds_elapsed=20)])
            (root/'events-rank0.jsonl').write_text('\n'.join(map(json.dumps,rows)))
            p=adapt(301,build_doc(root));self.assertEqual(p['update'],2);self.assertEqual(p['completed'],1)
            self.assertEqual([x['ref'] for x in p['phases']],[10,90,20,40,5]);self.assertEqual(p['phase'],1)
            self.assertEqual(p['sub']['done'],275);self.assertIn('12.5 submitted steps/s',p['sub']['text'])
            self.assertEqual(p['sub']['unit'],'positions')
            self.assertEqual(p['sub']['text'],'rank 0 GPU pair · submitted model-frame positions (includes prefill and media delay) · 12.5 submitted steps/s')
            (root/'supervisor').mkdir();(root/'supervisor/finished.json').write_text('{"exit_code":1,"time":305}')
            p=adapt(999,build_doc(root));self.assertTrue(p['failed']);self.assertEqual(p['terminal_at'],305)

    def test_startup_and_unknown_references_not_invented(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'events-rank0.jsonl').write_text('{"event":"phase","phase":"model_loading","time":100}')
            p=adapt(110,build_doc(root));self.assertTrue(p['idle']);self.assertIsNone(p['update_start'])
            self.assertTrue(all(x['ref'] is None for x in p['phases']))

    def test_partner_physical_clock_uses_actual_batch_prefill_and_team(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            (root/'config.json').write_text(json.dumps({'method':{'prefill_frames':38}}))
            rows=[dict(event='phase',phase='data_audio_begin',update=1,time=100),
                  dict(event='phase',phase='actor_rollout_begin',update=1,time=110),
                  dict(event='rollout_progress',update=1,time=120,current_frame=68,
                       total_frames=1500,seconds_elapsed=2,frame_clock='physical',
                       generated_start_frame=43,actor_group_gpus=4)]
            (root/'events-rank0.jsonl').write_text('\n'.join(map(json.dumps,rows)))
            doc=build_doc(root)
            self.assertEqual(doc['rollout_progress']['generated_start_frame'],43)
            p=adapt(121,doc)
            self.assertEqual(p['sub']['done'],68)
            self.assertEqual(p['sub']['total'],1500)
            self.assertEqual(p['sub']['unit'],'frames')
            self.assertEqual(p['sub']['text'],'rank 0 4-GPU team · submitted physical frames (includes prefill; excludes media-delay slots) · 12.5 submitted frames/s')

if __name__=='__main__':unittest.main()
