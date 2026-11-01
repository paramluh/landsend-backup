import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from pose_doc import build_doc

ROOT=Path(__file__).resolve().parents[1]


class PreviewIntegrationTest(unittest.TestCase):
    def test_new_telemetry_prefixes_survive_reader(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            (root/'metrics.jsonl').write_text(json.dumps({'update':1,'kl/lexical':.25,'kl/category':.03,
                'grad_norm/direct/layers':.7,'grad_norm/credit/layers':.2,
                'grad_norm/combined/layers':.8,'logit_grad_norm/direct/text':.01,
                'logit_grad_alignment/text':-.5,'logit_grad_credit_direct_ratio/text':2,
                'conversation/student/probability/word':.35,
                'conversation/partner/sampled_count/new_word':7,
                'conversation/partner/decisions':50})+'\n')
            doc=build_doc(root)
            self.assertIn('kl/lexical',doc['train_cols'])
            self.assertIn('grad_norm/credit/layers',doc['train_cols'])
            self.assertIn('grad_norm/combined/layers',doc['train_cols'])
            self.assertIn('logit_grad_norm/direct/text',doc['train_cols'])
            self.assertIn('logit_grad_alignment/text',doc['train_cols'])
            self.assertIn('logit_grad_credit_direct_ratio/text',doc['train_cols'])
            for key, expected in (('conversation/student/probability/word',.35),
                                  ('conversation/partner/sampled_count/new_word',7),
                                  ('conversation/partner/decisions',50)):
                self.assertIn(key,doc['train_cols'])
                self.assertEqual(doc['train'][0][doc['train_cols'].index(key)],expected)
            self.assertEqual(len(doc['train']),1)

    def test_live_build_removes_demo_and_uses_prepared_coefficients(self):
        spec=importlib.util.spec_from_file_location('preview_test',ROOT/'pose-preview'/'make_demo.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as d:
            work=Path(d)/'preview';shutil.copytree(ROOT/'pose-preview',work,ignore=shutil.ignore_patterns('dist','qa','__pycache__'))
            module.ROOT=work;module.SRC=work/'source';module.DIST=work/'dist'
            (work/'settings.json').write_text(json.dumps({'mode':'demo','run_dir':None,'samples_dir':None}))
            module.build()
            self.assertTrue(json.loads((module.DIST/'samples'/'manifest.json').read_text())['synthetic'])
            run=Path(d)/'run';run.mkdir()
            (run/'preparation.json').write_text(json.dumps({'identity':{'method':{'modality_weights':{'text':.5}}}}))
            (run/'metrics.jsonl').write_text('{"update":1,"loss/text":0.2}\n')
            (work/'settings.json').write_text(json.dumps({'mode':'live','run_dir':str(run),'samples_dir':None}))
            module.build()
            manifest=json.loads((module.DIST/'samples'/'manifest.json').read_text())
            self.assertFalse(manifest['synthetic']);self.assertEqual(manifest['rollouts'],[])
            self.assertFalse((module.DIST/'samples'/'assets').exists())
            doc=json.loads((module.DIST/'s3live_pose_opd_credit.json').read_text())
            self.assertAlmostEqual(doc['train'][0][doc['train_cols'].index('kl/text')],.4)
            self.assertIsNone(doc['train'][0][doc['train_cols'].index('kl/lexical')])
            self.assertNotIn('synthetic phase timings',(module.DIST/'opd_prog.json').read_text())
            # A new update reaches BOTH feeds without waiting for an HTML rebuild.
            html_before=(module.DIST/'index.html').read_text()
            (run/'metrics.jsonl').write_text('{"update":1,"loss/text":0.2}\n{"update":2,"loss/text":0.1}\n')
            module.refresh_progress()
            fresh=json.loads((module.DIST/'s3live_pose_opd_credit.json').read_text())
            progress=json.loads((module.DIST/'opd_prog.json').read_text())
            self.assertEqual(fresh['step'],2);self.assertEqual(progress['completed'],2)
            self.assertEqual((module.DIST/'index.html').read_text(),html_before)
            self.assertIn('setInterval(poll,5000)',html_before)
            self.assertNotIn('if(PA&&ASIGF()!==ASIG){location.reload()',html_before)
            self.assertIn('Combined parameter gradient',html_before)
            self.assertIn('not a parameter-gradient norm',html_before)
            self.assertIn("f.src='samples/index.html'",html_before)


if __name__=='__main__':unittest.main()
