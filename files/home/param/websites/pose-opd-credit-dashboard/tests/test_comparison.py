"""Historical curves and observed-range behavior on the rendered dashboard."""
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'pose-preview'))
sys.path.insert(0, str(ROOT))
import comparison
import make_demo
from pose_metrics import enrich_doc


class ReferenceTest(unittest.TestCase):
    def test_reference_uses_its_own_weights_and_does_not_invent_diagnostics(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            (run / 'metrics.jsonl').write_text('')
            method = {'modality_weights': {'text': 1/3, 'audio': 1/3, 'pose': 1/3},
                      'codebook_weights': [.5] + [.5/31]*31}
            (run / 'config.json').write_text(json.dumps({'method': method}))
            raw = {'run': 'old', 'train_cols': ['step', 'rows', 'loss/pose_cb0'],
                   'train': [[1, 320, .5]], 'step': 1}
            with patch('pose_doc.build_doc', return_value=raw):
                [doc] = comparison.reference_docs({'reference_runs': [{'run_dir': folder, 'label': 'Previous fixed-audio run'}]})
            values = dict(zip(doc['train_cols'], doc['train'][0]))
            self.assertEqual(values['kl/pose/tower'], 3)
            self.assertIsNone(values['kl/lexical'])
            self.assertIsNone(values['kl/category'])
            self.assertNotIn('grad_norm/credit/all', values)
            self.assertEqual(doc['label'], 'Previous fixed-audio run')
            self.assertFalse(doc.get('archived', False))


class Quiet(SimpleHTTPRequestHandler):
    def log_message(self, *args): pass


class ComparisonBrowserTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from playwright.sync_api import sync_playwright
        cls.tmp = tempfile.TemporaryDirectory()
        cls.source = Path(cls.tmp.name) / 'source'
        shutil.copytree(make_demo.SRC, cls.source)
        method = {'modality_weights': {'text': 1/3, 'audio': 1/3, 'pose': 1/3},
                  'codebook_weights': [.5] + [.5/31]*31}
        def doc(count, label):
            result = make_demo.demo_doc(1000)
            records = [make_demo.synthetic_row(i) for i in range(1, count+1)]
            keys = list(records[0])
            result.update(run=label, label=label, short=label, step=count, max_steps=123,
                          train_cols=['step','rows']+keys,
                          train=[[i,i*320]+[row[k] for k in keys] for i,row in enumerate(records,1)],
                          xwin=None, critical_charts=make_demo.CRITICAL, extra_charts=make_demo.EXTRAS)
            return enrich_doc(result, method)
        current, old = doc(3, 'Current run'), doc(89, 'Previous fixed-audio run')
        for key in ('kl/lexical','kl/category'):
            index=old['train_cols'].index(key)
            for row in old['train']:row[index]=None
        old.update(color='#b36b00', color_dark='#ffd43b', dash=[6,3])
        # A late historical outlier must not determine current-window y limits.
        old['train'][-1][old['train_cols'].index('kl/text')] = 1e6
        current['branches'] = [old]
        current['grad_components'] = [[g['component'],g['label']] for g in current['grad_groups']]
        cls.doc = current
        (cls.source/f's3branches_{make_demo.PAGE}.json').write_text(json.dumps(make_demo.config()))
        (cls.source/f's3live_{make_demo.PAGE}.json').write_text(json.dumps(current))
        result = subprocess.run([sys.executable, str(cls.source/'build_stage3.py'),str(cls.source),'--no-doc'],
            env=dict(os.environ,S3_PAGE=make_demo.PAGE,CANOPY_ROOT=str(cls.source/'empty'),CURVES_DIR=str(cls.source)),
            capture_output=True,text=True,check=True)
        if 'replace anchor missing:' in result.stdout:
            raise AssertionError(result.stdout)
        cls.server = ThreadingHTTPServer(('127.0.0.1',0),partial(Quiet,directory=str(cls.source)))
        cls.worker = threading.Thread(target=cls.server.serve_forever,daemon=True);cls.worker.start()
        cls.pw = sync_playwright().start()
        cls.browser = cls.pw.chromium.launch(headless=True,args=['--disable-gpu'])
        cls.url = f'http://127.0.0.1:{cls.server.server_port}/stage3_{make_demo.PAGE}.html'

    @classmethod
    def tearDownClass(cls):
        cls.browser.close();cls.pw.stop();cls.server.shutdown();cls.server.server_close();cls.tmp.cleanup()

    def setUp(self):
        self.context = self.browser.new_context()
        self.page = self.context.new_page()
        self.errors=[];self.page.on('pageerror',lambda e:self.errors.append(str(e)))
        self.page.goto(self.url,wait_until='domcontentloaded',timeout=15000)
        self.page.wait_for_function("typeof ch!=='undefined'&&ch.a_kl_text&&ch.a_kl_text.data.datasets.length>=2")

    def tearDown(self):
        self.assertFalse(self.errors,self.errors)
        self.context.close()

    def state(self):
        return self.page.evaluate("""()=>({range:POSE_COMPARE,x:ch.a_kl_text.options.scales.x.max,
          y:ch.a_kl_text.options.scales.y.max,curves:ch.a_kl_text.data.datasets.map(d=>({label:d.label,hidden:isHid('a_kl_text',d)}))})""")

    def test_current_default_and_all_runs_toggle_changes_visible_y_range(self):
        state=self.state()
        self.assertEqual(state['range'],'current');self.assertEqual(state['x'],3)
        self.assertLess(state['y'],1000)
        self.assertEqual(state['curves'][1],{'label':'Previous fixed-audio run','hidden':False})
        self.page.locator('#pose-comparison-range').select_option('all')
        state=self.state();self.assertEqual(state['x'],89);self.assertGreater(state['y'],1000)
        self.page.locator('#pose-comparison-range').select_option('current')
        self.assertEqual(self.state()['x'],3);self.assertLess(self.state()['y'],1000)

    def test_manual_x_max_overrides_both_ranges_and_setting_persists(self):
        self.page.locator('#xmaxs').evaluate("e=>{e.value='20';e.dispatchEvent(new Event('change',{bubbles:true}))}")
        self.assertEqual(self.state()['x'],20)
        self.page.locator('#pose-comparison-range').select_option('all')
        self.assertEqual(self.state()['x'],20)
        self.page.reload(wait_until='networkidle')
        self.assertEqual(self.state()['range'],'all');self.assertEqual(self.state()['x'],20)

    def test_all_charts_obey_global_range_and_missing_reference_is_absent(self):
        self.page.locator('#pose-comparison-range').select_option('all')
        self.page.evaluate("show('extras')")
        result=self.page.evaluate("Object.values(ch).map(c=>c.options.scales.x.max)")
        self.assertTrue(result);self.assertEqual(set(result),{89})
        self.page.locator('#pose-comparison-range').select_option('current')
        self.assertEqual(set(self.page.evaluate("Object.values(ch).map(c=>c.options.scales.x.max)")),{3})
        self.assertEqual(self.page.evaluate("chA('kl/lexical',0).filter(d=>!d._tr).length"),1)

    def test_new_points_advance_current_window_and_old_full_history_is_ignored(self):
        self.page.evaluate("LS.set('ppm7_afull',true)")
        self.page.reload(wait_until='networkidle');self.assertEqual(self.state()['range'],'current')
        self.page.evaluate("D.s3.step=5;upd('a_kl_text')")
        self.assertEqual(self.state()['x'],5)


if __name__ == '__main__': unittest.main()
