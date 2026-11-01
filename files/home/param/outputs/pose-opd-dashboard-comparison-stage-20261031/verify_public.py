from pathlib import Path
import hashlib,json,time
from playwright.sync_api import sync_playwright
ROOT=Path('/home/param/outputs/pose-opd-dashboard-comparison-stage-20261031')
url='https://mph-daniel-containers-afternoon.trycloudflare.com/'
with sync_playwright() as p:
 browser=p.chromium.launch(headless=True,args=['--disable-gpu']);page=browser.new_page(viewport={'width':1500,'height':1100})
 errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
 page.goto(url,wait_until='domcontentloaded');page.wait_for_selector('#pose-comparison-range',timeout=30000)
 page.wait_for_function("typeof ch!=='undefined'&&ch.a_kl_text&&S().branches?.length===1")
 current=page.evaluate("()=>({range:POSE_COMPARE,step:S().step,max:ch.a_kl_text.options.scales.x.max,previous:S().branches[0].step,curves:ch.a_kl_text.data.datasets.filter(d=>!d._tr).map(d=>({label:d.label,hidden:isHid('a_kl_text',d),points:d.data.length})),lexical:chA('kl/lexical',0).filter(d=>!d._tr).map(d=>d.label),category:chA('kl/category',0).filter(d=>!d._tr).map(d=>d.label)})")
 assert current['range']=='current' and current['max']==current['step'],current
 assert current['previous']==89 and current['curves'][1]['label']=='Previous fixed-audio run' and not current['curves'][1]['hidden'],current
 assert len(current['lexical'])==1 and len(current['category'])==1,current
 page.locator('#pose-comparison-range').select_option('all')
 all_runs=page.evaluate("({range:POSE_COMPARE,max:ch.a_kl_text.options.scales.x.max})")
 assert all_runs=={'range':'all','max':89},all_runs
 page.locator('#pose-comparison-range').select_option('current')
 page.screenshot(path=str(ROOT/'PUBLIC-COMPARISON.png'),full_page=False)
 assert not errors,errors
 live=Path('/home/param/websites/pose-opd-credit-dashboard/pose-preview/samples')
 before=json.loads((ROOT/'deployment.json').read_text())['samples_sha256_before']
 after={name:hashlib.sha256((live/name).read_bytes()).hexdigest() for name in before}
 assert before==after
 result={'url':url,'time':time.time(),'current':current,'all_runs':all_runs,'sample_assets_unchanged':True,'page_errors':errors}
 (ROOT/'PUBLIC-COMPARISON.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));browser.close()
