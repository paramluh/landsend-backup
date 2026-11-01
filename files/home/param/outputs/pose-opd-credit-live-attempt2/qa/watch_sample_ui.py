import json,time
from pathlib import Path
from playwright.sync_api import sync_playwright
out=Path('/home/param/outputs/pose-opd-credit-live-attempt2/qa')
with sync_playwright() as p:
 b=p.chromium.launch(headless=True,args=['--disable-gpu']);page=b.new_page(viewport={'width':1440,'height':1100});errors=[];requests=[];nav=[]
 page.on('pageerror',lambda e:errors.append(str(e)));page.on('request',lambda r:requests.append({'url':r.url,'time':time.time()}) if 'manifest.json' in r.url else None);page.on('framenavigated',lambda f:nav.append(f.url) if f==page.main_frame else None)
 page.goto('https://mph-daniel-containers-afternoon.trycloudflare.com/samples/index.html',wait_until='domcontentloaded');initial=page.locator('#notice').inner_text();print(json.dumps({'initial':initial}),flush=True)
 page.wait_for_function("document.querySelector('#notice').textContent.startsWith('4 of 4 samples published')",timeout=300000)
 page.wait_for_function("document.querySelectorAll('canvas').length===4",timeout=30000);page.wait_for_timeout(5000)
 assert page.locator('article.sample').count()==4;assert page.locator('.tag').all_inner_texts()==['ROLLOUT']*4
 text=page.locator('body').inner_text();assert text.count('2D normalized skeleton')==4,text
 assert len(nav)==1,nav;assert not errors,errors
 page.screenshot(path=str(out/'four-real-sample-ui.png'),full_page=True)
 (out/'four-real-sample-ui.json').write_text(json.dumps(dict(initial=initial,notice=page.locator('#notice').inner_text(),samples=4,navigation_count=len(nav),manifest_fetches=requests,errors=errors,representation='2D normalized skeleton (z=0), not3D motion'),indent=2));print('FOUR_REAL_SAMPLES_AUTOMATICALLY_VISIBLE',flush=True);b.close()
