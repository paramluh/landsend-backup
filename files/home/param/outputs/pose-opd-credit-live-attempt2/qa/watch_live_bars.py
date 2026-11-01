import json,time,re
from pathlib import Path
from playwright.sync_api import sync_playwright
out=Path('/home/param/outputs/pose-opd-credit-live-attempt2/qa')
with sync_playwright() as p:
 b=p.chromium.launch(headless=True,args=['--disable-gpu']);page=b.new_page(viewport={'width':1440,'height':1050});errors=[];page.on('pageerror',lambda e:errors.append(str(e)));responses=[]
 page.on('response',lambda r:responses.append({'time':time.time(),'url':r.url}) if 'opd_prog.json' in r.url else None)
 page.goto('https://mph-daniel-containers-afternoon.trycloudflare.com',wait_until='networkidle');seen=[];last=None
 for _ in range(120):
  page.wait_for_timeout(5000)
  text=page.locator('#opdp').inner_text();m=re.search(r'(\d+)/1502 positions',text)
  if m and int(m.group(1)) not in seen:
   seen.append(int(m.group(1)));print(json.dumps({'time':time.time(),'frame':seen[-1],'bar':text}),flush=True)
  state=page.evaluate('({run:D.s3.run,state:D.s3.state,step:D.s3.step,max:D.s3.max_steps,epochs:D.s3.epochs,learning_rate_peak:D.s3.learning_rate_peak})')
  if state!=last:print(json.dumps({'time':time.time(),'doc':state}),flush=True);last=state
  if len(seen)>=5:
   assert state['run'].endswith('attempt2') and state['max']==129 and state['epochs']==3
   assert abs(state['learning_rate_peak']-3e-6)<1e-15
   assert not errors,errors
   page.screenshot(path=str(out/'actual-advancing-bars.png'),full_page=False)
   (out/'actual-advancing-bars.json').write_text(json.dumps({'doc':state,'frames':seen,'fetches':responses,'errors':errors,'text':text},indent=2));break
 else:raise RuntimeError('No five actual rollout progress changes within ten minutes')
 b.close()
