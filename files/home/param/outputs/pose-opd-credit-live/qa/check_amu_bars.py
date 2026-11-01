import json,sys,threading,time,tempfile,shutil
from pathlib import Path
from http.server import ThreadingHTTPServer,SimpleHTTPRequestHandler
from playwright.sync_api import sync_playwright
ROOT=Path('/home/param/websites/pose-opd-credit-dashboard');sys.path.insert(0,str(ROOT))
from amu_progress import adapt
from pose_doc import build_doc
OUT=Path('/home/param/outputs/pose-opd-credit-live/qa');DIST=ROOT/'pose-preview/dist'
rows=[json.loads(x) for x in Path('/home/param/outputs/pose-opd-credit-live/current/events-rank0.jsonl').read_text().splitlines()]
positions=[i for i,r in enumerate(rows) if r.get('event')=='rollout_progress'][:2]
assert len(positions)==2
payloads=[]
with tempfile.TemporaryDirectory() as d:
    run=Path(d)
    for name in ['config.json','plan.json']:shutil.copy2(Path('/home/param/outputs/pose-opd-credit-live/current')/name,run/name)
    for i in positions:
        (run/'events-rank0.jsonl').write_text('\n'.join(map(json.dumps,rows[:i+1])))
        payloads.append(adapt(rows[i]['time'],build_doc(run)))
active=[payloads[0]]
class Handler(SimpleHTTPRequestHandler):
    def __init__(self,*a,**kw):super().__init__(*a,directory=str(DIST),**kw)
    def log_message(self,*a):pass
    def do_GET(self):
        path=self.path.split('?')[0]
        data=None;typ='application/json'
        if path=='/opd_prog.json':data=json.dumps(active[0]).encode()
        elif path=='/baseline.html':data=(DIST/'index.html').read_bytes().replace(b'opd_prog.js',b'upstream_prog.js');typ='text/html'
        elif path=='/upstream_prog.js':data=(ROOT/'pose-preview/source/opd_prog.js').read_bytes();typ='application/javascript'
        if data is None:return super().do_GET()
        self.send_response(200);self.send_header('Content-Type',typ);self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(data)
server=ThreadingHTTPServer(('127.0.0.1',9237),Handler);threading.Thread(target=server.serve_forever,daemon=True).start()
with sync_playwright() as p:
    browser=p.chromium.launch(headless=True,args=['--disable-gpu']);page=browser.new_page(viewport={'width':1440,'height':1050});errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
    page.goto('https://mph-daniel-containers-afternoon.trycloudflare.com',wait_until='networkidle');page.wait_for_selector('#opdp .ph')
    public=page.locator('#opdp').inner_text();assert 'failed · update 2 of 129' in public,public
    assert page.locator('#opdp .pc').count()==5
    assert page.locator('#pose-live-progress').count()==0
    before=page.locator('#opdp .mid').all_inner_texts();page.wait_for_timeout(6200);after=page.locator('#opdp .mid').all_inner_texts();assert before==after,(before,after)
    assert 'update 1 / 129' in page.locator('.hdr').inner_text()
    page.screenshot(path=str(OUT/'amu-public-failed.png'),full_page=False)
    page.goto('http://127.0.0.1:9237',wait_until='networkidle');page.wait_for_selector('#opdp .pc.cur')
    first=page.locator('#opdp').inner_text();w1=page.locator('#opdp .pc.cur .pf').get_attribute('style')
    assert '275/1502' in first,first
    page.screenshot(path=str(OUT/'amu-active-first.png'),full_page=False)
    active[0]=payloads[1];page.wait_for_timeout(5600)
    second=page.locator('#opdp').inner_text();w2=page.locator('#opdp .pc.cur .pf').get_attribute('style')
    assert '525/1502' in second,(first,second);assert w1!=w2
    assert 'submitted steps/s' in second
    page.screenshot(path=str(OUT/'amu-active-next.png'),full_page=False)
    css=page.locator('#opdp').evaluate('(e)=>{let s=getComputedStyle(e);return {padding:s.padding,borderRadius:s.borderRadius,fontSize:s.fontSize}}')
    phasecss=page.locator('#opdp .pb').first.evaluate('(e)=>{let s=getComputedStyle(e);return {height:s.height,borderRadius:s.borderRadius}}')
    page.goto('http://127.0.0.1:9237/baseline.html',wait_until='networkidle');page.wait_for_selector('#opdp .ph')
    baselinecss=page.locator('#opdp').evaluate('(e)=>{let s=getComputedStyle(e);return {padding:s.padding,borderRadius:s.borderRadius,fontSize:s.fontSize}}')
    baselinephasecss=page.locator('#opdp .pb').first.evaluate('(e)=>{let s=getComputedStyle(e);return {height:s.height,borderRadius:s.borderRadius}}')
    assert css==baselinecss and phasecss==baselinephasecss
    page.screenshot(path=str(OUT/'amu-upstream-baseline.png'),full_page=False)
    report=dict(public=public,public_clock_frozen=True,active_first=first,active_next=second,widths=[w1,w2],upstream_css=css,phase_css=phasecss,exact_css_match=True,errors=errors,fixture_source='actual attempt1 rank0 rollout progress events; isolated localhost:9237 only')
    assert not errors,errors
    (OUT/'amu-bars-browser-proof.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2));browser.close()
server.shutdown()
