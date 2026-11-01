"""headless check of the patched page on the fake runs (page/data.json built from test-pose by test_page.sh first).
   python test_browser.py   (pagecheck venv, compute node; serves page/ on 127.0.0.1)"""
import functools, http.server, json, threading, time
from playwright.sync_api import sync_playwright

P = "/data/home/amartyavarma/ppm/aloe-ft-32-samsep/poseduplex/samples/page"
H = functools.partial(http.server.SimpleHTTPRequestHandler, directory=P)
http.server.SimpleHTTPRequestHandler.log_message = lambda *a: None
srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
threading.Thread(target=srv.serve_forever, daemon=True).start()
url = f"http://127.0.0.1:{srv.server_address[1]}/index.html"
ST = """()=>({vids_hidden:document.getElementById('vids').hidden, vA:document.getElementById('vA').getAttribute('src'),
  vA_ready:document.getElementById('vA').readyState, vA_dur:document.getElementById('vA').duration, vB_dur:document.getElementById('vB').duration,
  vA_t:document.getElementById('vA').currentTime, vB_t:document.getElementById('vB').currentTime, aA_t:document.getElementById('aA').currentTime,
  aA_dur:document.getElementById('aA').duration, vA_rate:document.getElementById('vA').playbackRate, aA_paused:document.getElementById('aA').paused,
  vA_paused:document.getElementById('vA').paused, capA:document.getElementById('capA').textContent, params:document.getElementById('params').textContent.slice(-140),
  h264:document.createElement('video').canPlayType('video/mp4; codecs="avc1.64001e"'), vers:[...document.querySelectorAll('#vers button')].map(b=>b.textContent+':'+b.getAttribute('aria-pressed'))})"""
with sync_playwright() as p:
    b = p.chromium.launch(args=["--autoplay-policy=no-user-gesture-required"])
    pg = b.new_page()
    errs = []
    pg.on("console", lambda m: errs.append(f"console.{m.type}: {m.text}") if m.type in ("error", "warning") else None)
    pg.on("pageerror", lambda e: errs.append(f"pageerror: {e}"))
    pg.goto(url)
    time.sleep(4)
    print("loaded B-000 Model:", json.dumps(pg.evaluate(ST)))
    pg.click("#stage", position={"x": 300, "y": 30})                 # seek by clicking the waveform
    time.sleep(1)
    print("after stage click (seek):", json.dumps(pg.evaluate(ST)))
    pg.click("button.spd[data-r='2']")
    pg.click("#play")
    time.sleep(3)
    print("playing 2x 3 s:", json.dumps(pg.evaluate(ST)))
    pg.click("#play")
    time.sleep(0.5)
    print("paused:", json.dumps(pg.evaluate(ST)))
    pg.click("#vers button:nth-child(2)")                              # Real recording
    time.sleep(2)
    print("Real recording:", json.dumps(pg.evaluate(ST)))
    rail = pg.query_selector_all("#rail button, #rail a, #rail [role=button], #rail > *")
    print("rail items", len(rail))
    rail[-1].click()                                                   # B-001 = no POSE_DONE
    time.sleep(2)
    print("B-001 (no pose):", json.dumps(pg.evaluate(ST)))
    pg.click("#play"); time.sleep(1.5); pg.click("#play")
    print("B-001 play/pause:", json.dumps(pg.evaluate(ST)))
    rail[0].click(); time.sleep(2)
    print("back to B-000:", json.dumps(pg.evaluate(ST)))
    pg.screenshot(path="/data/home/amartyavarma/ppm/aloe-ft-32-samsep/poseduplex/samples/test-pose/page.png", full_page=True)
    print("ERRORS:", errs or "none")
    b.close()
srv.shutdown()
