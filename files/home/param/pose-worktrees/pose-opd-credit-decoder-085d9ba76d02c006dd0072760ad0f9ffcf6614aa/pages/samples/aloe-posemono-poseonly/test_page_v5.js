// (eval-set copy: real tile 'Real pose (data)', every tile's video = the rendered rv mp4, no Pose/Real-video toggle, one system-prompt chip = the voice clip)
// loaded; checks every tile of every clip (count + labels = data.json (labels, blob srcs, metric line, transcript words, cut marker), the params line, no JS errors.
// node test_page.js <dir with index.html + data.json>
const {JSDOM, VirtualConsole} = require('jsdom'); const fs = require('fs'); const path = require('path');
const dir = process.argv[2]; const html = fs.readFileSync(path.join(dir, 'index.html'), 'utf8'); const data = fs.readFileSync(path.join(dir, 'data.json'), 'utf8');
const errs = []; const vc = new VirtualConsole(); vc.on('jsdomError', e => errs.push(String(e && (e.stack || e.message || e)))); vc.on('error', e => errs.push('console.error ' + e));
const fetched = [];
const dom = new JSDOM('<!doctype html><html><head></head><body>' + html + '</body></html>', {runScripts: 'dangerously', pretendToBeVisual: true, virtualConsole: vc,
  beforeParse(w) {
    w.fetch = u => { fetched.push(u); return Promise.resolve(String(u).startsWith('data.json') ? {json: () => Promise.resolve(JSON.parse(data))} : {blob: () => Promise.resolve(new w.Blob(['x']))}); };
    let k = 0; w.URL.createObjectURL = () => 'blob:test/' + (k++);
    Object.defineProperty(w.HTMLMediaElement.prototype, 'readyState', {get() { return this._rs || 0; }});
    w.HTMLMediaElement.prototype.load = function () { this._rs = 0; setTimeout(() => { this._rs = 1; this.dispatchEvent(new w.Event('loadedmetadata')); }, 5); };
    w.HTMLMediaElement.prototype.pause = function () {}; w.HTMLMediaElement.prototype.play = function () { return Promise.resolve(); };
    w.HTMLCanvasElement.prototype.getContext = function () { return new Proxy({}, {get: (t, k) => (k === 'measureText' ? () => ({width: 10}) : () => {})}); };
    w.HTMLDialogElement && (w.HTMLDialogElement.prototype.showModal = function () { this.open = true; });
  }});
const w = dom.window, d = w.document;
const sleep = ms => new Promise(r => setTimeout(r, ms));
(async () => {
  await sleep(500);
  const rail = d.querySelectorAll('#rail button'); console.log('clips', rail.length);
  console.log('params:', d.getElementById('params').textContent);
  let bad = 0;
  for (let i = 0; i < rail.length; i++) {
    d.querySelectorAll('#rail button')[i].click(); await sleep(200);
    const tl = [...d.querySelectorAll('#tiles figure.tile')];
    const labs = tl.map(f => f.querySelector('.tl').textContent).join(' | ');
    const want = JSON.parse(data).convs[i].tiles.map(t => t.label).join(' | ');
    const ok = tl.length === JSON.parse(data).convs[i].tiles.length && labs === want && tl.length === 1 && /^first 2 s = your card \d+ clip, then generated$/.test(labs) && d.querySelectorAll('#tiles .tview').length === 0 && JSON.parse(data).convs[i].tiles.every(t => t.src.startsWith('rv/')) && d.querySelectorAll('#sp .chip').length === 1 && tl.every(f => f.querySelector('video').src.startsWith('blob:'))
      && tl.every(f => f.querySelectorAll('.tt .w').length > 5) && tl.every(f => f.querySelectorAll('.tspd button').length === 4);
    if (!ok) bad++;
    console.log(rail[i].textContent, ok ? 'OK' : 'BAD', '|', tl.map(f => f.querySelector('.tm').textContent + ' [' + f.querySelectorAll('.tt .w').length + ' words; ' +
      ((f.querySelector('.tt .cut') || {}).textContent || 'no cut') + ']').join(' || '));
  }
  d.getElementById('mxOpen').click();
  console.log('modal rows:', [...d.querySelectorAll('.mxt tbody tr')].map(r => r.textContent.replace(/\s+/g, ' ').trim()).join(' / '));
  console.log('fetched', fetched.length, 'unique media', new Set(fetched).size);
  console.log(errs.length ? 'JS ERRORS:\n' + errs.join('\n') : 'no JS errors', bad ? `; ${bad} BAD clips` : '; all clips OK');
  process.exit(errs.length || bad ? 1 : 0);
})();
