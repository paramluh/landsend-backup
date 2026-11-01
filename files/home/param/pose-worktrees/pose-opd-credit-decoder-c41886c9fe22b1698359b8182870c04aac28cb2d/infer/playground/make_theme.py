"""Build genweb/index.theme.html (staged white / product theme) from the live genweb/index.html. Same JS logic; new CSS + markup."""
import re, sys
SRC = "/data/home/amartyavarma/ppm/aloe-posedup-live/genweb/index.html"
DST = "/data/home/amartyavarma/ppm/aloe-posedup-live/genweb/index.theme.html"
s = open(SRC, encoding="utf-8").read()

INFO = '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="9"/><path d="M12 11v5"/><path d="M12 7.5h.01"/></svg>'
CHEV = "url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23667085' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='m6 9 6 6 6-6'/%3E%3C/svg%3E\")"

HEAD = '''<title>Generation Playground</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@600;700&display=swap" rel="stylesheet">
<style>
:root{color-scheme:light;--ink:#101828;--ink2:#344054;--mut:#667085;--mut2:#98a2b3;--line:#e9ebef;--field:#d0d5dd;--fieldh:#b9c0cb;--soft:#f7f8fa;--acc:#2563eb;--acch:#1d4ed8;--acct:#eff4ff;--ring:rgba(37,99,235,.28);
  --bad:#d92d20;--badt:#fef3f2;--badl:#fecdca;--r:12px;--rs:8px;
  --fb:"Inter",system-ui,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;--fh:"Plus Jakarta Sans","Inter",system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
*{box-sizing:border-box}
html,body{margin:0;background:#fff}
body{color:var(--ink);font:400 14px/20px var(--fb);font-variant-numeric:tabular-nums;-webkit-font-smoothing:antialiased}
[hidden]{display:none!important}
.ic{width:16px;height:16px;flex:none;fill:none;stroke:currentColor;stroke-width:1.8;stroke-linecap:round;stroke-linejoin:round}
button{font:500 14px/20px var(--fb);display:inline-flex;align-items:center;justify-content:center;gap:8px;border:1px solid var(--field);background:#fff;color:var(--ink2);border-radius:var(--rs);padding:7px 12px;cursor:pointer;white-space:nowrap;transition:background .12s,border-color .12s,box-shadow .12s,color .12s}
button:hover{background:var(--soft);border-color:var(--fieldh);color:var(--ink)}
button:focus-visible,a:focus-visible,.info:focus-visible,.run:focus-visible{outline:none;box-shadow:0 0 0 3px var(--ring)}
button:disabled,button:disabled:hover{opacity:.45;cursor:not-allowed}
button.pri{background:var(--acc);border-color:var(--acc);color:#fff;font-weight:600;font-size:15px;padding:9px 20px;box-shadow:0 1px 2px rgba(16,24,40,.08)}
button.pri:hover{background:var(--acch);border-color:var(--acch);color:#fff}
button.pri:disabled,button.pri:disabled:hover{background:var(--acc);border-color:var(--acc)}
button.pri .ic{width:18px;height:18px}
a.lnk{display:inline-flex;align-items:center;gap:6px;color:var(--ink2);text-decoration:none;font-weight:500;padding:7px 8px;border-radius:var(--rs)}
a.lnk:hover{color:var(--acc);background:var(--soft)}
input[type=number],input[type=text],select,textarea{font:400 14px/20px var(--fb);font-variant-numeric:tabular-nums;color:var(--ink);background:#fff;border:1px solid var(--field);border-radius:var(--rs);padding:7px 10px;width:100%;min-width:0;transition:border-color .12s,box-shadow .12s}
select{width:auto;max-width:100%;padding-right:32px;-webkit-appearance:none;appearance:none;background:#fff ''' + CHEV + ''' no-repeat right 10px center/14px 14px;cursor:pointer}
input[type=number]:hover,input[type=text]:hover,select:hover,textarea:hover{border-color:var(--fieldh)}
input[type=number]:focus,input[type=text]:focus,select:focus,textarea:focus{outline:none;border-color:var(--acc);box-shadow:0 0 0 3px var(--ring)}
input:disabled,select:disabled,textarea:disabled{background:var(--soft);color:var(--mut2);cursor:not-allowed}
textarea{min-height:72px;resize:vertical;display:block}
input[type=range]{accent-color:var(--acc);cursor:pointer}
input[type=checkbox]{accent-color:var(--acc);width:16px;height:16px;margin:0;cursor:pointer}
input[type=range]:focus-visible,input[type=checkbox]:focus-visible{outline:2px solid var(--ring);outline-offset:2px}
input[type=file]{font:400 14px/20px var(--fb);max-width:100%;color:var(--mut)}
input[type=file]::file-selector-button{font:500 14px/20px var(--fb);border:1px solid var(--field);background:#fff;color:var(--ink2);border-radius:var(--rs);padding:6px 12px;margin-right:12px;cursor:pointer}
input[type=file]::file-selector-button:hover{background:var(--soft)}
.note{color:var(--mut);font-size:12px;line-height:16px}
.lbl{font:600 12px/16px var(--fb);letter-spacing:.06em;text-transform:uppercase;color:var(--mut)}
.info{display:inline-flex;margin-left:6px;color:var(--mut2);cursor:help;flex:none;border-radius:50%;text-transform:none;letter-spacing:0}
.info svg{width:14px;height:14px;fill:none;stroke:currentColor;stroke-width:2;stroke-linecap:round}
.info:hover,.info:focus-visible{color:var(--acc)}
.spin{display:inline-block;width:16px;height:16px;border-radius:50%;border:2px solid currentColor;border-right-color:transparent;animation:spin .8s linear infinite;flex:none}
.spin.s{width:12px;height:12px;border-width:2px}
@keyframes spin{to{transform:rotate(360deg)}}
@keyframes pulse{50%{opacity:.35}}

/* header: product name + model / checkpoint chips */
header{display:flex;align-items:center;gap:8px 16px;flex-wrap:wrap;min-height:64px;padding:12px 24px;border-bottom:1px solid var(--line)}
.brand{display:flex;align-items:center;gap:12px;min-width:0}
.logo{width:32px;height:32px;border-radius:10px;background:var(--acc);color:#fff;display:grid;place-items:center;flex:none}
.logo .ic{width:18px;height:18px;stroke-width:2}
header h1{font:700 20px/28px var(--fh);letter-spacing:-.01em;margin:0;white-space:nowrap}
.chips{display:flex;gap:8px;flex-wrap:wrap;min-width:0}
.chip{display:inline-flex;align-items:center;gap:6px;height:24px;padding:0 10px;border:1px solid var(--line);border-radius:999px;background:var(--soft);color:var(--ink2);font-size:12px;line-height:16px;font-weight:500;white-space:nowrap}
.chip.k{color:var(--mut)}
header .q{margin-left:auto;color:var(--acc);background:var(--acct);border-color:transparent}

/* page grid: left = video panels + Generate, right = Output / Settings / History */
main{--fw:clamp(140px,min(calc((100vh - 248px) * .5333),calc((100vw - 720px) / 2)),320px);display:grid;grid-template-columns:calc(2 * var(--fw) + 64px) minmax(0,1fr)}
.left{padding:24px;border-right:1px solid var(--line);min-width:0}
.stage{display:flex;gap:16px}
.media{margin:0;width:var(--fw);flex:none;min-width:0}
.frame{position:relative;width:100%;aspect-ratio:8/15;border-radius:var(--r);overflow:hidden;background:linear-gradient(180deg,#f7f9fc 0%,#eef2f8 100%);box-shadow:0 1px 2px rgba(16,24,40,.05),0 12px 24px -12px rgba(16,24,40,.22)}
.frame video,.frame iframe{position:absolute;inset:0;width:100%;height:100%;border:0;display:block;background:#000;object-fit:contain}
.ph{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:4px;padding:24px 16px;text-align:center}
.pi{width:48px;height:48px;border-radius:50%;background:#fff;display:grid;place-items:center;color:var(--mut);box-shadow:0 0 0 1px var(--line),0 1px 2px rgba(16,24,40,.05);margin-bottom:8px}
.pi .ic{width:22px;height:22px}.pi .spin{width:20px;height:20px}
.pt{font:600 14px/20px var(--fh);color:var(--ink2)}
.ps{font-size:12px;line-height:16px;color:var(--mut);max-width:220px;overflow-wrap:anywhere}
.ph.run .pi{color:var(--acc)}.ph.bad .pi{color:var(--bad)}
.flash{position:absolute;inset:0;background:#fff;opacity:0;pointer-events:none;z-index:2}
figcaption{display:flex;align-items:center;gap:6px;flex-wrap:wrap;font-size:12px;line-height:16px;color:var(--mut);margin-top:8px;overflow-wrap:anywhere}
figcaption b{color:var(--ink2);font-weight:600}
.live{display:inline-flex;align-items:center;gap:4px;font-weight:700;font-size:11px;letter-spacing:.06em;color:var(--acc)}
.live::before{content:"";width:6px;height:6px;border-radius:50%;background:var(--acc);animation:pulse 1.2s ease-in-out infinite}
.genbar{display:flex;align-items:flex-end;gap:16px;flex-wrap:wrap;margin-top:24px}
.genknobs{display:flex;gap:16px;align-items:flex-end}
.fld{display:flex;flex-direction:column;gap:4px}
.fld>span{font-size:12px;line-height:16px;font-weight:500;color:var(--mut)}
.fld input{width:104px;padding:9px 10px}.fld[data-k=n_samples] input{width:72px}
.fld.changed>span::after{content:"";display:inline-block;width:6px;height:6px;margin-left:6px;border-radius:50%;background:var(--acc);vertical-align:1px}
.gomodel{display:flex;align-items:center;margin-top:12px;font-size:12px;line-height:16px;color:var(--mut)}
.gomodel b{color:var(--ink2);font-weight:600}

/* right column */
.right{display:flex;flex-direction:column;min-width:0}
.mtabs{display:flex;gap:24px;padding:0 24px;border-bottom:1px solid var(--line);flex:none}
.mtabs button{border:0;border-radius:0;background:none;box-shadow:none;padding:0;height:56px;color:var(--mut);font-weight:500;position:relative}
.mtabs button::after{content:"";position:absolute;left:0;right:0;bottom:-1px;height:2px;border-radius:2px 2px 0 0;background:transparent}
.mtabs button:hover{background:none;color:var(--ink)}
.mtabs button.on{color:var(--ink);font-weight:600}.mtabs button.on::after{background:var(--acc)}
.mtabs button:focus-visible{box-shadow:none;outline:2px solid var(--ring);outline-offset:-10px;border-radius:10px}
.mtabs .cnt{color:var(--mut2);font-weight:500}
.mtabs button.flag::before{content:"";position:absolute;top:16px;right:-8px;width:6px;height:6px;border-radius:50%;background:var(--acc)}
.mpanel{padding:24px;min-width:0}

/* Output */
.empty{display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;gap:4px;padding:64px 24px;color:var(--mut)}
.empty .pi{width:56px;height:56px;background:var(--soft)}
.empty .pt{font:600 16px/24px var(--fh);color:var(--ink)}
.empty .ps{font-size:14px;line-height:20px;max-width:360px}
.status{display:flex;align-items:center;gap:12px;padding:12px 16px;border:1px solid var(--line);border-radius:var(--r);background:#fff}
.sico{width:32px;height:32px;border-radius:50%;display:grid;place-items:center;flex:none;background:var(--soft);color:var(--mut)}
.stx{display:flex;flex-direction:column;min-width:0;flex:1}
.state{font-weight:600;color:var(--ink);white-space:nowrap}
.smsg{font-size:12px;line-height:16px;color:var(--mut);overflow-wrap:anywhere}
.status.s-run .sico{background:var(--acct);color:var(--acc)}
.status.s-ok .sico{color:var(--ink2)}
.status.s-bad{border-color:var(--badl);background:var(--badt)}.status.s-bad .sico{background:#fee4e2;color:var(--bad)}.status.s-bad .state{color:var(--bad)}
.prog{height:4px;background:var(--line);border-radius:2px;overflow:hidden;margin-top:8px}.prog>div{height:100%;background:var(--acc);width:0;transition:width .4s}
.stats{display:flex;flex-wrap:wrap;gap:8px 24px;margin-top:24px}
.stats:empty{display:none}
.stat{color:var(--mut);overflow-wrap:anywhere;max-width:100%}
.stat b{font:600 20px/28px var(--fb);font-variant-numeric:tabular-nums;color:var(--ink);margin-right:2px}
.meta{display:flex;flex-wrap:wrap;gap:4px 24px;margin-top:16px;font-size:12px;line-height:16px;color:var(--mut)}
.meta span{overflow-wrap:anywhere;min-width:0}.meta b{color:var(--ink2);font-weight:500}
.actions{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-top:16px}
.blk{margin-top:32px}
.blk h3{display:flex;align-items:center;margin:0 0 8px;font:600 12px/16px var(--fb);letter-spacing:.06em;text-transform:uppercase;color:var(--mut)}
.words{position:relative;max-height:192px;overflow:auto;line-height:28px;padding:8px 16px;background:var(--soft);border:1px solid var(--line);border-radius:var(--r);overflow-wrap:anywhere}
.words span[data-i]{padding:2px 3px;border-radius:4px;transition:background .1s,color .1s}.words span.pf{color:var(--mut2)}.words span.on{background:var(--acc);color:#fff}
.lane{display:grid;grid-template-columns:24px minmax(0,1fr);gap:8px}.lane+.lane{border-top:1px solid var(--line);margin-top:8px;padding-top:8px}
.lane>b{font:600 12px/28px var(--fb);color:var(--mut)}
audio{width:100%;display:block;height:40px}
.kv{color:var(--mut);overflow-wrap:anywhere}.kv b{color:var(--ink2);font-weight:600}

/* Settings */
.setbar{display:flex;align-items:center;gap:8px;flex-wrap:wrap}
.setbar .grow{flex:1}
.mdlpick{display:flex;align-items:center;gap:12px;min-width:0}
.mdlpick select{font-weight:600}
.copied{font-size:12px;color:var(--mut)}
.linknote{margin:16px 0 0;padding:12px 16px;border:1px solid var(--line);border-radius:var(--r);background:var(--soft);color:var(--ink2);font-size:12px;line-height:16px}
.summary{margin:16px 0 0;font-size:12px;line-height:16px;color:var(--mut);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;cursor:pointer}
.summary.open{white-space:normal;overflow-wrap:anywhere}.summary b{color:var(--ink2);font-weight:600}
#form{min-width:0}
.tabs{display:flex;flex-wrap:wrap;gap:4px;padding:4px;margin-top:16px;background:var(--soft);border:1px solid var(--line);border-radius:10px;align-self:flex-start;max-width:100%}
.tabs button{border:0;background:none;box-shadow:none;padding:6px 12px;border-radius:7px;color:var(--mut);font-weight:500}
.tabs button:hover{color:var(--ink);background:rgba(255,255,255,.7)}
.tabs button.on{background:#fff;color:var(--acc);font-weight:600;box-shadow:0 0 0 1px var(--line)}
.tabs button:focus-visible{box-shadow:0 0 0 3px var(--ring)}
.tabbody{min-width:0;margin-top:8px}
.tabp{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));column-gap:32px;align-content:start}
.tabp>.sub,.tabp>.pair,.tabp>.row.wide{grid-column:1/-1}
.sub{font:600 12px/16px var(--fb);letter-spacing:.06em;text-transform:uppercase;color:var(--mut);margin:24px 0 0;padding-bottom:4px}
.row{display:grid;grid-template-columns:minmax(0,1fr) minmax(120px,44%);gap:8px 16px;align-items:center;padding:8px 0;border-bottom:1px solid var(--line);min-width:0}
.row>.lab{display:flex;align-items:center;min-width:0;color:var(--ink2)}
.row>.lab label{overflow-wrap:anywhere;min-width:0}
.row.stk,.row.wide{grid-template-columns:minmax(0,1fr)}
.row.changed>.lab label{font-weight:600;color:var(--ink)}
.row.changed>.lab label::after{content:"";display:inline-block;width:6px;height:6px;margin-left:6px;border-radius:50%;background:var(--acc);vertical-align:2px}
.row input[type=checkbox]{justify-self:start}
.num{display:flex;gap:12px;align-items:center;min-width:0}.num input[type=range]{flex:1;min-width:0;margin:0}.num input[type=number]{width:80px;flex:none;padding:7px 8px}
.rowstat{font-size:12px;line-height:16px;color:var(--mut);overflow-wrap:anywhere}
details.opt>summary{cursor:pointer;color:var(--ink2);list-style:none;display:flex;align-items:center;gap:6px;border-radius:6px}
details.opt>summary::-webkit-details-marker{display:none}
details.opt>summary::before{content:"";width:6px;height:6px;border-right:1.6px solid var(--mut);border-bottom:1.6px solid var(--mut);transform:rotate(-45deg);margin:0 4px 0 2px;transition:transform .12s}
details.opt[open]>summary::before{transform:rotate(45deg)}
details.opt[open]>summary{margin-bottom:8px}
.pair{display:grid;grid-template-columns:minmax(80px,112px) minmax(0,1fr) minmax(0,1fr);gap:8px 16px;align-items:center;padding:16px 0;border-bottom:1px solid var(--line)}
.pair .hd{color:var(--ink2);display:flex;align-items:center;min-width:0}
.pair .hd.ch{font:600 12px/16px var(--fb);letter-spacing:.06em;text-transform:uppercase;color:var(--mut)}
.pair .row{grid-template-columns:minmax(0,1fr);padding:0;border:0}
.pair .row>.lab{display:none}

/* History */
#phist{padding:16px 12px}
.runs{display:flex;flex-direction:column}
.run{display:grid;grid-template-columns:40px minmax(0,1fr) auto;gap:16px;align-items:center;padding:8px 12px;border-radius:10px;cursor:pointer;position:relative;transition:background .12s}
.run+.run::before{content:"";position:absolute;left:12px;right:12px;top:0;height:1px;background:var(--line)}
.run:hover{background:var(--soft)}
.run.sel{background:var(--acct)}.run.sel::before,.run.sel+.run::before,.run:hover::before,.run:hover+.run::before{opacity:0}
.run.sel .thumb{box-shadow:0 0 0 2px var(--acc)}
.thumb{width:40px;aspect-ratio:8/15;border-radius:6px;overflow:hidden;background:linear-gradient(180deg,#f7f9fc,#eef2f8);position:relative;box-shadow:0 0 0 1px var(--line)}
.thumb video{position:absolute;inset:0;width:100%;height:100%;object-fit:cover;display:block;pointer-events:none}
.rb{min-width:0}
.run .hl{display:flex;gap:8px;align-items:baseline;color:var(--ink)}
.run .hx{min-width:0;overflow-wrap:anywhere;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}
.run .p{font-size:12px;line-height:16px;color:var(--mut);margin-top:2px;padding-left:16px;overflow-wrap:anywhere;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}
.run .t{font-size:12px;line-height:16px;color:var(--mut2);white-space:nowrap;align-self:start;margin-top:2px}
.dot{flex:none;width:8px;height:8px;border-radius:50%;background:#c4c9d2;transform:translateY(-1px)}
.dot.done{background:var(--mut2)}.dot.running,.dot.claimed,.dot.queued{background:var(--acc);animation:pulse 1.2s ease-in-out infinite}.dot.error{background:var(--bad)}

/* laptop / desktop: the page fits the window; each column scrolls inside itself */
@media (min-width:900px){
  body{height:100vh;height:100dvh;display:flex;flex-direction:column}
  header{flex:none}
  main{flex:1;min-height:0;overflow:hidden}
  .left{overflow:auto}
  .right{min-height:0;overflow:hidden}
  .mpanel{flex:1;min-height:0;overflow:auto}
  #pset:not([hidden]){display:flex;flex-direction:column;overflow:hidden;padding-bottom:0}
  #form{flex:1;min-height:0;display:flex;flex-direction:column}
  .tabs{flex:none}
  .tabbody{flex:1;min-height:0;overflow:auto;padding-bottom:24px}
}
/* phone / narrow: one column, the page scrolls */
@media (max-width:899px){
  header{padding:12px 16px}
  main{grid-template-columns:minmax(0,1fr);--fw:min(calc((100vw - 48px) / 2),260px)}
  .left{padding:16px;border-right:0;border-bottom:1px solid var(--line)}
  .stage,.genbar,.gomodel{justify-content:center}
  .mtabs{padding:0 16px}.mpanel{padding:16px}#phist{padding:8px 4px}
  .tabp{grid-template-columns:minmax(0,1fr)}
  .tabs{align-self:stretch}
}
</style></head><body>
<svg width="0" height="0" style="position:absolute" aria-hidden="true"><defs>
<symbol id="i-gen" viewBox="0 0 24 24"><path d="M12 3l1.9 5.6L19.5 10.5l-5.6 1.9L12 18l-1.9-5.6L4.5 10.5l5.6-1.9z"/><path d="M19 15.5l.7 1.8 1.8.7-1.8.7L19 20.5l-.7-1.8-1.8-.7 1.8-.7z"/></symbol>
<symbol id="i-link" viewBox="0 0 24 24"><path d="M10 13a5 5 0 0 0 7.5.5l3-3a5 5 0 0 0-7-7l-1.7 1.7"/><path d="M14 11a5 5 0 0 0-7.5-.5l-3 3a5 5 0 0 0 7 7l1.7-1.7"/></symbol>
<symbol id="i-sliders" viewBox="0 0 24 24"><path d="M4 6h9M17 6h3M4 12h3M11 12h9M4 18h11M19 18h1"/><circle cx="15" cy="6" r="2"/><circle cx="9" cy="12" r="2"/><circle cx="17" cy="18" r="2"/></symbol>
<symbol id="i-history" viewBox="0 0 24 24"><path d="M3 12a9 9 0 1 0 3-6.7L3 8"/><path d="M3 3v5h5"/><path d="M12 7v5l3 2"/></symbol>
<symbol id="i-play" viewBox="0 0 24 24"><rect x="3" y="3" width="18" height="18" rx="4"/><path d="M10 8.5v7l6-3.5z"/></symbol>
<symbol id="i-reset" viewBox="0 0 24 24"><path d="M3 12a9 9 0 1 0 3-6.7L3 8"/><path d="M3 3v5h5"/></symbol>
<symbol id="i-check" viewBox="0 0 24 24"><path d="M20 6 9 17l-5-5"/></symbol>
<symbol id="i-alert" viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="M12 7.5v5"/><path d="M12 16.5h.01"/></symbol>
<symbol id="i-clock" viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/></symbol>
<symbol id="i-user" viewBox="0 0 24 24"><circle cx="12" cy="8" r="4"/><path d="M4 21a8 8 0 0 1 16 0"/></symbol>
<symbol id="i-video" viewBox="0 0 24 24"><path d="m16 13 5.2 3.1a.5.5 0 0 0 .8-.4V8.3a.5.5 0 0 0-.8-.4L16 11"/><rect x="2" y="6" width="14" height="12" rx="2"/></symbol>
<symbol id="i-ext" viewBox="0 0 24 24"><path d="M15 3h6v6"/><path d="M10 14 21 3"/><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/></symbol>
</defs></svg>
<header>
 <div class="brand"><span class="logo"><svg class="ic"><use href="#i-gen"/></svg></span><h1>Generation playground</h1></div>
 <div class="chips"><span class="chip" id="mchip">Monoplex</span><span class="chip k" id="model">loading…</span></div>
 <span class="chip q" id="queue" hidden><span class="spin s"></span><span id="queuet"></span></span>
</header>
<main>
<section class="left" id="left">
 <div class="stage">
  <figure class="media" id="dav"><div class="frame"><div class="ph"><span class="pi"><svg class="ic"><use href="#i-user"/></svg></span><span class="pt">No run selected</span><span class="ps">Press Generate, or pick a run in History</span></div></div><figcaption><b>Generated</b></figcaption></figure>
  <figure class="media" id="dgt"><div class="frame"><div class="ph"><span class="pi"><svg class="ic"><use href="#i-video"/></svg></span><span class="pt">Ground truth</span><span class="ps">The actor's real clip appears with a run</span></div></div><figcaption><b>Ground truth</b></figcaption></figure>
 </div>
 <div class="genbar"><button class="pri" id="go"><svg class="ic"><use href="#i-gen"/></svg><span>Generate</span></button><div class="genknobs" id="genknobs"></div></div>
 <div class="gomodel"><b id="gomodel">Monoplex</b><span class="info" tabindex="0" title="Each press queues new runs (one per sample, seeds seed, seed+1, …); they run in parallel. Seed 0 = a random seed per sample. Pick the model in Settings.">INFO</span></div>
</section>
<section class="right" id="right">
 <nav class="mtabs" role="tablist"><button type="button" role="tab" data-m="out" id="mt-out"><svg class="ic"><use href="#i-play"/></svg>Output</button><button type="button" role="tab" data-m="set" id="mt-set"><svg class="ic"><use href="#i-sliders"/></svg>Settings</button><button type="button" role="tab" data-m="hist" id="mt-hist"><svg class="ic"><use href="#i-history"/></svg>History<span class="cnt" id="nruns" hidden></span></button></nav>
 <div class="mpanel" data-m="out" id="pout"><div id="detail"><div class="empty"><span class="pi"><svg class="ic"><use href="#i-play"/></svg></span><span class="pt">Nothing to show yet</span><span class="ps">Press <b>Generate</b>, or pick a run in <b>History</b> to replay it.</span></div></div></div>
 <div class="mpanel" data-m="set" id="pset" hidden>
  <div class="setbar">
   <label class="mdlpick"><span class="lbl">Model</span><select id="mdl" title="model"><option value="monoplex" title="B alone; actor-clip or conversation-start prefill">Monoplex (B alone)</option><option value="duplex" disabled title="hears A; conversation-start prefill">Duplex (starting…)</option></select><span class="info" tabindex="0" title="Monoplex: B alone (actor clip or conversation-start prefill). Duplex: B hears A (conversation-start prefill; the live demo's engine). Each model has its own settings; the pick is kept in the link.">INFO</span></label>
   <span class="grow"></span><span class="copied" id="copied"></span>
   <button type="button" class="sm" id="reset"><svg class="ic"><use href="#i-reset"/></svg>Reset to defaults</button><button type="button" class="sm" id="copy"><svg class="ic"><use href="#i-link"/></svg>Copy link</button>
  </div>
  <p class="linknote" id="linknote" hidden></p>
  <p class="summary" id="summary" title="click to show all"></p>
  <div id="form"><p class="note">Waiting for the worker's knob list…</p></div>
  <div id="uplhold" hidden><div class="row wide" id="uplrow"><div class="lab"><label for="upl">Upload a voice clip</label><span class="info" tabindex="0" title="3-10 s of it are used; normalised to -23 LUFS, then Mimi-encoded. The upload becomes the selected voice clip.">INFO</span></div><input type="file" id="upl" accept="audio/*"><div class="rowstat" id="uplnote"></div></div></div>
 </div>
 <div class="mpanel" data-m="hist" id="phist" hidden><div class="empty" id="runsempty" hidden><span class="pi"><svg class="ic"><use href="#i-history"/></svg></span><span class="pt">No runs yet</span><span class="ps">Runs you generate appear here.</span></div><div class="runs" id="runs"></div></div>
</section>
</main>
'''.replace("INFO", INFO)

i0 = s.index("<title>"); i1 = s.index("<script>")
s = s[:i0] + HEAD + s[i1:]

def rep(a, b):
    global s
    n = s.count(a)
    if n != 1:
        sys.exit(f"patch matched {n}x: {a[:90]!r}")
    s = s.replace(a, b)

# icon helpers
rep("const $=id=>document.getElementById(id);",
    "const $=id=>document.getElementById(id);\nconst INFO_SVG='" + INFO + "';\nfunction ic(n){ return '<svg class=\"ic\" aria-hidden=\"true\"><use href=\"#i-'+n+'\"/></svg>'; }\n"
    "function phHTML(icon,title,sub){ return '<span class=\"pi\">'+(icon==='spin'?'<span class=\"spin\"></span>':ic(icon))+'</span><span class=\"pt\">'+esc(title)+'</span>'+(sub?'<span class=\"ps\">'+esc(sub)+'</span>':''); }\n"
    "const STATE_ICON={wait:ic('clock'),run:'<span class=\"spin\"></span>',ok:ic('check'),bad:ic('alert')};")
rep("i.className='info'; i.tabIndex=0; i.textContent='i'; i.title=text; return i; }", "i.className='info'; i.tabIndex=0; i.innerHTML=INFO_SVG; i.title=text; return i; }")
rep("const h=document.createElement('span'); h.className='hd'; h.textContent=t;", "const h=document.createElement('span'); h.className='hd ch'; h.textContent=t;")
# header chips + queue chip
rep("function updModelUi(){ setTxt($('model'),modelLabel(MODEL)); setTxt($('gomodel'),MODEL==='duplex'?'Duplex (hears A)':'Monoplex (B alone)'); }",
    "function updModelUi(){ const ck=ckOf(MODEL); setTxt($('mchip'),MODEL==='duplex'?'Duplex':'Monoplex'); setTxt($('model'),ck?'step '+ck:'loading…'); $('model').title=modelLabel(MODEL); setTxt($('gomodel'),MODEL==='duplex'?'Duplex (hears A)':'Monoplex (B alone)'); }")
rep("setTxt($('queue'),d.pending?(d.pending+' run'+(d.pending>1?'s':'')+' queued / running'):'');",
    "$('queue').hidden=!d.pending; setTxt($('queuet'),d.pending?(d.pending+' run'+(d.pending>1?'s':'')+' in progress'):'');")
# Output: designed status card
rep("""  d.innerHTML='<div class="status"><span class="pill" id="dpill"></span><span class="smsg" id="dmsg"></span></div><div class="prog" id="dprog" hidden><div></div></div>'""",
    """  d.innerHTML='<div class="status" id="dstat"><span class="sico" id="dsico"></span><div class="stx"><span class="state" id="dpill"></span><span class="smsg" id="dmsg"></span><div class="prog" id="dprog" hidden><div></div></div></div></div>'""")
rep("""<span class="info" tabindex="0" title="B\\'s voice alone (the same audio is inside the avatar video)">i</span>""",
    """<span class="info" tabindex="0" title="B\\'s voice alone (the same audio is inside the avatar video)">'+INFO_SVG+'</span>""")
rep("const pl=$('dpill'); if(pl.className!=='pill p-'+S.k) pl.className='pill p-'+S.k; setTxt(pl,S.pill); setTxt($('dmsg'),S.msg);",
    "const sb=$('dstat'); if(sb.dataset.k!==S.k){ sb.dataset.k=S.k; sb.className='status s-'+S.k; $('dsico').innerHTML=STATE_ICON[S.k]||''; } setTxt($('dpill'),S.pill); setTxt($('dmsg'),S.msg);")
# left media: captions + designed placeholders
rep("""<figcaption title="generated avatar with B\\'s voice"><b>Generated</b> · B\\'s voice'+""", """<figcaption title="generated avatar with B\\'s voice"><b>Generated</b> B\\'s voice'+""")
rep("""<figcaption title="the avatar + B\\'s voice as the model generates; the saved replay video (with audio) replaces this when it is ready"><b>LIVE</b> · generating</figcaption>';""",
    """<figcaption title="the avatar + B\\'s voice as the model generates; the saved replay video (with audio) replaces this when it is ready"><span class="live">LIVE</span> generating</figcaption>';""")
rep("""    else fa.innerHTML='<div class="frame"><div class="ph"></div></div><figcaption><b>Generated</b></figcaption>'; }""",
    """    else fa.innerHTML='<div class="frame"><div class="ph"></div></div><figcaption><b>Generated</b></figcaption>'; }""")
rep("const ph=$('dav').querySelector('.ph'); if(ph) setTxt(ph,S.pill+(S.k==='bad'?'':'…'));",
    "const ph=$('dav').querySelector('.ph'); if(ph){ const pk=S.k+'|'+S.pill; if(ph.dataset.k!==pk){ ph.dataset.k=pk; ph.className='ph '+S.k; ph.innerHTML=phHTML(S.k==='bad'?'alert':S.k==='wait'?'clock':'spin',S.k==='bad'?'Generation failed':S.pill,S.k==='bad'?'details in Output':S.msg); } }")
rep("""'<div class="ph">'+(r.person?'No ground-truth clip rendered for '+esc(r.person):'No ground-truth clip for this run')+'</div>'""",
    """'<div class="ph">'+phHTML('video','No ground-truth clip',r.person?'none rendered for '+r.person:'this run has no actor clip')+'</div>'""")
rep("""'"><b>Ground truth</b>'+(r.person?' · '+esc(r.person):'')+'</figcaption>'; }""", """'"><b>Ground truth</b>'+(r.person?' '+esc(r.person):'')+'</figcaption>'; }""")
rep("c.innerHTML='<b>LIVE</b> · first frame '+window.__ttff.toFixed(1)+' s after Generate';", "c.innerHTML='<span class=\"live\">LIVE</span> first frame '+window.__ttff.toFixed(1)+' s after Generate';")
# actions with icons
rep("""$('dact').innerHTML='<button type="button" class="sm" id="exactlink">Copy exact link</button><button type="button" class="sm" id="reuse">Load these settings</button>'+(rnd.sxs_src?'<a href="'+esc(rnd.sxs_src)+'" target="_blank" rel="noopener">Video with skeleton ↗</a>':'');""",
    """$('dact').innerHTML='<button type="button" class="sm" id="exactlink">'+ic('link')+'<span>Copy exact link</span></button><button type="button" class="sm" id="reuse">'+ic('sliders')+'<span>Load these settings</span></button>'+(rnd.sxs_src?'<a class="lnk" href="'+esc(rnd.sxs_src)+'" target="_blank" rel="noopener"><span>Video with skeleton</span>'+ic('ext')+'</a>':'');""")
rep("el_.textContent='Copied (seed '+seed+')'; setTimeout(()=>{ el_.textContent='Copy exact link'; },2000);",
    "el_.lastChild.textContent='Copied (seed '+seed+')'; setTimeout(()=>{ el_.lastChild.textContent='Copy exact link'; },2000);")
open(DST, "w", encoding="utf-8").write(s)
print("wrote", DST, len(s))
