"""Page-local historical comparisons, using each run's own loss definition."""
import json
from pathlib import Path


def reference_docs(settings):
    from pose_doc import build_doc
    from pose_metrics import enrich_doc

    references = []
    for spec in settings.get('reference_runs', []):
        run = Path(spec['run_dir']).resolve()
        # An unavailable reference must never become a synthetic empty curve.
        if not (run / 'metrics.jsonl').is_file():
            continue
        method = {}
        if (run / 'config.json').is_file():
            method = json.loads((run / 'config.json').read_text()).get('method', {})
        elif (run / 'preparation.json').is_file():
            method = json.loads((run / 'preparation.json').read_text()).get('identity', {}).get('method', {})
        doc = enrich_doc(build_doc(run, {'run': spec.get('run_name', run.name)}), method)
        doc.update(label=spec.get('label', doc['run']), short=spec.get('label', doc['run']),
                   color=spec.get('color', '#b36b00'), color_dark=spec.get('color_dark', '#ffd43b'),
                   dash=spec.get('dash', [6, 3]), branches=[])
        references.append(doc)
    return references


def template_replacements(template):
    old_control = template[template.index('(Array.isArray(s.xwin)?'):template.index("}\nif(VOX())", template.index('(Array.isArray(s.xwin)?'))]
    old_handler = "{const ab=document.getElementById('afull');if(ab)ab.onclick=()=>{AFULL=!AFULL;LS.set('ppm7_afull',AFULL);hdr();for(const k in ch){try{ch[k].resetZoom('none')}catch(e){}ch[k].$z=false;upd(k)}}}"
    return [
        ["let AFULL=!!LS.get('ppm7_afull',false);", (Path(__file__).with_name('comparison.js')).read_text()],
        [old_control, 'poseCompareControl()'],
        [old_handler, 'poseCompareBind();'],
        ["if(XMAX)sx.max=evX(k,XMAX);", "sx.min=0;sx.max=poseCompareEnd();if(XMAX)sx.max=evX(k,XMAX);"],
        ['const x0=DX0!=null?DX0:', 'const x0=PAGE===\'pose_opd_credit\'?sx.min:DX0!=null?DX0:'],
    ]
