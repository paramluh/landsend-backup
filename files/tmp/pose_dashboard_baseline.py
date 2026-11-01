from pathlib import Path
import shutil
root=Path('/home/param/websites/pose-opd-credit-dashboard')
base=Path('/tmp/pose-dashboard-logging-baseline')
files=['pose_metrics.py','pose_doc.py','pose-preview/make_demo.py','pose-preview/chart_layout.js','pose-preview/page_notes.js','TELEMETRY_CONTRACT.txt','pose-preview/README.txt','tests/test_pose_metrics.py','tests/test_pose_preview.py']
for name in files:
 p=base/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(root/name,p)
def change(name, old, new):
 p=base/name;s=p.read_text()
 if old not in s:raise RuntimeError((name,old[:100]))
 p.write_text(s.replace(old,new))
def delete_between(name,start,end):
 p=base/name;s=p.read_text();a=s.index(start);b=s.index(end,a);p.write_text(s[:a]+s[b:])
change('pose_doc.py','MORE_PREFIXES = ("loss/", "credit/", "policy/", "kl/", "grad_norm/",\n                 "logit_grad_norm/", "logit_grad_alignment/", "logit_grad_credit_direct_ratio/")','MORE_PREFIXES = ("loss/", "credit/", "policy/", "kl/", "grad_norm/")')
delete_between('pose_metrics.py','    pose_codebooks = method.get("pose_codebook_weights")','    reduction = method.get("reduction")')
change('pose_metrics.py','    declare("kl/lexical_contribution", "Word-mass weighted lexical KL", "Student WORD probability times conditional lexical KL, multiplied before the same frame/row reduction as full text KL. Full text KL equals category KL plus this contribution.")\n    install("kl/lexical_contribution")\n','')
change('pose_metrics.py','media_codebooks[stream][cb]','codebooks[cb]')
change('pose_metrics.py','    for component, label in (("pose_head", "Optional pose head"), ("other", "Other trainable parameters")):\n        if any(f"grad_norm/{term}/{component}" in original_columns for term in ("combined", "direct", "credit")):\n            components.append((component, label))','    if any(key.startswith(("grad_norm/direct/pose_head", "grad_norm/credit/pose_head")) for key in original_columns):\n        components.append(("pose_head", "Optional pose head"))')
change('pose_metrics.py','for term in ("combined", "direct", "credit")','for term in ("direct", "credit")')
change('pose_metrics.py','                reason = ("Combined parameter gradients are not present in this log." if term == "combined" else\n                          "Separate per-loss parameter gradients are not measured by the current trainer; logit-gradient diagnostics are distinct quantities.")\n                missing.append({"metric": key, "reason": reason, "producer": "Trainer backward instrumentation before clipping and optimizer.step"})','                missing.append({"metric": key, "reason": "Separate per-term component gradients are not present in the supplied log.", "producer": "Trainer backward instrumentation before clipping and optimizer.step"})')
delete_between('pose_metrics.py','    logit_groups, logit_diagnostics = [], []\n','    for key in CRITICAL_METRICS:')
change('pose_metrics.py','        logit_grad_groups=logit_groups,\n        logit_grad_diagnostics=logit_diagnostics,\n','')
change('pose-preview/make_demo.py'," ['kl/lexical_contribution','Text KL · word-mass weighted lexical contribution'],\n",'')
change('pose-preview/make_demo.py',"               critical_charts=CRITICAL,\n               grad_components=[[group['component'],group['label']] for group in doc['grad_groups']],\n               extra_charts=EXTRAS)","               critical_charts=CRITICAL,grad_components=COMPONENTS,extra_charts=EXTRAS)")
change('pose-preview/make_demo.py',"    doc.update(critical_charts=CRITICAL,extra_charts=EXTRAS)\n    doc.setdefault('grad_components',COMPONENTS)","    doc.update(critical_charts=CRITICAL,grad_components=COMPONENTS,extra_charts=EXTRAS)")
change('pose-preview/chart_layout.js',"[['combined','Combined parameter gradient',css('--one')],['direct','Direct KL parameter gradient',css('--dm')],['credit','Credit parameter gradient',css('--muted')]]","[['direct','Direct KL',css('--one')],['credit','Credit assignment',css('--dm')]]")
delete_between('pose-preview/chart_layout.js','  for(const group of s.logit_grad_groups||[])','  const extras=')
change('pose-preview/page_notes.js',"gradnorms:'Parameter charts show the exact combined gradient for each component before clipping. Separate direct/credit parameter gradients remain unavailable unless explicitly measured. The categorical-logit charts below separate direct KL, credit from each KL modality, and credit on each action modality; these are logit derivatives, not parameter gradients. Norms do not add.',","gradnorms:'Direct KL and credit-assignment gradients, shown separately for each trainable component. Global L2 norms before clipping. The combined norm is measured independently; the two term norms do not add.',")
change('pose-preview/page_notes.js',"[['grad_norm/combined/'+component,'Combined parameters'],['grad_norm/direct/'+component,'Direct KL parameters'],['grad_norm/credit/'+component,'Credit parameters']]","[['grad_norm/direct/'+component,'Direct KL'],['grad_norm/credit/'+component,'Credit assignment']]")
change('pose-preview/page_notes.js',"    for(const group of doc.logit_grad_groups||[])entries.push([group.id,group.series.map(({key,label})=>[key,label])]);\n    for(const [key] of doc.logit_grad_diagnostics||[])entries.push([id(key),[[key,'']]]);\n",'')
change('pose-preview/page_notes.js','Separate direct/credit parameter gradients are not measured by this trainer. Logit-gradient diagnostics below are different quantities.','Separate term gradients have not been recorded yet.')
delete_between('pose-preview/page_notes.js','  for(const group of S().logit_grad_groups||[]){','  measurements();')
change('tests/test_pose_metrics.py','{"combined": False, "direct": True, "credit": True}','{"direct": True, "credit": True}')
change('tests/test_pose_metrics.py','len(got["missing_telemetry"]), 37','len(got["missing_telemetry"]), 27')
delete_between('tests/test_pose_metrics.py','    def test_pose_override_unweights_pose_independently_of_audio(self):','    def test_rejects_ambiguous_matrix(self):')
change('tests/test_pose_preview.py',"'grad_norm/direct/layers':.7,'grad_norm/credit/layers':.2,\n                'grad_norm/combined/layers':.8,'logit_grad_norm/direct/text':.01,\n                'logit_grad_alignment/text':-.5,'logit_grad_credit_direct_ratio/text':2}","'grad_norm/direct/layers':.7,'grad_norm/credit/layers':.2}")
change('tests/test_pose_preview.py',"            self.assertIn('grad_norm/combined/layers',doc['train_cols'])\n            self.assertIn('logit_grad_norm/direct/text',doc['train_cols'])\n            self.assertIn('logit_grad_alignment/text',doc['train_cols'])\n            self.assertIn('logit_grad_credit_direct_ratio/text',doc['train_cols'])\n",'')
change('tests/test_pose_preview.py',"            self.assertIn('Combined parameter gradient',html_before)\n            self.assertIn('not a parameter-gradient norm',html_before)\n            self.assertIn(\"f.src='samples/index.html'\",html_before)\n",'')
change('pose-preview/README.txt','  Grad norms: exact combined global and per-component parameter norms before\n              clipping; separately labeled categorical-logit direct/credit\n              diagnostics. Per-loss parameter norms stay empty unless measured.','  Grad norms: combined global norm and direct/credit norms for ten disjoint\n              trainable model components, before clipping.')
change('pose-preview/README.txt','  - New trainer logs include exact lexical/category KL and word-mass weighted\n    lexical contribution; earlier runs without these remain empty.\n  - Combined parameter-component norms and separately named categorical-logit\n    direct/credit diagnostics are supported. Separate direct/credit PARAMETER\n    component norms are unavailable without additional backward passes.','  - Lexical/category KL decomposition is not yet emitted by the trainer.\n  - Separate direct/credit component gradients are not yet emitted.')
change('TELEMETRY_CONTRACT.txt','independent CPU decoding are configured separately. The trainer emits exact\nlexical/category KL, combined parameter-component norms, and distinct\ncategorical-logit gradient diagnostics. Separate per-loss parameter-component\nnorms are not measured; historical runs lacking these producers remain empty.','independent CPU decoding are configured separately. Lexical/category metrics\nand separate direct/credit gradient components remain unimplemented and empty.')
change('TELEMETRY_CONTRACT.txt','method.modality_weights[stream] * the actual codebook coefficient. Audio uses\nmethod.codebook_weights[c]; pose uses method.pose_codebook_weights[c] when\nconfigured, otherwise the shared codebook_weights[c]. Zero/missing/','method.modality_weights[stream] * method.codebook_weights[c]. Zero/missing/')
change('TELEMETRY_CONTRACT.txt','2. Exact text decomposition','2. Text decomposition producer still needed')
change('TELEMETRY_CONTRACT.txt','Log conditional lexical KL as kl/lexical and the weighted lexical contribution\nas kl/lexical_contribution for checking this identity.','Log conditional lexical KL as kl/lexical and additionally log the weighted\nlexical contribution, e.g. kl/lexical_contribution, for checking this identity.')
p=base/'TELEMETRY_CONTRACT.txt';s=p.read_text();a=s.index('3. Parameter norms');b=s.index('4. Sample snapshot producer')
original='''3. Separate gradient producer still needed

Canonical scalar keys:
  grad_norm/direct/<component>
  grad_norm/credit/<component>

Student component IDs (parameter-name top-level prefixes):
  inp       input embeddings
  layers    backbone
  head      text head
  vadapter  audio input adapter
  vgain     audio gain
  tower_s   audio tower
  dec_s     audio depth decoder
  tower_p   pose tower
  dec_p     pose depth decoder
  decp_bb   backbone-to-pose-decoder bridge
Optional pose_head only if instantiated/used. Teachers are frozen.

Measure the Euclidean norm of the entire accumulated optimizer-update gradient,
after distributed gradient reduction, BEFORE gradient clipping. Each global
element is counted once: sum squares over FSDP shards; compensate for replicated
tensors before the scalar all-reduce; take one square root after summation.
Do not average rank norms or sum microbatch norms. Preserve the configured loss
normalization so direct and credit norms are comparable.

Current gradient_norm is the combined direct+credit preclip norm. Current
credit/absolute_sum is a coefficient magnitude, not a gradient norm. The credit
surrogate has scalar value zero while contributing a nonzero gradient. None of
these can reconstruct the requested separate gradients; norm subtraction is
invalid.

Actual instrumentation must separate the weighted analytic direct gradient and
the detached-coefficient score-function gradient on the SAME trajectories and
weights. A separate diagnostic backward/replay or split-gradient capture is
required; coordinate this with the trainer owner because it costs compute and
memory. If computing credit by subtraction, subtract the full gradient vectors
(combined minus direct), never their norms. Retain/restore the original combined
gradient used by the sole optimizer step. Do not zero modality weights to
disable direct gradients: doing that also changes the credit costs. Record the
measurement cadence; unmeasured updates are missing, not zero.

'''
p.write_text(s[:a]+original+s[b:])
print('Reconstructed pre-edit baseline:',base)
