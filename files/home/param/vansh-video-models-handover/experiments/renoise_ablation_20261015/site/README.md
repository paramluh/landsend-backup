# Resolver noise ablation viewer

Buildless static site. Serve `dist/` over HTTP; `experiment.json` supplies all run metadata, media, and observed metrics. No build dependencies. Site registration and publication are owned by the parent agent.

Each clip has `id`, optional `title`, `shortLabel`, `poster`, and `baseline` / `direct` objects with `src`, optional `poster`, and optional full-quality `original`. Both videos must have the same timeline. Gallery order is the predefined first 8 IDs.

`evaluation.aggregate` and `evaluation.perClip[id]` contain `{label, description, baseline, direct, delta, digits, ci}` rows. `ci` is the paired difference interval and appears only for aggregate rows. `evaluation.curveMetrics` contains `{id, label, description}`. `evaluation.curves[clipId][metricId]` has `baseline` and `direct` arrays of `[seconds, value]`. Results must come from full-quality inference output, never compressed website copies.

Website media can be transcoded for hosting; preserve original output elsewhere, expose full-duration web playback, and record the web codec/resolution/compression in `mediaNote`.

Validation: `node --check dist/app.js`; validate manifest JSON and referenced local assets before packaging. Browser QA depends on available local browser tooling.
