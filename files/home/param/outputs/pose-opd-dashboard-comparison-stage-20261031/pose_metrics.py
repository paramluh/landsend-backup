"""Turn documented weighted pose losses into raw KL curves, without guessing.

This module reads only the supplied dashboard document and training method.
It never imports the trainer and cannot manufacture lexical/category metrics,
separate direct/credit gradients, or rollout samples.
"""
from __future__ import annotations

import math


CRITICAL_METRICS = (
    "kl/text", "kl/lexical", "kl/category", "kl/audio/tower",
    "kl/audio/depth_mean", "kl/pose/tower", "kl/pose/depth_mean",
)
COMPONENTS = (
    ("inp", "Input embeddings"), ("layers", "Backbone"), ("head", "Text head"),
    ("vadapter", "Audio input adapter"), ("vgain", "Audio gain"),
    ("tower_s", "Audio tower"), ("dec_s", "Audio depth decoder"),
    ("tower_p", "Pose tower"), ("dec_p", "Pose depth decoder"),
    ("decp_bb", "Backbone → pose-decoder bridge"),
)


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _positive(value):
    return value if _finite(value) and value > 0 else None


def _ratio(value, *weights):
    if not _finite(value) or any(_positive(weight) is None for weight in weights):
        return None
    denominator = math.prod(weights)
    if not math.isfinite(denominator) or denominator <= 0:
        return None
    value = value / denominator
    return value if math.isfinite(value) else None


def enrich_doc(doc: dict, method: dict) -> dict:
    """Return a new S3-style document with raw Kls and availability metadata.

    Finite explicitly logged values always win. Missing modality/codebook
    weights, zero coefficients, and incomplete depth codebooks yield None.
    Raw means retain the run's row_mean or position_mean reduction; this is
    removal of configured objective coefficients, not a change of averaging.
    """
    method = method if isinstance(method, dict) else {}
    modalities = method.get("modality_weights")
    modalities = modalities if isinstance(modalities, dict) else {}
    codebooks = method.get("codebook_weights")
    codebooks = codebooks if isinstance(codebooks, (list, tuple)) and len(codebooks) == 32 else [None] * 32
    pose_codebooks = method.get("pose_codebook_weights")
    if pose_codebooks is None:
        pose_codebooks = codebooks
    elif not isinstance(pose_codebooks, (list, tuple)) or len(pose_codebooks) != 32:
        pose_codebooks = [None] * 32
    media_codebooks = {"audio": codebooks, "pose": pose_codebooks}
    reduction = method.get("reduction") if method.get("reduction") in {"row_mean", "position_mean"} else "unspecified"
    columns = list(doc.get("train_cols") or ["step", "rows"])
    if len(columns) != len(set(columns)):
        raise ValueError("train_cols contains duplicate metric names")
    original_columns = list(columns)
    matrix = [list(row) for row in doc.get("train", [])]
    if any(len(row) != len(columns) for row in matrix):
        raise ValueError("Each train row must match train_cols")
    original = [{key: value for key, value in zip(columns, row)} for row in matrix]
    records = [dict(row) for row in original]
    provenance = {}
    definitions = dict(doc.get("metric_definitions") or {})

    def declare(key, label, definition, dependencies=(), unit="nats", direction="student || teacher"):
        definitions[key] = {"label": label, "definition": definition, "direction": direction,
                            "reduction": reduction, "unit": unit, "dependencies": list(dependencies)}

    def install(key, derive=None):
        if key not in columns:
            columns.append(key)
        counts = {"logged_rows": 0, "derived_rows": 0, "missing_rows": 0}
        for before, row in zip(original, records):
            if _finite(before.get(key)):
                counts["logged_rows"] += 1
                row[key] = before[key]
            else:
                value = derive(row) if derive else None
                row[key] = value if _finite(value) else None
                counts["derived_rows" if row[key] is not None else "missing_rows"] += 1
        source = ("mixed" if counts["logged_rows"] and counts["derived_rows"] else
                  "logged" if counts["logged_rows"] else "derived" if counts["derived_rows"] else "missing")
        provenance[key] = {"source": source, **counts}
        if key in definitions:
            definitions[key].update(provenance[key])

    declare("kl/text", "Text KL", "Full executable text distribution: all 151,682 token IDs, against the audio teacher; objective modality coefficient removed.", ["loss/text"])
    install("kl/text", lambda row: _ratio(row.get("loss/text"), modalities.get("text")))
    declare("kl/lexical", "Lexical KL", "Conditional distribution over word IDs 0..151642, evaluated on the same sampled text decisions and reduction as full text KL. Not weighted by student WORD probability.")
    install("kl/lexical")
    declare("kl/category", "Category KL", "40 categories: WORD (sum of probabilities for IDs 0..151642), plus one category for each executable special ID 151643..151681. Uses the same audio teacher, text decisions and KL direction as full text KL.")
    install("kl/category")
    declare("kl/lexical_contribution", "Word-mass weighted lexical KL", "Student WORD probability times conditional lexical KL, multiplied before the same frame/row reduction as full text KL. Full text KL equals category KL plus this contribution.")
    install("kl/lexical_contribution")
    for stream in ("audio", "pose"):
        target = "audio teacher" if stream == "audio" else "configured pose teacher (including configured CFG)"
        for cb in range(32):
            key, weighted = f"kl/{stream}/cb{cb}", f"loss/{stream}_cb{cb}"
            declare(key, f"{stream.title()} cb{cb} KL", f"Raw categorical KL for {stream} codebook {cb} against the {target}; modality and codebook objective coefficients removed.", [weighted])
            install(key, lambda row, weighted=weighted, cb=cb, stream=stream: _ratio(row.get(weighted), modalities.get(stream), media_codebooks[stream][cb]))
        key, cb0 = f"kl/{stream}/tower", f"kl/{stream}/cb0"
        declare(key, f"{stream.title()} tower KL", f"{stream.title()} tower predicts codebook 0. This is raw cb0 KL against the {target}.", [cb0])
        install(key, lambda row, cb0=cb0: row.get(cb0))
        key = f"kl/{stream}/depth_mean"
        depth_keys = tuple(f"kl/{stream}/cb{cb}" for cb in range(1, 32))
        declare(key, f"{stream.title()} depth KL · mean cb1–31", "Arithmetic mean of all 31 raw depth-decoder codebook Kls (cb1..31). Excludes tower cb0; all 31 values are required.", depth_keys)

        def depth_mean(row, depth_keys=depth_keys):
            values = [row.get(part) for part in depth_keys]
            # Divide before summing to avoid overflowing a finite mean.
            return math.fsum(value / 31 for value in values) if all(_finite(value) for value in values) else None

        install(key, depth_mean)

    components = list(COMPONENTS)
    for component, label in (("pose_head", "Optional pose head"), ("other", "Other trainable parameters")):
        if any(key.startswith("grad_norm/") and key.endswith('/' + component) for key in original_columns):
            components.append((component, label))
    interval = method.get('loss_gradient_interval_steps')
    interval = interval if type(interval) is int and interval > 0 else None
    cadence = (f"Measured every {interval} completed updates; intermediate updates are unmeasured."
               if interval else "Measured only when explicitly recorded; no cadence is configured for this run.")
    groups, missing = [], []
    for component, label in components:
        keys = {term: f"grad_norm/{term}/{component}" for term in ("combined", "direct", "credit")}
        available = {}
        for term, key in keys.items():
            declare(key, f"{label} · {term}", f"Global full-update preclip L2 norm of the {term} loss gradient for student parameters in {component}. FSDP shards counted once; replicated tensors counted once. No per-rank norm averaging.", unit="L2 norm", direction=None)
            definitions[key]["reduction"] = "global_l2_preclip"
            definitions[key]['measurement_interval_steps'] = 1 if term == 'combined' else interval
            if term != 'combined':
                definitions[key]['definition'] += ' ' + cadence
            # No synthetic gradient columns/values: consumers use this metadata
            # to present a truthful empty state until the producer logs them.
            counts = {"logged_rows": sum(_finite(row.get(key)) for row in original), "derived_rows": 0,
                      "missing_rows": sum(not _finite(row.get(key)) for row in original)}
            available[term] = bool(counts["logged_rows"])
            definitions[key].update(source="logged" if available[term] else "missing", **counts)
            if not available[term]:
                reason = ("Combined parameter gradients are not present in this log." if term == "combined" else
                          "No separate per-loss parameter-gradient measurement is present for this run. " + cadence)
                missing.append({"metric": key, "reason": reason, "producer": "Trainer backward instrumentation before clipping and optimizer.step"})
        groups.append({"component": component, "label": label, "keys": keys, "available": available})

    parameter_loss_groups = []
    for modality in ('text', 'audio', 'pose'):
        series = []
        for term, label in (('direct', f'Direct {modality} KL'), ('credit', f'Credit from {modality} KL')):
            key = f'grad_norm/{term}/{modality}/all'
            series.append({'key': key, 'label': label})
        parameter_loss_groups.append({'id': f'parameter_loss_{modality}',
            'label': f'{modality.title()} KL · global parameter-gradient norms', 'series': series})
    for component, label in [('all', 'All trainable parameters'), *components]:
        series = []
        for modality in ('text', 'audio', 'pose'):
            for term in ('direct', 'credit'):
                key = f'grad_norm/{term}/{modality}/{component}'
                label_term = f'Direct {modality} KL' if term == 'direct' else f'Credit from {modality} KL'
                declare(key, f'{label} · {label_term}',
                    f'Exact full-update preclip parameter-gradient L2 norm for {component}, from the weighted {term} {modality} objective. Credit is grouped by originating KL modality, across every causal action path. FSDP and replicated parameters are counted once. ' + cadence,
                    unit='parameter-gradient L2 norm', direction=None)
                definitions[key].update(reduction='global_l2_preclip', measurement_interval_steps=interval)
                if term == 'credit' and modality == 'pose':
                    definitions[key]['definition'] += ' Reconstructed by subtracting the other five parameter-gradient vectors from the normal total, with floating-point residual roundoff.'
                install(key)
                series.append({'key': key, 'label': label_term})
        if component != 'all':
            parameter_loss_groups.append({'id': f'parameter_loss_component_{component}',
                'label': f'{label} · parameter gradients by KL origin', 'series': series})

    logit_groups, logit_diagnostics = [], []
    for modality in ("all", "text", "audio", "pose"):
        title = "All modalities" if modality == "all" else modality.title()
        series = []
        for term, label, detail in (
            ("combined", f"Combined {modality} objective", f"Direct plus credit gradients originating in {modality} KL, across the full action-logit space; includes their vector cross terms."),
            ("direct", f"Direct {modality} KL", f"Direct analytic {modality} KL gradient with respect to its prediction logits."),
            ("credit_from", f"Credit from {modality} KL", f"Credit originating in {modality} KL, across all causally preceding action logits in every modality."),
            ("credit_on", f"All credit on {modality} decisions", f"Credit from all KL modalities, restricted to sampled {modality} action logits."),
        ):
            if modality == "all" and term == "credit_on":
                continue
            key = f"logit_grad_norm/{term}/{modality}"
            declare(key, label, detail + " Exact L2 norm in categorical-logit space, with the training objective's weights and global normalization; squared entries sum across chunks, microbatches and ranks before one square root. This is not a parameter-gradient norm.", unit="logit-gradient L2 norm", direction=None)
            definitions[key]["reduction"] = "global_logit_l2"
            install(key)
            series.append({"key": key, "label": label})
        logit_groups.append({"id": f"logit_{modality}", "label": f"{title} · categorical-logit gradient norms", "series": series})
        for prefix, title, unit, detail in (
            ("logit_grad_alignment", "Direct / credit logit-gradient cosine", "cosine", "Cosine between the direct gradient of this modality's KL and credit originating from the same KL modality, in the full categorical-logit space."),
            ("logit_grad_credit_direct_ratio", "Credit / direct logit-gradient norm ratio", "ratio", "L2 norm of credit originating from this modality's KL divided by its direct logit-gradient L2 norm."),
        ):
            key = f"{prefix}/{modality}"
            declare(key, f"{modality.title()} · {title}", detail + " For all, use the total direct and total credit vectors, including cross-modality terms. Not a parameter-gradient statistic; undefined values remain missing.", unit=unit, direction=None)
            definitions[key]["reduction"] = "global_logit_diagnostic"
            install(key)
            logit_diagnostics.append([key, f"{modality.title()} · {title}"])
        key = f"logit_grad_dot/{modality}"
        declare(key, f"{modality.title()} · direct / credit logit-gradient dot product",
                "Inner product of the direct and originating-credit logit gradients; all uses the total vectors including cross-modality terms. Not a parameter-gradient statistic.",
                unit="logit-gradient inner product", direction=None)
        definitions[key]["reduction"] = "global_logit_dot"
        install(key)

    conversation_groups = []
    for measure, title in (("probability", "Conversation · sampling probabilities"),
                           ("sampled_fraction", "Conversation · sampled token fractions")):
        series = []
        for role in ("student", "partner"):
            for category, label in (("pad", "PAD"), ("new_word", "NEW_WORD"), ("word", "WORD")):
                key = f"conversation/{role}/{measure}/{category}"
                detail = ("Mean probability under the actual post-policy sampling distribution. WORD integrates all lexical token IDs 0..151642."
                          if measure == "probability" else "Actual sampled category count divided by the number of generated text decisions.")
                declare(key, f"{role.title()} · {label}", detail +
                        " Excludes prefill, padding, forced tails and repeated partner refreshes. PAD, NEW_WORD and WORD are not renormalized to sum to one.",
                        unit="probability" if measure == "probability" else "fraction", direction=None)
                definitions[key]["reduction"] = "global_decision_mean"
                install(key)
                series.append({"key": key, "label": f"{role.title()} · {label}"})
        conversation_groups.append({"id": f"conversation_{measure}", "label": title, "series": series})
    for role in ("student", "partner"):
        for suffix, label in (("decisions", "Generated text decisions"),
                              *((f"sampled_count/{category}", f"Sampled {category.upper()} count")
                                for category in ("pad", "new_word", "word"))):
            key = f"conversation/{role}/{suffix}"
            declare(key, f"{role.title()} · {label}", "Global number of unique generated text decisions in this update, excluding prefill, padding, forced tails and repeated partner refreshes.", unit="count", direction=None)
            definitions[key]["reduction"] = "global_sum"
            install(key)

    for key in CRITICAL_METRICS:
        if provenance[key]["source"] != "missing":
            continue
        if key in {"kl/lexical", "kl/category"}:
            reason = "Requires student and teacher text distributions; cannot be recovered from scalar full-text KL."
            producer = "No-gradient statistics inside the existing bounded text-head logits chunks"
        else:
            reason = "Requires finite raw telemetry, or weighted contributions with known positive modality/codebook coefficients."
            producer = "Current loss metrics plus this run's exact config.json method"
        missing.append({"metric": key, "reason": reason, "producer": producer})

    # Keep requested Critical curves together without changing existing column
    # positions. All source columns, including any explicit extra gradient
    # grouping, are preserved in their original order and values.
    enriched = dict(doc)
    enriched.update(
        train_cols=columns,
        train=[[row.get(key) for key in columns] for row in records],
        critical_metrics=list(CRITICAL_METRICS),
        metric_definitions=definitions,
        grad_groups=groups,
        parameter_loss_grad_groups=parameter_loss_groups,
        parameter_grad_cadence={'interval_steps': interval, 'extra_backward_passes': 5 if interval else None,
                               'residual_term': 'credit/pose' if interval else None,
                               'missing_updates': 'not measured'},
        logit_grad_groups=logit_groups,
        logit_grad_diagnostics=logit_diagnostics,
        conversation_groups=conversation_groups,
        missing_telemetry=missing,
        kl_metadata={"direction": "student || teacher", "unit": "nats", "reduction": reduction,
                     "teacher_text_audio": "audio_teacher", "teacher_pose": "pose_teacher",
                     "pose_teacher_cfg_scale": method.get("pose_teacher_cfg_scale"),
                     "tower_codebooks": [0], "depth_codebooks": list(range(1, 32)),
                     "lexical_vocab_size": 151643, "executable_vocab_size": 151682,
                     "category_count": 40,
                     "text_identity": "KL(text) = KL(category) + mean[P_student(WORD) * KL(lexical conditional)]"},
    )
    return enriched
