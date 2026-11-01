"""Pinned production pose-duplex recipe and local data-path adaptation."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path

DEFAULT_DATA_ROOT = Path("/home/param/datasets/aloe-duplex")
DEFAULT_CHECKPOINT_CFG = Path(
    "/home/param/models/ppm-aloe-poseduplex-v5-final/export-498-dcp/train_cfg.json"
)
STORE_DIGESTS = (
    "7318ffb3ea75cbcc3ff48b63b6cf8b88", "31ecb2273542310096936c0baf49efa8",
    "1759200f7818c6017dfb6a46e37fa5df", "611b2e7edb1503cc600a6489e49dcf40",
)
SOURCE_POSE_INDEX_SHA256 = "7114d5a04f43778b29244e2ef1867d6d960d619f1515bf3c25f4b7f155f8d8a1"
TRAIN_ITEMS_PER_EPOCH = 15_898
EVAL_ITEMS = 316
PRODUCTION_DIR = Path(__file__).resolve().parent / "production"


def load_recipe(checkpoint_cfg=DEFAULT_CHECKPOINT_CFG, *, world=48, epochs=2,
                micro_batch_size=1):
    """Use export-498 settings with balanced losses and shared depth sampling.

    Production pads each epoch separately. At world 56, two epochs take 568
    updates with one example/GPU or 284 with two, with 12 dummy examples total.
    """
    if int(world) != world or world < 1 or int(epochs) != epochs or epochs < 1:
        raise ValueError("world and epochs must be positive integers")
    if micro_batch_size not in (1, 2) or int(micro_batch_size) != micro_batch_size:
        raise ValueError("micro_batch_size must be one or two")
    source = json.loads(Path(checkpoint_cfg).read_text())
    if source.get("step") != 498 or source.get("run") != "ppm-aloe-poseduplex-v5-j30074":
        raise ValueError("Expected the production pose-duplex export-498 training config")
    cfg = copy.deepcopy(source["cfg"])
    # Production omits this optional key when QQA_LW_WORD has its default 1.
    cfg.setdefault("lw_word", 1.0)
    expected = {"crop_conv": False, "persona_sides": "none", "voice_frac": 0.3,
                "sysprompt": "v2_voiceB_only", "pose_delay": 2,
                "light_frac": 0.1, "pose_light_frac": 1.0, "clip_min": 38,
                "clip_max": 125, "voice_seed": 20261011, "sp_cap": 2000}
    for key, value in expected.items():
        if cfg["data"].get(key) != value:
            raise ValueError(f"Production data setting changed: {key}")
    if cfg["posedup"]["items"] != "all" or cfg["posedup"]["voice_rule"] != "v2_30":
        raise ValueError("Expected the complete production item set and v2_30 voice rule")
    if cfg["samsep"]["digests"] != list(STORE_DIGESTS):
        raise ValueError("Production store identities differ")
    global_batch_size = int(world) * int(micro_batch_size)
    steps = math.ceil(TRAIN_ITEMS_PER_EPOCH / global_batch_size) * int(epochs)
    warmup = round(10 * steps / 306)  # Original run_poseduplex.env schedule rule.
    cfg.update(steps=steps, warmup=warmup, world=int(world), epochs=int(epochs),
               micro_batch_size=int(micro_batch_size), global_batch_size=global_batch_size,
               gradient_accumulation=1, conversation_frames=7500,
               epoch_steps=math.ceil(TRAIN_ITEMS_PER_EPOCH / global_batch_size),
               expected_train_micros=TRAIN_ITEMS_PER_EPOCH * int(epochs),
               expected_eval_micros=EVAL_ITEMS,
               export_steps=[round(x * steps / 253) for x in (42, 84, 126, 169, 211)])
    for key in ("sched", "dec_sched", "dec_p_sched"):
        cfg[key].update(warmup=warmup, horizon=steps, wsd_decay_start=warmup,
                        wsd_decay_end=steps)
    cfg["accum"].update(logical_world=global_batch_size, physical_world=int(world), slots=1,
                        micro_batch_size=int(micro_batch_size), aloe_ranks=int(world), zoom_ranks=0)
    cfg["accum"].pop("host_ranks", None)
    cfg["cotrain"]["aloe_logical_world"] = global_batch_size
    cfg["samsep"]["expect_train_micros"] = cfg["expected_train_micros"]
    source_loss_weights = {key: cfg[key] for key in
                           ("lw_word", "w_text", "w_cb0", "w_cb", "w_pcb0", "w_pcb")}
    loss_overrides = {
        "w_cb0": 0.25, "w_cb": 0.75, "w_pcb0": 0.25, "w_pcb": 0.75,
        "modality_weights": {"text": 1 / 3, "audio": 1 / 3, "pose": 1 / 3},
    }
    cfg.update(copy.deepcopy(loss_overrides))
    # Keep the vendored production loader pinned. The adapter changes only its
    # training decoder-frame selection after constructing the original example.
    cfg["source_loader_data"] = copy.deepcopy(cfg["data"])
    cfg["source_loader_posedup"] = copy.deepcopy(cfg["posedup"])
    cfg["data"]["pose_light_frac"] = cfg["data"]["light_frac"]
    cfg["posedup"]["pose_light_frac"] = cfg["data"]["light_frac"]
    sampling_override = {
        "speech": 0.1, "pose": 0.1, "shared_conversation_frames": True,
        "source": "existing_production_speech_light",
        "speech_delay": int(cfg["cd"]), "pose_delay": int(cfg["data"]["pose_delay"]),
        "application": "postprocess_training_decoder_indices_only",
        "evaluation": "all_valid_then_cap", "evaluation_max": int(cfg["eval_light_max"]),
    }
    cfg["depth_frame_sampling"] = copy.deepcopy(sampling_override)
    checkpoint_overrides = {
        "eval_every": 0, "checkpoint_interval_seconds": 7200,
        "keep_latest_optimizer_state": True, "save_final": True, "export_steps": [],
    }
    cfg.update(copy.deepcopy(checkpoint_overrides))
    cfg["provenance"] = {"checkpoint_config": str(Path(checkpoint_cfg)),
                         "checkpoint_config_sha256": hashlib.sha256(Path(checkpoint_cfg).read_bytes()).hexdigest(),
                         "production_run": source["run"], "production_step": source["step"],
                         "source_loss_weights": source_loss_weights,
                         "user_loss_overrides": loss_overrides,
                         "loss_override_reason": "User requested equal one-third text/speech/pose contributions; speech and pose each use one-quarter cb0 plus three-quarters mean cb1..31.",
                         "source_training_sampling": {"speech": 0.1, "pose": 1.0, "shared_conversation_frames": False},
                         "user_sampling_override": sampling_override,
                         "sampling_override_reason": "User requested the same deterministic ten-percent conversation-frame sample for both training depth decoders. Original input sequences, cb0 coverage, and evaluation selection remain unchanged; source_loader_data/source_loader_posedup record the unchanged loader settings.",
                         "source_checkpoint_schedule": {"eval_every": source["cfg"]["eval_every"], "export_steps": source["cfg"]["export_steps"]},
                         "user_checkpoint_overrides": checkpoint_overrides,
                         "checkpoint_override_reason": "User requested no evaluations, checkpoints every two hours retaining the latest optimizer state, and a final checkpoint.",
                         "conditioning": "One shared B voice/prompt draw; monoplex teacher A is natively empty; student and audio teacher share raw A audio."}
    return cfg


def configure_environment(data_root=DEFAULT_DATA_ROOT):
    """Set original data module knobs before imports; never modify production code."""
    root = Path(data_root).resolve()
    index = root / "pose/POSE_INDEX.json"
    raw = index.read_bytes()
    doc = json.loads(raw)
    if doc["store_digests"] != list(STORE_DIGESTS):
        raise ValueError("Local dataset store digests differ from production")
    provenance = doc.get("local_provenance", {})
    if provenance.get("source_index_sha256") != SOURCE_POSE_INDEX_SHA256:
        raise ValueError("Local pose index is not derived from the pinned production index")
    roots = [str(root / f"s2h-m1-samsep/stores/m1n{k}") for k in range(4)]
    if doc["stores"] != roots or doc["data_root"] != str(root / "pose/data"):
        raise ValueError("Local pose index paths do not match data_root")
    settings = {
        "PPM_CD": "2", "ALOE_SAMSEP_STORES": str(root / "s2h-m1-samsep/stores"),
        "ALOE_STORES_FROM_DATA": "1", "ALOE_SAMSEP_SRC": str(root / "s2h-m1-samsep/stores"),
        "ALOE_SAMSEP_DIGESTS": ",".join(STORE_DIGESTS),
        "ALOE_M1VOX_INDEX": str(root / "pose/M1VOX_INDEX_samsep.json"),
        "ALOE_RAWGAIN": str(root / "pose/hf/provenance/RAWGAIN_samsep.original.json"),
        "ALOE_RAWGAIN_PLAN_DIGEST": "d1a3e3721328da0a", "VOX_A_GAIN": "raw",
        "ALOE_FLAC_ROOT": str(root / "original/flac"), "ALOE_ROWCACHE": "",
        "POSEDUP_POSE": "1", "POSEDUP_ITEMS": "all", "POSEDUP_POSE_INDEX": str(index),
        "POSEDUP_POSE_INDEX_SHA": hashlib.sha256(raw).hexdigest(),
        "POSEDUP_POSE_SHA_EVERY_READ": "1", "POSEDUP_POSE_LIGHT_FRAC": "1.0",
        "POSEDUP_SPEECH_LIGHT_FRAC": "0.1", "POSEDUP_VOICE_RULE": "v2_30",
        "POSEDUP_EPOCH_SEED": "20261011",
    }
    if os.environ.get("S4_FFMPEG"):
        settings["S4_FFMPEG"] = os.environ["S4_FFMPEG"]
    else:
        try:
            import imageio_ffmpeg
        except ImportError:
            pass  # Metadata-only checks do not need an audio decoder.
        else:
            settings["S4_FFMPEG"] = imageio_ffmpeg.get_ffmpeg_exe()
    os.environ.update(settings)
    return settings
