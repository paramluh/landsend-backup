"""rosie-ft (10-30): the single-host paths that replace the cluster's /data /scratch defaults of the copied modules. Imported FIRST by
train_rosie_cotrain.py and tools/scan_zoom.py. Nothing in the copied modules is edited: env defaults are set before their import and the
two hard-coded module globals (model_ppm.QWEN32 / QWEN4B_CFG, chat_ppm.Q1) and pinned_fa3's snapshot root are patched at runtime.
Every path can be overridden by its env var."""
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.environ.get("ROSIE_ASSETS", "/workspace/aloe-live/assets")
CODE_LIVE = os.environ.get("ROSIE_LIVE_CODE", "/workspace/aloe-live/code")
P = {
    "qwen32": os.environ.get("ROSIE_QWEN32", f"{CODE_LIVE}/models/Qwen3-32B"),             # config + tokenizer only (no weights needed)
    "qwen4b_cfg": os.environ.get("ROSIE_QWEN4B_CFG", f"{CODE_LIVE}/ppm/configs/Qwen3-4B"),
    "fa3": os.environ.get("ROSIE_FA3", f"{ASSETS}/vllm-flash-attn3"),
    "voxtral": os.environ.get("VOX_MODEL", f"{ASSETS}/voxtral-realtime"),
    "mimi": os.environ.get("ROSIE_MIMI", f"{ASSETS}/mimi"),
    "q1": os.environ.get("ROSIE_Q1", f"{HERE}/q1/qwen-1ch"),
    "zoom_src": os.environ.get("ZOOM_SRC", "/workspace/data/zoom-strangers-5k"),
    "zoom_table": os.environ.get("ZOOM_TABLE", "/workspace/data/zoom-strangers-5k/zoom_table.npz"),
}
os.environ.setdefault("ZOOM_SRC", P["zoom_src"])
os.environ.setdefault("ZOOM_TABLE", P["zoom_table"])
os.environ.setdefault("Q1_TOKENIZER", P["qwen32"])
os.environ.setdefault("S4_VOXTRAL_PATH", P["voxtral"])
os.environ.setdefault("VOX_MODEL", P["voxtral"])
os.environ.setdefault("PPM_CD", "2")
os.environ.setdefault("S3_MIN_FRAMES", "125")                    # = data_rosie.py (s3rows reads it at import; Rosie chunks >= 125 frames)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
for p in (os.path.join(HERE, "q1", "convgen"), os.path.join(HERE, "q1", "convgen_real"), os.path.join(HERE, "vendor"), os.path.join(HERE, "lib"), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)                                  # q1/convgen_real (the real synth_timing.py, if provided) before the shim


def _fa3_loader():
    import pinned_fa3 as PF
    if PF._module is not None:
        return PF._module
    package = os.path.join(P["fa3"], "build", PF.BUILD)
    assert os.path.isdir(package), package
    spec = importlib.util.spec_from_file_location("_text_tts_fa3", os.path.join(package, "__init__.py"), submodule_search_locations=[package])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    PF._module = mod
    return mod


def patch():
    """call after sys.path is set; idempotent."""
    import pinned_fa3 as PF
    PF._load_module = _fa3_loader
    import model_ppm as M
    M.QWEN32, M.QWEN4B_CFG = P["qwen32"], P["qwen4b_cfg"]
    import chat_ppm as CH
    CH.Q1 = P["q1"]
    import data_ppm as DP
    DP.TOKENIZER = P["qwen32"]
    return P
