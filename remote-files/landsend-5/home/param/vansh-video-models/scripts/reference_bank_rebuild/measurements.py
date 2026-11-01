"""Load the archived measurement implementation with explicit local model paths."""
import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace

ARCHIVE = Path('/mnt/paramfs/vansh/richmond-7/reference-bank/code')
MODELS = Path('/mnt/paramfs/vansh/richmond-7/evals/expressivity-v2/models')


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def initialize_measurements(archive_root=ARCHIVE, model_root=MODELS, data_root=None):
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    archive_root, model_root = Path(archive_root), Path(model_root)
    audit = _load('audit', archive_root / 'node/ref_audit/audit.py')
    audit59 = _load('audit59', archive_root / 'node/ref_audit/audit59.py')
    teeth = _load('teeth', archive_root / 'node/mouth_open_pilot/teeth.py')
    allteeth = _load('allteeth', archive_root / 'node/mouth_open_pilot/allteeth.py')
    pscan = _load('scan', archive_root / 'node/mouth_open_pilot/scan.py')
    audit.HERE = audit59.HERE = str(model_root)
    if data_root is not None:
        audit.DATA = str(Path(data_root) / 'reference-images-512x960-v1')
    audit59.init()
    # teeth.init hardcodes the old home directory. Its exact options are retained.
    from mediapipe.tasks.python import BaseOptions, vision
    teeth.FACE = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_root / 'face_landmarker.task')),
        running_mode=vision.RunningMode.IMAGE, num_faces=1,
        min_face_detection_confidence=.3, min_face_presence_confidence=.3))
    return SimpleNamespace(audit=audit, audit59=audit59, teeth=teeth, allteeth=allteeth, pscan=pscan)
