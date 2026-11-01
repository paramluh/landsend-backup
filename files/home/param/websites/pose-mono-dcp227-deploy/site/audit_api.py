"""Serve the curated pose-jump report and inspection clips."""
from pathlib import Path


def setup(app):
    root = Path('/home/param/runs/pose-runs/pose-jump-audit-suite-20261031/published')
    root.mkdir(parents=True, exist_ok=True)
    app.router.add_static('/diagnostics/pose-jumps', root, show_index=False, follow_symlinks=False)
