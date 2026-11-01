"""Keep managed run outputs and model checkpoints in separate home directories."""
from pathlib import Path


LEGACY_RUNS = Path('/home/param/pose-runs')
OUTPUT_RUNS = Path('/home/param/runs/pose-runs')
RUN_ROOTS = (OUTPUT_RUNS, Path('/home/param/runs/smokes'), Path('/home/param/runs/failed'))
MODEL_RUNS = Path('/home/param/models/trained-weights')


def canonical_run_dir(run_dir):
    """Normalize the old run-root alias without following arbitrary symlinks."""
    run = Path(run_dir).absolute()
    if '..' in run.parts:
        raise ValueError('Run paths must not contain parent traversal')
    if run.is_relative_to(LEGACY_RUNS):
        run = OUTPUT_RUNS / run.relative_to(LEGACY_RUNS)
    if run.parent in RUN_ROOTS and run.is_symlink():
        target = Path(run.readlink())
        if not target.is_absolute():
            target = run.parent / target
        if target.parent in RUN_ROOTS and target.name == run.name:
            run = target
    return run


def training_model_dir(run_dir):
    run = canonical_run_dir(run_dir)
    if run.parent in RUN_ROOTS:
        return MODEL_RUNS / run.name
    if run in RUN_ROOTS:
        raise ValueError('A training run needs its own directory')
    # Explicit scratch runs and CPU tests retain their self-contained layout.
    return run


def safe_model_section(run_dir, section):
    """Accept only the expected compatibility link; never a redirected target."""
    if section not in {'checkpoints', 'exports', 'saves'}:
        raise ValueError(f'Unknown model storage section: {section}')
    run = canonical_run_dir(run_dir)
    root = training_model_dir(run) / section
    for path in (run, root):
        if any(part.is_symlink() for part in (path, *path.parents)):
            raise ValueError('Model storage refuses symlink run or target paths')
    alias = run / section
    if alias != root and (alias.exists() or alias.is_symlink()):
        if not alias.is_symlink() or alias.resolve() != root:
            raise ValueError(f'Model storage requires migration of {alias} to {root}')
    return root


def prepare_training_storage(run_dir, sections):
    """Create canonical directories and race-safe legacy lookup links."""
    run = canonical_run_dir(run_dir)
    if training_model_dir(run) == run:
        return run
    run.mkdir(parents=True, exist_ok=True)
    for section in sections:
        root = safe_model_section(run, section)
        root.mkdir(parents=True, exist_ok=True)
        alias = run / section
        try:
            alias.symlink_to(root, target_is_directory=True)
        except FileExistsError:
            # Other local distributed ranks may create the same link first.
            safe_model_section(run, section)
    return training_model_dir(run)
