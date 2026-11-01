"""Keep subprocess output attached to the terminal and an append-only log."""
from pathlib import Path
import subprocess
import sys
from threading import Thread

CPU_ENV = ('OMP_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1', 'MKL_NUM_THREADS=1',
           'NUMEXPR_NUM_THREADS=1', 'RAYON_NUM_THREADS=1', 'PYTHONDONTWRITEBYTECODE=1')


def start_logged(argv, log_path, **kwargs):
    log = Path(log_path).open('ab')
    try:
        process = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, **kwargs)
    except BaseException:
        log.close()
        raise

    def stream():
        with log, process.stdout:
            for block in iter(lambda: process.stdout.read1(65536), b''):
                sys.stdout.buffer.write(block)
                sys.stdout.buffer.flush()
                log.write(block)
                log.flush()

    output = Thread(target=stream)
    output.start()
    return process, output
