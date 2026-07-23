import os
import sys


def setup_output_log(hydra_output_dir: str) -> None:
    """Tee stdout+stderr to {hydra_output_dir}/train.log.

    The .hydra/ config files already land in hydra_output_dir, so the log
    sits alongside them in the same directory under saves/slurm_outputs/.
    """
    log_path = os.path.join(hydra_output_dir, "train.log")
    os.makedirs(hydra_output_dir, exist_ok=True)

    log_file = open(log_path, "a", buffering=1)  # line-buffered

    class _Tee:
        def __init__(self, original, secondary):
            self._orig = original
            self._sec = secondary

        def write(self, data):
            self._orig.write(data)
            self._sec.write(data)

        def flush(self):
            self._orig.flush()
            self._sec.flush()

        def fileno(self):
            return self._orig.fileno()

        def isatty(self):
            return self._orig.isatty()

    sys.stdout = _Tee(sys.stdout, log_file)
    sys.stderr = _Tee(sys.stderr, log_file)
    print(f"Logging output to: {log_path}")
