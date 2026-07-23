"""SIGUSR1-based graceful checkpoint-and-requeue for Slurm jobs."""

import json
import os
import signal
import subprocess
import time

import torch

from .distributed import is_main_process

_early_sigusr1_received: bool = False


class ResubmitManager:
    """Coordinate a graceful Slurm requeue across distributed workers."""

    def __init__(self):
        self._received: bool = False
        self._end_time: int | None = None
        self._wall_time_buffer_secs: int = 600

    @staticmethod
    def register_early() -> None:
        """Record Slurm signals received before distributed setup completes."""
        def _handler(_signum, _frame):
            rank = os.environ.get("RANK", "unknown")
            os.write(1, f"\n[Rank {rank}] [ResubmitManager] Early signal received!\n".encode())
            global _early_sigusr1_received
            _early_sigusr1_received = True

        signal.signal(signal.SIGUSR1, _handler)
        signal.signal(signal.SIGUSR2, _handler)

    def register(self) -> None:
        """Register Slurm signal handlers once the trainer is ready."""
        import torch.distributed as dist

        rank = dist.get_rank() if dist.is_initialized() else 0
        global _early_sigusr1_received
        if _early_sigusr1_received:
            self._received = True
            if is_main_process():
                print(f"[Rank {rank}] [ResubmitManager] Picking up early SIGUSR1 flag.", flush=True)

        def _handler(signum, _frame):
            sig_name = "SIGUSR1" if signum == signal.SIGUSR1 else "SIGUSR2"
            os.write(
                1,
                f"\n[Rank {rank}] [ResubmitManager] {sig_name} received! Setting exit flag...\n".encode(),
            )
            self._received = True

        signal.signal(signal.SIGUSR1, _handler)
        signal.signal(signal.SIGUSR2, _handler)
        if is_main_process():
            print(f"[Rank {rank}] [ResubmitManager] SIGUSR1/SIGUSR2 handlers registered.", flush=True)

        end_time = os.environ.get("SLURM_JOB_END_TIME")
        if end_time:
            try:
                self._end_time = int(end_time)
            except ValueError:
                pass

    @property
    def received(self) -> bool:
        if self._received:
            return True
        if self._end_time is not None:
            if time.time() + self._wall_time_buffer_secs >= self._end_time:
                os.write(1, b"\n[ResubmitManager] Near wall time - setting requeue flag.\n")
                self._received = True
        return self._received

    def coordinate(self, device: torch.device) -> bool:
        """Return whether any distributed worker has requested a requeue."""
        local_received = self.received
        if torch.distributed.is_initialized():
            flag = torch.tensor([int(local_received)], dtype=torch.int32, device=device)
            torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MAX)
            self._received = bool(flag.item())
        return self._received

    def requeue(
        self,
        last_complete_ckpt: str,
        run_id: str,
        saving_root_dir: str,
        max_resubmit: int,
        epochs_done: int,
        total_epochs: int,
    ) -> None:
        """Persist resume state, then ask Slurm to requeue this job."""
        if not last_complete_ckpt or max_resubmit <= 0 or epochs_done >= total_epochs:
            return

        job_id = os.environ.get("SLURM_JOB_ID", "noslurm")
        sidecar_path = os.path.join(saving_root_dir, f"requeue_{job_id}.json")
        state = {
            "ckpt_path": last_complete_ckpt,
            "run_id": run_id,
            "resubmits_left": max_resubmit - 1,
        }
        try:
            os.makedirs(saving_root_dir, exist_ok=True)
            with open(sidecar_path, "w") as file:
                json.dump(state, file)
                file.flush()
                os.fsync(file.fileno())
        except OSError as exc:
            print(f"[ResubmitManager] could not write requeue state: {exc}", flush=True)
            return

        if job_id == "noslurm":
            return
        try:
            subprocess.run(["scontrol", "requeue", job_id], check=False, timeout=60)
        except (subprocess.TimeoutExpired, OSError) as exc:
            print(f"[ResubmitManager] requeue failed: {exc}", flush=True)
