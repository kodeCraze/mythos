"""
HaltDepthLogger — structured JSONL logging of halt depth statistics.

Separated from the pipeline to keep concerns clean (Requirement 7.2, 7.3, 7.4).
"""

from __future__ import annotations

import json

import torch


class HaltDepthLogger:
    """Logs per-step halt depth statistics to a JSONL file.

    Each call to ``log()`` appends one JSON object to the file, recording
    the mean, min, max halt depth and the early-halt rate for the batch.

    Args:
        log_path: Path to the JSONL file (opened for appending).
        n_loops:  Default maximum loop count; stored for reference but
                  ``log()`` accepts a per-call override.
    """

    def __init__(self, log_path: str, n_loops: int) -> None:
        self.log_path = log_path
        self.n_loops = n_loops
        # Open in append mode so multiple runs accumulate in the same file.
        self._file = open(log_path, "a", encoding="utf-8")  # noqa: WPS515

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log(
        self,
        step: int,
        halt_depths: torch.Tensor,  # (B, T) int
        n_loops: int,
    ) -> None:
        """Compute statistics and write one JSON line to the log file.

        Args:
            step:        Current training step.
            halt_depths: Integer tensor of shape ``(B, T)`` containing the
                         loop iteration at which each token halted.
            n_loops:     Maximum loop count used for this step (used to
                         compute the early-halt rate).
        """
        stats = self.compute_stats(halt_depths, n_loops)
        record = {"step": step, "n_loops": n_loops, **stats}
        self._file.write(json.dumps(record) + "\n")
        self._file.flush()

    @staticmethod
    def compute_stats(halt_depths: torch.Tensor, n_loops: int) -> dict:
        """Compute summary statistics over a batch of halt depths.

        Args:
            halt_depths: Integer tensor of shape ``(B, T)``.
            n_loops:     Maximum loop count; tokens with depth ``< n_loops``
                         are counted as early-halted.

        Returns:
            A dict with keys:
            - ``mean_halt_depth``  (float)
            - ``min_halt_depth``   (int)
            - ``max_halt_depth``   (int)
            - ``early_halt_rate``  (float) — fraction of tokens with depth < n_loops
        """
        flat = halt_depths.flatten().float()
        mean_halt_depth = float(flat.mean().item())
        min_halt_depth = int(halt_depths.min().item())
        max_halt_depth = int(halt_depths.max().item())
        early_halt_rate = float((halt_depths < n_loops).float().mean().item())

        return {
            "mean_halt_depth": mean_halt_depth,
            "min_halt_depth": min_halt_depth,
            "max_halt_depth": max_halt_depth,
            "early_halt_rate": early_halt_rate,
        }

    # ------------------------------------------------------------------
    # Resource management
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Flush and close the underlying log file."""
        self._file.close()

    def __enter__(self) -> "HaltDepthLogger":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
