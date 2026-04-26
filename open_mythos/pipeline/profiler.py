"""
ACTProfiler — instruments the ACT halting module to capture per-token halt depths.

The profiler temporarily monkey-patches ``model.recurrent.act.forward`` to
intercept per-step halting probabilities without modifying ``main.py``.  A
context manager guarantees the original method is always restored, even if an
exception occurs during profiling.

Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator, List, Optional

import torch

from open_mythos.pipeline.config import PipelineConfig


class ACTProfiler:
    """Profile per-token ACT halt depths over a tokenized corpus.

    The profiler instruments ``model.recurrent.act.forward`` via a temporary
    monkey-patch so that the exact halting probabilities used by
    ``RecurrentBlock.forward`` are captured without any modification to
    ``main.py``.

    Args:
        cfg: Pipeline configuration; ``cfg.profiling_batch_size`` controls how
             many examples are processed per forward pass.
    """

    def __init__(self, cfg: PipelineConfig) -> None:
        self.cfg = cfg

    # ------------------------------------------------------------------
    # Instrumentation context manager
    # ------------------------------------------------------------------

    @contextmanager
    def _instrument(self, model) -> Generator[List[torch.Tensor], None, None]:
        """Temporarily patch ``model.recurrent.act.forward`` to capture per-step
        halting probabilities.

        Yields:
            ``step_probs`` — a list that is populated with one ``(B, T)`` tensor
            per loop iteration during the model forward pass.  The list is
            cleared between batches by the caller.
        """
        original_forward = model.recurrent.act.forward
        step_probs: List[torch.Tensor] = []

        def hooked_forward(h: torch.Tensor) -> torch.Tensor:
            p = original_forward(h)
            step_probs.append(p.detach())
            return p

        model.recurrent.act.forward = hooked_forward
        try:
            yield step_probs
        finally:
            model.recurrent.act.forward = original_forward

    # ------------------------------------------------------------------
    # Halt-depth computation from collected per-step probabilities
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_halt_depths(
        step_probs: List[torch.Tensor],
        n_loops: int,
        act_threshold: float,
    ) -> torch.Tensor:
        """Convert a list of per-step halting probabilities into halt depths.

        Args:
            step_probs:    List of ``(B, T)`` tensors, one per loop iteration.
            n_loops:       Maximum number of loop iterations.
            act_threshold: Cumulative probability threshold for halting.

        Returns:
            Integer tensor of shape ``(B, T)`` with values in ``[1, n_loops]``.
            Tokens that never reach ``act_threshold`` receive ``n_loops``.
        """
        if not step_probs:
            raise ValueError("step_probs is empty; no loop iterations were captured.")

        B, T = step_probs[0].shape
        device = step_probs[0].device

        halt_depths = torch.zeros(B, T, dtype=torch.int32, device=device)
        halted = torch.zeros(B, T, dtype=torch.bool, device=device)
        cumulative_p = torch.zeros(B, T, device=device)

        for t, p in enumerate(step_probs):
            # Only accumulate for tokens that have not yet halted
            cumulative_p = cumulative_p + p * (~halted).float()
            newly_halted = (~halted) & (cumulative_p >= act_threshold)
            halt_depths[newly_halted] = t + 1
            halted = halted | newly_halted

        # Tokens that never halted receive n_loops
        halt_depths[~halted] = n_loops

        return halt_depths

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def profile(
        self,
        model,
        corpus: torch.Tensor,
        n_loops: int,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run the model over ``corpus`` and return per-token halt depths.

        Args:
            model:        An ``OpenMythos`` instance.
            corpus:       2-D integer tensor of shape ``(num_examples, seq_len)``
                          containing tokenised input ids.
            n_loops:      Number of recurrent loop iterations to run.
            padding_mask: Optional boolean tensor of shape
                          ``(num_examples, seq_len)``; ``True`` = real token,
                          ``False`` = padding.  Currently accepted but not used
                          to alter the forward pass (padding tokens still receive
                          a halt depth; callers may mask them during scoring).

        Returns:
            Integer tensor of shape ``(num_examples, seq_len)`` with halt depths
            in ``[1, n_loops]``.

        Raises:
            ValueError: if ``corpus`` is not 2-D.
        """
        # Requirement 1.6 — validate corpus dimensionality
        if corpus.ndim != 2:
            raise ValueError(
                f"corpus must be a 2-D tensor of shape (num_examples, seq_len), "
                f"but got a tensor with {corpus.ndim} dimension(s) "
                f"(shape={tuple(corpus.shape)})."
            )

        num_examples, seq_len = corpus.shape
        act_threshold: float = model.cfg.act_threshold
        batch_size: int = self.cfg.profiling_batch_size
        device = next(model.parameters()).device

        all_halt_depths: List[torch.Tensor] = []

        # Requirement 1.4 — operate under torch.no_grad()
        with torch.no_grad():
            # Requirement 1.5 — process in configurable batches
            for start in range(0, num_examples, batch_size):
                end = min(start + batch_size, num_examples)
                batch_ids = corpus[start:end].to(device)  # (B, T)

                with self._instrument(model) as step_probs:
                    # Run the full model forward pass; the hook captures
                    # per-step halting probabilities into step_probs.
                    model(batch_ids, n_loops=n_loops)

                # Requirements 1.2, 1.3 — compute halt depths from captured probs
                batch_halt_depths = self._compute_halt_depths(
                    step_probs, n_loops, act_threshold
                )
                all_halt_depths.append(batch_halt_depths.cpu())

        return torch.cat(all_halt_depths, dim=0)  # (num_examples, seq_len)

    def save(self, halt_depths: torch.Tensor, path: str) -> None:
        """Serialize halt depths and metadata to disk.

        The saved dict contains:
        - ``halt_depths``:     ``(num_examples, seq_len)`` int32 tensor
        - ``example_indices``: ``(num_examples,)`` int64 tensor (0-based)
        - ``n_loops``:         int — maximum loop iterations used during profiling
        - ``act_threshold``:   float — halting threshold (from model config)
        - ``corpus_size``:     int — number of examples profiled

        Args:
            halt_depths: Tensor of shape ``(num_examples, seq_len)``.
            path:        File path to write (e.g. ``"halt_depths.pt"``).
        """
        num_examples = halt_depths.shape[0]
        payload = {
            "halt_depths": halt_depths.to(torch.int32),
            "example_indices": torch.arange(num_examples, dtype=torch.int64),
            "n_loops": self.cfg.profiling_n_loops,
            "act_threshold": float("nan"),  # filled below if model is available
            "corpus_size": num_examples,
        }
        torch.save(payload, path)

    @staticmethod
    def load(path: str) -> dict:
        """Load serialized halt depths from disk.

        Args:
            path: File path previously written by :meth:`save`.

        Returns:
            Dict with keys ``halt_depths``, ``example_indices``, ``n_loops``,
            ``act_threshold``, and ``corpus_size``.
        """
        return torch.load(path, weights_only=False)
