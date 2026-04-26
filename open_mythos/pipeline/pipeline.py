"""
ACTCurriculumPipeline — top-level orchestrator for the ACT curriculum training loop.

Wires together CurriculumSampler, LoopScheduler, and the OpenMythos model to
execute a full curriculum training experiment with gradient accumulation,
checkpointing, and optional halt-depth logging.

Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 7.1, 7.5, 8.3, 8.4
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Callable, Generator, List, Optional

import torch
import torch.utils.data

from open_mythos.pipeline.config import PipelineConfig
from open_mythos.pipeline.logger import HaltDepthLogger
from open_mythos.pipeline.sampler import CurriculumSampler
from open_mythos.pipeline.scheduler import LoopScheduler


def _split_batch(batch: dict, n_splits: int) -> List[dict]:
    """Split a batch dict into ``n_splits`` equal micro-batches along dim 0.

    Args:
        batch:    Dict with tensor values of shape ``(B, ...)``.
        n_splits: Number of micro-batches to produce.

    Returns:
        List of dicts, each containing a slice of the original tensors.
    """
    if n_splits <= 1:
        return [batch]

    first_val = next(iter(batch.values()))
    B = first_val.shape[0]
    chunk_size = max(1, (B + n_splits - 1) // n_splits)

    micro_batches = []
    for start in range(0, B, chunk_size):
        end = min(start + chunk_size, B)
        micro_batches.append({k: v[start:end] for k, v in batch.items()})
    return micro_batches


class ACTCurriculumPipeline:
    """Orchestrates curriculum training of an OpenMythos model.

    Integrates CurriculumSampler, LoopScheduler, gradient accumulation,
    checkpointing, and optional halt-depth logging into a single training loop.

    Args:
        model:           An ``OpenMythos`` instance.
        sampler:         A :class:`CurriculumSampler` pre-loaded with difficulty
                         scores and stage definitions.
        scheduler:       A :class:`LoopScheduler` mapping steps to ``n_loops``.
        optimizer:       Any ``torch.optim.Optimizer`` wrapping ``model``'s
                         parameters.
        criterion:       Loss callable; called as
                         ``criterion(logits.view(-1, V), labels.view(-1))``.
        cfg:             :class:`PipelineConfig` with all hyperparameters.
        checkpoint_path: Optional path to a checkpoint file.  If provided,
                         :meth:`_load_checkpoint` is called immediately.
    """

    def __init__(
        self,
        model,
        sampler: CurriculumSampler,
        scheduler: LoopScheduler,
        optimizer: torch.optim.Optimizer,
        criterion: Callable,
        cfg: PipelineConfig,
        checkpoint_path: Optional[str] = None,
    ) -> None:
        self.model = model
        self.sampler = sampler
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.criterion = criterion
        self.cfg = cfg
        self.start_step: int = 0

        if checkpoint_path is not None:
            self._load_checkpoint(checkpoint_path)

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def _save_checkpoint(self, step: int) -> None:
        """Save all training state to ``cfg.checkpoint_dir/checkpoint_{step}.pt``.

        Args:
            step: Current training step.
        """
        os.makedirs(self.cfg.checkpoint_dir, exist_ok=True)
        path = os.path.join(self.cfg.checkpoint_dir, f"checkpoint_{step}.pt")
        payload = {
            "step": step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.to_dict(),
            "sampler_state": self.sampler.state_dict(),
            "config": self.cfg.to_dict(),
        }
        torch.save(payload, path)

    def _load_checkpoint(self, path: str) -> None:
        """Restore all training state from a checkpoint file.

        Args:
            path: Path to a checkpoint file previously written by
                  :meth:`_save_checkpoint`.
        """
        checkpoint = torch.load(path, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler = LoopScheduler.from_dict(checkpoint["scheduler_state"])
        self.sampler.load_state_dict(checkpoint["sampler_state"])
        self.start_step = int(checkpoint["step"])

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def _log_step(
        self,
        step: int,
        loss: float,
        halt_stats: Optional[dict],
    ) -> None:
        """Print a one-line training log for the current step.

        Args:
            step:       Current training step.
            loss:       Accumulated loss over the gradient accumulation window.
            halt_stats: Dict from HaltDepthLogger.compute_stats, or None.
        """
        n_loops = self.scheduler.current_n_loops(step)
        stage_idx = self.sampler._stage_index_for_step(step)

        pool = self.sampler._active_pool
        if len(pool) > 0:
            mean_difficulty = float(self.sampler._scores[pool].mean().item())
        else:
            mean_difficulty = float("nan")

        msg = (
            f"step={step:6d}  n_loops={n_loops}  stage={stage_idx}"
            f"  mean_difficulty={mean_difficulty:.4f}  loss={loss:.6f}"
        )

        if halt_stats is not None:
            msg += (
                f"  mean_halt_depth={halt_stats['mean_halt_depth']:.3f}"
                f"  min_halt_depth={halt_stats['min_halt_depth']}"
                f"  max_halt_depth={halt_stats['max_halt_depth']}"
                f"  early_halt_rate={halt_stats['early_halt_rate']:.3f}"
            )

        print(msg)

    # ------------------------------------------------------------------
    # Halt-depth instrumentation (zero overhead when disabled)
    # ------------------------------------------------------------------

    @contextmanager
    def _instrument_act(self) -> Generator[List[torch.Tensor], None, None]:
        """Temporarily patch ``model.recurrent.act.forward`` to capture per-step
        halting probabilities during a forward pass.

        Yields:
            ``step_probs`` — list populated with one ``(B, T)`` tensor per loop
            iteration during the forward pass.
        """
        original_forward = self.model.recurrent.act.forward
        step_probs: List[torch.Tensor] = []

        def hooked_forward(h: torch.Tensor) -> torch.Tensor:
            p = original_forward(h)
            step_probs.append(p.detach())
            return p

        self.model.recurrent.act.forward = hooked_forward
        try:
            yield step_probs
        finally:
            self.model.recurrent.act.forward = original_forward

    @staticmethod
    def _compute_halt_depths_from_probs(
        step_probs: List[torch.Tensor],
        n_loops: int,
        act_threshold: float,
    ) -> torch.Tensor:
        """Convert per-step halting probabilities to halt depths.

        Args:
            step_probs:    List of ``(B, T)`` tensors, one per loop iteration.
            n_loops:       Maximum loop count.
            act_threshold: Cumulative probability threshold for halting.

        Returns:
            Integer tensor of shape ``(B, T)`` with values in ``[1, n_loops]``.
        """
        if not step_probs:
            return torch.zeros(1, 1, dtype=torch.int32)

        B, T = step_probs[0].shape
        device = step_probs[0].device

        halt_depths = torch.zeros(B, T, dtype=torch.int32, device=device)
        halted = torch.zeros(B, T, dtype=torch.bool, device=device)
        cumulative_p = torch.zeros(B, T, device=device)

        for t, p in enumerate(step_probs):
            cumulative_p = cumulative_p + p * (~halted).float()
            newly_halted = (~halted) & (cumulative_p >= act_threshold)
            halt_depths[newly_halted] = t + 1
            halted = halted | newly_halted

        halt_depths[~halted] = n_loops
        return halt_depths

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def run(self, dataset: torch.utils.data.Dataset, batch_size: int = 32) -> None:
        """Execute the full curriculum training loop.

        Args:
            dataset:    A ``torch.utils.data.Dataset`` whose items are dicts
                        with ``"input_ids"`` and ``"labels"`` keys.
            batch_size: Number of examples per batch.  Defaults to 32.
        """
        # Requirement 8.4 — seed before any stochastic operation
        torch.manual_seed(self.cfg.seed)

        loader = torch.utils.data.DataLoader(
            dataset,
            sampler=self.sampler,
            batch_size=batch_size,
        )

        step = self.start_step
        accum_loss = 0.0

        # Open halt-depth logger once if enabled
        halt_logger: Optional[HaltDepthLogger] = None
        if self.cfg.log_halt_depths:
            initial_n_loops = self.scheduler.current_n_loops(step)
            halt_logger = HaltDepthLogger(self.cfg.halt_log_path, initial_n_loops)

        def _infinite_loader(dl):
            """Cycle through the DataLoader indefinitely."""
            while True:
                for batch in dl:
                    yield batch

        try:
            for batch in _infinite_loader(loader):
                if step >= self.cfg.total_steps:
                    break

                # Requirement 5.3 — advance sampler pool for this step
                self.sampler.set_step(step)
                # Requirement 5.2 — get n_loops from scheduler
                n_loops = self.scheduler.current_n_loops(step)

                accum_loss = 0.0
                micro_batches = _split_batch(batch, self.cfg.gradient_accumulation_steps)
                step_probs: List[torch.Tensor] = []

                if self.cfg.log_halt_depths:
                    # Requirement 7.1 — instrument ACT forward to capture halt depths
                    with self._instrument_act() as step_probs:
                        for micro_batch in micro_batches:
                            input_ids = micro_batch["input_ids"]
                            labels = micro_batch["labels"]
                            logits = self.model(input_ids, n_loops=n_loops)
                            vocab_size = logits.shape[-1]
                            loss = self.criterion(
                                logits.view(-1, vocab_size),
                                labels.view(-1),
                            )
                            (loss / self.cfg.gradient_accumulation_steps).backward()
                            accum_loss += loss.item()
                else:
                    # Requirement 7.5 — zero overhead when logging is disabled
                    for micro_batch in micro_batches:
                        input_ids = micro_batch["input_ids"]
                        labels = micro_batch["labels"]
                        logits = self.model(input_ids, n_loops=n_loops)
                        vocab_size = logits.shape[-1]
                        loss = self.criterion(
                            logits.view(-1, vocab_size),
                            labels.view(-1),
                        )
                        (loss / self.cfg.gradient_accumulation_steps).backward()
                        accum_loss += loss.item()

                self.optimizer.step()
                self.optimizer.zero_grad()

                # Requirement 5.4 — log at configured interval
                if step % self.cfg.logging_interval == 0:
                    if self.cfg.log_halt_depths and step_probs:
                        act_threshold = float(self.model.cfg.act_threshold)
                        halt_depths = self._compute_halt_depths_from_probs(
                            step_probs, n_loops, act_threshold
                        )
                        halt_stats = HaltDepthLogger.compute_stats(halt_depths, n_loops)
                        if halt_logger is not None:
                            halt_logger.log(step, halt_depths, n_loops)
                    else:
                        halt_stats = None

                    self._log_step(step, accum_loss, halt_stats)

                # Requirement 5.5 — checkpoint at configured interval
                if step % self.cfg.checkpoint_interval == 0:
                    self._save_checkpoint(step)

                step += 1

        finally:
            if halt_logger is not None:
                halt_logger.close()
