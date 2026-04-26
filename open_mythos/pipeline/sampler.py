"""
CurriculumSampler — stage-based curriculum sampling for the ACT pipeline.

Implements Requirements 3.1–3.7:
  - Stage-based active pool management (cumulative curriculum)
  - Uniform and competence-weighted sampling modes
  - PyTorch DataLoader compatibility as a Sampler subclass
  - Checkpoint state_dict / load_state_dict support
"""

from __future__ import annotations

from typing import Iterator

import torch
import torch.utils.data

from open_mythos.pipeline.config import StageConfig


class CurriculumSampler(torch.utils.data.Sampler):
    """Samples training examples according to a curriculum defined by difficulty stages.

    The sampler maintains an *active pool* of example indices whose difficulty
    scores fall within the cumulative percentile range of the current stage.
    At each training step the caller must invoke :meth:`set_step` to advance
    the pool before iterating.

    Args:
        scores: Difficulty scores for each example, shape ``(num_examples,)``,
            values in ``[0.0, 1.0]``.  Higher = harder.
        stages: Ordered list of :class:`~open_mythos.pipeline.config.StageConfig`
            objects defining the curriculum progression.
        sampling_mode: ``"uniform"`` for uniform random sampling within the
            active pool, or ``"weighted"`` for competence-based sampling where
            ``w[i] = 1 - scores[i]`` (easier examples drawn more often).
        seed: Random seed for reproducibility.

    Raises:
        ValueError: If ``sampling_mode`` is not ``"uniform"`` or ``"weighted"``.
    """

    def __init__(
        self,
        scores: torch.Tensor,
        stages: list[StageConfig],
        sampling_mode: str = "uniform",
        seed: int = 42,
    ) -> None:
        if sampling_mode not in ("uniform", "weighted"):
            raise ValueError(
                f"sampling_mode must be 'uniform' or 'weighted', got {sampling_mode!r}"
            )

        self._scores = scores.float()
        self._stages = stages
        self._sampling_mode = sampling_mode
        self._seed = seed

        # Build cumulative step boundaries: [0, s0, s0+s1, s0+s1+s2, ...]
        self._cumulative_steps: list[int] = [0]
        for stage in stages:
            self._cumulative_steps.append(self._cumulative_steps[-1] + stage.max_steps)

        # Active pool — populated by set_step()
        self._active_pool: torch.Tensor = torch.empty(0, dtype=torch.long)
        self._current_step: int = 0

        # Seeded generator for reproducible sampling
        self._generator = torch.Generator()
        self._generator.manual_seed(seed)

    # ------------------------------------------------------------------
    # Stage resolution helpers
    # ------------------------------------------------------------------

    def _stage_index_for_step(self, step: int) -> int:
        """Return the index of the active stage for *step* (0-based)."""
        stage_idx = 0
        for i, cum in enumerate(self._cumulative_steps):
            if cum <= step:
                stage_idx = i
            else:
                break
        # Clamp to valid stage range
        return min(stage_idx, len(self._stages) - 1)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_step(self, step: int) -> None:
        """Update the active pool for *step*.

        Must be called before iterating the sampler at a new training step.

        Args:
            step: Current training step (0-based).

        Raises:
            RuntimeError: If the resulting active pool is empty.
        """
        self._current_step = step
        pool = self.get_active_pool(step)
        if len(pool) == 0:
            stage_idx = self._stage_index_for_step(step)
            raise RuntimeError(
                f"Empty pool at step {step}, stage {stage_idx}"
            )
        self._active_pool = pool

    def get_active_pool(self, step: int) -> torch.Tensor:
        """Return indices of examples in the active pool for *step*.

        Implements the cumulative curriculum: all examples whose score is at or
        below the ``high_percentile`` of the current stage are included.

        Args:
            step: Training step to compute the pool for.

        Returns:
            1-D ``LongTensor`` of example indices in the active pool.
        """
        stage_idx = self._stage_index_for_step(step)
        active_stage = self._stages[stage_idx]

        # Percentile thresholds
        low_threshold = torch.quantile(self._scores, 0.0).item()
        high_threshold = torch.quantile(
            self._scores, float(active_stage.high_percentile)
        ).item()

        mask = (self._scores >= low_threshold) & (self._scores <= high_threshold)
        return torch.where(mask)[0]

    # ------------------------------------------------------------------
    # Sampler protocol
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[int]:
        """Yield indices from the active pool in sampled order.

        Sampling strategy is determined by ``sampling_mode``:

        * ``"uniform"`` — each index in the pool is equally likely.
        * ``"weighted"`` — index ``i`` is drawn with probability proportional
          to ``1 - scores[i]`` (easier examples are sampled more often).

        Yields:
            Integer example indices.
        """
        pool = self._active_pool
        n = len(pool)
        if n == 0:
            return

        if self._sampling_mode == "uniform":
            perm = torch.randperm(n, generator=self._generator)
            for idx in perm:
                yield int(pool[idx].item())
        else:
            # Competence-weighted: w[i] = 1 - scores[i]
            weights = 1.0 - self._scores[pool]
            # Guard against all-zero weights (all scores == 1.0)
            if weights.sum() == 0:
                weights = torch.ones(n)
            sampled = torch.multinomial(
                weights,
                num_samples=n,
                replacement=True,
                generator=self._generator,
            )
            for idx in sampled:
                yield int(pool[idx].item())

    def __len__(self) -> int:
        """Return the size of the current active pool."""
        return len(self._active_pool)

    # ------------------------------------------------------------------
    # Checkpoint support
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        """Serialize sampler state for checkpoint saving.

        Returns:
            A plain Python dict containing all state needed to resume sampling
            from the same position.
        """
        return {
            "current_step": self._current_step,
            "active_pool": self._active_pool.tolist(),
            "generator_state": self._generator.get_state().tolist(),
            "sampling_mode": self._sampling_mode,
            "seed": self._seed,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore sampler state from a checkpoint.

        Args:
            state: Dict previously returned by :meth:`state_dict`.
        """
        self._current_step = int(state["current_step"])
        self._active_pool = torch.tensor(state["active_pool"], dtype=torch.long)
        gen_state = torch.tensor(state["generator_state"], dtype=torch.uint8)
        self._generator.set_state(gen_state)
        self._sampling_mode = str(state["sampling_mode"])
        self._seed = int(state["seed"])
