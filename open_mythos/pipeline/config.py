"""
PipelineConfig and StageConfig dataclasses for the ACT Curriculum Pipeline.

All hyperparameters for the entire experiment live here.  The config is
serializable to/from a plain Python dict so it can be saved as JSON and
restored without loss of information (Requirement 8.1, 8.2, 8.3, 8.5).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class StageConfig:
    """Defines one curriculum training stage.

    Attributes:
        low_percentile:  lower bound of difficulty percentile range [0, 1)
        high_percentile: upper bound of difficulty percentile range (0, 1]
        max_steps:       number of training steps allocated to this stage
    """

    low_percentile: float
    high_percentile: float
    max_steps: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "low_percentile": self.low_percentile,
            "high_percentile": self.high_percentile,
            "max_steps": self.max_steps,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "StageConfig":
        return cls(
            low_percentile=float(d["low_percentile"]),
            high_percentile=float(d["high_percentile"]),
            max_steps=int(d["max_steps"]),
        )


@dataclass
class PipelineConfig:
    """Central configuration for the ACT Curriculum Proof Extrapolation pipeline.

    Validation (Requirement 8.5):
        ``eval_n_loops`` must contain at least one value strictly greater than
        the maximum ``n_loops`` in ``loop_schedule``.  If this condition is not
        met, ``__post_init__`` raises ``ValueError`` because no depth-
        extrapolation evaluation would occur.

    Serialization (Requirements 8.1, 8.2, 8.3):
        ``to_dict()`` / ``from_dict()`` provide a lossless JSON round-trip.
        ``stages`` is serialized as a list of dicts; ``loop_schedule`` is
        serialized as a list of two-element lists (JSON arrays) and restored
        as a list of tuples.
    """

    # ------------------------------------------------------------------
    # Profiling
    # ------------------------------------------------------------------
    profiling_batch_size: int = 32
    profiling_n_loops: int = 8

    # ------------------------------------------------------------------
    # Difficulty scoring
    # ------------------------------------------------------------------
    difficulty_mode: str = "mean"  # "mean" | "p90"

    # ------------------------------------------------------------------
    # Curriculum stages
    # ------------------------------------------------------------------
    stages: list[StageConfig] = field(
        default_factory=lambda: [
            StageConfig(0.0, 0.33, 1000),
            StageConfig(0.0, 0.66, 2000),
            StageConfig(0.0, 1.00, 3000),
        ]
    )

    # ------------------------------------------------------------------
    # Loop depth schedule: list of (step_threshold, n_loops) pairs
    # ------------------------------------------------------------------
    loop_schedule: list[tuple[int, int]] = field(
        default_factory=lambda: [(0, 4), (1000, 8), (3000, 16)]
    )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    total_steps: int = 6000
    logging_interval: int = 100
    checkpoint_interval: int = 500
    gradient_accumulation_steps: int = 1
    seed: int = 42

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    eval_n_loops: list[int] = field(default_factory=lambda: [8, 16, 32])

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    halt_depth_path: str = "halt_depths.pt"
    scores_path: str = "scores.pt"
    checkpoint_dir: str = "checkpoints/"
    results_path: str = "results.json"
    halt_log_path: str = "halt_log.jsonl"

    # ------------------------------------------------------------------
    # Halt depth logging
    # ------------------------------------------------------------------
    log_halt_depths: bool = True

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        """Validate that at least one eval_n_loops value exceeds the schedule max.

        Requirement 8.5: eval_n_loops must contain at least one value greater
        than the maximum n_loops in loop_schedule so that depth-extrapolation
        evaluation actually occurs.

        Raises:
            ValueError: if no eval_n_loops value exceeds the schedule maximum.
        """
        max_schedule_loops = max(n for _, n in self.loop_schedule)
        if not any(n > max_schedule_loops for n in self.eval_n_loops):
            raise ValueError(
                f"eval_n_loops must contain at least one value greater than "
                f"the maximum n_loops in the schedule ({max_schedule_loops}). "
                f"Got eval_n_loops={self.eval_n_loops}."
            )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain Python dict suitable for JSON encoding.

        ``stages`` → list of dicts via ``StageConfig.to_dict()``.
        ``loop_schedule`` → list of two-element lists (JSON-safe; tuples are
        not natively JSON-serializable).

        Returns:
            A dict that can be passed to ``json.dumps`` without further
            conversion and round-tripped losslessly via ``from_dict``.
        """
        return {
            "profiling_batch_size": self.profiling_batch_size,
            "profiling_n_loops": self.profiling_n_loops,
            "difficulty_mode": self.difficulty_mode,
            "stages": [s.to_dict() for s in self.stages],
            "loop_schedule": [list(pair) for pair in self.loop_schedule],
            "total_steps": self.total_steps,
            "logging_interval": self.logging_interval,
            "checkpoint_interval": self.checkpoint_interval,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "seed": self.seed,
            "eval_n_loops": list(self.eval_n_loops),
            "halt_depth_path": self.halt_depth_path,
            "scores_path": self.scores_path,
            "checkpoint_dir": self.checkpoint_dir,
            "results_path": self.results_path,
            "halt_log_path": self.halt_log_path,
            "log_halt_depths": self.log_halt_depths,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PipelineConfig":
        """Deserialize from a plain Python dict (e.g. loaded from JSON).

        ``stages`` is restored from a list of dicts via ``StageConfig.from_dict``.
        ``loop_schedule`` is restored from a list of two-element lists/tuples
        and converted back to a list of ``(int, int)`` tuples.

        Args:
            d: dict previously produced by ``to_dict()`` (or equivalent JSON).

        Returns:
            A fully validated ``PipelineConfig`` instance.
        """
        return cls(
            profiling_batch_size=int(d["profiling_batch_size"]),
            profiling_n_loops=int(d["profiling_n_loops"]),
            difficulty_mode=str(d["difficulty_mode"]),
            stages=[StageConfig.from_dict(s) for s in d["stages"]],
            loop_schedule=[tuple(pair) for pair in d["loop_schedule"]],  # type: ignore[misc]
            total_steps=int(d["total_steps"]),
            logging_interval=int(d["logging_interval"]),
            checkpoint_interval=int(d["checkpoint_interval"]),
            gradient_accumulation_steps=int(d["gradient_accumulation_steps"]),
            seed=int(d["seed"]),
            eval_n_loops=[int(n) for n in d["eval_n_loops"]],
            halt_depth_path=str(d["halt_depth_path"]),
            scores_path=str(d["scores_path"]),
            checkpoint_dir=str(d["checkpoint_dir"]),
            results_path=str(d["results_path"]),
            halt_log_path=str(d["halt_log_path"]),
            log_halt_depths=bool(d["log_halt_depths"]),
        )
