"""
DifficultyScorer — aggregates per-token halt depths into per-example difficulty
scores and normalizes them to [0.0, 1.0].

Implements Requirements 2.1 – 2.6.
"""

from __future__ import annotations

from typing import Optional

import torch


class DifficultyScorer:
    """Converts a halt-depth tensor into normalized per-example difficulty scores.

    Args:
        mode: Aggregation strategy over non-padding tokens.
              ``"mean"`` — arithmetic mean of halt depths.
              ``"p90"``  — 90th-percentile of halt depths.

    Raises:
        ValueError: if *mode* is not ``"mean"`` or ``"p90"``.
    """

    def __init__(self, mode: str = "mean") -> None:
        if mode not in ("mean", "p90"):
            raise ValueError(
                f"mode must be 'mean' or 'p90', got {mode!r}"
            )
        self.mode = mode

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(
        self,
        halt_depths: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Aggregate halt depths and normalize to [0.0, 1.0].

        Args:
            halt_depths:  Integer tensor of shape ``(num_examples, seq_len)``.
                          Each value is the halt depth for that token.
            padding_mask: Boolean tensor of shape ``(num_examples, seq_len)``.
                          ``True`` marks a real token; ``False`` marks padding.
                          When *None*, all positions are treated as real tokens.

        Returns:
            Float tensor of shape ``(num_examples,)`` with values in
            ``[0.0, 1.0]``.  The hardest example receives score ``1.0``;
            all others are proportionally scaled (Requirement 2.4).

        Raises:
            ValueError: if *halt_depths* is not 2-D.
        """
        if halt_depths.dim() != 2:
            raise ValueError(
                f"halt_depths must be 2-D (num_examples, seq_len), "
                f"got shape {tuple(halt_depths.shape)}"
            )

        num_examples = halt_depths.size(0)
        depths_float = halt_depths.float()

        # Build a default all-True mask when none is provided.
        if padding_mask is None:
            padding_mask = torch.ones_like(halt_depths, dtype=torch.bool)

        raw_scores = torch.zeros(num_examples, dtype=torch.float32)

        for i in range(num_examples):
            valid = depths_float[i][padding_mask[i]]  # exclude padding (Req 2.3)
            if valid.numel() == 0:
                # Degenerate case: all tokens are padding — score 0.
                raw_scores[i] = 0.0
            elif self.mode == "mean":
                raw_scores[i] = valid.mean()
            else:  # p90
                raw_scores[i] = torch.quantile(valid, 0.90)

        # Normalize to [0, 1] by dividing by corpus-wide maximum (Req 2.4).
        max_score = raw_scores.max()
        if max_score == 0.0:
            # All examples have zero halt depth — return zeros.
            return raw_scores

        scores = raw_scores / max_score
        return scores

    def save(
        self,
        scores: torch.Tensor,
        indices: torch.Tensor,
        path: str,
    ) -> None:
        """Serialize scores and example indices to disk.

        The file is a ``torch.save`` dict with keys ``"scores"``,
        ``"indices"``, and ``"mode"`` (Requirement 2.6).

        Args:
            scores:  Float tensor of shape ``(num_examples,)`` in ``[0, 1]``.
            indices: Int64 tensor of shape ``(num_examples,)`` with example ids.
            path:    Destination file path (e.g. ``"scores.pt"``).
        """
        torch.save(
            {
                "scores": scores.to(torch.float32),
                "indices": indices.to(torch.int64),
                "mode": self.mode,
            },
            path,
        )

    @staticmethod
    def load(path: str) -> dict:
        """Load serialized scores from disk.

        Args:
            path: Path to a file previously written by :meth:`save`.

        Returns:
            A dict with keys ``"scores"`` (float32 Tensor) and
            ``"indices"`` (int64 Tensor).
        """
        data = torch.load(path, weights_only=True)
        return {"scores": data["scores"], "indices": data["indices"]}
