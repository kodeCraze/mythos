"""
DepthExtrapolationEvaluator and ProofTask for the ACT Curriculum Pipeline.

Evaluates a trained model at multiple loop depths (including depths beyond
those seen during training) to measure depth-extrapolation capability.

Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

import torch

from open_mythos.pipeline.config import PipelineConfig


@dataclass
class ProofTask:
    """A formal proof verification example.

    Attributes:
        input_ids:             (seq_len,) int64 token ids representing the premise.
        label:                 0 = invalid proof step, 1 = valid proof step.
        verification_token_id: vocab index of the token whose logit determines
                               the classification (score > 0 → predicted valid).
    """

    input_ids: torch.Tensor        # (seq_len,) int64 token ids
    label: int                     # 0 = invalid, 1 = valid proof step
    verification_token_id: int     # vocab index of the verification token


class DepthExtrapolationEvaluator:
    """Evaluates a trained model at multiple loop depths.

    Runs the model at each value in ``eval_n_loops`` and computes binary
    classification accuracy on a list of :class:`ProofTask` instances.
    Classification is based on the sign of the logit at the last sequence
    position for each task's ``verification_token_id``.

    Requirements: 6.1 – 6.7
    """

    def __init__(self, cfg: PipelineConfig) -> None:
        """
        Args:
            cfg: Pipeline configuration; ``cfg.eval_n_loops`` is used as the
                 default list of loop depths to evaluate.
        """
        self.cfg = cfg

    @torch.no_grad()
    def evaluate(
        self,
        model,
        tasks: list,
        eval_n_loops: Optional[list] = None,
    ) -> list:
        """Evaluate the model on proof tasks at each specified loop depth.

        For each ``n_loops`` value the model is called with that value and each
        task is classified as valid (1) when the verification-token logit at the
        last position is positive, and invalid (0) otherwise.  Accuracy is
        ``correct / total`` for each depth.

        The decorator ``@torch.no_grad()`` ensures no gradients are stored
        during evaluation (Requirement 6.3).

        Args:
            model:        An OpenMythos instance.
            tasks:        List of ProofTask instances to evaluate.
            eval_n_loops: Loop depths to evaluate at.  Defaults to
                          ``cfg.eval_n_loops`` when ``None``.

        Returns:
            A list of dicts, one per loop depth, each with keys:
            ``"n_loops"`` (int) and ``"accuracy"`` (float in [0.0, 1.0]).

        Raises:
            ValueError: if ``tasks`` is empty.
        """
        if not tasks:
            raise ValueError("tasks must be non-empty")

        depths = eval_n_loops if eval_n_loops is not None else self.cfg.eval_n_loops

        results: list = []

        for n in depths:
            correct = 0
            total = len(tasks)

            for task in tasks:
                # (1, seq_len, vocab_size) — Requirement 6.2, 6.6
                logits = model(task.input_ids.unsqueeze(0), n_loops=n)

                # Score is the logit at the last position for the verification token
                score = logits[0, -1, task.verification_token_id]

                prediction = 1 if score > 0 else 0
                if prediction == task.label:
                    correct += 1

            accuracy = correct / total  # Requirement 6.4
            results.append({"n_loops": n, "accuracy": accuracy})

        return results  # Requirement 6.5

    def save_results(self, results: list, path: str) -> None:
        """Serialize evaluation results to a JSON file.

        Args:
            results: List of dicts as returned by :meth:`evaluate`.
            path:    Destination file path.
        """
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)

    @staticmethod
    def load_results(path: str) -> list:
        """Load evaluation results from a JSON file.

        Args:
            path: Path to a JSON file previously written by :meth:`save_results`.

        Returns:
            List of dicts with ``"n_loops"`` and ``"accuracy"`` keys.
        """
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
