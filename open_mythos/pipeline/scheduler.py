"""
LoopScheduler: maps training step → n_loops using a piecewise-constant schedule.

Implements Requirements 4.1–4.6.
"""

from __future__ import annotations

from typing import Any


class LoopScheduler:
    """Maps a training step to the number of recurrent loops to use.

    The schedule is a list of ``(step_threshold, n_loops)`` pairs in strictly
    ascending order of ``step_threshold``.  At each training step the scheduler
    returns the ``n_loops`` value of the highest threshold that does not exceed
    the current step (piecewise-constant / staircase function).

    Example::

        scheduler = LoopScheduler([(0, 4), (1000, 8), (3000, 16)])
        scheduler.current_n_loops(0)    # → 4
        scheduler.current_n_loops(999)  # → 4
        scheduler.current_n_loops(1000) # → 8
        scheduler.current_n_loops(3000) # → 16

    Args:
        schedule: list of ``(step_threshold, n_loops)`` pairs in ascending
                  order of ``step_threshold``.

    Raises:
        ValueError: if ``schedule`` is empty, if any ``n_loops`` value is not a
                    positive integer, or if ``step_threshold`` values are not
                    non-negative integers in strictly ascending order.
    """

    def __init__(self, schedule: list[tuple[int, int]]) -> None:
        if not schedule:
            raise ValueError("schedule must contain at least one entry.")

        # Validate each entry and collect thresholds for ordering check.
        prev_threshold: int | None = None
        for i, (threshold, n_loops) in enumerate(schedule):
            # --- step_threshold validation ---
            if not isinstance(threshold, int) or isinstance(threshold, bool):
                raise ValueError(
                    f"schedule[{i}]: step_threshold must be a non-negative integer, "
                    f"got {threshold!r}."
                )
            if threshold < 0:
                raise ValueError(
                    f"schedule[{i}]: step_threshold must be non-negative, "
                    f"got {threshold}."
                )
            if prev_threshold is not None and threshold <= prev_threshold:
                raise ValueError(
                    f"schedule[{i}]: step_threshold values must be strictly "
                    f"ascending; {threshold} is not greater than {prev_threshold}."
                )
            prev_threshold = threshold

            # --- n_loops validation ---
            if not isinstance(n_loops, int) or isinstance(n_loops, bool):
                raise ValueError(
                    f"schedule[{i}]: n_loops must be a positive integer, "
                    f"got {n_loops!r}."
                )
            if n_loops <= 0:
                raise ValueError(
                    f"schedule[{i}]: n_loops must be a positive integer, "
                    f"got {n_loops}."
                )

        # Store as a tuple of tuples for immutability.
        self._schedule: tuple[tuple[int, int], ...] = tuple(
            (int(t), int(n)) for t, n in schedule
        )

    # ------------------------------------------------------------------
    # Core lookup
    # ------------------------------------------------------------------

    def current_n_loops(self, step: int) -> int:
        """Return the ``n_loops`` value for the given training step.

        Uses a piecewise-constant (staircase) lookup: iterates through all
        schedule entries and keeps updating the result whenever the current
        step meets or exceeds a threshold.  The last matching entry wins,
        which corresponds to the highest threshold ≤ step.

        Args:
            step: current training step (non-negative integer).

        Returns:
            The ``n_loops`` value active at ``step``.
        """
        result = self._schedule[0][1]  # default: first entry's n_loops
        for threshold, n_loops in self._schedule:
            if step >= threshold:
                result = n_loops
        return result

    # ------------------------------------------------------------------
    # Serialization (Requirement 4.6)
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain Python dict for checkpoint storage.

        The schedule is stored as a list of two-element lists so the result
        is directly JSON-serializable.

        Returns:
            ``{"schedule": [[threshold, n_loops], ...]}``
        """
        return {
            "schedule": [list(pair) for pair in self._schedule],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LoopScheduler":
        """Deserialize from a plain Python dict (e.g. loaded from a checkpoint).

        Args:
            d: dict previously produced by ``to_dict()``.

        Returns:
            A fully validated ``LoopScheduler`` instance.
        """
        schedule = [tuple(pair) for pair in d["schedule"]]
        return cls(schedule)  # type: ignore[arg-type]
