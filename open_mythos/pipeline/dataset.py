"""
MathCorpusDataset — synthetic arithmetic/algebra token sequences.

Generates self-contained math reasoning sequences without requiring any
external downloads.  Each example is a tokenised arithmetic expression
with a step-by-step solution encoded as a flat token sequence.

Token vocabulary (128 tokens):
  0-9    : digits 0-9
  10     : '+'
  11     : '-'
  12     : '*'
  13     : '='
  14     : '?'  (unknown / query marker)
  15     : PAD
  16-127 : reserved / noise tokens

Difficulty levels (1-5) map to expression complexity:
  1 — single-digit addition:          3 + 4 = 7
  2 — two-digit addition:             23 + 41 = 64
  3 — multi-step addition chain:      3 + 4 + 5 + 2 = 14
  4 — mixed add/subtract:             23 - 7 + 15 - 4 = 27
  5 — nested multi-step with carries: 97 + 86 - 43 + 12 = 152

The model is trained as a language model (next-token prediction), so
labels = input_ids shifted by one position.
"""

from __future__ import annotations

import random
from typing import Optional

import torch
import torch.utils.data


# ---------------------------------------------------------------------------
# Vocabulary constants
# ---------------------------------------------------------------------------

PAD_ID = 15
DIGIT_OFFSET = 0   # digits 0-9 → token ids 0-9
PLUS_ID = 10
MINUS_ID = 11
MUL_ID = 12
EQ_ID = 13
QUERY_ID = 14
VOCAB_SIZE = 128


def _encode_number(n: int) -> list[int]:
    """Encode a non-negative integer as a list of digit token ids (MSD first)."""
    return [DIGIT_OFFSET + int(d) for d in str(abs(n))]


def _make_example(difficulty: int, rng: random.Random) -> list[int]:
    """Generate a tokenised math expression at the given difficulty level.

    Args:
        difficulty: Integer in [1, 5].
        rng:        Seeded random number generator.

    Returns:
        List of token ids representing the full expression including the answer.
    """
    if difficulty == 1:
        a = rng.randint(0, 9)
        b = rng.randint(0, 9)
        result = a + b
        tokens = _encode_number(a) + [PLUS_ID] + _encode_number(b) + [EQ_ID] + _encode_number(result)

    elif difficulty == 2:
        a = rng.randint(10, 99)
        b = rng.randint(10, 99)
        result = a + b
        tokens = _encode_number(a) + [PLUS_ID] + _encode_number(b) + [EQ_ID] + _encode_number(result)

    elif difficulty == 3:
        terms = [rng.randint(1, 9) for _ in range(rng.randint(3, 5))]
        result = sum(terms)
        tokens = []
        for i, t in enumerate(terms):
            tokens += _encode_number(t)
            if i < len(terms) - 1:
                tokens.append(PLUS_ID)
        tokens += [EQ_ID] + _encode_number(result)

    elif difficulty == 4:
        a = rng.randint(10, 99)
        b = rng.randint(1, 20)
        c = rng.randint(1, 30)
        d = rng.randint(1, 10)
        result = a - b + c - d
        tokens = (
            _encode_number(a) + [MINUS_ID] + _encode_number(b)
            + [PLUS_ID] + _encode_number(c)
            + [MINUS_ID] + _encode_number(d)
            + [EQ_ID] + _encode_number(abs(result))
        )

    else:  # difficulty == 5
        a = rng.randint(50, 99)
        b = rng.randint(50, 99)
        c = rng.randint(10, 50)
        d = rng.randint(1, 20)
        result = a + b - c + d
        tokens = (
            _encode_number(a) + [PLUS_ID] + _encode_number(b)
            + [MINUS_ID] + _encode_number(c)
            + [PLUS_ID] + _encode_number(d)
            + [EQ_ID] + _encode_number(abs(result))
        )

    return tokens


class MathCorpusDataset(torch.utils.data.Dataset):
    """Synthetic math corpus for ACT curriculum training experiments.

    Generates ``num_examples`` tokenised arithmetic expressions across five
    difficulty levels.  Each item is a dict with ``"input_ids"`` and
    ``"labels"`` (next-token prediction targets) tensors of length ``seq_len``.

    Args:
        num_examples: Total number of examples to generate.
        seq_len:      Fixed sequence length; examples are padded or truncated.
        difficulty_distribution: Probability for each difficulty level [1-5].
            Defaults to uniform.
        seed: Random seed for reproducibility.
    """

    VOCAB_SIZE = VOCAB_SIZE

    def __init__(
        self,
        num_examples: int = 1000,
        seq_len: int = 32,
        difficulty_distribution: Optional[list[float]] = None,
        seed: int = 42,
    ) -> None:
        self.seq_len = seq_len
        rng = random.Random(seed)

        if difficulty_distribution is None:
            difficulty_distribution = [0.2, 0.2, 0.2, 0.2, 0.2]

        assert len(difficulty_distribution) == 5
        assert abs(sum(difficulty_distribution) - 1.0) < 1e-6

        # Pre-generate all examples
        self._input_ids: list[torch.Tensor] = []
        self._labels: list[torch.Tensor] = []
        self._difficulties: list[int] = []

        for _ in range(num_examples):
            # Sample difficulty level
            r = rng.random()
            cumulative = 0.0
            diff = 1
            for level, prob in enumerate(difficulty_distribution, start=1):
                cumulative += prob
                if r <= cumulative:
                    diff = level
                    break

            tokens = _make_example(diff, rng)

            # Pad or truncate to seq_len
            if len(tokens) < seq_len:
                tokens = tokens + [PAD_ID] * (seq_len - len(tokens))
            else:
                tokens = tokens[:seq_len]

            ids = torch.tensor(tokens, dtype=torch.long)
            # Labels: shift by 1 for next-token prediction; last label is PAD
            labels = torch.cat([ids[1:], torch.tensor([PAD_ID], dtype=torch.long)])

            self._input_ids.append(ids)
            self._labels.append(labels)
            self._difficulties.append(diff)

    def __len__(self) -> int:
        return len(self._input_ids)

    def __getitem__(self, idx: int) -> dict:
        return {
            "input_ids": self._input_ids[idx],
            "labels": self._labels[idx],
        }

    @property
    def difficulties(self) -> list[int]:
        """Ground-truth difficulty level (1-5) for each example."""
        return self._difficulties

    def get_proof_tasks(
        self,
        n_tasks: int = 50,
        seed: int = 99,
    ) -> list:
        """Generate ProofTask instances for depth extrapolation evaluation.

        Each task is a math expression where the model must predict whether
        the answer token (EQ_ID followed by result digits) is correct.
        We use EQ_ID (token 13) as the verification token — a high logit
        for EQ_ID at the last position indicates the model "expects" an
        answer to follow, which correlates with correct reasoning.

        Args:
            n_tasks: Number of proof tasks to generate.
            seed:    Random seed.

        Returns:
            List of ProofTask instances.
        """
        from open_mythos.pipeline.evaluator import ProofTask

        rng = random.Random(seed)
        tasks = []
        for i in range(n_tasks):
            # Alternate valid (label=1) and invalid (label=0) tasks
            label = i % 2
            diff = rng.randint(3, 5)  # harder tasks for evaluation
            tokens = _make_example(diff, rng)

            if not label:
                # Corrupt the answer: flip the last digit
                if len(tokens) > 1:
                    last = tokens[-1]
                    tokens[-1] = DIGIT_OFFSET + ((last - DIGIT_OFFSET + 1) % 10)

            if len(tokens) < self.seq_len:
                tokens = tokens + [PAD_ID] * (self.seq_len - len(tokens))
            else:
                tokens = tokens[:self.seq_len]

            tasks.append(ProofTask(
                input_ids=torch.tensor(tokens, dtype=torch.long),
                label=label,
                verification_token_id=EQ_ID,
            ))

        return tasks
