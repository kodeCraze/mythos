"""
GSM8K + MATH Dataset loader for ACT Curriculum Proof Extrapolation research.

Uses real benchmark datasets:
  - GSM8K (Cobbe et al., 2021): 8.5k grade-school math word problems
    https://huggingface.co/datasets/openai/gsm8k
  - MATH (Hendrycks et al., 2021): 12.5k competition problems, 5 difficulty levels
    https://huggingface.co/datasets/hendrycks/competition_math

Tokenization: character-level BPE via tiktoken (cl100k_base, same as GPT-4).
This gives a 100k vocab and handles math notation, LaTeX, and code naturally.

Fallback: if datasets/tiktoken not installed, falls back to the synthetic
MathCorpusDataset so the pipeline always runs.
"""

from __future__ import annotations

import random
from typing import Optional

import torch
import torch.utils.data

# ---------------------------------------------------------------------------
# Try to import real dataset dependencies
# ---------------------------------------------------------------------------
try:
    import tiktoken
    _TIKTOKEN_AVAILABLE = True
except ImportError:
    _TIKTOKEN_AVAILABLE = False

try:
    from datasets import load_dataset
    _DATASETS_AVAILABLE = True
except ImportError:
    _DATASETS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Tokenizer wrapper
# ---------------------------------------------------------------------------

class MathTokenizer:
    """Thin wrapper around tiktoken cl100k_base for math text."""

    PAD_ID: int = 0   # tiktoken has no pad; we use 0 and mask it in loss

    def __init__(self):
        if not _TIKTOKEN_AVAILABLE:
            raise ImportError("tiktoken not installed. Run: pip install tiktoken")
        self._enc = tiktoken.get_encoding("cl100k_base")
        self.vocab_size: int = self._enc.n_vocab  # 100277

    def encode(self, text: str, max_len: int) -> list[int]:
        ids = self._enc.encode(text, disallowed_special=())
        if len(ids) >= max_len:
            ids = ids[:max_len]
        else:
            ids = ids + [self.PAD_ID] * (max_len - len(ids))
        return ids

    def decode(self, ids: list[int]) -> str:
        return self._enc.decode([i for i in ids if i != self.PAD_ID])


# ---------------------------------------------------------------------------
# GSM8K Dataset
# ---------------------------------------------------------------------------

class GSM8KDataset(torch.utils.data.Dataset):
    """GSM8K grade-school math word problems.

    Each example is a concatenation of:
        "Question: {question}\\nAnswer: {answer}"

    tokenised with tiktoken cl100k_base and padded/truncated to seq_len.

    Args:
        split:   "train" (7473 examples) or "test" (1319 examples)
        seq_len: Fixed sequence length for padding/truncation
        seed:    Random seed for shuffling
    """

    def __init__(
        self,
        split: str = "train",
        seq_len: int = 256,
        seed: int = 42,
    ):
        if not _DATASETS_AVAILABLE:
            raise ImportError("datasets not installed. Run: pip install datasets")
        if not _TIKTOKEN_AVAILABLE:
            raise ImportError("tiktoken not installed. Run: pip install tiktoken")

        self.seq_len = seq_len
        self.tokenizer = MathTokenizer()
        self.vocab_size = self.tokenizer.vocab_size

        raw = load_dataset("openai/gsm8k", "main", split=split)
        rng = random.Random(seed)
        indices = list(range(len(raw)))
        rng.shuffle(indices)

        self._input_ids: list[torch.Tensor] = []
        self._labels: list[torch.Tensor] = []
        self._difficulties: list[int] = []

        for idx in indices:
            item = raw[idx]
            # Format: question + answer chain-of-thought
            text = f"Question: {item['question']}\nAnswer: {item['answer']}"
            ids = self.tokenizer.encode(text, seq_len)
            t = torch.tensor(ids, dtype=torch.long)
            # Next-token prediction labels
            labels = torch.cat([t[1:], torch.tensor([self.tokenizer.PAD_ID])])
            self._input_ids.append(t)
            self._labels.append(labels)
            # Difficulty proxy: answer length (longer = more steps = harder)
            ans_len = len(item['answer'])
            diff = min(5, max(1, ans_len // 100 + 1))
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
        return self._difficulties


# ---------------------------------------------------------------------------
# MATH Dataset (competition problems, 5 difficulty levels)
# ---------------------------------------------------------------------------

class MATHDataset(torch.utils.data.Dataset):
    """Hendrycks MATH competition dataset.

    5 difficulty levels (1=easiest, 5=hardest) across 7 subject areas.
    Used for depth extrapolation evaluation — harder problems should benefit
    more from additional recurrent loops.

    Args:
        split:   "train" (7500 examples) or "test" (5000 examples)
        seq_len: Fixed sequence length
        subjects: Optional list of subjects to filter (e.g. ["algebra"])
        seed:    Random seed
    """

    SUBJECTS = [
        "algebra", "counting_and_probability", "geometry",
        "intermediate_algebra", "number_theory", "prealgebra", "precalculus"
    ]

    def __init__(
        self,
        split: str = "test",
        seq_len: int = 256,
        subjects: Optional[list[str]] = None,
        seed: int = 42,
    ):
        if not _DATASETS_AVAILABLE:
            raise ImportError("datasets not installed. Run: pip install datasets")
        if not _TIKTOKEN_AVAILABLE:
            raise ImportError("tiktoken not installed. Run: pip install tiktoken")

        self.seq_len = seq_len
        self.tokenizer = MathTokenizer()
        self.vocab_size = self.tokenizer.vocab_size

        # qwedsacf/competition_math is the standard Parquet mirror of the
        # Hendrycks MATH dataset (12.5k problems, no loading script required).
        # It only has a 'train' split — we do an 80/20 manual split.
        full = load_dataset("qwedsacf/competition_math", split="train")
        rng_split = random.Random(seed + 1)
        indices = list(range(len(full)))
        rng_split.shuffle(indices)
        if split == "train":
            indices = indices[:int(0.8 * len(indices))]
        else:  # test
            indices = indices[int(0.8 * len(indices)):]
        raw = [full[i] for i in indices]
        rng = random.Random(seed)

        self._input_ids: list[torch.Tensor] = []
        self._labels: list[torch.Tensor] = []
        self._difficulties: list[int] = []
        self._subjects: list[str] = []

        items = list(raw)
        rng.shuffle(items)

        for item in items:
            if subjects and item.get("type") not in subjects:
                continue
            text = f"Problem: {item['problem']}\nSolution: {item['solution']}"
            ids = self.tokenizer.encode(text, seq_len)
            t = torch.tensor(ids, dtype=torch.long)
            labels = torch.cat([t[1:], torch.tensor([self.tokenizer.PAD_ID])])
            self._input_ids.append(t)
            self._labels.append(labels)
            # MATH has explicit difficulty levels "Level 1" through "Level 5"
            level_str = item.get("level", "Level 3")
            try:
                diff = int(str(level_str).split()[-1])
                diff = max(1, min(5, diff))
            except (ValueError, IndexError):
                diff = 3
            self._difficulties.append(diff)
            self._subjects.append(str(item.get("type", "unknown")))

    def __len__(self) -> int:
        return len(self._input_ids)

    def __getitem__(self, idx: int) -> dict:
        return {
            "input_ids": self._input_ids[idx],
            "labels": self._labels[idx],
        }

    @property
    def difficulties(self) -> list[int]:
        return self._difficulties

    def get_proof_tasks_by_difficulty(
        self,
        n_per_level: int = 20,
        verification_token: str = "=",
        seed: int = 99,
    ) -> list:
        """Get ProofTask instances stratified by difficulty level.

        For each difficulty level 1-5, sample n_per_level examples.
        The verification token is '=' (token id in cl100k_base) — a high
        logit for '=' at the last position indicates the model expects
        an equation to follow, correlating with correct mathematical reasoning.

        Args:
            n_per_level:         Examples per difficulty level.
            verification_token:  String token to use as verification signal.
            seed:                Random seed.

        Returns:
            List of ProofTask instances with label=difficulty (1-5 mapped to 0/1
            by thresholding at level 3: levels 1-2 = easy=1, levels 4-5 = hard=0).
        """
        from open_mythos.pipeline.evaluator import ProofTask

        enc = self.tokenizer._enc
        # Get token id for '=' sign
        eq_ids = enc.encode("=", disallowed_special=())
        verification_token_id = eq_ids[0] if eq_ids else 28

        rng = random.Random(seed)
        tasks = []

        for level in range(1, 6):
            level_indices = [i for i, d in enumerate(self._difficulties) if d == level]
            rng.shuffle(level_indices)
            selected = level_indices[:n_per_level]

            for idx in selected:
                # Binary label: easy problems (level 1-2) = valid/solvable (1)
                #               hard problems (level 4-5) = challenging (0)
                #               level 3 = skip (ambiguous)
                if level <= 2:
                    label = 1
                elif level >= 4:
                    label = 0
                else:
                    label = 1  # level 3 treated as positive

                tasks.append(ProofTask(
                    input_ids=self._input_ids[idx],
                    label=label,
                    verification_token_id=verification_token_id,
                ))

        rng.shuffle(tasks)
        return tasks


# ---------------------------------------------------------------------------
# Availability check and fallback
# ---------------------------------------------------------------------------

def load_gsm8k_or_fallback(split="train", seq_len=256, seed=42):
    """Load GSM8K if available, otherwise return synthetic MathCorpusDataset."""
    if _DATASETS_AVAILABLE and _TIKTOKEN_AVAILABLE:
        print(f"  Loading GSM8K ({split} split) with tiktoken tokenizer...")
        return GSM8KDataset(split=split, seq_len=seq_len, seed=seed)
    else:
        print("  WARNING: datasets/tiktoken not available. Using synthetic fallback.")
        print("  Install with: pip install datasets tiktoken")
        from open_mythos.pipeline.dataset import MathCorpusDataset
        return MathCorpusDataset(num_examples=1000, seq_len=min(seq_len, 32), seed=seed)


def load_math_or_fallback(split="test", seq_len=256, seed=42):
    """Load MATH dataset if available, otherwise return synthetic MathCorpusDataset."""
    if _DATASETS_AVAILABLE and _TIKTOKEN_AVAILABLE:
        print(f"  Loading MATH competition dataset ({split} split)...")
        return MATHDataset(split=split, seq_len=seq_len, seed=seed)
    else:
        print("  WARNING: datasets/tiktoken not available. Using synthetic fallback.")
        from open_mythos.pipeline.dataset import MathCorpusDataset
        return MathCorpusDataset(num_examples=500, seq_len=min(seq_len, 32), seed=seed)
