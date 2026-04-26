# Design Document: ACT Curriculum Proof Extrapolation Pipeline

## Overview

This document describes the design of a research pipeline built on top of the OpenMythos Recurrent-Depth Transformer. The pipeline exploits the ACT (Adaptive Computation Time) halting mechanism as a free, unsupervised difficulty signal: tokens that require more recurrent loops to converge are harder, and examples with higher mean halt depth are more challenging for the model.

The pipeline has five coordinated stages:

1. **ACT Profiling** — run the model over a math/proof corpus and record per-token halt depths.
2. **Difficulty Scoring** — aggregate per-token halt depths into a single per-example difficulty score.
3. **Curriculum Sampling** — order training examples from easy to hard using those scores.
4. **Loop Depth Scheduling** — simultaneously ramp `n_loops` during training (e.g., 4 → 8 → 16).
5. **Depth Extrapolation Evaluation** — evaluate a model trained at `n_loops=8` on tasks requiring `n_loops=32`.

All new code lives in `open_mythos/pipeline/` and treats `open_mythos/main.py` as a read-only dependency. No modifications to `main.py` are required.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ACTCurriculumPipeline                                │
│                                                                             │
│  ┌──────────────┐    halt_depths.pt    ┌──────────────────┐                │
│  │ ACTProfiler  │ ──────────────────► │ DifficultyScorer │                │
│  │              │                     │                  │                │
│  │ instruments  │                     │ mean / p90 agg   │                │
│  │ RecurrentBlk │                     │ normalize [0,1]  │                │
│  └──────────────┘                     └────────┬─────────┘                │
│         ▲                                      │ scores.pt                │
│         │ n_loops                              ▼                          │
│  ┌──────┴───────┐                     ┌──────────────────┐                │
│  │ LoopScheduler│                     │CurriculumSampler │                │
│  │              │                     │                  │                │
│  │ step→n_loops │                     │ stage-based pool │                │
│  │ schedule     │                     │ uniform/weighted │                │
│  └──────┬───────┘                     └────────┬─────────┘                │
│         │ n_loops(step)                        │ batch indices            │
│         ▼                                      ▼                          │
│  ┌──────────────────────────────────────────────────────┐                 │
│  │                   Training Loop                      │                 │
│  │                                                      │                 │
│  │  for step in range(total_steps):                     │                 │
│  │    n = scheduler.current_n_loops(step)               │                 │
│  │    batch = sampler[step]                             │                 │
│  │    logits = model(batch, n_loops=n)                  │                 │
│  │    loss = criterion(logits, labels)                  │                 │
│  │    loss.backward(); optimizer.step()                 │                 │
│  └──────────────────────────────────────────────────────┘                 │
│                                                                             │
│  ┌──────────────────────────────────────────────────────┐                 │
│  │            DepthExtrapolationEvaluator               │                 │
│  │                                                      │                 │
│  │  for n_loops in eval_n_loops:                        │                 │
│  │    acc = evaluate(model, proof_tasks, n_loops)       │                 │
│  │  → results.json                                      │                 │
│  └──────────────────────────────────────────────────────┘                 │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
MathCorpus (tokenized)
        │
        ▼
  ACTProfiler.profile(model, corpus, n_loops)
        │
        ▼
  halt_depths: Tensor(num_examples, seq_len)   [saved to disk]
        │
        ▼
  DifficultyScorer.score(halt_depths, mask, mode)
        │
        ▼
  scores: Tensor(num_examples,)  ∈ [0.0, 1.0]  [saved to disk]
        │
        ▼
  CurriculumSampler(scores, stages)
        │  ← step
        ▼
  batch_indices: List[int]
        │
        ▼
  OpenMythos.forward(batch, n_loops=scheduler(step))
        │
        ▼
  loss → backward → optimizer.step()
```

---

## Components and Interfaces

### `PipelineConfig` (`open_mythos/pipeline/config.py`)

Central configuration dataclass. All hyperparameters for the entire experiment live here.

```python
@dataclass
class StageConfig:
    low_percentile: float        # lower bound of difficulty percentile range [0, 1)
    high_percentile: float       # upper bound of difficulty percentile range (0, 1]
    max_steps: int               # number of training steps in this stage

@dataclass
class PipelineConfig:
    # Profiling
    profiling_batch_size: int = 32
    profiling_n_loops: int = 8

    # Difficulty scoring
    difficulty_mode: str = "mean"          # "mean" | "p90"

    # Curriculum stages
    stages: list[StageConfig] = field(default_factory=lambda: [
        StageConfig(0.0, 0.33, 1000),
        StageConfig(0.0, 0.66, 2000),
        StageConfig(0.0, 1.00, 3000),
    ])

    # Loop depth schedule: list of (step_threshold, n_loops) pairs
    loop_schedule: list[tuple[int, int]] = field(default_factory=lambda: [
        (0, 4), (1000, 8), (3000, 16)
    ])

    # Training
    total_steps: int = 6000
    logging_interval: int = 100
    checkpoint_interval: int = 500
    gradient_accumulation_steps: int = 1
    seed: int = 42

    # Evaluation
    eval_n_loops: list[int] = field(default_factory=lambda: [8, 16, 32])

    # Paths
    halt_depth_path: str = "halt_depths.pt"
    scores_path: str = "scores.pt"
    checkpoint_dir: str = "checkpoints/"
    results_path: str = "results.json"
    halt_log_path: str = "halt_log.jsonl"

    # Halt depth logging
    log_halt_depths: bool = True

    def __post_init__(self):
        max_schedule_loops = max(n for _, n in self.loop_schedule)
        if not any(n > max_schedule_loops for n in self.eval_n_loops):
            raise ValueError(
                f"eval_n_loops must contain at least one value greater than "
                f"the maximum n_loops in the schedule ({max_schedule_loops}). "
                f"Got eval_n_loops={self.eval_n_loops}."
            )

    def to_dict(self) -> dict: ...
    @classmethod
    def from_dict(cls, d: dict) -> "PipelineConfig": ...
```

---

### `ACTProfiler` (`open_mythos/pipeline/profiler.py`)

Instruments `RecurrentBlock` via a forward hook to capture per-token halt depths without modifying `main.py`.

**Key design decision:** Rather than modifying `RecurrentBlock.forward()`, the profiler registers a `register_forward_hook` on the `RecurrentBlock` instance. However, since the halt depth is computed *inside* the loop (not at the block boundary), the profiler instead patches the `ACTHalting` module's forward method temporarily to intercept per-step halting probabilities.

**Halt depth capture algorithm:**

```
halt_depths = zeros(B, T)          # 0 = not yet halted
halted = zeros(B, T, dtype=bool)

for t in range(n_loops):
    p = act_module(h)              # (B, T) — intercepted by hook
    cumulative_p += p * ~halted
    newly_halted = (~halted) & (cumulative_p >= act_threshold)
    halt_depths[newly_halted] = t + 1
    halted |= newly_halted

# tokens that never halted get n_loops
halt_depths[~halted] = n_loops
```

**Interface:**

```python
class ACTProfiler:
    def __init__(self, cfg: PipelineConfig): ...

    def profile(
        self,
        model: OpenMythos,
        corpus: torch.Tensor,          # (num_examples, seq_len) token ids
        n_loops: int,
        padding_mask: Optional[torch.Tensor] = None,  # (num_examples, seq_len) bool
    ) -> torch.Tensor:                 # (num_examples, seq_len) int halt depths
        """Run model over corpus under no_grad, return halt-depth tensor."""
        ...

    def save(self, halt_depths: torch.Tensor, path: str) -> None:
        """Serialize halt_depths + metadata to disk."""
        ...

    @staticmethod
    def load(path: str) -> dict:
        """Load serialized halt depths. Returns {"halt_depths": Tensor, "meta": dict}."""
        ...
```

**Instrumentation approach:**

The profiler temporarily replaces `model.recurrent.act.forward` with a wrapper that records the halting probability at each loop step. It uses a context manager to ensure the original method is always restored:

```python
@contextmanager
def _instrument(self, model: OpenMythos):
    original_forward = model.recurrent.act.forward
    step_probs = []

    def hooked_forward(h):
        p = original_forward(h)
        step_probs.append(p.detach())
        return p

    model.recurrent.act.forward = hooked_forward
    try:
        yield step_probs
    finally:
        model.recurrent.act.forward = original_forward
```

This approach is non-invasive: it captures the exact same `p` values that `RecurrentBlock.forward()` uses internally, so the recorded halt depths are consistent with the model's actual ACT behavior.

---

### `DifficultyScorer` (`open_mythos/pipeline/scorer.py`)

Aggregates per-token halt depths into a single scalar difficulty score per example.

**Interface:**

```python
class DifficultyScorer:
    def __init__(self, mode: str = "mean"):
        """
        Args:
            mode: "mean" — mean halt depth over non-padding tokens
                  "p90"  — 90th-percentile halt depth over non-padding tokens
        """
        ...

    def score(
        self,
        halt_depths: torch.Tensor,          # (num_examples, seq_len) int
        padding_mask: Optional[torch.Tensor] = None,  # (num_examples, seq_len) bool
                                                       # True = real token, False = pad
    ) -> torch.Tensor:                      # (num_examples,) float in [0.0, 1.0]
        ...

    def save(self, scores: torch.Tensor, indices: torch.Tensor, path: str) -> None:
        """Serialize scores + example indices to disk."""
        ...

    @staticmethod
    def load(path: str) -> dict:
        """Load serialized scores. Returns {"scores": Tensor, "indices": Tensor}."""
        ...
```

**Scoring algorithm:**

```
For each example i:
    valid_depths = halt_depths[i][padding_mask[i]]   # exclude padding
    if mode == "mean":
        raw_score[i] = mean(valid_depths.float())
    elif mode == "p90":
        raw_score[i] = percentile(valid_depths.float(), 90)

# Normalize to [0, 1]
scores = raw_score / max(raw_score)
```

The normalization divides by the corpus-wide maximum, so the hardest example always gets score 1.0 and all others are proportionally scaled.

---

### `CurriculumSampler` (`open_mythos/pipeline/sampler.py`)

A `torch.utils.data.Sampler` subclass that manages stage-based example pools and supports both uniform and competence-weighted sampling.

**Interface:**

```python
class CurriculumSampler(torch.utils.data.Sampler):
    def __init__(
        self,
        scores: torch.Tensor,           # (num_examples,) float in [0.0, 1.0]
        stages: list[StageConfig],
        sampling_mode: str = "uniform", # "uniform" | "weighted"
        seed: int = 42,
    ): ...

    def set_step(self, step: int) -> None:
        """Update the active pool for the given training step."""
        ...

    def get_active_pool(self, step: int) -> torch.Tensor:
        """Return indices of examples in the active pool for the given step."""
        ...

    def __iter__(self) -> Iterator[int]: ...
    def __len__(self) -> int: ...

    def state_dict(self) -> dict: ...
    def load_state_dict(self, state: dict) -> None: ...
```

**Stage transition algorithm:**

```
stages = [Stage(0.0, 0.33, 1000), Stage(0.0, 0.66, 2000), Stage(0.0, 1.00, 3000)]

cumulative_steps = [0, 1000, 3000, 6000]

For step s:
    stage_idx = largest i such that cumulative_steps[i] <= s
    active_stage = stages[stage_idx]

    # Cumulative curriculum: include all examples up to current high_percentile
    pool = indices where scores[i] <= percentile(scores, active_stage.high_percentile)
         AND scores[i] >= percentile(scores, 0.0)  # always 0 for cumulative

    if len(pool) == 0:
        raise RuntimeError(f"Empty pool at step {s}, stage {stage_idx}")
```

**Weighted sampling:**

```
For weighted mode, sampling weights w[i] = 1 - scores[i] for i in pool
(easier examples have higher weight)
```

---

### `LoopScheduler` (`open_mythos/pipeline/scheduler.py`)

Maps training step → `n_loops` using a piecewise-constant schedule.

**Interface:**

```python
class LoopScheduler:
    def __init__(self, schedule: list[tuple[int, int]]):
        """
        Args:
            schedule: list of (step_threshold, n_loops) pairs in ascending order.
                      Example: [(0, 4), (1000, 8), (3000, 16)]
        Raises:
            ValueError: if n_loops values are not positive integers, or
                        step_threshold values are not non-negative integers
                        in strictly ascending order.
        """
        ...

    def current_n_loops(self, step: int) -> int:
        """Return the n_loops value for the given training step."""
        ...

    def to_dict(self) -> dict: ...

    @classmethod
    def from_dict(cls, d: dict) -> "LoopScheduler": ...
```

**Lookup algorithm:**

```
schedule = [(0, 4), (1000, 8), (3000, 16)]

current_n_loops(step):
    result = schedule[0][1]   # default: first entry
    for (threshold, n) in schedule:
        if step >= threshold:
            result = n
    return result
```

This is O(|schedule|) but schedules are tiny (typically 3–10 entries).

---

### `ACTCurriculumPipeline` (`open_mythos/pipeline/pipeline.py`)

The top-level orchestrator. Wires all components together and runs the training loop.

**Interface:**

```python
class ACTCurriculumPipeline:
    def __init__(
        self,
        model: OpenMythos,
        sampler: CurriculumSampler,
        scheduler: LoopScheduler,
        optimizer: torch.optim.Optimizer,
        criterion: Callable,
        cfg: PipelineConfig,
        checkpoint_path: Optional[str] = None,
    ): ...

    def run(self, dataset: torch.utils.data.Dataset) -> None:
        """Execute the full training loop."""
        ...

    def _train_step(self, step: int, batch: dict) -> float:
        """Execute one training step (or micro-step). Returns loss."""
        ...

    def _save_checkpoint(self, step: int) -> None: ...
    def _load_checkpoint(self, path: str) -> None: ...
    def _log_step(self, step: int, loss: float, halt_stats: Optional[dict]) -> None: ...
```

**Training loop pseudocode:**

```python
def run(self, dataset):
    torch.manual_seed(self.cfg.seed)
    loader = DataLoader(dataset, sampler=self.sampler, batch_size=self.cfg.batch_size)

    step = self.start_step
    accum_loss = 0.0

    for batch in loader:
        if step >= self.cfg.total_steps:
            break

        self.sampler.set_step(step)
        n_loops = self.scheduler.current_n_loops(step)

        # Gradient accumulation
        for micro_step, micro_batch in enumerate(split_batch(batch, self.cfg.gradient_accumulation_steps)):
            logits = self.model(micro_batch["input_ids"], n_loops=n_loops)
            loss = self.criterion(logits, micro_batch["labels"])
            (loss / self.cfg.gradient_accumulation_steps).backward()
            accum_loss += loss.item()

        self.optimizer.step()
        self.optimizer.zero_grad()

        if step % self.cfg.logging_interval == 0:
            halt_stats = self._collect_halt_stats() if self.cfg.log_halt_depths else None
            self._log_step(step, accum_loss, halt_stats)

        if step % self.cfg.checkpoint_interval == 0:
            self._save_checkpoint(step)

        step += 1
```

---

### `DepthExtrapolationEvaluator` (`open_mythos/pipeline/evaluator.py`)

Evaluates a trained model at multiple loop depths, including depths beyond those seen during training.

**Interface:**

```python
@dataclass
class ProofTask:
    input_ids: torch.Tensor    # (seq_len,) tokenized premise
    label: int                 # 0 = invalid step, 1 = valid step
    verification_token_id: int # token id whose logit determines the classification

class DepthExtrapolationEvaluator:
    def __init__(self, cfg: PipelineConfig): ...

    @torch.no_grad()
    def evaluate(
        self,
        model: OpenMythos,
        tasks: list[ProofTask],
        eval_n_loops: Optional[list[int]] = None,  # defaults to cfg.eval_n_loops
    ) -> list[dict]:
        """
        Returns list of {"n_loops": int, "accuracy": float} dicts,
        one per eval_n_loops value.
        """
        ...

    def save_results(self, results: list[dict], path: str) -> None:
        """Serialize results to JSON."""
        ...

    @staticmethod
    def load_results(path: str) -> list[dict]:
        """Load results from JSON."""
        ...
```

**Classification algorithm:**

```
For each task t and each eval_n_loops value n:
    logits = model(t.input_ids.unsqueeze(0), n_loops=n)  # (1, seq_len, vocab_size)
    # Use the logit at the last position for the verification token
    score = logits[0, -1, t.verification_token_id]
    prediction = 1 if score > 0 else 0
    correct += (prediction == t.label)

accuracy[n] = correct / len(tasks)
```

---

### `HaltDepthLogger` (`open_mythos/pipeline/logger.py`)

Handles structured logging of halt depth statistics during training. Separated from the pipeline to keep concerns clean.

**Interface:**

```python
class HaltDepthLogger:
    def __init__(self, log_path: str, n_loops: int): ...

    def log(
        self,
        step: int,
        halt_depths: torch.Tensor,   # (B, T) int
        n_loops: int,
    ) -> None:
        """Compute and write statistics to the JSONL log file."""
        ...

    @staticmethod
    def compute_stats(halt_depths: torch.Tensor, n_loops: int) -> dict:
        """
        Returns:
            {
                "mean_halt_depth": float,
                "min_halt_depth": int,
                "max_halt_depth": int,
                "early_halt_rate": float,  # fraction with depth < n_loops
            }
        """
        ...
```

---

## Data Models

### `StageConfig`

```python
@dataclass
class StageConfig:
    low_percentile: float   # ∈ [0.0, 1.0), lower bound of difficulty range
    high_percentile: float  # ∈ (0.0, 1.0], upper bound of difficulty range
    max_steps: int          # steps allocated to this stage
```

### `ProofTask`

```python
@dataclass
class ProofTask:
    input_ids: torch.Tensor        # (seq_len,) int64 token ids
    label: int                     # 0 = invalid, 1 = valid proof step
    verification_token_id: int     # vocab index of the verification token
```

### Serialization Formats

**`halt_depths.pt`** — saved with `torch.save`:
```python
{
    "halt_depths": Tensor(num_examples, seq_len),  # int32
    "example_indices": Tensor(num_examples,),       # int64
    "n_loops": int,
    "act_threshold": float,
    "corpus_size": int,
}
```

**`scores.pt`** — saved with `torch.save`:
```python
{
    "scores": Tensor(num_examples,),    # float32 in [0.0, 1.0]
    "indices": Tensor(num_examples,),   # int64 example indices
    "mode": str,                        # "mean" | "p90"
}
```

**`results.json`** — evaluation results:
```json
[
    {"n_loops": 8,  "accuracy": 0.72},
    {"n_loops": 16, "accuracy": 0.81},
    {"n_loops": 32, "accuracy": 0.85}
]
```

**`halt_log.jsonl`** — one JSON object per line:
```json
{"step": 100, "n_loops": 4, "mean_halt_depth": 2.3, "min_halt_depth": 1, "max_halt_depth": 4, "early_halt_rate": 0.87}
{"step": 200, "n_loops": 4, "mean_halt_depth": 2.7, "min_halt_depth": 1, "max_halt_depth": 4, "early_halt_rate": 0.82}
```

**`checkpoint_{step}.pt`** — saved with `torch.save`:
```python
{
    "step": int,
    "model_state_dict": dict,
    "optimizer_state_dict": dict,
    "scheduler_state": dict,    # LoopScheduler.to_dict()
    "sampler_state": dict,      # CurriculumSampler.state_dict()
    "config": dict,             # PipelineConfig.to_dict()
}
```

---

## File / Module Structure

```
open_mythos/
├── main.py                          # existing — read-only dependency
├── __init__.py                      # existing
└── pipeline/
    ├── __init__.py                  # exports public API
    ├── config.py                    # PipelineConfig, StageConfig
    ├── profiler.py                  # ACTProfiler
    ├── scorer.py                    # DifficultyScorer
    ├── sampler.py                   # CurriculumSampler
    ├── scheduler.py                 # LoopScheduler
    ├── pipeline.py                  # ACTCurriculumPipeline
    ├── evaluator.py                 # DepthExtrapolationEvaluator, ProofTask
    └── logger.py                    # HaltDepthLogger
```

`open_mythos/pipeline/__init__.py` exports:
```python
from open_mythos.pipeline.config import PipelineConfig, StageConfig
from open_mythos.pipeline.profiler import ACTProfiler
from open_mythos.pipeline.scorer import DifficultyScorer
from open_mythos.pipeline.sampler import CurriculumSampler
from open_mythos.pipeline.scheduler import LoopScheduler
from open_mythos.pipeline.pipeline import ACTCurriculumPipeline
from open_mythos.pipeline.evaluator import DepthExtrapolationEvaluator, ProofTask
from open_mythos.pipeline.logger import HaltDepthLogger
```

