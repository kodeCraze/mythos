# Implementation Plan: ACT Curriculum Proof Extrapolation Pipeline

## Overview

Implement the `open_mythos/pipeline/` package in Python/PyTorch. Each task builds incrementally on the previous ones, starting with the configuration layer and ending with the fully wired `ACTCurriculumPipeline`. All code lives in `open_mythos/pipeline/`; `open_mythos/main.py` is a read-only dependency.

## Tasks

- [x] 1. Create the `open_mythos/pipeline/` package skeleton and `PipelineConfig`
  - Create `open_mythos/pipeline/__init__.py` (empty for now; filled in the final task)
  - Create `open_mythos/pipeline/config.py` with `StageConfig` and `PipelineConfig` dataclasses
  - Implement `PipelineConfig.__post_init__` validation: raise `ValueError` when no `eval_n_loops` value exceeds the maximum `n_loops` in `loop_schedule`
  - Implement `PipelineConfig.to_dict()` and `PipelineConfig.from_dict()` for lossless JSON round-trip (convert `stages` list and `loop_schedule` list of tuples correctly)
  - _Requirements: 8.1, 8.2, 8.3, 8.5_

  - [ ]* 1.1 Write property test for `PipelineConfig` JSON round-trip
    - **Property 1: Config round-trip consistency** — `PipelineConfig.from_dict(cfg.to_dict()) == cfg` for any valid config
    - **Validates: Requirements 8.2, 8.3**

  - [ ]* 1.2 Write unit tests for `PipelineConfig` validation
    - Test that `__post_init__` raises `ValueError` when all `eval_n_loops` values are ≤ max schedule `n_loops`
    - Test that a valid config with at least one extrapolation depth constructs without error
    - _Requirements: 8.5_

- [x] 2. Implement `LoopScheduler`
  - Create `open_mythos/pipeline/scheduler.py`
  - Implement `LoopScheduler.__init__` with validation: raise `ValueError` if any `n_loops` is not a positive integer, or if `step_threshold` values are not non-negative integers in strictly ascending order
  - Implement `current_n_loops(step: int) -> int` using the piecewise-constant lookup algorithm from the design
  - Implement `to_dict()` and `from_dict()` for checkpoint serialization
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6_

  - [ ]* 2.1 Write property test for `LoopScheduler` monotonicity
    - **Property 2: Monotone step→n_loops** — for any two steps `s1 ≤ s2`, `scheduler.current_n_loops(s1) ≤ scheduler.current_n_loops(s2)`
    - **Validates: Requirements 4.1, 4.3**

  - [ ]* 2.2 Write property test for `LoopScheduler` round-trip
    - **Property 3: Scheduler round-trip consistency** — `LoopScheduler.from_dict(s.to_dict()).current_n_loops(step) == s.current_n_loops(step)` for any step
    - **Validates: Requirements 4.6**

  - [ ]* 2.3 Write unit tests for `LoopScheduler` validation and boundary conditions
    - Test that steps below the first threshold return the first `n_loops` value
    - Test that steps at exact thresholds return the correct `n_loops`
    - Test that invalid schedules (non-ascending thresholds, non-positive `n_loops`) raise `ValueError`
    - _Requirements: 4.2, 4.3, 4.4_

- [x] 3. Implement `DifficultyScorer`
  - Create `open_mythos/pipeline/scorer.py`
  - Implement `DifficultyScorer.__init__` accepting `mode: str` (`"mean"` or `"p90"`)
  - Implement `score(halt_depths, padding_mask)` — exclude padding positions, aggregate per example, normalize to `[0.0, 1.0]` by dividing by corpus-wide maximum
  - Implement `save(scores, indices, path)` using `torch.save` with the format from the design
  - Implement `load(path)` returning `{"scores": Tensor, "indices": Tensor}`
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6_

  - [ ]* 3.1 Write property test for `DifficultyScorer` output range
    - **Property 4: Scores in [0, 1]** — for any halt-depth tensor and any valid mask, all output scores are in `[0.0, 1.0]`
    - **Validates: Requirements 2.4**

  - [ ]* 3.2 Write property test for `DifficultyScorer` determinism
    - **Property 5: Scorer determinism** — calling `score()` twice on the same inputs returns identical tensors
    - **Validates: Requirements 2.5**

  - [ ]* 3.3 Write unit tests for `DifficultyScorer`
    - Test `mean` mode with and without padding mask
    - Test `p90` mode with and without padding mask
    - Test that the hardest example always receives score `1.0`
    - Test `save`/`load` round-trip preserves scores and indices exactly
    - _Requirements: 2.1, 2.2, 2.3, 2.6_

- [x] 4. Implement `ACTProfiler`
  - Create `open_mythos/pipeline/profiler.py`
  - Implement `ACTProfiler.__init__` accepting `cfg: PipelineConfig`
  - Implement the `_instrument(model)` context manager that monkey-patches `model.recurrent.act.forward` to capture per-step halting probabilities and restores the original on exit
  - Implement `profile(model, corpus, n_loops, padding_mask)`:
    - Validate that `corpus` shape is consistent; raise `ValueError` with a descriptive message on mismatch
    - Process examples in batches of `cfg.profiling_batch_size` under `torch.no_grad()`
    - Apply the halt-depth capture algorithm from the design to produce a `(num_examples, seq_len)` int tensor
    - Tokens that never reach `act_threshold` within `n_loops` iterations receive `n_loops` as their halt depth
  - Implement `save(halt_depths, path)` and `load(path)` using `torch.save`/`torch.load`
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7_

  - [ ]* 4.1 Write property test for `ACTProfiler` halt depth bounds
    - **Property 6: Halt depths in [1, n_loops]** — for any model and corpus, every value in the returned halt-depth tensor is an integer in `[1, n_loops]`
    - **Validates: Requirements 1.1, 1.2, 1.3**

  - [ ]* 4.2 Write unit tests for `ACTProfiler`
    - Test that profiling runs under `no_grad` (no gradients on model parameters after profiling)
    - Test that the original `act.forward` is restored after profiling (even if an exception occurs)
    - Test that a shape mismatch raises `ValueError`
    - Test `save`/`load` round-trip preserves halt depths and metadata
    - _Requirements: 1.4, 1.6, 1.7_

- [x] 5. Checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 6. Implement `CurriculumSampler`
  - Create `open_mythos/pipeline/sampler.py`
  - Implement `CurriculumSampler` as a `torch.utils.data.Sampler` subclass
  - Implement `__init__` accepting `scores`, `stages`, `sampling_mode`, and `seed`
  - Implement `set_step(step)` to update the active pool using the cumulative curriculum algorithm from the design; raise `RuntimeError` with step and stage info if the pool is empty
  - Implement `get_active_pool(step)` returning the indices of examples in the active pool
  - Implement `__iter__` for uniform sampling and competence-weighted sampling (`w[i] = 1 - scores[i]`)
  - Implement `__len__` returning the size of the current active pool
  - Implement `state_dict()` and `load_state_dict()` for checkpoint resume
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7_

  - [ ]* 6.1 Write property test for `CurriculumSampler` pool monotonicity
    - **Property 7: Cumulative pool growth** — for any two steps `s1 ≤ s2` in the same or later stage, `len(get_active_pool(s1)) ≤ len(get_active_pool(s2))`
    - **Validates: Requirements 3.3**

  - [ ]* 6.2 Write property test for `CurriculumSampler` pool membership
    - **Property 8: Pool difficulty bounds** — all examples in the active pool at any step have `DifficultyScore` within the cumulative percentile range `[0, active_stage.high_percentile]`
    - **Validates: Requirements 3.2, 3.3**

  - [ ]* 6.3 Write unit tests for `CurriculumSampler`
    - Test that an empty pool raises `RuntimeError`
    - Test uniform sampling produces indices only from the active pool
    - Test weighted sampling assigns higher probability to easier examples
    - Test `state_dict`/`load_state_dict` round-trip restores sampler state
    - _Requirements: 3.4, 3.5, 3.6, 3.7_

- [x] 7. Implement `HaltDepthLogger`
  - Create `open_mythos/pipeline/logger.py`
  - Implement `HaltDepthLogger.__init__` accepting `log_path: str` and `n_loops: int`; open the JSONL file for appending
  - Implement `compute_stats(halt_depths, n_loops)` as a static method returning `mean_halt_depth`, `min_halt_depth`, `max_halt_depth`, and `early_halt_rate` (fraction of tokens with depth < `n_loops`)
  - Implement `log(step, halt_depths, n_loops)` that calls `compute_stats` and writes one JSON object per line to the log file
  - _Requirements: 7.2, 7.3, 7.4_

  - [ ]* 7.1 Write unit tests for `HaltDepthLogger`
    - Test `compute_stats` with known halt-depth tensors (all halted early, none halted early, mixed)
    - Test that `log` writes valid JSON lines to the file
    - _Requirements: 7.2, 7.3, 7.4_

- [x] 8. Implement `DepthExtrapolationEvaluator` and `ProofTask`
  - Create `open_mythos/pipeline/evaluator.py`
  - Define the `ProofTask` dataclass with `input_ids`, `label`, and `verification_token_id` fields
  - Implement `DepthExtrapolationEvaluator.__init__` accepting `cfg: PipelineConfig`
  - Implement `evaluate(model, tasks, eval_n_loops)` decorated with `@torch.no_grad()`:
    - For each `n_loops` value, run `model.forward(task.input_ids.unsqueeze(0), n_loops=n)` and classify using the logit at the last position for `verification_token_id`
    - Compute accuracy as `correct / total` for each depth
    - Return a list of `{"n_loops": int, "accuracy": float}` dicts
  - Implement `save_results(results, path)` serializing to JSON and `load_results(path)` deserializing from JSON
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7_

  - [ ]* 8.1 Write property test for `DepthExtrapolationEvaluator` accuracy bounds
    - **Property 9: Accuracy in [0, 1]** — for any model and any list of `ProofTask`s, all reported accuracy values are in `[0.0, 1.0]`
    - **Validates: Requirements 6.4**

  - [ ]* 8.2 Write unit tests for `DepthExtrapolationEvaluator`
    - Test that evaluation runs under `no_grad` (no gradients accumulated)
    - Test that extrapolation depths (> max training `n_loops`) are passed directly to `model.forward` without modification
    - Test `save_results`/`load_results` round-trip preserves all entries
    - _Requirements: 6.3, 6.6, 6.7_

- [x] 9. Implement `ACTCurriculumPipeline`
  - Create `open_mythos/pipeline/pipeline.py`
  - Implement `ACTCurriculumPipeline.__init__` accepting `model`, `sampler`, `scheduler`, `optimizer`, `criterion`, `cfg`, and optional `checkpoint_path`; if `checkpoint_path` is provided, call `_load_checkpoint` immediately
  - Implement `_save_checkpoint(step)` saving model weights, optimizer state, scheduler state, sampler state, and config to `cfg.checkpoint_dir/checkpoint_{step}.pt`
  - Implement `_load_checkpoint(path)` restoring all state and setting `self.start_step`
  - Implement `_log_step(step, loss, halt_stats)` printing current `n_loops`, stage index, mean batch difficulty score, and loss; if `halt_stats` is not None, also log halt depth statistics
  - Implement `run(dataset)`:
    - Call `torch.manual_seed(cfg.seed)` before any stochastic operation
    - Build a `DataLoader` using `self.sampler`
    - For each step, call `scheduler.current_n_loops(step)`, call `sampler.set_step(step)`, run the forward pass with gradient accumulation over `cfg.gradient_accumulation_steps` micro-steps
    - If `cfg.log_halt_depths`, instrument `model.recurrent.act.forward` via a lightweight hook to collect halt depths for the current batch and pass stats to `_log_step`; incur zero overhead when disabled
    - Log at `cfg.logging_interval` and checkpoint at `cfg.checkpoint_interval`
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 7.1, 7.5, 8.3, 8.4_

  - [ ]* 9.1 Write unit tests for `ACTCurriculumPipeline` checkpointing
    - Test that a checkpoint saved at step N can be loaded and training resumes from step N
    - Test that `torch.manual_seed` is called before the training loop
    - _Requirements: 5.5, 5.6, 8.3, 8.4_

  - [ ]* 9.2 Write unit tests for gradient accumulation
    - Test that gradients are accumulated over `gradient_accumulation_steps` micro-steps before `optimizer.step()`
    - _Requirements: 5.7_

- [x] 10. Checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 11. Wire public API in `open_mythos/pipeline/__init__.py`
  - Update `open_mythos/pipeline/__init__.py` to export all public symbols:
    - `PipelineConfig`, `StageConfig` from `config`
    - `ACTProfiler` from `profiler`
    - `DifficultyScorer` from `scorer`
    - `CurriculumSampler` from `sampler`
    - `LoopScheduler` from `scheduler`
    - `ACTCurriculumPipeline` from `pipeline`
    - `DepthExtrapolationEvaluator`, `ProofTask` from `evaluator`
    - `HaltDepthLogger` from `logger`
  - Verify that `from open_mythos.pipeline import ACTCurriculumPipeline` works without errors
  - _Requirements: 8.1_

- [x] 12. Final checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for a faster MVP
- Property tests use `hypothesis` (install via `pip install hypothesis`); unit tests use `pytest`
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation after each major component
- `open_mythos/main.py` must not be modified; all instrumentation uses monkey-patching via context managers
