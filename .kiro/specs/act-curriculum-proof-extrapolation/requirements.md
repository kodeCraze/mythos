# Requirements Document

## Introduction

This feature adds a research pipeline on top of the OpenMythos Recurrent-Depth Transformer that combines two complementary ideas: using the ACT (Adaptive Computation Time) halting mechanism as an unsupervised curriculum signal, and training with progressively increasing loop depth to enable depth extrapolation on formal proof verification tasks.

The pipeline has five stages:
1. **ACT Profiling** — run a model over a math/proof corpus and log per-token halt depths.
2. **Difficulty Scoring** — rank training examples by their mean halt depth (harder = more loops needed).
3. **Curriculum Training** — train starting from easy examples and progressively introduce harder ones.
4. **Loop Depth Curriculum** — simultaneously increase `n_loops` during training (e.g., 4 → 8 → 16).
5. **Depth Extrapolation Evaluation** — evaluate a model trained at `n_loops=8` on tasks requiring `n_loops=32`.

---

## Glossary

- **Pipeline**: The end-to-end `ACTCurriculumPipeline` orchestrator that coordinates all five stages.
- **Profiler**: The `ACTProfiler` component that runs a model over a corpus and records per-token halt depths.
- **HaltDepth**: The loop iteration index at which a token's cumulative ACT halting probability first crosses `act_threshold`. An integer in `[1, n_loops]`.
- **DifficultyScorer**: The `DifficultyScorer` component that aggregates per-token halt depths into a per-example difficulty score.
- **DifficultyScore**: A non-negative float assigned to each training example; higher means harder. Derived from mean or percentile halt depth across the example's tokens.
- **CurriculumSampler**: The `CurriculumSampler` component that orders or weights training examples according to their difficulty scores and the current training stage.
- **LoopScheduler**: The `LoopScheduler` component that maps a training step or epoch to the `n_loops` value to use for that step.
- **DepthExtrapolationEvaluator**: The `DepthExtrapolationEvaluator` component that evaluates a trained model at loop depths higher than those seen during training.
- **ProofTask**: A formal proof verification example consisting of a tokenized premise sequence and a binary label (valid / invalid step).
- **MathCorpus**: A dataset of tokenized mathematical text or proof steps used for ACT profiling and curriculum training.
- **TrainingStage**: A discrete phase of curriculum training defined by a difficulty percentile range and a target `n_loops` value.
- **OpenMythos**: The `OpenMythos` model class from `open_mythos.main`.
- **RecurrentBlock**: The `RecurrentBlock` class from `open_mythos.main`; its `forward()` accepts `n_loops`.
- **ACTHalting**: The `ACTHalting` class from `open_mythos.main`; produces per-position halting probabilities.

---

## Requirements

### Requirement 1: ACT Halt Depth Profiling

**User Story:** As a researcher, I want to run a pre-trained or randomly initialized OpenMythos model over a math corpus and record the ACT halt depth for every token, so that I can derive an unsupervised difficulty signal without any labels.

#### Acceptance Criteria

1. THE Profiler SHALL accept an OpenMythos model instance, a tokenized MathCorpus, and a `n_loops` parameter, and produce a halt-depth tensor of shape `(num_examples, seq_len)` where each value is the HaltDepth for that token.
2. WHEN a token's cumulative ACT halting probability reaches or exceeds `act_threshold` at loop iteration `t`, THE Profiler SHALL record `t + 1` as the HaltDepth for that token.
3. WHEN a token's cumulative ACT halting probability does not reach `act_threshold` within `n_loops` iterations, THE Profiler SHALL record `n_loops` as the HaltDepth for that token.
4. THE Profiler SHALL operate under `torch.no_grad()` to avoid storing gradients during profiling.
5. THE Profiler SHALL process examples in configurable batches to support corpora that do not fit in GPU memory.
6. IF the model or corpus is provided with mismatched sequence lengths, THEN THE Profiler SHALL raise a `ValueError` with a descriptive message identifying the mismatch.
7. THE Profiler SHALL serialize the halt-depth tensor and associated example indices to disk in a format that the DifficultyScorer can load without re-running the model.

---

### Requirement 2: Difficulty Scoring from Halt Depths

**User Story:** As a researcher, I want to convert per-token halt depths into a single difficulty score per training example, so that I can rank examples for curriculum ordering.

#### Acceptance Criteria

1. THE DifficultyScorer SHALL accept a halt-depth tensor of shape `(num_examples, seq_len)` and produce a DifficultyScore vector of shape `(num_examples,)`.
2. THE DifficultyScorer SHALL support at least two aggregation modes: `mean` (mean halt depth across non-padding tokens) and `p90` (90th-percentile halt depth across non-padding tokens).
3. WHEN a padding mask is provided, THE DifficultyScorer SHALL exclude padding positions from the aggregation.
4. THE DifficultyScorer SHALL normalize DifficultyScores to the range `[0.0, 1.0]` by dividing by the maximum observed score within the corpus.
5. THE DifficultyScorer SHALL produce identical DifficultyScores when called twice on the same inputs (deterministic).
6. THE DifficultyScorer SHALL serialize the scored dataset (example indices + scores) to disk so that the CurriculumSampler can load it without re-running the scorer.

---

### Requirement 3: Curriculum Sampling

**User Story:** As a researcher, I want to sample training examples in order of increasing difficulty, so that the model first masters easy examples before being exposed to harder ones.

#### Acceptance Criteria

1. THE CurriculumSampler SHALL accept a scored dataset and a list of TrainingStages, where each stage specifies a difficulty percentile range `[low, high)` and a maximum number of training steps.
2. WHEN the current training step falls within a TrainingStage, THE CurriculumSampler SHALL sample exclusively from examples whose DifficultyScore falls within that stage's percentile range.
3. WHEN transitioning between TrainingStages, THE CurriculumSampler SHALL include all examples from all previous stages in the sampling pool (cumulative curriculum).
4. THE CurriculumSampler SHALL support uniform random sampling within the active pool.
5. THE CurriculumSampler SHALL support competence-based weighted sampling where sampling probability is proportional to `(1 - DifficultyScore)` for examples in the active pool, so that easier examples within the pool are sampled more frequently.
6. IF the active pool for a given step is empty, THEN THE CurriculumSampler SHALL raise a `RuntimeError` with a message identifying the step and stage.
7. THE CurriculumSampler SHALL be compatible with PyTorch `DataLoader` as a drop-in `Sampler` subclass.

---

### Requirement 4: Loop Depth Scheduling

**User Story:** As a researcher, I want to progressively increase the number of recurrent loops during training, so that the model learns to reason at increasing depths and develops the depth-extrapolation property.

#### Acceptance Criteria

1. THE LoopScheduler SHALL accept a schedule defined as a list of `(step_threshold, n_loops)` pairs and return the correct `n_loops` value for any given training step.
2. WHEN the training step is below the first threshold, THE LoopScheduler SHALL return the `n_loops` value associated with the first entry.
3. WHEN the training step equals or exceeds a threshold, THE LoopScheduler SHALL return the `n_loops` value of the highest threshold not exceeding the current step.
4. THE LoopScheduler SHALL validate that all `n_loops` values in the schedule are positive integers and that all `step_threshold` values are non-negative integers in strictly ascending order; IF validation fails, THEN THE LoopScheduler SHALL raise a `ValueError` with a descriptive message.
5. THE LoopScheduler SHALL expose a `current_n_loops(step: int) -> int` method that the training loop calls each step to retrieve the active loop count.
6. THE LoopScheduler SHALL be serializable to and deserializable from a plain Python dict so that training can be resumed from a checkpoint.

---

### Requirement 5: Curriculum Training Loop

**User Story:** As a researcher, I want a training loop that integrates the CurriculumSampler and LoopScheduler with the OpenMythos model, so that I can run the full curriculum training experiment end-to-end.

#### Acceptance Criteria

1. THE Pipeline SHALL accept an OpenMythos model, a CurriculumSampler, a LoopScheduler, an optimizer, and a loss function, and execute a training loop for a configurable number of steps.
2. WHEN executing a training step, THE Pipeline SHALL call `LoopScheduler.current_n_loops(step)` and pass the result as `n_loops` to `OpenMythos.forward()`.
3. WHEN executing a training step, THE Pipeline SHALL call `CurriculumSampler` to obtain the batch for that step.
4. THE Pipeline SHALL log the current `n_loops`, current TrainingStage index, mean batch DifficultyScore, and training loss at a configurable logging interval.
5. THE Pipeline SHALL save a checkpoint containing model weights, optimizer state, LoopScheduler state, CurriculumSampler state, and current step at a configurable checkpoint interval.
6. IF a checkpoint path is provided at initialization, THEN THE Pipeline SHALL resume training from that checkpoint, restoring all state before continuing.
7. THE Pipeline SHALL support gradient accumulation over a configurable number of micro-steps to allow effective batch sizes larger than GPU memory permits.

---

### Requirement 6: Depth Extrapolation Evaluation

**User Story:** As a researcher, I want to evaluate a model trained at a low loop depth on proof tasks requiring a higher loop depth, so that I can measure the depth-extrapolation capability of the curriculum-trained model.

#### Acceptance Criteria

1. THE DepthExtrapolationEvaluator SHALL accept an OpenMythos model, a list of ProofTasks, and a list of `eval_n_loops` values, and produce an accuracy metric for each `eval_n_loops` value.
2. WHEN evaluating at a given `eval_n_loops`, THE DepthExtrapolationEvaluator SHALL call `OpenMythos.forward()` with that `n_loops` value and classify each ProofTask as valid or invalid based on the logit of the designated verification token.
3. THE DepthExtrapolationEvaluator SHALL operate under `torch.no_grad()` during evaluation.
4. THE DepthExtrapolationEvaluator SHALL report per-depth accuracy as `correct_predictions / total_examples` for each `eval_n_loops` value.
5. THE DepthExtrapolationEvaluator SHALL produce a summary table mapping each `eval_n_loops` to its accuracy, formatted as a list of dicts with keys `n_loops` and `accuracy`.
6. WHEN `eval_n_loops` contains a value greater than `cfg.max_loop_iters`, THE DepthExtrapolationEvaluator SHALL pass that value directly to `OpenMythos.forward()` without modifying the model, relying on the RecurrentBlock's ability to accept arbitrary `n_loops` at inference.
7. THE DepthExtrapolationEvaluator SHALL serialize the evaluation results to disk as a JSON file.

---

### Requirement 7: Halt Depth Logging During Training

**User Story:** As a researcher, I want to monitor per-token halt depths during training, so that I can observe how the model's computation allocation evolves as curriculum difficulty increases.

#### Acceptance Criteria

1. WHEN halt depth logging is enabled, THE Pipeline SHALL instrument the RecurrentBlock to record the loop iteration at which each token halts during each training forward pass.
2. THE Pipeline SHALL log the mean, minimum, and maximum halt depth across all tokens in the batch at each logging step.
3. THE Pipeline SHALL log the fraction of tokens that halted before the maximum loop count (early-halt rate) at each logging step.
4. THE Pipeline SHALL write halt depth statistics to a structured log file (one JSON object per line) so that they can be post-processed independently of the training run.
5. WHEN halt depth logging is disabled, THE Pipeline SHALL incur no additional computation or memory overhead beyond the standard forward pass.

---

### Requirement 8: Pipeline Configuration and Reproducibility

**User Story:** As a researcher, I want to configure the entire pipeline from a single configuration object and reproduce any experiment from a saved config, so that experiments are auditable and shareable.

#### Acceptance Criteria

1. THE Pipeline SHALL accept a `PipelineConfig` dataclass that specifies all hyperparameters: profiling batch size, difficulty aggregation mode, curriculum stage definitions, loop depth schedule, training steps, logging interval, checkpoint interval, and evaluation loop depths.
2. THE PipelineConfig SHALL be serializable to and deserializable from JSON without loss of information.
3. WHEN a `PipelineConfig` is deserialized from JSON and used to initialize the Pipeline, THE Pipeline SHALL produce identical training behavior to a Pipeline initialized from the original `PipelineConfig`, given the same random seed.
4. THE Pipeline SHALL accept a `seed` parameter and call `torch.manual_seed(seed)` before any stochastic operation to ensure reproducibility.
5. THE PipelineConfig SHALL validate that `eval_n_loops` contains at least one value greater than the maximum `n_loops` in the loop depth schedule; IF this condition is not met, THEN THE PipelineConfig SHALL raise a `ValueError` indicating that no extrapolation evaluation would occur.
