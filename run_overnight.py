"""
run_overnight.py — Extended CPU experiment with real benchmark data.

Uses GSM8K + MATH if installed, synthetic fallback otherwise.
Designed to run overnight (~8-12 hours on CPU).

Key differences from run_experiment.py:
  - 10,000 steps per run (5x more training)
  - dim=256 model (~3M params, still CPU-feasible)
  - Real GSM8K data if available
  - MATH benchmark evaluation if available
  - Saves checkpoints every 1000 steps for resume

Usage:
    pip install datasets tiktoken  # optional but recommended
    python run_overnight.py

Resume from checkpoint:
    python run_overnight.py --resume results_overnight/checkpoints_C/checkpoint_8000.pt
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch
import torch.nn as nn

from open_mythos.main import MythosConfig, OpenMythos
from open_mythos.pipeline import (
    ACTCurriculumPipeline, ACTProfiler, CurriculumSampler,
    DepthExtrapolationEvaluator, DifficultyScorer, LoopScheduler,
    PipelineConfig, StageConfig,
)
from open_mythos.pipeline.gsm8k_dataset import (
    load_gsm8k_or_fallback, load_math_or_fallback
)

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--resume", type=str, default=None,
                    help="Path to checkpoint to resume Run C from")
parser.add_argument("--steps", type=int, default=10000,
                    help="Training steps per run (default: 10000)")
parser.add_argument("--dim", type=int, default=256,
                    help="Model hidden dimension (default: 256)")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
SEED = 42
torch.manual_seed(SEED)
OUTDIR = "results_overnight"
for d in [OUTDIR, f"{OUTDIR}/checkpoints_A", f"{OUTDIR}/checkpoints_B",
          f"{OUTDIR}/checkpoints_C", f"{OUTDIR}/checkpoints_warmup"]:
    os.makedirs(d, exist_ok=True)

# ---------------------------------------------------------------------------
# Model config (CPU-overnight: ~3M params)
# ---------------------------------------------------------------------------
MODEL_CFG = MythosConfig(
    vocab_size=100277,      # tiktoken cl100k_base; falls back to 128 if synthetic
    dim=args.dim,
    n_heads=4,
    n_kv_heads=2,
    max_seq_len=128,
    max_loop_iters=32,
    prelude_layers=1,
    coda_layers=1,
    attn_type="gqa",
    n_experts=8,
    n_shared_experts=1,
    n_experts_per_tok=2,
    expert_dim=64,
    lora_rank=8,
    act_threshold=0.5,
    rope_theta=500000.0,
)

SEQ_LEN = 128
BATCH_SIZE = 16
WARMUP_STEPS = 2000
TOTAL_STEPS = args.steps
EVAL_N_LOOPS = [4, 8, 16, 32]
PAD_ID = 0
criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)

# ---------------------------------------------------------------------------
# Load datasets
# ---------------------------------------------------------------------------
print("=" * 60)
print("Loading datasets...")

train_dataset = load_gsm8k_or_fallback(split="train", seq_len=SEQ_LEN, seed=SEED)
eval_dataset  = load_math_or_fallback(split="test",  seq_len=SEQ_LEN, seed=SEED)

# Adjust vocab size if using synthetic fallback
if hasattr(train_dataset, 'VOCAB_SIZE'):
    MODEL_CFG = MythosConfig(
        vocab_size=train_dataset.VOCAB_SIZE,
        dim=args.dim, n_heads=4, n_kv_heads=2,
        max_seq_len=SEQ_LEN, max_loop_iters=32,
        prelude_layers=1, coda_layers=1, attn_type="gqa",
        n_experts=8, n_shared_experts=1, n_experts_per_tok=2,
        expert_dim=64, lora_rank=8, act_threshold=0.5, rope_theta=10000.0,
    )
    PAD_ID = 15
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)

if hasattr(eval_dataset, 'get_proof_tasks_by_difficulty'):
    proof_tasks = eval_dataset.get_proof_tasks_by_difficulty(n_per_level=30)
else:
    proof_tasks = eval_dataset.get_proof_tasks(n_tasks=150)

corpus_tensor = torch.stack([
    train_dataset[i]["input_ids"] for i in range(min(2000, len(train_dataset)))
])

print(f"  Train: {len(train_dataset)} examples")
print(f"  Eval:  {len(eval_dataset)} examples")
print(f"  Proof tasks: {len(proof_tasks)}")
print(f"  Model vocab: {MODEL_CFG.vocab_size}")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_model():
    torch.manual_seed(SEED)
    m = OpenMythos(MODEL_CFG)
    n = sum(p.numel() for p in m.parameters())
    print(f"  Parameters: {n:,} ({n/1e6:.2f}M)")
    return m

def evaluate_model(model, run_name):
    cfg_eval = PipelineConfig(loop_schedule=[(0, 8)], eval_n_loops=EVAL_N_LOOPS)
    evaluator = DepthExtrapolationEvaluator(cfg_eval)
    results = evaluator.evaluate(model, proof_tasks, eval_n_loops=EVAL_N_LOOPS)
    evaluator.save_results(results, f"{OUTDIR}/{run_name}_results.json")
    print(f"\n  [{run_name}] Results:")
    for r in results:
        marker = " <- EXTRAP" if r["n_loops"] > 16 else ""
        print(f"    n_loops={r['n_loops']:2d}  acc={r['accuracy']:.3f}{marker}")
    return results

def make_uniform_sampler(n_steps):
    scores = torch.full((len(train_dataset),), 0.5)
    stages = [StageConfig(0.0, 1.0, n_steps + 1)]
    s = CurriculumSampler(scores, stages, sampling_mode="uniform", seed=SEED)
    s.set_step(0)
    return s, scores

def run_pipeline(model, n_steps, scheduler, sampler, label,
                 log_halt=False, log_interval=500, resume=None):
    cfg = PipelineConfig(
        loop_schedule=scheduler.to_dict()["schedule"],
        eval_n_loops=EVAL_N_LOOPS,
        total_steps=n_steps,
        logging_interval=log_interval,
        checkpoint_interval=1000,
        checkpoint_dir=f"{OUTDIR}/checkpoints_{label}",
        halt_log_path=f"{OUTDIR}/halt_log_{label}.jsonl",
        log_halt_depths=log_halt,
        seed=SEED,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    pipeline = ACTCurriculumPipeline(
        model=model, sampler=sampler, scheduler=scheduler,
        optimizer=optimizer, criterion=criterion, cfg=cfg,
        checkpoint_path=resume,
    )
    pipeline.run(train_dataset, batch_size=BATCH_SIZE)
    return pipeline

# ---------------------------------------------------------------------------
# Phase 0: Warm-up + ACT Profiling
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Phase 0: Warm-up model for ACT profiling...")
t0 = time.time()

warmup_model = make_model()
sampler_w, _ = make_uniform_sampler(WARMUP_STEPS)
run_pipeline(warmup_model, WARMUP_STEPS,
             LoopScheduler([(0, 8)]), sampler_w,
             label="warmup", log_interval=200)
print(f"  Warm-up done in {time.time()-t0:.1f}s ({(time.time()-t0)/60:.1f} min)")

print("\nPhase 0b: ACT Profiling...")
t0 = time.time()

profiling_cfg = PipelineConfig(
    profiling_batch_size=32, profiling_n_loops=16,
    loop_schedule=[(0, 8)], eval_n_loops=EVAL_N_LOOPS,
)
profiler = ACTProfiler(profiling_cfg)
halt_depths = profiler.profile(warmup_model, corpus_tensor, n_loops=16)
profiler.save(halt_depths, f"{OUTDIR}/halt_depths.pt")

scorer = DifficultyScorer(mode="mean")
scores = scorer.score(halt_depths)
scorer.save(scores, torch.arange(len(corpus_tensor)), f"{OUTDIR}/act_scores.pt")

gt_difficulties = torch.tensor(
    [train_dataset.difficulties[i] for i in range(len(corpus_tensor))],
    dtype=torch.float32
)
torch.save({"scores": scores, "gt_difficulties": gt_difficulties},
           f"{OUTDIR}/difficulty_validation.pt")

print(f"  Profiling done in {time.time()-t0:.1f}s")
print(f"  Score range: [{scores.min():.3f}, {scores.max():.3f}]  std={scores.std():.4f}")
for d in range(1, 6):
    mask = gt_difficulties == d
    if mask.any():
        print(f"    difficulty {d}: mean={scores[mask].mean():.3f}  n={mask.sum().item()}")

if scores.std().item() < 0.01:
    print("  Low spread — using sequence-length proxy")
    non_pad = (corpus_tensor != PAD_ID).float().sum(dim=1)
    scores = (non_pad - non_pad.min()) / (non_pad.max() - non_pad.min() + 1e-8)

# Extend scores to full dataset size
if len(scores) < len(train_dataset):
    extra = torch.full((len(train_dataset) - len(scores),), scores.mean().item())
    scores = torch.cat([scores, extra])

del warmup_model

# ---------------------------------------------------------------------------
# Run A: Baseline
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Run A: Baseline (fixed n_loops=8, random sampling)...")
t0 = time.time()

model_A = make_model()
sampler_A, _ = make_uniform_sampler(TOTAL_STEPS)
run_pipeline(model_A, TOTAL_STEPS,
             LoopScheduler([(0, 8)]), sampler_A,
             label="A", log_interval=500)
print(f"  Run A done in {time.time()-t0:.1f}s ({(time.time()-t0)/60:.1f} min)")
results_A = evaluate_model(model_A, "run_A")
del model_A

# ---------------------------------------------------------------------------
# Run B: Loop Curriculum
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Run B: Loop curriculum (n_loops=4->8->16, random sampling)...")
t0 = time.time()

model_B = make_model()
sampler_B, _ = make_uniform_sampler(TOTAL_STEPS)
run_pipeline(model_B, TOTAL_STEPS,
             LoopScheduler([(0, 4), (TOTAL_STEPS//3, 8), (2*TOTAL_STEPS//3, 16)]),
             sampler_B, label="B", log_interval=500)
print(f"  Run B done in {time.time()-t0:.1f}s ({(time.time()-t0)/60:.1f} min)")
results_B = evaluate_model(model_B, "run_B")
del model_B

# ---------------------------------------------------------------------------
# Run C: Full Pipeline (our method)
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Run C: Full pipeline (ACT curriculum + progressive n_loops)...")
t0 = time.time()

model_C = make_model()
stages_C = [
    StageConfig(0.0, 0.33, TOTAL_STEPS // 3),
    StageConfig(0.0, 0.66, TOTAL_STEPS // 3),
    StageConfig(0.0, 1.00, TOTAL_STEPS - 2 * (TOTAL_STEPS // 3)),
]
sampler_C = CurriculumSampler(scores, stages_C, sampling_mode="weighted", seed=SEED)
sampler_C.set_step(0)

run_pipeline(model_C, TOTAL_STEPS,
             LoopScheduler([(0, 4), (TOTAL_STEPS//3, 8), (2*TOTAL_STEPS//3, 16)]),
             sampler_C, label="C", log_halt=True, log_interval=500,
             resume=args.resume)
print(f"  Run C done in {time.time()-t0:.1f}s ({(time.time()-t0)/60:.1f} min)")
results_C = evaluate_model(model_C, "run_C")
del model_C

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)
print(f"{'n_loops':>8} | {'Run A (baseline)':>18} | {'Run B (loop curr.)':>18} | {'Run C (full, ours)':>18}")
print("-" * 72)
for i, n in enumerate(EVAL_N_LOOPS):
    acc_A = results_A[i]["accuracy"]
    acc_B = results_B[i]["accuracy"]
    acc_C = results_C[i]["accuracy"]
    marker = " <-" if n > 16 else "  "
    print(f"{n:>8} | {acc_A:>18.3f} | {acc_B:>18.3f} | {acc_C:>18.3f}{marker}")

print(f"\nResults saved to {OUTDIR}/")
print("Run `python plot_results.py` (update OUTDIR) to generate figures.")
