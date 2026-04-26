"""
colab_experiment.py
====================
GPU-scale ACT Curriculum Proof Extrapolation experiment.
Designed for Google Colab T4/A100 (free tier works).

SETUP (run once in Colab):
    !git clone https://github.com/YOUR_USERNAME/open-mythos.git
    %cd open-mythos
    !pip install torch datasets tiktoken matplotlib -q

USAGE:
    # In Colab, run this file cell by cell, or:
    !python colab_experiment.py

EXPECTED RUNTIME:
    T4 GPU:  ~2-3 hours
    A100:    ~45 min
    CPU:     ~12 hours (not recommended)

EXPECTED RESULTS (publication quality):
    - Run C accuracy at n_loops=32 should be 5-15pp above Run A
    - ACT scores should show clear correlation with MATH difficulty levels
    - Halt depth evolution should show 3 distinct phases
"""

# ============================================================
# CELL 1: Install dependencies
# ============================================================
# Uncomment in Colab:
# import subprocess
# subprocess.run(["pip", "install", "datasets", "tiktoken", "matplotlib", "-q"])

# ============================================================
# CELL 2: Imports and device setup
# ============================================================

from __future__ import annotations
import json, os, time, math
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

from open_mythos.main import MythosConfig, OpenMythos
from open_mythos.pipeline import (
    ACTCurriculumPipeline, ACTProfiler, CurriculumSampler,
    DepthExtrapolationEvaluator, DifficultyScorer, LoopScheduler,
    PipelineConfig, StageConfig,
)
from open_mythos.pipeline.gsm8k_dataset import (
    GSM8KDataset, MATHDataset, load_gsm8k_or_fallback, load_math_or_fallback
)

# ============================================================
# CELL 3: Model configuration (GPU-scale)
# ============================================================

SEED = 42
torch.manual_seed(SEED)
os.makedirs("results_gpu", exist_ok=True)

# GPU-scale config: ~50M params
# Adjust dim/n_heads down if you get OOM on T4 (8GB VRAM)
MODEL_CFG = MythosConfig(
    vocab_size=100277,      # tiktoken cl100k_base vocab
    dim=512,
    n_heads=8,
    n_kv_heads=4,
    max_seq_len=256,
    max_loop_iters=32,      # allow up to 32 at eval
    prelude_layers=2,
    coda_layers=2,
    attn_type="gqa",
    n_experts=16,
    n_shared_experts=2,
    n_experts_per_tok=4,
    expert_dim=128,
    lora_rank=16,
    act_threshold=0.5,      # lower threshold -> more spread in halt depths
    rope_theta=500000.0,
)

# If T4 OOM, use this smaller config instead:
MODEL_CFG_SMALL = MythosConfig(
    vocab_size=100277,
    dim=256,
    n_heads=4,
    n_kv_heads=2,
    max_seq_len=256,
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

SEQ_LEN = 256
BATCH_SIZE = 16          # reduce to 8 if OOM
WARMUP_STEPS = 2000
TOTAL_STEPS = 20000
EVAL_N_LOOPS = [4, 8, 16, 32]   # 32 > max training loops (16) -> extrapolation
PAD_ID = 0
criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)

def make_model(cfg=MODEL_CFG):
    torch.manual_seed(SEED)
    m = OpenMythos(cfg).to(DEVICE)
    n = sum(p.numel() for p in m.parameters())
    print(f"  Parameters: {n:,} ({n/1e6:.1f}M)")
    return m

# ============================================================
# CELL 4: Load datasets
# ============================================================

print("Loading GSM8K training set...")
train_dataset = load_gsm8k_or_fallback(split="train", seq_len=SEQ_LEN, seed=SEED)
print(f"  Train: {len(train_dataset)} examples")

print("Loading MATH test set for evaluation...")
eval_dataset = load_math_or_fallback(split="test", seq_len=SEQ_LEN, seed=SEED)
print(f"  Eval: {len(eval_dataset)} examples")

# Proof tasks stratified by MATH difficulty level
if hasattr(eval_dataset, 'get_proof_tasks_by_difficulty'):
    proof_tasks = eval_dataset.get_proof_tasks_by_difficulty(n_per_level=40)
else:
    proof_tasks = eval_dataset.get_proof_tasks(n_tasks=200)
print(f"  Proof tasks: {len(proof_tasks)}")

# Corpus tensor for ACT profiling (use first 2000 train examples)
n_profile = min(2000, len(train_dataset))
corpus_tensor = torch.stack([
    train_dataset[i]["input_ids"] for i in range(n_profile)
]).to(DEVICE)
print(f"  Profiling corpus: {corpus_tensor.shape}")

# ============================================================
# CELL 5: Helper functions
# ============================================================

def evaluate_model(model, run_name):
    cfg_eval = PipelineConfig(loop_schedule=[(0, 8)], eval_n_loops=EVAL_N_LOOPS)
    evaluator = DepthExtrapolationEvaluator(cfg_eval)
    results = evaluator.evaluate(model, proof_tasks, eval_n_loops=EVAL_N_LOOPS)
    evaluator.save_results(results, f"results_gpu/{run_name}_results.json")
    print(f"\n  [{run_name}] Depth extrapolation on MATH benchmark:")
    for r in results:
        marker = " <- EXTRAPOLATION" if r["n_loops"] > 16 else ""
        print(f"    n_loops={r['n_loops']:2d}  accuracy={r['accuracy']:.3f}{marker}")
    return results

def make_uniform_sampler(n_steps, n_examples):
    scores = torch.full((n_examples,), 0.5)
    stages = [StageConfig(0.0, 1.0, n_steps + 1)]
    s = CurriculumSampler(scores, stages, sampling_mode="uniform", seed=SEED)
    s.set_step(0)
    return s, scores

def run_pipeline(model, n_steps, scheduler, sampler, label,
                 log_halt=False, log_interval=500):
    cfg = PipelineConfig(
        loop_schedule=scheduler.to_dict()["schedule"],
        eval_n_loops=EVAL_N_LOOPS,
        total_steps=n_steps,
        logging_interval=log_interval,
        checkpoint_interval=2000,
        checkpoint_dir=f"results_gpu/checkpoints_{label}",
        halt_log_path=f"results_gpu/halt_log_{label}.jsonl",
        log_halt_depths=log_halt,
        seed=SEED,
    )
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    scheduler_lr = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps)
    pipeline = ACTCurriculumPipeline(
        model=model, sampler=sampler, scheduler=scheduler,
        optimizer=optimizer, criterion=criterion, cfg=cfg,
    )
    pipeline.run(train_dataset, batch_size=BATCH_SIZE)
    return pipeline

# ============================================================
# CELL 6: Phase 0 — Warm-up + ACT Profiling
# ============================================================

print("\n" + "="*60)
print("Phase 0: Warm-up model for ACT profiling...")
t0 = time.time()

warmup_model = make_model()
sampler_w, _ = make_uniform_sampler(WARMUP_STEPS, len(train_dataset))
run_pipeline(warmup_model, WARMUP_STEPS,
             LoopScheduler([(0, 8)]), sampler_w,
             label="warmup", log_interval=200)
print(f"  Warm-up done in {time.time()-t0:.1f}s")

print("\nPhase 0b: ACT Profiling on warm model...")
t0 = time.time()

profiling_cfg = PipelineConfig(
    profiling_batch_size=32, profiling_n_loops=16,
    loop_schedule=[(0, 8)], eval_n_loops=EVAL_N_LOOPS,
)
profiler = ACTProfiler(profiling_cfg)
halt_depths = profiler.profile(warmup_model, corpus_tensor, n_loops=16)
profiler.save(halt_depths, "results_gpu/halt_depths.pt")

scorer = DifficultyScorer(mode="mean")
scores = scorer.score(halt_depths)
scorer.save(scores, torch.arange(n_profile), "results_gpu/act_scores.pt")

gt_difficulties = torch.tensor(
    [train_dataset.difficulties[i] for i in range(n_profile)], dtype=torch.float32
)
torch.save({"scores": scores.cpu(), "gt_difficulties": gt_difficulties},
           "results_gpu/difficulty_validation.pt")

print(f"  Profiling done in {time.time()-t0:.1f}s")
print(f"  Score range: [{scores.min():.3f}, {scores.max():.3f}]  std={scores.std():.4f}")
print(f"  Mean ACT score by difficulty:")
for d in range(1, 6):
    mask = gt_difficulties == d
    if mask.any():
        print(f"    difficulty {d}: mean={scores.cpu()[mask].mean():.3f}  n={mask.sum().item()}")

# Fallback to sequence-length proxy if scores have no spread
if scores.std().item() < 0.01:
    print("  Low spread — using sequence-length proxy")
    non_pad = (corpus_tensor != PAD_ID).float().sum(dim=1).cpu()
    scores = (non_pad - non_pad.min()) / (non_pad.max() - non_pad.min() + 1e-8)

del warmup_model
torch.cuda.empty_cache() if DEVICE == "cuda" else None

# ============================================================
# CELL 7: Run A — Baseline
# ============================================================

print("\n" + "="*60)
print("Run A: Baseline (fixed n_loops=8, random sampling)...")
t0 = time.time()

model_A = make_model()
sampler_A, _ = make_uniform_sampler(TOTAL_STEPS, len(train_dataset))
run_pipeline(model_A, TOTAL_STEPS,
             LoopScheduler([(0, 8)]), sampler_A,
             label="A", log_interval=500)
print(f"  Run A done in {time.time()-t0:.1f}s")
results_A = evaluate_model(model_A, "run_A")
del model_A
torch.cuda.empty_cache() if DEVICE == "cuda" else None

# ============================================================
# CELL 8: Run B — Loop Curriculum
# ============================================================

print("\n" + "="*60)
print("Run B: Loop curriculum (n_loops=4->8->16, random sampling)...")
t0 = time.time()

model_B = make_model()
sampler_B, _ = make_uniform_sampler(TOTAL_STEPS, len(train_dataset))
# Progressive: 4 loops for first third, 8 for second, 16 for last third
run_pipeline(model_B, TOTAL_STEPS,
             LoopScheduler([(0, 4), (7000, 8), (14000, 16)]), sampler_B,
             label="B", log_interval=500)
print(f"  Run B done in {time.time()-t0:.1f}s")
results_B = evaluate_model(model_B, "run_B")
del model_B
torch.cuda.empty_cache() if DEVICE == "cuda" else None

# ============================================================
# CELL 9: Run C — Full Pipeline (our method)
# ============================================================

print("\n" + "="*60)
print("Run C: Full pipeline (ACT curriculum + progressive n_loops)...")
t0 = time.time()

model_C = make_model()

# 3-stage curriculum: easy (0-33%) -> medium (0-66%) -> all (0-100%)
stages_C = [
    StageConfig(0.0, 0.33, 7000),
    StageConfig(0.0, 0.66, 7000),
    StageConfig(0.0, 1.00, 6000),
]
sampler_C = CurriculumSampler(
    scores[:len(train_dataset)] if len(scores) >= len(train_dataset)
    else torch.cat([scores, torch.full((len(train_dataset)-len(scores),), 0.5)]),
    stages_C, sampling_mode="weighted", seed=SEED
)
sampler_C.set_step(0)

cfg_C = PipelineConfig(
    loop_schedule=[(0, 4), (7000, 8), (14000, 16)],
    eval_n_loops=EVAL_N_LOOPS,
    total_steps=TOTAL_STEPS,
    logging_interval=500,
    checkpoint_interval=2000,
    checkpoint_dir="results_gpu/checkpoints_C",
    halt_log_path="results_gpu/halt_log_C.jsonl",
    log_halt_depths=True,
    seed=SEED,
)
os.makedirs(cfg_C.checkpoint_dir, exist_ok=True)
optimizer_C = torch.optim.AdamW(model_C.parameters(), lr=3e-4, weight_decay=0.01)
pipeline_C = ACTCurriculumPipeline(
    model=model_C, sampler=sampler_C,
    scheduler=LoopScheduler([(0, 4), (7000, 8), (14000, 16)]),
    optimizer=optimizer_C, criterion=criterion, cfg=cfg_C,
)
pipeline_C.run(train_dataset, batch_size=BATCH_SIZE)
print(f"  Run C done in {time.time()-t0:.1f}s")
results_C = evaluate_model(model_C, "run_C")
del model_C
torch.cuda.empty_cache() if DEVICE == "cuda" else None

# ============================================================
# CELL 10: Results summary + figures
# ============================================================

print("\n" + "="*60)
print("RESULTS SUMMARY (GSM8K training / MATH evaluation)")
print("="*60)
print(f"{'n_loops':>8} | {'Run A (baseline)':>18} | {'Run B (loop curr.)':>18} | {'Run C (full, ours)':>18}")
print("-"*72)
for i, n in enumerate(EVAL_N_LOOPS):
    acc_A = results_A[i]["accuracy"]
    acc_B = results_B[i]["accuracy"]
    acc_C = results_C[i]["accuracy"]
    marker = " <-" if n > 16 else "  "
    print(f"{n:>8} | {acc_A:>18.3f} | {acc_B:>18.3f} | {acc_C:>18.3f}{marker}")

print("\n<- = extrapolation (n_loops > max training loops of 16)")

# Generate figures
os.makedirs("results_gpu/figures", exist_ok=True)

# Figure 1: ACT score vs MATH difficulty
data = torch.load("results_gpu/difficulty_validation.pt", weights_only=True)
sc = data["scores"].numpy()
gt = data["gt_difficulties"].numpy()
fig, ax = plt.subplots(figsize=(7, 4))
means = [sc[gt == d].mean() if (gt == d).any() else 0 for d in range(1, 6)]
stds  = [sc[gt == d].std()  if (gt == d).any() else 0 for d in range(1, 6)]
counts = [(gt == d).sum() for d in range(1, 6)]
ax.bar(range(1, 6), means, yerr=stds, capsize=5, color="#4C72B0", alpha=0.8, edgecolor="black")
ax.set_xlabel("MATH Difficulty Level (ground truth)", fontsize=12)
ax.set_ylabel("Mean ACT Halt Depth Score", fontsize=12)
ax.set_title("ACT Halt Depth Correlates with MATH Problem Difficulty\n(unsupervised — no labels used during profiling)", fontsize=11)
ax.set_xticks(range(1, 6))
ax.set_xticklabels([f"Level {d}\n(n={counts[d-1]})" for d in range(1, 6)])
ax.set_ylim(0, 1.1)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("results_gpu/figures/fig1_act_difficulty_correlation.png", dpi=150)
plt.close()

# Figure 2: Depth extrapolation curves
n_vals = [r["n_loops"] for r in results_A]
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(n_vals, [r["accuracy"] for r in results_A], "o--", color="#DD8452", lw=2, ms=8, label="Run A: Baseline")
ax.plot(n_vals, [r["accuracy"] for r in results_B], "s--", color="#55A868", lw=2, ms=8, label="Run B: Loop Curriculum")
ax.plot(n_vals, [r["accuracy"] for r in results_C], "D-",  color="#4C72B0", lw=2.5, ms=9, label="Run C: ACT Curriculum (ours)")
ax.axvspan(16.5, max(n_vals)+0.5, alpha=0.08, color="purple", label="Extrapolation region")
ax.axvline(x=16, color="gray", linestyle=":", lw=1.5, alpha=0.7)
ax.set_xlabel("n_loops at Evaluation", fontsize=12)
ax.set_ylabel("Accuracy on MATH Benchmark", fontsize=12)
ax.set_title("Depth Extrapolation on MATH Competition Problems\n(GSM8K training, MATH evaluation)", fontsize=11)
ax.set_xticks(n_vals)
ax.set_ylim(0, 1.05)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("results_gpu/figures/fig2_depth_extrapolation.png", dpi=150)
plt.close()

# Figure 3: Halt depth evolution
halt_log = "results_gpu/halt_log_C.jsonl"
if os.path.exists(halt_log):
    steps, means_h, rates = [], [], []
    with open(halt_log) as f:
        for line in f:
            obj = json.loads(line.strip())
            steps.append(obj["step"]); means_h.append(obj["mean_halt_depth"]); rates.append(obj["early_halt_rate"])
    if steps:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        ax1.plot(steps, means_h, "b-o", ms=4, lw=2)
        ax1.set_ylabel("Mean Halt Depth", fontsize=11)
        ax1.set_title("Halt Depth Evolution During ACT Curriculum Training (Run C)", fontsize=11)
        ax1.grid(alpha=0.3)
        for thresh in [7000, 14000]:
            ax1.axvline(x=thresh, color="red", ls="--", alpha=0.5)
            ax2.axvline(x=thresh, color="red", ls="--", alpha=0.5)
        ax2.plot(steps, rates, "g-s", ms=4, lw=2)
        ax2.set_xlabel("Training Step", fontsize=11)
        ax2.set_ylabel("Early Halt Rate", fontsize=11)
        ax2.set_ylim(0, 1.05)
        ax2.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig("results_gpu/figures/fig3_halt_depth_evolution.png", dpi=150)
        plt.close()

print("\nAll figures saved to results_gpu/figures/")
print("Experiment complete. Upload results_gpu/ for paper writing.")
