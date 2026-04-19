# 🌌 OpenMythos

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)

> **The Silent Reasoner**: A theoretical reconstruction of the Claude Mythos architecture focusing on Recurrent-Depth Transformers and Systematic Generalization.

---

## 🏛️ Architecture Overview

OpenMythos implements the cutting-edge **Recurrent-Depth Transformer (RDT)** architecture, suspected to be the engine behind the legendary reasoning capabilities of Claude Mythos. Unlike traditional fixed-depth models, OpenMythos recycles its parameters through a looped recurrent block, enabling "latent chain-of-thought" within a single forward pass.

### Key Features:
- **🌀 Recurrent-Depth (RDT)**: A 3-stage pipeline: Prelude → [Looped Recurrent Block]×T → Coda.
- **⚡ Flash Attention Ready**: Native support for `torch.nn.functional.scaled_dot_product_attention` for maximal inference speed.
- **🧠 Hybrid Attention**: Switchable between **MLA** (Multi-Latent Attention) for memory-efficient KV caching and **GQA** (Grouped Query Attention).
- **🎭 Sparse MoE**: DeepSeek-style Mixture of Experts with fine-grained routed experts and always-on shared experts.
- **🛡️ LTI Stability**: Guarantees dynamical stability across arbitrary loop depths using Linear Time-Invariant (LTI) constrained injection.
- **⏱️ ACT Halting**: Adaptive Computation Time allows the model to dynamically decide when to stop "thinking" per token.

---

## 🚀 Getting Started

### Installation

Install directly via pip for the latest version:

```bash
pip install open-mythos
```

Or for development:

```bash
git clone https://github.com/kodeCraze/mythos.git
cd mythos
pip install -e .
```

### Basic Usage

```python
import torch
from open_mythos import OpenMythos, MythosConfig

# Initialize with Multi-Latent Attention (MLA)
config = MythosConfig(attn_type="mla")
model = OpenMythos(config)

# Run a forward pass with 8 reasoning loops
input_ids = torch.randint(0, config.vocab_size, (1, 32))
logits = model(input_ids, n_loops=8)

# Generate with adaptive depth
output = model.generate(input_ids, max_new_tokens=20, n_loops=16)
```

---

## 🗺️ Roadmap to 2026 Mythos

To transform this architectural skeleton into a powerhouse comparable to the hypothesized 2026 Claude Mythos, the following components are recommended:

### 🧩 Recommended Tokenizer
In 2026, efficient tokenization is paramount. I recommend using a **sentencepiece** or **tiktoken** based tokenizer with a large vocabulary (e.g., **128k+ tokens**). 
- **Llama-3/4 Tokenizer**: Excellent for multi-lingual and code support.
- **Claude-style Byte-Pair Encoding**: Optimized for reasoning and long-context windows.

### 📚 Data Strategy for Systematic Generalization
Mythos-level reasoning isn't just about scale; it's about **compositionality**.
1. **Algorithmic Traces**: 100B+ tokens of step-by-step mathematical proofs and logical derivations.
2. **Synthetic Reasoning Chains**: Generated data where the model must learn the *rule* rather than the *answer*.
3. **Multi-Horizon Planning**: Data containing long-term dependencies and nested problem-solving.
4. **Code-Reasoning Interleaving**: Python/Rust code paired with natural language explanations of the "why" behind the logic.

---

## 📜 License

MIT License. See [LICENSE](LICENSE) for details.

---

Created with ❤️ by **kodeCraze**
