# DEMHC: Deep Equilibrium Models with Manifold-Constrained Hyper Connections

A novel LLM architecture combining two recent innovations:

1. **Deep Equilibrium Models (DEQ)**: Models that find fixed points of layer transformations, effectively creating infinite-depth networks with finite parameters.

2. **Manifold-Constrained Hyper Connections (mHC)**: A generalization of residual connections using multiple "lanes" of information flow, with learnable mixing constrained to the Birkhoff polytope (doubly stochastic matrices).

## Key Features

- **Anderson Acceleration** for fast fixed-point convergence
- **Implicit Differentiation** for memory-efficient backpropagation through equilibrium
- **Sinkhorn-Knopp Algorithm** for projecting mixing matrices onto the Birkhoff manifold
- **Multi-lane information flow** with learnable, stable mixing
- Parameter-matched baseline transformer for fair comparison

## Installation

```bash
# Using uv (recommended)
uv pip install -e .

# Or using pip
pip install -e .
```

## Quick Start

### Training

```bash
# Train baseline transformer (small, ~10M params)
demhc-train --model baseline --size small --name baseline_small

# Train DEQ-mHC model (small, ~10M params, parameter-matched)
demhc-train --model demhc --size small --name demhc_small

# Train medium models (~85M params)
demhc-train --model baseline --size medium --name baseline_medium
demhc-train --model demhc --size medium --name demhc_medium
```

### Hyperparameter Tuning

```bash
# Customize DEQ parameters
demhc-train --model demhc \
    --anderson-m 5 \
    --deq-max-iters 30 \
    --deq-tol 1e-5

# Customize mHC parameters  
demhc-train --model demhc \
    --num-lanes 6 \
    --sinkhorn-iters 10

# Customize training
demhc-train --model demhc \
    --batch-size 64 \
    --learning-rate 3e-4 \
    --max-steps 100000 \
    --warmup-steps 1000
```

### Monitoring with TensorBoard

```bash
tensorboard --logdir outputs/
```

This will show loss curves for all experiments, allowing direct comparison.

## Architecture Overview

### Baseline Transformer
Standard GPT-style transformer with:
- Pre-norm architecture
- Multi-head self-attention
- Feed-forward networks with GELU activation
- Learned positional embeddings

### DEQ-mHC Model
Each DEQ-mHC layer:
1. **Lane Mixing**: Combines k lanes using a learnable doubly stochastic matrix (Sigmoid → Sinkhorn-Knopp)
2. **Fixed-Point Iteration**: Finds equilibrium of the layer transformation using Anderson Acceleration
3. **Implicit Differentiation**: Backprop through the fixed point without unrolling iterations

Final aggregation collapses k lanes to 1 using a learnable simplex (softmax).

## Configuration

Configurations are managed through dataclasses in `src/demhc/config.py`:

- `DEQConfig`: Fixed-point solver settings (anderson_m, max_iters, tol)
- `mHCConfig`: Hyper-connection settings (num_lanes, sinkhorn_iters)
- `ModelConfig`: Architecture settings (hidden_dim, num_heads, num_layers)
- `TrainConfig`: Training settings (batch_size, learning_rate, etc.)

## Project Structure

```
src/demhc/
├── config.py              # Configuration dataclasses
├── train.py               # Main training loop
├── data/
│   └── datasets.py        # TinyStories data loading
├── models/
│   ├── baseline.py        # Standard transformer
│   ├── demhc.py           # DEQ-mHC model
│   └── components/
│       ├── attention.py   # Multi-head attention
│       ├── feedforward.py # FFN layers
│       ├── deq.py         # Anderson acceleration + implicit diff
│       ├── sinkhorn.py    # Birkhoff projection
│       └── hyper_conn.py  # Lane mixing
└── utils/
    ├── logging.py         # TensorBoard utilities
    └── checkpointing.py   # Save/load checkpoints
```

## Metrics Logged

- `train/loss`, `val/loss`: Cross-entropy loss
- `train/perplexity`, `val/perplexity`: Perplexity (2^loss)
- `train/grad_norm`: Gradient norm (for monitoring stability)
- `deq/layer_*_iters`: Fixed-point iterations per layer
- `deq/layer_*_residual`: Convergence residual per layer
- `mhc/layer_*_entropy`: Entropy of mixing matrices

## Hardware Requirements

- Designed for single H100 GPU (80GB VRAM)
- Uses bfloat16 mixed precision by default
- Supports torch.compile for additional speedup

## License

MIT
