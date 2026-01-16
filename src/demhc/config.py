"""Configuration dataclasses for DEMHC."""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class DEQConfig:
    """Configuration for Deep Equilibrium solver."""

    solver: Literal["anderson", "fixed_point"] = "anderson"
    anderson_m: int = 3  # History size for Anderson acceleration (smaller = faster)
    max_iters: int = 8  # Maximum forward iterations (reduced for speed)
    tol: float = 0.15  # Convergence tolerance (relaxed for practical training)
    beta: float = 1.0  # Mixing parameter for Anderson

    # Implicit differentiation settings
    implicit_diff_max_iters: int = 8  # Max iterations for backward solve
    implicit_diff_tol: float = 0.15  # Tolerance for backward solve


@dataclass
class mHCConfig:
    """Configuration for Manifold-Constrained Hyper Connections."""

    num_lanes: int = 4  # Number of parallel lanes
    sinkhorn_iters: int = 10  # Iterations for Sinkhorn-Knopp projection
    sinkhorn_eps: float = 1e-8  # Numerical stability epsilon


@dataclass
class ModelConfig:
    """Configuration for model architecture."""

    # Architecture
    hidden_dim: int = 512
    num_heads: int = 8
    num_layers: int = 4
    ffn_mult: float = 4.0  # FFN dimension = hidden_dim * ffn_mult
    vocab_size: int = 50257  # GPT-2 tokenizer vocab size
    max_seq_len: int = 512
    dropout: float = 0.1

    # DEQ and mHC configs
    deq: DEQConfig = field(default_factory=DEQConfig)
    mhc: mHCConfig = field(default_factory=mHCConfig)

    @property
    def ffn_dim(self) -> int:
        return int(self.hidden_dim * self.ffn_mult)

    @property
    def head_dim(self) -> int:
        return self.hidden_dim // self.num_heads


@dataclass
class TrainConfig:
    """Configuration for training."""

    # Optimization
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    gradient_clip: float = 1.0
    mhc_lr_mult: float = 30.0  # Learning rate multiplier for mHC params

    # Schedule
    warmup_steps: int = 1000
    max_steps: int = 100000
    lr_decay: Literal["cosine", "linear", "constant"] = "cosine"

    # Logging and checkpointing
    eval_interval: int = 500
    log_interval: int = 10
    checkpoint_interval: int = 5000
    num_eval_batches: int = 50

    # Hardware
    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    compile: bool = True  # Use torch.compile


@dataclass
class DataConfig:
    """Configuration for data loading."""

    dataset: str = "roneneldan/TinyStories"
    tokenizer: str = "gpt2"
    max_seq_len: int = 512
    num_workers: int = 4


@dataclass
class ExperimentConfig:
    """Full experiment configuration."""

    name: str = "default"
    model_type: Literal["baseline", "demhc"] = "demhc"
    size: Literal["small", "medium"] = "small"

    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)

    # Output paths
    output_dir: str = "outputs"
    seed: int = 42

    @classmethod
    def small_baseline(cls, name: str = "baseline_small") -> "ExperimentConfig":
        """Create a small baseline transformer config (~10M params)."""
        return cls(
            name=name,
            model_type="baseline",
            size="small",
            model=ModelConfig(
                hidden_dim=384,
                num_heads=6,
                num_layers=6,
                ffn_mult=4.0,
                max_seq_len=512,
                dropout=0.1,
            ),
        )

    @classmethod
    def small_demhc(cls, name: str = "demhc_small") -> "ExperimentConfig":
        """Create a small DEQ-mHC config (~10M params, parameter-matched to baseline)."""
        return cls(
            name=name,
            model_type="demhc",
            size="small",
            model=ModelConfig(
                hidden_dim=384,
                num_heads=6,
                num_layers=6,  # Match baseline layer count for fair comparison
                ffn_mult=4.0,
                max_seq_len=512,
                dropout=0.1,
                deq=DEQConfig(),  # Use relaxed defaults for faster training
                mhc=mHCConfig(num_lanes=6, sinkhorn_iters=10),
            ),
        )

    @classmethod
    def medium_baseline(cls, name: str = "baseline_medium") -> "ExperimentConfig":
        """Create a medium baseline transformer config (~50M params)."""
        return cls(
            name=name,
            model_type="baseline",
            size="medium",
            model=ModelConfig(
                hidden_dim=768,
                num_heads=12,
                num_layers=12,
                ffn_mult=4.0,
                max_seq_len=512,
                dropout=0.1,
            ),
        )

    @classmethod
    def medium_demhc(cls, name: str = "demhc_medium") -> "ExperimentConfig":
        """Create a medium DEQ-mHC config (~50M params, parameter-matched to baseline)."""
        return cls(
            name=name,
            model_type="demhc",
            size="medium",
            model=ModelConfig(
                hidden_dim=768,
                num_heads=12,
                num_layers=12,  # Match baseline layer count for fair comparison
                ffn_mult=4.0,
                max_seq_len=512,
                dropout=0.1,
                deq=DEQConfig(),  # Use relaxed defaults for faster training
                mhc=mHCConfig(num_lanes=12, sinkhorn_iters=10),
            ),
        )
