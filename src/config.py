from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SplitConfig:
    test_size: float = 0.2
    validation_size: float = 0.1
    random_state: int = 42


@dataclass(frozen=True)
class CVConfig:
    # K for the primary Stratified K-Fold loop used by baselines and (optionally) the LSTM.
    n_splits: int = 5
    # Outer/inner K for nested CV (outer = honest generalisation, inner = hyperparameter search).
    outer_splits: int = 5
    inner_splits: int = 3
    # Seeds used by multi-seed stability sweeps. Three is the minimum to compute a meaningful std.
    # 💡 tuple (not list) because frozen dataclasses can't hold mutable defaults.
    seeds: tuple[int, ...] = (13, 42, 2024)
    # Metrics tracked across every fold. Macro-F1 is the primary headline for imbalanced multi-class.
    scoring: tuple[str, ...] = ("accuracy", "f1_macro", "f1_weighted")
    # Single-process CV avoids joblib/loky semaphore issues in restricted environments.
    n_jobs: int = 1


@dataclass(frozen=True)
class TextCleaningConfig:
    lowercase: bool = True
    strip_html: bool = True
    remove_punctuation: bool = True
    collapse_whitespace: bool = True
    remove_stopwords: bool = False
    lemmatize: bool = False


@dataclass(frozen=True)
class LSTMConfig:
    # Vocabulary / sequence
    max_words: int = 15000
    max_len: int = 250

    # Embedding
    embedding_dim: int = 100        # 100-d matches GloVe 6B.100d
    use_glove: bool = True          # load pre-trained GloVe vectors if available
    glove_path: str = "data/glove.6B.100d.txt"

    # Conv1D feature extractor (applied before LSTM)
    use_conv: bool = True
    conv_filters: int = 128
    conv_kernel_size: int = 5

    # LSTM
    bidirectional: bool = True      # read sequences forward AND backward
    lstm_units: int = 64            # smaller = less overfitting on small datasets
    recurrent_dropout: float = 0.0  # 0 on small datasets — kills gradients otherwise

    # Dense head
    dense_units: int = 64
    dropout: float = 0.2            # reduced from 0.3

    # Training — Phase 1 (frozen embeddings)
    epochs: int = 20                # more epochs; early-stopping will cut early
    batch_size: int = 32
    learning_rate: float = 5e-4    # slightly lower than Adam default (1e-3)

    # Training — Phase 2 (GloVe fine-tuning)
    # After Phase 1 converges, unfreeze the embedding layer and train at a much
    # lower learning rate so the GloVe vectors adapt to genre vocabulary without
    # "forgetting" the general semantics they learned on 6B tokens.
    finetune_glove: bool = True     # only applies when use_glove=True
    finetune_epochs: int = 10       # max additional epochs (EarlyStopping cuts early)
    finetune_lr: float = 5e-5      # ~10x smaller than Phase 1 lr


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory
