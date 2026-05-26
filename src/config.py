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
    # n_jobs=-1 → use all cores. Set to 1 if you hit memory issues on small machines.
    n_jobs: int = -1


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
    max_words: int = 15000
    max_len: int = 250
    embedding_dim: int = 128
    lstm_units: int = 128
    dense_units: int = 64
    dropout: float = 0.3
    recurrent_dropout: float = 0.2
    epochs: int = 10
    batch_size: int = 32


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory
