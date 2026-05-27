"""
train_lstm.py
─────────────
Improved TensorFlow LSTM classifier with:
  • Pre-trained GloVe embeddings (falls back to random if file not found)
  • Conv1D feature extractor before the LSTM
  • Bidirectional LSTM (reads sequences forward AND backward)
  • No recurrent dropout (harmful on small datasets)
  • Reduced model size and dropout to fight overfitting
  • Tunable learning rate

Architecture:
  Input → GloVe Embedding → Conv1D → MaxPooling1D
        → Bidirectional LSTM → Dense(relu) → Dropout → Softmax
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from .runtime import prepare_runtime, set_global_seed

prepare_runtime()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (
    LSTM,
    Bidirectional,
    Conv1D,
    Dense,
    Dropout,
    Embedding,
    GlobalMaxPooling1D,
    MaxPooling1D,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from .config import LSTMConfig, SplitConfig, ensure_dir
from .cross_validation import make_stratified_kfold
from .data_processing import add_clean_columns, load_dataset, stratified_split
from .evaluate import compute_metrics, save_confusion_matrix, save_metrics


# ── GloVe embedding loader ─────────────────────────────────────────────────────

def load_glove_embeddings(
    glove_path: str | Path,
    tokenizer: Tokenizer,
    embedding_dim: int,
) -> np.ndarray | None:
    """
    Load GloVe vectors and build an embedding matrix aligned to the tokenizer
    vocabulary.

    Returns None if the file is missing — caller falls back to random init.

    Download GloVe:
        python -m scripts.download_glove          # included in this repo
      OR manually from https://nlp.stanford.edu/data/glove.6B.zip
      and place glove.6B.100d.txt in the data/ directory.
    """
    path = Path(glove_path)
    if not path.exists():
        print(f"[GloVe] File not found at '{path}'. Falling back to random embeddings.")
        print("[GloVe] Run:  python -m scripts.download_glove  to download automatically.")
        return None

    print(f"[GloVe] Loading vectors from '{path}' …", end=" ", flush=True)
    glove_index: dict[str, np.ndarray] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip().split(" ")
            word = parts[0]
            try:
                vector = np.array(parts[1:], dtype="float32")
                glove_index[word] = vector
            except ValueError:
                continue
    print(f"loaded {len(glove_index):,} vectors.")

    vocab_size = min(tokenizer.num_words or len(tokenizer.word_index) + 1,
                     len(tokenizer.word_index) + 1)
    # Small random values for words not in GloVe
    rng = np.random.default_rng(42)
    embedding_matrix = rng.normal(scale=0.01, size=(vocab_size, embedding_dim)).astype("float32")
    embedding_matrix[0] = 0  # padding token → zero vector

    hits, misses = 0, 0
    for word, idx in tokenizer.word_index.items():
        if idx >= vocab_size:
            continue
        vec = glove_index.get(word)
        if vec is not None and len(vec) == embedding_dim:
            embedding_matrix[idx] = vec
            hits += 1
        else:
            misses += 1

    coverage = hits / max(hits + misses, 1) * 100
    print(f"[GloVe] Vocabulary coverage: {hits:,} hits / {misses:,} misses ({coverage:.1f}%)")
    return embedding_matrix


# ── Model builder ──────────────────────────────────────────────────────────────

def build_model(
    num_classes: int,
    config: LSTMConfig,
    embedding_matrix: np.ndarray | None = None,
) -> tf.keras.Model:
    """
    Build the improved text-classification model.

    Architecture explanation:
    ┌─────────────────────────────────────────────────────┐
    │ Embedding  (GloVe pre-trained, frozen)              │
    │   → gives each token a meaningful 100-d vector      │
    ├─────────────────────────────────────────────────────┤
    │ Conv1D  (optional, use_conv=True)                   │
    │   → extracts local n-gram patterns (like bigrams)   │
    │ MaxPooling1D                                         │
    │   → keeps the strongest signal, halves sequence     │
    ├─────────────────────────────────────────────────────┤
    │ Bidirectional LSTM                                  │
    │   → models long-range context in both directions    │
    │   recurrent_dropout=0 avoids killing small-data     │
    │   gradients                                         │
    ├─────────────────────────────────────────────────────┤
    │ Dense(relu) → Dropout → Dense(softmax)              │
    └─────────────────────────────────────────────────────┘
    """
    vocab_size = (
        embedding_matrix.shape[0]
        if embedding_matrix is not None
        else config.max_words
    )

    layers: list = []

    # Embedding layer
    if embedding_matrix is not None:
        layers.append(
            Embedding(
                input_dim=vocab_size,
                output_dim=config.embedding_dim,
                weights=[embedding_matrix],
                input_length=config.max_len,
                trainable=False,   # freeze GloVe weights; set True to fine-tune
                name="glove_embedding",
            )
        )
    else:
        layers.append(
            Embedding(
                input_dim=config.max_words,
                output_dim=config.embedding_dim,
                input_length=config.max_len,
                name="random_embedding",
            )
        )

    # Optional Conv1D block
    if config.use_conv:
        layers.append(
            Conv1D(
                filters=config.conv_filters,
                kernel_size=config.conv_kernel_size,
                activation="relu",
                padding="same",
                name="conv1d",
            )
        )
        layers.append(MaxPooling1D(pool_size=2, name="maxpool"))

    # Bidirectional LSTM
    lstm_cell = LSTM(
        config.lstm_units,
        dropout=config.dropout,
        recurrent_dropout=config.recurrent_dropout,  # 0.0
        name="lstm",
    )
    if config.bidirectional:
        layers.append(Bidirectional(lstm_cell, name="bidirectional_lstm"))
    else:
        layers.append(lstm_cell)

    # Dense head
    layers.append(Dense(config.dense_units, activation="relu", name="dense"))
    layers.append(Dropout(config.dropout, name="dropout"))
    layers.append(Dense(num_classes, activation="softmax", name="output"))

    model = Sequential(layers)
    model.compile(
        optimizer=Adam(learning_rate=config.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ── GloVe fine-tuning ─────────────────────────────────────────────────────────

def finetune_glove_embeddings(
    model: tf.keras.Model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    config: LSTMConfig,
) -> tf.keras.callbacks.History | None:
    """
    Phase 2 training: unfreeze the GloVe embedding layer and fine-tune at a
    much lower learning rate.

    Why this works:
    - Phase 1 trains the Conv1D, LSTM, and Dense layers while keeping GloVe
      frozen. Once those layers have good weights, we allow the embeddings to
      shift slightly toward genre-specific vocabulary.
    - The 10x lower learning rate (finetune_lr vs learning_rate) prevents
      catastrophic forgetting of GloVe's general semantics.
    - EarlyStopping with patience=3 stops as soon as val_loss stops improving.

    Returns None if fine-tuning is skipped (no GloVe or config.finetune_glove=False).
    """
    # Find the embedding layer
    embedding_layer = next(
        (layer for layer in model.layers if isinstance(layer, tf.keras.layers.Embedding)),
        None,
    )
    if embedding_layer is None or not config.finetune_glove:
        return None

    # Was it frozen in Phase 1?
    if embedding_layer.trainable:
        print("[fine-tune] Embedding was already trainable — skipping Phase 2.")
        return None

    print("\n── Phase 2: GloVe fine-tuning (unfreezing embedding layer) ──")
    embedding_layer.trainable = True

    # Recompile at the lower learning rate
    model.compile(
        optimizer=Adam(learning_rate=config.finetune_lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    ft_callbacks = [
        EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1, min_lr=1e-7, verbose=1),
    ]

    history_ft = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=config.finetune_epochs,
        batch_size=config.batch_size,
        callbacks=ft_callbacks,
        verbose=1,
    )
    return history_ft


def plot_combined_history(
    history_p1: tf.keras.callbacks.History,
    history_p2: tf.keras.callbacks.History | None,
    output_dir: Path,
) -> None:
    """Plot Phase 1 and Phase 2 training curves on the same axes."""
    p1_acc  = history_p1.history.get("accuracy", [])
    p1_vacc = history_p1.history.get("val_accuracy", [])
    p1_loss = history_p1.history.get("loss", [])
    p1_vloss= history_p1.history.get("val_loss", [])

    p2_acc  = (history_p2.history.get("accuracy",     []) if history_p2 else [])
    p2_vacc = (history_p2.history.get("val_accuracy", []) if history_p2 else [])
    p2_loss = (history_p2.history.get("loss",         []) if history_p2 else [])
    p2_vloss= (history_p2.history.get("val_loss",     []) if history_p2 else [])

    epochs_p1 = list(range(1, len(p1_acc) + 1))
    epochs_p2 = list(range(len(p1_acc) + 1, len(p1_acc) + len(p2_acc) + 1))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for metric_idx, (metric_label, p1_vals, p2_vals, p1_vvals, p2_vvals) in enumerate([
        ("Accuracy", p1_acc, p2_acc, p1_vacc, p2_vacc),
        ("Loss",     p1_loss, p2_loss, p1_vloss, p2_vloss),
    ]):
        ax = axes[metric_idx]
        ax.plot(epochs_p1, p1_vals,  label="Train (Phase 1)",  color="steelblue",  linewidth=2)
        ax.plot(epochs_p1, p1_vvals, label="Val (Phase 1)",    color="steelblue",  linewidth=2, linestyle="--")
        if epochs_p2:
            ax.plot(epochs_p2, p2_vals,  label="Train (Phase 2 fine-tune)", color="tomato", linewidth=2)
            ax.plot(epochs_p2, p2_vvals, label="Val (Phase 2 fine-tune)",   color="tomato", linewidth=2, linestyle="--")
            ax.axvline(len(p1_acc) + 0.5, color="gray", linestyle=":", linewidth=1.2, label="Fine-tune start")
        ax.set_title(f"LSTM {metric_label} — Phase 1 + Fine-tune", fontsize=12)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric_label)
        ax.legend(fontsize=8)
        if metric_label == "Accuracy":
            ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(output_dir / "figures" / "lstm_training_curves.png", dpi=200)
    plt.close()

    # Also save individual metrics for backward compat
    all_acc  = p1_acc  + p2_acc
    all_vacc = p1_vacc + p2_vacc
    all_loss = p1_loss + p2_loss
    all_vloss= p1_vloss + p2_vloss
    history_df = pd.DataFrame({
        "accuracy": all_acc, "val_accuracy": all_vacc,
        "loss": all_loss,    "val_loss": all_vloss,
    })
    history_df.to_csv(output_dir / "tables" / "lstm_history.csv", index=False)


# ── Architecture diagram ───────────────────────────────────────────────────────

def plot_model_architecture(model: tf.keras.Model, output_dir: Path) -> None:
    arch_path = output_dir / "figures" / "lstm_architecture.png"
    try:
        tf.keras.utils.plot_model(
            model,
            to_file=str(arch_path),
            show_shapes=True,
            show_layer_names=True,
            rankdir="TB",
            dpi=150,
        )
        if arch_path.exists():
            print(f"Saved model architecture diagram → {arch_path}")
        else:
            print(
                "[warn] plot_model unavailable; install pydot==1.4.2 and "
                "system Graphviz (macOS: brew install graphviz)."
            )
    except Exception as exc:
        print(f"[warn] plot_model unavailable ({exc}); skipping architecture diagram.")


# ── Training curves ────────────────────────────────────────────────────────────

def plot_history(history: tf.keras.callbacks.History, output_dir: Path) -> None:
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(output_dir / "tables" / "lstm_history.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    axes[0].plot(history_df["accuracy"],     label="Train",      marker="o", linewidth=2)
    axes[0].plot(history_df["val_accuracy"], label="Validation", marker="s",
                 linestyle="--", linewidth=2)
    axes[0].set_title("LSTM Accuracy Over Epochs", fontsize=12)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].set_ylim(0, 1.05)

    axes[1].plot(history_df["loss"],     label="Train",      marker="o",
                 color="tomato", linewidth=2)
    axes[1].plot(history_df["val_loss"], label="Validation", marker="s",
                 color="salmon", linestyle="--", linewidth=2)
    axes[1].set_title("LSTM Loss Over Epochs", fontsize=12)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_dir / "figures" / "lstm_training_curves.png", dpi=200)
    plt.close()

    # Also save individual files for backward compat
    for metric in ("accuracy", "loss"):
        plt.figure(figsize=(8, 5))
        plt.plot(history_df[metric],         label=f"train_{metric}")
        plt.plot(history_df[f"val_{metric}"], label=f"val_{metric}")
        plt.title(f"LSTM {metric.title()} Over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel(metric.title())
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "figures" / f"lstm_{metric}.png", dpi=200)
        plt.close()


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an improved TensorFlow LSTM classifier.")
    parser.add_argument("--input",        required=True)
    parser.add_argument("--text-column",  required=True)
    parser.add_argument("--label-column", required=True)
    parser.add_argument("--output-dir",   default="outputs")
    parser.add_argument("--model-dir",    default="models")
    parser.add_argument("--glove-path",   default="data/glove.6B.100d.txt",
                        help="Path to GloVe vectors. Download with: python -m scripts.download_glove")
    parser.add_argument("--epochs",       type=int, default=20)
    parser.add_argument("--no-glove",     action="store_true",
                        help="Skip GloVe loading and use random embeddings.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    ensure_dir(Path(output_dir) / "figures")
    ensure_dir(Path(output_dir) / "tables")
    model_dir = ensure_dir(args.model_dir)

    config = LSTMConfig(
        epochs=args.epochs,
        glove_path=args.glove_path,
        use_glove=not args.no_glove,
    )

    # ── Data ──────────────────────────────────────────────────────────────────
    df = add_clean_columns(load_dataset(args.input), args.text_column)

    # Mode 1 + 2: single canonical run (always), plus optional extra seeds.
    seeds: Iterable[int] = args.seeds if args.seeds else [42]
    rows = []
    for i, seed in enumerate(seeds):
        save_artifacts = (i == 0)  # Only the first seed produces the canonical model.
        print(f"\n=== LSTM run with seed={seed} (save_artifacts={save_artifacts}) ===")
        rows.append(train_one_run(
            df, args.label_column, args.text_column,
            config, seed, Path(output_dir), model_dir,
            save_artifacts=save_artifacts,
        ))

    def vectorize(texts):
        seqs = tokenizer.texts_to_sequences(texts)
        return pad_sequences(seqs, maxlen=config.max_len, padding="post", truncating="post")

    x_train = vectorize(train_df["text_clean_neural"])
    x_val   = vectorize(val_df["text_clean_neural"])
    x_test  = vectorize(test_df["text_clean_neural"])

    encoder = LabelEncoder()
    y_train = encoder.fit_transform(train_df[args.label_column])
    y_val   = encoder.transform(val_df[args.label_column])
    y_test  = encoder.transform(test_df[args.label_column])
    labels  = encoder.classes_.tolist()

    # ── GloVe ─────────────────────────────────────────────────────────────────
    embedding_matrix = None
    if config.use_glove:
        embedding_matrix = load_glove_embeddings(config.glove_path, tokenizer, config.embedding_dim)

    glove_used = embedding_matrix is not None
    print(f"Embedding: {'GloVe (pre-trained, frozen)' if glove_used else 'random (trainable)'}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(len(labels), config, embedding_matrix)
    model.summary()
    plot_model_architecture(model, Path(output_dir))

    # ── Training ──────────────────────────────────────────────────────────────
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True,
                      verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2,
                          min_lr=1e-6, verbose=1),
    ]

    print("\n── Phase 1: training with frozen GloVe embeddings ──")
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=config.epochs,
        batch_size=config.batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    # ── Phase 2: GloVe fine-tuning ────────────────────────────────────────────
    history_ft = None
    if glove_used and config.finetune_glove:
        history_ft = finetune_glove_embeddings(
            model, x_train, y_train, x_val, y_val, config
        )

    # Plot combined Phase 1 + Phase 2 curves (or just Phase 1 if no fine-tuning)
    if history_ft is not None:
        plot_combined_history(history, history_ft, Path(output_dir))
    else:
        plot_history(history, Path(output_dir))

    # ── Evaluation ────────────────────────────────────────────────────────────
    test_probs    = model.predict(x_test, verbose=0)
    test_pred_ids = np.argmax(test_probs, axis=1)
    test_preds    = encoder.inverse_transform(test_pred_ids)
    y_true        = encoder.inverse_transform(y_test)

    metrics = compute_metrics(y_true, test_preds, labels)
    save_metrics(metrics, model_dir / "lstm_metrics.json")
    save_confusion_matrix(
        y_true, test_preds, labels,
        "LSTM confusion matrix",
        Path(output_dir) / "figures" / "lstm_confusion_matrix.png",
    )

    # ── Save artefacts ────────────────────────────────────────────────────────
    model.save(model_dir / "lstm_text_classifier.keras")
    (model_dir / "lstm_label_classes.json").write_text(json.dumps(labels, indent=2))
    (model_dir / "lstm_tokenizer.json").write_text(tokenizer.to_json())
    (model_dir / "lstm_config.json").write_text(json.dumps({
        **config.__dict__,
        "glove_used": glove_used,
    }, indent=2))

    summary_lines: list[str] = []
    model.summary(print_fn=summary_lines.append)
    (model_dir / "lstm_model_summary.txt").write_text("\n".join(summary_lines))

    pd.DataFrame({
        args.text_column:   test_df[args.text_column].tolist(),
        args.label_column:  y_true.tolist(),
        "lstm_prediction":  test_preds.tolist(),
    }).to_csv(Path(output_dir) / "tables" / "lstm_predictions.csv", index=False)

    print(json.dumps({k: v for k, v in metrics.items() if k != "report"}, indent=2))


if __name__ == "__main__":
    main()
