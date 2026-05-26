"""LSTM text classifier with optional multi-seed stability sweep and K-Fold CV.

Three modes (mutually compatible — modes 2/3 ON TOP of mode 1):
    1. Default: a single training run on the 70/10/20 stratified split.
       This is the *original* behaviour — produces all the original
       artifacts (lstm_metrics.json, lstm_confusion_matrix.png, …).
    2. ``--seeds 13 42 2024``: runs the same default split & training
       under each seed → produces ``outputs/tables/lstm_seed_stability.csv``
       with mean ± std for accuracy / macro-F1.
    3. ``--cv-folds N``: Stratified K-Fold on the train+val pool (keeping
       the 20% hold-out untouched); inside each fold an additional 10%
       sliver is used for early-stopping. Produces
       ``outputs/tables/lstm_cv_summary.csv``.

⚠ LSTMs are expensive. Default ``--seeds`` and ``--cv-folds`` to OFF so
   ``python -m src.train_lstm ...`` stays as fast as it was before.
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
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from .config import LSTMConfig, SplitConfig, ensure_dir
from .cross_validation import make_stratified_kfold
from .data_processing import add_clean_columns, load_dataset, stratified_split
from .evaluate import compute_metrics, save_confusion_matrix, save_metrics


# ---------------------------------------------------------------------------
# Model + plotting helpers (unchanged)
# ---------------------------------------------------------------------------

def build_model(num_classes: int, config: LSTMConfig) -> tf.keras.Model:
    model = Sequential([
        Embedding(config.max_words, config.embedding_dim, input_length=config.max_len),
        LSTM(config.lstm_units, dropout=config.dropout, recurrent_dropout=config.recurrent_dropout),
        Dense(config.dense_units, activation="relu"),
        Dropout(config.dropout),
        Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def plot_model_architecture(model: tf.keras.Model, output_dir: Path) -> None:
    try:
        tf.keras.utils.plot_model(
            model,
            to_file=str(output_dir / "figures" / "lstm_architecture.png"),
            show_shapes=True, show_layer_names=True, rankdir="TB", dpi=150,
        )
    except Exception as exc:
        print(f"[warn] plot_model unavailable ({exc}); skipping architecture diagram.")


def plot_history(history: tf.keras.callbacks.History, output_dir: Path) -> None:
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(output_dir / "tables" / "lstm_history.csv", index=False)
    for metric in ("accuracy", "loss"):
        plt.figure(figsize=(8, 5))
        plt.plot(history.history.get(metric, []), label=f"train_{metric}")
        plt.plot(history.history.get(f"val_{metric}", []), label=f"val_{metric}")
        plt.title(f"LSTM {metric.title()} Over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel(metric.title())
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "figures" / f"lstm_{metric}.png", dpi=200)
        plt.close()


# ---------------------------------------------------------------------------
# Single-run training (extracted so multi-seed / K-Fold can reuse it).
# ---------------------------------------------------------------------------

def _prepare_data(df: pd.DataFrame, label_column: str, config: LSTMConfig, seed: int):
    """Stratified 70/10/20 split → tokeniser → padded sequences. Train-fit tokeniser only."""
    train_df, val_df, test_df = stratified_split(
        df, label_column, SplitConfig(random_state=seed),
    )
    tokenizer = Tokenizer(num_words=config.max_words, oov_token="<OOV>")
    # ⚠ fit on training text ONLY — otherwise val/test vocabulary leaks into the model.
    tokenizer.fit_on_texts(train_df["text_clean_neural"])

    def vec(s):
        return pad_sequences(
            tokenizer.texts_to_sequences(s),
            maxlen=config.max_len, padding="post", truncating="post",
        )

    encoder = LabelEncoder()
    y_train = encoder.fit_transform(train_df[label_column])
    y_val = encoder.transform(val_df[label_column])
    y_test = encoder.transform(test_df[label_column])

    return {
        "x_train": vec(train_df["text_clean_neural"]),
        "x_val": vec(val_df["text_clean_neural"]),
        "x_test": vec(test_df["text_clean_neural"]),
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "labels": encoder.classes_.tolist(),
        "encoder": encoder, "tokenizer": tokenizer,
        "train_df": train_df, "val_df": val_df, "test_df": test_df,
    }


def train_one_run(
    df: pd.DataFrame,
    label_column: str,
    text_column: str,
    config: LSTMConfig,
    seed: int,
    output_dir: Path,
    model_dir: Path,
    *,
    save_artifacts: bool = True,
) -> dict:
    """Train one LSTM. Returns the metrics dict; optionally saves all artifacts.

    Setting ``save_artifacts=False`` is what makes multi-seed / K-Fold cheap —
    only the headline run writes models, figures, and predictions to disk.
    """
    set_global_seed(seed)
    data = _prepare_data(df, label_column, config, seed)

    model = build_model(len(data["labels"]), config)
    if save_artifacts:
        plot_model_architecture(model, output_dir)

    early_stop = EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)
    history = model.fit(
        data["x_train"], data["y_train"],
        validation_data=(data["x_val"], data["y_val"]),
        epochs=config.epochs, batch_size=config.batch_size,
        callbacks=[early_stop], verbose=1 if save_artifacts else 0,
    )

    test_probs = model.predict(data["x_test"], verbose=0)
    test_pred_ids = np.argmax(test_probs, axis=1)
    test_preds = data["encoder"].inverse_transform(test_pred_ids)
    y_true = data["encoder"].inverse_transform(data["y_test"])
    metrics = compute_metrics(y_true, test_preds, data["labels"])

    if save_artifacts:
        save_metrics(metrics, model_dir / "lstm_metrics.json")
        save_confusion_matrix(
            y_true, test_preds, data["labels"],
            "LSTM confusion matrix",
            output_dir / "figures" / "lstm_confusion_matrix.png",
        )
        plot_history(history, output_dir)
        model.save(model_dir / "lstm_text_classifier.keras")
        (model_dir / "lstm_label_classes.json").write_text(json.dumps(data["labels"], indent=2))
        (model_dir / "lstm_tokenizer.json").write_text(data["tokenizer"].to_json())
        (model_dir / "lstm_config.json").write_text(json.dumps(config.__dict__, indent=2))
        summary_lines: list[str] = []
        model.summary(print_fn=summary_lines.append)
        (model_dir / "lstm_model_summary.txt").write_text("\n".join(summary_lines))
        pd.DataFrame({
            text_column: data["test_df"][text_column].tolist(),
            label_column: y_true.tolist(),
            "lstm_prediction": test_preds.tolist(),
        }).to_csv(output_dir / "tables" / "lstm_predictions.csv", index=False)

    return {
        "seed": seed,
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
    }


# ---------------------------------------------------------------------------
# K-Fold mode (optional)
# ---------------------------------------------------------------------------

def train_kfold(
    df: pd.DataFrame,
    label_column: str,
    config: LSTMConfig,
    n_splits: int,
    output_dir: Path,
    val_fraction_within_fold: float = 0.1,
) -> pd.DataFrame:
    """K-Fold over the train+val pool. The 20% hold-out is excluded."""
    # Carve out the same hold-out as the default split, then K-Fold on the remainder.
    train_df, val_df, _test_df = stratified_split(df, label_column, SplitConfig())
    pool = pd.concat([train_df, val_df], ignore_index=True)

    skf = make_stratified_kfold(n_splits=n_splits, random_state=42)
    y_pool = pool[label_column].values
    rows = []
    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(pool)), y_pool)):
        print(f"\n── LSTM fold {fold_idx+1}/{n_splits} ──")
        # 💡 Within each outer fold, peel off a small sliver as the
        # early-stopping validation set so we never use the fold's val
        # data for both training AND scoring.
        fold_train_full = pool.iloc[tr_idx].reset_index(drop=True)
        fold_val_eval = pool.iloc[va_idx].reset_index(drop=True)
        val_size = max(1, int(len(fold_train_full) * val_fraction_within_fold))
        fold_val_es = fold_train_full.iloc[:val_size].reset_index(drop=True)
        fold_train = fold_train_full.iloc[val_size:].reset_index(drop=True)

        set_global_seed(42 + fold_idx)
        tok = Tokenizer(num_words=config.max_words, oov_token="<OOV>")
        tok.fit_on_texts(fold_train["text_clean_neural"])
        vec = lambda s: pad_sequences(
            tok.texts_to_sequences(s),
            maxlen=config.max_len, padding="post", truncating="post",
        )
        enc = LabelEncoder()
        y_tr = enc.fit_transform(fold_train[label_column])
        y_es = enc.transform(fold_val_es[label_column])
        y_eval = enc.transform(fold_val_eval[label_column])

        model = build_model(len(enc.classes_), config)
        model.fit(
            vec(fold_train["text_clean_neural"]), y_tr,
            validation_data=(vec(fold_val_es["text_clean_neural"]), y_es),
            epochs=config.epochs, batch_size=config.batch_size,
            callbacks=[EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)],
            verbose=0,
        )
        pred = np.argmax(model.predict(vec(fold_val_eval["text_clean_neural"]), verbose=0), axis=1)
        y_true = enc.inverse_transform(y_eval)
        pred_lbl = enc.inverse_transform(pred)
        m = compute_metrics(y_true, pred_lbl, enc.classes_.tolist())
        rows.append({
            "fold": fold_idx,
            "accuracy": m["accuracy"],
            "macro_f1": m["macro_f1"],
            "weighted_f1": m["weighted_f1"],
        })
        print(f"  fold {fold_idx} → acc={m['accuracy']:.4f} macro-F1={m['macro_f1']:.4f}")

    cv_df = pd.DataFrame(rows)
    cv_df.to_csv(output_dir / "tables" / "lstm_cv_summary.csv", index=False)
    print("\n── LSTM K-Fold summary ──")
    print(cv_df.to_string(index=False))
    print(cv_df.describe().loc[["mean", "std"]])
    return cv_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a TensorFlow LSTM classifier.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--text-column", required=True)
    parser.add_argument("--label-column", required=True)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--seeds",
        type=int, nargs="*", default=None,
        help="If given, run the default split under each seed and report between-seed std. "
             "First seed is the canonical run that saves models + figures.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int, default=0,
        help="If > 0, also run Stratified K-Fold on the train+val pool (expensive).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    ensure_dir(Path(output_dir) / "figures")
    ensure_dir(Path(output_dir) / "tables")
    model_dir = ensure_dir(args.model_dir)

    config = LSTMConfig(epochs=args.epochs)
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

    if args.seeds:
        stab = pd.DataFrame(rows)
        stab.to_csv(Path(output_dir) / "tables" / "lstm_seed_stability.csv", index=False)
        print("\n── LSTM seed stability ──")
        print(stab.to_string(index=False))
        print(f"accuracy  : {stab['accuracy'].mean():.4f} ± {stab['accuracy'].std():.4f}")
        print(f"macro_f1  : {stab['macro_f1'].mean():.4f} ± {stab['macro_f1'].std():.4f}")

    # Mode 3: optional K-Fold (totally separate from the canonical run).
    if args.cv_folds and args.cv_folds > 0:
        train_kfold(df, args.label_column, config, args.cv_folds, Path(output_dir))

    # Echo the canonical-run metrics (back-compat with the original script).
    canonical = rows[0]
    print(json.dumps({k: v for k, v in canonical.items() if k != "seed"}, indent=2))


if __name__ == "__main__":
    main()
