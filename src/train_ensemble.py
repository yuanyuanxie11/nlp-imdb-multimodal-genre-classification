from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences

from .app_helpers import load_lstm_bundle
from .config import ensure_dir
from .data_processing import add_clean_columns, load_dataset, stratified_split
from .evaluate import compute_metrics, save_confusion_matrix, save_metrics

ENSEMBLE_MODEL_NAME = "ensemble_soft_vote"


def align_probabilities_to_labels(
    probabilities: np.ndarray,
    source_labels: list[str],
    target_labels: list[str],
) -> np.ndarray:
    label_to_index = {label: idx for idx, label in enumerate(source_labels)}
    missing = [label for label in target_labels if label not in label_to_index]
    if missing:
        raise ValueError(f"Probability labels are missing expected classes: {missing}")
    ordered_indices = [label_to_index[label] for label in target_labels]
    return probabilities[:, ordered_indices]


def normalize_weights(weights: list[float] | None, member_count: int) -> np.ndarray:
    if member_count <= 0:
        raise ValueError("At least one ensemble member is required.")
    if weights is None:
        return np.full(member_count, 1.0 / member_count)
    if len(weights) != member_count:
        raise ValueError("Weights must have the same length as ensemble members.")

    weight_array = np.asarray(weights, dtype=float)
    if np.any(weight_array < 0):
        raise ValueError("Ensemble weights must be non-negative.")
    weight_sum = float(weight_array.sum())
    if weight_sum <= 0:
        raise ValueError("At least one ensemble weight must be greater than zero.")
    return weight_array / weight_sum


def soft_vote_probabilities(
    probability_arrays: list[np.ndarray],
    weights: list[float] | np.ndarray | None = None,
) -> np.ndarray:
    if not probability_arrays:
        raise ValueError("At least one model probability array is required.")
    first_shape = probability_arrays[0].shape
    if any(probs.shape != first_shape for probs in probability_arrays):
        shapes = [probs.shape for probs in probability_arrays]
        raise ValueError(f"All probability arrays must have the same shape; got {shapes}")
    normalized_weights = normalize_weights(None if weights is None else list(weights), len(probability_arrays))
    stacked = np.stack(probability_arrays, axis=0)
    return np.tensordot(normalized_weights, stacked, axes=(0, 0))


def merge_comparison_row(comparison: pd.DataFrame, row: dict) -> pd.DataFrame:
    updated = comparison[comparison["model"] != row["model"]].copy() if not comparison.empty else comparison
    updated = pd.concat([pd.DataFrame([row]), updated], ignore_index=True)
    sort_cols = [col for col in ["accuracy", "macro_f1"] if col in updated.columns]
    if sort_cols:
        updated = updated.sort_values(by=sort_cols, ascending=False, ignore_index=True)
    return updated


def baseline_probabilities(pipeline, texts: pd.Series, labels: list[str]) -> np.ndarray:
    if not hasattr(pipeline, "predict_proba"):
        raise ValueError(
            f"{type(pipeline).__name__} does not expose predict_proba. "
            "Use calibrated classifiers for soft voting."
        )
    probs = pipeline.predict_proba(texts)
    return align_probabilities_to_labels(probs, list(pipeline.classes_), labels)


def lstm_probabilities(bundle: dict, texts: pd.Series, labels: list[str]) -> np.ndarray:
    tokenizer = bundle["tokenizer"]
    config = bundle["config"]
    sequences = tokenizer.texts_to_sequences(texts.fillna("").astype(str))
    padded = pad_sequences(
        sequences,
        maxlen=config["max_len"],
        padding="post",
        truncating="post",
    )
    probs = bundle["model"].predict(padded, verbose=0)
    return align_probabilities_to_labels(probs, list(bundle["classes"]), labels)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a soft-voting ensemble over trained models.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--text-column", required=True)
    parser.add_argument("--label-column", required=True)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument(
        "--members",
        nargs="*",
        default=["logistic_regression", "linear_svm", "lstm"],
        help="Models to include. Supported: saved baseline joblib stems and lstm.",
    )
    parser.add_argument(
        "--weights",
        nargs="*",
        type=float,
        default=None,
        help="Optional soft-voting weights, in the same order as --members. Values are normalized.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    figures_dir = ensure_dir(Path(output_dir) / "figures")
    tables_dir = ensure_dir(Path(output_dir) / "tables")
    model_dir = ensure_dir(args.model_dir)

    df = add_clean_columns(load_dataset(args.input), args.text_column)
    _, _, test_df = stratified_split(df, args.label_column)
    labels = sorted(df[args.label_column].dropna().astype(str).unique().tolist())

    probability_arrays: list[np.ndarray] = []
    member_names: list[str] = []

    for member in args.members:
        if member == "lstm":
            bundle = load_lstm_bundle(model_dir)
            if bundle is None:
                raise FileNotFoundError(
                    "Missing LSTM model bundle. Run src.train_lstm before ensemble evaluation."
                )
            probability_arrays.append(
                lstm_probabilities(bundle, test_df["text_clean_neural"], labels)
            )
            member_names.append("lstm")
            continue

        model_path = model_dir / f"{member}.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing baseline model artifact: {model_path}")
        pipeline = joblib.load(model_path)
        probability_arrays.append(
            baseline_probabilities(pipeline, test_df["text_clean_classical"], labels)
        )
        member_names.append(member)

    ensemble_weights = normalize_weights(args.weights, len(probability_arrays))
    ensemble_probs = soft_vote_probabilities(probability_arrays, ensemble_weights)
    pred_indices = np.argmax(ensemble_probs, axis=1)
    predictions = [labels[idx] for idx in pred_indices]
    y_true = test_df[args.label_column].astype(str).tolist()

    metrics = compute_metrics(y_true, predictions, labels)
    save_metrics(metrics, model_dir / "ensemble_metrics.json")
    save_confusion_matrix(
        y_true,
        predictions,
        labels,
        "Soft-voting ensemble confusion matrix",
        figures_dir / "ensemble_confusion_matrix.png",
    )

    prediction_df = test_df[[args.text_column, args.label_column]].copy()
    prediction_df[f"{ENSEMBLE_MODEL_NAME}_prediction"] = predictions
    for label_index, label in enumerate(labels):
        prediction_df[f"{ENSEMBLE_MODEL_NAME}_prob_{label}"] = ensemble_probs[:, label_index]
    prediction_df.to_csv(tables_dir / "ensemble_predictions.csv", index=False)

    row = {
        "model": ENSEMBLE_MODEL_NAME,
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "ensemble_members": "+".join(member_names),
        "ensemble_weights": "+".join(
            f"{member}:{weight:.3f}" for member, weight in zip(member_names, ensemble_weights)
        ),
    }
    comparison_path = tables_dir / "model_comparison.csv"
    comparison = pd.read_csv(comparison_path) if comparison_path.exists() else pd.DataFrame()
    merge_comparison_row(comparison, row).to_csv(comparison_path, index=False)

    metadata = {
        "model": ENSEMBLE_MODEL_NAME,
        "members": member_names,
        "weights": {
            member: float(weight) for member, weight in zip(member_names, ensemble_weights)
        },
        "labels": labels,
    }
    (model_dir / "ensemble_config.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps({k: v for k, v in row.items() if k != "ensemble_members"}, indent=2))


if __name__ == "__main__":
    main()
