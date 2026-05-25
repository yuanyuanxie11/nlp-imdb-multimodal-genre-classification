from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .config import ensure_dir
from .data_processing import add_clean_columns, load_dataset, stratified_split
from .evaluate import comparison_frame, compute_metrics, save_confusion_matrix, save_metrics


def build_models() -> dict[str, Pipeline]:
    common_vectorizer = dict(
        lowercase=False,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )
    return {
        "naive_bayes": Pipeline(
            [("tfidf", TfidfVectorizer(**common_vectorizer)), ("model", MultinomialNB())]
        ),
        "logistic_regression": Pipeline(
            [
                ("tfidf", TfidfVectorizer(**common_vectorizer)),
                ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
            ]
        ),
        "linear_svm": Pipeline(
            [("tfidf", TfidfVectorizer(**common_vectorizer)), ("model", LinearSVC())]
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train baseline text classifiers.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--text-column", required=True)
    parser.add_argument("--label-column", required=True)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--model-dir", default="models")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    figures_dir = ensure_dir(Path(output_dir) / "figures")
    tables_dir = ensure_dir(Path(output_dir) / "tables")
    model_dir = ensure_dir(args.model_dir)

    df = add_clean_columns(load_dataset(args.input), args.text_column)
    train_df, val_df, test_df = stratified_split(df, args.label_column)
    labels = sorted(df[args.label_column].dropna().unique().tolist())

    x_train = train_df["text_clean_classical"]
    y_train = train_df[args.label_column]
    x_val = val_df["text_clean_classical"]
    y_val = val_df[args.label_column]
    x_test = test_df["text_clean_classical"]
    y_test = test_df[args.label_column]

    results = []
    prediction_table = test_df[[args.text_column, args.label_column]].copy()

    for name, pipeline in build_models().items():
        pipeline.fit(x_train, y_train)
        val_pred = pipeline.predict(x_val)
        test_pred = pipeline.predict(x_test)

        metrics = compute_metrics(y_test, test_pred, labels)
        save_metrics(metrics, model_dir / f"{name}_metrics.json")
        save_confusion_matrix(
            y_test,
            test_pred,
            labels,
            f"{name} confusion matrix",
            figures_dir / f"{name}_confusion_matrix.png",
        )
        joblib.dump(pipeline, model_dir / f"{name}.joblib")
        prediction_table[f"{name}_prediction"] = test_pred

        val_metrics = compute_metrics(y_val, val_pred, labels)
        results.append(
            {
                "model": name,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
                "validation_accuracy": val_metrics["accuracy"],
            }
        )

    comparison = comparison_frame(results)
    comparison.to_csv(tables_dir / "model_comparison.csv", index=False)
    prediction_table.to_csv(tables_dir / "baseline_predictions.csv", index=False)
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
