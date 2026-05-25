from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from wordcloud import WordCloud

from .config import ensure_dir
from .data_processing import add_clean_columns, load_dataset, stratified_split


def top_features_from_linear_pipeline(pipeline, top_n: int = 15) -> pd.DataFrame:
    vectorizer = pipeline.named_steps["tfidf"]
    model = pipeline.named_steps["model"]
    feature_names = vectorizer.get_feature_names_out()

    if not hasattr(model, "coef_"):
        if hasattr(model, "feature_log_prob_"):
            scores = model.feature_log_prob_
        else:
            raise ValueError("Model does not expose linear coefficients.")
    else:
        scores = model.coef_

    rows = []
    for class_index, class_name in enumerate(model.classes_):
        class_scores = scores[class_index]
        top_ids = class_scores.argsort()[-top_n:][::-1]
        for rank, feature_id in enumerate(top_ids, start=1):
            rows.append(
                {
                    "class": class_name,
                    "rank": rank,
                    "feature": feature_names[feature_id],
                    "score": float(class_scores[feature_id]),
                }
            )
    return pd.DataFrame(rows)


def save_wordclouds(feature_df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for class_name, group in feature_df.groupby("class"):
        frequencies = {row["feature"]: max(row["score"], 0.001) for _, row in group.iterrows()}
        wordcloud = WordCloud(width=1000, height=600, background_color="white").generate_from_frequencies(
            frequencies
        )
        wordcloud.to_file(str(output_dir / f"{class_name}_wordcloud.png"))


def disagreement_table(prediction_table: pd.DataFrame, label_column: str) -> pd.DataFrame:
    prediction_cols = [col for col in prediction_table.columns if col.endswith("_prediction")]
    if not prediction_cols:
        return pd.DataFrame()

    disagree_mask = prediction_table[prediction_cols].nunique(axis=1) > 1
    disagreements = prediction_table.loc[disagree_mask].copy()
    disagreements["num_unique_predictions"] = disagreements[prediction_cols].nunique(axis=1)
    disagreements["all_wrong"] = disagreements.apply(
        lambda row: all(row[col] != row[label_column] for col in prediction_cols),
        axis=1,
    )
    return disagreements.sort_values(by=["num_unique_predictions", "all_wrong"], ascending=False)


def merge_prediction_tables(
    baseline_path: Path,
    lstm_path: Path,
    text_column: str,
    label_column: str,
) -> pd.DataFrame:
    baseline_df = pd.read_csv(baseline_path) if baseline_path.exists() else pd.DataFrame()
    lstm_df = pd.read_csv(lstm_path) if lstm_path.exists() else pd.DataFrame()
    if baseline_df.empty:
        return lstm_df
    if lstm_df.empty:
        return baseline_df
    return baseline_df.merge(lstm_df, on=[text_column, label_column], how="left")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate interpretability artifacts.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--text-column", required=True)
    parser.add_argument("--label-column", required=True)
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--output-dir", default="outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    output_dir = ensure_dir(args.output_dir)
    figures_dir = ensure_dir(output_dir / "figures")
    tables_dir = ensure_dir(output_dir / "tables")

    model_names = ["naive_bayes", "logistic_regression", "linear_svm"]
    for name in model_names:
        model_path = model_dir / f"{name}.joblib"
        if not model_path.exists():
            continue
        pipeline = joblib.load(model_path)
        feature_df = top_features_from_linear_pipeline(pipeline)
        feature_df.to_csv(tables_dir / f"{name}_top_features.csv", index=False)
        save_wordclouds(feature_df, figures_dir / f"{name}_wordclouds")

    baseline_path = tables_dir / "baseline_predictions.csv"
    lstm_path = tables_dir / "lstm_predictions.csv"
    if baseline_path.exists() or lstm_path.exists():
        predictions = merge_prediction_tables(
            baseline_path,
            lstm_path,
            args.text_column,
            args.label_column,
        )
    else:
        df = add_clean_columns(load_dataset(args.input), args.text_column)
        _, _, test_df = stratified_split(df, args.label_column)
        predictions = test_df[[args.text_column, args.label_column]].copy()

    disagreements = disagreement_table(predictions, args.label_column)
    disagreements.to_csv(tables_dir / "model_disagreements.csv", index=False)
    print("Saved explanation artifacts to", output_dir)


if __name__ == "__main__":
    main()
