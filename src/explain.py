from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from wordcloud import WordCloud

from .config import ensure_dir
from .data_processing import add_clean_columns, clean_text, load_dataset, stratified_split


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


def lstm_gradient_saliency(
    model_dir: Path,
    texts: list[str],
    labels: list[str],
    top_n: int = 20,
) -> Optional[pd.DataFrame]:
    """
    Compute gradient-based saliency for an LSTM model.

    For each genre class, average the gradient magnitude of the loss w.r.t.
    the embedding layer across all examples of that class. Sum across the
    embedding dimension to get a per-token importance score, then map token
    ids back to words.

    Returns a DataFrame with columns [class, rank, feature, score],
    matching the shape produced by top_features_from_linear_pipeline.
    """
    try:
        import tensorflow as tf
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        from tensorflow.keras.preprocessing.text import tokenizer_from_json
    except ImportError:
        print("[warn] TensorFlow not available; skipping LSTM saliency.")
        return None

    model_path = model_dir / "lstm_text_classifier.keras"
    tokenizer_path = model_dir / "lstm_tokenizer.json"
    classes_path = model_dir / "lstm_label_classes.json"
    config_path = model_dir / "lstm_config.json"

    for p in [model_path, tokenizer_path, classes_path, config_path]:
        if not p.exists():
            print(f"[warn] Missing {p}; skipping LSTM saliency.")
            return None

    model = tf.keras.models.load_model(model_path)
    tokenizer = tokenizer_from_json(tokenizer_path.read_text())
    class_names = json.loads(classes_path.read_text())
    config = json.loads(config_path.read_text())
    max_len = config.get("max_len", 250)

    # Build a sub-model that outputs (embedding_output, final_logits)
    embedding_layer = model.layers[0]  # Embedding is always first
    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[embedding_layer.output, model.output],
    )

    # Accumulate word-level importance per class: {class_name: {word: total_score}}
    class_word_scores: dict[str, dict[str, float]] = {c: {} for c in class_names}
    class_counts: dict[str, int] = {c: 0 for c in class_names}

    index_to_word = {v: k for k, v in tokenizer.word_index.items()}

    for text, label in zip(texts, labels):
        if label not in class_names:
            continue
        cleaned = clean_text(text)
        seq = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(seq, maxlen=max_len, padding="post", truncating="post")
        x = tf.constant(padded, dtype=tf.int32)

        class_idx = class_names.index(label)
        with tf.GradientTape() as tape:
            emb_out, logits = grad_model(x, training=False)
            tape.watch(emb_out)
            loss = logits[0, class_idx]

        grads = tape.gradient(loss, emb_out)
        if grads is None:
            continue
        # Shape: (1, max_len, embed_dim) → sum across embed_dim → (max_len,)
        token_importance = tf.reduce_sum(tf.abs(grads[0]), axis=-1).numpy()

        token_ids = padded[0]
        for token_id, importance in zip(token_ids, token_importance):
            if token_id == 0:  # padding
                continue
            word = index_to_word.get(int(token_id), "")
            if not word or word == "<OOV>":
                continue
            class_word_scores[label][word] = class_word_scores[label].get(word, 0.0) + float(importance)
        class_counts[label] += 1

    # Normalize by count and pick top_n
    rows = []
    for class_name, word_scores in class_word_scores.items():
        count = max(class_counts[class_name], 1)
        normalized = {w: s / count for w, s in word_scores.items()}
        top_words = sorted(normalized.items(), key=lambda x: x[1], reverse=True)[:top_n]
        for rank, (word, score) in enumerate(top_words, start=1):
            rows.append({"class": class_name, "rank": rank, "feature": word, "score": score})

    return pd.DataFrame(rows) if rows else None


def generate_disagreement_prose(
    disagreements: pd.DataFrame,
    label_column: str,
    text_column: str,
    n_cases: int = 5,
) -> str:
    """
    Generate a markdown prose explanation for the top-N disagreement cases.

    For each case it identifies which models agreed and which differed,
    then lists the most distinctive words in the summary that could explain
    why each model leaned toward its prediction.
    """
    prediction_cols = [c for c in disagreements.columns if c.endswith("_prediction")]
    if not prediction_cols or disagreements.empty:
        return "No disagreement cases found to analyze."

    # Genre keyword hints for explanation heuristics
    genre_signals: dict[str, list[str]] = {
        "action": ["fight", "chase", "weapon", "battle", "explosion", "agent", "war",
                   "mission", "hero", "villain", "shoot", "escape", "thriller", "attack"],
        "comedy": ["funny", "joke", "laugh", "hilarious", "comic", "awkward", "absurd",
                   "ridiculous", "mishap", "prank", "wedding", "date", "quirky"],
        "horror": ["killer", "ghost", "terror", "monster", "evil", "dark", "blood",
                   "demon", "haunt", "murder", "corpse", "supernatural", "scream", "dead"],
        "romance": ["love", "heart", "relationship", "affair", "passion", "kiss",
                    "couple", "marriage", "wedding", "feeling", "attract", "desire"],
    }

    lines = ["# Model Disagreement Analysis\n",
             "These are movies where different models predicted different genres. "
             "The analysis explains which words in each summary likely drove the disagreement.\n"]

    sample = disagreements.head(n_cases)
    for i, (_, row) in enumerate(sample.iterrows(), start=1):
        true_label = str(row.get(label_column, "?")).lower()
        summary = str(row.get(text_column, ""))
        words_lower = set(summary.lower().split())

        pred_map = {col.replace("_prediction", ""): str(row[col]).lower() for col in prediction_cols}
        unique_preds = set(pred_map.values())

        lines.append(f"## Case {i}")
        lines.append(f"**True genre:** {true_label.title()}")
        lines.append(f"**Summary excerpt:** *{summary[:300].strip()}...*\n")
        lines.append("**Model predictions:**")
        for model_name, pred in pred_map.items():
            correct = "✓" if pred == true_label else "✗"
            lines.append(f"- {model_name.replace('_', ' ').title()}: **{pred.title()}** {correct}")

        lines.append("\n**Why the models may disagree:**")

        # Find signals for each predicted genre
        seen_genres = set()
        for pred_genre in unique_preds:
            if pred_genre in seen_genres:
                continue
            seen_genres.add(pred_genre)
            signals = genre_signals.get(pred_genre, [])
            matched = [w for w in signals if w in words_lower]
            if matched:
                lines.append(
                    f"- Words suggesting **{pred_genre.title()}**: {', '.join(f'`{w}`' for w in matched[:6])}."
                )

        # Explain why models split
        if len(unique_preds) > 1:
            # Models that predicted the true label
            correct_models = [m for m, p in pred_map.items() if p == true_label]
            wrong_models = [m for m, p in pred_map.items() if p != true_label]
            if correct_models and wrong_models:
                lines.append(
                    f"\nThe {', '.join(m.replace('_',' ').title() for m in correct_models)} "
                    f"model(s) correctly identified this as **{true_label.title()}**, "
                    f"while {', '.join(m.replace('_',' ').title() for m in wrong_models)} "
                    f"model(s) were misled — likely because the summary uses vocabulary "
                    f"more commonly associated with {next(pred_map[m] for m in wrong_models).title()} "
                    f"films. This is a classic genre-overlap case where surface-level word "
                    f"patterns conflict with deeper narrative context."
                )
        lines.append("")

    lines.append("---\n")
    lines.append(
        "**Key takeaway:** Model disagreements most often occur in genre-blended summaries "
        "where vocabulary from multiple genres overlaps. Classical models (Naive Bayes, Logistic "
        "Regression, SVM) rely entirely on word frequency patterns, while the LSTM can capture "
        "sequential context — but neither is immune to misleading surface signals."
    )
    return "\n".join(lines)


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

    # ── Classical models: top features + word clouds ──────────────────────────
    model_names = ["naive_bayes", "logistic_regression", "linear_svm"]
    for name in model_names:
        model_path = model_dir / f"{name}.joblib"
        if not model_path.exists():
            continue
        pipeline = joblib.load(model_path)
        feature_df = top_features_from_linear_pipeline(pipeline)
        feature_df.to_csv(tables_dir / f"{name}_top_features.csv", index=False)
        save_wordclouds(feature_df, figures_dir / f"{name}_wordclouds")

    # ── LSTM: gradient saliency word importance + word clouds ─────────────────
    df = add_clean_columns(load_dataset(args.input), args.text_column)
    _, _, test_df = stratified_split(df, args.label_column)
    lstm_features = lstm_gradient_saliency(
        model_dir,
        texts=test_df[args.text_column].fillna("").tolist(),
        labels=test_df[args.label_column].fillna("").tolist(),
        top_n=20,
    )
    if lstm_features is not None:
        lstm_features.to_csv(tables_dir / "lstm_top_features.csv", index=False)
        save_wordclouds(lstm_features, figures_dir / "lstm_wordclouds")

    # ── Disagreement table ────────────────────────────────────────────────────
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
        predictions = test_df[[args.text_column, args.label_column]].copy()

    disagreements = disagreement_table(predictions, args.label_column)
    disagreements.to_csv(tables_dir / "model_disagreements.csv", index=False)

    # ── Written disagreement analysis ─────────────────────────────────────────
    if not disagreements.empty:
        prose = generate_disagreement_prose(
            disagreements,
            label_column=args.label_column,
            text_column=args.text_column,
            n_cases=5,
        )
        (tables_dir / "disagreement_analysis.md").write_text(prose, encoding="utf-8")

    print("Saved explanation artifacts to", output_dir)


if __name__ == "__main__":
    main()
