from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from wordcloud import WordCloud

from .config import ensure_dir
from .data_processing import add_clean_columns, clean_text, load_dataset, stratified_split

SALIENCY_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by",
    "for", "from", "has", "have", "he", "her", "hers", "him", "his", "i",
    "in", "into", "is", "it", "its", "of", "on", "or", "our", "she", "that",
    "the", "their", "them", "they", "this", "to", "was", "we", "were", "who",
    "will", "with", "you", "your",
}


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


def _shap_values_array(shap_output, feature_count: int) -> np.ndarray:
    """Normalize SHAP outputs across SHAP versions into samples x features x classes."""
    if isinstance(shap_output, list):
        return np.stack(shap_output, axis=-1)

    values = getattr(shap_output, "values", shap_output)
    values = np.asarray(values)
    if values.ndim == 2:
        return values[:, :, np.newaxis]
    if values.ndim != 3:
        raise ValueError(f"Unexpected SHAP value shape: {values.shape}")

    if values.shape[1] == feature_count:
        return values
    if values.shape[2] == feature_count:
        return np.swapaxes(values, 1, 2)
    raise ValueError(f"Could not align SHAP values with {feature_count} features: {values.shape}")


def _base_value_for_class(explainer, shap_output, sample_index: int, class_index: int, n_classes: int) -> float:
    """Extract the expected value for one class across SHAP versions."""
    base_values = getattr(shap_output, "base_values", None)
    if base_values is None:
        base_values = getattr(explainer, "expected_value", 0.0)

    arr = np.asarray(base_values)
    if arr.ndim == 0:
        return float(arr)
    if arr.ndim == 1:
        if len(arr) == n_classes:
            return float(arr[class_index])
        return float(arr[sample_index])
    if arr.ndim == 2:
        return float(arr[sample_index, class_index])
    raise ValueError(f"Unexpected SHAP base value shape: {arr.shape}")


def _top_shap_features(
    feature_names: np.ndarray,
    shap_vector: np.ndarray,
    candidate_ids: np.ndarray,
    top_n: int = 5,
) -> tuple[str, str]:
    """Return compact positive and negative feature summaries for one prediction."""
    if len(candidate_ids) == 0:
        candidate_ids = np.arange(len(shap_vector))

    positive_ids = sorted(candidate_ids, key=lambda idx: shap_vector[idx], reverse=True)
    negative_ids = sorted(candidate_ids, key=lambda idx: shap_vector[idx])

    positive = [
        f"{feature_names[idx]} ({shap_vector[idx]:+.3f})"
        for idx in positive_ids
        if shap_vector[idx] > 0
    ][:top_n]
    negative = [
        f"{feature_names[idx]} ({shap_vector[idx]:+.3f})"
        for idx in negative_ids
        if shap_vector[idx] < 0
    ][:top_n]
    return "; ".join(positive), "; ".join(negative)


def logistic_regression_shap_artifacts(
    pipeline,
    texts: list[str],
    labels: list[str],
    output_dir: str | Path,
    raw_texts: list[str] | None = None,
    top_n: int = 20,
    background_size: int = 100,
    sample_size: int = 200,
    force_plot_count: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame] | tuple[None, None]:
    """
    Generate SHAP explanations for the saved Logistic Regression TF-IDF pipeline.

    Saves:
    - outputs/tables/logistic_regression_shap_top_features.csv
    - outputs/tables/logistic_regression_shap_force_cases.csv
    - outputs/figures/logistic_regression_shap_top_features.png
    - outputs/figures/logistic_regression_shap_force_plots/*.html
    """
    try:
        import shap
    except ImportError:
        print("[warn] SHAP not available; install `shap` to generate Logistic Regression explanations.")
        return None, None

    output_dir = ensure_dir(output_dir)
    figures_dir = ensure_dir(output_dir / "figures")
    tables_dir = ensure_dir(output_dir / "tables")
    force_dir = ensure_dir(figures_dir / "logistic_regression_shap_force_plots")

    vectorizer = pipeline.named_steps["tfidf"]
    model = pipeline.named_steps["model"]
    if not hasattr(model, "predict_proba"):
        print("[warn] Logistic Regression SHAP requires class probabilities; skipping.")
        return None, None

    clean_texts = pd.Series(texts).fillna("").astype(str).reset_index(drop=True)
    label_series = pd.Series(labels).fillna("").astype(str).reset_index(drop=True)
    display_texts = (
        pd.Series(raw_texts).fillna("").astype(str).reset_index(drop=True)
        if raw_texts is not None
        else clean_texts
    )

    if clean_texts.empty:
        print("[warn] No texts available for Logistic Regression SHAP.")
        return None, None

    x_all = vectorizer.transform(clean_texts)
    feature_names = vectorizer.get_feature_names_out()
    class_names = np.asarray(model.classes_)

    n_background = min(background_size, x_all.shape[0])
    n_samples = min(sample_size, x_all.shape[0])
    background = x_all[:n_background]
    x_explain = x_all[:n_samples]

    explainer = shap.LinearExplainer(model, background)
    try:
        shap_output = explainer(x_explain)
    except TypeError:
        shap_output = explainer.shap_values(x_explain)

    shap_values = _shap_values_array(shap_output, x_explain.shape[1])
    if shap_values.shape[2] == 1 and len(class_names) > 1:
        print("[warn] Unexpected binary-only SHAP output for multiclass Logistic Regression.")

    # Global explanation: highest mean absolute SHAP values per genre.
    global_rows = []
    for class_index, class_name in enumerate(class_names):
        value_index = min(class_index, shap_values.shape[2] - 1)
        class_values = shap_values[:, :, value_index]
        mean_abs = np.abs(class_values).mean(axis=0)
        signed_mean = class_values.mean(axis=0)
        top_ids = mean_abs.argsort()[-top_n:][::-1]
        for rank, feature_id in enumerate(top_ids, start=1):
            global_rows.append(
                {
                    "class": class_name,
                    "rank": rank,
                    "feature": feature_names[feature_id],
                    "mean_abs_shap": float(mean_abs[feature_id]),
                    "mean_signed_shap": float(signed_mean[feature_id]),
                }
            )

    global_df = pd.DataFrame(global_rows)
    global_df.to_csv(tables_dir / "logistic_regression_shap_top_features.csv", index=False)

    fig, axes = plt.subplots(len(class_names), 1, figsize=(10, 3.2 * len(class_names)))
    axes = np.atleast_1d(axes)
    for ax, class_name in zip(axes, class_names):
        group = global_df[global_df["class"] == class_name].sort_values("mean_abs_shap")
        ax.barh(group["feature"], group["mean_abs_shap"], color="#4C78A8")
        ax.set_title(f"Logistic Regression SHAP: {class_name}")
        ax.set_xlabel("Mean |SHAP value|")
    plt.tight_layout()
    plt.savefig(figures_dir / "logistic_regression_shap_top_features.png", dpi=180)
    plt.close(fig)

    # Local explanation: one high-confidence force plot per predicted genre when possible.
    probabilities = model.predict_proba(x_explain)
    predicted_indices = probabilities.argmax(axis=1)
    predicted_labels = class_names[predicted_indices]

    selected_indices: list[int] = []
    labels_array = label_series.iloc[:n_samples].to_numpy()
    for class_name in class_names:
        matches = np.where((predicted_labels == class_name) & (labels_array == class_name))[0]
        if len(matches):
            selected_indices.append(int(matches[0]))

    if len(selected_indices) < force_plot_count:
        confidence_order = np.argsort(probabilities.max(axis=1))[::-1]
        for idx in confidence_order:
            idx = int(idx)
            if idx not in selected_indices:
                selected_indices.append(idx)
            if len(selected_indices) >= force_plot_count:
                break

    force_rows = []
    for case_number, sample_index in enumerate(selected_indices[:force_plot_count], start=1):
        class_index = int(predicted_indices[sample_index])
        value_index = min(class_index, shap_values.shape[2] - 1)
        shap_vector = shap_values[sample_index, :, value_index]
        x_row = x_explain[sample_index]
        dense_row = x_row.toarray()[0]
        nonzero_ids = x_row.nonzero()[1]
        base_value = _base_value_for_class(
            explainer,
            shap_output,
            sample_index=sample_index,
            class_index=value_index,
            n_classes=shap_values.shape[2],
        )

        predicted_label = str(class_names[class_index])
        html_name = f"case_{case_number}_{predicted_label}.html"
        force_plot = shap.force_plot(
            base_value,
            shap_vector,
            dense_row,
            feature_names=feature_names,
            out_names=predicted_label,
            matplotlib=False,
            show=False,
        )
        shap.save_html(str(force_dir / html_name), force_plot)

        top_positive, top_negative = _top_shap_features(feature_names, shap_vector, nonzero_ids)
        force_rows.append(
            {
                "case": case_number,
                "row_index": sample_index,
                "true_label": labels_array[sample_index],
                "predicted_label": predicted_label,
                "predicted_probability": float(probabilities[sample_index, class_index]),
                "force_plot_path": str(force_dir / html_name),
                "top_positive_features": top_positive,
                "top_negative_features": top_negative,
                "text_excerpt": display_texts.iloc[sample_index][:300],
            }
        )

    force_df = pd.DataFrame(force_rows)
    force_df.to_csv(tables_dir / "logistic_regression_shap_force_cases.csv", index=False)
    return global_df, force_df


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

    # Build a sub-model that outputs (embedding_output, final_logits).
    # Replaying the layers from a fresh Input is compatible with Keras versions
    # where a loaded Sequential model may not expose model.output immediately.
    inputs = tf.keras.Input(shape=(max_len,), dtype=tf.int32)
    x = inputs
    embedding_output = None
    for layer in model.layers:
        x = layer(x)
        if embedding_output is None and isinstance(layer, tf.keras.layers.Embedding):
            embedding_output = x

    if embedding_output is None:
        print("[warn] No Embedding layer found; skipping LSTM saliency.")
        return None

    grad_model = tf.keras.Model(inputs=inputs, outputs=[embedding_output, x])

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
        # 💡 Manual forward pass so we can watch the embedding output without
        # constructing a Functional submodel. Each `layer(h, training=False)`
        # call is just `layer.__call__`, which works on any Keras layer
        # regardless of how the parent model was built.
        with tf.GradientTape() as tape:
            emb_out = embedding_layer(x)
            tape.watch(emb_out)
            h = emb_out
            for layer in remaining_layers:
                h = layer(h, training=False)
            logits = h
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
            normalized_word = word.lower()
            if (
                not word
                or word == "<OOV>"
                or normalized_word in SALIENCY_STOPWORDS
                or len(normalized_word) < 3
            ):
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
    df = add_clean_columns(load_dataset(args.input), args.text_column)
    _, _, test_df = stratified_split(df, args.label_column)

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
        if name == "logistic_regression":
            logistic_regression_shap_artifacts(
                pipeline,
                texts=test_df["text_clean_classical"].fillna("").tolist(),
                labels=test_df[args.label_column].fillna("").tolist(),
                raw_texts=test_df[args.text_column].fillna("").tolist(),
                output_dir=output_dir,
            )

    # ── LSTM: gradient saliency word importance + word clouds ─────────────────
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
