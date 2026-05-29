from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from .runtime import prepare_runtime

prepare_runtime()

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split

from .config import SplitConfig, TextCleaningConfig, ensure_dir

# 💡 sklearn ships a 318-word English stopword list. Using it (instead of a
# hand-rolled 36-word set) removes high-frequency pronouns / auxiliaries /
# prepositions like "she", "him", "when", "but" that were drowning out the
# real signal in LDA topics and downstream bag-of-words models.
# ⚠️ ENGLISH_STOP_WORDS is a frozenset — wrap it in a mutable set before
# adding domain words.
DOMAIN_STOPWORDS = {
    # Movie-plot filler that is genre-agnostic and adds no discriminative signal.
    # Add or remove items here when EDA surfaces new noise.
    "film", "movie", "story", "life", "lives", "live",
    "year", "years", "day", "days", "night", "time",
    "one", "two", "three", "first", "second",
    "young", "old", "new",
    "man", "woman", "men", "women", "people", "person",
    "find", "finds", "found", "tell", "tells", "told",
    "say", "says", "said", "go", "goes", "went", "get", "gets", "got",
    "make", "makes", "made", "take", "takes", "took",
    "back", "way", "world",
}

STOPWORDS = set(ENGLISH_STOP_WORDS) | DOMAIN_STOPWORDS


def load_dataset(input_path: str | Path) -> pd.DataFrame:
    path = Path(input_path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file type: {path.suffix}")


def clean_text(text: str, config: TextCleaningConfig | None = None) -> str:
    config = config or TextCleaningConfig()
    value = "" if text is None else str(text)

    if config.lowercase:
        value = value.lower()
    if config.strip_html:
        value = re.sub(r"<[^>]+>", " ", value)
    value = re.sub(r"http\S+|www\.\S+", " ", value)
    value = re.sub(r"[^a-z0-9\s']", " ", value) if config.remove_punctuation else value
    if config.remove_stopwords:
        value = " ".join(token for token in value.split() if token not in STOPWORDS)
    if config.collapse_whitespace:
        value = re.sub(r"\s+", " ", value).strip()
    return value


def basic_profile(df: pd.DataFrame, text_column: str, label_column: str) -> dict:
    """Lightweight dataset profile. Backward-compatible: new keys added, none removed/renamed."""
    text_series = df[text_column].fillna("").astype(str)
    token_counts = text_series.str.split().map(len)
    char_counts = text_series.str.len()
    # Cheap sentence count → split on .!? without dragging in heavy NLP tooling.
    sentence_counts = text_series.map(lambda s: len(re.findall(r"[.!?]+", s)) if s else 0)
    label_counts = df[label_column].value_counts(dropna=False).to_dict()
    total_labels = int(df[label_column].notna().sum())
    label_pct = {
        str(k): float(v) / total_labels * 100 if total_labels else 0.0
        for k, v in label_counts.items()
    }
    vocab = Counter()
    for tokens in text_series.str.lower().str.split():
        vocab.update(tokens)
    total_tokens = sum(vocab.values()) or 1

    return {
        # ---- legacy keys (unchanged) ----
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": df.columns.tolist(),
        "missing_by_column": {k: int(v) for k, v in df.isna().sum().to_dict().items()},
        "duplicate_rows": int(df.duplicated().sum()),
        "label_distribution": label_counts,
        "summary_word_count": {
            "mean": float(token_counts.mean()),
            "median": float(token_counts.median()),
            "min": int(token_counts.min()),
            "max": int(token_counts.max()),
        },
        "top_words_before_cleaning": vocab.most_common(30),
        # ---- new keys (additive only) ----
        "summary_char_count": {
            "mean": float(char_counts.mean()),
            "median": float(char_counts.median()),
            "min": int(char_counts.min()),
            "max": int(char_counts.max()),
        },
        "sentence_count": {
            "mean": float(sentence_counts.mean()),
            "median": float(sentence_counts.median()),
            "min": int(sentence_counts.min()),
            "max": int(sentence_counts.max()),
        },
        "label_distribution_pct": label_pct,
        "null_label_count": int(df[label_column].isna().sum()),
        "unique_word_ratio": float(len(vocab) / total_tokens),  # corpus-level type/token ratio
    }


def add_clean_columns(df: pd.DataFrame, text_column: str) -> pd.DataFrame:
    classical_cfg = TextCleaningConfig(remove_stopwords=True)
    light_cfg = TextCleaningConfig(remove_stopwords=False)
    enriched = df.copy()
    enriched["text_clean_classical"] = enriched[text_column].fillna("").map(
        lambda x: clean_text(x, classical_cfg)
    )
    enriched["text_clean_neural"] = enriched[text_column].fillna("").map(
        lambda x: clean_text(x, light_cfg)
    )
    enriched["text_word_count"] = enriched[text_column].fillna("").astype(str).str.split().map(len)
    return enriched


def save_eda_artifacts(
    df: pd.DataFrame,
    text_column: str,
    label_column: str,
    output_dir: str | Path,
    *,
    run_advanced: bool = True,
) -> None:
    """Generate the full EDA artifact set.

    Legacy artifacts (genre_distribution.png, summary_length_distribution.png,
    word_count_by_genre.csv, dataset_preview.csv) are always produced for
    backward compatibility.

    When ``run_advanced=True`` (default), also calls into ``eda_advanced`` to
    produce the deeper analyses (n-grams, distinctive terms, vocab uniqueness,
    quality report, OOV curve, class overlap, cleaning impact, readability,
    POS/entity — last two are graceful no-ops without their optional deps).
    """
    output_dir = ensure_dir(output_dir)
    figures_dir = ensure_dir(output_dir / "figures")
    tables_dir = ensure_dir(output_dir / "tables")

    # ---- legacy artifacts (unchanged outputs) ----
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x=label_column, order=df[label_column].value_counts().index)
    plt.title("Genre Distribution")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(figures_dir / "genre_distribution.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.histplot(df["text_word_count"], bins=30, kde=True)
    plt.title("Summary Length Distribution")
    plt.xlabel("Word count")
    plt.tight_layout()
    plt.savefig(figures_dir / "summary_length_distribution.png", dpi=200)
    plt.close()

    by_genre = (
        df.groupby(label_column)["text_word_count"]
        .agg(["mean", "median", "min", "max", "count"])
        .reset_index()
    )
    by_genre.to_csv(tables_dir / "word_count_by_genre.csv", index=False)
    df.head(20).to_csv(tables_dir / "dataset_preview.csv", index=False)

    if not run_advanced:
        return

    # ---- advanced artifacts (delegate to src.eda_advanced) ----
    # 💡 Lazy import → avoids a circular-import risk and keeps `data_processing`
    # importable even if a downstream user strips eda_advanced.
    try:
        from . import eda_advanced as ea
    except ImportError as exc:
        print(f"[data_processing] eda_advanced unavailable ({exc}); skipping advanced EDA.")
        return

    # The classical-cleaned column is what baselines actually see, so most
    # linguistic analyses use that. Raw column is used for quality checks.
    classical_col = "text_clean_classical" if "text_clean_classical" in df.columns else text_column
    neural_col = "text_clean_neural" if "text_clean_neural" in df.columns else text_column

    ea.duplicate_and_quality_report(df, text_column, label_column, output_dir)
    ea.length_outliers(df, text_column, label_column, output_dir)
    ea.ngram_frequency(df, classical_col, label_column, output_dir)
    ea.tfidf_top_terms_per_genre(df, classical_col, label_column, output_dir)
    ea.vocabulary_uniqueness(df, classical_col, label_column, output_dir)
    ea.class_overlap_text(df, classical_col, label_column, output_dir)
    ea.oov_curve(df, text_column, label_column, output_dir)
    if {classical_col, neural_col, text_column} <= set(df.columns):
        ea.cleaning_pipeline_impact(df, text_column, classical_col, neural_col, label_column, output_dir)
    # These two no-op gracefully if their optional deps are missing.
    ea.readability_metrics(df, text_column, label_column, output_dir)
    ea.pos_and_entity_distribution(df, text_column, label_column, output_dir)


def stratified_split(
    df: pd.DataFrame,
    label_column: str,
    split_config: SplitConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    split_config = split_config or SplitConfig()
    train_df, test_df = train_test_split(
        df,
        test_size=split_config.test_size,
        stratify=df[label_column],
        random_state=split_config.random_state,
    )
    val_fraction = split_config.validation_size / (1.0 - split_config.test_size)
    train_df, val_df = train_test_split(
        train_df,
        test_size=val_fraction,
        stratify=train_df[label_column],
        random_state=split_config.random_state,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def _write_json(payload: dict, path: Path) -> None:
    path.write_text(json.dumps(payload, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect and preprocess the movie dataset.")
    parser.add_argument("--input", required=True, help="Path to CSV or Parquet dataset.")
    parser.add_argument("--text-column", required=True, help="Name of the summary text column.")
    parser.add_argument("--label-column", required=True, help="Name of the genre label column.")
    parser.add_argument("--output-dir", default="outputs", help="Directory for figures and tables.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)

    df = load_dataset(args.input)
    required = {args.text_column, args.label_column}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    processed = add_clean_columns(df, args.text_column)
    profile = basic_profile(processed, args.text_column, args.label_column)
    _write_json(profile, output_dir / "dataset_profile.json")
    processed.to_csv(output_dir / "cleaned_dataset_preview.csv", index=False)
    save_eda_artifacts(processed, args.text_column, args.label_column, output_dir)

    train_df, val_df, test_df = stratified_split(processed, args.label_column)
    train_df.to_csv(output_dir / "train_split.csv", index=False)
    val_df.to_csv(output_dir / "validation_split.csv", index=False)
    test_df.to_csv(output_dir / "test_split.csv", index=False)

    print("Saved preprocessing artifacts to", output_dir)


if __name__ == "__main__":
    main()
