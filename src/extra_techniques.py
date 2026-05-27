"""
extra_techniques.py
────────────────────
Additional NLP techniques applied to the IMDB genre dataset:

  1. t-SNE 2D visualization of TF-IDF document embeddings
  2. LDA topic modeling per genre (top topic words per genre)
  3. Sentiment analysis distribution across genres

Run as a script:
    python -m src.extra_techniques \
        --input data/movies.csv \
        --text-column Plot \
        --label-column Genre \
        --output-dir outputs
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .runtime import prepare_runtime

prepare_runtime()

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.manifold import TSNE

from .config import ensure_dir
from .data_processing import add_clean_columns, load_dataset, stratified_split


# ── 1. t-SNE embedding visualization ──────────────────────────────────────────

def tsne_embedding_plot(
    df: pd.DataFrame,
    text_column: str,
    label_column: str,
    output_dir: Path,
    n_components_svd: int = 50,
    perplexity: float = 30.0,
    random_state: int = 42,
) -> None:
    """
    Reduce TF-IDF vectors to 2D with SVD → t-SNE and scatter-plot by genre.

    We first apply Truncated SVD (LSA) to reduce dimensionality from the
    full TF-IDF space to 50 dimensions, then t-SNE to 2D. This is a common
    and fast approach for visualizing high-dimensional text data.
    """
    texts = df[text_column].fillna("").astype(str).tolist()
    labels = df[label_column].fillna("unknown").astype(str).tolist()
    unique_labels = sorted(set(labels))

    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(max_features=5000, sublinear_tf=True, ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(texts)

    # Dimensionality reduction: LSA → t-SNE
    n_components_svd = min(n_components_svd, tfidf_matrix.shape[1] - 1, tfidf_matrix.shape[0] - 1)
    svd = TruncatedSVD(n_components=n_components_svd, random_state=random_state)
    reduced = svd.fit_transform(tfidf_matrix)

    tsne = TSNE(
        n_components=2,
        perplexity=min(perplexity, len(texts) - 1),
        random_state=random_state,
        max_iter=1000,
        init="pca",
        learning_rate="auto",
    )
    coords = tsne.fit_transform(reduced)

    # Plot
    palette = cm.get_cmap("tab10", len(unique_labels))
    color_map = {label: palette(i) for i, label in enumerate(unique_labels)}

    fig, ax = plt.subplots(figsize=(10, 8))
    for label in unique_labels:
        mask = [l == label for l in labels]
        xs = coords[mask, 0]
        ys = coords[mask, 1]
        ax.scatter(xs, ys, label=label, alpha=0.6, s=20, color=color_map[label])

    ax.set_title("t-SNE Visualization of Movie Plot Summaries (TF-IDF → LSA → t-SNE)", fontsize=14)
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    ax.legend(title="Genre", markerscale=2)
    plt.tight_layout()

    out_path = output_dir / "figures" / "tsne_embeddings.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved t-SNE plot → {out_path}")


# ── 2. LDA topic modeling per genre ───────────────────────────────────────────

def lda_topic_modeling(
    df: pd.DataFrame,
    text_column: str,
    label_column: str,
    output_dir: Path,
    n_topics: int = 3,
    top_words: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Fit one LDA model per genre on the cleaned text.

    For each genre we show `n_topics` latent topics, each described by its
    `top_words` highest-probability words. This reveals genre-specific
    thematic clusters beyond simple word frequency.

    Returns a DataFrame with columns [genre, topic, words].
    """
    rows = []
    for genre, group in df.groupby(label_column):
        texts = group[text_column].fillna("").astype(str).tolist()
        if len(texts) < n_topics:
            continue

        # CountVectorizer for LDA (LDA works on raw counts, not TF-IDF)
        cv = CountVectorizer(
            max_features=2000,
            min_df=2,
            max_df=0.9,
            ngram_range=(1, 2),
        )
        try:
            doc_term = cv.fit_transform(texts)
        except ValueError:
            continue  # too few docs

        n_fit_topics = min(n_topics, doc_term.shape[0] - 1, doc_term.shape[1] - 1)
        if n_fit_topics < 1:
            continue

        lda = LatentDirichletAllocation(
            n_components=n_fit_topics,
            max_iter=20,
            random_state=random_state,
        )
        lda.fit(doc_term)

        feature_names = cv.get_feature_names_out()
        for topic_idx, topic in enumerate(lda.components_):
            top_word_ids = topic.argsort()[-top_words:][::-1]
            word_list = ", ".join(feature_names[i] for i in top_word_ids)
            rows.append({"genre": genre, "topic": topic_idx + 1, "top_words": word_list})

    result_df = pd.DataFrame(rows)

    # Save table
    tables_dir = ensure_dir(output_dir / "tables")
    result_df.to_csv(tables_dir / "lda_topics_by_genre.csv", index=False)

    # Plot: one heatmap-style bar chart per genre showing topic word clouds alternative
    _plot_lda_topics(result_df, output_dir)
    print(f"Saved LDA topic table → {tables_dir / 'lda_topics_by_genre.csv'}")
    return result_df


def _plot_lda_topics(topic_df: pd.DataFrame, output_dir: Path) -> None:
    """Save a text-table figure of LDA topics per genre for easy presentation."""
    genres = topic_df["genre"].unique()
    n_genres = len(genres)
    if n_genres == 0:
        return

    fig, axes = plt.subplots(1, n_genres, figsize=(6 * n_genres, 4), squeeze=False)
    for ax, genre in zip(axes[0], genres):
        genre_rows = topic_df[topic_df["genre"] == genre]
        text_lines = [f"Genre: {genre.upper()}", ""]
        for _, row in genre_rows.iterrows():
            text_lines.append(f"Topic {int(row['topic'])}:")
            # wrap words at ~30 chars per line
            words = row["top_words"].split(", ")
            line = ""
            for w in words:
                if len(line) + len(w) + 2 > 32:
                    text_lines.append(f"  {line.rstrip(', ')}")
                    line = w + ", "
                else:
                    line += w + ", "
            if line:
                text_lines.append(f"  {line.rstrip(', ')}")
            text_lines.append("")

        ax.axis("off")
        ax.text(
            0.05, 0.95, "\n".join(text_lines),
            transform=ax.transAxes,
            va="top", ha="left",
            fontsize=9,
            fontfamily="monospace",
        )
        ax.set_title(f"LDA Topics — {genre.title()}", fontsize=11, fontweight="bold")

    plt.suptitle("Latent Dirichlet Allocation Topics by Genre", fontsize=14, y=1.02)
    plt.tight_layout()
    out_path = output_dir / "figures" / "lda_topics.png"
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved LDA topic figure → {out_path}")


# ── 3. Sentiment analysis distribution by genre ───────────────────────────────

def sentiment_by_genre(
    df: pd.DataFrame,
    text_column: str,
    label_column: str,
    output_dir: Path,
) -> pd.DataFrame:
    """
    Compute a simple lexicon-based sentiment score for each summary
    and compare distributions across genres.

    Uses TextBlob if available; falls back to a simple positive/negative
    word count approach using the VADER-inspired word lists.
    """
    try:
        from textblob import TextBlob

        def polarity(text: str) -> float:
            return TextBlob(str(text)).sentiment.polarity

        score_label = "TextBlob Polarity"
    except ImportError:
        # Minimal fallback: difference of positive vs. negative word counts
        POSITIVE_WORDS = {
            "love", "happy", "wonderful", "great", "joy", "beautiful",
            "funny", "laugh", "smile", "romantic", "sweet", "kind", "warm",
        }
        NEGATIVE_WORDS = {
            "kill", "murder", "dead", "dark", "terror", "evil", "blood",
            "horror", "fear", "scream", "death", "monster", "violent",
        }

        def polarity(text: str) -> float:
            tokens = str(text).lower().split()
            pos = sum(1 for t in tokens if t in POSITIVE_WORDS)
            neg = sum(1 for t in tokens if t in NEGATIVE_WORDS)
            denom = len(tokens) or 1
            return (pos - neg) / denom

        score_label = "Sentiment Polarity (lexicon)"

    enriched = df[[text_column, label_column]].copy()
    enriched["sentiment"] = enriched[text_column].fillna("").map(polarity)

    # Plot violin + strip chart
    fig, ax = plt.subplots(figsize=(9, 6))
    order = sorted(enriched[label_column].dropna().unique())
    sns.violinplot(
        data=enriched,
        x=label_column,
        y="sentiment",
        order=order,
        palette="Set2",
        inner=None,
        alpha=0.6,
        ax=ax,
    )
    sns.stripplot(
        data=enriched,
        x=label_column,
        y="sentiment",
        order=order,
        color="k",
        alpha=0.25,
        size=2,
        jitter=True,
        ax=ax,
    )
    ax.axhline(0, color="red", linestyle="--", linewidth=0.8, label="Neutral")
    ax.set_title(f"Sentiment Distribution by Genre ({score_label})", fontsize=13)
    ax.set_xlabel("Genre")
    ax.set_ylabel(score_label)
    ax.legend()
    plt.tight_layout()

    out_path = output_dir / "figures" / "sentiment_by_genre.png"
    plt.savefig(out_path, dpi=200)
    plt.close()

    # Save summary stats
    stats = enriched.groupby(label_column)["sentiment"].describe().reset_index()
    stats.to_csv(output_dir / "tables" / "sentiment_by_genre.csv", index=False)
    print(f"Saved sentiment plot → {out_path}")
    return enriched


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extra NLP techniques: t-SNE, LDA, sentiment.")
    parser.add_argument("--input", required=True, help="Path to CSV dataset.")
    parser.add_argument("--text-column", required=True)
    parser.add_argument("--label-column", required=True)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--no-tsne", action="store_true", help="Skip t-SNE (slow on large datasets).")
    parser.add_argument("--n-topics", type=int, default=3, help="LDA topics per genre.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    ensure_dir(output_dir / "figures")
    ensure_dir(output_dir / "tables")

    df = add_clean_columns(load_dataset(args.input), args.text_column)

    if not args.no_tsne:
        print("Running t-SNE embedding visualization …")
        tsne_embedding_plot(df, "text_clean_classical", args.label_column, output_dir)

    print("Running LDA topic modeling …")
    topic_df = lda_topic_modeling(
        df, "text_clean_classical", args.label_column, output_dir, n_topics=args.n_topics
    )
    print(topic_df.to_string(index=False))

    print("Running sentiment analysis …")
    sentiment_by_genre(df, args.text_column, args.label_column, output_dir)

    print("Done. Outputs saved to", output_dir)


if __name__ == "__main__":
    main()
