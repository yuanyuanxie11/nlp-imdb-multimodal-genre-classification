"""Advanced, reusable EDA functions.

Every public function follows the same shape::

    def some_eda(df, ..., output_dir) -> pd.DataFrame:
        # compute
        df_out.to_csv(tables_dir / "...csv", index=False)
        # plot
        plt.savefig(figures_dir / "...png", dpi=200); plt.close()
        return df_out

so they can be called from both the CLI (``data_processing.main``) and the
notebook with the same expectations: writes a CSV + a PNG, returns the
table for further use.

Optional dependencies (``textstat``, ``spacy``, ``datasketch``) are
imported lazily inside the function that needs them — missing deps log a
warning and the function returns an empty DataFrame instead of crashing.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .config import ensure_dir


# ---------------------------------------------------------------------------
# Small shared utilities
# ---------------------------------------------------------------------------

def _figs_tables(output_dir) -> tuple[Path, Path]:
    out = Path(output_dir)
    return ensure_dir(out / "figures"), ensure_dir(out / "tables")


def _safe_slug(value: str) -> str:
    """File-system-safe slug for genre names with spaces / slashes / etc."""
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(value)).strip("_") or "unknown"


def _genres(df: pd.DataFrame, label_column: str) -> list:
    return sorted(df[label_column].dropna().unique().tolist())


# ---------------------------------------------------------------------------
# 1. N-gram frequency
# ---------------------------------------------------------------------------

def ngram_frequency(
    df: pd.DataFrame,
    text_column: str,
    label_column: str,
    output_dir,
    *,
    n_values: Sequence[int] = (2, 3),
    top_k: int = 20,
    min_df: int = 2,
) -> pd.DataFrame:
    """Top-K bigrams / trigrams overall and per genre.

    Why n-grams matter
    ------------------
    Unigrams collapse "no good" / "very good" into the same bag-of-words —
    bigrams preserve sentiment-shifting modifiers and short phrases.
    Trigrams catch genre-defining patterns ("based on true", "fall in
    love", "set out to"). Looking at *per-genre* n-grams surfaces lexical
    signatures the classifier will rely on.
    """
    figs, tables = _figs_tables(output_dir)
    all_rows = []

    for n in n_values:
        # 💡 CountVectorizer(ngram_range=(n,n)) constrains to *only* n-grams
        # of length n, not all 1..n. Cleaner per-n analysis.
        for scope_label, sub_df in [("overall", df)] + [
            (str(g), df[df[label_column] == g]) for g in _genres(df, label_column)
        ]:
            if sub_df.empty:
                continue
            cv = CountVectorizer(ngram_range=(n, n), min_df=min_df)
            try:
                matrix = cv.fit_transform(sub_df[text_column].fillna("").astype(str))
            except ValueError:
                # Happens when min_df > documents — degrade to min_df=1.
                cv = CountVectorizer(ngram_range=(n, n), min_df=1)
                matrix = cv.fit_transform(sub_df[text_column].fillna("").astype(str))
            counts = np.asarray(matrix.sum(axis=0)).ravel()
            vocab = cv.get_feature_names_out()
            order = counts.argsort()[::-1][:top_k]
            top = pd.DataFrame({
                "ngram_length": n,
                "scope": scope_label,
                "ngram": vocab[order],
                "count": counts[order],
            })
            all_rows.append(top)

            slug = _safe_slug(scope_label)
            top.to_csv(tables / f"top_{n}grams_{slug}.csv", index=False)

            # Per-scope horizontal bar chart.
            plt.figure(figsize=(7, 0.3 * len(top) + 1))
            sns.barplot(data=top, y="ngram", x="count", color="steelblue")
            plt.title(f"Top {top_k} {n}-grams — {scope_label}")
            plt.tight_layout()
            plt.savefig(figs / f"top_{n}grams_{slug}.png", dpi=200)
            plt.close()

    combined = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    combined.to_csv(tables / "ngram_frequency_combined.csv", index=False)
    return combined


# ---------------------------------------------------------------------------
# 2. Distinctive TF-IDF terms per genre
# ---------------------------------------------------------------------------

def tfidf_top_terms_per_genre(
    df: pd.DataFrame,
    text_column: str,
    label_column: str,
    output_dir,
    *,
    top_k: int = 15,
    max_features: int = 20000,
) -> pd.DataFrame:
    """Genre-distinctive terms via *mean TF-IDF deviation from the corpus*.

    Naively reporting raw TF-IDF per genre gives you mostly the same
    common terms across all genres ("the", "movie", "story"). What you
    want are terms whose mean TF-IDF in the genre is *much higher* than
    in the rest of the corpus — those carry discriminative signal.
    """
    figs, tables = _figs_tables(output_dir)
    texts = df[text_column].fillna("").astype(str)
    vec = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    X = vec.fit_transform(texts)
    terms = vec.get_feature_names_out()

    # 💡 X[mask].mean(axis=0) → mean TF-IDF *within* the genre; subtract the
    # corresponding mean over the rest of the corpus → "lift".
    rows = []
    for g in _genres(df, label_column):
        mask = (df[label_column] == g).values
        if mask.sum() == 0:
            continue
        in_mean = np.asarray(X[mask].mean(axis=0)).ravel()
        out_mean = np.asarray(X[~mask].mean(axis=0)).ravel() if (~mask).any() else np.zeros_like(in_mean)
        lift = in_mean - out_mean
        top_idx = lift.argsort()[::-1][:top_k]
        for rank, idx in enumerate(top_idx):
            rows.append({
                "genre": g, "rank": rank + 1, "term": terms[idx],
                "in_genre_tfidf": float(in_mean[idx]),
                "out_genre_tfidf": float(out_mean[idx]),
                "lift": float(lift[idx]),
            })

    out = pd.DataFrame(rows)
    out.to_csv(tables / "distinctive_terms_per_genre.csv", index=False)

    # Small-multiple bar plot.
    genres = out["genre"].unique()
    cols = min(2, max(1, len(genres)))
    rows_ = int(np.ceil(len(genres) / cols))
    fig, axes = plt.subplots(rows_, cols, figsize=(7 * cols, 0.35 * top_k * rows_ + 1))
    axes = np.atleast_1d(axes).flatten()
    for i, g in enumerate(genres):
        sub = out[out["genre"] == g].sort_values("lift")
        axes[i].barh(sub["term"], sub["lift"], color="steelblue")
        axes[i].set_title(f"{g}: most distinctive (TF-IDF lift)")
    for j in range(len(genres), len(axes)):
        axes[j].axis("off")
    plt.tight_layout()
    plt.savefig(figs / "distinctive_terms_per_genre.png", dpi=200)
    plt.close()
    return out


# ---------------------------------------------------------------------------
# 3. Vocabulary uniqueness + pairwise Jaccard
# ---------------------------------------------------------------------------

def vocabulary_uniqueness(
    df: pd.DataFrame,
    text_column: str,
    label_column: str,
    output_dir,
    *,
    min_freq: int = 2,
    top_exclusive: int = 20,
) -> pd.DataFrame:
    """Genre vocabulary overlap + genre-exclusive terms.

    For each genre we build the set of tokens that appear at least
    ``min_freq`` times in that genre, then compare to other genres.
    """
    figs, tables = _figs_tables(output_dir)
    vocab_sets: dict = {}
    exclusive_terms: dict = {}

    for g in _genres(df, label_column):
        sub = df[df[label_column] == g][text_column].fillna("").astype(str)
        c = Counter()
        for s in sub:
            c.update(s.lower().split())
        vocab_sets[g] = {w for w, n in c.items() if n >= min_freq}

    # Genre-exclusive = present in this genre's vocab but in no other genre's vocab.
    others_union = {g: set().union(*[v for k, v in vocab_sets.items() if k != g]) for g in vocab_sets}
    exclusive_terms = {g: vocab_sets[g] - others_union[g] for g in vocab_sets}

    rows = []
    for g in vocab_sets:
        rows.append({
            "genre": g,
            "vocab_size": len(vocab_sets[g]),
            "exclusive_count": len(exclusive_terms[g]),
            "exclusive_pct": (len(exclusive_terms[g]) / max(1, len(vocab_sets[g]))) * 100,
            "top_exclusive_terms": ", ".join(list(exclusive_terms[g])[:top_exclusive]),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(tables / "vocabulary_uniqueness.csv", index=False)

    # Pairwise Jaccard heatmap.
    genres = list(vocab_sets.keys())
    jacc = np.zeros((len(genres), len(genres)))
    for i, a in enumerate(genres):
        for j, b in enumerate(genres):
            inter = len(vocab_sets[a] & vocab_sets[b])
            union = len(vocab_sets[a] | vocab_sets[b]) or 1
            jacc[i, j] = inter / union
    plt.figure(figsize=(6, 5))
    sns.heatmap(jacc, annot=True, fmt=".2f", xticklabels=genres, yticklabels=genres, cmap="Blues")
    plt.title("Genre vocabulary Jaccard similarity")
    plt.tight_layout()
    plt.savefig(figs / "genre_vocab_jaccard.png", dpi=200)
    plt.close()
    pd.DataFrame(jacc, index=genres, columns=genres).to_csv(tables / "genre_vocab_jaccard.csv")
    return summary


# ---------------------------------------------------------------------------
# 4. Length outliers
# ---------------------------------------------------------------------------

def length_outliers(
    df: pd.DataFrame,
    text_column: str,
    label_column: str,
    output_dir,
    *,
    z_thresh: float = 3.0,
) -> pd.DataFrame:
    """Per-genre z-score outliers on summary word-count.

    Outliers in either tail can both be informative:
    * Very short → likely truncated / scraping error / placeholder.
    * Very long → may be spoiler-heavy "synopsis" rows that aren't really
      summaries; could be confusing the model.
    """
    figs, tables = _figs_tables(output_dir)
    work = df.copy()
    work["__wc"] = work[text_column].fillna("").astype(str).str.split().map(len)
    out_rows = []
    for g, sub in work.groupby(label_column):
        mu, sd = sub["__wc"].mean(), sub["__wc"].std() or 1.0
        sub = sub.assign(z=(sub["__wc"] - mu) / sd)
        outliers = sub[sub["z"].abs() >= z_thresh]
        out_rows.append(outliers.assign(genre_mean=mu, genre_std=sd))
    out = pd.concat(out_rows, ignore_index=True) if out_rows else pd.DataFrame()
    keep_cols = [c for c in (label_column, text_column, "__wc", "z", "genre_mean", "genre_std") if c in out.columns]
    if not out.empty:
        out = out[keep_cols].rename(columns={"__wc": "word_count"})
    out.to_csv(tables / "length_outliers.csv", index=False)
    return out


# ---------------------------------------------------------------------------
# 5. Data-quality report
# ---------------------------------------------------------------------------

def duplicate_and_quality_report(
    df: pd.DataFrame,
    text_column: str,
    label_column: str,
    output_dir,
) -> pd.DataFrame:
    """Single tidy table of dataset health checks.

    Categories:
        * exact_duplicates — full-row duplicates.
        * normalised_text_duplicates — same text after .lower() + whitespace collapse.
        * html_residue — text still contains <tag> patterns.
        * non_ascii_heavy — text with >20% non-ASCII chars.
        * very_short — < 5 tokens (likely truncated).
        * missing_text — null or empty.
        * missing_label — null label.
    """
    _, tables = _figs_tables(output_dir)
    text = df[text_column].fillna("").astype(str)
    norm = text.str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    has_html = text.str.contains(r"<[^>]+>", regex=True, na=False)
    non_ascii_ratio = text.map(
        lambda s: (sum(1 for c in s if ord(c) > 127) / max(1, len(s)))
    )
    word_count = text.str.split().map(len)

    report = [
        {"category": "rows_total", "count": int(len(df))},
        {"category": "exact_duplicates", "count": int(df.duplicated().sum())},
        {"category": "normalised_text_duplicates", "count": int(norm.duplicated().sum())},
        {"category": "html_residue", "count": int(has_html.sum())},
        {"category": "non_ascii_heavy", "count": int((non_ascii_ratio > 0.2).sum())},
        {"category": "very_short_lt5_tokens", "count": int((word_count < 5).sum())},
        {"category": "missing_text", "count": int(text.eq("").sum())},
        {"category": "missing_label", "count": int(df[label_column].isna().sum())},
    ]
    out = pd.DataFrame(report)
    out.to_csv(tables / "data_quality_report.csv", index=False)
    return out


# ---------------------------------------------------------------------------
# 6. Readability metrics (uses textstat if available)
# ---------------------------------------------------------------------------

def readability_metrics(
    df: pd.DataFrame,
    text_column: str,
    label_column: str,
    output_dir,
) -> pd.DataFrame:
    """Per-genre readability stats. Gracefully degrades if ``textstat`` is missing."""
    figs, tables = _figs_tables(output_dir)
    try:
        import textstat  # type: ignore
    except ImportError:
        print("[eda_advanced.readability_metrics] textstat not installed — skipping.")
        return pd.DataFrame()

    text = df[text_column].fillna("").astype(str)
    rows = []
    for i, s in enumerate(text):
        if not s.strip():
            continue
        rows.append({
            "idx": i,
            "genre": df.iloc[i][label_column],
            "flesch_reading_ease": textstat.flesch_reading_ease(s),
            "flesch_kincaid_grade": textstat.flesch_kincaid_grade(s),
            "sentence_count": textstat.sentence_count(s),
            "avg_sentence_length": textstat.avg_sentence_length(s),
            "ttr": (len(set(s.lower().split())) / max(1, len(s.split()))),
        })
    full = pd.DataFrame(rows)
    if full.empty:
        return full
    summary = full.groupby("genre").agg(["mean", "median", "std"]).reset_index()
    summary.columns = ["_".join(c).rstrip("_") for c in summary.columns]
    summary.to_csv(tables / "readability_by_genre.csv", index=False)

    long = full.melt(id_vars=["idx", "genre"], var_name="metric", value_name="value")
    plt.figure(figsize=(11, 6))
    sns.violinplot(data=long, x="metric", y="value", hue="genre", inner="quartile")
    plt.xticks(rotation=20)
    plt.title("Readability metrics per genre")
    plt.tight_layout()
    plt.savefig(figs / "readability_violins.png", dpi=200)
    plt.close()
    return summary


# ---------------------------------------------------------------------------
# 7. Cleaning-pipeline impact
# ---------------------------------------------------------------------------

def cleaning_pipeline_impact(
    df: pd.DataFrame,
    raw_column: str,
    classical_column: str,
    neural_column: str,
    label_column: str,
    output_dir,
) -> pd.DataFrame:
    """Quantify how much each cleaning pipeline changes the data."""
    figs, tables = _figs_tables(output_dir)

    def vocab(series: pd.Series) -> set:
        c: Counter = Counter()
        for s in series.fillna("").astype(str):
            c.update(s.lower().split())
        return set(c.keys())

    raw_v, cls_v, neu_v = vocab(df[raw_column]), vocab(df[classical_column]), vocab(df[neural_column])
    wc = lambda s: s.fillna("").astype(str).str.split().map(len)
    raw_wc, cls_wc, neu_wc = wc(df[raw_column]).sum(), wc(df[classical_column]).sum(), wc(df[neural_column]).sum()

    impact = pd.DataFrame([
        {"pipeline": "raw",       "vocab_size": len(raw_v), "total_tokens": int(raw_wc), "pct_tokens_dropped_vs_raw": 0.0},
        {"pipeline": "classical", "vocab_size": len(cls_v), "total_tokens": int(cls_wc), "pct_tokens_dropped_vs_raw": (1 - cls_wc / max(1, raw_wc)) * 100},
        {"pipeline": "neural",    "vocab_size": len(neu_v), "total_tokens": int(neu_wc), "pct_tokens_dropped_vs_raw": (1 - neu_wc / max(1, raw_wc)) * 100},
    ])
    impact.to_csv(tables / "cleaning_pipeline_impact.csv", index=False)

    plt.figure(figsize=(8, 4))
    x = np.arange(len(impact)); w = 0.35
    plt.bar(x - w / 2, impact["vocab_size"], w, label="vocab size")
    plt.bar(x + w / 2, impact["total_tokens"], w, label="total tokens")
    plt.xticks(x, impact["pipeline"])
    plt.yscale("log")
    plt.title("Cleaning pipeline impact (log scale)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figs / "cleaning_impact.png", dpi=200)
    plt.close()
    return impact


# ---------------------------------------------------------------------------
# 8. Class overlap in text space
# ---------------------------------------------------------------------------

def class_overlap_text(
    df: pd.DataFrame,
    text_column: str,
    label_column: str,
    output_dir,
    *,
    max_features: int = 20000,
) -> pd.DataFrame:
    """Pre-training prediction: which genres will the model confuse?

    Compute one TF-IDF centroid per genre, then the cosine similarity
    matrix between centroids. Pairs with high similarity tend to dominate
    confusion-matrix off-diagonals.
    """
    figs, tables = _figs_tables(output_dir)
    vec = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    X = vec.fit_transform(df[text_column].fillna("").astype(str))
    genres = _genres(df, label_column)
    centroids = []
    for g in genres:
        mask = (df[label_column] == g).values
        centroids.append(np.asarray(X[mask].mean(axis=0)).ravel())
    centroids = np.vstack(centroids)
    sim = cosine_similarity(centroids)

    plt.figure(figsize=(6, 5))
    sns.heatmap(sim, annot=True, fmt=".2f", xticklabels=genres, yticklabels=genres, cmap="Reds")
    plt.title("Genre TF-IDF centroid cosine similarity (higher → harder to separate)")
    plt.tight_layout()
    plt.savefig(figs / "class_overlap_heatmap.png", dpi=200)
    plt.close()

    out = pd.DataFrame(sim, index=genres, columns=genres)
    out.to_csv(tables / "class_overlap_text.csv")
    return out


# ---------------------------------------------------------------------------
# 9. OOV curve
# ---------------------------------------------------------------------------

def oov_curve(
    df: pd.DataFrame,
    text_column: str,
    label_column: str,
    output_dir,
    *,
    vocab_sizes: Iterable[int] = (1000, 5000, 10000, 15000, 30000),
) -> pd.DataFrame:
    """OOV rate vs. tokenizer vocab cap.

    Informs the LSTM ``max_words`` choice. If the curve has clearly
    flattened before 15000 (the current default), 15000 is fine; if it
    is still descending steeply, increasing ``max_words`` will probably
    help the LSTM.
    """
    figs, tables = _figs_tables(output_dir)
    text = df[text_column].fillna("").astype(str)
    # Token-level frequency across the corpus.
    counter: Counter = Counter()
    docs = [s.lower().split() for s in text]
    for tokens in docs:
        counter.update(tokens)
    sorted_terms = [w for w, _ in counter.most_common()]
    total_tokens = sum(counter.values())

    rows = []
    for k in vocab_sizes:
        kept = set(sorted_terms[:k])
        # Sum frequencies of terms NOT in kept → OOV-by-token rate.
        oov_tokens = sum(n for w, n in counter.items() if w not in kept)
        rows.append({"vocab_size": k, "oov_token_rate": oov_tokens / max(1, total_tokens)})
    out = pd.DataFrame(rows)
    out.to_csv(tables / "oov_curve.csv", index=False)

    plt.figure(figsize=(7, 4))
    plt.plot(out["vocab_size"], out["oov_token_rate"], marker="o")
    plt.xlabel("Tokenizer vocab cap")
    plt.ylabel("OOV token rate")
    plt.title("OOV rate vs. tokenizer vocabulary size")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figs / "oov_curve.png", dpi=200)
    plt.close()
    return out


# ---------------------------------------------------------------------------
# 10. POS + entity distribution (spacy, optional)
# ---------------------------------------------------------------------------

def pos_and_entity_distribution(
    df: pd.DataFrame,
    text_column: str,
    label_column: str,
    output_dir,
    *,
    sample_size: int = 2000,
    spacy_model: str = "en_core_web_sm",
) -> pd.DataFrame:
    """Per-genre POS-tag distribution and top entity types.

    Optional; skipped with a warning if ``spacy`` or the model is missing.
    """
    figs, tables = _figs_tables(output_dir)
    try:
        import spacy  # type: ignore
        try:
            nlp = spacy.load(spacy_model, disable=["parser"])
        except Exception:
            print(f"[eda_advanced.pos_and_entity_distribution] spacy model '{spacy_model}' not installed — skipping.")
            return pd.DataFrame()
    except ImportError:
        print("[eda_advanced.pos_and_entity_distribution] spacy not installed — skipping.")
        return pd.DataFrame()

    work = df.sample(min(sample_size, len(df)), random_state=42)
    rows = []
    for g, sub in work.groupby(label_column):
        pos_counter: Counter = Counter()
        ent_counter: Counter = Counter()
        for doc in nlp.pipe(sub[text_column].fillna("").astype(str), batch_size=64):
            pos_counter.update(t.pos_ for t in doc if not t.is_space)
            ent_counter.update(e.label_ for e in doc.ents)
        total_pos = sum(pos_counter.values()) or 1
        for pos, n in pos_counter.most_common(12):
            rows.append({"genre": g, "kind": "POS", "tag": pos, "count": n, "pct": n / total_pos * 100})
        for ent, n in ent_counter.most_common(10):
            rows.append({"genre": g, "kind": "ENT", "tag": ent, "count": n})

    out = pd.DataFrame(rows)
    out.to_csv(tables / "pos_entity_distribution.csv", index=False)
    if not out.empty and "pct" in out.columns:
        pos_only = out[out["kind"] == "POS"]
        plt.figure(figsize=(10, 5))
        sns.barplot(data=pos_only, x="tag", y="pct", hue="genre")
        plt.title("POS distribution by genre (sample)")
        plt.tight_layout()
        plt.savefig(figs / "pos_distribution_by_genre.png", dpi=200)
        plt.close()
    return out
