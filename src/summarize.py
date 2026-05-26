from __future__ import annotations

import argparse
import math
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from .config import ensure_dir
from .data_processing import clean_text, load_dataset


def split_sentences(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def extractive_summary(text: str, max_sentences: int = 3) -> str:
    sentences = split_sentences(text)
    if len(sentences) <= max_sentences:
        return text.strip()

    tokens = clean_text(text).split()
    freq = Counter(tokens)
    if not freq:
        return " ".join(sentences[:max_sentences])

    sentence_scores = []
    for sentence in sentences:
        sentence_tokens = clean_text(sentence).split()
        if not sentence_tokens:
            continue
        score = sum(freq[token] for token in sentence_tokens) / math.sqrt(len(sentence_tokens))
        sentence_scores.append((score, sentence))

    top_sentences = sorted(sentence_scores, key=lambda x: x[0], reverse=True)[:max_sentences]
    chosen = {sentence for _, sentence in top_sentences}
    ordered = [sentence for sentence in sentences if sentence in chosen]
    return " ".join(ordered)


def transformer_summary(text: str, max_length: int = 90, min_length: int = 30) -> str:
    try:
        from transformers import pipeline
    except ImportError as exc:
        raise RuntimeError("transformers is not installed.") from exc

    summarizer = pipeline("summarization")
    result = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
    return result[0]["summary_text"]


def summarize_examples(
    input_path: str | Path,
    text_column: str,
    output_dir: str | Path,
    sample_size: int = 5,
) -> pd.DataFrame:
    df = load_dataset(input_path)
    samples = df[[text_column]].dropna().head(sample_size).copy()
    samples["extractive_summary"] = samples[text_column].map(extractive_summary)

    try:
        samples["transformer_summary"] = samples[text_column].map(transformer_summary)
    except Exception:
        samples["transformer_summary"] = "Transformer summarizer unavailable"

    output_dir = ensure_dir(output_dir)
    samples.to_csv(output_dir / "tables" / "sample_summaries.csv", index=False)
    return samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build summary examples from movie plots.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--text-column", required=True)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--sample-size", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(Path(args.output_dir) / "tables")
    samples = summarize_examples(args.input, args.text_column, args.output_dir, args.sample_size)
    print(samples.head(args.sample_size).to_string(index=False))


if __name__ == "__main__":
    main()
