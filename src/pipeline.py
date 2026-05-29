from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

from .config import ensure_dir
from .runtime import prepare_runtime

prepare_runtime()

DEFAULT_DATASET = "zulkarnainsaurav/imdb-multimodal-vision-and-nlp-genre-classification"
TEXT_COLUMN_CANDIDATES = (
    "summary",
    "plot",
    "plot summary",
    "plot_summary",
    "description",
    "overview",
    "synopsis",
)
LABEL_COLUMN_CANDIDATES = ("genre", "genres", "label", "class", "category")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def _normalized(value: str) -> str:
    return value.strip().lower().replace("-", " ").replace("_", " ")


def _read_columns(path: Path) -> list[str]:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, nrows=0).columns.tolist()
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path).columns.tolist()
    return []


def infer_columns(path: str | Path) -> tuple[str, str]:
    columns = _read_columns(Path(path))
    normalized = {_normalized(column): column for column in columns}

    text_column = next(
        (normalized[name] for name in TEXT_COLUMN_CANDIDATES if name in normalized),
        None,
    )
    label_column = next(
        (normalized[name] for name in LABEL_COLUMN_CANDIDATES if name in normalized),
        None,
    )

    if not text_column or not label_column:
        raise ValueError(
            "Could not infer text and genre columns. "
            f"Available columns: {columns}. "
            "Pass --text-column and --label-column explicitly."
        )
    return text_column, label_column


def select_dataset_file(data_dir: str | Path) -> Path:
    data_dir = Path(data_dir)
    candidates = sorted(
        [
            path
            for suffix in ("*.csv", "*.parquet", "*.pq")
            for path in data_dir.rglob(suffix)
            if path.is_file()
        ],
        key=lambda path: path.stat().st_size,
        reverse=True,
    )
    for path in candidates:
        try:
            infer_columns(path)
        except Exception:
            continue
        return path
    raise FileNotFoundError(f"No supported dataset file with text and genre columns found in {data_dir}.")


def select_poster_dir(data_dir: str | Path) -> Path | None:
    data_dir = Path(data_dir)
    directories = sorted(
        [path for path in data_dir.rglob("*") if path.is_dir()],
        key=lambda path: len(list(path.rglob("*"))),
        reverse=True,
    )
    for directory in directories:
        genre_dirs = [path for path in directory.iterdir() if path.is_dir()]
        image_count = sum(
            1
            for genre_dir in genre_dirs
            for image_path in genre_dir.rglob("*")
            if image_path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if len(genre_dirs) >= 2 and image_count > 0:
            return directory
    return None


def build_image_manifest(poster_dir: str | Path, output_path: str | Path) -> Path:
    poster_dir = Path(poster_dir)
    rows = []
    for genre_dir in sorted(path for path in poster_dir.iterdir() if path.is_dir()):
        for image_path in sorted(genre_dir.rglob("*")):
            if image_path.suffix.lower() in IMAGE_EXTENSIONS:
                rows.append({"image_path": str(image_path), "genre": genre_dir.name})
    if not rows:
        raise FileNotFoundError(f"No poster images found in {poster_dir}.")
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return output_path


def download_dataset(dataset: str, data_dir: str | Path) -> Path:
    import kagglehub

    data_dir = ensure_dir(data_dir)
    downloaded_path = Path(kagglehub.dataset_download(dataset))
    target_dir = data_dir / downloaded_path.name
    if downloaded_path.resolve() != target_dir.resolve():
        shutil.copytree(downloaded_path, target_dir, dirs_exist_ok=True)
    return target_dir


def run_step(args: list[str]) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run(args, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Kaggle data and train project models.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="KaggleHub dataset slug.")
    parser.add_argument("--data-dir", default="data", help="Directory where dataset files are stored.")
    parser.add_argument("--input", help="Dataset CSV/Parquet path. If omitted, the pipeline downloads and infers it.")
    parser.add_argument("--text-column", help="Summary/plot text column. Inferred when omitted.")
    parser.add_argument("--label-column", help="Genre label column. Inferred when omitted.")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-lstm", action="store_true")
    parser.add_argument("--skip-ensemble", action="store_true")
    parser.add_argument("--skip-image", action="store_true")
    parser.add_argument("--lstm-epochs", type=int, default=10)
    parser.add_argument("--image-epochs", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.data_dir)
    ensure_dir(args.output_dir)
    ensure_dir(args.model_dir)

    if not args.skip_download and not args.input:
        dataset_dir = download_dataset(args.dataset, args.data_dir)
        print(f"Dataset files copied to: {dataset_dir}", flush=True)

    input_path = Path(args.input) if args.input else select_dataset_file(args.data_dir)
    text_column, label_column = (
        (args.text_column, args.label_column)
        if args.text_column and args.label_column
        else infer_columns(input_path)
    )

    print(f"Using dataset: {input_path}", flush=True)
    print(f"Using text column: {text_column}", flush=True)
    print(f"Using label column: {label_column}", flush=True)

    common = [
        "--input",
        str(input_path),
        "--text-column",
        text_column,
        "--label-column",
        label_column,
        "--output-dir",
        args.output_dir,
    ]
    run_step([sys.executable, "-m", "src.data_processing", *common])
    run_step([sys.executable, "-m", "src.train_baselines", *common, "--model-dir", args.model_dir])
    if not args.skip_lstm:
        run_step(
            [
                sys.executable,
                "-m",
                "src.train_lstm",
                *common,
                "--model-dir",
                args.model_dir,
                "--epochs",
                str(args.lstm_epochs),
            ]
        )
    if not args.skip_ensemble and not args.skip_lstm:
        run_step(
            [
                sys.executable,
                "-m",
                "src.train_ensemble",
                *common,
                "--model-dir",
                args.model_dir,
            ]
        )
    if not args.skip_image:
        poster_dir = select_poster_dir(args.data_dir)
        if poster_dir is None:
            print("[warn] No poster image directory found; skipping image classifier.", flush=True)
        else:
            print(f"Using poster directory: {poster_dir}", flush=True)
            run_step(
                [
                    sys.executable,
                    "-m",
                    "src.bonus_image_classifier",
                    "--input",
                    str(input_path),
                    "--poster-dir",
                    str(poster_dir),
                    "--output-dir",
                    args.output_dir,
                    "--model-dir",
                    args.model_dir,
                    "--epochs",
                    str(args.image_epochs),
                ]
            )
    run_step(
        [
            sys.executable,
            "-m",
            "src.explain",
            *common,
            "--model-dir",
            args.model_dir,
        ]
    )


if __name__ == "__main__":
    main()
