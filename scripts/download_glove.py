"""
scripts/download_glove.py
─────────────────────────
Downloads GloVe 6B 100-dimensional word vectors from Stanford and extracts
them into the data/ directory.

Usage:
    python -m scripts.download_glove          # from project root
    python scripts/download_glove.py          # direct

File produced: data/glove.6B.100d.txt  (~347 MB unzipped)
Download size: ~822 MB zip

Reference: https://nlp.stanford.edu/projects/glove/
"""
from __future__ import annotations

import sys
import urllib.request
import zipfile
from pathlib import Path

GLOVE_URL  = "https://nlp.stanford.edu/data/glove.6B.zip"
TARGET_FILE = "glove.6B.100d.txt"


def _progress(block_num: int, block_size: int, total_size: int) -> None:
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(downloaded / total_size * 100, 100)
        mb  = downloaded / 1_048_576
        bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
        print(f"\r  [{bar}] {pct:5.1f}%  {mb:6.1f} MB", end="", flush=True)


def download_glove(data_dir: Path = Path("data")) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    target = data_dir / TARGET_FILE

    if target.exists():
        print(f"✓ GloVe already downloaded: {target}")
        return target

    zip_path = data_dir / "glove.6B.zip"

    if not zip_path.exists():
        print(f"Downloading GloVe 6B from Stanford (~822 MB)…")
        print(f"  URL : {GLOVE_URL}")
        print(f"  Dest: {zip_path}\n")
        try:
            urllib.request.urlretrieve(GLOVE_URL, zip_path, reporthook=_progress)
            print()  # newline after progress bar
        except Exception as exc:
            print(f"\n✗ Download failed: {exc}")
            print("\nManual download:")
            print(f"  1. Go to {GLOVE_URL}")
            print(f"  2. Save the zip to {zip_path}")
            print(f"  3. Re-run this script to extract.")
            sys.exit(1)
    else:
        print(f"✓ Zip already present: {zip_path}. Extracting…")

    print(f"Extracting {TARGET_FILE} …", end=" ", flush=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        # Only extract the 100d file to save disk space
        zf.extract(TARGET_FILE, data_dir)
    print("done.")

    # Clean up zip to save disk space (optional)
    try:
        zip_path.unlink()
        print(f"Removed zip file ({zip_path}).")
    except OSError:
        pass

    print(f"\n✓ GloVe ready at: {target}")
    return target


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download GloVe 6B 100d embeddings.")
    parser.add_argument("--data-dir", default="data", help="Directory to save GloVe file.")
    args = parser.parse_args()
    download_glove(Path(args.data_dir))
