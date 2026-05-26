"""Poster image EDA for the multimodal bonus.

Source-detection chain (first hit wins):
    1. Folder-per-class layout: ``data/posters/<genre>/*.{jpg,png,jpeg}``.
    2. A DataFrame / CSV with an image-path column.
    3. Network fetch (TMDB API or Kaggle dataset) — only if explicitly requested.

If no posters are available and the CLI was not asked to fetch, the module
exits with a clear message rather than crashing. This makes it safe to wire
into ``save_eda_artifacts`` as a best-effort step.
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError

from .config import ensure_dir

VALID_EXT = {".jpg", ".jpeg", ".png", ".webp"}


# ---------------------------------------------------------------------------
# Source detection
# ---------------------------------------------------------------------------

def discover_local_posters(posters_dir: Path) -> pd.DataFrame:
    """Folder-per-class layout: returns a DataFrame with columns ``image_path``, ``genre``."""
    rows = []
    if not posters_dir.exists():
        return pd.DataFrame(columns=["image_path", "genre"])
    for genre_dir in sorted(posters_dir.iterdir()):
        if not genre_dir.is_dir():
            continue
        for img in genre_dir.iterdir():
            if img.suffix.lower() in VALID_EXT:
                rows.append({"image_path": str(img), "genre": genre_dir.name})
    return pd.DataFrame(rows)


def posters_from_csv(csv_path: Path, image_column: str, label_column: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if image_column not in df.columns or label_column not in df.columns:
        raise KeyError(f"CSV must contain '{image_column}' and '{label_column}'.")
    df = df[[image_column, label_column]].dropna().copy()
    df = df[df[image_column].map(lambda p: Path(p).exists())]
    return df.rename(columns={image_column: "image_path", label_column: "genre"})


# ---------------------------------------------------------------------------
# Optional fetch backends — both are best-effort and skip on any error.
# ---------------------------------------------------------------------------

def fetch_from_tmdb(
    genres: Iterable[str],
    target_dir: Path,
    *,
    per_genre: int = 20,
    api_key: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch a small poster sample from TMDB.

    Requires ``TMDB_API_KEY`` env var (v3 key). Uses ``discover/movie``
    with ``with_genres`` per supported genre. Skips with a printed
    message if the network call fails for any reason.
    """
    api_key = api_key or os.environ.get("TMDB_API_KEY")
    if not api_key:
        print("[eda_images.fetch_from_tmdb] TMDB_API_KEY not set — skipping fetch.")
        return pd.DataFrame()
    try:
        import requests  # type: ignore
    except ImportError:
        print("[eda_images.fetch_from_tmdb] requests not installed — skipping.")
        return pd.DataFrame()

    # TMDB genre IDs for the four classes this project uses. If the
    # caller passes a genre name we don't know, we just skip that genre.
    GENRE_IDS = {"Action": 28, "Comedy": 35, "Horror": 27, "Romance": 10749}
    rows = []
    for genre in genres:
        gid = GENRE_IDS.get(str(genre))
        if gid is None:
            print(f"[eda_images.fetch_from_tmdb] unknown genre '{genre}' — skipping.")
            continue
        out_dir = ensure_dir(target_dir / str(genre))
        try:
            r = requests.get(
                "https://api.themoviedb.org/3/discover/movie",
                params={"api_key": api_key, "with_genres": gid, "language": "en-US", "page": 1},
                timeout=20,
            )
            r.raise_for_status()
            results = r.json().get("results", [])[:per_genre]
        except Exception as exc:
            print(f"[eda_images.fetch_from_tmdb] {genre}: discover failed ({exc}) — skipping.")
            continue
        for item in results:
            poster_path = item.get("poster_path")
            if not poster_path:
                continue
            url = f"https://image.tmdb.org/t/p/w342{poster_path}"
            try:
                img_resp = requests.get(url, timeout=20)
                img_resp.raise_for_status()
                dest = out_dir / f"{item['id']}.jpg"
                dest.write_bytes(img_resp.content)
                rows.append({"image_path": str(dest), "genre": str(genre)})
            except Exception as exc:
                print(f"[eda_images.fetch_from_tmdb] failed to download {url}: {exc}")
                continue
    return pd.DataFrame(rows)


def fetch_from_kaggle(target_dir: Path) -> pd.DataFrame:
    """Stub for the Kaggle 'Movie Genre from its Poster' dataset.

    Implemented as a graceful no-op unless the ``kaggle`` CLI is
    available and authenticated; otherwise prints what the user would
    need to set up.
    """
    try:
        import subprocess
        cmd = ["kaggle", "datasets", "download", "-d", "neha1703/movie-genre-from-its-poster",
               "-p", str(target_dir), "--unzip"]
        subprocess.run(cmd, check=True, capture_output=True)
        # The Kaggle dataset has its own structure — surfacing it is left
        # to the user (intentional: we don't want to silently reshape it).
        print(f"[eda_images.fetch_from_kaggle] dataset extracted to {target_dir}; "
              "rearrange into folder-per-class before re-running EDA.")
    except Exception as exc:
        print(f"[eda_images.fetch_from_kaggle] kaggle fetch unavailable ({exc}).")
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Inventory + size distribution
# ---------------------------------------------------------------------------

def poster_inventory(catalog: pd.DataFrame, output_dir) -> pd.DataFrame:
    """Counts per genre + how many files are missing / unreadable."""
    out_dir = ensure_dir(output_dir)
    tables = ensure_dir(out_dir / "tables")
    rows = []
    for genre, sub in catalog.groupby("genre"):
        missing = sum(1 for p in sub["image_path"] if not Path(p).exists())
        broken = 0
        for p in sub["image_path"]:
            try:
                with Image.open(p) as im:
                    im.verify()
            except (FileNotFoundError, UnidentifiedImageError, OSError):
                broken += 1
        rows.append({"genre": genre, "count": len(sub), "missing": missing, "broken": broken})
    out = pd.DataFrame(rows)
    out.to_csv(tables / "poster_inventory.csv", index=False)
    return out


def poster_size_distribution(catalog: pd.DataFrame, output_dir) -> pd.DataFrame:
    out_dir = ensure_dir(output_dir)
    figs = ensure_dir(out_dir / "figures")
    tables = ensure_dir(out_dir / "tables")
    rows = []
    for _, r in catalog.iterrows():
        try:
            with Image.open(r["image_path"]) as im:
                w, h = im.size
                rows.append({"genre": r["genre"], "width": w, "height": h, "aspect": w / max(1, h)})
        except Exception:
            continue
    out = pd.DataFrame(rows)
    out.to_csv(tables / "poster_size_distribution.csv", index=False)
    if not out.empty:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for ax, col in zip(axes, ["width", "height", "aspect"]):
            for g in out["genre"].unique():
                vals = out[out["genre"] == g][col]
                ax.hist(vals, bins=15, alpha=0.5, label=str(g))
            ax.set_title(col)
            ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(figs / "poster_size_distribution.png", dpi=200)
        plt.close()
    return out


# ---------------------------------------------------------------------------
# Color signature
# ---------------------------------------------------------------------------

def poster_color_signature(
    catalog: pd.DataFrame,
    output_dir,
    *,
    n_colors: int = 5,
    pixels_per_image: int = 400,
    sample_size_per_genre: int = 30,
) -> pd.DataFrame:
    """Dominant-color palette per genre via KMeans over sampled pixels."""
    out_dir = ensure_dir(output_dir)
    figs = ensure_dir(out_dir / "figures")
    tables = ensure_dir(out_dir / "tables")
    try:
        from sklearn.cluster import KMeans
    except ImportError:
        print("[eda_images.poster_color_signature] sklearn unavailable — skipping.")
        return pd.DataFrame()

    palette_rows = []
    fig_palettes = {}
    rng = random.Random(42)
    for genre, sub in catalog.groupby("genre"):
        paths = sub["image_path"].tolist()
        rng.shuffle(paths)
        sampled = paths[:sample_size_per_genre]
        pixel_buffer = []
        for p in sampled:
            try:
                with Image.open(p) as im:
                    im = im.convert("RGB").resize((64, 96))  # downsample → speed
                    arr = np.asarray(im).reshape(-1, 3)
                    idx = np.random.default_rng(0).choice(len(arr), size=min(pixels_per_image, len(arr)), replace=False)
                    pixel_buffer.append(arr[idx])
            except Exception:
                continue
        if not pixel_buffer:
            continue
        pixels = np.vstack(pixel_buffer)
        km = KMeans(n_clusters=n_colors, n_init=4, random_state=42).fit(pixels)
        centers = km.cluster_centers_.astype(int)
        counts = np.bincount(km.labels_, minlength=n_colors)
        order = counts.argsort()[::-1]
        for rank, idx in enumerate(order):
            r, g, b = centers[idx]
            palette_rows.append({
                "genre": genre, "rank": rank + 1,
                "r": int(r), "g": int(g), "b": int(b),
                "hex": "#{:02x}{:02x}{:02x}".format(r, g, b),
                "share": float(counts[idx] / counts.sum()),
            })
        fig_palettes[genre] = centers[order]

    palette = pd.DataFrame(palette_rows)
    palette.to_csv(tables / "poster_color_palette.csv", index=False)

    if fig_palettes:
        fig, axes = plt.subplots(len(fig_palettes), 1, figsize=(6, 1.2 * len(fig_palettes)))
        axes = np.atleast_1d(axes)
        for ax, (genre, swatch) in zip(axes, fig_palettes.items()):
            ax.imshow(swatch.reshape(1, -1, 3))
            ax.set_yticks([]); ax.set_xticks([])
            ax.set_ylabel(genre, rotation=0, ha="right", va="center", fontsize=10)
        plt.suptitle("Dominant-color palette per genre")
        plt.tight_layout()
        plt.savefig(figs / "poster_color_palette.png", dpi=200)
        plt.close()
    return palette


# ---------------------------------------------------------------------------
# Sample grid
# ---------------------------------------------------------------------------

def poster_grid(catalog: pd.DataFrame, output_dir, *, per_genre: int = 6) -> None:
    out_dir = ensure_dir(output_dir)
    figs = ensure_dir(out_dir / "figures")
    genres = sorted(catalog["genre"].unique())
    if not genres:
        return
    fig, axes = plt.subplots(len(genres), per_genre, figsize=(per_genre * 2, 3 * len(genres)))
    axes = np.atleast_2d(axes)
    rng = random.Random(0)
    for row_idx, g in enumerate(genres):
        paths = catalog[catalog["genre"] == g]["image_path"].tolist()
        rng.shuffle(paths)
        for col_idx in range(per_genre):
            ax = axes[row_idx, col_idx]
            ax.axis("off")
            if col_idx < len(paths):
                try:
                    with Image.open(paths[col_idx]) as im:
                        ax.imshow(im.convert("RGB"))
                except Exception:
                    pass
            if col_idx == 0:
                ax.set_ylabel(g, rotation=0, ha="right", va="center", fontsize=11)
    plt.tight_layout()
    plt.savefig(figs / "poster_grid.png", dpi=200)
    plt.close()


# ---------------------------------------------------------------------------
# Orchestrator + CLI
# ---------------------------------------------------------------------------

def run_full_image_eda(
    catalog: pd.DataFrame,
    output_dir,
) -> dict:
    """Convenience: run inventory + size + color + grid in one call."""
    if catalog.empty:
        print("[eda_images] empty catalog — nothing to do.")
        return {}
    return {
        "inventory": poster_inventory(catalog, output_dir),
        "sizes": poster_size_distribution(catalog, output_dir),
        "palette": poster_color_signature(catalog, output_dir),
        # poster_grid returns None (writes a figure)
        "grid": poster_grid(catalog, output_dir),
    }


def build_catalog(
    posters_dir: Optional[Path] = None,
    csv_path: Optional[Path] = None,
    image_column: str = "poster_path",
    label_column: str = "genre",
    fetch: Optional[str] = None,
    genres_to_fetch: Iterable[str] = ("Action", "Comedy", "Horror", "Romance"),
) -> pd.DataFrame:
    """Walk the source-detection chain and return a catalog DataFrame."""
    # 💡 Default path is relative to THIS file's parent's parent (= project root),
    # not the CWD. Avoids the classic "works from project root, fails from
    # notebooks/" bug when the caller doesn't pass an explicit path.
    if posters_dir is None:
        posters_dir = Path(__file__).resolve().parent.parent / "data" / "posters"
    else:
        posters_dir = Path(posters_dir)
    # 1. local folder layout
    if posters_dir.exists():
        cat = discover_local_posters(posters_dir)
        if not cat.empty:
            print(f"[eda_images] found {len(cat)} local posters under {posters_dir}.")
            return cat
    # 2. CSV
    if csv_path is not None and Path(csv_path).exists():
        cat = posters_from_csv(Path(csv_path), image_column, label_column)
        if not cat.empty:
            print(f"[eda_images] loaded {len(cat)} poster paths from {csv_path}.")
            return cat
    # 3. fetch
    if fetch == "tmdb":
        return fetch_from_tmdb(genres_to_fetch, posters_dir)
    if fetch == "kaggle":
        return fetch_from_kaggle(posters_dir)
    print(
        "[eda_images] no poster data available. "
        "Place posters under data/posters/<genre>/, pass --csv with an image path column, "
        "or set TMDB_API_KEY and re-run with --fetch tmdb."
    )
    return pd.DataFrame(columns=["image_path", "genre"])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Poster image EDA for the multimodal bonus.")
    p.add_argument("--posters-dir", type=Path, default=None, help="Folder-per-class root.")
    p.add_argument("--csv", type=Path, default=None, help="CSV with image_path + label columns.")
    p.add_argument("--image-column", default="poster_path")
    p.add_argument("--label-column", default="genre")
    p.add_argument("--fetch", choices=["tmdb", "kaggle"], default=None,
                   help="Network fetch fallback if no local data is found.")
    p.add_argument("--output-dir", default="outputs")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    catalog = build_catalog(
        posters_dir=args.posters_dir,
        csv_path=args.csv,
        image_column=args.image_column,
        label_column=args.label_column,
        fetch=args.fetch,
    )
    if catalog.empty:
        return
    run_full_image_eda(catalog, output_dir)
    print(f"Poster EDA written to {output_dir}/figures and {output_dir}/tables.")


if __name__ == "__main__":
    main()
