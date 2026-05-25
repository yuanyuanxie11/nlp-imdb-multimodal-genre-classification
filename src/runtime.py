from __future__ import annotations

import os
from pathlib import Path


def prepare_runtime() -> None:
    """Set writable cache locations for plotting libraries in sandboxed environments."""
    cache_dir = Path.cwd() / ".cache"
    mpl_dir = cache_dir / "matplotlib"
    font_dir = cache_dir / "fontconfig"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    font_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
