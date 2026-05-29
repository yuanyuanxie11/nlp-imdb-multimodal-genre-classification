from __future__ import annotations

import os
import random
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


def set_global_seed(seed: int, *, deterministic_tf: bool = True) -> None:
    """Seed every RNG that matters so multi-seed sweeps are meaningful.

    Why all of this?
    ----------------
    A "fixed random_state" only nails down numpy/sklearn behaviour. TensorFlow
    has its own RNGs (per-op kernel init, dropout masks, shuffle buffers) and
    Python's ``random`` is used by some preprocessing helpers. Without seeding
    all of them, swapping ``random_state=42`` for ``random_state=2024`` would
    still produce identical TF runs → "multi-seed std" would falsely read 0.

    ⚠ ``TF_DETERMINISTIC_OPS`` must be set BEFORE TensorFlow is imported for
    the first time; if TF is already imported this call still seeds it but
    full op-level determinism may not engage. The function is therefore safe
    to call once per training run (just don't set the env var, re-import TF,
    and expect bitwise reproducibility — that's a stronger guarantee than we
    need here).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic_tf:
        os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import tensorflow as tf
        tf.keras.utils.set_random_seed(seed)
    except ImportError:
        pass
