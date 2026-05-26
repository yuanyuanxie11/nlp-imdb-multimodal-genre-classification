"""Reusable cross-validation utilities.

Centralises every CV protocol used by the project (baselines, LSTM, image),
so all training scripts share the same fold construction, the same seed
sweep, and the same nested-CV semantics.

Design notes
------------
* `pipeline_factory` arguments are callables, NOT pre-built pipelines:
  multi-seed / nested CV need a *fresh* estimator per outer iteration to
  avoid leaked state (fitted vectorizers, learned coefficients). A
  factory `lambda: Pipeline([...])` makes that explicit.
* All public functions return tidy `pandas.DataFrame`s (long format,
  one row per metric per fold) so they can be concatenated, grouped,
  and plotted without further reshaping.
"""

from __future__ import annotations

from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate

from .config import CVConfig, SplitConfig
from .data_processing import stratified_split


# ---------------------------------------------------------------------------
# Fold construction
# ---------------------------------------------------------------------------

def make_stratified_kfold(
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: int = 42,
) -> StratifiedKFold:
    """Build a Stratified K-Fold splitter.

    Thin wrapper so the rest of the project imports from one place; if we
    ever swap to a different CV strategy (e.g. multilabel-stratified), we
    only change it here. ⚠ shuffle=True is critical — without it, the
    split is order-dependent and a sorted dataset would yield
    catastrophically biased folds.
    """
    return StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)


# ---------------------------------------------------------------------------
# Single-pass K-Fold scoring
# ---------------------------------------------------------------------------

def cv_scores(
    estimator,
    X,
    y,
    *,
    cv: StratifiedKFold | None = None,
    scoring: Sequence[str] = ("accuracy", "f1_macro", "f1_weighted"),
    n_jobs: int = -1,
) -> pd.DataFrame:
    """Run K-Fold scoring and return a tidy DataFrame.

    Output columns: ``fold``, ``metric``, ``score``. One row per (fold,
    metric) pair → easy to `groupby('metric').agg(['mean','std'])` for
    summary tables, or to feed straight into `sns.boxplot(x='metric',
    y='score')` for figures.
    """
    cv = cv or make_stratified_kfold()
    # 💡 cross_validate (not cross_val_score) → supports multiple metrics in one pass.
    results = cross_validate(
        estimator,
        X,
        y,
        scoring=list(scoring),
        cv=cv,
        n_jobs=n_jobs,
        return_train_score=False,
    )
    rows = []
    for metric in scoring:
        # sklearn key format is "test_<metric>"
        per_fold = results[f"test_{metric}"]
        for fold_idx, score in enumerate(per_fold):
            rows.append({"fold": fold_idx, "metric": metric, "score": float(score)})
    return pd.DataFrame(rows)


def summarise_cv(scores: pd.DataFrame, group_cols: Sequence[str] = ("metric",)) -> pd.DataFrame:
    """Aggregate `cv_scores` output into mean ± std per metric (and any extra grouping)."""
    summary = (
        scores.groupby(list(group_cols))["score"]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
    )
    return summary


# ---------------------------------------------------------------------------
# Multi-seed stability
# ---------------------------------------------------------------------------

def multi_seed_cv(
    pipeline_factory: Callable[[], object],
    X,
    y,
    *,
    seeds: Iterable[int] = (13, 42, 2024),
    n_splits: int = 5,
    scoring: Sequence[str] = ("accuracy", "f1_macro", "f1_weighted"),
    n_jobs: int = -1,
) -> pd.DataFrame:
    """K-Fold across multiple seeds → measures between-seed variance.

    A single Stratified K-Fold tells you fold-to-fold variance (data
    sensitivity). Re-running across different seeds tells you whether
    your reported number is sensitive to the *random partitioning*
    itself. A wide between-seed std means "one lucky split could have
    shifted my reported number by X" — important context for any final
    report.
    """
    frames = []
    for seed in seeds:
        # Fresh estimator per seed: pipeline_factory() returns an UNFITTED instance.
        # ⚠ if you pass a pre-built Pipeline, the second seed would refit a
        # model that already has state from seed 1 — silently invalid.
        estimator = pipeline_factory()
        cv = make_stratified_kfold(n_splits=n_splits, random_state=seed)
        per_seed = cv_scores(estimator, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs)
        per_seed.insert(0, "seed", seed)
        frames.append(per_seed)
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Nested CV (outer = honest score, inner = hyperparameter search)
# ---------------------------------------------------------------------------

def nested_cv(
    pipeline_factory: Callable[[], object],
    param_grid: dict,
    X,
    y,
    *,
    outer_splits: int = 5,
    inner_splits: int = 3,
    scoring: str = "f1_macro",
    random_state: int = 42,
    n_jobs: int = -1,
) -> tuple[pd.DataFrame, list[dict]]:
    """Honest hyperparameter evaluation.

    Why nested CV?
    --------------
    Plain GridSearchCV with cross_val_score reports the *best* CV score
    found during the search. That number is biased upward: every config
    you tried got to "peek" at the validation folds. Nested CV fixes
    this by selecting hyperparameters inside an inner loop and scoring
    the chosen model on a held-out outer fold the inner loop never saw.

    Returns
    -------
    scores_df : DataFrame with columns ``outer_fold``, ``score``.
    best_params_per_fold : list of dicts — one chosen config per outer
        fold. If these dicts disagree across folds, your "best" hyper-
        parameters are unstable and you should not trust any single
        winner. This is exactly the kind of finding plain GridSearch
        hides.
    """
    outer = make_stratified_kfold(n_splits=outer_splits, random_state=random_state)
    inner = make_stratified_kfold(n_splits=inner_splits, random_state=random_state)

    # 💡 X/y might be pandas — convert to numpy arrays for index-based slicing.
    X_arr = np.asarray(X) if not hasattr(X, "iloc") else X
    y_arr = np.asarray(y) if not hasattr(y, "iloc") else y

    rows = []
    best_params: list[dict] = []
    for fold_idx, (train_idx, test_idx) in enumerate(outer.split(X_arr, y_arr)):
        if hasattr(X_arr, "iloc"):
            X_tr, X_te = X_arr.iloc[train_idx], X_arr.iloc[test_idx]
            y_tr, y_te = y_arr.iloc[train_idx], y_arr.iloc[test_idx]
        else:
            X_tr, X_te = X_arr[train_idx], X_arr[test_idx]
            y_tr, y_te = y_arr[train_idx], y_arr[test_idx]

        search = GridSearchCV(
            estimator=pipeline_factory(),
            param_grid=param_grid,
            scoring=scoring,
            cv=inner,
            n_jobs=n_jobs,
            refit=True,
        )
        search.fit(X_tr, y_tr)
        outer_score = search.score(X_te, y_te)
        rows.append({"outer_fold": fold_idx, "score": float(outer_score)})
        best_params.append(search.best_params_)
    return pd.DataFrame(rows), best_params


# ---------------------------------------------------------------------------
# Hold-out + K-Fold composition
# ---------------------------------------------------------------------------

def holdout_then_cv(
    df: pd.DataFrame,
    label_column: str,
    *,
    split_config: SplitConfig | None = None,
    cv_config: CVConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, StratifiedKFold]:
    """Carve out the existing 20% hold-out, then return a K-Fold splitter for the rest.

    Lets us keep the project's existing test set *exactly as is* (for
    final reporting) while still doing K-Fold model selection on the
    remaining 80% (= train + val combined).

    Returns
    -------
    train_pool_df : the 80% used for K-Fold.
    test_df : the untouched 20% hold-out.
    skf : a StratifiedKFold instance already configured.
    """
    split_config = split_config or SplitConfig()
    cv_config = cv_config or CVConfig()
    train_df, val_df, test_df = stratified_split(df, label_column, split_config)
    # Merge train+val into one "training pool" for K-Fold to subdivide.
    train_pool = pd.concat([train_df, val_df], ignore_index=True)
    skf = make_stratified_kfold(
        n_splits=cv_config.n_splits, random_state=split_config.random_state
    )
    return train_pool, test_df, skf


# ---------------------------------------------------------------------------
# Sanity-check helper: did stratification actually hold?
# ---------------------------------------------------------------------------

def check_fold_balance(
    cv: StratifiedKFold,
    X,
    y,
) -> pd.DataFrame:
    """Per-fold class-count table; lets you verify stratification visually.

    For a perfectly stratified split, each fold's class proportions
    should match the corpus proportions to within ±1 sample. If not,
    something is off (e.g. you passed a non-stratified splitter by
    mistake, or y contains NaNs).
    """
    y_arr = np.asarray(y) if not hasattr(y, "iloc") else y.reset_index(drop=True)
    rows = []
    for fold_idx, (_, val_idx) in enumerate(cv.split(np.zeros(len(y_arr)), y_arr)):
        if hasattr(y_arr, "iloc"):
            fold_y = y_arr.iloc[val_idx]
        else:
            fold_y = y_arr[val_idx]
        counts = pd.Series(fold_y).value_counts().to_dict()
        for cls, n in counts.items():
            rows.append({"fold": fold_idx, "class": cls, "count": int(n)})
    return pd.DataFrame(rows)
