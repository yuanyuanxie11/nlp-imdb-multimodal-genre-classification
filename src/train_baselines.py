"""Baseline text classifiers with full CV protocol.

Pipeline per model:
    1. Carve out the held-out 20% test set (untouched throughout).
    2. Run Stratified K-Fold on the remaining 80% train pool → CV mean ± std.
    3. Run GridSearchCV on the train pool → pick best hyperparameters.
    4. Fit the best-config pipeline on the full train pool.
    5. Score that fitted pipeline on the held-out test set → unbiased final number.
    6. Optionally run nested CV on a chosen model (`--nested`) to also report an
       *unbiased* estimate of the hyperparameter-tuned model itself.

⚠ The held-out test is consulted exactly once per model (step 5). No model
   selection, no early stopping, no threshold tuning touches it.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .config import CVConfig, SplitConfig, ensure_dir
from .cross_validation import (
    check_fold_balance,
    cv_scores,
    holdout_then_cv,
    make_stratified_kfold,
    nested_cv,
    summarise_cv,
)
from .data_processing import add_clean_columns, load_dataset
from .evaluate import comparison_frame, compute_metrics, save_confusion_matrix, save_metrics


# ---------------------------------------------------------------------------
# Model factories — return UNFITTED Pipelines.
# ---------------------------------------------------------------------------

# 💡 Pulled out as factories (callables) so multi-seed / nested CV can build
# a fresh Pipeline per iteration. See cross_validation.py for the rationale.
def _common_vectorizer_kwargs() -> dict:
    return dict(lowercase=False, ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True)


def naive_bayes_factory() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(**_common_vectorizer_kwargs())),
        ("model", MultinomialNB()),
    ])


def logistic_regression_factory() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(**_common_vectorizer_kwargs())),
        ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])


def linear_svm_factory() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(**_common_vectorizer_kwargs())),
        ("model", LinearSVC()),
    ])


MODEL_REGISTRY: dict[str, Callable[[], Pipeline]] = {
    "naive_bayes": naive_bayes_factory,
    "logistic_regression": logistic_regression_factory,
    "linear_svm": linear_svm_factory,
}


# Small but meaningful grids → kept under a minute on a laptop.
# Shared TF-IDF axis lets us spot whether TF-IDF or model-side hyperparams matter more.
_SHARED_TFIDF_GRID = {
    "tfidf__min_df": [1, 2, 5],
    "tfidf__ngram_range": [(1, 1), (1, 2)],
}
PARAM_GRIDS: dict[str, dict] = {
    "naive_bayes": {**_SHARED_TFIDF_GRID, "model__alpha": [0.1, 0.5, 1.0]},
    "logistic_regression": {**_SHARED_TFIDF_GRID, "model__C": [0.1, 1.0, 10.0]},
    "linear_svm": {**_SHARED_TFIDF_GRID, "model__C": [0.1, 1.0, 10.0]},
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train baseline text classifiers with full CV.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--text-column", required=True)
    parser.add_argument("--label-column", required=True)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--cv-folds", type=int, default=5, help="K for the primary Stratified K-Fold.")
    parser.add_argument(
        "--grid-search",
        action="store_true",
        help="Run GridSearchCV per model (slower but picks better hyperparameters).",
    )
    parser.add_argument(
        "--nested",
        action="store_true",
        help="Run nested CV on logistic_regression (slowest; gives unbiased tuned score).",
    )
    parser.add_argument(
        "--skip-models",
        nargs="*",
        default=[],
        help="Model names to skip (useful for fast iteration).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Plotting helper
# ---------------------------------------------------------------------------

def _save_cv_boxplot(cv_scores_df: pd.DataFrame, output_path: Path) -> None:
    """Box plot of per-fold macro-F1 per model → visual variance comparison."""
    macro = cv_scores_df[cv_scores_df["metric"] == "f1_macro"]
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=macro, x="model", y="score")
    sns.stripplot(data=macro, x="model", y="score", color="black", size=4, alpha=0.6)
    plt.title("Per-fold macro-F1 across baselines (K-Fold CV)")
    plt.ylabel("macro-F1")
    plt.xlabel("")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    figures_dir = ensure_dir(Path(output_dir) / "figures")
    tables_dir = ensure_dir(Path(output_dir) / "tables")
    model_dir = ensure_dir(args.model_dir)

    df = add_clean_columns(load_dataset(args.input), args.text_column)
    labels = sorted(df[args.label_column].dropna().unique().tolist())

    # ---- Step 1: hold-out test + K-Fold splitter on remaining 80% ----
    cv_config = CVConfig(n_splits=args.cv_folds)
    train_pool, test_df, skf = holdout_then_cv(
        df, args.label_column, split_config=SplitConfig(), cv_config=cv_config
    )
    x_pool = train_pool["text_clean_classical"]
    y_pool = train_pool[args.label_column]
    x_test = test_df["text_clean_classical"]
    y_test = test_df[args.label_column]

    # Sanity: confirm stratification holds across folds (writes a CSV).
    fold_balance = check_fold_balance(skf, x_pool, y_pool)
    fold_balance.to_csv(tables_dir / "fold_class_distribution.csv", index=False)

    # ---- Step 2-5: per model ----
    all_cv_rows: list[pd.DataFrame] = []
    holdout_rows: list[dict] = []
    grid_rows: list[dict] = []
    prediction_table = test_df[[args.text_column, args.label_column]].copy()

    selected_models = {n: f for n, f in MODEL_REGISTRY.items() if n not in args.skip_models}

    for name, factory in selected_models.items():
        print(f"\n── {name} ─────────────────────────────────────────────")

        # Step 2: K-Fold on the train pool (uses the default config from the factory).
        per_fold = cv_scores(
            factory(),
            x_pool,
            y_pool,
            cv=skf,
            scoring=cv_config.scoring,
            n_jobs=cv_config.n_jobs,
        )
        per_fold.insert(0, "model", name)
        all_cv_rows.append(per_fold)
        cv_summary = summarise_cv(per_fold[["metric", "score"]])
        print("CV summary:")
        print(cv_summary.to_string(index=False))

        # Step 3: GridSearchCV (optional but on by default for the headline run).
        if args.grid_search:
            search = GridSearchCV(
                estimator=factory(),
                param_grid=PARAM_GRIDS[name],
                scoring="f1_macro",
                cv=skf,
                n_jobs=cv_config.n_jobs,
                refit=True,
            )
            search.fit(x_pool, y_pool)
            best_estimator = search.best_estimator_
            grid_rows.append({
                "model": name,
                "best_cv_f1_macro": float(search.best_score_),
                **{f"best_{k}": str(v) for k, v in search.best_params_.items()},
            })
            print(f"GridSearch best macro-F1 (inner CV): {search.best_score_:.4f}")
            print(f"GridSearch best params: {search.best_params_}")
        else:
            # Step 4 with default config (no grid search): fit fresh on the full pool.
            best_estimator = factory()
            best_estimator.fit(x_pool, y_pool)

        # Step 5: score on the untouched hold-out test set.
        test_pred = best_estimator.predict(x_test)
        metrics = compute_metrics(y_test, test_pred, labels)
        save_metrics(metrics, model_dir / f"{name}_metrics.json")
        save_confusion_matrix(
            y_test, test_pred, labels,
            f"{name} confusion matrix (hold-out test)",
            figures_dir / f"{name}_confusion_matrix.png",
        )
        joblib.dump(best_estimator, model_dir / f"{name}.joblib")
        prediction_table[f"{name}_prediction"] = test_pred

        holdout_rows.append({
            "model": name,
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "cv_f1_macro_mean": float(per_fold[per_fold["metric"] == "f1_macro"]["score"].mean()),
            "cv_f1_macro_std": float(per_fold[per_fold["metric"] == "f1_macro"]["score"].std()),
        })

    # ---- Step 6 (optional): nested CV on logistic_regression as the canonical example ----
    if args.nested and "logistic_regression" in selected_models:
        print("\n── nested CV on logistic_regression ──────────────────")
        nested_df, best_params_per_fold = nested_cv(
            logistic_regression_factory,
            PARAM_GRIDS["logistic_regression"],
            x_pool, y_pool,
            outer_splits=cv_config.outer_splits,
            inner_splits=cv_config.inner_splits,
            scoring="f1_macro",
            n_jobs=cv_config.n_jobs,
        )
        nested_df.to_csv(tables_dir / "nested_cv_logistic_regression.csv", index=False)
        print(nested_df.to_string(index=False))
        print(f"nested mean ± std (macro-F1): {nested_df['score'].mean():.4f} ± {nested_df['score'].std():.4f}")
        # ⚠ if the best_params disagree across outer folds → your "winning" config is not stable.
        print("best params per outer fold:")
        for i, p in enumerate(best_params_per_fold):
            print(f"  fold {i}: {p}")
        pd.DataFrame(best_params_per_fold).to_csv(
            tables_dir / "nested_cv_logistic_regression_best_params.csv", index=False
        )

    # ---- Persist CV artifacts ----
    cv_all = pd.concat(all_cv_rows, ignore_index=True)
    cv_all.to_csv(tables_dir / "baseline_cv_scores.csv", index=False)
    summarise_cv(cv_all, group_cols=["model", "metric"]).to_csv(
        tables_dir / "baseline_cv_summary.csv", index=False
    )
    if grid_rows:
        pd.DataFrame(grid_rows).to_csv(tables_dir / "baseline_grid_search.csv", index=False)
    _save_cv_boxplot(cv_all, figures_dir / "baseline_cv_boxplot.png")

    # ---- Final comparison table (legacy artifact, plus new CV columns) ----
    comparison = comparison_frame(holdout_rows)
    comparison.to_csv(tables_dir / "model_comparison.csv", index=False)
    prediction_table.to_csv(tables_dir / "baseline_predictions.csv", index=False)
    print("\n── Final comparison (hold-out test + CV summary) ──")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
