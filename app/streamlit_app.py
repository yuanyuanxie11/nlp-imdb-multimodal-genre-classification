from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.runtime import prepare_runtime

prepare_runtime()

from src.app_helpers import (
    load_ensemble_bundle,
    load_image_bundle,
    load_baseline_models,
    load_lstm_bundle,
    predict_with_baselines,
    predict_with_ensemble,
    predict_with_image,
    predict_with_lstm,
    summarize_text,
)
from src.config import TextCleaningConfig
from src.data_processing import clean_text

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"


@st.cache_resource
def _baseline_models():
    return load_baseline_models(MODEL_DIR)


@st.cache_resource
def _lstm_bundle():
    return load_lstm_bundle(MODEL_DIR)


@st.cache_resource
def _image_bundle():
    return load_image_bundle(MODEL_DIR)


@st.cache_resource
def _ensemble_bundle():
    return load_ensemble_bundle(MODEL_DIR)


def _probability_frame(probabilities: dict[str, float]) -> pd.DataFrame:
    return (
        pd.DataFrame(
            [{"genre": genre, "probability": probability} for genre, probability in probabilities.items()]
        )
        .sort_values("probability", ascending=False)
        .reset_index(drop=True)
    )


def _render_prediction_result(model_name: str, result: dict) -> None:
    st.markdown(f"**{model_name}**")
    st.metric("Prediction", result["prediction"])
    probability_df = _probability_frame(result["probabilities"])
    st.bar_chart(probability_df, x="genre", y="probability", height=220)
    st.dataframe(
        probability_df.assign(probability=lambda df: (df["probability"] * 100).round(2)),
        use_container_width=True,
        hide_index=True,
    )


def _model_results_frame() -> pd.DataFrame | None:
    comparison_path = OUTPUT_DIR / "tables" / "model_comparison.csv"
    if comparison_path.exists():
        comparison = pd.read_csv(comparison_path)
    else:
        comparison = pd.DataFrame()

    rows = []
    for model_name, metrics_file in [
        ("naive_bayes", "naive_bayes_metrics.json"),
        ("logistic_regression", "logistic_regression_metrics.json"),
        ("linear_svm", "linear_svm_metrics.json"),
        ("lstm", "lstm_metrics.json"),
        ("ensemble_soft_vote", "ensemble_metrics.json"),
    ]:
        if not comparison.empty and model_name in comparison["model"].astype(str).values:
            continue
        metrics_path = MODEL_DIR / metrics_file
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text())
            rows.append(
                {
                    "model": model_name,
                    "accuracy": metrics.get("accuracy"),
                    "macro_f1": metrics.get("macro_f1"),
                    "weighted_f1": metrics.get("weighted_f1"),
                }
            )

    if rows:
        comparison = pd.concat([comparison, pd.DataFrame(rows)], ignore_index=True)
    if comparison.empty:
        return None
    sort_cols = [col for col in ["accuracy", "macro_f1"] if col in comparison.columns]
    if sort_cols:
        comparison = comparison.sort_values(by=sort_cols, ascending=False, ignore_index=True)
    return comparison


st.set_page_config(page_title="Movie Genre Predictor", layout="wide")
st.title("Movie Genre Classifier and Summarizer")
st.caption("Northwestern Text Analytics Final Project demo scaffold")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Genre Predictor", "Summarizer", "Poster Predictor", "Model Results", "Dataset Notes"]
)

with tab1:
    st.subheader("Predict a movie genre from a plot summary")
    sample_text = st.text_area(
        "Paste a real or fake movie summary",
        height=240,
        placeholder="A retired detective discovers that the charming neighbor next door may be hiding a deadly secret...",
    )
    if st.button("Predict Genre", type="primary"):
        if not sample_text.strip():
            st.warning("Enter a summary first.")
        else:
            st.markdown("**Cleaned text preview**")
            st.write(clean_text(sample_text, TextCleaningConfig(remove_stopwords=True)))

            baseline_models = _baseline_models()
            prediction_results = []
            if baseline_models:
                baseline_preds = predict_with_baselines(sample_text, baseline_models)
                for model_name, result in baseline_preds.items():
                    prediction_results.append((model_name.replace("_", " ").title(), result))
            else:
                st.info("No baseline models found in the models directory yet.")

            lstm_bundle = _lstm_bundle()
            lstm_result = predict_with_lstm(sample_text, lstm_bundle)
            if lstm_result:
                prediction_results.append(("TensorFlow LSTM", lstm_result))
            else:
                st.info("No LSTM model bundle found yet.")

            ensemble_result = predict_with_ensemble(
                sample_text,
                _ensemble_bundle(),
                baseline_models,
                lstm_bundle,
            )
            if ensemble_result:
                prediction_results.append(("Soft-Voting Ensemble", ensemble_result))
            else:
                st.info("No ensemble config found yet.")

            if prediction_results:
                st.markdown("**Model predictions**")
                columns = st.columns(2)
                for index, (model_name, result) in enumerate(prediction_results):
                    with columns[index % 2]:
                        _render_prediction_result(model_name, result)

with tab2:
    st.subheader("Summarize a movie plot")
    summary_text = st.text_area(
        "Paste a full movie description",
        height=240,
        key="summary_box",
    )
    max_summary_sentences = st.slider("Summary length", min_value=1, max_value=3, value=2)
    if st.button("Summarize"):
        if not summary_text.strip():
            st.warning("Enter text first.")
        else:
            st.markdown("**Extractive summary**")
            st.write(summarize_text(summary_text, max_sentences=max_summary_sentences))

with tab3:
    st.subheader("Predict a movie genre from a poster")
    uploaded_poster = st.file_uploader(
        "Upload a movie poster",
        type=["jpg", "jpeg", "png", "webp"],
    )
    if uploaded_poster is not None:
        st.image(uploaded_poster, width=260)
        image_bundle = _image_bundle()
        image_result = predict_with_image(uploaded_poster, image_bundle)
        if image_result:
            _render_prediction_result("Poster Image Model", image_result)
        else:
            st.info("No poster image classifier found in the models directory yet.")

with tab4:
    st.subheader("Saved training results")
    model_results = _model_results_frame()
    if model_results is not None:
        st.dataframe(model_results, use_container_width=True)
    else:
        st.info("Run the training scripts first to populate model results.")

    figures_dir = OUTPUT_DIR / "figures"
    for image_name in [
        "genre_distribution.png",
        "summary_length_distribution.png",
        "naive_bayes_confusion_matrix.png",
        "logistic_regression_confusion_matrix.png",
        "linear_svm_confusion_matrix.png",
        "ensemble_confusion_matrix.png",
        "lstm_confusion_matrix.png",
        "lstm_accuracy.png",
        "lstm_loss.png",
        "image_classifier_confusion_matrix.png",
        "image_model_training_curves.png",
        "image_model_accuracy.png",
    ]:
        image_path = figures_dir / image_name
        if image_path.exists():
            st.image(str(image_path), caption=image_name)

with tab5:
    st.subheader("Project checklist")
    st.markdown(
        """
        - Load the dataset and confirm the summary and genre columns
        - Run preprocessing and save EDA artifacts
        - Train baseline models and the LSTM model
        - Generate explanations and disagreement analysis
        - Use this app to demo predictions and summarization
        - Add the optional poster-image classifier for bonus credit
        """
    )
