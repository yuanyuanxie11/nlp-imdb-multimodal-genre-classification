from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.app_helpers import (
    load_baseline_models,
    load_lstm_bundle,
    predict_with_baselines,
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


st.set_page_config(page_title="Movie Genre Predictor", layout="wide")
st.title("Movie Genre Classifier and Summarizer")
st.caption("Northwestern Text Analytics Final Project demo scaffold")

tab1, tab2, tab3, tab4 = st.tabs(
    ["Genre Predictor", "Summarizer", "Model Results", "Dataset Notes"]
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
            if baseline_models:
                baseline_preds = predict_with_baselines(sample_text, baseline_models)
                st.markdown("**Baseline model predictions**")
                st.json(baseline_preds)
            else:
                st.info("No baseline models found in the models directory yet.")

            lstm_bundle = _lstm_bundle()
            lstm_result = predict_with_lstm(sample_text, lstm_bundle)
            if lstm_result:
                st.markdown("**LSTM prediction**")
                st.write(lstm_result["prediction"])
                st.bar_chart(pd.DataFrame.from_dict(lstm_result["probabilities"], orient="index"))
            else:
                st.info("No LSTM model bundle found yet.")

with tab2:
    st.subheader("Summarize a movie plot")
    summary_text = st.text_area(
        "Paste a full movie description",
        height=240,
        key="summary_box",
    )
    if st.button("Summarize"):
        if not summary_text.strip():
            st.warning("Enter text first.")
        else:
            st.markdown("**Extractive summary**")
            st.write(summarize_text(summary_text))

with tab3:
    st.subheader("Saved training results")
    comparison_path = OUTPUT_DIR / "tables" / "model_comparison.csv"
    if comparison_path.exists():
        st.dataframe(pd.read_csv(comparison_path), use_container_width=True)
    else:
        st.info("Run the training scripts first to populate model results.")

    figures_dir = OUTPUT_DIR / "figures"
    for image_name in [
        "genre_distribution.png",
        "summary_length_distribution.png",
        "naive_bayes_confusion_matrix.png",
        "logistic_regression_confusion_matrix.png",
        "linear_svm_confusion_matrix.png",
        "lstm_confusion_matrix.png",
        "lstm_accuracy.png",
        "lstm_loss.png",
    ]:
        image_path = figures_dir / image_name
        if image_path.exists():
            st.image(str(image_path), caption=image_name)

with tab4:
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
