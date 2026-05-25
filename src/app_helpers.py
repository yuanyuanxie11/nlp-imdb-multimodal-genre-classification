from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

from .config import TextCleaningConfig
from .data_processing import clean_text
from .summarize import extractive_summary


def load_baseline_models(model_dir: str | Path) -> dict:
    model_dir = Path(model_dir)
    models = {}
    for path in model_dir.glob("*.joblib"):
        models[path.stem] = joblib.load(path)
    return models


def load_lstm_bundle(model_dir: str | Path) -> dict | None:
    model_dir = Path(model_dir)
    model_path = model_dir / "lstm_text_classifier.keras"
    tokenizer_path = model_dir / "lstm_tokenizer.json"
    classes_path = model_dir / "lstm_label_classes.json"
    config_path = model_dir / "lstm_config.json"
    if not all(path.exists() for path in [model_path, tokenizer_path, classes_path, config_path]):
        return None

    tokenizer = tokenizer_from_json(tokenizer_path.read_text())
    return {
        "model": tf.keras.models.load_model(model_path),
        "tokenizer": tokenizer,
        "classes": json.loads(classes_path.read_text()),
        "config": json.loads(config_path.read_text()),
    }


def predict_with_baselines(text: str, models: dict) -> dict[str, str]:
    cleaned = clean_text(text, TextCleaningConfig(remove_stopwords=True))
    return {name: model.predict([cleaned])[0] for name, model in models.items()}


def predict_with_lstm(text: str, bundle: dict | None) -> dict | None:
    if bundle is None:
        return None
    cleaned = clean_text(text, TextCleaningConfig(remove_stopwords=False))
    tokenizer = bundle["tokenizer"]
    config = bundle["config"]
    sequence = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(sequence, maxlen=config["max_len"], padding="post", truncating="post")
    probs = bundle["model"].predict(padded, verbose=0)[0]
    best_idx = int(np.argmax(probs))
    return {
        "prediction": bundle["classes"][best_idx],
        "probabilities": {
            label: float(prob) for label, prob in zip(bundle["classes"], probs)
        },
    }


def summarize_text(text: str) -> str:
    return extractive_summary(text)
