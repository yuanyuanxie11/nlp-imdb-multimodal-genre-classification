from __future__ import annotations

import json
from pathlib import Path

from .runtime import prepare_runtime

prepare_runtime()

import joblib
import numpy as np
import tensorflow as tf
from PIL import Image
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


def load_image_bundle(model_dir: str | Path) -> dict | None:
    model_dir = Path(model_dir)
    model_path = model_dir / "image_genre_classifier.keras"
    classes_path = model_dir / "image_label_classes.json"
    config_path = model_dir / "image_config.json"
    if not all(path.exists() for path in [model_path, classes_path]):
        return None

    config = json.loads(config_path.read_text()) if config_path.exists() else {"image_size": [224, 224]}
    return {
        "model": tf.keras.models.load_model(model_path),
        "classes": json.loads(classes_path.read_text()),
        "config": config,
    }


def load_ensemble_bundle(model_dir: str | Path) -> dict | None:
    config_path = Path(model_dir) / "ensemble_config.json"
    if not config_path.exists():
        return None
    config = json.loads(config_path.read_text())
    members = config.get("members", [])
    weights = config.get("weights", {})
    labels = config.get("labels", [])
    if not members or not labels:
        return None
    weight_values = np.asarray([float(weights.get(member, 1.0)) for member in members], dtype=np.float32)
    weight_sum = float(weight_values.sum())
    if weight_sum <= 0:
        return None
    return {
        "members": members,
        "weights": weight_values / weight_sum,
        "labels": labels,
    }


def _prediction_result(labels: list[str], probabilities: np.ndarray) -> dict:
    best_idx = int(np.argmax(probabilities))
    return {
        "prediction": labels[best_idx],
        "probabilities": {
            label: float(prob) for label, prob in zip(labels, probabilities)
        },
    }


def predict_with_baselines(text: str, models: dict) -> dict[str, dict]:
    cleaned = clean_text(text, TextCleaningConfig(remove_stopwords=True))
    predictions = {}
    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba([cleaned])[0]
            labels = model.classes_.tolist()
            predictions[name] = _prediction_result(labels, probabilities)
        else:
            label = model.predict([cleaned])[0]
            predictions[name] = {"prediction": label, "probabilities": {label: 1.0}}
    return predictions


def _align_probability_dict(probabilities: dict[str, float], labels: list[str]) -> np.ndarray:
    return np.asarray([float(probabilities.get(label, 0.0)) for label in labels], dtype=np.float32)


def predict_with_lstm(text: str, bundle: dict | None) -> dict | None:
    if bundle is None:
        return None
    cleaned = clean_text(text, TextCleaningConfig(remove_stopwords=False))
    tokenizer = bundle["tokenizer"]
    config = bundle["config"]
    sequence = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(sequence, maxlen=config["max_len"], padding="post", truncating="post")
    probs = bundle["model"].predict(padded, verbose=0)[0]
    return _prediction_result(bundle["classes"], probs)


def predict_with_ensemble(
    text: str,
    ensemble_bundle: dict | None,
    baseline_models: dict,
    lstm_bundle: dict | None,
) -> dict | None:
    if ensemble_bundle is None:
        return None

    labels = ensemble_bundle["labels"]
    member_probabilities = []
    usable_weights = []
    baseline_predictions = None

    for member, weight in zip(ensemble_bundle["members"], ensemble_bundle["weights"]):
        if member == "lstm":
            result = predict_with_lstm(text, lstm_bundle)
        else:
            if baseline_predictions is None:
                baseline_predictions = predict_with_baselines(text, baseline_models)
            result = baseline_predictions.get(member)
        if result is None:
            continue
        member_probabilities.append(_align_probability_dict(result["probabilities"], labels))
        usable_weights.append(float(weight))

    if not member_probabilities:
        return None

    weights = np.asarray(usable_weights, dtype=np.float32)
    weights = weights / weights.sum()
    probabilities = np.tensordot(weights, np.stack(member_probabilities, axis=0), axes=(0, 0))
    return _prediction_result(labels, probabilities)


def predict_with_image(image_file, bundle: dict | None) -> dict | None:
    if bundle is None:
        return None
    if hasattr(image_file, "seek"):
        image_file.seek(0)
    image_size = tuple(bundle["config"].get("image_size", [224, 224]))
    image = Image.open(image_file).convert("RGB").resize(image_size)
    array = np.asarray(image, dtype=np.float32)
    batch = np.expand_dims(array, axis=0)
    batch = tf.keras.applications.mobilenet_v2.preprocess_input(batch)
    probs = bundle["model"].predict(batch, verbose=0)[0]
    return _prediction_result(bundle["classes"], probs)


def summarize_text(text: str, max_sentences: int = 2) -> str:
    return extractive_summary(text, max_sentences=max_sentences)
