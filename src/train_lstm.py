from __future__ import annotations

import argparse
import json
from pathlib import Path

from .runtime import prepare_runtime

prepare_runtime()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from .config import LSTMConfig, ensure_dir
from .data_processing import add_clean_columns, load_dataset, stratified_split
from .evaluate import compute_metrics, save_confusion_matrix, save_metrics


def build_model(num_classes: int, config: LSTMConfig) -> tf.keras.Model:
    model = Sequential(
        [
            Embedding(config.max_words, config.embedding_dim, input_length=config.max_len),
            LSTM(
                config.lstm_units,
                dropout=config.dropout,
                recurrent_dropout=config.recurrent_dropout,
            ),
            Dense(config.dense_units, activation="relu"),
            Dropout(config.dropout),
            Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def plot_history(history: tf.keras.callbacks.History, output_dir: Path) -> None:
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(output_dir / "tables" / "lstm_history.csv", index=False)

    for metric in ("accuracy", "loss"):
        plt.figure(figsize=(8, 5))
        plt.plot(history.history.get(metric, []), label=f"train_{metric}")
        plt.plot(history.history.get(f"val_{metric}", []), label=f"val_{metric}")
        plt.title(f"LSTM {metric.title()} Over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel(metric.title())
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "figures" / f"lstm_{metric}.png", dpi=200)
        plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a TensorFlow LSTM classifier.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--text-column", required=True)
    parser.add_argument("--label-column", required=True)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--epochs", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    ensure_dir(Path(output_dir) / "figures")
    ensure_dir(Path(output_dir) / "tables")
    model_dir = ensure_dir(args.model_dir)

    config = LSTMConfig(epochs=args.epochs)
    df = add_clean_columns(load_dataset(args.input), args.text_column)
    train_df, val_df, test_df = stratified_split(df, args.label_column)

    tokenizer = Tokenizer(num_words=config.max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(train_df["text_clean_neural"])

    def vectorize(texts):
        sequences = tokenizer.texts_to_sequences(texts)
        return pad_sequences(sequences, maxlen=config.max_len, padding="post", truncating="post")

    x_train = vectorize(train_df["text_clean_neural"])
    x_val = vectorize(val_df["text_clean_neural"])
    x_test = vectorize(test_df["text_clean_neural"])

    encoder = LabelEncoder()
    y_train = encoder.fit_transform(train_df[args.label_column])
    y_val = encoder.transform(val_df[args.label_column])
    y_test = encoder.transform(test_df[args.label_column])
    labels = encoder.classes_.tolist()

    model = build_model(len(labels), config)
    early_stop = EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=config.epochs,
        batch_size=config.batch_size,
        callbacks=[early_stop],
        verbose=1,
    )

    test_probs = model.predict(x_test, verbose=0)
    test_pred_ids = np.argmax(test_probs, axis=1)
    test_preds = encoder.inverse_transform(test_pred_ids)
    y_true = encoder.inverse_transform(y_test)

    metrics = compute_metrics(y_true, test_preds, labels)
    save_metrics(metrics, model_dir / "lstm_metrics.json")
    save_confusion_matrix(
        y_true,
        test_preds,
        labels,
        "LSTM confusion matrix",
        Path(output_dir) / "figures" / "lstm_confusion_matrix.png",
    )
    plot_history(history, Path(output_dir))

    model.save(model_dir / "lstm_text_classifier.keras")
    Path(model_dir / "lstm_label_classes.json").write_text(json.dumps(labels, indent=2))
    Path(model_dir / "lstm_tokenizer.json").write_text(tokenizer.to_json())
    Path(model_dir / "lstm_config.json").write_text(json.dumps(config.__dict__, indent=2))

    summary_lines: list[str] = []
    model.summary(print_fn=summary_lines.append)
    Path(model_dir / "lstm_model_summary.txt").write_text("\n".join(summary_lines))
    pd.DataFrame(
        {
            args.text_column: test_df[args.text_column].tolist(),
            args.label_column: y_true.tolist(),
            "lstm_prediction": test_preds.tolist(),
        }
    ).to_csv(Path(output_dir) / "tables" / "lstm_predictions.csv", index=False)
    print(json.dumps({k: v for k, v in metrics.items() if k != "report"}, indent=2))


if __name__ == "__main__":
    main()
