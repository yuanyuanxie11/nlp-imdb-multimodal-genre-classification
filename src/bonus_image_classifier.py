from __future__ import annotations

import argparse
import json
from pathlib import Path

from .runtime import prepare_runtime

prepare_runtime()

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

from .config import ensure_dir
from .evaluate import compute_metrics, save_confusion_matrix, save_metrics


def build_image_model(image_size: tuple[int, int], num_classes: int) -> tf.keras.Model:
    base = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(image_size[0], image_size[1], 3),
    )
    base.trainable = False
    x = GlobalAveragePooling2D()(base.output)
    outputs = Dense(num_classes, activation="softmax")(x)
    model = Model(inputs=base.input, outputs=outputs)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bonus image-only genre classifier.")
    parser.add_argument("--input", required=True, help="Dataset with image path and label columns.")
    parser.add_argument("--image-column", required=True)
    parser.add_argument("--label-column", required=True)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--img-size", type=int, nargs=2, default=[224, 224])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    ensure_dir(Path(output_dir) / "figures")
    ensure_dir(Path(output_dir) / "tables")
    model_dir = ensure_dir(args.model_dir)

    df = pd.read_csv(args.input)
    df = df[[args.image_column, args.label_column]].dropna().copy()
    df = df[df[args.image_column].map(lambda p: Path(p).exists())]
    if df.empty:
        raise ValueError("No valid image paths found.")

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df[args.label_column],
        random_state=42,
    )
    train_df, val_df = train_test_split(
        train_df,
        test_size=0.125,
        stratify=train_df[args.label_column],
        random_state=42,
    )

    encoder = LabelEncoder()
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    train_df["label_id"] = encoder.fit_transform(train_df[args.label_column])
    val_df["label_id"] = encoder.transform(val_df[args.label_column])
    test_df["label_id"] = encoder.transform(test_df[args.label_column])

    image_size = tuple(args.img_size)
    train_ds = tf.keras.utils.image_dataset_from_directory
    del train_ds

    def make_dataset(frame: pd.DataFrame, shuffle: bool) -> tf.data.Dataset:
        paths = frame[args.image_column].astype(str).tolist()
        labels = frame["label_id"].tolist()
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))

        def _load(path, label):
            image = tf.io.read_file(path)
            image = tf.image.decode_image(image, channels=3, expand_animations=False)
            image = tf.image.resize(image, image_size)
            image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
            return image, label

        ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
        if shuffle:
            ds = ds.shuffle(512, seed=42)
        return ds.batch(32).prefetch(tf.data.AUTOTUNE)

    train_ds = make_dataset(train_df, shuffle=True)
    val_ds = make_dataset(val_df, shuffle=False)
    test_ds = make_dataset(test_df, shuffle=False)

    model = build_image_model(image_size, len(encoder.classes_))
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=[EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)],
        verbose=1,
    )

    probs = model.predict(test_ds, verbose=0)
    pred_ids = probs.argmax(axis=1)
    preds = encoder.inverse_transform(pred_ids)
    y_true = test_df[args.label_column].tolist()
    labels = encoder.classes_.tolist()

    metrics = compute_metrics(y_true, preds, labels)
    save_metrics(metrics, model_dir / "image_classifier_metrics.json")
    save_confusion_matrix(
        y_true,
        preds,
        labels,
        "Image classifier confusion matrix",
        Path(output_dir) / "figures" / "image_classifier_confusion_matrix.png",
    )

    plt.figure(figsize=(8, 5))
    plt.plot(history.history.get("accuracy", []), label="train_accuracy")
    plt.plot(history.history.get("val_accuracy", []), label="val_accuracy")
    plt.title("Image Model Accuracy Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "figures" / "image_model_accuracy.png", dpi=200)
    plt.close()

    model.save(model_dir / "image_genre_classifier.keras")
    Path(model_dir / "image_label_classes.json").write_text(json.dumps(labels, indent=2))
    print(json.dumps({k: v for k, v in metrics.items() if k != "report"}, indent=2))


if __name__ == "__main__":
    main()
