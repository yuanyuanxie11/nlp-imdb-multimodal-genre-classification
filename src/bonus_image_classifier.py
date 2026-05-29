"""
bonus_image_classifier.py
─────────────────────────
Bonus multimodal extension: train a MobileNetV2 image classifier on IMDB
movie posters, then compare against the text-only models.

Poster folder layout expected:
    data/IMDB four_genre_posters/
        Action/   tt0000001.jpg  tt0000002.jpg  …
        Comedy/   …
        Horror/   …
        Romance/  …

The CSV only needs columns  movie_id  and  genre  (lowercase).
Image paths are built by capitalising the genre and appending  movie_id.jpg.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from .runtime import prepare_runtime, set_global_seed

prepare_runtime()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model

from .config import SplitConfig, ensure_dir
from .data_processing import stratified_split
from .evaluate import compute_metrics, save_confusion_matrix, save_metrics


# ── Dataset builder ────────────────────────────────────────────────────────────

def build_image_dataframe(
    csv_path: str | Path,
    poster_dir: str | Path,
    movie_id_col: str = "movie_id",
    label_col: str = "genre",
) -> pd.DataFrame:
    """
    Join the CSV with poster files.

    Returns a DataFrame with columns:
        movie_id, genre, image_path
    Only rows where the poster file actually exists are kept.
    """
    df = pd.read_csv(csv_path)

    poster_root = Path(poster_dir)
    # Genre in CSV is lowercase; folder names are capitalised
    df["image_path"] = df.apply(
        lambda r: str(
            poster_root / r[label_col].capitalize() / f"{r[movie_id_col]}.jpg"
        ),
        axis=1,
    )
    before = len(df)
    df = df[df["image_path"].map(lambda p: Path(p).exists())].reset_index(drop=True)
    after = len(df)
    print(f"[image] Matched {after}/{before} movies to poster files.")
    print(df[label_col].value_counts().to_string())
    return df


# ── Model ──────────────────────────────────────────────────────────────────────

def build_image_model(num_classes: int, image_size: tuple[int, int] = (224, 224)) -> tf.keras.Model:
    """
    MobileNetV2 transfer-learning classifier.
    Base is frozen; we add a GlobalAveragePooling2D → Dropout → Dense head.
    """
    base = MobileNetV2(
        weights="imagenet", include_top=False,
        input_shape=(image_size[0], image_size[1], 3),
    )
    base.trainable = False   # freeze ImageNet weights

    x = GlobalAveragePooling2D()(base.output)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base.input, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ── tf.data pipeline ──────────────────────────────────────────────────────────

def make_tf_dataset(
    frame: pd.DataFrame,
    image_size: tuple[int, int],
    batch_size: int,
    shuffle: bool,
) -> tf.data.Dataset:
    paths  = frame["image_path"].astype(str).tolist()
    labels = frame["label_id"].tolist()

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def _load(path: tf.Tensor, label: tf.Tensor):
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, image_size)
        image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
        return image, label

    def _augment(image: tf.Tensor, label: tf.Tensor):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=0.1)
        return image, label

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(512, seed=42)
        ds = ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# ── Training curves ────────────────────────────────────────────────────────────

def plot_image_history(history: tf.keras.callbacks.History, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["accuracy"],     label="Train",      marker="o", linewidth=2)
    axes[0].plot(history.history["val_accuracy"], label="Validation", marker="s",
                 linestyle="--", linewidth=2)
    axes[0].set_title("Poster Classifier — Accuracy", fontsize=12)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy")
    axes[0].legend(); axes[0].set_ylim(0, 1.05)

    axes[1].plot(history.history["loss"],     label="Train",      marker="o",
                 color="tomato", linewidth=2)
    axes[1].plot(history.history["val_loss"], label="Validation", marker="s",
                 color="salmon", linestyle="--", linewidth=2)
    axes[1].set_title("Poster Classifier — Loss", fontsize=12)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_dir / "figures" / "image_model_training_curves.png", dpi=150)
    plt.close()


# ── Main training function (callable from notebook) ───────────────────────────

def train_image_classifier(
    csv_path: str | Path,
    poster_dir: str | Path,
    output_dir: str | Path,
    model_dir: str | Path,
    image_size: tuple[int, int] = (224, 224),
    batch_size: int = 32,
    epochs: int = 15,
    label_col: str = "genre",
    movie_id_col: str = "movie_id",
) -> dict:
    """
    Full pipeline: load → split → train MobileNetV2 → evaluate → save.

    Returns a metrics dict (accuracy, macro_f1, …).
    """
    output_dir = ensure_dir(output_dir)
    ensure_dir(Path(output_dir) / "figures")
    ensure_dir(Path(output_dir) / "tables")
    model_dir  = ensure_dir(model_dir)

    # ── Dataset ───────────────────────────────────────────────────────────────
    df = build_image_dataframe(csv_path, poster_dir, movie_id_col, label_col)
    if len(df) < 50:
        raise ValueError(f"Too few matched images ({len(df)}). Check poster_dir path.")

    train_df, val_df, test_df = stratified_split(df, label_col)

    encoder = LabelEncoder()
    train_df = train_df.copy()
    val_df   = val_df.copy()
    test_df  = test_df.copy()
    train_df["label_id"] = encoder.fit_transform(train_df[label_col])
    val_df["label_id"]   = encoder.transform(val_df[label_col])
    test_df["label_id"]  = encoder.transform(test_df[label_col])
    labels = encoder.classes_.tolist()

    # Class weights to handle poster count imbalance (Horror > Comedy/Romance)
    cw_values = compute_class_weight(
        "balanced",
        classes=np.arange(len(labels)),
        y=train_df["label_id"].values,
    )
    class_weight = dict(enumerate(cw_values))
    print(f"\n[image] Class weights: { {labels[k]: round(v, 3) for k, v in class_weight.items()} }")

    train_ds = make_tf_dataset(train_df, image_size, batch_size, shuffle=True)
    val_ds   = make_tf_dataset(val_df,   image_size, batch_size, shuffle=False)
    test_ds  = make_tf_dataset(test_df,  image_size, batch_size, shuffle=False)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_image_model(len(labels), image_size)
    model.summary()

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1),
    ]

    print(f"\n[image] Training on {len(train_df)} images, validating on {len(val_df)} …")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )

    plot_image_history(history, Path(output_dir))

    # ── Evaluation ────────────────────────────────────────────────────────────
    print(f"\n[image] Evaluating on {len(test_df)} test images …")
    probs    = model.predict(test_ds, verbose=0)
    pred_ids = probs.argmax(axis=1)
    preds    = encoder.inverse_transform(pred_ids)
    y_true   = test_df[label_col].tolist()

    metrics = compute_metrics(y_true, preds, labels)
    save_metrics(metrics, Path(model_dir) / "image_classifier_metrics.json")
    save_confusion_matrix(
        y_true, preds, labels,
        "Poster Classifier Confusion Matrix",
        Path(output_dir) / "figures" / "image_classifier_confusion_matrix.png",
    )

    # Save predictions
    test_df["image_prediction"] = preds
    test_df[["movie_id", label_col, "image_prediction"]].to_csv(
        Path(output_dir) / "tables" / "image_predictions.csv", index=False
    )

    # ── Save model ────────────────────────────────────────────────────────────
    model.save(Path(model_dir) / "image_genre_classifier.keras")
    (Path(model_dir) / "image_label_classes.json").write_text(json.dumps(labels, indent=2))
    (Path(model_dir) / "image_config.json").write_text(
        json.dumps({"image_size": list(image_size)}, indent=2)
    )

    print(f"\n[image] Test accuracy: {metrics['accuracy']:.4f}  "
          f"Macro-F1: {metrics['macro_f1']:.4f}")
    return metrics


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bonus MobileNetV2 poster genre classifier.")
    p.add_argument("--input",       required=True, help="CSV with movie_id and genre columns.")
    p.add_argument("--poster-dir",  required=True,
                   help="Root folder with Action/Comedy/Horror/Romance subfolders.")
    p.add_argument("--output-dir",  default="outputs")
    p.add_argument("--model-dir",   default="models")
    p.add_argument("--epochs",      type=int, default=15)
    p.add_argument("--batch-size",  type=int, default=32)
    p.add_argument("--img-size",    type=int, nargs=2, default=[224, 224])
    return p.parse_args()


def _make_dataset(frame: pd.DataFrame, image_column: str, image_size, *, shuffle: bool, seed: int) -> tf.data.Dataset:
    paths = frame[image_column].astype(str).tolist()
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
        ds = ds.shuffle(512, seed=seed)
    return ds.batch(32).prefetch(tf.data.AUTOTUNE)


def train_one_run(
    df: pd.DataFrame,
    image_column: str,
    label_column: str,
    image_size: tuple[int, int],
    epochs: int,
    seed: int,
    output_dir: Path,
    model_dir: Path,
    *,
    save_artifacts: bool = True,
) -> dict:
    """Single full training run. Returns metrics; optionally writes artifacts."""
    set_global_seed(seed)
    train_df, val_df, test_df = stratified_split(
        df, label_column, SplitConfig(random_state=seed),
    )

    encoder = LabelEncoder()
    for sub in (train_df, val_df, test_df):
        sub["label_id"] = encoder.fit_transform(sub[label_column]) if sub is train_df \
            else encoder.transform(sub[label_column])

    train_ds = _make_dataset(train_df, image_column, image_size, shuffle=True, seed=seed)
    val_ds = _make_dataset(val_df, image_column, image_size, shuffle=False, seed=seed)
    test_ds = _make_dataset(test_df, image_column, image_size, shuffle=False, seed=seed)

    model = build_image_model(len(encoder.classes_), image_size)
    history = model.fit(
        train_ds, validation_data=val_ds, epochs=epochs,
        callbacks=[EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)],
        verbose=1 if save_artifacts else 0,
    )

    probs = model.predict(test_ds, verbose=0)
    pred_ids = probs.argmax(axis=1)
    preds = encoder.inverse_transform(pred_ids)
    y_true = test_df[label_column].tolist()
    labels = encoder.classes_.tolist()
    metrics = compute_metrics(y_true, preds, labels)

    if save_artifacts:
        save_metrics(metrics, model_dir / "image_classifier_metrics.json")
        save_confusion_matrix(
            y_true, preds, labels,
            "Image classifier confusion matrix",
            output_dir / "figures" / "image_classifier_confusion_matrix.png",
        )
        plt.figure(figsize=(8, 5))
        plt.plot(history.history.get("accuracy", []), label="train_accuracy")
        plt.plot(history.history.get("val_accuracy", []), label="val_accuracy")
        plt.title("Image Model Accuracy Over Epochs")
        plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "figures" / "image_model_accuracy.png", dpi=200)
        plt.close()
        model.save(model_dir / "image_genre_classifier.keras")
        (model_dir / "image_label_classes.json").write_text(json.dumps(labels, indent=2))
        (model_dir / "image_config.json").write_text(
            json.dumps({"image_size": list(image_size)}, indent=2)
        )

    return {
        "seed": seed,
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
    }


def main() -> None:
    args = parse_args()
    train_image_classifier(
        csv_path   = args.input,
        poster_dir = args.poster_dir,
        output_dir = args.output_dir,
        model_dir  = args.model_dir,
        image_size = tuple(args.img_size),
        batch_size = args.batch_size,
        epochs     = args.epochs,
    )


if __name__ == "__main__":
    main()
