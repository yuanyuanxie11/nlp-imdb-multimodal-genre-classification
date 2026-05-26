"""Bonus poster-image classifier (MobileNetV2 + transfer learning).

Now shares the project's standard ``stratified_split`` so train/val/test
ratios match the text models, and supports an optional ``--seeds`` sweep
for between-seed stability reporting (same protocol as ``train_lstm``).

⚠ Full K-Fold is *not* added — MobileNetV2 fine-tuning is too slow to
   run 5× on a class-project budget. The seed sweep gives most of the
   benefit (between-run variance) at a small fraction of the cost.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from .runtime import prepare_runtime, set_global_seed

prepare_runtime()

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

from .config import SplitConfig, ensure_dir
from .data_processing import stratified_split
from .evaluate import compute_metrics, save_confusion_matrix, save_metrics


def build_image_model(image_size: tuple[int, int], num_classes: int) -> tf.keras.Model:
    base = MobileNetV2(
        weights="imagenet", include_top=False,
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
    parser.add_argument(
        "--seeds",
        type=int, nargs="*", default=None,
        help="Run each seed and report between-seed std. First seed saves the canonical artifacts.",
    )
    return parser.parse_args()


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

    model = build_image_model(image_size, len(encoder.classes_))
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

    return {
        "seed": seed,
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
    }


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

    image_size = tuple(args.img_size)
    seeds: Iterable[int] = args.seeds if args.seeds else [42]
    rows = []
    for i, seed in enumerate(seeds):
        print(f"\n=== Image classifier run seed={seed} (save={i == 0}) ===")
        rows.append(train_one_run(
            df, args.image_column, args.label_column, image_size,
            args.epochs, seed, Path(output_dir), model_dir,
            save_artifacts=(i == 0),
        ))

    if args.seeds:
        stab = pd.DataFrame(rows)
        stab.to_csv(Path(output_dir) / "tables" / "image_seed_stability.csv", index=False)
        print("\n── Image classifier seed stability ──")
        print(stab.to_string(index=False))
        print(f"accuracy  : {stab['accuracy'].mean():.4f} ± {stab['accuracy'].std():.4f}")
        print(f"macro_f1  : {stab['macro_f1'].mean():.4f} ± {stab['macro_f1'].std():.4f}")

    print(json.dumps({k: v for k, v in rows[0].items() if k != "seed"}, indent=2))


if __name__ == "__main__":
    main()
