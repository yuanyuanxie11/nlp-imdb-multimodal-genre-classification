from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import LSTMConfig
from src.train_lstm import build_model
from tensorflow.keras.preprocessing.text import Tokenizer
from src.pipeline import build_image_manifest, infer_columns, select_dataset_file, select_poster_dir
from src.explain import lstm_gradient_saliency


class PipelineTest(unittest.TestCase):
    def test_select_dataset_file_prefers_largest_csv_with_required_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            small = tmp_path / "notes.csv"
            small.write_text("name,value\nx,1\n")
            dataset = tmp_path / "movies.csv"
            pd.DataFrame(
                {
                    "plot": ["A detective hunts a killer.", "Two friends fall in love."],
                    "genre": ["Action", "Romance"],
                }
            ).to_csv(dataset, index=False)

            self.assertEqual(select_dataset_file(tmp_path), dataset)

    def test_infer_columns_detects_plot_and_genre_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "imdb.csv"
            pd.DataFrame(
                {
                    "Plot Summary": ["A haunted house waits.", "A comedian takes the stage."],
                    "Genre": ["Horror", "Comedy"],
                }
            ).to_csv(dataset, index=False)

            self.assertEqual(infer_columns(dataset), ("Plot Summary", "Genre"))

    def test_lstm_saliency_handles_saved_sequential_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            tokenizer = Tokenizer(num_words=20, oov_token="<OOV>")
            tokenizer.fit_on_texts(["hero mission", "haunted night"])
            config = LSTMConfig(max_words=20, max_len=6, embedding_dim=4, lstm_units=4, dense_units=4)
            model = build_model(num_classes=2, config=config)
            model(np.zeros((1, config.max_len), dtype=np.int32))
            model.save(model_dir / "lstm_text_classifier.keras")
            (model_dir / "lstm_tokenizer.json").write_text(tokenizer.to_json())
            (model_dir / "lstm_label_classes.json").write_text(json.dumps(["Action", "Horror"]))
            (model_dir / "lstm_config.json").write_text(json.dumps(config.__dict__))

            features = lstm_gradient_saliency(
                model_dir,
                texts=["hero mission", "haunted night"],
                labels=["Action", "Horror"],
                top_n=3,
            )

            self.assertIsNotNone(features)
            self.assertFalse(features.empty)

    def test_build_image_manifest_from_genre_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            poster_dir = Path(tmp) / "posters"
            action = poster_dir / "Action"
            comedy = poster_dir / "Comedy"
            action.mkdir(parents=True)
            comedy.mkdir(parents=True)
            (action / "a.jpg").write_bytes(b"fake")
            (comedy / "c.png").write_bytes(b"fake")

            self.assertEqual(select_poster_dir(tmp), poster_dir)
            manifest = build_image_manifest(poster_dir, Path(tmp) / "manifest.csv")
            manifest_df = pd.read_csv(manifest)

            self.assertEqual(set(manifest_df["genre"]), {"Action", "Comedy"})
            self.assertEqual(len(manifest_df), 2)


if __name__ == "__main__":
    unittest.main()
