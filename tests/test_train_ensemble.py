import unittest

import numpy as np
import pandas as pd

from src.train_ensemble import (
    align_probabilities_to_labels,
    merge_comparison_row,
    normalize_weights,
    soft_vote_probabilities,
)


class EnsembleHelpersTest(unittest.TestCase):
    def test_align_probabilities_to_labels_reorders_columns(self):
        probs = np.array([[0.2, 0.7, 0.1]])
        aligned = align_probabilities_to_labels(
            probs,
            source_labels=["Comedy", "Romance", "Action"],
            target_labels=["Action", "Comedy", "Romance"],
        )

        np.testing.assert_allclose(aligned, np.array([[0.1, 0.2, 0.7]]))

    def test_soft_vote_probabilities_averages_model_outputs(self):
        voted = soft_vote_probabilities(
            [
                np.array([[0.8, 0.2], [0.1, 0.9]]),
                np.array([[0.4, 0.6], [0.3, 0.7]]),
            ]
        )

        np.testing.assert_allclose(voted, np.array([[0.6, 0.4], [0.2, 0.8]]))

    def test_soft_vote_probabilities_uses_normalized_weights(self):
        voted = soft_vote_probabilities(
            [
                np.array([[0.8, 0.2], [0.1, 0.9]]),
                np.array([[0.4, 0.6], [0.3, 0.7]]),
            ],
            weights=[3, 1],
        )

        np.testing.assert_allclose(voted, np.array([[0.7, 0.3], [0.15, 0.85]]))

    def test_normalize_weights_rejects_member_count_mismatch(self):
        with self.assertRaisesRegex(ValueError, "same length"):
            normalize_weights([0.7, 0.3], member_count=3)

    def test_normalize_weights_rejects_negative_weights(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            normalize_weights([0.7, -0.3], member_count=2)

    def test_merge_comparison_row_replaces_existing_ensemble_row(self):
        comparison = pd.DataFrame(
            [
                {"model": "ensemble_soft_vote", "accuracy": 0.1, "macro_f1": 0.1},
                {"model": "logistic_regression", "accuracy": 0.7, "macro_f1": 0.6},
            ]
        )
        merged = merge_comparison_row(
            comparison,
            {"model": "ensemble_soft_vote", "accuracy": 0.8, "macro_f1": 0.75},
        )

        self.assertEqual(merged["model"].tolist(), ["ensemble_soft_vote", "logistic_regression"])
        self.assertEqual(float(merged.loc[0, "accuracy"]), 0.8)
        self.assertEqual(len(merged), 2)


if __name__ == "__main__":
    unittest.main()
