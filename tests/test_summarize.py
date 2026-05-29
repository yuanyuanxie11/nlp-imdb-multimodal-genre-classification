from __future__ import annotations

import unittest

from src.app_helpers import summarize_text


class SummarizeHelpersTest(unittest.TestCase):
    def test_summarize_text_respects_target_sentence_count(self):
        text = (
            "A detective returns home. "
            "The city is full of secrets. "
            "His neighbor is suspicious. "
            "A missing child changes the case."
        )

        summary = summarize_text(text, max_sentences=2)

        self.assertLessEqual(summary.count("."), 2)
        self.assertNotEqual(summary, text)


if __name__ == "__main__":
    unittest.main()
