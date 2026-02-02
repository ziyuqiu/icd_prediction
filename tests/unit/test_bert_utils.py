"""
Unit tests for BERT/utils.py

Tests the BERT-specific utility functions including embedding extraction,
word frequency computation, and sequence length analysis.
"""
import pytest
import sys
import os
from unittest.mock import patch, MagicMock
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from BERT.utils import (
    trunc,
    compute_word_freq,
    remove_meanless_words,
    to_remove_words,
)


class TestTruncFunction:
    """Tests for the trunc function that truncates sequences to 512 tokens."""

    def test_trunc_shorter_than_limit(self):
        input_str = str(list(range(100)))
        result = trunc(input_str)

        assert len(result) == 100
        assert result == list(range(100))

    def test_trunc_equal_to_limit(self):
        input_str = str(list(range(512)))
        result = trunc(input_str)

        assert len(result) == 512

    def test_trunc_longer_than_limit(self):
        input_str = str(list(range(1000)))
        result = trunc(input_str)

        assert len(result) == 512
        assert result == list(range(512))

    def test_trunc_empty_list(self):
        input_str = str([])
        result = trunc(input_str)

        assert result == []

    def test_trunc_preserves_order(self):
        input_str = str([5, 4, 3, 2, 1])
        result = trunc(input_str)

        assert result == [5, 4, 3, 2, 1]


class TestComputeWordFreq:
    """Tests for the compute_word_freq function."""

    def test_compute_word_freq_basic(self):
        # Reset global word_freq
        import BERT.utils as bert_utils
        bert_utils.word_freq = {}

        # Process some text
        compute_word_freq("['hello', 'world', 'hello']")

        assert bert_utils.word_freq["hello"] == 2
        assert bert_utils.word_freq["world"] == 1

    def test_compute_word_freq_accumulates(self):
        import BERT.utils as bert_utils
        bert_utils.word_freq = {}

        compute_word_freq("['test', 'word']")
        compute_word_freq("['test', 'another']")

        assert bert_utils.word_freq["test"] == 2
        assert bert_utils.word_freq["word"] == 1
        assert bert_utils.word_freq["another"] == 1

    def test_compute_word_freq_empty_list(self):
        import BERT.utils as bert_utils
        bert_utils.word_freq = {}

        compute_word_freq("[]")

        assert bert_utils.word_freq == {}


class TestRemoveMeanlessWords:
    """Tests for the remove_meanless_words function."""

    def test_remove_meanless_words_removes_stopwords(self):
        # Test removal of words in to_remove_words set
        result = remove_meanless_words("['admission', 'chest', 'date', 'pain']")

        assert "admission" not in result
        assert "date" not in result
        assert "chest" in result
        assert "pain" in result

    def test_remove_meanless_words_keeps_medical_terms(self):
        result = remove_meanless_words("['hypertension', 'diabetes', 'chest', 'pain']")

        assert "hypertension" in result
        assert "diabetes" in result
        assert "chest" in result
        assert "pain" in result

    def test_remove_meanless_words_removes_patient(self):
        result = remove_meanless_words("['patient', 'presents', 'with', 'symptoms']")

        assert "patient" not in result
        assert "presents" in result

    def test_remove_meanless_words_removes_also(self):
        result = remove_meanless_words("['also', 'has', 'history']")

        assert "also" not in result
        assert "has" in result
        assert "history" in result

    def test_remove_meanless_words_empty_list(self):
        result = remove_meanless_words("[]")

        assert result == []

    def test_remove_meanless_words_all_stopwords(self):
        result = remove_meanless_words("['admission', 'date', 'discharge', 'service', 'patient']")

        assert len(result) == 0


class TestToRemoveWordsSet:
    """Tests for the to_remove_words constant."""

    def test_to_remove_words_is_set(self):
        assert isinstance(to_remove_words, set)

    def test_to_remove_words_contains_common_stopwords(self):
        expected_words = {'admission', 'date', 'discharge', 'service', 'also', 'x', 'patient'}

        for word in expected_words:
            assert word in to_remove_words


class TestIntegration:
    """Integration tests for BERT utilities."""

    def test_preprocessing_pipeline(self):
        """Test a typical preprocessing sequence."""
        import BERT.utils as bert_utils
        bert_utils.word_freq = {}

        # Simulate processing multiple documents
        docs = [
            "['admission', 'patient', 'chest', 'pain', 'history']",
            "['date', 'discharge', 'hypertension', 'diabetes']",
            "['chest', 'pain', 'shortness', 'breath', 'patient']",
        ]

        # Remove meaningless words and compute frequency
        cleaned_docs = []
        for doc in docs:
            cleaned = remove_meanless_words(doc)
            cleaned_docs.append(cleaned)
            compute_word_freq(str(cleaned))

        # Verify meaningful medical terms are preserved
        assert bert_utils.word_freq.get("chest", 0) == 2
        assert bert_utils.word_freq.get("pain", 0) == 2
        assert bert_utils.word_freq.get("hypertension", 0) == 1

        # Verify stopwords were removed (not counted)
        assert bert_utils.word_freq.get("patient", 0) == 0
        assert bert_utils.word_freq.get("admission", 0) == 0
