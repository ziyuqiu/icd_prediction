"""
Unit tests for LSTM/util.py

Tests the data preprocessing utilities for LSTM models including
vocabulary building, sequence encoding, and data transformation.
"""
import pytest
import sys
import os
import torch
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from LSTM.util import (
    build_vocab,
    text_to_seq,
    pad_seq,
    get_X_y_ids,
)


class TestBuildVocab:
    """Tests for the build_vocab function."""

    def test_build_vocab_basic(self):
        texts = ["hello world", "hello python", "world python"]
        vocab = build_vocab(texts, min_df=1, max_df=1.0, keep_n=100)

        # Should include padding and unknown tokens
        assert "<pad>" in vocab
        assert "<unk>" in vocab
        assert vocab["<pad>"] == 0
        assert vocab["<unk>"] == 1

    def test_build_vocab_contains_frequent_words(self):
        texts = ["hello world", "hello python", "hello java"]
        vocab = build_vocab(texts, min_df=1, max_df=1.0, keep_n=100)

        # "hello" appears 3 times, should be in vocab
        assert "hello" in vocab

    def test_build_vocab_min_df_filter(self):
        texts = ["hello world", "hello python", "rare_word"]
        vocab = build_vocab(texts, min_df=2, max_df=1.0, keep_n=100)

        # "rare_word" appears only once, should be filtered out
        assert "rare_word" not in vocab
        # "hello" appears twice, should be included
        assert "hello" in vocab

    def test_build_vocab_max_df_filter(self):
        texts = ["common word", "common rare", "common unique"]
        vocab = build_vocab(texts, min_df=1, max_df=0.5, keep_n=100)

        # "common" appears in all 3 docs (100%), should be filtered
        assert "common" not in vocab

    def test_build_vocab_keep_n_limit(self):
        texts = [f"word{i}" for i in range(100)]
        vocab = build_vocab(texts, min_df=1, max_df=1.0, keep_n=10)

        # Should only keep top 10 words plus <pad> and <unk>
        # Actually keep_n-2 words plus pad/unk = keep_n total
        assert len(vocab) <= 10

    def test_build_vocab_empty_texts(self):
        texts = []
        vocab = build_vocab(texts, min_df=1, max_df=1.0, keep_n=100)

        # Should still have pad and unk tokens
        assert "<pad>" in vocab
        assert "<unk>" in vocab

    def test_build_vocab_indices_are_unique(self):
        texts = ["hello world", "foo bar", "hello foo"]
        vocab = build_vocab(texts, min_df=1, max_df=1.0, keep_n=100)

        # All indices should be unique
        indices = list(vocab.values())
        assert len(indices) == len(set(indices))

    def test_build_vocab_indices_are_sequential(self):
        texts = ["hello world", "foo bar", "hello foo"]
        vocab = build_vocab(texts, min_df=1, max_df=1.0, keep_n=100)

        # Indices should start from 0 and be sequential
        indices = sorted(vocab.values())
        assert indices == list(range(len(indices)))


class TestTextToSeq:
    """Tests for the text_to_seq function."""

    def test_text_to_seq_basic(self, sample_word2index):
        text = "patient chest pain"
        seq = text_to_seq(text, sample_word2index)

        assert seq == [2, 3, 4]  # patient=2, chest=3, pain=4

    def test_text_to_seq_unknown_words(self, sample_word2index):
        text = "patient unknown_word chest"
        seq = text_to_seq(text, sample_word2index)

        # Unknown words should map to <unk> index (1)
        assert seq == [2, 1, 3]

    def test_text_to_seq_all_unknown(self, sample_word2index):
        text = "xyz abc def"
        seq = text_to_seq(text, sample_word2index)

        assert seq == [1, 1, 1]

    def test_text_to_seq_empty_string(self, sample_word2index):
        text = ""
        seq = text_to_seq(text, sample_word2index)

        assert seq == []

    def test_text_to_seq_single_word(self, sample_word2index):
        text = "patient"
        seq = text_to_seq(text, sample_word2index)

        assert seq == [2]


class TestPadSeq:
    """Tests for the pad_seq function."""

    def test_pad_seq_shorter_than_max(self):
        seq = [1, 2, 3]
        padded = pad_seq(seq, max_len=5)

        assert len(padded) == 5
        assert padded == [1, 2, 3, 0, 0]

    def test_pad_seq_equal_to_max(self):
        seq = [1, 2, 3, 4, 5]
        padded = pad_seq(seq, max_len=5)

        assert len(padded) == 5
        assert padded == [1, 2, 3, 4, 5]

    def test_pad_seq_longer_than_max(self):
        seq = [1, 2, 3, 4, 5, 6, 7]
        padded = pad_seq(seq, max_len=5)

        # Should truncate to max_len
        assert len(padded) == 5
        assert padded == [1, 2, 3, 4, 5]

    def test_pad_seq_empty_sequence(self):
        seq = []
        padded = pad_seq(seq, max_len=5)

        assert len(padded) == 5
        assert padded == [0, 0, 0, 0, 0]

    def test_pad_seq_max_len_zero(self):
        seq = [1, 2, 3]
        padded = pad_seq(seq, max_len=0)

        assert len(padded) == 0
        assert padded == []

    def test_pad_seq_preserves_order(self):
        seq = [5, 4, 3, 2, 1]
        padded = pad_seq(seq, max_len=7)

        assert padded[:5] == [5, 4, 3, 2, 1]


class TestGetXYIds:
    """Tests for the get_X_y_ids function."""

    def test_get_X_y_ids_basic(self, sample_word2index):
        data = [
            ("patient chest pain", np.array([1, 0, 1, 0, 0]), 1),
            ("history hypertension", np.array([0, 1, 0, 1, 0]), 2),
        ]
        X, y, ids_tensor, ids_list = get_X_y_ids(data, sample_word2index, max_seq_len=10)

        # Check shapes
        assert X.shape == (2, 10)
        assert y.shape == (2, 5)
        assert ids_tensor.shape == (2,)
        assert len(ids_list) == 2

    def test_get_X_y_ids_tensor_types(self, sample_word2index):
        data = [
            ("patient chest", np.array([1, 0, 0, 0, 0]), 1),
        ]
        X, y, ids_tensor, ids_list = get_X_y_ids(data, sample_word2index, max_seq_len=5)

        # Check tensor types
        assert X.dtype == torch.long
        assert y.dtype == torch.float
        assert ids_tensor.dtype == torch.long

    def test_get_X_y_ids_padding(self, sample_word2index):
        data = [
            ("patient", np.array([1, 0, 0, 0, 0]), 1),
        ]
        X, y, ids_tensor, ids_list = get_X_y_ids(data, sample_word2index, max_seq_len=5)

        # Check that sequence is padded
        assert X[0, 0].item() == sample_word2index["patient"]
        assert X[0, 1].item() == 0  # Padding

    def test_get_X_y_ids_truncation(self, sample_word2index):
        # Long text should be truncated
        data = [
            ("patient chest pain history hypertension diabetes admission discharge",
             np.array([1, 1, 1, 1, 1]), 1),
        ]
        X, y, ids_tensor, ids_list = get_X_y_ids(data, sample_word2index, max_seq_len=3)

        # Should only keep first 3 tokens
        assert X.shape[1] == 3

    def test_get_X_y_ids_preserves_doc_ids(self, sample_word2index):
        data = [
            ("patient", np.array([1, 0, 0, 0, 0]), 100),
            ("chest", np.array([0, 1, 0, 0, 0]), 200),
            ("pain", np.array([0, 0, 1, 0, 0]), 300),
        ]
        X, y, ids_tensor, ids_list = get_X_y_ids(data, sample_word2index, max_seq_len=5)

        assert ids_list == [100, 200, 300]
        assert ids_tensor.tolist() == [100, 200, 300]

    def test_get_X_y_ids_empty_data(self, sample_word2index):
        """Test that empty data raises IndexError (known limitation)."""
        import pytest
        data = []
        # The current implementation doesn't handle empty data gracefully
        with pytest.raises(IndexError):
            get_X_y_ids(data, sample_word2index, max_seq_len=5)


class TestIntegration:
    """Integration tests combining multiple utility functions."""

    def test_end_to_end_preprocessing(self):
        """Test the full preprocessing pipeline."""
        # Build vocabulary from sample texts
        texts = [
            "patient has chest pain and shortness of breath",
            "history of hypertension and diabetes mellitus",
            "chest pain radiating to left arm",
            "patient with history of heart disease",
        ]
        vocab = build_vocab(texts, min_df=1, max_df=1.0, keep_n=100)

        # Convert text to sequence
        text = "patient has chest pain"
        seq = text_to_seq(text, vocab)

        # Pad sequence
        padded = pad_seq(seq, max_len=10)

        # Verify result
        assert len(padded) == 10
        assert all(isinstance(idx, int) for idx in padded)

    def test_batch_processing(self, sample_word2index):
        """Test processing multiple documents in a batch."""
        documents = [
            ("patient chest pain", np.array([1, 0, 1, 0, 0]), i)
            for i in range(10)
        ]

        X, y, ids_tensor, ids_list = get_X_y_ids(
            documents, sample_word2index, max_seq_len=20
        )

        # Verify batch dimensions
        assert X.shape[0] == 10
        assert y.shape[0] == 10
        assert len(ids_list) == 10
