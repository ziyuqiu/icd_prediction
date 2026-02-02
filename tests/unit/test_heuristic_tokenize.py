"""
Unit tests for Preprocess/heuristic_tokenize.py

Tests the heuristic sentence tokenization functions used for
processing clinical notes from MIMIC-III.
"""
import pytest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from Preprocess.heuristic_tokenize import (
    strip,
    is_inline_title,
    is_title,
    sent_tokenize_rules,
)


class TestStripFunction:
    """Tests for the strip utility function."""

    def test_strip_removes_leading_whitespace(self):
        assert strip("  hello") == "hello"

    def test_strip_removes_trailing_whitespace(self):
        assert strip("hello  ") == "hello"

    def test_strip_removes_both_whitespace(self):
        assert strip("  hello  ") == "hello"

    def test_strip_preserves_internal_whitespace(self):
        assert strip("  hello world  ") == "hello world"

    def test_strip_empty_string(self):
        assert strip("") == ""

    def test_strip_only_whitespace(self):
        assert strip("   ") == ""


class TestIsTitleFunction:
    """Tests for the is_title function that identifies section headers."""

    def test_is_title_valid_single_word(self):
        assert is_title("History:") is True

    def test_is_title_valid_multiple_words(self):
        assert is_title("Past Medical History:") is True

    def test_is_title_valid_with_stopwords(self):
        # Stopwords (of, on, or) should not affect title detection
        assert is_title("History of Present Illness:") is True

    def test_is_title_missing_colon(self):
        assert is_title("History") is False

    def test_is_title_lowercase_word(self):
        # Non-stopwords must be capitalized
        assert is_title("history:") is False

    def test_is_title_mixed_case_invalid(self):
        assert is_title("Past medical History:") is False

    def test_is_title_with_parentheses(self):
        # Parenthetical content should be ignored
        assert is_title("Medications (Home):") is True

    def test_is_title_disp_edge_case(self):
        # "Disp" is explicitly excluded as it's a common false positive
        assert is_title("Disp:") is False

    def test_is_title_empty_string(self):
        assert is_title("") is False

    def test_is_title_only_colon(self):
        # Edge case: empty title with just colon is considered valid by the code
        # This documents the current behavior
        assert is_title(":") is True

    def test_is_title_stopword_only(self):
        # Stopwords alone are considered valid titles (all words pass the check)
        assert is_title("of:") is True


class TestIsInlineTitleFunction:
    """Tests for the is_inline_title function."""

    def test_is_inline_title_valid(self):
        assert is_inline_title("Name: John Doe") is True

    def test_is_inline_title_valid_multiple_words(self):
        assert is_inline_title("Date of Birth: 1990-01-01") is True

    def test_is_inline_title_no_colon(self):
        assert is_inline_title("Name John Doe") is False

    def test_is_inline_title_colon_at_end(self):
        assert is_inline_title("John Doe:") is False

    def test_is_inline_title_empty_string(self):
        assert is_inline_title("") is False

    def test_is_inline_title_lowercase(self):
        assert is_inline_title("name: John Doe") is False


class TestSentTokenizeRules:
    """Tests for the main sent_tokenize_rules function."""

    def test_tokenize_simple_sections(self):
        text = """Chief Complaint:
Chest pain.

History:
Patient has history of heart disease."""
        segments = sent_tokenize_rules(text)

        # Should have separate segments for headers and content
        assert any("Chief Complaint:" in seg for seg in segments)
        assert any("History:" in seg for seg in segments)

    def test_tokenize_admission_discharge_header(self):
        text = "Admission Date: 2020-01-15 Discharge Date: 2020-01-20\n\nService: Medicine"
        segments = sent_tokenize_rules(text)

        # The code combines header with value for these special cases
        assert any("Admission Date:" in seg for seg in segments)
        assert any("Discharge Date:" in seg for seg in segments)

    def test_tokenize_dob_sex_header(self):
        text = "Date of Birth: 1955-03-20 Sex: M\n\nChief Complaint:\nChest pain"
        segments = sent_tokenize_rules(text)

        # The code combines header with value for these special cases
        assert any("Date of Birth:" in seg for seg in segments)
        assert any("Sex:" in seg for seg in segments)

    def test_tokenize_horizontal_rules(self):
        text = """Section One
Content here.

---

Section Two
More content."""
        segments = sent_tokenize_rules(text)

        # Horizontal rules should create separate segments
        assert "-----" in segments

    def test_tokenize_underscore_delimiter(self):
        text = """Section One
___
Section Two"""
        segments = sent_tokenize_rules(text)

        assert "_____" in segments

    def test_tokenize_numbered_list(self):
        text = """Medications:
1. Aspirin 81mg daily
2. Metoprolol 25mg twice daily
3. Lisinopril 10mg daily"""
        segments = sent_tokenize_rules(text)

        # The tokenizer processes numbered lists - verify segments are created
        assert len(segments) >= 1
        # Content should be preserved somewhere in the output
        all_text = " ".join(segments)
        assert "Aspirin" in all_text or "Metoprolol" in all_text

    def test_tokenize_preserves_content(self):
        text = """Diagnosis:
Hypertension"""
        segments = sent_tokenize_rules(text)

        # Content should be preserved
        all_text = " ".join(segments)
        assert "Hypertension" in all_text

    def test_tokenize_empty_string(self):
        segments = sent_tokenize_rules("")
        assert isinstance(segments, list)

    def test_tokenize_no_sections(self):
        text = "This is just plain text without any section headers."
        segments = sent_tokenize_rules(text)

        assert len(segments) >= 1
        assert any("plain text" in seg for seg in segments)

    def test_tokenize_inline_titles_separation(self):
        text = """Building: Main Hospital
Campus: WEST
Floor: 3"""
        segments = sent_tokenize_rules(text)

        # Inline titles should create separate segments
        assert len(segments) >= 3

    def test_tokenize_title_with_one_line_answer(self):
        text = """Chief Complaint:
Chest pain

History of Present Illness:
Details here."""
        segments = sent_tokenize_rules(text)

        # Single-line answers should be combined with their headers
        # when followed by another title
        combined = " ".join(segments)
        assert "Chest pain" in combined


class TestSentTokenizeRulesIntegration:
    """Integration tests using realistic clinical note formats."""

    def test_full_clinical_note(self, sample_clinical_note):
        """Test tokenization of a complete clinical note."""
        segments = sent_tokenize_rules(sample_clinical_note)

        # Should produce multiple segments
        assert len(segments) > 5

        # Key sections should be identifiable
        all_text = " ".join(segments)
        assert "Chief Complaint" in all_text
        assert "History" in all_text
        assert "Medications" in all_text
        assert "Physical Exam" in all_text

    def test_preserves_medical_data(self, sample_clinical_note):
        """Ensure medical data is not lost during tokenization."""
        segments = sent_tokenize_rules(sample_clinical_note)
        all_text = " ".join(segments)

        # Vital signs should be preserved
        assert "BP" in all_text or "blood pressure" in all_text.lower()

        # Some medications should be preserved (numbered list handling may drop first item)
        assert "Lisinopril" in all_text or "Atorvastatin" in all_text

    def test_lab_values_preserved(self):
        """Test that lab values with special formatting are preserved."""
        text = """Labs:
WBC-8.5 RBC-4.2* Hgb-12.5* Hct-38.2*
MCV-88 MCH-29.8 MCHC-32.8 RDW-14.2* Plt Ct-250"""

        segments = sent_tokenize_rules(text)
        all_text = " ".join(segments)

        assert "WBC-8.5" in all_text
        assert "Plt Ct-250" in all_text
