"""
Shared pytest fixtures for icd_prediction tests.
"""
import pytest
import torch
import numpy as np


@pytest.fixture
def sample_clinical_note():
    """Sample clinical note for testing preprocessing."""
    return """Admission Date: 2020-01-15 Discharge Date: 2020-01-20

Service: Medicine

Chief Complaint:
Chest pain and shortness of breath.

History of Present Illness:
Patient is a 65 year old male with history of hypertension and diabetes presenting
with chest pain for the past 2 days.

Past Medical History:
-- Hypertension
-- Type 2 Diabetes
-- Hyperlipidemia

Medications on Admission:
1. Metformin 500mg twice daily
2. Lisinopril 10mg daily
3. Atorvastatin 20mg daily

Physical Exam:
Vitals - T 98.6 BP 145/90 HR 88 R 18 SpO2 96% on RA
General - Alert, oriented, no acute distress
CV - Regular rate and rhythm, no murmurs

Labs:
WBC-8.5 RBC-4.2 Hgb-12.5 Hct-38.2

Discharge Diagnosis:
Unstable angina
"""


@pytest.fixture
def sample_word2index():
    """Sample word to index mapping."""
    return {
        "<pad>": 0,
        "<unk>": 1,
        "patient": 2,
        "chest": 3,
        "pain": 4,
        "history": 5,
        "hypertension": 6,
        "diabetes": 7,
        "admission": 8,
        "discharge": 9,
    }


@pytest.fixture
def sample_code2index():
    """Sample ICD code to index mapping."""
    return {
        "410.0": 0,
        "401.9": 1,
        "250.00": 2,
        "272.4": 3,
        "786.50": 4,
    }


@pytest.fixture
def sample_labels():
    """Sample multi-hot encoded labels."""
    return np.array([
        [1, 0, 1, 0, 1],  # Document 1: codes 410.0, 250.00, 786.50
        [0, 1, 1, 1, 0],  # Document 2: codes 401.9, 250.00, 272.4
        [1, 1, 0, 0, 1],  # Document 3: codes 410.0, 401.9, 786.50
    ])


@pytest.fixture
def sample_tensors(sample_word2index):
    """Sample tensor data for model testing."""
    batch_size = 4
    seq_len = 32
    num_classes = 5

    # Random input IDs
    X = torch.randint(0, len(sample_word2index), (batch_size, seq_len))

    # Random multi-hot labels
    y = torch.zeros(batch_size, num_classes)
    for i in range(batch_size):
        # Each sample has 1-3 positive labels
        num_positive = torch.randint(1, 4, (1,)).item()
        positive_indices = torch.randperm(num_classes)[:num_positive]
        y[i, positive_indices] = 1

    return X, y


@pytest.fixture
def mock_bert_output():
    """Mock BERT model output for testing."""
    batch_size = 4
    hidden_size = 768

    # Mock pooled output
    pooled_output = torch.randn(batch_size, hidden_size)

    return pooled_output
