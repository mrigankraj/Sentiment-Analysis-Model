import pytest
from src.model import build_model

def test_build_model():
    model = build_model(vocab_size=10000, embedding_dim=16, max_length=100)
    assert model is not None
    assert len(model.layers) == 5
