import pytest
from src.data_loader import load_data, preprocess_data

def test_load_data():
    data = load_data('data/sentiment_data.csv')
    assert not data.empty

def test_preprocess_data():
    data = load_data('data/sentiment_data.csv')
    train_dataset, test_dataset = preprocess_data(data)
    assert len(train_dataset) > 0
    assert len(test_dataset) > 0
