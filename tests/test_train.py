import pytest
from src.train import train_model
from src.model import build_model
from src.data_loader import load_data, preprocess_data

def test_train_model():
    data = load_data('data/sentiment_data.csv')
    train_dataset, _ = preprocess_data(data)
    model = build_model(vocab_size=10000, embedding_dim=16, max_length=100)
    history = train_model(model, train_dataset, epochs=1)
    assert len(history.history['loss']) > 0
