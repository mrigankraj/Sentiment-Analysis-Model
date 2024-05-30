import os
from src.data_loader import load_data, preprocess_data
from src.model import build_model
from src.train import train_model
from src.evaluate import evaluate_model

# Set file path
file_path = 'data/sentiment_data.csv'

# Load and preprocess data
data = load_data(file_path)
train_dataset, test_dataset = preprocess_data(data)

# Build and train the model
model = build_model(vocab_size=10000, embedding_dim=16, max_length=100)
train_model(model, train_dataset)

# Evaluate the model
results = evaluate_model(model, test_dataset)
print(f'Test Loss: {results[0]}, Test Accuracy: {results[1]}')
