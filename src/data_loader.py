import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    data = data.dropna()
    le = LabelEncoder()
    data['sentiment'] = le.fit_transform(data['sentiment'])
    X_train, X_test, y_train, y_test = train_test_split(
        data['text'], data['sentiment'], test_size=0.2, random_state=42)
    
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train.values, y_train.values))
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test.values, y_test.values))
    
    return train_dataset, test_dataset
