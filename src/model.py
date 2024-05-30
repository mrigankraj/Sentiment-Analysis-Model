import tensorflow as tf
from tensorflow.keras.layers import TextVectorization, Embedding, GlobalAveragePooling1D, Dense

def build_model(vocab_size, embedding_dim, max_length):
    model = tf.keras.Sequential([
        TextVectorization(max_tokens=vocab_size, output_mode='int', output_sequence_length=max_length),
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
        GlobalAveragePooling1D(),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
