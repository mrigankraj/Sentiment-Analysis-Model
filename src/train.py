def train_model(model, train_dataset, epochs=10):
    train_dataset = train_dataset.batch(32).prefetch(tf.data.experimental.AUTOTUNE)
    history = model.fit(train_dataset, epochs=epochs)
    return history
