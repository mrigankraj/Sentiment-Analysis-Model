def evaluate_model(model, test_dataset):
    test_dataset = test_dataset.batch(32).prefetch(tf.data.experimental.AUTOTUNE)
    results = model.evaluate(test_dataset)
    return results
