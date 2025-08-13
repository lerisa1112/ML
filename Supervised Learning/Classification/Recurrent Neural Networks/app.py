# pip install tensorflow
import tensorflow as tf
from tensorflow.keras import layers

# 1) Load IMDB (already tokenized as integers)
vocab_size = 20000
max_len = 200
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=vocab_size)

x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_len)
x_test  = tf.keras.preprocessing.sequence.pad_sequences(x_test,  maxlen=max_len)

# 2) Build model: Embedding -> LSTM -> Dense
model = tf.keras.Sequential([
    layers.Embedding(vocab_size, 128, input_length=max_len),
    layers.LSTM(128, return_sequences=False),     # swap to GRU(128) or SimpleRNN(128) if you like
    layers.Dropout(0.3),
    layers.Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

# 3) Train & Evaluate
history = model.fit(x_train, y_train, batch_size=64, epochs=3, validation_split=0.2)
print("Test accuracy:", model.evaluate(x_test, y_test, verbose=0)[1])

# 4) Inference on a single review (using raw IDs for simplicity)
sample = x_test[0:1]
print("Predicted prob (positive):", float(model.predict(sample, verbose=0)))
