import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# Example corpus and preprocessing
corpus = [
    "Artificial intelligence is the future.",
    "The world is evolving with technology.",
    "Machine learning is a subset of AI.",
]
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(corpus)
vocab_size = len(tokenizer.word_index) + 1

sequences = []
for sentence in corpus:
    encoded = tokenizer.texts_to_sequences([sentence])[0]
    for i in range(1, len(encoded)):
        sequences.append(encoded[:i+1])

sequences = pad_sequences(sequences, maxlen=5, padding="pre")
X, y = sequences[:, :-1], sequences[:, -1]
y = to_categorical(y, num_classes=vocab_size)

# Build LSTM model
model = Sequential([
    Embedding(vocab_size, 10, input_length=4),
    LSTM(50),
    Dense(vocab_size, activation="softmax")
])
model.compile(optimizer="adam", loss="categorical_crossentropy")
model.fit(X, y, epochs=200, verbose=1)

# Generate text using LSTM
def generate_lstm_text(seed_text, next_words=10):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=4, padding="pre")
        predicted = np.argmax(model.predict(token_list), axis=-1)
        output_word = tokenizer.index_word[predicted[0]]
        seed_text += " " + output_word
    return seed_text

print(generate_lstm_text("Artificial intelligence"))
