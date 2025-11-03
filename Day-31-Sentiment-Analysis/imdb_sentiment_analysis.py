import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.optimizers import Adam

# 1. Load and preprocess IMDB dataset

vocab_size = 10000  # Only use the top 10k words
max_len = 200       # Each review is padded to 200 words

print("📦 Loading IMDB dataset...")
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)
print(f"Training samples: {len(x_train)}, Test samples: {len(x_test)}")

x_train = pad_sequences(x_train, maxlen=max_len)
x_test = pad_sequences(x_test, maxlen=max_len)

# ===============================
# 2. Build LSTM model
# ===============================
model = Sequential([
    Embedding(vocab_size, 128, input_length=max_len),
    LSTM(64, dropout=0.2, recurrent_dropout=0.2),
    Dense(1, activation='sigmoid')
])

model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(learning_rate=0.001),
    metrics=['accuracy']
)

model.summary()

# ===============================
# 3. Train the model
# ===============================
history = model.fit(
    x_train, y_train,
    epochs=4,
    batch_size=128,
    validation_split=0.2,
    verbose=2
)

# ===============================
# 4. Evaluate model
# ===============================
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"\n✅ Test Accuracy: {test_acc:.4f}")

# ===============================
# 5. Plot training performance
# ===============================
plt.figure(figsize=(8,4))
plt.plot(history.history["accuracy"], label="Train Acc")
plt.plot(history.history["val_accuracy"], label="Val Acc")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.show()

# 6. Try predictions on custom text
# ===============================
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb

# Retrieve the word index mapping used by IMDB
word_index = imdb.get_word_index()
reverse_word_index = {value + 3: key for key, value in word_index.items()}
reverse_word_index[0] = "<PAD>"
reverse_word_index[1] = "<START>"
reverse_word_index[2] = "<UNK>"
reverse_word_index[3] = "<UNUSED>"

def encode_review(text):
    tokens = text.lower().split()
    encoded = [1]  # Start token
    for word in tokens:
        if word in word_index and word_index[word] < vocab_size:
            encoded.append(word_index[word] + 3)
        else:
            encoded.append(2)  # Unknown word
    return pad_sequences([encoded], maxlen=max_len)

# Test custom reviews
samples = [
    "I absolutely loved this movie, it was fantastic!",
    "The plot was terrible and the acting was worse."
]

for s in samples:
    seq = encode_review(s)
    pred = model.predict(seq, verbose=0)[0][0]
    sentiment = "Positive" if pred > 0.5 else "Negative"
    print(f"\n💬 Review: {s}")
    print(f"🧠 Prediction: {sentiment} ({pred:.2f})")