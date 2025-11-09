import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Step 1: Load text
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read().lower()

# Step 2: Tokenize text into characters
chars = sorted(list(set(text)))
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for i, c in enumerate(chars)}

# Step 3: Prepare sequences
seq_length = 40
sequences = []
next_chars = []

for i in range(0, len(text) - seq_length):
    sequences.append(text[i: i + seq_length])
    next_chars.append(text[i + seq_length])

print("Total sequences:", len(sequences))

# Step 4: Vectorize sequences
X = np.zeros((len(sequences), seq_length, len(chars)), dtype=np.bool_)
y = np.zeros((len(sequences), len(chars)), dtype=np.bool_)

for i, seq in enumerate(sequences):
    for t, char in enumerate(seq):
        X[i, t, char_to_idx[char]] = 1
    y[i, char_to_idx[next_chars[i]]] = 1

# Step 5: Build model
model = Sequential([
    LSTM(128, input_shape=(seq_length, len(chars))),
    Dense(len(chars), activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy")

# Step 6: Train model
model.fit(X, y, batch_size=128, epochs=10)

# Step 7: Generate text
def generate_text(seed, length=200):
    generated = seed
    for _ in range(length):
        x_pred = np.zeros((1, seq_length, len(chars)))
        for t, char in enumerate(seed):
            if char in char_to_idx:
                x_pred[0, t, char_to_idx[char]] = 1
        preds = model.predict(x_pred, verbose=0)[0]
        next_char = idx_to_char[np.argmax(preds)]
        generated += next_char
        seed = generated[-seq_length:]
    return generated

# Step 8: Try generating text
seed_text = "once upon a time there was a brave hero"
print("\n🧠 Generated text:\n")
print(generate_text(seed_text.lower(), length=300))
