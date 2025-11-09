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
print("\n🏗️ Building LSTM model...")
model = Sequential([
    LSTM(128, input_shape=(seq_length, len(chars)), return_sequences=False),
    Dense(len(chars), activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy'])

print("📊 Model Summary:")
model.summary()

# Step 6: Train model
print("\n🚀 Training model...")
history = model.fit(X, y, batch_size=128, epochs=20, validation_split=0.1, verbose=1)

print("\n✅ Training complete!")
print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")

# Step 7: Generate text with temperature sampling
def sample_with_temperature(preds, temperature=1.0):
    """Sample from probability distribution with temperature"""
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds + 1e-10) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate_text(seed, length=200, temperature=0.5):
    """
    Generate text using the trained model
    
    Args:
        seed: Starting text
        length: Number of characters to generate
        temperature: Controls randomness (lower = more conservative, higher = more random)
                    0.2 - Very conservative, repetitive
                    0.5 - Balanced (recommended)
                    1.0 - More random
                    1.5 - Very random, creative
    """
    generated = seed
    for _ in range(length):
        x_pred = np.zeros((1, seq_length, len(chars)))
        for t, char in enumerate(seed):
            if char in char_to_idx:
                x_pred[0, t, char_to_idx[char]] = 1
        
        preds = model.predict(x_pred, verbose=0)[0]
        next_idx = sample_with_temperature(preds, temperature)
        next_char = idx_to_char[next_idx]
        
        generated += next_char
        seed = generated[-seq_length:]
    return generated

# Step 8: Try generating text with different temperatures
seed_text = "once upon a time there was a brave hero"

print("\n" + "="*80)
print("🧠 TEXT GENERATION EXAMPLES")
print("="*80)

# Conservative generation (temperature = 0.5)
print("\n📝 Conservative Generation (temperature = 0.5):")
print("-" * 80)
print(generate_text(seed_text.lower(), length=300, temperature=0.5))

# Balanced generation (temperature = 0.7)
print("\n\n📝 Balanced Generation (temperature = 0.7):")
print("-" * 80)
print(generate_text(seed_text.lower(), length=300, temperature=0.7))

# Creative generation (temperature = 1.0)
print("\n\n📝 Creative Generation (temperature = 1.0):")
print("-" * 80)
print(generate_text(seed_text.lower(), length=300, temperature=1.0))

print("\n" + "="*80)
print("✅ Generation complete!")
print("="*80)