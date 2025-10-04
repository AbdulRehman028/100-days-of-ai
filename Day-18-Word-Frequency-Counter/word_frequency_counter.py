import re
from collections import Counter

# ------------------------
# Step 1: Load Text
# ------------------------
with open("sample_text.txt", "r", encoding="utf-8") as f:
    text = f.read()

print("Original Text:\n", text[:200], "...")  # show first 200 chars

# ------------------------
# Step 2: Preprocessing
# ------------------------
def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)  # keep only letters
    return text.split()

tokens = preprocess(text)
print("\nTokens (first 20):", tokens[:20])

# ------------------------
# Step 3: Word Frequency
# ------------------------
word_counts = Counter(tokens)

# Top 10 words
print("\nTop 10 Words:")
for word, freq in word_counts.most_common(10):
    print(f"{word}: {freq}")

# ------------------------
# Step 4: Save Results
# ------------------------
with open("word_frequencies.txt", "w", encoding="utf-8") as f:
    for word, freq in word_counts.most_common():
        f.write(f"{word}: {freq}\n")

print("\n✅ Word frequencies saved to word_frequencies.txt")
