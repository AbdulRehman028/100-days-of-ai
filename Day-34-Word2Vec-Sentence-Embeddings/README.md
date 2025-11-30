import numpy as np
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# Load Dataset
with open("dataset.txt", "r", encoding="utf-8") as f:
    corpus = f.read().lower().split("\n")

sentences = [line.split() for line in corpus]

# Train Word2Vec Model
model = Word2Vec(
    sentences=sentences,
    vector_size=50,
    window=5,
    min_count=1,
    workers=4,
    epochs=50
)

print("\nTraining complete.")

## Word Embedding Example

print("\nVector for 'language':")
print(model.wv['language'])

# Word Similarity
print("\nSimilarity between 'machine' and 'learning':")
print(model.wv.similarity('machine', 'learning'))

# Sentence Embedding (avg of words)

def sentence_embedding(sentence):
    words = sentence.lower().split()
    vecs = [model.wv[word] for word in words if word in model.wv]
    if len(vecs) == 0:
        return np.zeros(model.vector_size)
    return np.mean(vecs, axis=1)

# Example Sentences
s1 = "machine learning enables computers"
s2 = "deep learning is a branch of ai"
s3 = "generative models create text"

# Get embeddings
e1 = sentence_embedding(s1)
e2 = sentence_embedding(s2)
e3 = sentence_embedding(s3)

# Cosine similarity between sentences
print("\nSentence Similarity (s1 vs s2):", cosine_similarity([e1], [e2])[0][0])
print("Sentence Similarity (s1 vs s3):", cosine_similarity([e1], [e3])[0][0])

# Visualization with t-SNE

words = list(model.wv.index_to_key)
X = np.array([model.wv[word] for word in words])

tsne = TSNE(n_components=2, perplexity=5, random_state=42)
X_2d = tsne.fit_transform(X)

plt.figure(figsize=(10, 7))
plt.scatter(X_2d[:, 0], X_2d[:, 1])

for i, word in enumerate(words):
    plt.annotate(word, xy=(X_2d[i, 0], X_2d[i, 1]))

plt.title("Word2Vec Embeddings Visualization (t-SNE)")
plt.show()
