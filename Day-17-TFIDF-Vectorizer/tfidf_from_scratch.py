import numpy as np
import pandas as pd
import re
from collections import Counter

documents = [
    "Machine learning is amazing",
    "Deep learning drives AI advancements",
    "AI and machine learning are related fields",
    "Neural networks are part of deep learning"
]

def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)  # remove punctuation/numbers
    return text.split()

tokenized_docs = [preprocess(doc) for doc in documents]
print("Tokenized Docs:", tokenized_docs)

vocab = sorted(set(word for doc in tokenized_docs for word in doc))
print("\nVocabulary:", vocab)

def compute_tf(doc):
    word_counts = Counter(doc)
    total_words = len(doc)
    return {word: word_counts[word] / total_words for word in vocab}

tf_scores = [compute_tf(doc) for doc in tokenized_docs]
print("\nTF Scores (First Doc):", tf_scores[0])

def compute_idf(tokenized_docs):
    N = len(tokenized_docs)
    idf_scores = {}
    for word in vocab:
        containing_docs = sum(1 for doc in tokenized_docs if word in doc)
        idf_scores[word] = np.log((N + 1) / (containing_docs + 1)) + 1  # smooth
    return idf_scores

idf_scores = compute_idf(tokenized_docs)
print("\nIDF Scores:", idf_scores)

def compute_tfidf(tf_scores, idf_scores):
    return {word: tf_scores[word] * idf_scores[word] for word in vocab}

tfidf_matrix = [compute_tfidf(tf, idf_scores) for tf in tf_scores]
df_tfidf = pd.DataFrame(tfidf_matrix)
print("\nTF-IDF Matrix:\n", df_tfidf)
