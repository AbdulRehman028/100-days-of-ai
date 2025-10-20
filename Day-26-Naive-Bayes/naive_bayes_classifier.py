import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------------
# Step 1: Load Dataset
# -----------------------------
data = pd.read_csv("emails.csv")
print("✅ Data Loaded Successfully:\n")
print(data.head())

# -----------------------------
# Step 2: Prepare Features
# -----------------------------
X = data["text"]
y = data["label"]

# Convert text to numerical form
vectorizer = CountVectorizer()
X_vec = vectorizer.fit_transform(X)

# -----------------------------
# Step 3: Split Data
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.3, random_state=42
)

# -----------------------------
# Step 4: Train Naive Bayes Model
# -----------------------------
model = MultinomialNB()
model.fit(X_train, y_train)

# -----------------------------
# Step 5: Evaluate Model
# -----------------------------
y_pred = model.predict(X_test)
print("\n🎯 Accuracy:", accuracy_score(y_test, y_pred))
print("\n📄 Classification Report:\n", classification_report(y_test, y_pred))

# -----------------------------
# Step 6: Confusion Matrix
# -----------------------------
cm = confusion_matrix(y_test, y_pred, labels=["ham", "spam"])
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"])
plt.title("Confusion Matrix - Naive Bayes Classifier")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# -----------------------------
# Step 7: Test Predictions
# -----------------------------
samples = [
    "Claim your free prize now!",
    "Let's go for dinner tomorrow.",
    "Earn money easily from home"
]

samples_vec = vectorizer.transform(samples)
predictions = model.predict(samples_vec)

for text, label in zip(samples, predictions):
    print(f"📧 '{text}' → Predicted: {label}")
