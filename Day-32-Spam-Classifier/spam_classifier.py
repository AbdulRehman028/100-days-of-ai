import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load dataset
data = pd.read_csv("spam.csv")

# Step 2: Clean column names
data.columns = ["label", "message"]

# Step 3: Encode labels
data["label_num"] = data["label"].map({"ham": 0, "spam": 1})

# Step 4: Split data
X_train, X_test, y_train, y_test = train_test_split(
    data["message"], data["label_num"], test_size=0.2, random_state=42
)

# Step 5: Vectorize text
vectorizer = CountVectorizer(stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Step 6: Train model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Step 7: Evaluate
y_pred = model.predict(X_test_vec)

print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

# Step 8: Confusion matrix visualization
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Spam Classifier Confusion Matrix")
plt.show()

# Step 9: Try a sample message
sample = ["Congratulations! You have won free tickets to Maldives!"]
sample_vec = vectorizer.transform(sample)
prediction = model.predict(sample_vec)
print("\n💬 Sample Prediction:", "Spam" if prediction[0] else "Ham")
