import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Step 1: Load Data
  
data = pd.read_csv("data.csv")
print("✅ Data Loaded Successfully:\n", data.head())
  
# Step 2: Encode Categorical Columns
  
le = LabelEncoder()
for column in data.columns:
    data[column] = le.fit_transform(data[column])

X = data.drop("buys_computer", axis=1)
y = data["buys_computer"]
  
# Step 3: Split Dataset
  
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Step 4: Train Decision Tree
  
clf = DecisionTreeClassifier(criterion="entropy", random_state=42)
clf.fit(X_train, y_train)

# Step 5: Evaluate Model
  
y_pred = clf.predict(X_test)
print("\n🎯 Accuracy:", accuracy_score(y_test, y_pred))
print("\n📄 Classification Report:\n", classification_report(y_test, y_pred))

# Step 6: Visualize the Tree
  
plt.figure(figsize=(10, 6))
plot_tree(
    clf,
    feature_names=X.columns,
    class_names=["No", "Yes"],
    filled=True,
    rounded=True,
    fontsize=10
)
plt.title("Decision Tree Visualization")
plt.show()

# Step 7: Test Prediction
sample = [[1, 2, 1, 0]]  # Example input
prediction = clf.predict(sample)
print(f"\n🧠 Prediction for {sample}: {prediction[0]} (1=Yes, 0=No)")