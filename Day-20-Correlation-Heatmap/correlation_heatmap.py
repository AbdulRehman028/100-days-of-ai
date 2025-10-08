import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load Dataset

data = pd.read_csv("sample_data.csv")
print("✅ Data Loaded Successfully:\n")
print(data.head())

# Step 2: Compute Correlation

corr_matrix = data.corr(numeric_only=True)
print("\n📊 Correlation Matrix:\n", corr_matrix)

# Step 3: Plot Heatmap

plt.figure(figsize=(8, 6))
sns.heatmap(
    corr_matrix,
    annot=True,
    cmap="coolwarm",
    linewidths=0.5,
    fmt=".2f"
)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.show()

print("\n✅ Correlation Heatmap generated and saved as correlation_heatmap.png")