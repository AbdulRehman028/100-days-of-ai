import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Load Dataset
data = pd.read_csv("sample_data.csv")
print("Data Loaded Successfully:\n")
print(data.head())

# Step 2: Basic Line Plot
plt.figure(figsize=(8, 5))
plt.plot(data["Month"], data["Sales"], marker='o', label="Sales")
plt.plot(data["Month"], data["Expenses"], marker='o', label="Expenses")
plt.title("Monthly Sales vs Expenses")
plt.xlabel("Month")
plt.ylabel("Amount (USD)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("sales_vs_expenses.png")
plt.show()

# Step 3: Bar Chart
plt.figure(figsize=(8, 5))
plt.bar(data["Month"], data["Profit"], color='green')
plt.title("Monthly Profit")
plt.xlabel("Month")
plt.ylabel("Profit (USD)")
plt.tight_layout()
plt.savefig("monthly_profit.png")
plt.show()

# Step 4: Pie Chart
plt.figure(figsize=(6, 6))
plt.pie(data["Profit"], labels=data["Month"], autopct="%1.1f%%", startangle=90)
plt.title("Profit Distribution by Month")
plt.savefig("profit_pie_chart.png")
plt.show()

print("\n✅ Charts generated successfully and saved as PNG files.")