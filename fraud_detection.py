import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("fraud_transactions.csv")

print("FRAUD & ANOMALY DETECTION SYSTEM")
print("=" * 50)
print("Machine Learning-Based Transaction Monitoring")
print("=" * 50)

# Select features for anomaly detection
# Features used for anomaly detection
features = [
    "Amount",
    "Hour",
    "Distance_From_Home",
    "Transaction_Frequency"
]

print("\nFeatures Used:")
for feature in features:
    print("-", feature)

X = df[features]

y = df["Is_Fraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nDataset Split:")
print("Training transactions:", len(X_train))
print("Testing transactions:", len(X_test))

# Create Isolation Forest model
# Initialize Isolation Forest
model = IsolationForest(
    n_estimators=200,
    contamination=0.03,
    random_state=42
)

print("\nIsolation Forest Configuration:")
print("Number of trees:", 200)
print("Expected anomaly rate:", "3%")

# Train the model and predict anomalies
model.fit(X_train)

df["Anomaly_Prediction"] = model.predict(X)
df["Anomaly_Score"] = model.decision_function(X)

# Convert predictions:
# -1 = anomaly
#  1 = normal
df["Predicted_Fraud"] = df["Anomaly_Prediction"].apply(
    lambda x: 1 if x == -1 else 0
)

# Convert anomaly score into a 0-100 risk score
df["Risk_Score"] = (
    (1 - (df["Anomaly_Score"] - df["Anomaly_Score"].min()) /
     (df["Anomaly_Score"].max() - df["Anomaly_Score"].min()))
    * 100
).round(2)

print("\nModel trained successfully!")

print("\nPredicted Fraud Distribution:")
print(df["Predicted_Fraud"].value_counts())

print("\nSample Predictions:")
print(
    df[
        [
            "Transaction_ID",
            "Amount",
            "Hour",
            "Distance_From_Home",
            "Transaction_Frequency",
            "Predicted_Fraud",
            "Risk_Score"
        ]
    ].head(10)
)

print("\nFraud Detection Summary:")
print("Actual fraudulent transactions:", df["Is_Fraud"].sum())
print("Predicted fraudulent transactions:", df["Predicted_Fraud"].sum())

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

print("\nModel Performance")
print("=" * 40)

accuracy = accuracy_score(y_test, df.loc[y_test.index, "Predicted_Fraud"])
precision = precision_score(y_test, df.loc[y_test.index, "Predicted_Fraud"])
recall = recall_score(y_test, df.loc[y_test.index, "Predicted_Fraud"])
f1 = f1_score(y_test, df.loc[y_test.index, "Predicted_Fraud"])

print(f"Accuracy:  {accuracy:.2%}")
print(f"Precision: {precision:.2%}")
print(f"Recall:    {recall:.2%}")
print(f"F1 Score:  {f1:.2%}")

print("\nConfusion Matrix:")
print(confusion_matrix(df["Is_Fraud"], df["Predicted_Fraud"]))

# Save predicted fraudulent transactions
fraud_transactions = df[df["Predicted_Fraud"] == 1]

fraud_transactions.to_csv(
    "detected_fraud_transactions.csv",
    index=False
)

print("\nDetected fraud transactions saved successfully!")
print("Output file: detected_fraud_transactions.csv")

import matplotlib.pyplot as plt

# Fraud vs Normal Transaction Count
plt.figure(figsize=(7, 5))

df["Is_Fraud"].value_counts().sort_index().plot(
    kind="bar",
    xlabel="Transaction Type",
    ylabel="Number of Transactions",
    title="Normal vs Fraudulent Transactions"
)

plt.xticks([0, 1], ["Normal", "Fraud"], rotation=0)
plt.tight_layout()
plt.savefig("fraud_vs_normal.png")
plt.show()

# Transaction Amount Distribution
plt.figure(figsize=(8, 5))

plt.hist(
    df[df["Is_Fraud"] == 0]["Amount"],
    bins=30,
    alpha=0.7,
    label="Normal"
)

plt.hist(
    df[df["Is_Fraud"] == 1]["Amount"],
    bins=30,
    alpha=0.7,
    label="Fraud"
)

plt.xlabel("Transaction Amount")
plt.ylabel("Number of Transactions")
plt.title("Transaction Amount Distribution")
plt.legend()

plt.tight_layout()
plt.savefig("transaction_amount_distribution.png")
plt.show()

# Fraud Transactions by Hour
fraud_by_hour = df[df["Is_Fraud"] == 1]["Hour"].value_counts().sort_index()

plt.figure(figsize=(9, 5))

fraud_by_hour.plot(
    kind="bar",
    xlabel="Hour of Day",
    ylabel="Number of Fraudulent Transactions",
    title="Fraudulent Transactions by Hour"
)

plt.tight_layout()
plt.savefig("fraud_by_hour.png")
plt.show()

import seaborn as sns

# Confusion Matrix Heatmap
cm = confusion_matrix(df["Is_Fraud"], df["Predicted_Fraud"])

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Normal", "Fraud"],
    yticklabels=["Normal", "Fraud"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Fraud Detection Confusion Matrix")

plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# Risk Score Distribution
plt.figure(figsize=(8, 5))

plt.hist(
    df["Risk_Score"],
    bins=20,
    alpha=0.8
)

plt.xlabel("Risk Score")
plt.ylabel("Number of Transactions")
plt.title("Transaction Risk Score Distribution")

plt.tight_layout()
plt.savefig("risk_score_distribution.png")
plt.show()