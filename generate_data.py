import pandas as pd
import numpy as np

np.random.seed(42)

transactions = []

for i in range(1, 5001):

    amount = np.random.randint(100, 50000)

    hour = np.random.randint(0, 24)

    distance_from_home = np.random.randint(1, 100)

    transaction_frequency = np.random.randint(1, 20)

    transactions.append({
        "Transaction_ID": f"T{i:05d}",
        "Amount": amount,
        "Hour": hour,
        "Distance_From_Home": distance_from_home,
        "Transaction_Frequency": transaction_frequency
    })

df = pd.DataFrame(transactions)

# Create realistic fraud cases
fraud_indices = np.random.choice(df.index, size=150, replace=False)

for index in fraud_indices:

    fraud_type = np.random.choice(["high_amount", "unusual_location", "high_frequency"])

    if fraud_type == "high_amount":
        df.loc[index, "Amount"] = np.random.randint(40000, 80000)

    elif fraud_type == "unusual_location":
        df.loc[index, "Distance_From_Home"] = np.random.randint(80, 250)

    elif fraud_type == "high_frequency":
        df.loc[index, "Transaction_Frequency"] = np.random.randint(15, 35)

df["Is_Fraud"] = 0
df.loc[fraud_indices, "Is_Fraud"] = 1

df.to_csv("fraud_transactions.csv", index=False)

print("Realistic fraud transaction dataset generated successfully!")
print("Total transactions:", len(df))
print("Fraudulent transactions:", df["Is_Fraud"].sum())
print("Dataset saved as fraud_transactions.csv")