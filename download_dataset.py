#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      am199
#
# Created:     12-10-2026
# Copyright:   (c) am199 2026
# Licence:     <your licence>
#-------------------------------------------------------------------------------


import pandas as pd
from ucimlrepo import fetch_ucirepo

print("Downloading official credit-risk dataset...")

# Fetch dataset from UCI Machine Learning Repository
dataset = fetch_ucirepo(id=350)

# Separate features and target
X = dataset.data.features
y = dataset.data.targets

# Combine into one DataFrame
df = pd.concat([X, y], axis=1)

# Save dataset in the project folder
df.to_csv("credit_data.csv", index=False)

print("\nDataset downloaded successfully!")
print("Dataset shape:", df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst five rows:")
print(df.head())

print("\nTarget distribution:")
print(df.iloc[:, -1].value_counts())
