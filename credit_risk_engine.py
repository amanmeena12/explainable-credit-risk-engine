#-------------------------------------------------------------------------------
# Name:        module2
# Purpose:
#
# Author:      am199
#
# Created:     12-10-2026
# Copyright:   (c) am199 2026
# Licence:     <your licence>
#-------------------------------------------------------------------------------



import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Sample practice dataset
data = {
    "income": [45000, 65000, 30000, 80000, 25000,
               55000, 40000, 90000, 35000, 70000],
    "debt": [18000, 10000, 22000, 12000, 20000,
             15000, 25000, 8000, 18000, 11000],
    "credit_history_years": [3, 8, 1, 10, 1,
                              5, 2, 12, 2, 7],
    "missed_payments": [2, 0, 4, 0, 5,
                        1, 3, 0, 4, 0],
    "default": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
}

df = pd.DataFrame(data)

# Separate features and target
X = df.drop(columns=["default"])
y = df["default"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.4,
    random_state=42,
    stratify=y
)

# Create and train the model
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000, random_state=42)
)

model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)

print("CREDIT RISK ENGINE - MODEL RESULTS")
print("=" * 45)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

print("\nAccuracy:", round(accuracy_score(y_test, y_pred), 2))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# Predict for a sample applicant
applicant = pd.DataFrame([{
    "income": 45000,
    "debt": 18000,
    "credit_history_years": 3,
    "missed_payments": 2
}])

prediction = model.predict(applicant)[0]
probabilities = model.predict_proba(applicant)[0]
default_probability = probabilities[
    list(model.named_steps["logisticregression"].classes_).index(1)
]

print("\nSAMPLE APPLICANT")
print("-" * 45)
print("Predicted class:", int(prediction))
print("Estimated default probability:",
      f"{default_probability:.1%}")

print("\nNote: This is a tiny synthetic dataset.")
print("The results are for learning purposes only.")

import shap
import matplotlib.pyplot as plt

# Explain the trained Logistic Regression model
classifier = model.named_steps["logisticregression"]
scaler = model.named_steps["standardscaler"]

# Transform the sample applicant using the trained scaler
applicant_scaled = scaler.transform(applicant)

# SHAP LinearExplainer explains the model's output
explainer = shap.LinearExplainer(
    classifier,
    scaler.transform(X_train)
)

shap_values = explainer(applicant_scaled)

# Display feature contributions
print("\nEXPLAINABLE AI - SHAP RESULTS")
print("=" * 45)

for feature, contribution in zip(
    X.columns, shap_values.values[0]
):
    print(f"{feature}: {contribution:.4f}")

# Plot the feature contributions
shap.plots.waterfall(shap_values[0], show=False)
plt.tight_layout()
plt.show()
