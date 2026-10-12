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

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

print("=" * 55)
print("EXPLAINABLE CREDIT RISK ENGINE")
print("REAL DATASET MODEL TRAINING")
print("=" * 55)

# 1. Load dataset
df = pd.read_csv("credit_data.csv")

print("\nDataset shape:", df.shape)
print("\nAvailable columns:")
print(df.columns.tolist())

# 2. Identify target column
possible_targets = [
    "Y",
    "default.payment.next.month",
    "default_payment_next_month"
]

target = next(
    (col for col in possible_targets if col in df.columns),
    None
)

if target is None:
    raise ValueError(
        "Target column not found. Check the printed column names."
    )

# 3. Separate features and target
X = df.drop(columns=[target])
y = df[target]

# Remove identifier columns if present
id_columns = [
    col for col in X.columns
    if col.strip().lower() in ["id", "client_id", "customer_id"]
]

X = X.drop(columns=id_columns, errors="ignore")

# Remove rows with missing target labels
valid_rows = y.notna()
X = X.loc[valid_rows].copy()
y = y.loc[valid_rows].astype(int)

print("\nTarget column:", target)
print("\nTarget distribution:")
print(y.value_counts())

# 4. Identify numeric and categorical features
numeric_columns = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_columns = X.select_dtypes(
    exclude=["number"]
).columns.tolist()

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_columns),
    ("categorical", categorical_pipeline, categorical_columns)
])

# 5. Split before fitting preprocessing to avoid leakage
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 6. Train baseline model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    ))
])

print("\nTraining model...")
model.fit(X_train, y_train)

# 7. Evaluate on held-out test data
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

print("\n" + "=" * 55)
print("MODEL EVALUATION")
print("=" * 55)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print("Accuracy:", round(accuracy_score(y_test, predictions), 4))
print(
    "Precision:",
    round(precision_score(y_test, predictions, zero_division=0), 4)
)
print(
    "Recall:",
    round(recall_score(y_test, predictions, zero_division=0), 4)
)
print("ROC-AUC:", round(roc_auc_score(y_test, probabilities), 4))

print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    zero_division=0
))

print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))

print("\nTraining completed successfully.")
print("Note: These metrics are not proof of production readiness.")
