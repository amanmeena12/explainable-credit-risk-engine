#-------------------------------------------------------------------------------
# Name:        Untitled1
# Purpose:
#
# Author:      am199
#
# Created:     12-10-2026
# Copyright:   (c) am199 2026
# Licence:     <your licence>
#-------------------------------------------------------------------------------

# Explainable Credit Risk Engine

An end-to-end machine learning project that predicts credit card default risk and explains individual predictions through feature contributions and SHAP.

## Overview

The Explainable Credit Risk Engine uses a Logistic Regression model trained on a public credit card default dataset. It combines model evaluation, an interactive Streamlit dashboard, and explainability techniques to make model behavior easier to inspect.

**This is an educational portfolio project, not a production lending system.**

## Key Features

- Trains a Logistic Regression classification model.
- Uses a public dataset containing 30,000 records.
- Reports Accuracy, Precision, Recall, and ROC-AUC.
- Displays a confusion matrix.
- Provides an interactive Streamlit interface for entering applicant feature values.
- Estimates default probability for an input record.
- Visualizes feature contributions.
- Uses SHAP to explain individual model predictions.

## Technology Stack

- Python
- Pandas
- Scikit-learn
- Streamlit
- Matplotlib
- SHAP
- Joblib

## Dataset

The project uses the **Default of Credit Card Clients** dataset from the UCI Machine Learning Repository.

- Number of records: 30,000
- Input features: 23
- Target: `Y`
- `Y = 1`: Default
- `Y = 0`: No default

Dataset source: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients

## Model Performance

The initial Logistic Regression evaluation produced the following results:

| Metric | Result |
|---|---:|
| Accuracy | 67.97% |
| Precision | 36.72% |
| Recall | 62.02% |
| ROC-AUC | 0.7081 |

These are the initial evaluation results. Results shown in the dashboard should be checked against the same held-out test split before publishing final performance claims.

## Project Structure

```text
Explainable-Credit-Risk-Engine/
├── app.py
├── train_model.py
├── credit_risk_engine.py
├── data.py
├── download_dataset.py
├── credit_data.csv
├── credit_risk_model.joblib
└── README.md
```

## Installation

Python 3.11 is recommended.

1. Clone or download this repository.
2. Open a terminal in the project directory.
3. Install the required packages:

```bash
python -m pip install pandas scikit-learn streamlit matplotlib shap joblib ucimlrepo
```

## Run the Project

### 1. Download the dataset

If the dataset is not already present, run:

```bash
python download_dataset.py
```

### 2. Train the model

```bash
python train_model.py
```

This should create `credit_risk_model.joblib`.

### 3. Launch the dashboard

```bash
python -m streamlit run app.py
```

Open the local URL shown in the terminal.

## Explainability

The dashboard presents feature contributions and SHAP values for an individual prediction.

- Positive contributions push the model output toward default.
- Negative contributions push the model output away from default.

These explanations describe model behavior. They do not establish causation or guarantee that a prediction is correct.

## Limitations and Responsible Use

- The model is an educational prototype and must not be used to approve or reject real credit applications.
- Performance depends on the dataset, preprocessing, and decision threshold.
- Historical data may contain bias and may not represent future applicants.
- The dataset contains demographic-related features; their use requires careful fairness and legal review.
- Predicted probabilities should not be assumed to be calibrated without additional testing.

## Future Improvements

- Compare Logistic Regression with tree-based models.
- Evaluate probability calibration and alternative decision thresholds.
- Add fairness and subgroup performance analysis.
- Save and report a consistent held-out test evaluation.
- Improve automated testing and error handling.

## Author

Created as a machine learning and data science portfolio project.
