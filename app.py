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
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix
)
import shap
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Explainable Credit Risk Engine",
    page_icon="📊",
    layout="wide"
)

st.title("Explainable Credit Risk Engine")
st.caption(
    "Credit card default prediction using a trained Logistic Regression model. "
    "Educational portfolio project — not for real lending decisions."
)

# Load the saved model
MODEL_PATH = Path(__file__).parent / "credit_risk_model.joblib"

try:
    saved = joblib.load(MODEL_PATH)
    model = saved["model"]
    feature_columns = saved["feature_columns"]
except FileNotFoundError:
    st.error(
        "Model file not found. Run train_model.py first "
        "to create credit_risk_model.joblib."
    )
    st.stop()
except Exception as error:
    st.error(f"Could not load the model: {error}")
    st.stop()

# Feature descriptions from the UCI credit-card dataset
feature_labels = {
    "X1": "Credit limit",
    "X2": "Sex code (1 or 2)",
    "X3": "Education code (1–6)",
    "X4": "Marital status code (1–3)",
    "X5": "Age (years)",
    "X6": "Repayment status — most recent month",
    "X7": "Repayment status — month 2",
    "X8": "Repayment status — month 3",
    "X9": "Repayment status — month 4",
    "X10": "Repayment status — month 5",
    "X11": "Repayment status — month 6",
    "X12": "Bill amount — most recent month",
    "X13": "Bill amount — month 2",
    "X14": "Bill amount — month 3",
    "X15": "Bill amount — month 4",
    "X16": "Bill amount — month 5",
    "X17": "Bill amount — month 6",
    "X18": "Payment amount — most recent month",
    "X19": "Payment amount — month 2",
    "X20": "Payment amount — month 3",
    "X21": "Payment amount — month 4",
    "X22": "Payment amount — month 5",
    "X23": "Payment amount — month 6",
}

# Illustrative starting values only; edit these for each analysis.
defaults = {
    "X1": 50000,
    "X2": 2,
    "X3": 2,
    "X4": 1,
    "X5": 30,
    "X6": 0, "X7": 0, "X8": 0,
    "X9": 0, "X10": 0, "X11": 0,
    "X12": 20000, "X13": 18000, "X14": 15000,
    "X15": 12000, "X16": 10000, "X17": 8000,
    "X18": 2000, "X19": 2000, "X20": 1500,
    "X21": 1000, "X22": 1000, "X23": 1000,
}

st.sidebar.header("Applicant Details")
st.sidebar.caption(
    "Values must follow the dataset's numeric coding. "
    "Starting values are illustrative, not a real applicant."
)

with st.sidebar.form("applicant_form"):
    values = {}

    st.sidebar.subheader("Profile")
    for feature in ["X1", "X2", "X3", "X4", "X5"]:
        values[feature] = st.number_input(
            feature + " — " + feature_labels[feature],
            value=float(defaults[feature]),
            step=1.0,
            key="input_" + feature
        )

    st.sidebar.subheader("Repayment history")
    for feature in ["X6", "X7", "X8", "X9", "X10", "X11"]:
        values[feature] = st.number_input(
            feature + " — " + feature_labels[feature],
            value=float(defaults[feature]),
            step=1.0,
            key="input_" + feature
        )

    st.sidebar.subheader("Bill amounts")
    for feature in ["X12", "X13", "X14", "X15", "X16", "X17"]:
        values[feature] = st.number_input(
            feature + " — " + feature_labels[feature],
            value=float(defaults[feature]),
            step=1000.0,
            key="input_" + feature
        )

    st.sidebar.subheader("Previous payments")
    for feature in ["X18", "X19", "X20", "X21", "X22", "X23"]:
        values[feature] = st.number_input(
            feature + " — " + feature_labels[feature],
            value=float(defaults[feature]),
            step=500.0,
            key="input_" + feature
        )

    analyze = st.form_submit_button("Analyze Credit Risk")

if analyze:
    applicant = pd.DataFrame(
        [[values[column] for column in feature_columns]],
        columns=feature_columns
    )

    # Predict probability and class
    probabilities = model.predict_proba(applicant)[0]
    classes = list(model.named_steps["classifier"].classes_)
    default_probability = float(probabilities[classes.index(1)])
    prediction = int(model.predict(applicant)[0])
        # Model Performance Dashboard
    st.subheader("Model Performance Dashboard")

    test_data = pd.read_csv(
        Path(__file__).parent / "credit_data.csv"
    )

    X_test_all = test_data[feature_columns]
    y_test_all = test_data["Y"]

    X_train, X_test, y_train, y_test = train_test_split(
        X_test_all,
        y_test_all,
        test_size=0.2,
        random_state=42,
        stratify=y_test_all
    )

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    m1, m2, m3, m4 = st.columns(4)

    m1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2%}")
    m2.metric(
        "Precision",
        f"{precision_score(y_test, y_pred, zero_division=0):.2%}"
    )
    m3.metric(
        "Recall",
        f"{recall_score(y_test, y_pred, zero_division=0):.2%}"
    )
    m4.metric("ROC-AUC", f"{roc_auc_score(y_test, y_prob):.3f}")

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y_test, y_pred)

    fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
    image = ax_cm.imshow(cm)

    ax_cm.set_xticks([0, 1])
    ax_cm.set_yticks([0, 1])
    ax_cm.set_xticklabels(["Predicted: No Default", "Predicted: Default"])
    ax_cm.set_yticklabels(["Actual: No Default", "Actual: Default"])
    ax_cm.set_ylabel("Actual")
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_title("Confusion Matrix")

    for row in range(2):
        for col in range(2):
            ax_cm.text(
                col, row, str(cm[row, col]),
                ha="center", va="center"
            )

    fig_cm.tight_layout()
    st.pyplot(fig_cm)
    plt.close(fig_cm)
    st.subheader("Risk Assessment")
    col1, col2 = st.columns(2)

    col1.metric(
        "Estimated default probability",
        f"{default_probability:.1%}"
    )

    col2.metric(
        "Model classification",
        "Default predicted" if prediction == 1 else "No default predicted"
    )

    st.progress(default_probability)

    st.warning(
        "This is a model estimate, not a guarantee or a lending decision. "
        "The model was trained on historical data and may be biased or inaccurate."
    )

    st.subheader("Applicant Input Summary")
    summary = applicant.T.reset_index()
    summary.columns = ["Feature", "Entered value"]
    summary["Description"] = summary["Feature"].map(feature_labels)
    st.dataframe(summary, use_container_width=True, hide_index=True)

    # Explain the model using standardized-feature contributions
    st.subheader("Model Feature Contributions")

    # SHAP Explainability
    st.subheader("SHAP Explainability")
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    try:
        # Use the fitted preprocessing pipeline
        background = pd.read_csv(
            Path(__file__).parent / "credit_data.csv"
        )[feature_columns].sample(
            n=100,
            random_state=42
        )

        background_transformed = preprocessor.transform(background)
        applicant_transformed = preprocessor.transform(applicant)

        if hasattr(background_transformed, "toarray"):
            background_transformed = background_transformed.toarray()

        if hasattr(applicant_transformed, "toarray"):
            applicant_transformed = applicant_transformed.toarray()

        explainer = shap.LinearExplainer(
            classifier,
            background_transformed
        )

        shap_values = explainer(applicant_transformed)

        values_to_plot = shap_values.values[0]

        shap_result = pd.DataFrame({
            "Feature": feature_columns,
            "SHAP value": values_to_plot
        })

        shap_result = shap_result.sort_values("SHAP value")

        fig_shap, ax_shap = plt.subplots(figsize=(9, 6))

        bar_colors = [
            "tomato" if value > 0 else "steelblue"
            for value in shap_result["SHAP value"]
        ]

        ax_shap.barh(
            shap_result["Feature"],
            shap_result["SHAP value"],
            color=bar_colors
        )

        ax_shap.axvline(0, color="black", linewidth=0.8)
        ax_shap.set_xlabel("SHAP value (log-odds scale)")
        ax_shap.set_title("SHAP Explanation for This Applicant")

        fig_shap.tight_layout()
        st.pyplot(fig_shap)
        plt.close(fig_shap)

        st.caption(
            "Positive SHAP values push the prediction toward default; "
            "negative values push it away. SHAP explains model behavior, "
            "not cause and effect."
        )

    except Exception as error:
        st.error(f"SHAP explanation error: {error}")

    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    transformed = preprocessor.transform(applicant)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    contributions = transformed[0] * classifier.coef_[0]

    explanation = pd.DataFrame({
        "Feature": feature_columns,
        "Description": [
            feature_labels.get(column, column)
            for column in feature_columns
        ],
        "Contribution to default log-odds": contributions
    })

    explanation = explanation.sort_values(
        "Contribution to default log-odds"
    )

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = [
        "tomato" if value > 0 else "steelblue"
        for value in explanation["Contribution to default log-odds"]
    ]

    ax.barh(
        explanation["Feature"],
        explanation["Contribution to default log-odds"],
        color=colors
    )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Contribution to model log-odds")
    ax.set_title("Feature Contributions for This Prediction")
    fig.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "Positive contributions push the model toward predicting default; "
        "negative contributions push it away. These are model-specific "
        "contributions, not proof of causation."
    )

    st.subheader("Largest Contributions")
    st.dataframe(
        explanation.assign(
            **{
                "Contribution to default log-odds":
                explanation["Contribution to default log-odds"].round(4)
            }
        ).sort_values(
            "Contribution to default log-odds",
            key=lambda column: column.abs(),
            ascending=False
        ).head(10),
        use_container_width=True,
        hide_index=True
    )

else:
    st.info(
        "Enter the feature values in the sidebar and click "
        "'Analyze Credit Risk' to see a prediction and explanation."
    )
