import os

import pandas as pd
import pickle
from model import train_models
from model.train_models import X_train
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix
)
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

#Common Methods

def metrics_calculation(y_true, y_pred, y_prob):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    auc_score = roc_auc_score(y_true, y_prob)
    col1, col2, col3 = st.columns(3)
    col4, col5, col6 = st.columns(3)
    col1.metric("Accuracy", f"{accuracy:.3f}")
    col2.metric("AUC Score", f"{auc_score:.3f}")
    col3.metric("Precision", f"{precision:.3f}")
    col4.metric("Recall", f"{recall:.3f}")
    col5.metric("F1 Score", f"{f1:.3f}")
    col6.metric("MCC Score", f"{mcc:.3f}")

def confusion_matrix_plot(y_true, y_pred, color):
    st.subheader("Confusion Matrix (Test Set)")
    fig, ax = plt.subplots(figsize=(3, 3))
    sns.heatmap(
        confusion_matrix(y_true, y_pred).T,
        annot=True,
        fmt="d",
        cmap=color,
        ax=ax
    )
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.set_xticklabels(["Negative", "Positive"])
    ax.set_yticklabels(["Negative", "Positive"])
    ax.set_xlabel("Predicted Label")
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    ax.set_ylabel("True Label")
    st.pyplot(fig)

def report_description(report_header, y_test, y_pred):
    st.subheader(report_header)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)

st.set_page_config(page_title="Credit Rating ML App", layout="wide")
st.title("Corporate Credit Rating Prediction (Binary Classification)")

# --------------------------------------------------
# Load trained models and test set
# --------------------------------------------------
MODEL_PATH = os.path.join("model", "saved_models.pkl")
with open(MODEL_PATH, "rb") as f:
    bundle = pickle.load(f)

models = bundle["models"]
scaler = bundle["scaler"]
X_test = bundle["X_test"]
y_test = bundle["y_test"]

# Sidebar model selection
model_name = st.sidebar.selectbox("Select ML Model", list(models.keys()))
model = models[model_name]

# --------------------------------------------------
# Section 1: Metrics on original test set
# --------------------------------------------------
st.subheader("Evaluation on Original Validation Set of Training Data")

y_pred_test = model.predict(X_test)
y_prob_test = model.predict_proba(X_test)[:, 1]

#Metrics Calculation
metrics_calculation(y_test, y_pred_test, y_prob_test)
# Confusion Matrix
confusion_matrix_plot(y_test, y_pred_test, "Reds")
# Classification Report
report_description("Detailed Classification Report (Test Set)", y_test, y_pred_test)

# --------------------------------------------------
# Section 2: CSV Upload - dynamic inference & evaluation
# --------------------------------------------------
st.subheader("Upload New Dataset for Predictions / Evaluation")

uploaded = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    st.write("Preview of Uploaded Data")
    st.dataframe(df.head())

    # Drop target if exists, keep features same as training data
    if "Binary Rating" in df.columns:
        X_uploaded, y_uploaded = train_models.data_preprocessing(df)
    else:
        X_uploaded = df
        y_uploaded = None

    # Check if uploaded CSV has the correct number of features
    if X_uploaded.shape[1] != X_train.shape[1]:
        st.error(f"Uploaded CSV must have {X_train.shape[1]} features (found {X_uploaded.shape[1]}).")
    else:
        # Scale features
        X_scaled = scaler.transform(X_uploaded)

        # Predict
        y_pred_uploaded = model.predict(X_scaled)
        y_prob_uploaded = model.predict_proba(X_scaled)[:, 1]

        if y_uploaded is not None:
            st.subheader("Evaluation Metrics on Uploaded(Test) Data file")
            # Compute metrics
            metrics_calculation(y_uploaded, y_pred_uploaded, y_prob_uploaded)
            # Confusion Matrix
            confusion_matrix_plot(y_uploaded, y_pred_uploaded, "Blues")
            # Classification Report
            report_description("Classification Report (Uploaded Data)", y_uploaded, y_pred_uploaded)

        else:
            # If no target column, just show predictions and probabilities
            df_output = X_uploaded.copy()
            df_output["Predicted_Class"] = y_pred_uploaded
            df_output["Positive_Class_Probability"] = y_prob_uploaded
            st.subheader("Prediction Results (No Target Column)")
            st.dataframe(df_output)
