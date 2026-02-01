import os

import numpy as np
import pandas as pd
import pickle
from model import train_models
from model.train_models import X_train, FEATURE_COLUMNS
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
    col1, col2= st.columns(2)
    col3, col4 = st.columns(2)
    col5, col6 = st.columns(2)
    col1.metric("Accuracy", f"{accuracy:.3f}")
    col2.metric("AUC Score", f"{auc_score:.3f}")
    col3.metric("Precision", f"{precision:.3f}")
    col4.metric("Recall", f"{recall:.3f}")
    col5.metric("F1 Score", f"{f1:.3f}")
    col6.metric("MCC Score", f"{mcc:.3f}")

def confusion_matrix_plot(y_true, y_pred, color):
    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    sns.heatmap(
        confusion_matrix(y_true, y_pred).T,
        annot=True,
        fmt="d",
        cmap=color,
        ax=ax,
        cbar=False,
        annot_kws={"size": 9}
    )
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.set_xticklabels(["Negative", "Positive"],fontsize=9)
    ax.set_yticklabels(["Negative", "Positive"], fontsize=9)
    ax.set_xlabel("Predicted Label", fontsize=9)
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    ax.set_ylabel("True Label", fontsize=9)
    st.pyplot(fig)

def report_description(report_header, y_test, y_pred):
    st.subheader(report_header)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)

def weighted_metrics_calculation(mteric_header, y_true, y_pred):
    st.subheader(mteric_header)
    precision_w = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall_w = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_w = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    col1, col2, col3 = st.columns(3)
    col1.metric("Weighted Precision", f"{precision_w:.3f}")
    col2.metric("Weighted Recall", f"{recall_w:.3f}")
    col3.metric("Weighted F1 Score", f"{f1_w:.3f}")


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

st.set_page_config(
    page_title="Corporate Credit Rating ML App",
    layout="wide"
)

st.title("Corporate Credit Rating Prediction (Binary Classification)")

page = st.radio(
    "Select Page",
    ["ML App", "Dataset Info", ],
    index=0,
    horizontal=True
)
if page == "Dataset Info":
    st.markdown(
        '<div class="section-header">Dataset Information</div>',
        unsafe_allow_html=True
    )

    st.markdown("**Training Feature Matrix Shape:**")
    st.write(f"Rows: {X_train.shape[0]}, Columns: {X_train.shape[1]}")

    st.markdown("**Target Distribution (Binary Rating):**")
    target_df = pd.Series(y_test).value_counts().rename(
        index={0: "Speculative Grade", 1: "Investment Grade"}
    )
    st.bar_chart(target_df)

    st.markdown("**Feature Preview (Training Data):**")
    X_train_df = pd.DataFrame(X_train, columns=FEATURE_COLUMNS)
    st.dataframe(X_train_df.head())

    st.markdown(
        '<div class="custom-divider"></div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        **Dataset Notes**
        - Binary target derived from credit rating
        - Features standardized using training-set statistics
        """
    )

elif page == "ML App":

    # Sidebar model selection
    model_name = st.sidebar.selectbox("Select ML Model", list(models.keys()))
    model = models[model_name]
    tab_eval, tab_predict  = st.tabs(
        ["Model Evaluation", "Predict on Test Data"]
    )

    with tab_eval:
        st.markdown("### Selected Model: " + model_name)
        st.subheader("Model Evaluation")

        y_pred_test = model.predict(X_test)
        y_prob_test = model.predict_proba(X_test)[:, 1]

        col_metrics, col_chart = st.columns([3, 3])

        # LEFT: Metrics
        with col_metrics:
            st.markdown(
                '<div class="section-header" style="font-size:20px; font-weight:bold;">Evaluation Metrics</div>',
                unsafe_allow_html=True
            )
            st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
            metrics_calculation(y_test, y_pred_test, y_prob_test)

        with col_chart:
            st.markdown(
                '<div class="section-header" style="font-size:20px; font-weight:bold;">Confusion Matrix (Train Set)</div>',
                unsafe_allow_html=True
            )
            confusion_matrix_plot(y_test, y_pred_test, "Reds")

        report_description(
            "Detailed Classification Report (Validation Set)",
            y_test,
            y_pred_test
        )

        weighted_metrics_calculation(
            "Weighted Metrics (Validation Set)",
            y_test,
            y_pred_test
        )

    with tab_predict:
        st.markdown("### Selected Model: " + model_name)
        st.markdown(
            '<div class="section-header">Prediction on Test Data</div>',
            unsafe_allow_html=True
        )

        st.markdown("### Download Sample Test Dataset")

        TEST_CSV_URL = (
            "https://raw.githubusercontent.com/hemeshjoshi25"
            "/ml_assignment_2/refs/heads/master/data/CreditRatingPrediction_test.csv"
        )

        df = pd.read_csv(TEST_CSV_URL)

        st.download_button(
            label="Download Test Data CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="CreditRatingPrediction_test.csv",
            mime="text/csv"
        )

        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        st.caption(
            "Upload an test dataset to generate credit rating predictions. It should not have target Column"
        )

        uploaded = st.file_uploader(
            "Upload CSV file",
            type=["csv"],
            key="inference_upload"
        )

        if uploaded:
            df = pd.read_csv(uploaded)

            st.subheader("Preview of Uploaded Data")
            st.dataframe(df.head())

            data = df.drop(columns=[
                "Rating Date", "CIK", "Ticker", "Sector",
                "SIC Code", "Corporation", "Rating Agency", "Rating"
            ])

            if data.shape[1] != X_train.shape[1]:
                st.error(
                    f"Uploaded CSV must have {X_train.shape[1]} features "
                    f"(found {data.shape[1]})."
                )
            else:
                X_scaled = scaler.transform(data)

                y_pred_uploaded = model.predict(X_scaled)
                y_prob_uploaded = model.predict_proba(X_scaled)[:, 1]

                df_output = df.copy()
                df_output["Predicted_Class"] = y_pred_uploaded
                df_output["Predicted_Label"] = np.where(
                    y_pred_uploaded == 1,
                    "Investment Grade",
                    "Speculative Grade"
                )
                df_output["Investment_Grade_Probability"] = y_prob_uploaded

                cols = [
                    "Predicted_Class",
                    "Predicted_Label",
                    "Investment_Grade_Probability"
                ] + [
                    c for c in df_output.columns
                    if c not in [
                        "Predicted_Class",
                        "Predicted_Label",
                        "Investment_Grade_Probability"
                    ]
                ]

                st.subheader("Predictions Mapped to Original Data")
                st.dataframe(df_output[cols])

