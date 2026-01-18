import numpy as np
import streamlit as st
import pandas as pd
import pickle
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
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Credit Rating ML App", layout="wide")
st.title("Corporate Credit Rating Prediction (Binary Classification)")

# --------------------------------------------------
# Load trained models
# --------------------------------------------------
with open(r"model\saved_models.pkl", "rb") as f:
    bundle = pickle.load(f)

models = bundle["models"]
scaler = bundle["scaler"]
X_test = bundle["X_test"]
y_test = bundle["y_test"]

# --------------------------------------------------
# Sidebar model selection
# --------------------------------------------------
model_name = st.sidebar.selectbox("Select ML Model", list(models.keys()))
model = models[model_name]

# --------------------------------------------------
# Predictions on test data
# --------------------------------------------------
y_pred = model.predict(X_test)

# Probability of positive class (class = 1)
y_prob = model.predict_proba(X_test)[:, 1]

# --------------------------------------------------
# Metrics (Binary)
# --------------------------------------------------
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
mcc = matthews_corrcoef(y_test, y_pred)
auc_score = roc_auc_score(y_test, y_prob)

# --------------------------------------------------
# Display metrics
# --------------------------------------------------
st.subheader("Model Evaluation Metrics")

col1, col2, col3 = st.columns(3)
col4, col5, col6 = st.columns(3)

col1.metric("Accuracy", f"{accuracy:.3f}")
col2.metric("AUC Score", f"{auc_score:.3f}")
col3.metric("Precision", f"{precision:.3f}")

col4.metric("Recall", f"{recall:.3f}")
col5.metric("F1 Score", f"{f1:.3f}")
col6.metric("MCC Score", f"{mcc:.3f}")

# --------------------------------------------------
# Confusion Matrix
# --------------------------------------------------
st.subheader("Confusion Matrix")

fig, ax = plt.subplots(figsize=(3, 3))
sns.heatmap(
    confusion_matrix(y_test, y_pred).T,
    annot=True,
    fmt="d",
    cmap="Reds",
    ax=ax
)

ax.invert_xaxis()
ax.invert_yaxis()

ax.set_xticklabels(["Negative", "Positive"])
ax.set_yticklabels(["Negative", "Positive"])
ax.set_xlabel("Predicted Label")
ax.xaxis.set_label_position('top')   # 👈 move label to top
ax.xaxis.tick_top()

ax.set_ylabel("True Label")
st.pyplot(fig)

# --------------------------------------------------
# Classification Report
# --------------------------------------------------
st.subheader("Detailed Classification Report")

report = classification_report(
    y_test,
    y_pred,
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report).transpose()
st.dataframe(report_df)

# --------------------------------------------------
# CSV Upload - Inference only
# --------------------------------------------------
st.subheader("Upload New Dataset for Prediction (Unseen Data)")

uploaded = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)

    df_scaled = scaler.transform(df)
    predictions = model.predict(df_scaled)
    probabilities = model.predict_proba(df_scaled)[:, 1]

    df_output = df.copy()
    df_output["Predicted_Class"] = predictions
    df_output["Positive_Class_Probability"] = probabilities

    st.write("Prediction Results")
    st.dataframe(df_output)
