from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, roc_auc_score
)
import pandas as pd

def evaluate_model(model, X, y):
    y_pred = model.predict(X)

    return {
        "Accuracy": accuracy_score(y, y_pred),
        "Precision": precision_score(y, y_pred, average="weighted"),
        "Recall": recall_score(y, y_pred, average="weighted"),
        "F1": f1_score(y, y_pred, average="weighted"),
        "MCC": matthews_corrcoef(y, y_pred)
    }
