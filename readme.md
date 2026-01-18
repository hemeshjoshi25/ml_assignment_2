# Corporate Credit Rating Prediction (Binary Classification) - ML Assignment 2

[Live Demo on Streamlit](https://mlassignment2-hemeshjoshi.streamlit.app/)

---
## Business Problem Framing: Binary Credit Rating Classification

Credit rating agencies assign detailed ratings (e.g., AAA, AA+, AA, …, D) to assess a company’s creditworthiness. However, in real-world financial decision-making—such as investment screening, credit risk assessment, and lending approvals—the primary concern is whether a company qualifies as investment grade or non-investment grade, rather than its exact rating category.

To align the machine learning solution with this business objective, the original multi-class credit rating labels were intentionally dropped and the problem was reformulated as a binary classification task only for Standard & Poor's Credit Rating.

### Target Variable Definition

Column Binary Rating used as target variable where Investment Grade (Label = 1) and Non-Investment Grade (Label = 0)
Credit ratings of BBB− and above, indicating financially stable companies with relatively low default risk.


### Why Binary Classification?

Reframing the problem as a binary classification task provides several practical advantages:

Business relevance: Directly answers the key question—Is this company safe to invest in?

Improved model stability: Reduces noise and ambiguity present in fine-grained rating classes.

Lower overfitting risk: Avoids sparsity issues associated with multiple low-frequency rating categories.

Clear interpretability: Produces outputs that are easily understood by non-technical stakeholders.

This approach ensures that the machine learning models deliver actionable, decision-ready insights aligned with real-world financial and investment use cases


## 1. Project Overview

This project implements multiple **machine learning models** to predict corporate credit ratings as **binary classification** (Positive / Negative).  

The app allows users to:

- Upload a CSV dataset for evaluation or prediction  
- Select from multiple trained models  
- View evaluation metrics dynamically  
- Visualize confusion matrix and classification reports  

It is deployed on **Streamlit Community Cloud** for interactive use.

---

## 2. Features

### ✅ Core Features

1. **Dataset Upload (CSV)** - Data Referred from Kaggle
    ```bash
    import kagglehub
    
    # Download latest version
    path = kagglehub.dataset_download("agewerc/corporate-credit-rating")
    
    print("Path to dataset files:", path)
   ```
   - Users can upload new datasets with the same features used in training.
   - If the CSV contains a target column (`Binary Rating`), evaluation metrics are computed.  
   - If the CSV does not contain a target column, the app only provides predictions and probabilities.

2. **Model Selection**
   - Available models:
     - Logistic Regression
     - Decision Tree
     - K-Nearest Neighbors
     - Naive Bayes
     - Random Forest
     - XGBoost
   - Users can select the model to apply on the uploaded data.

3. **Evaluation Metrics**
   - Accuracy  
   - Precision  
   - Recall  
   - F1 Score  
   - Matthews Correlation Coefficient (MCC)  
   - ROC AUC Score  

4. **Visualizations**
   - Confusion Matrix (Heatmap)  
   - Detailed Classification Report (DataFrame)  

5. **Prediction Download**
   - Users can download the predictions for uploaded datasets as a CSV.

---

## 3. Repository Structure
ml_assignment_2/
├─ app.py # Streamlit app
├─ model/
│ └─ saved_models.pkl # Pickled trained models & scaler
├─ data/
│ └─ CreditRatingPrediction_train.csv # Training dataset
├─ requirements.txt # Required Python packages
└─ README.md


---

## 4. Requirements
### 4.1 Required packages include:
streamlit
pandas
scikit-learn
matplotlib
seaborn
xgboost
numpy



## 5. Getting Started
### 5.1 Clone the Repository

```bash
git clone https://github.com/hemeshjoshi25/ml_assignment_2.git
cd ml_assignment_2
```

### 5.2 Install Dependencies
``` bash
pip install -r requirements.txt
```
## 6. Run Locally
```bash
streamlit run app.py
```

## 6. CSV Upload Instructions

Uploaded CSV must contain the same features as the training dataset, excluding these columns:
Rating Date, CIK, Ticker, Sector, SIC Code, Corporation, Rating Agency, Rating

If your CSV includes the target column Binary Rating, the app will compute evaluation metrics dynamically.
If the CSV does not include the target, the app will only generate predictions and probabilities.
Feature order must match the training dataset.

## 7. Model Details
Logistic Regression: Simple linear model for binary classification
Decision Tree: Max depth = 10, interpretable tree structure
KNN: K = 7, predicts based on nearest neighbors
Naive Bayes: Gaussian assumption for feature distributions
Random Forest: 200 trees, ensemble model
XGBoost: Gradient boosting, objective = binary:logistic

## 8. Deployment
The app is deployed on Streamlit Community Cloud.
https://mlassignment2-hemeshjoshi.streamlit.app/

## 9. Author Hemesh Joshi
M.Tech (AI & ML) - Work Integrated Learning Program (BITS WILP)
GitHub: https://github.com/hemeshjoshi25