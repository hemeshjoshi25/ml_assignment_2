# Corporate Credit Rating Prediction (Binary Classification) - ML Assignment 2

🔗 Live Streamlit App:
https://mlassignment2-hemeshjoshi.streamlit.app/

---
## 1. Problem Statement

**Business Problem Framing:** Binary Credit Rating Classification

Credit rating agencies such as Standard & Poor’s (S&P) assign detailed credit ratings (AAA, AA+, AA, …, D) to evaluate a company’s creditworthiness. However, in many real-world financial decision-making scenarios—such as investment screening, credit risk assessment, and lending approvals—the primary requirement is to determine whether a company is Investment Grade or Non-Investment Grade, rather than predicting the exact rating category.

To align the machine learning solution with this practical business objective, the original multi-class credit rating labels were intentionally dropped, and the problem was reformulated as a binary classification task based on Standard & Poor’s Credit Ratings.

**Target Variable Definition**

The target variable used is Binary Rating, defined as:

1 – Investment Grade: Credit ratings of BBB− and above

0 – Non-Investment Grade: Ratings below BBB−

This formulation enables effective prediction of corporate creditworthiness using financial ratios while simplifying the decision boundary for real-world applications.

## 2. Dataset Description

Dataset Name: Corporate Credit Rating with Financial Ratios

Source: Kaggle

Type: Binary Classification

Number of Records: > 500

Number of Features: > 12

The dataset satisfies all assignment constraints regarding minimum instance size and feature count.

```bash
import kagglehub

# Download latest version
path = kagglehub.dataset_download("agewerc/corporate-credit-rating")

print("Path to dataset files:", path)
```
## 3. Machine Learning Models Implemented

The following six classification models were implemented on the same dataset and evaluated using a consistent train-test split:

1. Logistic Regression 
2. Decision Tree Classifier 
3. K-Nearest Neighbors (KNN)
4. Naive Bayes (Gaussian)
5. Random Forest (Ensemble Model)
6. XGBoost (Ensemble Model)

### Evaluation Metrics
Each model is evaluated using the following metrics as required by the assignment:
1. Accuracy 
2. Precision 
3. Recall 
4. F1 Score 
5. AUC Score 
6. Matthews Correlation Coefficient (MCC)

| ML Model                  | Accuracy | AUC   | Precision | Recall | F1 Score | MCC   |
| ------------------------- | -------- | ----- | --------- | ------ | -------- | ----- |
| Logistic Regression       | 0.792    | 0.823 | 0.799     | 0.922  | 0.856    | 0.504 |
| Decision Tree             | 0.847    | 0.902 | 0.879     | 0.895  | 0.887    | 0.650 |
| K-Nearest Neighbors (KNN) | 0.919    | 0.941 | 0.932     | 0.948  | 0.940    | 0.815 |
| Naive Bayes               | 0.713    | 0.742 | 0.713     | 0.959  | 0.818    | 0.271 |
| Random Forest (Ensemble)  | 0.810    | 0.866 | 0.808     | 0.941  | 0.869    | 0.551 |
| XGBoost (Ensemble)        | 0.848    | 0.907 | 0.860     | 0.925  | 0.891    | 0.647 | 


## 4. Model-Wise Observation

| ML Model                      | Observation                                                                                                                                                                               |
| ----------------------------- |-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Logistic Regression**       | Performs well as a baseline model with high recall, making it suitable for minimizing false negatives in credit risk assessment. However, it assumes linear separability between classes. |
| **Decision Tree**             | Captures non-linear relationships effectively but shows signs of overfitting when depth increases. Performance is sensitive to tree depth.                                                |
| **K-Nearest Neighbors (KNN)** | Performance depends heavily on feature scaling and choice of K. Computationally expensive for large datasets but effective for localized decision boundaries.                             |
| **Naive Bayes**               | Fast and efficient with reasonable performance. Assumes feature independence, which limits its predictive power for correlated financial ratios.                                          |
| **Random Forest**             | Demonstrates strong and stable performance due to ensemble averaging. Reduces overfitting compared to a single decision tree.                                                             |
| **XGBoost**                   | Achieves the best overall performance in terms of accuracy and AUC. Effectively captures complex feature interactions using gradient boosting.                                            |



## 5. Repository Structure
ml_assignment_2/
- app.py # Streamlit app
- model/
  -  saved_models.pkl # Pickled trained models & scaler
  -  train_models.py
  -  evaluate.py
- data/
  - CreditRatingPrediction_train.csv # Training dataset
  - CreditRatingPrediction_test.csv # Training dataset
- requirements.txt # Required Python packages
- README.md

---

## 6. Requirements
### 6.1 Required packages include:
streamlit – version 1.52.2
pandas – version 2.3.3
scikit-learn – version 1.7.2
matplotlib – version 3.10.7
seaborn – version – 0.13.2
xgboost -version 3.1.2
numpy – version 2.3.4

## 7. Getting Started
### 7.1 Clone the Repository

```bash
git clone https://github.com/hemeshjoshi25/ml_assignment_2.git
cd ml_assignment_2
```

### 7.2 Install Dependencies
``` bash
pip install -r requirements.txt

```
## 8. Run Locally
```bash
streamlit run app.py
```

## 9. CSV Upload Instructions

Uploaded CSV must contain the same features as the training dataset, excluding these columns:
Rating Date, CIK, Ticker, Sector, SIC Code, Corporation, Rating Agency, Rating

If your CSV includes the target column Binary Rating, the app will compute evaluation metrics dynamically.
If the CSV does not include the target, the app will only generate predictions and probabilities.
Feature order must match the training dataset.

## 10. Model Details
Logistic Regression: Simple linear model for binary classification
Decision Tree: Max depth = 10, interpretable tree structure
KNN: K = 7, predicts based on nearest neighbors
Naive Bayes: Gaussian assumption for feature distributions
Random Forest: 200 trees, ensemble model
XGBoost: Gradient boosting, objective = binary:logistic

## 11. Deployment
The app is deployed on Streamlit Community Cloud.
https://mlassignment2-hemeshjoshi.streamlit.app/

## 12. Author Hemesh Joshi
M.Tech (AI & ML) - Work Integrated Learning Program (BITS WILP)
GitHub: https://github.com/hemeshjoshi25
