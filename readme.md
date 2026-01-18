a. Problem Statement
## Problem Statement
This project predicts corporate credit ratings based on financial ratios
such as profitability, leverage, and liquidity indicators. Accurate credit
rating prediction helps investors and financial institutions assess credit
risk and make informed lending and investment decisions.

## Dataset Description
Dataset Name: Corporate Credit Rating Prediction using Financial Ratios  
Source: Kaggle  
Total Samples: 7805  
Total Features: 23 (excluding target variable)  
Problem Type: Multiclass Classification  

The dataset contains corporate financial indicators and corresponding
credit ratings issued by Standard & Poor’s.


Predict corporate credit ratings using financial ratios to assist investors and financial institutions in assessing credit risk.

b. Dataset Description

The dataset contains 7,805 corporate records with 23 financial features such as profitability, leverage, and liquidity ratios. Ratings are sourced from Standard & Poor’s.

| Model               | Accuracy | AUC | Precision | Recall | F1 | MCC |
| ------------------- | -------- | --- | --------- | ------ | -- | --- |
| Logistic Regression | ✔        | ✔   | ✔         | ✔      | ✔  | ✔   |
| Decision Tree       | ✔        | ✔   | ✔         | ✔      | ✔  | ✔   |
| KNN                 | ✔        | ✔   | ✔         | ✔      | ✔  | ✔   |
| Naive Bayes         | ✔        | ✔   | ✔         | ✔      | ✔  | ✔   |
| Random Forest       | ✔        | ✔   | ✔         | ✔      | ✔  | ✔   |
| XGBoost             | ✔        | ✔   | ✔         | ✔      | ✔  | ✔   |


| Model               | Observation                                           |
| ------------------- | ----------------------------------------------------- |
| Logistic Regression | Performs well but limited for nonlinear relationships |
| Decision Tree       | Captures nonlinearity but prone to overfitting        |
| KNN                 | Sensitive to scaling and data size                    |
| Naive Bayes         | Fast but assumes feature independence                 |
| Random Forest       | Strong performance and robustness                     |
| XGBoost             | Best overall performance with high generalization     |
