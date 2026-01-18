import os

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Load data
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "CreditRatingPrediction_train.csv")

data = pd.read_csv(DATA_PATH)

def data_preprocessing(data):
    data = (data.drop
            (columns=["Rating Date", "CIK", "Ticker", "Sector", "SIC Code", "Corporation", "Rating Agency", 'Rating']))
    X = data.drop(['Binary Rating'], axis=1)
    y = data['Binary Rating']
    return X, y

X, y =data_preprocessing(data)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Decision Tree": DecisionTreeClassifier(max_depth=10),
    "KNN": KNeighborsClassifier(n_neighbors=7),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(n_estimators=200),
    "XGBoost": XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        use_label_encoder=False
    )
}

# Train
for model in models.values():
    model.fit(X_train, y_train)

# Save everything
with open(r"saved_models.pkl", "wb") as f:
    pickle.dump({
        "models": models,
        "scaler": scaler,
        "X_test": X_test,
        "y_test": y_test
    }, f)

print("Models trained and saved successfully")
