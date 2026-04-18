import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import json

def generate_synthetic_data(filepath):
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        "CustomerID": range(1, n_samples + 1),
        "CreditScore": np.random.randint(400, 850, n_samples),
        "Geography": np.random.choice(["France", "Spain", "Germany"], n_samples),
        "Gender": np.random.choice(["Male", "Female"], n_samples),
        "Age": np.random.randint(18, 92, n_samples),
        "Tenure": np.random.randint(0, 11, n_samples),
        "Balance": np.random.uniform(0.0, 250000.0, n_samples),
        "NumOfProducts": np.random.randint(1, 5, n_samples),
        "HasCrCard": np.random.randint(0, 2, n_samples),
        "IsActiveMember": np.random.randint(0, 2, n_samples),
        "EstimatedSalary": np.random.uniform(10000.0, 200000.0, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Generate some realistic logic for churn
    # Older, more balance in Germany, lower active members tend to churn
    churn_prob = (
        (df["Age"] > 45).astype(int) * 0.3 + 
        (df["IsActiveMember"] == 0).astype(int) * 0.2 + 
        (df["NumOfProducts"] >= 3).astype(int) * 0.4 +
        (df["Geography"] == "Germany").astype(int) * 0.1
    )
    
    df["Exited"] = (np.random.rand(n_samples) < churn_prob).astype(int)
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"✅ Generated synthetic data at {filepath}")
    return df

def train_and_save_model(data_path, model_path):
    if not os.path.exists(data_path):
        df = generate_synthetic_data(data_path)
    else:
        df = pd.read_csv(data_path)
        
    X = df.drop(["CustomerID", "Exited"], axis=1)
    y = df["Exited"]
    
    numeric_features = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "EstimatedSalary"]
    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    categorical_features = ["Geography", "Gender", "HasCrCard", "IsActiveMember"]
    categorical_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ])

    clf = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(random_state=42, class_weight="balanced"))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    score = clf.score(X_test, y_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    print(f"✅ Model trained! Test Accuracy: {score:.2f}")
    
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(clf, model_path)
    
    report_path = os.path.join(os.path.dirname(model_path), "evaluation_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)
        
    print(f"✅ Model saved to {model_path}")
    print(f"✅ Evaluation report saved to {report_path}")

if __name__ == "__main__":
    train_and_save_model(
        data_path="data/customer_churn_sample.csv",
        model_path="src/ml/model.pkl"
    )
