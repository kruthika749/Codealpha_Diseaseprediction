# --------------------------------------------------------------
#  Disease Prediction from Medical Data (Complete Project Code)
# --------------------------------------------------------------

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb

# -------------------------------------------------------------------
# 1. LOAD DATASETS
# -------------------------------------------------------------------
def load_dataset(name):
    if name == "diabetes":
        df = pd.read_csv("diabetes.csv")   # from Kaggle/UCI
        X = df.drop("Outcome", axis=1)
        y = df["Outcome"]
        return X, y

    elif name == "heart":
        df = pd.read_csv("heart.csv")
        X = df.drop("target", axis=1)
        y = df["target"]
        return X, y

    elif name == "breast_cancer":
        from sklearn.datasets import load_breast_cancer
        data = load_breast_cancer()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target)
        return X, y

    else:
        raise ValueError("Dataset name must be: diabetes / heart / breast_cancer")


# -------------------------------------------------------------------
# 2. TRAIN MULTIPLE MODELS
# -------------------------------------------------------------------
def train_models(X_train, y_train):
    models = {
        "Logistic Regression": LogisticRegression(),
        "SVM": SVC(kernel='linear', probability=True),
        "Random Forest": RandomForestClassifier(n_estimators=200),
        "XGBoost": xgb.XGBClassifier(eval_metric='logloss')
    }

    trained_models = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model

    return trained_models


# -------------------------------------------------------------------
# 3. EVALUATION
# -------------------------------------------------------------------
def evaluate_models(models, X_test, y_test):
    print("\n\n==================== MODEL PERFORMANCE ====================\n")
    for name, model in models.items():
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"\n--------------- {name} ---------------")
        print(f"Accuracy: {acc*100:.2f}%")
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
        print("Classification Report:\n", classification_report(y_test, y_pred))


# -------------------------------------------------------------------
# 4. PREDICT NEW DATA
# -------------------------------------------------------------------
def predict_new(model, scaler, input_data):
    input_arr = np.array(input_data).reshape(1, -1)
    input_scaled = scaler.transform(input_arr)
    pred = model.predict(input_scaled)

    return pred[0]


# -------------------------------------------------------------------
# MAIN PROGRAM
# -------------------------------------------------------------------
if __name__ == "__main__":

    print("Choose dataset: diabetes / heart / breast_cancer")
    dataset_name = input("Enter dataset name: ")

    # 1. Load Dataset
    X, y = load_dataset(dataset_name)

    # 2. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. Feature Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 4. Train Models
    models = train_models(X_train, y_train)

    # 5. Evaluate Models
    evaluate_models(models, X_test, y_test)

    # 6. Predict New Sample
    ask = input("\nDo you want to test a new input? (yes/no): ").lower()

    if ask == "yes":
        print("\nEnter values separated by comma (in same order as dataset features):")
        print(list(X.columns))

        raw = input("Input: ")
        values = list(map(float, raw.split(",")))

        model_name = input("Choose model: Logistic Regression / SVM / Random Forest / XGBoost\n")
        model = models[model_name]

        result = predict_new(model, scaler, values)

        print("\nPrediction:", "Disease Present" if result == 1 else "No Disease")

    print("\nDone ✔")
