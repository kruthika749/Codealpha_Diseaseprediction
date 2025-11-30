# Streamlit Disease Prediction App
# --------------------------------
# - Loads datasets (Diabetes, Heart Disease, Breast Cancer) from built-ins or uploaded CSV
# - Preprocesses (optional scaling)
# - Trains Logistic Regression, SVM, Random Forest, XGBoost
# - Shows evaluation metrics, confusion matrix, ROC
# - Allows single-sample prediction and model download
#  -typing the issues and predicting the disease
# Save this file as streamlit_disease_predictor.py and run:
#    streamlit run streamlit_disease_predictor.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import joblib
import base64

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# XGBoost is optional; app will handle if not installed
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

st.set_page_config(page_title="Disease Predictor", layout="wide")
# 5) Symptom-Based Disease Prediction
# -------------------------------------------------------------
st.markdown("---")
st.header("🩺 Symptom-Based Quick Disease Prediction")

st.write("Enter symptoms (comma-separated) and get a quick rule-based prediction. "
         "This is separate from the ML model training above.")

symptom_input = st.text_input(
    "Enter symptoms:",
    placeholder="e.g., fever, cough, headache, nausea"
)

def predict_from_symptoms(symptoms):
    symptoms = [s.strip().lower() for s in symptoms]

    # --- Flu / Viral Infections ---
    if {"fever", "cough", "headache"} & set(symptoms):
        return "Flu / Viral Infection"

    # --- Dengue ---
    if ("fever" in symptoms and "headache" in symptoms and "joint pain" in symptoms) or \
       ("fever" in symptoms and "rash" in symptoms):
        return "Dengue (possible)"

    # --- Malaria ---
    if ("fever" in symptoms and "chills" in symptoms) or \
       ("fever" in symptoms and "sweating" in symptoms):
        return "Malaria (possible)"

    # --- Typhoid ---
    if "fever" in symptoms and "headache" in symptoms and "stomach pain" in symptoms:
        return "Typhoid (possible)"

    # --- Food Poisoning ---
    if "vomiting" in symptoms and "diarrhea" in symptoms:
        return "Food Poisoning"

    # --- Migraine ---
    if "headache" in symptoms and ("nausea" in symptoms or "sensitivity to light" in symptoms):
        return "Migraine"

    # --- COVID ---
    if "cough" in symptoms and "fever" in symptoms and "loss of smell" in symptoms:
        return "COVID-19 (possible)"

    # --- Heart Disease ---
    if "chest pain" in symptoms or ("breathlessness" in symptoms and "fatigue" in symptoms):
        return "Heart Disease (possible)"

    # --- Stomach Infection ---
    if "stomach pain" in symptoms and ("vomiting" in symptoms or "nausea" in symptoms):
        return "Stomach Infection"

    # --- Appendicitis ---
    if "stomach pain" in symptoms and "fever" in symptoms and "loss of appetite" in symptoms:
        return "Appendicitis (possible)"

    # --- Kidney stones ---
    if "back pain" in symptoms and "vomiting" in symptoms:
        return "Kidney Stones (possible)"
    
    # Default
    return "Unknown — need more specific symptoms"

    
    return "Unknown — need more symptoms"

if st.button("Predict Disease from Symptoms"):
    if not symptom_input.strip():
        st.error("Please enter symptoms first.")
    else:
        user_symptoms = symptom_input.split(",")
        prediction = predict_from_symptoms(user_symptoms)
        st.success(f"Predicted Disease: **{prediction}**")



# -----------------------------
# Helpers
# -----------------------------
@st.cache_data
def load_breast_cancer():
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name='target')
    return pd.concat([X, y], axis=1)

@st.cache_data
def load_diabetes_csv(path="diabetes.csv"):
    return pd.read_csv(path)

@st.cache_data
def load_heart_csv(path="heart.csv"):
    return pd.read_csv(path)


def train_models(X_train, y_train, random_state=42):
    models = {}

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=random_state)
    lr.fit(X_train, y_train)
    models['Logistic Regression'] = lr

    # SVM (probability True for ROC)
    svm = SVC(kernel='linear', probability=True, random_state=random_state)
    svm.fit(X_train, y_train)
    models['SVM'] = svm

    # Random Forest
    rf = RandomForestClassifier(n_estimators=200, random_state=random_state)
    rf.fit(X_train, y_train)
    models['Random Forest'] = rf

    # XGBoost if available
    if HAS_XGB:
        xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=random_state)
        xgb.fit(X_train, y_train)
        models['XGBoost'] = xgb

    return models


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        # fallback
        try:
            scores = model.decision_function(X_test)
            proba = (scores - scores.min()) / (scores.max() - scores.min())
        except Exception:
            proba = None

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, output_dict=False)

    fpr, tpr, thresholds, roc_auc = (None, None, None, None)
    if proba is not None:
        fpr, tpr, thresholds = roc_curve(y_test, proba)
        roc_auc = auc(fpr, tpr)

    return {
        'accuracy': acc,
        'confusion_matrix': cm,
        'classification_report': cr,
        'fpr': fpr,
        'tpr': tpr,
        'roc_auc': roc_auc,
    }


def plot_confusion_matrix(cm, labels=None):
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest')
    ax.figure.colorbar(im, ax=ax)
    if labels is None:
        labels = [0, 1]
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
           xticklabels=labels, yticklabels=labels,
           ylabel='True label', xlabel='Predicted label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return fig


def plot_roc(fpr, tpr, roc_auc, title="ROC Curve"):
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
    ax.plot([0, 1], [0, 1], linestyle='--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc='lower right')
    return fig


def get_table_download_link(obj, filename, text):
    """Generates a link to download the given object as a file
    obj: bytes or text
    filename: name for download
    text: clickable text to display
    """
    if isinstance(obj, bytes):
        b64 = base64.b64encode(obj).decode()
    else:
        b64 = base64.b64encode(obj.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">{text}</a>'
    return href

# -----------------------------
# UI
# -----------------------------
st.title("🩺 Disease Prediction — Streamlit App")
st.write(
    "Train and compare classification models (Logistic Regression, SVM, Random Forest, XGBoost) on medical datasets."
)

# Sidebar controls
st.sidebar.header("1) Dataset")
dataset_option = st.sidebar.selectbox(
    "Choose dataset source",
    ("Breast Cancer (sklearn)", "Upload CSV", "Use sample CSV files")
)

uploaded_file = None
sample_choice = None
if dataset_option == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file (must contain label column)", type=['csv'])
else:
    sample_choice = st.sidebar.selectbox("Or pick a sample CSV", ("diabetes.csv", "heart.csv"))

label_column = st.sidebar.text_input("Label column name (target)", value='target')
scale_features = st.sidebar.checkbox("Scale features (StandardScaler)", value=True)
random_state = st.sidebar.number_input("Random state (for reproducibility)", min_value=0, value=42, step=1)

st.sidebar.markdown("---")
st.sidebar.header("2) Models & Training")
train_button = st.sidebar.button("Train models")
show_models = st.sidebar.multiselect("Which models to show", ["Logistic Regression", "SVM", "Random Forest", "XGBoost"], default=["Logistic Regression", "Random Forest"])

st.sidebar.markdown("\n")

# Load dataframe
df = None
if dataset_option == "Breast Cancer (sklearn)":
    df = load_breast_cancer()
    if label_column == 'target':
        label_column = 'target'
elif dataset_option == "Upload CSV" and uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Couldn't read uploaded CSV: {e}")
elif dataset_option == "Use sample CSV files":
    # attempt to load local files
    try:
        if sample_choice == 'diabetes.csv':
            df = load_diabetes_csv('diabetes.csv')
        elif sample_choice == 'heart.csv':
            df = load_heart_csv('heart.csv')
    except FileNotFoundError:
        st.warning("Sample CSV not found locally. You can upload your CSV instead.")

if df is None:
    st.info("No dataset loaded yet. Choose an option in the sidebar or upload a CSV.")
    st.stop()

st.write("### Preview dataset")
st.dataframe(df.head())

# Ensure label exists
if label_column not in df.columns:
    st.error(f"Label column '{label_column}' not found in dataset columns: {list(df.columns)}")
    st.stop()

# Prepare X and y
X = df.drop(columns=[label_column])
y = df[label_column]

# Drop non-numeric columns automatically (simple approach)
non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
if non_numeric:
    st.warning(f"Dropping non-numeric columns: {non_numeric}")
    X = X.drop(columns=non_numeric)

# Train/test split
test_size = st.sidebar.slider("Test size (%)", 5, 50, 20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100.0, random_state=int(random_state))

# Scaling
scaler = None
if scale_features:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
else:
    X_train_scaled = X_train.values
    X_test_scaled = X_test.values

models = None
metrics = {}

if train_button:
    with st.spinner('Training models — this may take a few seconds...'):
        models = train_models(X_train_scaled, y_train, random_state=int(random_state))

    st.success('Training complete!')

    # Evaluate selected models
    st.write("## Model evaluation")
    cols = st.columns(2)
    for i, (name, model) in enumerate(models.items()):
        if name not in show_models:
            continue
        res = evaluate_model(model, X_test_scaled, y_test)
        metrics[name] = res

        with cols[i % 2]:
            st.subheader(name)
            st.write(f"**Accuracy:** {res['accuracy']*100:.2f}%")
            st.write("**Classification report:**")
            st.text(res['classification_report'])
            st.write("**Confusion matrix:**")
            fig_cm = plot_confusion_matrix(res['confusion_matrix'])
            st.pyplot(fig_cm)

            if res['roc_auc'] is not None:
                fig_roc = plot_roc(res['fpr'], res['tpr'], res['roc_auc'], title=f"ROC - {name}")
                st.pyplot(fig_roc)

            # Download model
            buf = io.BytesIO()
            joblib.dump(model, buf)
            buf.seek(0)
            b = buf.read()
            dl_link = get_table_download_link(b, f"{name.replace(' ', '_')}_model.joblib", f"Download {name} model")
            st.markdown(dl_link, unsafe_allow_html=True)

# Allow single-sample prediction if models exist
if models is not None:
    st.write("---")
    st.write("## Predict on a single sample")
    st.write("Enter values for each feature in the order shown below (numeric only).")

    cols = st.columns(2)
    features = list(X.columns)
    sample_values = []
    for i, feat in enumerate(features):
        # make two-column layout
        with cols[i % 2]:
            sample_values.append(st.text_input(f"{feat}", value="0"))

    model_choice = st.selectbox("Choose model for prediction", list(models.keys()))
    if st.button("Predict"):
        try:
            sample_floats = [float(x) for x in sample_values]
            sample_arr = np.array(sample_floats).reshape(1, -1)
            if scaler is not None:
                sample_arr = scaler.transform(sample_arr)
            model = models[model_choice]
            pred = model.predict(sample_arr)[0]
            proba = None
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(sample_arr)[0, 1]
            st.write(f"**Prediction:** {pred}")
            if proba is not None:
                st.write(f"**Probability (positive class):** {proba:.4f}")
        except Exception as e:
            st.error(f"Failed to predict: {e}")

st.write("\n---\nBuilt with ❤️  — Streamlit")



# End of file

