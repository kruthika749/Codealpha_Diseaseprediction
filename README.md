# Codealpha_Diseaseprediction
A complete and interactive Disease Prediction System powered by machine learning and simple rule-based medical logic. This Streamlit application helps users explore medical datasets, train ML classification models, evaluate them visually, and even get quick predictions from symptoms typed in manually. The app is flexible enough for beginners, students, and professionals who want a hands-on demonstration of healthcare-focused classification models.
⭐ Overview
This project lets you:
✔️ Load Data
*Use built-in Breast Cancer dataset
*Upload your own CSV medical dataset
*Use provided sample datasets: diabetes.csv or heart.csv
✔️ Train ML Models
*The app automatically trains multiple classification models:
*Logistic Regression
*Support Vector Machine (SVM)
*Random Forest Classifier
*XGBoost (optional, if installed)
*Users can scale features using StandardScaler and control test size and random state.
✔️ Evaluate Models Visually
*For each selected classifier, the app displays:
*Accuracy score
*Classification report
*Confusion matrix heatmap
*ROC Curve with AUC score
*Instant model download (as .joblib)
✔️ Predict on New Samples
*after training, users can enter values for all features and instantly generate:
*Model prediction
*Prediction probability (when available)
✔️ Instant Symptom-Based Disease Prediction
*A unique feature of this app is a rule-based medical symptom checker.
🚀 How to Run the App
1️⃣ Install dependencies
pip install streamlit pandas numpy scikit-learn matplotlib joblib
pip install xgboost   # optional

2️⃣ Save your file as:
streamlit_disease_predictor.py

3️⃣ Run the app
streamlit run streamlit_disease_predictor.py
