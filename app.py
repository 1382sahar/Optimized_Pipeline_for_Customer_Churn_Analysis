# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import shap
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="ML Dashboard", layout="wide")
st.title("📊 Machine Learning Dashboard - Telco Churn")

# Sidebar
st.sidebar.header("Upload & Settings")
file = st.sidebar.file_uploader("Upload your CSV", type=["csv"])

if file:
    df = pd.read_csv(file)
    st.subheader("Data Preview")
    st.write(df.head())

    # Preprocessing
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)

    for col in df.select_dtypes(include='object').columns:
        if col != 'Churn':
            df[col] = LabelEncoder().fit_transform(df[col])

    X = df.drop('Churn', axis=1)
    y = LabelEncoder().fit_transform(df['Churn'])

    st.subheader("Feature Selection")
    selector = SelectKBest(score_func=f_classif, k=10)
    X_selected = selector.fit_transform(X, y)
    selected_cols = X.columns[selector.get_support()]
    X = pd.DataFrame(X_selected, columns=selected_cols)
    st.write(X.head())

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    st.subheader("Model Training & Evaluation")
    models = {
        'Logistic Regression': LogisticRegression(),
        'Random Forest': RandomForestClassifier(),
        'XGBoost': XGBClassifier(verbosity=0),
        'LightGBM': LGBMClassifier(),
        'CatBoost': CatBoostClassifier(verbose=0)
    }

    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probas = model.predict_proba(X_test)[:,1]
        results.append({
            'Model': name,
            'Accuracy': accuracy_score(y_test, preds),
            'Precision': precision_score(y_test, preds),
            'Recall': recall_score(y_test, preds),
            'F1 Score': f1_score(y_test, preds),
            'AUC': roc_auc_score(y_test, probas)
        })

    st.write(pd.DataFrame(results).sort_values("F1 Score", ascending=False))

    # Visualization
    st.subheader("ROC Curves")
    plt.figure(figsize=(10,6))
    for name, model in models.items():
        fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:,1])
        plt.plot(fpr, tpr, label=name)
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    st.pyplot(plt)

    # SHAP Analysis
    st.subheader("SHAP Feature Importance (CatBoost)")
    explainer = shap.TreeExplainer(models['CatBoost'])
    shap_values = explainer.shap_values(X_test)
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    st.pyplot(fig)

    # Prediction on new data
    st.subheader("Predict on New Data")
    predict_file = st.file_uploader("Upload new data for prediction", type=["csv"], key="predict")
    if predict_file:
        new_data = pd.read_csv(predict_file)
        new_data = new_data[selected_cols]  # use selected features only
        prediction = models['CatBoost'].predict(new_data)
        st.write("Predictions:", prediction)
        output_df = new_data.copy()
        output_df['Prediction'] = prediction
        st.download_button("Download Predictions", output_df.to_csv(index=False), "predictions.csv", "text/csv")
