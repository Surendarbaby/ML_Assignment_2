import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score, classification_report

st.set_page_config(page_title="ML Evaluator", layout="wide")
st.title("📊 ML Assignment 2: Classification Models")

uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### 1. Dataset Preview", df.head())
    
    target_col = st.selectbox("Select Target Column (Select 'y')", df.columns, index=len(df.columns)-1)
    selected_model_name = st.selectbox("Select Model for Detailed Report", 
                                      ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Random Forest", "XGBoost"])

    if st.button("Run Evaluation"):
        X = df.drop(target_col, axis=1)
        y = df[target_col]
        
        le = LabelEncoder()
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = le.fit_transform(X[col])
        if y.dtype == 'object': y = le.fit_transform(y)
            
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        models_dict = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Decision Tree": DecisionTreeClassifier(),
            "KNN": KNeighborsClassifier(),
            "Naive Bayes": GaussianNB(),
            "Random Forest": RandomForestClassifier(),
            "XGBoost": XGBClassifier()
        }
        
        # --- ALL MODELS COMPARISON [Requirement 4c] ---
        st.write("### 2. Mandatory Model Comparison Table")
        all_results = []
        for name, m in models_dict.items():
            m.fit(X_train, y_train)
            pred = m.predict(X_test)
            # AUC requires probabilities
            prob = m.predict_proba(X_test)[:, 1] if hasattr(m, "predict_proba") else pred
            
            all_results.append({
                "Model": name,
                "Accuracy": accuracy_score(y_test, pred),
                "AUC": roc_auc_score(y_test, prob),
                "Precision": precision_score(y_test, pred, average='weighted'),
                "Recall": recall_score(y_test, pred, average='weighted'),
                "F1 Score": f1_score(y_test, pred, average='weighted'),
                "MCC": matthews_corrcoef(y_test, pred)
            })
        
        res_df = pd.DataFrame(all_results)
        st.dataframe(res_df.style.format(precision=4)) 

        # --- INDIVIDUAL REPORT [Requirement 4d] ---
        st.write(f"### 3. Detailed Report: {selected_model_name}")
        # Re-fit the specific selected model to show its report
        specific_model = models_dict[selected_model_name]
        specific_model.fit(X_train, y_train)
        st.text(classification_report(y_test, specific_model.predict(X_test)))

        st.write("### 4. Accuracy Comparison Chart")
        st.bar_chart(res_df.set_index('Model')['Accuracy'])

else:
    st.info("Please upload the banking.csv file to begin.")
