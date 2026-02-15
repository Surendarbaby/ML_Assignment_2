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
st.title("ML Assignment 2: Classification Models")

# 1. Sidebar Upload [Requirement 4a]
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview", df.head())
    
    # Preprocessing
    target_col = st.selectbox("Select Target Column (Select 'y')", df.columns, index=len(df.columns)-1)
    
    # 2. Model Selection Dropdown [Requirement 4b]
    selected_model_name = st.selectbox("Select Model for Detailed Report", 
                                      ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Random Forest", "XGBoost"])

    if st.button("Run Evaluation"):
        X = df.drop(target_col, axis=1)
        y = df[target_col]
        
        # Encoding and Scaling
        le = LabelEncoder()
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = le.fit_transform(X[col])
        if y.dtype == 'object': y = le.fit_transform(y)
            
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Define Model Object
        models_dict = {
            "Logistic Regression": LogisticRegression(),
            "Decision Tree": DecisionTreeClassifier(),
            "KNN": KNeighborsClassifier(),
            "Naive Bayes": GaussianNB(),
            "Random Forest": RandomForestClassifier(),
            "XGBoost": XGBClassifier()
        }
        
        # Train and Show Specific Results
        model = models_dict[selected_model_name]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # 3. Display Metrics [Requirement 4c]
        st.write(f"### Results for {selected_model_name}")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
        col2.metric("F1 Score", f"{f1_score(y_test, y_pred, average='weighted'):.4f}")
        col3.metric("MCC", f"{matthews_corrcoef(y_test, y_pred):.4f}")
        col4.metric("Recall", f"{recall_score(y_test, y_pred, average='weighted'):.4f}")
        
        # 4. Classification Report [Requirement 4d]
        st.write("### Classification Report")
        st.text(classification_report(y_test, y_pred))

else:
    st.info("Please upload the banking.csv file to begin.")
