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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score

st.set_page_config(page_title="ML Model Evaluator", layout="wide")

st.title("📊 ML Assignment 2: Classification Model Comparison")
st.markdown("This app trains 6 models on an uploaded dataset and compares their performance.")

# 1. Sidebar - File Upload
st.sidebar.header("1. Upload Data")
uploaded_file = st.sidebar.file_saver = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview", df.head())
    
    # Simple Preprocessing (Logic from our Lab)
    target_col = st.selectbox("Select Target Column", df.columns, index=len(df.columns)-1)
    
    if st.button("Train Models"):
        X = df.drop(target_col, axis=1)
        y = df[target_col]
        
        # Basic Encoding for Categorical
        le = LabelEncoder()
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = le.fit_transform(X[col])
        
        # Split and Scale
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Models Dictionary
        models = {
            "Logistic Regression": LogisticRegression(),
            "Decision Tree": DecisionTreeClassifier(),
            "KNN": KNeighborsClassifier(),
            "Naive Bayes": GaussianNB(),
            "Random Forest": RandomForestClassifier(),
            "XGBoost": XGBClassifier()
        }
        
        results = []
        
        with st.spinner('Training 6 models...'):
            for name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                # Metrics
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, average='weighted')
                rec = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')
                mcc = matthews_corrcoef(y_test, y_pred)
                
                results.append([name, acc, prec, rec, f1, mcc])
        
        # Display Results
        res_df = pd.DataFrame(results, columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1', 'MCC'])
        st.write("### Model Comparison Results")
        st.dataframe(res_df.style.highlight_max(axis=0))
        
        # Mandatory UI Element: Bar Chart of Accuracy
        st.write("### Accuracy Comparison Chart")
        st.bar_chart(res_df.set_index('Model')['Accuracy'])
else:
    st.info("Awaiting CSV file upload. Please upload a dataset to begin.")
