# ML Assignment 2 - Bank Marketing Classification

## 1. Project Overview
This project implements six different machine learning models to predict whether a client will subscribe to a term deposit based on the Bank Marketing dataset.

## 2. Dataset Description
- **Source:** UCI Machine Learning Repository (Bank Marketing)
- **Instances:** 41,188
- **Features:** 20 (Input features including age, job, marital status, education, etc.)
- **Target:** 'y' (Binary: Yes/No for subscription)

## 3. Mandatory Preprocessing Steps
- **Label Encoding:** Converted categorical text data into numerical format for model compatibility.
- **Feature Scaling:** Applied `StandardScaler` to normalize feature distributions for models like KNN and Logistic Regression.
- **Data Splitting:** 80% Training and 20% Testing split to ensure robust evaluation.

## 4. Model Comparison Table
| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Logistic Regression | 0.9068 | 0.9272 | 0.8953 | 0.9068 | 0.8978 | 0.4696 |
| Decision Tree | 0.8849 | 0.7282 | 0.8877 | 0.8849 | 0.8863 | 0.4459 |
| KNN | 0.8990 | 0.8654 | 0.8879 | 0.8990 | 0.8916 | 0.4394 |
| Naive Bayes | 0.8484 | 0.8595 | 0.8858 | 0.8484 | 0.8627 | 0.4207 |
| Random Forest | 0.9130 | 0.9433 | 0.9065 | 0.9130 | 0.9088 | 0.5349 |
| **XGBoost** | **0.9143** | **0.9453** | **0.9092** | **0.9143** | **0.9112** | **0.5495** |



## 5. Observations on Model Performance (3 Marks)
| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Solid baseline performance with **0.9068** accuracy; works well but assumes linear relationships between banking features. |
| **Decision Tree** | Faster to train but showed the lowest AUC (**0.7282**), indicating it is less effective at separating the classes than ensemble methods. |
| **kNN** | Performance is stable (**0.8990** accuracy) but prediction speed is slower due to distance calculations across 20 features. |
| **Naive Bayes** | Lowest accuracy at **0.8484**. Its assumption of feature independence likely hinders performance on this complex socio-economic dataset. |
| **Random Forest** | Strong ensemble performer with **0.9130** accuracy; effectively reduced overfitting through multiple tree bagging. |
| **XGBoost** | **Best Overall Performer.** Highest Accuracy (**0.9143**) and AUC (**0.9453**), demonstrating the best ability to handle complex patterns. |

## 6. How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Run the app: `streamlit run app.py`
3. View the live version here: [https://mlassignment2-irhu8kzrmqwxs6hhkdwmx7.streamlit.app/]
