# ML Assignment 2 - Bank Marketing Classification

## 1. Project Overview
This project implements six different machine learning models to predict whether a client will subscribe to a term deposit based on the Bank Marketing dataset. 

## 2. Dataset Description
- **Source:** UCI Machine Learning Repository (Bank Marketing)
- **Instances:** 41,188
- **Features:** 20 (Input features including age, job, marital status, education, etc.)
- **Target:** 'y' (Binary: Yes/No for subscription)

## 3. Mandatory Preprocessing Steps
- **Label Encoding:** Converted categorical text data into numerical format.
- **Feature Scaling:** Applied `StandardScaler` to normalize feature distributions for models like KNN and Logistic Regression.
- **Data Splitting:** 80% Training and 20% Testing split.

## 4. Model Comparison Table
| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9067 | 0.9272 | 0.6436 | 0.4156 | 0.5051 | 0.4696 |
| Decision Tree | 0.8843 | 0.7241 | 0.4949 | 0.5164 | 0.5054 | 0.4401 |
| KNN | 0.8993 | 0.8652 | 0.5843 | 0.4188 | 0.4879 | 0.4412 |
| Naive Bayes | 0.8483 | 0.8594 | 0.3981 | 0.6341 | 0.4891 | 0.4207 |
| Random Forest | 0.9109 | 0.9438 | 0.6373 | 0.5143 | 0.5692 | 0.5239 |
| XGBoost | 0.9143 | 0.9453 | 0.6483 | 0.5493 | 0.5947 | 0.5495 |

## 5. Observations
1. **Best Model:** XGBoost achieved the highest Accuracy (91.43%) and AUC (0.945), making it the most reliable model for this dataset.
2. **Recall vs Precision:** Naive Bayes showed the highest Recall (0.634), which is useful if the bank wants to minimize missing potential customers, though it has more false positives.
3. **Complexity:** Ensemble methods (Random Forest/XGBoost) significantly outperformed linear and distance-based models.

## 6. How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Run the app: `streamlit run app.py`
