# 📊 Customer Churn Prediction using Machine Learning

## 👤 Student Details
- **Name:** Tejas  
- **Registration No:** 2427030729  
- **Department:** Computer Science and Engineering  
- **Project Guide:** Varda Pareek  
- **Course:** PBL 2  

---

## 🔍 Project Overview

Customer churn is a critical challenge in the telecom industry, where losing customers directly impacts revenue.  
This project develops an end-to-end machine learning pipeline to predict whether a customer is likely to churn.

The system compares multiple classification algorithms and evaluates their performance using standard metrics.

---

## 🎯 Objectives

- Perform data preprocessing and feature engineering  
- Implement and compare 10 classification algorithms  
- Evaluate models using Accuracy, Precision, Recall, and F1-score  
- Identify the most effective model for churn prediction  
- Provide a reproducible live implementation  

---

## 🗂 Dataset

- **Source:** IBM Telco Customer Churn Dataset  
- **Records:** ~7043 customers  
- **Type:** Binary classification (Churn vs No Churn)  

### Key Features

- Customer demographics  
- Service subscriptions  
- Billing information  
- Contract details  

---

## ⚙️ Methodology

The project follows an end-to-end machine learning workflow:

1. Data Loading  
2. Data Cleaning  
3. Feature Encoding  
4. Train-Test Split (80–20)  
5. Model Training  
6. Performance Evaluation  
7. Model Comparison  

---

## 🤖 Models Implemented

- Logistic Regression  
- K-Nearest Neighbors (KNN)  
- Support Vector Machine (SVM)  
- Decision Tree  
- Random Forest  
- Gradient Boosting  
- AdaBoost  
- Naive Bayes  
- Extra Trees  
- XGBoost  

---

## 📈 Evaluation Metrics

Models were evaluated using:

- Accuracy  
- Precision  
- Recall  
- F1-score  
- Confusion Matrix  

> Special emphasis was placed on **Recall for churn class**, as missing churn customers leads to revenue loss.

---

## 🏆 Results Summary

| Model | Accuracy | Recall (Churn) |
|------|----------|----------------|
| Gradient Boosting | **79.46%** | 0.50 |
| Random Forest | 79.46% | 0.47 |
| XGBoost | 77.83% | **0.55** |

### ✅ Best Balanced Model
**Gradient Boosting** achieved the best trade-off between accuracy and churn detection.

### 🔍 Highest Churn Detection
**XGBoost** achieved the highest recall for churn customers.

---

## 🚀 Live Implementation

🔗 **GitHub Pages:**  
https://tejas-dev.github.io/customer-churn-prediction-pbl2/

🔗 **Google Colab Notebook:**  
https://colab.research.google.com/drive/1XXfMyim-oR1xj877Q8vOSwkpK06Lu5EP?usp=sharing

---

## 🛠️ Technologies Used

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- XGBoost  
- Matplotlib / Seaborn  
- Google Colab  
- GitHub Pages  

---

## 📚 References

1. IBM Telco Customer Churn Dataset (Kaggle)  
2. Scikit-learn Documentation  
3. XGBoost Documentation  
4. Telecom churn prediction research papers  

---

## ✅ Project Status

- ✔ Data preprocessing completed  
- ✔ 10 models implemented  
- ✔ Performance comparison done  
- ✔ Live deployment completed  

---

## 🙏 Acknowledgment

I would like to thank **Varda Pareek** for guidance and support throughout the completion of this project.

---
