# Customer Churn Prediction using Ensemble Learning

## 📌 Project Overview
This project focuses on predicting customer churn in the telecom industry using machine learning techniques. Customer churn refers to the loss of clients or subscribers, which directly impacts business revenue.

The system leverages **ensemble learning models** to improve prediction accuracy and identify high-risk customers, enabling proactive retention strategies.

---

## 🚀 Open in Google Colab

👉 Click below to run the project instantly:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/19QIrOr28jKceanpbXorOOCThPQd5FYaQ#scrollTo=FWH1VQXiu_r_)

---

## 🎯 Objectives
- Develop a robust churn prediction model  
- Compare multiple machine learning algorithms  
- Improve performance using ensemble techniques  
- Optimize recall for churn class  
- Provide interpretability using SHAP  

---

## 📊 Dataset
- **Dataset Used:** IBM Telco Customer Churn Dataset  
- **Size:** 7043 records  
- **Target Variable:** `Churn (Yes/No)`  

Includes:
- Demographics  
- Service usage  
- Billing information  
- Tenure  

---

## ⚙️ Methodology

### 🔹 Data Preprocessing
- Missing value handling  
- Encoding categorical variables  
- Feature scaling  
- SMOTE for class imbalance  

### 🔹 Models Used
- Logistic Regression  
- Decision Tree  
- Random Forest  
- XGBoost  
- LightGBM  
- CatBoost  

### 🔹 Ensemble Model
- **Stacking Ensemble**
- Combines multiple base learners  
- Improves robustness and accuracy  

### 🔹 Optimization
- Hyperparameter tuning using **Optuna**

### 🔹 Interpretability
- SHAP for feature importance analysis  

---

## 📈 Results

| Model              | Score |
|------------------|------|
| Logistic Regression | 0.74 |
| Random Forest       | 0.78 |
| XGBoost             | 0.82 |
| **Stacking Ensemble** | **0.83** |

- **ROC-AUC:** 0.83  
- **F1 Score:** 0.62  

---

## 🌐 Live Project
🔗 https://tejasc-dev.github.io/customer-churn-prediction-pbl2/

---

## 🧠 Key Insights
- Ensemble models outperform individual models  
- Feature interactions are crucial  
- Recall optimization is key in churn prediction  

---

## 🔮 Future Work
- Deploy as a web app  
- Real-time prediction system  
- Improve visualization dashboard  
- Explore deep learning  

---

## 👨‍💻 Author
**Tejas Chaudhary**  
Reg No: 2427030729  
Manipal University Jaipur  

---

## 🎓 Guide
**Dr. Varda Pareek**

---

## 📜 License
Academic project – for educational use only.
