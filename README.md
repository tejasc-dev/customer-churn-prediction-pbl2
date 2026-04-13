# Customer Churn Prediction using Ensemble Learning

## 📌 Project Overview
This project focuses on predicting customer churn in the telecom industry using machine learning techniques. Customer churn refers to the loss of clients or subscribers, which directly impacts business revenue.

The system leverages **ensemble learning models** to improve prediction accuracy and identify high-risk customers, enabling proactive retention strategies.

---

## 🎯 Objectives
- Develop a robust churn prediction model
- Compare multiple machine learning algorithms
- Improve performance using ensemble techniques
- Optimize recall for churn class (important for business)
- Provide interpretable insights using SHAP

---

## 📊 Dataset
- **Dataset Used:** IBM Telco Customer Churn Dataset  
- **Size:** 7043 records  
- **Features Include:**
  - Customer demographics
  - Service usage
  - Billing information
  - Tenure

- **Target Variable:**  
  `Churn (Yes/No)`

---

## ⚙️ Methodology

### 🔹 Data Preprocessing
- Handling missing values
- Encoding categorical variables
- Feature scaling
- Class imbalance handled using **SMOTE**

### 🔹 Models Used
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM
- CatBoost

### 🔹 Ensemble Technique
- **Stacking Ensemble Model**
- Combines predictions of multiple base learners
- Improves generalization and robustness

### 🔹 Optimization
- Hyperparameter tuning using **Optuna**

### 🔹 Interpretability
- Model explanations using **SHAP (SHapley Additive Explanations)**

---

## 📈 Results

| Model              | Performance |
|------------------|------------|
| Logistic Regression | 0.74 |
| Random Forest       | 0.78 |
| XGBoost             | 0.82 |
| **Stacking Ensemble** | **0.83 (Best)** |

- **ROC-AUC Score:** 0.83  
- **F1 Score (Churn):** 0.62  
- Improved recall for churn class

---

## 🌐 Live Project
🔗 https://tejasc-dev.github.io/customer-churn-prediction-pbl2/

---

## 🧠 Key Insights
- Ensemble models outperform individual models
- Feature interactions significantly impact churn
- Recall optimization is critical in churn prediction problems

---

## 🔮 Future Work
- Deploy model as a web application
- Integrate real-time prediction system
- Explore deep learning models
- Improve interpretability dashboards

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
This project is for academic purposes.
