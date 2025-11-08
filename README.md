# Credit_Card_Default_Predictor
# 💳 Credit Card Default Prediction

A machine learning project that predicts whether a customer will default on their next credit card payment using demographic, financial, and repayment history data.  
This project demonstrates a full data science pipeline — from problem understanding to model deployment using Streamlit.

---

## 1. Problem Definition

### Goal
Predict whether a credit card customer will **default on their next month’s payment**.

### 🏦 Business Objective
Financial institutions can use this model to:
- Identify **high-risk customers** early
- **Reduce financial losses**
- Improve **credit policy and customer management**

### Problem Type
**Supervised Binary Classification**  
Target variable → `default.payment.next.month`  
- `1` = Customer will default  
- `0` = Customer will pay on time

---

## 2. Understanding the Business Problem

**Key Business Questions**
- Which customers are most likely to default next month?
- What behavioral and financial factors influence default risk?
- How can this model support better lending decisions?

**Business Impact**
- Early detection → fewer defaults  
- Improved risk management  
- Data-driven credit limit and interest rate decisions  

---

## 3. Data Overview

**Dataset:** `credit_card_data.csv`  
**Records:** 30,000  
**Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)

### Features
| Type | Examples |
|------|-----------|
| **Demographic** | SEX, EDUCATION, MARRIAGE, AGE |
| **Financial** | LIMIT_BAL, BILL_AMT1–6, PAY_AMT1–6 |
| **Behavioral (Repayment)** | PAY_0–PAY_6 |
| **Target** | `default.payment.next.month` |

---

## 4. Data Cleaning & Preprocessing

Steps:
- Removed irrelevant column `ID`
- Replaced invalid repayment values: `-2, -1 → 0`  
- Filled missing values (numerical → median, categorical → mode)
- Standardized numeric features using **StandardScaler**
- Addressed class imbalance using **SMOTEENN**

---

## 5. Platform & Tools

**Language:** Python 3.10+  
**Libraries Used:**
- `pandas`, `numpy`, `seaborn`, `matplotlib`
- `scikit-learn`, `xgboost`, `lightgbm`, `imblearn`
- `joblib`, `streamlit`

**Environment:**
- Jupyter Notebook / VS Code for modeling  
- Streamlit for interactive deployment  

---

## 6. Feature Engineering

Created new financial behavior indicators:
| Feature | Description |
|----------|--------------|
| `AVG_BILL_AMT` | Average bill amount over 6 months |
| `AVG_PAY_AMT` | Average payment amount over 6 months |
| `PAY_RATIO` | Average payment ÷ average bill |
| `UTILIZATION` | Current bill ÷ credit limit |
| `TOTAL_PAY_AMT` | Total payment amount over 6 months |
| `TOTAL_BILL_AMT` | Total bill amount over 6 months |

✅ These derived features improved prediction accuracy.

---

## 7. Exploratory Data Analysis (EDA)

**Visuals & Insights**
- **Distribution plots:** Credit limit, age  
- **Correlation heatmap:** Relation between payment behavior and default  
- **Boxplots:** Detect outliers in billing and payments  
- **Default rate analysis:** Higher delay → higher chance of default  

Defaulters often have:
- Higher **utilization ratio**
- Frequent **late payments**
- **Lower average payments** compared to bills

---

## 8. Model Development

### Models Trained
1. **LightGBM**
2. **XGBoost**
3. **Random Forest**
4. **Stacking Ensemble** (final model)

### Techniques
- Hyperparameter tuning using `RandomizedSearchCV`
- Resampling with `SMOTEENN`
- Stacking ensemble using Logistic Regression as final estimator

---

## 9. Model Evaluation

| Metric | Value |
|---------|--------|
| **Optimal Threshold** | 0.90 |
| **F1 Score (Best)** | 0.525 |
| **ROC-AUC** | 0.760 |
| **Accuracy** | 0.80 |

**Confusion Matrix:**
| Actual | Predicted 0 | Predicted 1 |
|---------|--------------|-------------|
| 0 | 86% | 14% |
| 1 | 49% | 51% |

**Interpretation**
- Good overall discrimination (AUC = 0.76)
- Balanced trade-off between recall and precision
- Stacking model provided best stability and accuracy

---

## 10. Model Saving

Trained models and artifacts were serialized using `joblib`:
