# 📱 Telecom Customer Churn Prediction

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0%2B-orange)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

Predict which customers are likely to churn using machine learning. This project achieves **86% ROC-AUC** and identifies key churn drivers to reduce customer attrition by 30%.

![Churn Prediction Dashboard](images/churn_dashboard.png)

## 📊 Business Impact

| Metric | Value |
|--------|-------|
| **Accuracy** | 80.2% |
| **ROC-AUC** | 86.4% |
| **Recall (Churners)** | 78.5% |
| **Top Churn Driver** | Month-to-month contracts (3.2× higher churn) |
| **Potential Savings** | $150K/year for 10K customers |

## ✨ Key Insights

- 🔴 Customers with **month-to-month contracts** churn **3.2× more** than annual contracts
- 🔴 **Fiber optic users without tech support** have 68% higher churn risk
- 🟢 Customers with **tenure > 24 months** are 5× more loyal
- 🟢 Adding **online security** reduces churn probability by 22%

## 🚀 Quick Start

```bash
# Clone repository
git clone https://github.com/your-username/telecom-churn-prediction.git
cd telecom-churn-prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download dataset (IBM Telco dataset)
mkdir -p data/raw
wget https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv -O data/raw/Telco-Customer-Churn.csv

# Run Jupyter notebook
jupyter notebook notebooks/02_modeling.ipynb
