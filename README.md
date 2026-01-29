 🏦 Credit Risk Assessment ML Model

An AI-powered credit risk classification system built with Python following RBI (Reserve Bank of India) lending guidelines.

![Python](https://img.shields.io/badge/Python-3.14-blue)
![ML](https://img.shields.io/badge/ML-Decision%20Tree-green)
![Status](https://img.shields.io/badge/Status-Complete-success)

## 📊 Project Overview

This machine learning model automates credit risk evaluation for loan applications by analyzing key financial indicators and predicting risk categories (Low/Medium/High).

🎯 Key Features

- ✅ **Automated Risk Classification** - Predicts Low/Medium/High risk categories
- ✅ **RBI-Compliant** - Follows Reserve Bank of India credit assessment principles  
- ✅ **8 Financial Indicators** - Comprehensive data analysis
- ✅ **75% Accuracy** - Validated prediction performance
- ✅ **Instant Recommendations** - Automated loan approval/rejection guidance

 📈 Financial Indicators Analyzed

| Feature | Description |
|---------|-------------|
| 👤 Age | Customer's age (22-60 years) |
| 💰 Monthly Income | Salary/income in INR |
| 💼 Employment Type | Salaried (0) or Self-Employed (1) |
| 🏠 Loan Amount | Requested loan amount in INR |
| 💳 EMI Amount | Monthly installment amount |
| 📋 Existing Loans | Number of current loans (0-3) |
| ⚠️ Missed Payments | Payment defaults in last 12 months |
| 📊 Credit Utilization | Percentage of credit limit used |

🛠️ Tech Stack

- Language: Python 3.14
- Libraries:
  - `pandas` - Data manipulation and analysis
  - `scikit-learn` - Machine learning algorithms
- Algorithm: Decision Tree Classifier
- IDE: Visual Studio Code

 📊 Model Performance
```
Training Data: 16 customers
Testing Data: 4 customers
Accuracy: 75%
Features: 8
Risk Categories: 3 (Low/Medium/High)
```

🚀 How to Run

 Prerequisites
```bash
Python 3.x installed
pip package manager
```

 Installation

1. Clone the repository
```bash
git clone https://github.com/YOUR-USERNAME/credit-risk-ml-model.git
cd credit-risk-ml-model
```

2. Install required libraries
```bash
pip install pandas scikit-learn
```

3. Run the model
```bash
python credit_risk_model.py
```

💡 Sample Output
```
==================================================
CREDIT RISK MODEL - RBI Style
==================================================

New Customer Details:
   Age: 32 years
   Monthly Income: Rs.45,000
   Employment: Salaried
   Loan Amount: Rs.2,50,000
   
PREDICTION RESULT
Credit Risk Category: MEDIUM RISK
RECOMMENDATION: Proceed with caution - Medium risk customer

==================================================
Credit Risk Model Complete!
==================================================
```

 🎓 What I Learned

- Machine Learning classification techniques
- Financial risk assessment methodology  
- Data preprocessing and feature engineering
- Model training, testing, and evaluation
- Practical AI application in banking sector
- Python programming with pandas and scikit-learn

🔮 Future Enhancements

- Increase training dataset size for improved accuracy
- Implement additional ML algorithms (Random Forest, XGBoost)
- Create web interface for user-friendly input
-  Add data visualization dashboards
-  Deploy model as REST API

 📧 Contact

Rupali Kumari
- LinkedIn: www.linkedin.com/in/rupali-singh-hr27
- Email: rups2122@gmail.com
  

---

⭐ If you found this project interesting, please give it a star!



📝 **License:** MIT License - feel free to use this for learning purposes
