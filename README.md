# Bank Customer Churn Prediction 📊🏦

This project analyzes bank customer data to identify patterns behind customer churn and builds machine learning models to predict which customers are likely to leave. The goal is to help businesses take data-driven actions for improving customer retention.

---

## 🔍 Problem Statement

Customer churn is a major challenge for banks, as acquiring new customers is significantly more expensive than retaining existing ones.  
This project aims to:
- Analyze customer behavior and demographics
- Identify key factors influencing churn
- Build predictive models to detect at-risk customers

---

## 📂 Dataset

The dataset contains **10,000 customer records** with the following features:

- **Customer Demographics:** Age, Gender, Geography  
- **Financial Information:** Credit Score, Balance, Estimated Salary  
- **Account Details:** Tenure, Number of Products, Credit Card Ownership, Active Membership  
- **Target Variable:** `Exited` (1 = Churned, 0 = Not Churned)

---

## 🧪 Exploratory Data Analysis (EDA)

Key insights from the data:
- Churned customers tend to be **older on average**
- Customers with **higher balances** show a greater likelihood of churn
- **Inactive members** are significantly more likely to leave
- Geography and number of products have a noticeable impact on churn behavior

Visualizations and statistical summaries were used to understand:
- Distribution of numeric features
- Churn vs non-churn comparisons
- Correlation between variables

---

## ⚙️ Feature Engineering

To improve model performance, additional features were created:
- **Tenure-to-Age ratio**
- Binned variables using quantiles for:
  - Credit Score
  - Age
  - Balance
  - Estimated Salary

Categorical features were encoded using **One-Hot Encoding**, and numerical features were scaled using a **robust scaling technique** to handle outliers.

---

## 🤖 Machine Learning Models

Multiple models were trained and evaluated using cross-validation:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- Gradient Boosting
- LightGBM

### 🏆 Best Performing Models:
- **Gradient Boosting**
- **LightGBM**

Both models achieved approximately **86% accuracy**, with strong recall for churned customers, making them suitable for identifying high-risk customers.

---

## 📈 Model Evaluation

Evaluation metrics used:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC-AUC

The final models demonstrated:
- High overall accuracy
- Good ability to detect churned customers
- Balanced performance between false positives and false negatives

---

## 🛠️ Technologies Used

- **Programming:** Python  
- **Data Analysis:** Pandas, NumPy  
- **Visualization:** Matplotlib, Seaborn  
- **Machine Learning:** Scikit-Learn, LightGBM, XGBoost  
- **Environment:** Jupyter Notebook  

---

## 🚀 How to Run the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/Prabhnoor4/Bank-Customer-Churn.git
   cd Bank-Customer-Churn
   ```
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Open the **Jupyter Notebook**
   
4. Run the notebook step by step to:

  - Perform EDA
  - Apply feature engineering
  - Train models
  - Evaluate performance
---
## 📌 Key Takeaways

- Customer activity level is one of the strongest indicators of churn.
- Feature engineering significantly improves model performance.
- Ensemble models outperform traditional classifiers for this problem.
- The pipeline demonstrates a complete data science workflow from raw data to actionable insights.

## 👤 Author

Prabhnoor Singh
GitHub: @Prabhnoor4
   
