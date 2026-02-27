🏥 Healthcare Readmission Prediction (30-Day)
📌 Project Overview

This project is an end-to-end healthcare analytics solution focused on predicting 30-day hospital readmissions for diabetes patients.

Hospital readmissions significantly increase healthcare costs and indicate potential gaps in treatment quality. By building a predictive model, this project aims to help healthcare institutions proactively identify high-risk patients and improve patient outcomes.

🎯 Problem Statement

Predict whether a diabetes patient will be readmitted to the hospital within 30 days of discharge.

Why This Matters:

Reduces hospital penalties

Improves patient care quality

Optimizes hospital resource allocation

Supports data-driven healthcare decisions

📊 Dataset Information

Dataset Name: Diabetes 130-US Hospitals (1999–2008)

Source: UCI Machine Learning Repository

Records: 100,000+ patient encounters

Features: 50+ medical and administrative attributes

📥 Dataset Download Instructions

Download diabetic_data.csv from the UCI ML Repository.

Place the file in the project root directory.

Run the project.

(Note: Dataset is not uploaded due to size constraints.)

🛠️ Tech Stack

Python

Pandas

NumPy

Matplotlib

Seaborn

Scikit-learn

XGBoost (if used)

🔄 Project Pipeline
1️⃣ Data Acquisition & Validation

Loaded real-world healthcare dataset

Validated data structure and integrity

Checked missing values and anomalies

2️⃣ Data Preprocessing & Cleaning

Handled missing values

Removed duplicates

Treated invalid entries (e.g., '?')

Feature engineering

Categorical encoding

Data transformation & normalization

3️⃣ Exploratory Data Analysis (EDA)

Univariate analysis

Bivariate analysis

Correlation study

Readmission distribution analysis

Identification of high-risk patient groups

Professional visualizations

Key Insights:

Certain medications influence readmission rates

Age groups show different readmission patterns

Length of stay impacts risk probability

4️⃣ Model Building

Built multiple classification models:

Logistic Regression

Decision Tree

Random Forest

XGBoost (if implemented)

📈 Model Evaluation

Evaluation Metrics Used:

Accuracy

Precision

Recall

F1-Score

ROC-AUC

Confusion Matrix

Best model selected based on overall performance and recall (important in healthcare use cases).

5️⃣ Feature Importance Analysis

Identified key drivers of hospital readmission such as:

Number of inpatient visits

Number of emergency visits

Diagnosis category

Medication changes

Age group

This helps stakeholders understand why predictions are made.

📊 Sample Visualizations

(Add screenshots in your repo and reference them like below)

ROC Curve

Confusion Matrix

Feature Importance Plot

Readmission Distribution Chart

🚀 How to Run the Project
git clone https://github.com/YOUR_USERNAME/healthcare-readmission-prediction.git
cd healthcare-readmission-prediction
pip install -r requirements.txt
python main.py
📁 Recommended Project Structure
healthcare-readmission-prediction/
│
├── data/
├── notebooks/
├── src/
│   ├── preprocessing.py
│   ├── model.py
│   └── evaluation.py
├── requirements.txt
├── README.md
└── main.py
📌 Key Learnings

Handling messy real-world healthcare datasets

Building a complete ML pipeline

Feature engineering for structured medical data

Model evaluation in imbalanced classification

Writing professional documentation

GitHub project structuring best practices

🔮 Future Improvements

Deploy model using Streamlit

Hyperparameter tuning with GridSearchCV

Implement SMOTE for class imbalance

Add model monitoring

Create REST API for predictions

👩‍💻 Author

Sanjana Dhage
Aspiring Data Analyst | Healthcare Analytics Enthusiast
LinkedIn: (Add your link)
GitHub: (Add your link)
