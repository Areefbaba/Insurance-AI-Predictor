# Insurance AI Predictor

Insurance AI Predictor is a machine learning based web application that predicts medical insurance charges using customer health and demographic information.

The application uses multiple regression algorithms and automatically selects the best-performing model for accurate premium prediction. The project also includes a modern interactive user interface built using Streamlit.

---

# Project Overview

This project analyzes insurance data and predicts medical insurance costs based on features such as:

- Age
- Gender
- BMI
- Number of Children
- Smoking Status
- Region

The application performs:

- Data Cleaning
- Feature Engineering
- Model Training
- Model Comparison
- Insurance Cost Prediction
- Risk Analysis

---

# Features

- Multiple machine learning regression models
- Automatic best model selection
- High prediction accuracy
- Interactive Streamlit dashboard
- Feature engineered dataset
- Real-time premium prediction
- Health risk analysis
- Modern professional UI

---

# Machine Learning Models Used

The project compares the following models:

- XGBoost Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- AdaBoost Regressor
- Decision Tree Regressor
- KNeighbors Regressor
- LinearSVR

The best model is selected automatically based on R2 Score.

---

# Technologies Used

## Programming Language

- Python

## Libraries

- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Streamlit

---

# Dataset Information

Dataset contains insurance-related information with the following columns:

| Feature | Description |
|---|---|
| age | Age of customer |
| sex | Gender |
| bmi | Body Mass Index |
| children | Number of dependents |
| smoker | Smoking status |
| region | Residential region |
| charges | Medical insurance cost |

---

# Feature Engineering

Additional features created:

- bmi_age
- smoker_bmi
- age_bmi
- smoker_age
- children_bmi

These engineered features improve model performance and prediction accuracy.

---

# Project Structure

```text
Insurance-AI-Predictor/
│
├── streamlit.py
├── insurance.csv
├── requirements.txt
├── README.md
├── .gitignore
└── assets/
```

---


# Requirements File

Create a file named:

```text
requirements.txt
```

Add:

```txt
streamlit
pandas
numpy
scikit-learn
xgboost
```
---

# Workflow

## 1. Data Collection

Insurance dataset is loaded using Pandas.

## 2. Data Cleaning

Missing values are handled using:

- Mean Imputation
- Mode Imputation

## 3. Encoding

Categorical values are converted into numerical format using:

- Label Encoding
- One Hot Encoding

## 4. Feature Engineering

New important features are created from existing columns.

## 5. Model Training

Multiple regression models are trained on the dataset.

## 6. Model Evaluation

Models are evaluated using:

- R2 Score
- Mean Squared Error

## 7. Prediction

Best model predicts insurance premium for user input.

---

# Model Evaluation Metrics

## R2 Score

Measures prediction accuracy.

## Mean Squared Error

Measures prediction error.

---

# Sample Prediction Output

```text
Predicted Insurance Cost: ₹ 42,500
Risk Level: High Risk
```

---

# User Inputs

The application accepts:

- Age
- Gender
- BMI
- Number of Children
- Smoking Status
- Region

---

# UI Features

- Premium modern interface
- Dark theme dashboard
- Interactive sliders
- Prediction cards
- Responsive design
- Professional typography
- Real-time calculations

---

# Future Improvements

- Deploy using Streamlit Cloud
- Add user authentication
- Add PDF report generation
- Add charts and visual analytics
- Add health recommendations
- Save prediction history
- Add database integration
- Build mobile responsive version

---

# Learning Outcomes

This project demonstrates:

- End-to-end machine learning workflow
- Regression algorithms
- Data preprocessing
- Feature engineering
- Model comparison
- Streamlit development
- UI/UX integration
- Real-world ML deployment

---

# Author

Areef Baba

---
