# Insurance AI Predictor

Insurance AI Predictor is a machine learning web application that predicts medical insurance costs based on user information such as age, BMI, smoking status, region, and number of dependents.

The project uses a Random Forest Regressor model for high prediction accuracy and provides a professional interactive user interface built with Streamlit.

---

# Features

- High accuracy machine learning model
- Real-time insurance premium prediction
- Professional modern user interface
- Health impact analysis
- Smoking and BMI comparison scenarios
- Interactive sliders and dropdowns
- Responsive layout
- Feature engineered prediction system

---

# Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn

---

# Machine Learning Model

Model Used:

```python
RandomForestRegressor
```

Accuracy:

```text
Approximately 88% - 92%
```

---

# Dataset Features

Input Features:

- age
- sex
- bmi
- children
- smoker
- region

Engineered Features:

- bmi_age
- smoker_bmi

Target Feature:

- charges

---

# Project Structure
```text
PROJECT/
│
├── streamlit.py
├── insurance.csv
├── requirements.txt
└── README.md
---
---

# Requirements

Create a `requirements.txt` file with:
```txt
streamlit
pandas
numpy
scikit-learn
```
---

# UI Features

- Glassmorphism design
- Gradient backgrounds
- Responsive layout
- Modern typography
- Interactive cards
- Scenario analysis dashboard
- Premium styling

---

# Future Improvements

- Add user authentication
- Deploy using Streamlit Cloud
- Add database integration
- Add downloadable reports
- Add charts and visual analytics
- Use XGBoost for higher accuracy
- Add model persistence using Pickle

---

# Author

Areef Baba

---

# License

This project is open source and available under the MIT License.
