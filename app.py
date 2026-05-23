from flask import Flask, render_template, request
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

app = Flask(__name__)

# =========================================
# LOAD DATASET
# =========================================

df = pd.read_csv("insurance.csv")

# =========================================
# CLEANING
# =========================================

df['age'] = df['age'].fillna(df['age'].mean())
df['bmi'] = df['bmi'].fillna(df['bmi'].mean())
df['children'] = df['children'].fillna(df['children'].mean())

df['sex'] = df['sex'].fillna(df['sex'].mode()[0])
df['smoker'] = df['smoker'].fillna(df['smoker'].mode()[0])
df['region'] = df['region'].fillna(df['region'].mode()[0])

# =========================================
# ENCODING
# =========================================

df['sex'] = df['sex'].map({
    'male': 1,
    'female': 0
})

df['smoker'] = df['smoker'].map({
    'yes': 1,
    'no': 0
})

df = pd.get_dummies(
    df,
    columns=['region'],
    drop_first=True
)

# =========================================
# FEATURE ENGINEERING
# =========================================

df['bmi_age'] = df['bmi'] * df['age']
df['smoker_bmi'] = df['smoker'] * df['bmi']

# =========================================
# FEATURES & TARGET
# =========================================

x = df.drop('charges', axis=1)
y = df['charges']

# =========================================
# TRAIN TEST SPLIT
# =========================================

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

# =========================================
# MODEL
# =========================================

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=12,
    random_state=42
)

model.fit(x_train, y_train)

accuracy = r2_score(
    y_test,
    model.predict(x_test)
)

# =========================================
# HOME ROUTE
# =========================================

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    risk = None

    if request.method == "POST":

        age = int(request.form["age"])

        sex = request.form["sex"]

        bmi = float(request.form["bmi"])

        children = int(request.form["children"])

        smoker = request.form["smoker"]

        region = request.form["region"]

        # =========================================
        # ENCODING USER INPUT
        # =========================================

        sex_value = 1 if sex == "male" else 0

        smoker_value = 1 if smoker == "yes" else 0

        region_northwest = 1 if region == "northwest" else 0

        region_southeast = 1 if region == "southeast" else 0

        region_southwest = 1 if region == "southwest" else 0

        bmi_age = bmi * age

        smoker_bmi = smoker_value * bmi

        # =========================================
        # PREDICTION
        # =========================================

        data = [[
            age,
            sex_value,
            bmi,
            children,
            smoker_value,
            region_northwest,
            region_southeast,
            region_southwest,
            bmi_age,
            smoker_bmi
        ]]

        prediction = model.predict(data)[0]

        prediction = round(prediction, 2)

        # =========================================
        # RISK LEVEL
        # =========================================

        if prediction < 10000:
            risk = "Low Risk 🟢"

        elif prediction < 25000:
            risk = "Medium Risk 🟡"

        else:
            risk = "High Risk 🔴"

    return render_template(
        "index.html",
        prediction=prediction,
        accuracy=round(accuracy * 100, 2),
        risk=risk
    )

# =========================================
# RUN APP
# =========================================

if __name__ == "__main__":
    app.run(debug=True)