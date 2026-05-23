# =========================================================
# PREMIUM INSURANCE AI PREDICTOR
# HIGH ACCURACY + PROFESSIONAL UI
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Insurance AI Predictor",
    page_icon="💎",
    layout="wide"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

/* Main Background */
.stApp {
    background: linear-gradient(135deg,#f8fafc,#e2e8f0);
}

/* Hide Streamlit Header */
header {
    visibility: hidden;
}

/* Main Title */
.main-title {
    font-size: 60px;
    font-weight: 800;
    color: #0f172a;
    text-align: center;
    margin-top: -40px;
}

.sub-title {
    text-align: center;
    font-size: 22px;
    color: #475569;
    margin-bottom: 40px;
}

/* Glass Card */
.glass {
    background: rgba(255,255,255,0.75);
    backdrop-filter: blur(14px);
    border-radius: 24px;
    padding: 30px;
    box-shadow: 0px 8px 32px rgba(0,0,0,0.08);
    border: 1px solid rgba(255,255,255,0.3);
}

/* Metric Card */
.metric-card {
    background: white;
    padding: 25px;
    border-radius: 20px;
    text-align: center;
    box-shadow: 0px 5px 15px rgba(0,0,0,0.06);
}

.metric-title {
    color: #64748b;
    font-size: 18px;
}

.metric-value {
    color: #0f172a;
    font-size: 42px;
    font-weight: bold;
}

/* Prediction Box */
.prediction-box {
    background: linear-gradient(135deg,#2563eb,#06b6d4);
    padding: 40px;
    border-radius: 25px;
    color: white;
    text-align: center;
    box-shadow: 0px 10px 30px rgba(37,99,235,0.35);
}

/* Risk Box */
.risk-box {
    background: #f1f5f9;
    padding: 25px;
    border-radius: 20px;
    text-align: center;
}

/* Button */
.stButton>button {
    width: 100%;
    height: 60px;
    border-radius: 16px;
    border: none;
    background: linear-gradient(135deg,#2563eb,#06b6d4);
    color: white;
    font-size: 22px;
    font-weight: bold;
    transition: 0.3s;
}

.stButton>button:hover {
    transform: scale(1.02);
    box-shadow: 0px 10px 25px rgba(37,99,235,0.3);
}

/* Input Labels */
label {
    font-weight: 600 !important;
    color: #0f172a !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0f172a;
}

[data-testid="stSidebar"] * {
    color: white;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD DATASET
# =========================================================

df = pd.read_csv("insurance.csv")

# =========================================================
# CLEANING
# =========================================================

df['age'] = df['age'].fillna(df['age'].mean())
df['bmi'] = df['bmi'].fillna(df['bmi'].mean())
df['children'] = df['children'].fillna(df['children'].mean())

df['sex'] = df['sex'].fillna(df['sex'].mode()[0])
df['smoker'] = df['smoker'].fillna(df['smoker'].mode()[0])
df['region'] = df['region'].fillna(df['region'].mode()[0])

# =========================================================
# ENCODING
# =========================================================

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

# =========================================================
# FEATURE ENGINEERING
# =========================================================

df['bmi_age'] = df['bmi'] * df['age']

df['smoker_bmi'] = df['smoker'] * df['bmi']

# =========================================================
# FEATURES & TARGET
# =========================================================

x = df.drop('charges', axis=1)

y = df['charges']

# =========================================================
# SPLIT DATA
# =========================================================

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

# =========================================================
# HIGH ACCURACY MODEL
# =========================================================

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=12,
    random_state=42
)

model.fit(x_train, y_train)

# =========================================================
# MODEL SCORE
# =========================================================

y_pred = model.predict(x_test)

accuracy = r2_score(y_test, y_pred)

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("💎 Insurance AI")

st.sidebar.markdown("""
### Features

✅ High Accuracy ML Model  
✅ Random Forest Regressor  
✅ Modern Premium UI  
✅ Real-Time Prediction  
✅ Risk Analysis  
✅ Scenario Comparison  
""")

st.sidebar.success(
    f"Model Accuracy: {accuracy:.2f}"
)

# =========================================================
# HEADER
# =========================================================

st.markdown("""
<div class='main-title'>
🏥 Insurance Premium Predictor
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class='sub-title'>
AI Powered Medical Insurance Estimation Platform
</div>
""", unsafe_allow_html=True)

# =========================================================
# TOP METRICS
# =========================================================

m1, m2, m3 = st.columns(3)

with m1:
    st.markdown(f"""
    <div class='metric-card'>
    <div class='metric-title'>Dataset Size</div>
    <div class='metric-value'>{df.shape[0]}</div>
    </div>
    """, unsafe_allow_html=True)

with m2:
    st.markdown(f"""
    <div class='metric-card'>
    <div class='metric-title'>Model Accuracy</div>
    <div class='metric-value'>{accuracy:.2f}</div>
    </div>
    """, unsafe_allow_html=True)

with m3:
    st.markdown(f"""
    <div class='metric-card'>
    <div class='metric-title'>Features</div>
    <div class='metric-value'>{x.shape[1]}</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")
st.write("")

# =========================================================
# MAIN LAYOUT
# =========================================================

left, right = st.columns([1,1])

# =========================================================
# INPUT FORM
# =========================================================

with left:

    st.markdown("""
    <div class='glass'>
    <h2 style='color:#2563eb;'>
    📝 Customer Information
    </h2>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    age = st.slider(
        "Age",
        18,
        100,
        25
    )

    sex = st.selectbox(
        "Gender",
        ["male", "female"]
    )

    bmi = st.slider(
        "BMI",
        10.0,
        50.0,
        25.0
    )

    children = st.slider(
        "Number of Dependents",
        0,
        10,
        0
    )

    smoker = st.selectbox(
        "Smoking Status",
        ["yes", "no"]
    )

    region = st.selectbox(
        "Region",
        [
            "northwest",
            "southeast",
            "southwest",
            "northeast"
        ]
    )

# =========================================================
# PREDICTION
# =========================================================

with right:

    st.markdown("""
    <div class='glass'>
    <h2 style='color:#2563eb;'>
    🤖 AI Prediction Engine
    </h2>
    </div>
    """, unsafe_allow_html=True)

    st.write("")
    st.write("")

    if st.button("🚀 Generate Insurance Quote"):

        # Binary Encoding
        sex_value = 1 if sex == 'male' else 0

        smoker_value = 1 if smoker == 'yes' else 0

        # One Hot Encoding
        region_northwest = 1 if region == 'northwest' else 0

        region_southeast = 1 if region == 'southeast' else 0

        region_southwest = 1 if region == 'southwest' else 0

        # Feature Engineering
        bmi_age = bmi * age

        smoker_bmi = smoker_value * bmi

        # User Data
        user_data = [[
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

        # Prediction
        prediction = model.predict(user_data)

        prediction_value = prediction[0]

        # Risk Level
        if prediction_value < 10000:
            risk = "Low Risk 🟢"
        elif prediction_value < 25000:
            risk = "Medium Risk 🟡"
        else:
            risk = "High Risk 🔴"

        # Prediction Box
        st.markdown(f"""
        <div class='prediction-box'>

        <h2>💰 Estimated Insurance Premium</h2>

        <h1 style='font-size:60px;'>
        ₹ {prediction_value:,.2f}
        </h1>

        <p style='font-size:22px;'>
        {risk}
        </p>

        </div>
        """, unsafe_allow_html=True)

        st.write("")
        st.write("")

        # Scenario Comparison
        st.markdown("""
        <div class='glass'>
        <h3 style='color:#2563eb;'>
        📊 What-If Scenario Comparison
        </h3>
        </div>
        """, unsafe_allow_html=True)

        st.write("")

        c1, c2, c3 = st.columns(3)

        # Quit Smoking Scenario
        no_smoke_data = [[
            age,
            sex_value,
            bmi,
            children,
            0,
            region_northwest,
            region_southeast,
            region_southwest,
            bmi_age,
            0
        ]]

        no_smoke_pred = model.predict(no_smoke_data)[0]

        # Lose Weight Scenario
        low_bmi = bmi - 5

        low_bmi_data = [[
            age,
            sex_value,
            low_bmi,
            children,
            smoker_value,
            region_northwest,
            region_southeast,
            region_southwest,
            low_bmi * age,
            smoker_value * low_bmi
        ]]

        low_bmi_pred = model.predict(low_bmi_data)[0]

        with c1:
            st.markdown(f"""
            <div class='risk-box'>
            <h4>Current</h4>
            <h2>₹ {prediction_value:,.0f}</h2>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
            <div class='risk-box'>
            <h4>If Quit Smoking</h4>
            <h2>₹ {no_smoke_pred:,.0f}</h2>
            <p style='color:green;'>
            -₹ {prediction_value-no_smoke_pred:,.0f}
            </p>
            </div>
            """, unsafe_allow_html=True)

        with c3:
            st.markdown(f"""
            <div class='risk-box'>
            <h4>If Lose Weight</h4>
            <h2>₹ {low_bmi_pred:,.0f}</h2>
            <p style='color:green;'>
            -₹ {prediction_value-low_bmi_pred:,.0f}
            </p>
            </div>
            """, unsafe_allow_html=True)

        st.balloons()

# =========================================================
# FOOTER
# =========================================================

st.write("")
st.write("")
st.markdown("""
<hr>

<center>
<p style='color:#64748b;'>
© 2026 Insurance AI Predictor • Built with Streamlit & Random Forest ML
</p>
</center>
""", unsafe_allow_html=True)