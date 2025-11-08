import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =========================================================
# ⚙️ Streamlit Configuration
# =========================================================
st.set_page_config(page_title="Credit Card Default Predictor", layout="wide")

st.title("💳 Credit Card Default Prediction App")
st.markdown("""
Predict whether a **credit card customer** is likely to default on their **next payment**  
using advanced **machine learning models** trained on real financial data.
""")

# =========================================================
# 🧠 Load Models and Scaler
# =========================================================
@st.cache_resource
def load_models():
    try:
        stack_model = joblib.load("stacking_model.pkl")
        xgb_model = joblib.load("xgboost_model.pkl")
        lgb_model = joblib.load("lightgbm_model.pkl")
        rf_model = joblib.load("randomforest_model.pkl")
        scaler = joblib.load("scaler.pkl")
        best_threshold = joblib.load("best_threshold.pkl")
        return stack_model, xgb_model, lgb_model, rf_model, scaler, best_threshold
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        return None, None, None, None, None, None


stack_model, xgb_model, lgb_model, rf_model, scaler, best_threshold = load_models()

# =========================================================
# ✅ Sidebar for Input Parameters
# =========================================================
st.sidebar.header("📋 Input Customer Details")

st.sidebar.markdown("""
**Repayment Status Legend**
- `-2`: No consumption  
- `-1`: Paid duly (before statement)  
- `0`: Paid on time  
- `1`: 1 month delay  
- `2`: 2 months delay  
- `...` up to `8`: 8 months delay
""")

sex_options = {"Male": 1, "Female": 2}
education_options = {
    "Graduate School": 1,
    "University": 2,
    "High School": 3,
    "Others": 4
}
marriage_options = {"Married": 1, "Single": 2, "Others": 3}

LIMIT_BAL = st.sidebar.number_input("💰 Credit Limit (NTD)", 10000, 1000000, 200000, step=10000)
SEX = sex_options[st.sidebar.selectbox("👤 Gender", list(sex_options.keys()))]
EDUCATION = education_options[st.sidebar.selectbox("🎓 Education Level", list(education_options.keys()))]
MARRIAGE = marriage_options[st.sidebar.selectbox("💍 Marital Status", list(marriage_options.keys()))]
AGE = st.sidebar.slider("🎂 Age", 18, 80, 35)

st.sidebar.markdown("---")
st.sidebar.subheader("📅 Repayment Status (Last 6 Months)")
PAY_0 = st.sidebar.slider("September (Most Recent)", -2, 8, 0)
PAY_2 = st.sidebar.slider("August", -2, 8, 0)
PAY_3 = st.sidebar.slider("July", -2, 8, 0)
PAY_4 = st.sidebar.slider("June", -2, 8, 0)
PAY_5 = st.sidebar.slider("May", -2, 8, 0)
PAY_6 = st.sidebar.slider("April", -2, 8, 0)

st.sidebar.markdown("---")
st.sidebar.subheader("💵 Bill Amounts (NTD)")
BILL_AMT1 = st.sidebar.number_input("September Bill", 0, 1000000, 50000)
BILL_AMT2 = st.sidebar.number_input("August Bill", 0, 1000000, 48000)
BILL_AMT3 = st.sidebar.number_input("July Bill", 0, 1000000, 46000)
BILL_AMT4 = st.sidebar.number_input("June Bill", 0, 1000000, 45000)
BILL_AMT5 = st.sidebar.number_input("May Bill", 0, 1000000, 44000)
BILL_AMT6 = st.sidebar.number_input("April Bill", 0, 1000000, 42000)

st.sidebar.markdown("---")
st.sidebar.subheader("💸 Payment Amounts (NTD)")
PAY_AMT1 = st.sidebar.number_input("September Payment", 0, 1000000, 20000)
PAY_AMT2 = st.sidebar.number_input("August Payment", 0, 1000000, 18000)
PAY_AMT3 = st.sidebar.number_input("July Payment", 0, 1000000, 16000)
PAY_AMT4 = st.sidebar.number_input("June Payment", 0, 1000000, 15000)
PAY_AMT5 = st.sidebar.number_input("May Payment", 0, 1000000, 14000)
PAY_AMT6 = st.sidebar.number_input("April Payment", 0, 1000000, 12000)

# =========================================================
# 🧮 Derived Features (same as training)
# =========================================================
AVG_BILL_AMT = np.mean([BILL_AMT1, BILL_AMT2, BILL_AMT3, BILL_AMT4, BILL_AMT5, BILL_AMT6])
AVG_PAY_AMT = np.mean([PAY_AMT1, PAY_AMT2, PAY_AMT3, PAY_AMT4, PAY_AMT5, PAY_AMT6])
PAY_RATIO = AVG_PAY_AMT / (AVG_BILL_AMT + 1)
UTILIZATION = BILL_AMT1 / (LIMIT_BAL + 1)
TOTAL_PAY_AMT = sum([PAY_AMT1, PAY_AMT2, PAY_AMT3, PAY_AMT4, PAY_AMT5, PAY_AMT6])
TOTAL_BILL_AMT = sum([BILL_AMT1, BILL_AMT2, BILL_AMT3, BILL_AMT4, BILL_AMT5, BILL_AMT6])

# =========================================================
# 🧩 Input DataFrame (must match training features)
# =========================================================
input_data = pd.DataFrame([[
    LIMIT_BAL, SEX, EDUCATION, MARRIAGE, AGE,
    PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6,
    BILL_AMT1, BILL_AMT2, BILL_AMT3, BILL_AMT4, BILL_AMT5, BILL_AMT6,
    PAY_AMT1, PAY_AMT2, PAY_AMT3, PAY_AMT4, PAY_AMT5, PAY_AMT6,
    AVG_BILL_AMT, AVG_PAY_AMT, PAY_RATIO,
    UTILIZATION, TOTAL_PAY_AMT, TOTAL_BILL_AMT
]], columns=[
    'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE',
    'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
    'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
    'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6',
    'AVG_BILL_AMT', 'AVG_PAY_AMT', 'PAY_RATIO',
    'UTILIZATION', 'TOTAL_PAY_AMT', 'TOTAL_BILL_AMT'
])

# =========================================================
# 🧮 Scale Data
# =========================================================
try:
    X_scaled = scaler.transform(input_data)
except Exception as e:
    st.error(f"Error scaling data: {e}")
    st.stop()

# =========================================================
# 🚀 Model Selection + Prediction
# =========================================================
st.sidebar.markdown("---")
model_choice = st.sidebar.radio(
    "🧠 Select Model for Prediction",
    ("Stacking Model", "XGBoost", "LightGBM", "Random Forest")
)

if st.sidebar.button("🔮 Predict Default Probability"):
    try:
        if model_choice == "Stacking Model":
            model = stack_model
        elif model_choice == "XGBoost":
            model = xgb_model
        elif model_choice == "LightGBM":
            model = lgb_model
        else:
            model = rf_model

        prob = model.predict_proba(X_scaled)[:, 1][0]
        final_pred = int(prob >= best_threshold)

        st.subheader("🧾 Prediction Result")
        st.metric(f"Default Probability ({model_choice})", f"{prob:.2%}")
        st.write(f"**Optimal Decision Threshold:** {best_threshold:.2f}")

        if final_pred == 1:
            st.error("⚠️ High Risk: Customer likely to default on next payment.")
        else:
            st.success("✅ Low Risk: Customer likely to pay on time.")

    except Exception as e:
        st.error(f"Error during prediction: {e}")

# =========================================================
# ℹ️ Info Section
# =========================================================
st.markdown("---")
with st.expander("ℹ️ About This App"):
    st.write("""
    - Inputs are taken from real credit card billing data.  
    - Repayment status (`PAY_X`) indicates delays in payments.  
    - Derived features like `PAY_RATIO`, `UTILIZATION`, and `AVG_BILL_AMT` are auto-calculated.  
    - Models trained using **XGBoost, LightGBM, RandomForest**, and an ensemble **Stacking model**.  
    """)
