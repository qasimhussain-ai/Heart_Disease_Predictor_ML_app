import streamlit as st
import numpy as np
import joblib

# Load model & scaler
model = joblib.load("heart_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Heart Disease Predictor", layout="wide")

# --- Styling ---
st.markdown(
    """
    <style>
    .title {text-align: center; font-size: 32px; font-weight: 700;}
    .card {background: #f8f9fb; padding: 16px; border-radius: 8px;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div class='title'>🫀 Heart Disease Prediction</div>", unsafe_allow_html=True)
st.write("Use the controls in the left sidebar to enter patient information. Click Predict to evaluate risk.")

# ---- Sidebar: Inputs grouped into a form ----
with st.sidebar.form(key="patient_form"):
    st.header("Patient Details")
    age = st.number_input("Age", 18, 100, 40)
    sex_m = st.selectbox("Sex", options=[("Female", 0), ("Male", 1)], index=1)[1]

    st.subheader("Vitals & Labs")
    resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 50, 250, 120)
    cholesterol = st.number_input("Cholesterol (mg/dL)", 80, 600, 200)
    fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", options=[("No", 0), ("Yes", 1)], index=0)[1]

    st.subheader("Exercise & ECG")
    max_hr = st.number_input("Max Heart Rate", 60, 220, 150)
    oldpeak = st.number_input("Oldpeak (ST Depression)", 0.0, 6.0, 1.0, step=0.1)
    exercise_angina = st.selectbox("Exercise-induced Angina", options=[("No", 0), ("Yes", 1)], index=0)[1]

    st.subheader("Chest Pain Type (select one or multiple)")
    cp_ata = st.checkbox("ATA")
    cp_nap = st.checkbox("NAP")
    cp_ta = st.checkbox("TA")

    st.subheader("Resting ECG & ST Slope")
    ecg_normal = st.checkbox("Normal")
    ecg_st = st.checkbox("ST")
    slope_flat = st.checkbox("Flat")
    slope_up = st.checkbox("Up")

    submit = st.form_submit_button("Predict")

# ---- Helper: build feature vector ----
def build_feature_vector():
    # Scale numeric inputs only (scaler was fitted on numeric columns)
    numeric = np.array([[age, resting_bp, cholesterol, max_hr, oldpeak]])
    numeric_scaled = scaler.transform(numeric)[0]

    model_cols = list(model.feature_names_in_)

    age_group_middle = 1 if 45 <= age < 65 else 0
    age_group_old = 1 if age >= 65 else 0
    high_cholesterol = 1 if cholesterol > 240 else 0
    high_bp = 1 if resting_bp > 140 else 0
    risk_score = 0

    scaled_map = {
        'Age': numeric_scaled[0],
        'RestingBP': numeric_scaled[1],
        'Cholesterol': numeric_scaled[2],
        'MaxHR': numeric_scaled[3],
        'Oldpeak': numeric_scaled[4]
    }

    value_map = {
        'FastingBS': int(fasting_bs),
        'Sex_M': int(sex_m),
        'ChestPainType_ATA': int(cp_ata),
        'ChestPainType_NAP': int(cp_nap),
        'ChestPainType_TA': int(cp_ta),
        'RestingECG_Normal': int(ecg_normal),
        'RestingECG_ST': int(ecg_st),
        'ExerciseAngina_Y': int(exercise_angina),
        'ST_Slope_Flat': int(slope_flat),
        'ST_Slope_Up': int(slope_up),
        'Age_Group_Middle': age_group_middle,
        'Age_Group_Old': age_group_old,
        'High_Cholesterol': high_cholesterol,
        'High_BP': high_bp,
        'Risk_Score': risk_score
    }

    full_features = [value_map.get(col, scaled_map.get(col, 0)) for col in model_cols]
    return np.array([full_features])

# ---- Prediction and Results UI ----
if submit:
    try:
        X = build_feature_vector()
        pred = model.predict(X)[0]
        proba = None
        try:
            proba = model.predict_proba(X)[0][1]
        except Exception:
            proba = None

        # Layout: result card + details
        col1, col2 = st.columns([1, 2])
        with col1:
            if pred == 1:
                st.markdown('<div class="card"><h3 style="color:#7f1d1d">⚠️ High Risk</h3><p>Model predicts presence of heart disease.</p></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="card"><h3 style="color:#0b6623">✅ Low Risk</h3><p>Model predicts low risk of heart disease.</p></div>', unsafe_allow_html=True)
            if proba is not None:
                st.metric(label="Predicted Risk Probability", value=f"{proba*100:.1f}%")

        with col2:
            st.subheader("Top feature importances")
            try:
                importances = getattr(model, 'feature_importances_', None)
                if importances is not None:
                    names = list(model.feature_names_in_)
                    idx = np.argsort(importances)[::-1][:6]
                    for i in idx:
                        st.write(f"- {names[i]}: {importances[i]:.3f}")
                else:
                    st.write("Feature importances not available for this model.")
            except Exception:
                st.write("Unable to compute feature importances.")

        with st.expander("Input summary"):
            st.write({
                'Age': age,
                'Sex_M': sex_m,
                'RestingBP': resting_bp,
                'Cholesterol': cholesterol,
                'MaxHR': max_hr,
                'Oldpeak': oldpeak,
                'ExerciseAngina': exercise_angina,
            })

        st.markdown("---")
        st.caption("Note: This is a demo UI. For clinical use, validate preprocessing/pipelines and thresholds against trained model artifacts.")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
