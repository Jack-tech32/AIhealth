import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from PIL import Image

st.set_page_config(page_title="HealthAI Suite", layout="centered")
st.title("🧠 HealthAI Suite — Clinical AI Dashboard")


# ================= LOAD MODELS =================
import os


# Base directory of current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths to models (relative, cross-platform)
clf_model_path = os.path.join(BASE_DIR, "../models/classifier_xgb.pkl")
reg_model_path = os.path.join(BASE_DIR, "../models/regressor_xgb.pkl")
cnn_model_path = os.path.join(BASE_DIR, "../models/pneumonia_cnn_final.h5")

# Load models
clf_model = joblib.load(clf_model_path)
reg_model = joblib.load(reg_model_path)
cnn_model = load_model(cnn_model_path)

# ================= TABULAR AI =================

st.header("🩺 Patient Risk & Hospital Stay")

age = st.number_input("Age", 1, 120)
gender = st.selectbox("Gender", ["M", "F"])
bmi = st.number_input("BMI", 10.0, 50.0)
sbp = st.number_input("Systolic BP")
dbp = st.number_input("Diastolic BP")
hr = st.number_input("Heart Rate")
chol = st.number_input("Cholesterol")
sugar = st.number_input("Blood Sugar")
htn = st.selectbox("Hypertension", ["No", "Yes"])
dia = st.selectbox("Diabetes", ["No", "Yes"])

if st.button("Predict Outcome"):

    gender_feM = 1 if gender == "F" else 0
    has_htn = 1 if htn == "Yes" else 0
    has_dia = 1 if dia == "Yes" else 0

    df = pd.DataFrame([{
        'age': age,
        'bmi': bmi,
        'systolic_bp': sbp,
        'diastolic_bp': dbp,
        'heart_rate': hr,
        'cholesterol': chol,
        'blood_sugar': sugar,
        'has_hypertension': has_htn,
        'has_diabetes': has_dia,
        'gender_feM': gender_feM
    }])

    # ---- Raw model predictions ----
    risk = clf_model.predict(df)[0]
    los_raw = float(reg_model.predict(df)[0])

    # ---------- Clinical Rule Engine ----------
    critical = (
        age >= 55 and sugar >= 160 and (has_htn == 1 or has_dia == 1)
    ) or (
        bmi >= 30 and sbp >= 150
    )

    # ---------- Final Clinical Decision ----------
    if critical:
        risk = 1   # Critical / Readmitted
        los = max(los_raw, np.random.uniform(12,18))

    elif age < 35 and bmi < 25 and sugar < 110 and has_htn == 0 and has_dia == 0:
        risk = 2
        los = min(los_raw, 3)

    elif age < 50 and has_htn == 0 and has_dia == 0:
        risk = 2
        los = min(los_raw, 6)

    else:
        los = los_raw

    # ---------- Display ----------
    outcome = {0:"Deceased",1:"Critical / Readmitted",2:"Recovered"}
    st.success(f"Predicted Outcome: {outcome[risk]}")
    st.info(f"Estimated Length of Stay: {round(los,2)} days")

# ================= CNN XRAY =================

st.header("🫁 Chest X-ray Pneumonia Detection")

img = st.file_uploader("Upload Chest X-ray Image", type=["jpg","jpeg","png"])

if img:
    from PIL import Image
    import numpy as np

    # Load and display uploaded image
    image = Image.open(img).convert("RGB")
    st.image(image, caption="Uploaded X-ray", use_column_width=True)

    # Preprocess for CNN
    image_resized = image.resize((64,64))
    image_array = np.array(image_resized)/255.0
    image_array = np.expand_dims(image_array, axis=0)  # shape (1,64,64,3)

    # Prediction
    pred = cnn_model.predict(image_array, verbose=0)[0][0]

    # Thresholding
    label = "PNEUMONIA" if pred > 0.5 else "NORMAL"
    confidence = round(pred*100,2) if pred > 0.5 else round((1-pred)*100,2)

    st.warning(f"Prediction: {label} ({confidence}%)")