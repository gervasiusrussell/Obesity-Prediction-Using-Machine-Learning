import streamlit as st
import pandas as pd
import json
import requests

# Streamlit App Title
st.title("🔍 Prediksi Tingkat Obesitas")
st.write("Masukkan data untuk memprediksi tingkat obesitas berdasarkan gaya hidup.")

# Form Input
Gender = st.selectbox("Gender", ["Male", "Female"])
Age = st.slider("Age", min_value=14, max_value=55, value=25)
Height = st.slider("Height (m)", min_value=1.45, max_value=1.95, value=1.70)
Weight = st.slider("Weight (kg)", min_value=39.0, max_value=173.0, value=70.0)
family_history_with_overweight = st.selectbox("Family history with overweight?", ["yes", "no"])
FAVC = st.selectbox("Frequent consumption of high caloric food?", ["yes", "no"])
FCVC = st.slider("Vegetable consumption (1-3)", min_value=1.0, max_value=3.0, value=2.0)
NCP = st.slider("Main meals per day (1-4)", min_value=1.0, max_value=4.0, value=3.0)
CAEC = st.selectbox("Eating between meals?", ["No", "Sometimes", "Frequently", "Always"])
SMOKE = st.selectbox("Do you smoke?", ["yes", "no"])
CH2O = st.slider("Water intake (1-3 liters)", min_value=1.0, max_value=3.0, value=2.0)
SCC = st.selectbox("Do you monitor calorie intake?", ["yes", "no"])
FAF = st.slider("Physical activity (0-3)", min_value=0.0, max_value=3.0, value=1.0)
TUE = st.slider("Time using tech devices (0-2)", min_value=0.0, max_value=2.0, value=1.0)
CALC = st.selectbox("Alcohol consumption?", ["No", "Sometimes", "Frequently", "Always"])
MTRANS = st.selectbox("Primary transport method", ["Automobile", "Bike", "Motorbike", "Public_Transportation", "Walking"])

# Assemble inputs into dictionary
inputs = {
    "Gender": Gender,
    "Age": Age,
    "Height": Height,
    "Weight": Weight,
    "family_history_with_overweight": family_history_with_overweight,
    "FAVC": FAVC,
    "FCVC": FCVC,
    "NCP": NCP,
    "CAEC": CAEC,
    "SMOKE": SMOKE,
    "CH2O": CH2O,
    "SCC": SCC,
    "FAF": FAF,
    "TUE": TUE,
    "CALC": CALC,
    "MTRANS": MTRANS
}

# Submit button
if st.button("🚀 Predict"):
    try:
        res = requests.post(
            "http://127.0.0.1:8000/predict",
            data=json.dumps(inputs),
            headers={"Content-Type": "application/json"}
        )

        if res.status_code == 200:
            result = res.json()
            prediction = result.get("prediction")
            probabilities = result.get("probabilities")

            st.success(f"✅ Prediction Result: **{prediction}**")

            if probabilities:
                st.subheader("📊 Class Probabilities")
                st.dataframe(pd.DataFrame([probabilities]).T.rename(columns={0: "Probability"}))

        else:
            st.error(f"❌ API returned status code: {res.status_code}")
            st.error(res.text)

    except Exception as e:
        st.error(f"❌ API call failed: {e}")
