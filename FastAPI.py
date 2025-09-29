from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

# Load model pipeline
model = joblib.load("lgbm_best_pipeline.pkl")

# Inisialisasi FastAPI
app = FastAPI(title="Obesity Prediction API", version="1.0")

# Input schema dengan Pydantic
class ObesityInput(BaseModel):
    Gender: str
    Age: int
    Height: float
    Weight: float
    family_history_with_overweight: str
    FAVC: str
    FCVC: float
    NCP: float
    CAEC: str
    SMOKE: str
    CH2O: float
    SCC: str
    FAF: float
    TUE: float
    CALC: str
    MTRANS: str

# Endpoint root
@app.get("/")
def read_root():
    return {"message": "Welcome to Gervasius's Obesity Prediction API"}

# Endpoint prediksi
@app.post("/predict")
def predict_obesity(data: ObesityInput):
    input_df = pd.DataFrame([data.dict()])
    
    try:
        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]
        class_labels = model.classes_

        proba_dict = {label: round(float(p), 4) for label, p in zip(class_labels, proba)}

        return {
            "prediction": prediction,
            "probabilities": proba_dict
        }

    except Exception as e:
        return {"error": str(e)}

