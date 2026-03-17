from fastapi import FastAPI
import pandas as pd
import numpy as np

from backend.model_loader import model, scaler, columns
from backend.schemas import PredictionRequest

app = FastAPI()


@app.get("/")
def root():
    return {"message": "Autism Screening API running"}


@app.post("/predict")
def predict(request: PredictionRequest):

    # Step 1: convert request → dict
    input_data = request.dict()

    # Step 2: convert to dataframe
    df = pd.DataFrame([input_data])

    # Step 3: one-hot encode (same as training)
    df = pd.get_dummies(df)

    # Step 4: align columns with training
    df = df.reindex(columns=columns, fill_value=0)

    # Step 5: scale
    features_scaled = scaler.transform(df)

    # Step 6: predict
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0][1]

    return {
        "prediction": int(prediction),
        "probability": float(probability)
    }