import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Literal
from fastapi.middleware.cors import CORSMiddleware

model = joblib.load('titanic_logistic_model.pkl')

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic model describing what the frontend sends us
class PassengerData(BaseModel):
    Pclass: Literal[1, 2, 3]
    Sex: Literal[0, 1]
    Age: float = Field(..., ge=0, le=100)
    SibSp: int = Field(..., ge=0)
    Parch: int = Field(..., ge=0)
    Fare: float = Field(..., ge=0)

# Describe what we send back
class PredictionResponse(BaseModel):
    survived: int
    survival_probability: float

@app.get('/')
def greet():
    return {'message': 'Titanic Survival Prediction API'}

@app.post('/predict', response_model=PredictionResponse)
def predict(data: PassengerData):
    input_row = pd.DataFrame([{
        'Pclass': data.Pclass,
        'Sex': data.Sex,
        'Age': data.Age,
        'SibSp': data.SibSp,
        'Parch': data.Parch,
        'Fare': data.Fare
    }])

    prediction = model.predict(input_row)[0]
    probability = model.predict_proba(input_row)[0][1]

    return PredictionResponse(
        survived=int(prediction),
        survival_probability=round(float(probability), 2)
    )
