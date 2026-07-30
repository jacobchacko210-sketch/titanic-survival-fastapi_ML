import os
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import joblib
import pandas as pd

# Load the logistic regression model
model = joblib.load("Titanic_Survival_Predictor.pkl")

app = FastAPI()

# 1. Define the expected data format
class PassengerData(BaseModel):
    Pclass: int
    Sex: str
    Age: float
    SibSp: int
    Parch: int
    Fare: float

# 2. GET route to serve the HTML interface
@app.get("/", response_class=HTMLResponse)
def serve_home():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    html_file_path = os.path.join(current_dir, "index.html")
    
    with open(html_file_path, "r") as f:
        return f.read()

# 3. POST route to handle predictions
@app.post("/predict")
def predict_survival(data: PassengerData):
    # Convert 'Sex' string to numerical input expected by the model (e.g., Male = 0, Female = 1)
    sex_encoded = 1 if data.Sex.lower() == "female" else 0

    # Create the DataFrame
    df = pd.DataFrame([{
        "Pclass": data.Pclass,
        "Sex": sex_encoded,
        "Age": data.Age,
        "SibSp": data.SibSp,
        "Parch": data.Parch,
        "Fare": data.Fare
    }])
    
    # Generate prediction and probability
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]
    
    return {
        "prediction": int(prediction),
        "probability": float(probability)
    }
