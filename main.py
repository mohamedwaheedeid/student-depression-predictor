import os, joblib, numpy as np, pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title='Student Depression Prediction API', version='1.0')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

def load_artifact(filename):
    path = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(path): raise FileNotFoundError(f'Not found: {path}')
    return joblib.load(path)

scaler = load_artifact('scaler.pkl')
pca = load_artifact('pca.pkl')
model = load_artifact('final_student_dep_model.pkl')

class StudentInput(BaseModel):
    Gender: str = Field(..., example='Male')
    Age: float = Field(..., example=22.0)
    Academic_Pressure: float = Field(..., example=3.0)
    CGPA: float = Field(..., example=3.6)
    Study_Satisfaction: float = Field(..., example=4.0)
    Job_Satisfaction: float = Field(..., example=0.0)
    Sleep_Duration: str = Field(..., example='5-6 hours')
    Dietary_Habits: str = Field(..., example='Healthy')
    Work_Study_Hours: float = Field(..., example=6.0)
    Financial_Stress: float = Field(..., example=2.0)
    Family_History_of_Mental_Illness: str = Field(..., example='No')

GENDER_MAP = {'Male': 1, 'Female': 0}
SLEEP_MAP = {'Less than 5 hours': 0, '5-6 hours': 1, '7-8 hours': 2, 'More than 8 hours': 3}
DIET_MAP = {'Unhealthy': 0, 'Moderate': 1, 'Healthy': 2}
FAMILY_MAP = {'No': 0, 'Yes': 1}

@app.get('/')
def home():
    return {'message': 'Student Depression Prediction API is active.'}

@app.post('/predict')
def predict_depression(data: StudentInput):
    try:
        df = pd.DataFrame([{
            'Gender': GENDER_MAP.get(data.Gender, 0),
            'Age': data.Age,
            'Academic Pressure': data.Academic_Pressure,
            'CGPA': data.CGPA,
            'Study Satisfaction': data.Study_Satisfaction,
            'Job Satisfaction': data.Job_Satisfaction,
            'Sleep Duration': SLEEP_MAP.get(data.Sleep_Duration, 1),
            'Dietary Habits': DIET_MAP.get(data.Dietary_Habits, 1),
            'Work/Study Hours': data.Work_Study_Hours,
            'Financial Stress': data.Financial_Stress,
            'Family History of Mental Illness': FAMILY_MAP.get(data.Family_History_of_Mental_Illness, 0)
        }])
        df = df[scaler.feature_names_in_]
        X_scaled = scaler.transform(df)
        X_pca = pca.transform(X_scaled)
        prediction = model.predict(X_pca)[0]
        probability = model.predict_proba(X_pca)[0][1]
        return {'prediction': int(prediction), 'status': 'High Risk' if prediction == 1 else 'Low Risk', 'probability': round(float(probability), 4)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f'Pipeline processing error: {str(e)}')
