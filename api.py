from fastapi import FastAPI
import pandas as pd
from threshold.threshold_classifier import ThresholdClassifier
from threshold.tuning import find_best_threshold_f1
from pydantic import BaseModel
from typing import List
import pickle




# FastAPI app
app = FastAPI(title="Employee Attrition Predictor")

with open("models/model.pkl","rb") as f:
    model = pickle.load(f)





class AttritionInput(BaseModel):
    Age: int
    BusinessTravel: str
    DailyRate: int
    Department: str
    DistanceFromHome: int
    Education: int
    EducationField: str
    EnvironmentSatisfaction: int
    Gender: str
    HourlyRate: int
    JobInvolvement: int
    JobLevel: int
    JobRole: str
    JobSatisfaction: int
    MaritalStatus: str
    MonthlyIncome: int
    MonthlyRate: int
    NumCompaniesWorked: int
    OverTime: str
    PercentSalaryHike: int
    PerformanceRating: int
    RelationshipSatisfaction: int
    StockOptionLevel: int
    TotalWorkingYears: int
    TrainingTimesLastYear: int
    WorkLifeBalance: int
    YearsAtCompany: int
    YearsInCurrentRole: int
    YearsSinceLastPromotion: int
    YearsWithCurrManager: int


# Health check
# -----------------------------
@app.get("/")
def health():
    return {"status": "API is live"}


@app.post("/predict")
def predict_attrition(data: AttritionInput):
    df = pd.DataFrame([data.dict()])
    pred = model.predict(df)[0]
    prob = model.predict_proba(df)[0][1]

    return {
        "attrition_prediction": int(pred),
        "attrition_probability": round(float(prob), 4)
    }

@app.post("/predict-batch")
def predict_attrition_batch(data: List[AttritionInput]):
    df = pd.DataFrame([d.dict()for d in data])
    preds = model.predict(df)
    probs = model.predict_proba(df)[:, 1]

    results = [
        {
            "record_id": i,
            "attrition_prediction": int(preds[i]),
            "attrition_probability": round(float(probs[i]), 4)
        }
        for i in range(len(df))
    ]

    return {
        "total_records": len(df),
        "results": results
    }
    
    
    
    
    
    
    
    
    
    
    
    