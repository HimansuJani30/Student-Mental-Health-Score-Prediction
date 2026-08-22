from pathlib import Path

import joblib
import uvicorn
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field
from typing import Literal
from fastapi.middleware.cors import CORSMiddleware
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "Mental_Health_Model.pkl"

model = joblib.load(MODEL_PATH)


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


# A first pydantic model-in this we are going to make a VALIDATION for the input data that we are goin to send to the model for prediction.

class StudentData(BaseModel):
     model_config = ConfigDict(populate_by_name=True)

     Age                    : int = Field(..., ge=10, le=100, alias='age')
     Gender                 : Literal['Male', 'Female'] = Field(alias='gender')
     Country                : str = Field(alias='country')
     Academic_Level         : Literal['Undergraduate', 'Graduate', 'High School'] = Field(alias='academic_level')
     Most_Used_Platform     : Literal['Facebook', 'LinkedIn', 'Instagram', 'Snapchat', 'Twitter',
       'YouTube', 'TikTok', 'LINE', 'KakaoTalk', 'VKontakte', 'WhatsApp',
         'WeChat'] = Field(alias='most_used_platform')
     Purpose_Of_Use         : Literal['Networking', 'Education', 'Entertainment', 'News'] = Field(alias='purpose_of_use')
     Avg_Daily_Usage_Hours  : float = Field(..., ge=0, le=24, alias='avg_daily_usage_hours')
     Daily_Unlocks          : int = Field(..., ge=0, alias='daily_unlocks')
     Study_Hours            : float = Field(..., ge=0, le=24, alias='study_hours')
     Physical_Activity_Hours: float = Field(..., ge=0, le=24, alias='physical_activity_hours')
     Sleep_Hours_Per_Night  : float = Field(..., ge=0, le=24, alias='sleep_hours_per_night')
     Stress_Level           : Literal['Medium', 'Low', 'Very High', 'High'] = Field(alias='stress_level')


# we are going to create a response body.
class PredictionResponse(BaseModel):
    predicted_mental_health_score: float


  

@app.get('/')
def greet():
    return {"Welcome to D-town Guys"}


top_countries = ['Other',
 'India',
 'USA',
 'Canada',
 'Australia',
 'UK',
 'Germany',
 'Mexico',
 'Turkey',
 'France']


@app.post('/predict',response_model = PredictionResponse)

def predict(data: StudentData):# studentdata is a class.
    country_group = data.Country if data.Country in top_countries else 'Other'
    
    input_row = pd.DataFrame([{
        "Age"                   : data.Age,
        "Gender"                : data.Gender,
        "Country"               : data.Country,
        "Academic_Level"        : data.Academic_Level,
        "Most_Used_Platform"    : data.Most_Used_Platform,
        "Purpose_Of_Use"        : data.Purpose_Of_Use,
        "Avg_Daily_Usage_Hours" : data.Avg_Daily_Usage_Hours,
        "Daily_Unlocks"         : data.Daily_Unlocks,
        "Study_Hours"           : data.Study_Hours,
        "Physical_Activity_Hours" : data.Physical_Activity_Hours,
        "Sleep_Hours_Per_Night" : data.Sleep_Hours_Per_Night,
        "Stress_Level"          : data.Stress_Level,
        "Grouped_country"       : country_group
    }])

    prediction = model.predict(input_row)[0]
    return PredictionResponse(predicted_mental_health_score=round(float(prediction),2))


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
   