from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from xgboost import XGBRegressor
import numpy as np

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model
model = XGBRegressor()
model.load_model("model/model.json")

class SalesInput(BaseModel):
    sales_lag_1: float
    sales_lag_2: float
    sales_lag_3: float
    sales_lag_4: float
    sales_lag_5: float
    sales_lag_6: float

@app.post("/predict")
def predict_sales(data: SalesInput):
    features = np.array([[
        data.sales_lag_1,
        data.sales_lag_2,
        data.sales_lag_3,
        data.sales_lag_4,
        data.sales_lag_5,
        data.sales_lag_6
    ]])
    
    prediction = model.predict(features)[0]
    
    return {
        "predicted_sales": float(prediction)
    }

@app.get("/")
def read_root():
    return {"message": "Sales forecasting API is running."}