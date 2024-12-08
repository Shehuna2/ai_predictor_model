from fastapi import FastAPI
from pydantic import BaseModel
from api.model import load_model, predict
# from api.data_fetch import fetch_live_data



app = FastAPI()

# Define the input structure
class MarketData(BaseModel):
    price: float
    volume: float
    time: str

# Load the model at startup
model = load_model()

@app.post("/predict")
async def predict_market(data: MarketData):
    # Use input data for predictions
    prediction = predict(model, data.dict())
    return {"prediction": prediction}
