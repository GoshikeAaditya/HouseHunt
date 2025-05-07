from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import joblib
import numpy as np
import pandas as pd

from house_recommender import HouseRecommender

# Load price prediction model and encoders
price_model = joblib.load('price_model.joblib')
city_encoder = joblib.load('city_encoder.joblib')
scaler = joblib.load('scaler.joblib')

# Initialize recommender
recommender = HouseRecommender('dataset_indian.csv')

app = FastAPI()

class Preferences(BaseModel):
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    bedrooms: Optional[int] = None
    city: Optional[str] = None
    amenities: Optional[List[str]] = []

class PricePreferences(BaseModel):
    area: float
    bedrooms: int
    city: str
    amenities: Optional[List[str]] = []

@app.post("/recommend")
def recommend(preferences: Preferences):
    prefs = preferences.dict(exclude_none=True)
    try:
        recommendations = recommender.get_recommendations(prefs)
        if recommendations.empty:
            return {"recommendations": []}
        # Convert DataFrame to list of dicts for JSON response
        return {"recommendations": recommendations.to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_price")
def predict_price(preferences: PricePreferences):
    amenities = [amenity.strip().lower() for amenity in preferences.amenities]
    try:
        input_dict = {
            'Area': preferences.area,
            'No. of Bedrooms': preferences.bedrooms,
            'City': city_encoder.transform([preferences.city])[0],
            'Gymnasium': 1 if 'gymnasium' in amenities else 0,
            'SwimmingPool': 1 if 'swimmingpool' in amenities else 0,
            'LandscapedGardens': 1 if 'landscapedgardens' in amenities else 0,
            'JoggingTrack': 1 if 'joggingtrack' in amenities else 0,
            'RainWaterHarvesting': 1 if 'rainwaterharvesting' in amenities else 0,
            'ClubHouse': 1 if 'clubhouse' in amenities else 0,
            'CarParking': 1 if 'carparking' in amenities else 0,
        }
        input_df = pd.DataFrame([input_dict])
        # Scale area and bedrooms
        dummy = np.zeros((1, 5))
        dummy[0, 1] = input_dict['Area']
        dummy[0, 2] = input_dict['No. of Bedrooms']
        scaled = scaler.transform(dummy)
        input_df['Area'] = scaled[0, 1]
        input_df['No. of Bedrooms'] = scaled[0, 2]
        # Predict scaled price
        predicted_price_scaled = price_model.predict(input_df)[0]
        dummy[0, 0] = predicted_price_scaled
        price = scaler.inverse_transform(dummy)[0, 0]
        return {"predicted_price": price}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))