from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import random

app = FastAPI()

class RecommendationRequest(BaseModel):
    min_price: float
    max_price: float
    bedrooms: int
    city: str
    amenities: List[str]

class PricePredictionRequest(BaseModel):
    area: float
    bedrooms: int
    city: str
    amenities: List[str]

@app.post("/recommend")
async def recommend(request: RecommendationRequest):
    # This is a mock implementation - in a real app, you would query a database
    # or use a machine learning model to get recommendations
    
    # Generate some mock data based on the request parameters
    num_recommendations = random.randint(3, 8)
    recommendations = []
    
    for _ in range(num_recommendations):
        price = random.uniform(request.min_price, request.max_price)
        area = random.uniform(800, 3000)
        location = f"Area {random.randint(1, 10)}"
        
        recommendations.append({
            "Price": price,
            "Area": area,
            "No. of Bedrooms": request.bedrooms,
            "Location": location,
            "City": request.city,
            "Amenities": request.amenities
        })
    
    return {"recommendations": recommendations}

@app.post("/predict_price")
async def predict_price(request: PricePredictionRequest):
    # This is a mock implementation - in a real app, you would use a 
    # trained machine learning model to predict the price
    
    # Simple formula for demonstration purposes
    base_price = request.area * 100  # Base price per sq ft
    bedroom_factor = request.bedrooms * 50000  # Additional price per bedroom
    amenity_factor = len(request.amenities) * 25000  # Additional price per amenity
    
    # City factor (could be more sophisticated in a real app)
    city_factors = {
        "Mumbai": 1.5,
        "Delhi": 1.3,
        "Bangalore": 1.4,
        "Hyderabad": 1.2,
        "Chennai": 1.1,
        "Pune": 1.0,
        "Kolkata": 0.9
    }
    
    city_factor = city_factors.get(request.city, 1.0)
    
    predicted_price = (base_price + bedroom_factor + amenity_factor) * city_factor
    
    return {"predicted_price": predicted_price}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)  # Changed port from 8000 to 8001