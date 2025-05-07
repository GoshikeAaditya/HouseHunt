import joblib
import numpy as np
import pandas as pd

# Load the trained model and encoders
price_model = joblib.load('price_model.joblib')
city_encoder = joblib.load('city_encoder.joblib')
scaler = joblib.load('scaler.joblib')

def predict_price(preferences):
    # Prepare input features
    input_dict = {
        'Area': preferences.get('area', 0),
        'No. of Bedrooms': preferences.get('bedrooms', 0),
        'City': city_encoder.transform([preferences['city']])[0] if 'city' in preferences else 0,
        'Gymnasium': 1 if 'amenities' in preferences and 'Gymnasium' in preferences['amenities'] else 0,
        'SwimmingPool': 1 if 'amenities' in preferences and 'SwimmingPool' in preferences['amenities'] else 0,
        'LandscapedGardens': 1 if 'amenities' in preferences and 'LandscapedGardens' in preferences['amenities'] else 0,
        'JoggingTrack': 1 if 'amenities' in preferences and 'JoggingTrack' in preferences['amenities'] else 0,
        'RainWaterHarvesting': 1 if 'amenities' in preferences and 'RainWaterHarvesting' in preferences['amenities'] else 0,
        'ClubHouse': 1 if 'amenities' in preferences and 'ClubHouse' in preferences['amenities'] else 0,
        'CarParking': 1 if 'amenities' in preferences and 'CarParking' in preferences['amenities'] else 0,
    }
    input_df = pd.DataFrame([input_dict])

    # The model expects scaled features, so scale the numerical columns
    # We'll use the scaler to transform only the relevant columns
    # The scaler was fit on ['Price', 'Area', 'No. of Bedrooms', 'Latitude', 'Longitude']
    # We'll create a dummy row for scaling
    dummy = np.zeros((1, 5))
    dummy[0, 1] = input_dict['Area']
    dummy[0, 2] = input_dict['No. of Bedrooms']
    scaled = scaler.transform(dummy)
    input_df['Area'] = scaled[0, 1]
    input_df['No. of Bedrooms'] = scaled[0, 2]

    # Predict the scaled price
    predicted_price_scaled = price_model.predict(input_df)[0]

    # Inverse transform to get the original price
    dummy[0, 0] = predicted_price_scaled
    price = scaler.inverse_transform(dummy)[0, 0]
    return price

if __name__ == "__main__":
    # Example usage
    preferences = {
        'area': 1200,
        'bedrooms': 3,
        'city': 'Hyderabad',
        'amenities': ['Gymnasium', 'SwimmingPool']
    }
    predicted_price = predict_price(preferences)
    print(f"Predicted Price for given preferences: {predicted_price:.2f}")