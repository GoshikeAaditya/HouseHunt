import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load and preprocess data
df = pd.read_csv('dataset_indian.csv')
df = df.drop(['Unnamed: 0', 'Date'], axis=1)
df = df.fillna(0)

# Encode categorical variables
location_encoder = LabelEncoder()
city_encoder = LabelEncoder()
df['Location'] = df['Location'].astype(str)
df['City'] = df['City'].astype(str)
df['Location'] = location_encoder.fit_transform(df['Location'])
df['City'] = city_encoder.fit_transform(df['City'])

# Scale numerical features
scaler = StandardScaler()
numerical_features = ['Price', 'Area', 'No. of Bedrooms', 'Latitude', 'Longitude']
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# Prepare features and target for price prediction
feature_columns = ['Area', 'No. of Bedrooms', 'City', 
                   'Gymnasium', 'SwimmingPool', 'LandscapedGardens', 
                   'JoggingTrack', 'RainWaterHarvesting', 'ClubHouse',
                   'CarParking']
X = df[feature_columns]
y = df['Price']

# Train Random Forest regression model
price_model = RandomForestRegressor(n_estimators=100, random_state=42)
price_model.fit(X, y)

# Save the model and encoders
joblib.dump(price_model, 'price_model.joblib')
joblib.dump(city_encoder, 'city_encoder.joblib')
joblib.dump(scaler, 'scaler.joblib')

print("Random Forest price prediction model, city encoder, and scaler saved successfully.")