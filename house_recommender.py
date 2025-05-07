import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

class HouseRecommender:
    def __init__(self, dataset_path):
        """Initialize the recommender system."""
        self.df = pd.read_csv(dataset_path)
        self.preprocess_data()
        self.train_model()
        
    def preprocess_data(self):
        """Preprocess the dataset."""
        # Drop unnecessary columns
        self.df = self.df.drop(['Unnamed: 0', 'Date'], axis=1)
        
        # Handle missing values
        self.df = self.df.fillna(0)
        
        # Encode categorical variables
        self.location_encoder = LabelEncoder()
        self.city_encoder = LabelEncoder()
        
        # Convert to string type before encoding
        self.df['Location'] = self.df['Location'].astype(str)
        self.df['City'] = self.df['City'].astype(str)
        
        self.df['Location'] = self.location_encoder.fit_transform(self.df['Location'])
        self.df['City'] = self.city_encoder.fit_transform(self.df['City'])
        
        # Scale numerical features
        self.scaler = StandardScaler()
        numerical_features = ['Price', 'Area', 'No. of Bedrooms', 'Latitude', 'Longitude']
        self.df[numerical_features] = self.scaler.fit_transform(self.df[numerical_features])

    def train_model(self):
        """Train the recommendation model."""
        feature_columns = ['Price', 'Area', 'No. of Bedrooms', 'Location', 
                         'Gymnasium', 'SwimmingPool', 'LandscapedGardens', 
                         'JoggingTrack', 'RainWaterHarvesting', 'ClubHouse',
                         'CarParking', 'Latitude', 'Longitude']
        
        # Convert to DataFrame to preserve feature names
        features_df = self.df[feature_columns].copy()
        self.model = NearestNeighbors(n_neighbors=5, metric='cosine')
        self.model.fit(features_df)
        
    def get_recommendations(self, preferences):
        """
        Get house recommendations based on user preferences.
        """
        # Filter dataset based on hard constraints
        df_filtered = self.df.copy()
        if 'min_price' in preferences and 'max_price' in preferences:
            min_price_scaled = self.scaler.transform([[preferences['min_price'], 0, 0, 0, 0]])[0][0]
            max_price_scaled = self.scaler.transform([[preferences['max_price'], 0, 0, 0, 0]])[0][0]
            df_filtered = df_filtered[(df_filtered['Price'] >= min_price_scaled) & (df_filtered['Price'] <= max_price_scaled)]
        if 'bedrooms' in preferences:
            bedrooms_scaled = self.scaler.transform([[0, 0, preferences['bedrooms'], 0, 0]])[0][2]
            df_filtered = df_filtered[df_filtered['No. of Bedrooms'] == bedrooms_scaled]
        if 'city' in preferences:
            try:
                city_encoded = self.city_encoder.transform([preferences['city']])[0]
                df_filtered = df_filtered[df_filtered['City'] == city_encoded]
            except ValueError:
                pass
        if 'amenities' in preferences:
            for amenity in preferences['amenities']:
                if amenity in df_filtered.columns:
                    df_filtered = df_filtered[df_filtered[amenity] == 1]

        # If no matches, fall back to original dataset
        if len(df_filtered) == 0:
            print("No strict matches found. Showing closest matches instead.")
            df_filtered = self.df

        # Create a reference point based on preferences
        reference_point = self.create_reference_point(preferences)

        # Find nearest neighbors in the filtered set
        feature_columns = ['Price', 'Area', 'No. of Bedrooms', 'Location', 
                           'Gymnasium', 'SwimmingPool', 'LandscapedGardens', 
                           'JoggingTrack', 'RainWaterHarvesting', 'ClubHouse',
                           'CarParking', 'Latitude', 'Longitude']
        n_neighbors = min(5, len(df_filtered))
        if n_neighbors == 0:
            print("No matches found. Try relaxing your preferences.")
            return pd.DataFrame()
        model = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
        model.fit(df_filtered[feature_columns])
        distances, indices = model.kneighbors([reference_point])

        # Get unique recommendations
        recommendations = df_filtered.iloc[indices[0]].copy()
        recommendations = recommendations.drop_duplicates()

        # Decode location and city for better readability
        recommendations['Location'] = self.location_encoder.inverse_transform(recommendations['Location'].astype(int))
        recommendations['City'] = self.city_encoder.inverse_transform(recommendations['City'].astype(int))

        # Unscale numerical features for better readability
        numerical_features = ['Price', 'Area', 'No. of Bedrooms', 'Latitude', 'Longitude']
        recommendations[numerical_features] = self.scaler.inverse_transform(recommendations[numerical_features])

        return recommendations
        
    def create_reference_point(self, preferences):
        """Create a reference point based on user preferences."""
        # Get mean values for all features
        reference_point = self.df.mean().to_dict()
        
        # Update with user preferences
        if 'min_price' in preferences and 'max_price' in preferences:
            reference_point['Price'] = (preferences['min_price'] + preferences['max_price']) / 2
        if 'bedrooms' in preferences:
            reference_point['No. of Bedrooms'] = preferences['bedrooms']
        if 'city' in preferences:
            try:
                reference_point['City'] = self.city_encoder.transform([preferences['city']])[0]
            except ValueError:
                reference_point['City'] = self.df['City'].mean()
        if 'amenities' in preferences:
            for amenity in preferences['amenities']:
                if amenity in reference_point:
                    reference_point[amenity] = 1
        
        # Convert to array in correct order
        feature_columns = ['Price', 'Area', 'No. of Bedrooms', 'Location', 
                           'Gymnasium', 'SwimmingPool', 'LandscapedGardens', 
                           'JoggingTrack', 'RainWaterHarvesting', 'ClubHouse',
                           'CarParking', 'Latitude', 'Longitude']
        return np.array([reference_point[col] for col in feature_columns])

# Example usage
if __name__ == "__main__":
    # Initialize recommender
    recommender = HouseRecommender('dataset_indian.csv')
    
    # Example preferences
    preferences = {
        'min_price': 50,
        'max_price': 200,
        'bedrooms': 3,
        'city': 'Hyderabad',
        'amenities': []
    }
    
    # Get recommendations
    recommendations = recommender.get_recommendations(preferences)
    print("\nTop Recommendations:")
    print(recommendations[['Price', 'Area', 'No. of Bedrooms', 'Location', 'City']].to_string(index=False))