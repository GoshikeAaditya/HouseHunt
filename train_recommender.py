import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

def load_dataset(file_path):
    """Load the preprocessed dataset from a CSV file."""
    return pd.read_csv(file_path)

def train_recommender(df):
    """Train a recommender model using the preprocessed dataset."""
    # Check available columns
    print("Available columns in the dataset:")
    print(df.columns)

    # Define features and target
    # Adjust the feature list based on available columns
    features = ["City", "Price", "No. of Bedrooms", "CarParking"]
    target = "Price"

    # Ensure features exist in the dataset
    missing_features = [feature for feature in features if feature not in df.columns]
    if missing_features:
        print(f"Missing features in the dataset: {missing_features}")
        return

    # Prepare the data
    X = df[features]
    y = df[target]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate the model
    score = model.score(X_test, y_test)
    print(f"Model R^2 score: {score}")

    # Save the model
    with open("recommender_model.pkl", "wb") as f:
        pickle.dump(model, f)

def main():
    # Load the preprocessed dataset
    file_path = "/Users/aaditya/Documents/househunt/dataset_indian.csv"
    df = load_dataset(file_path)

    # Train the recommender model
    train_recommender(df)

if __name__ == "__main__":
    main()