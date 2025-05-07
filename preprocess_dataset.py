import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_dataset(file_path):
    """Load the dataset from a CSV file."""
    return pd.read_csv(file_path)

def explore_dataset(df):
    """Perform exploratory data analysis on the dataset."""
    print("First few rows of the dataset:")
    print(df.head())

    print("\nDataset Info:")
    print(df.info())

    print("\nSummary Statistics:")
    print(df.describe())

    print("\nColumn Names:")
    print(df.columns)

def preprocess_dataset(df):
    """Preprocess the dataset."""
    # Drop unnecessary columns
    df = df.drop(columns=["Unnamed: 0", "Date"])

    # Handle missing values
    df = df.fillna(df.mean())

    # Encode categorical variables
    label_encoders = {}
    for column in ["Location", "City"]:
        # Fill missing values with a placeholder
        df[column] = df[column].fillna("Unknown")
        # Ensure the column is of type object
        df[column] = df[column].astype(str)
        label_encoders[column] = LabelEncoder()
        df[column] = label_encoders[column].fit_transform(df[column])

    # Scale numerical features
    scaler = StandardScaler()
    numerical_features = ["Price", "Area", "No. of Bedrooms", "Latitude", "Longitude"]
    df[numerical_features] = scaler.fit_transform(df[numerical_features])

    return df

def main():
    # Load the dataset
    file_path = "dataset_indian.csv"
    df = load_dataset(file_path)

    # Explore the dataset
    explore_dataset(df)

    # Preprocess the dataset
    df_preprocessed = preprocess_dataset(df)
    print("\nPreprocessed Dataset:")
    print(df_preprocessed.head())

    # Save the preprocessed dataset
    df_preprocessed.to_csv("preprocessed_dataset.csv", index=False)

if __name__ == "__main__":
    main()