import pandas as pd

def load_dataset(file_path):
    """Load the dataset from a CSV file."""
    return pd.read_csv(file_path)

def explore_dataset(df):
    """Explore the dataset by printing basic information and statistics."""
    print("First few rows of the dataset:")
    print(df.head())

    print("\nDataset Info:")
    print(df.info())

    print("\nSummary Statistics:")
    print(df.describe())

    print("\nColumn Names:")
    print(df.columns)

def main():
    # Load the dataset
    file_path = "dataset_indian.csv"  # Change this to your dataset file path
    df = load_dataset(file_path)

    # Explore the dataset
    explore_dataset(df)

if __name__ == "__main__":
    main()