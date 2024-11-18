# src/data_processing.py

import pandas as pd


def load_data(file_path):
    """
    Load the dataset from the specified file path.

    Parameters:
    - file_path (str): The path to the CSV file to load.

    Returns:
    - pd.DataFrame: The loaded dataset as a DataFrame.
    """
    data = pd.read_csv(file_path)
    print("Data loaded successfully.")
    return data
