import pandas as pd


def load_data(file_path):
    """
    Load the dataset from the specified file path.

    Parameters:
    - file_path (str): The path to the CSV file to load.

    Returns:
    - pd.DataFrame: The loaded dataset as a DataFrame.
    """
    raw_data = pd.read_csv(file_path)
    data = raw_data[raw_data['troponin'] < 10]
    print("Data loaded successfully.")
    return data
