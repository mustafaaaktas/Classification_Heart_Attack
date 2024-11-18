from src.data_processing import load_data

# Define the file path as a direct relative path
file_path = '../data/heart_attack.csv'

# Load the data
data = load_data(file_path)

print(data.head())
print(data.dtypes)
print(data.describe())

