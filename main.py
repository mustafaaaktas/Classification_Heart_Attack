from src.data_processing import load_data
from src.imports import *
from src.visuals import plot_column_distribution

# Define the file path as a direct relative path
file_path = '../data/heart_attack.csv'

# Load the data
data = load_data(file_path)

# Quick EDA
print(data.head())
print(data.dtypes)
print(data.describe())

# Visually looking into the relation of the features with class parameter
plot_column_distribution(data, 'age', 'class')



