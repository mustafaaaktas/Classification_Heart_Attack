from src.imports import *
from src.data_processing import load_data
from src.visuals import plot_column_distribution

# Define the file path as a direct relative path
file_path = '../data/heart_attack.csv'

# Load the data
raw_data = load_data(file_path)
# There are samples and one positive one negative for troponin level 10.
data = raw_data[raw_data['troponin'] < 10]

# Quick EDA
print(data.head())
print(data.dtypes)
print(data.describe())

# Visually looking into the relation of the features with class parameter
plot_column_distribution(data, 'troponin', 'class')



