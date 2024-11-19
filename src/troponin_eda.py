import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = '../data/heart_attack.csv'
raw_data = pd.read_csv(file_path)
data = raw_data[raw_data['troponin'] < 10]

# 1. Summary statistics of troponin
print(data['troponin'].describe())

# 2. Distribution of Troponin
plt.figure(figsize=(10,6))
sns.histplot(data['troponin'], kde=True, bins=30, color='skyblue')
plt.title('Distribution of Troponin Levels')
plt.xlabel('Troponin Level')
plt.ylabel('Frequency')
plt.show()

# 3. Boxplot of Troponin by Class (positive/negative)
# Updated: Assign 'class' to hue explicitly to avoid the FutureWarning
plt.figure(figsize=(10,6))
sns.boxplot(x='class', y='troponin', data=data, hue='class', palette='Set2', legend=False)
plt.title('Troponin Level by Heart Attack Class')
plt.xlabel('Class (Positive/Negative)')
plt.ylabel('Troponin Level')
plt.show()

# 4. Violin Plot for better distribution visualization
# Updated: Assign 'class' to hue explicitly to avoid the FutureWarning
plt.figure(figsize=(10,6))
sns.violinplot(x='class', y='troponin', data=data, hue='class', palette='Set2', legend=False)
plt.title('Troponin Level Distribution by Heart Attack Class')
plt.xlabel('Class (Positive/Negative)')
plt.ylabel('Troponin Level')
plt.show()

# 5. Scatter plot of Troponin vs Other Features (e.g., Age, Glucose)
# Scatter plot of Troponin vs Age
plt.figure(figsize=(10,6))
sns.scatterplot(x='age', y='troponin', data=data, hue='class', palette='Set1')
plt.title('Troponin vs Age')
plt.xlabel('Age')
plt.ylabel('Troponin Level')
plt.legend(title='Class')
plt.show()

# Scatter plot of Troponin vs Glucose
plt.figure(figsize=(10,6))
sns.scatterplot(x='glucose', y='troponin', data=data, hue='class', palette='Set1')
plt.title('Troponin vs Glucose')
plt.xlabel('Glucose Level')
plt.ylabel('Troponin Level')
plt.legend(title='Class')
plt.show()

# 6. Correlation Matrix to see how Troponin relates to other features
# Filter out non-numeric columns for correlation calculation
numerical_data = data.select_dtypes(include=[float, int])
correlation_matrix = numerical_data.corr()

plt.figure(figsize=(10,8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()
