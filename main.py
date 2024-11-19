from src.imports import *
from src.data_processing import load_data
from src.k_neighbors_classifier import find_best_knn_with_gridsearch
from src.random_forest_classifier import find_best_rf_with_gridsearch
from src.support_vector_machine import find_best_svm_with_gridsearch
from src.visuals import plot_column_distribution
from src.decision_tree_classifier import find_best_dt_with_gridsearch

file_path = '../data/heart_attack.csv'

# Load the data
data = load_data(file_path)


# Quick EDA
print("Data preview (first 5 rows):")
print(data.head())

print("\nData types:")
print(data.dtypes)

print("\nData summary:")
print(data.describe())


# Visually looking into the relation of the features with class parameter
plot_column_distribution(data, 'troponin', 'class')


# MODELS #


# Decision Tree #

best_params_dt, best_score_dt, test_accuracy_dt, best_model_dt \
    = find_best_dt_with_gridsearch(data)

print("Model:", best_model_dt)
print("Test Accuracy:", test_accuracy_dt)


# Random Forest #

best_params_rf, best_score_rf, test_accuracy_rf, best_model_rf \
    = find_best_rf_with_gridsearch(data)

print("Model:", best_model_rf)
print("Test Accuracy:", test_accuracy_rf)


# K-Nearest Neighbors (KNN) #

best_params_knn, best_score_knn, test_accuracy_knn, best_model_knn \
    = find_best_knn_with_gridsearch(data)

print("Model:", best_model_knn)
print("Test Accuracy:", test_accuracy_knn)


# Support Vector Machine (SVM) #

best_params_svm, best_score_svm, test_accuracy_svm, best_model_svm \
    = find_best_svm_with_gridsearch(data)

print("Model:", best_model_svm)
print("Test Accuracy:", test_accuracy_svm)


# Results #

result = pd.DataFrame({
    'Algorithms': ['Decision Tree', 'Random Forest', 'KNN', 'SVM'],
    'Accuracy': [test_accuracy_dt, test_accuracy_rf,
                 test_accuracy_knn, test_accuracy_svm]
})

print(f"Results:", result)
