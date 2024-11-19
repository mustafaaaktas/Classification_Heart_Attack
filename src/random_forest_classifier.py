from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from src.data_processing import load_data


def find_best_rf_with_gridsearch(data):
    """
    Function to find the best parameters for a Random Forest Classifier using GridSearchCV with cross-validation.

    Parameters:
    - data: The complete dataset with features and target variable

    Returns:
    - best_params: The best hyperparameters found by GridSearchCV
    - best_score: The best cross-validation score
    """
    # Split data into features (X) and target (y)
    X = data.drop(columns='class')  # Features (excluding target column)
    y = data['class']  # Target variable (class)

    # Label encode the target variable (class)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Scale numerical features (e.g., age, glucose, etc.)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)  # Apply scaling

    # Split data into train and test sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=0)

    # Define hyperparameter grid for GridSearchCV
    param_grid = {
        'n_estimators': [50, 100, 200],
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 3, 5, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Initialize the Random Forest Classifier
    rf_classifier = RandomForestClassifier(n_jobs=-1, random_state=0)

    # Initialize GridSearchCV with 5-fold cross-validation
    grid_search = GridSearchCV(rf_classifier, param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)

    # Fit the model using GridSearchCV
    grid_search.fit(X_train, y_train)

    # Get the best parameters and best cross-validation score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    # Get the best model (estimator)
    best_model = grid_search.best_estimator_

    # Evaluate the best model on the test set
    y_pred = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)

    # Print the results
    print("Best Parameters from GridSearchCV:", best_params)
    print("Best Cross-Validation Accuracy:", best_score)
    print("Test Accuracy on Hold-Out Set:", test_accuracy)
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Visualization 1: Feature Importance Plot
    feature_importances = best_model.feature_importances_
    features = X.columns
    feature_importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10,6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
    plt.title("Feature Importance in Random Forest Classifier")
    plt.show()

    # Visualization 2: Confusion Matrix Heatmap
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(7,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    return best_params, best_score, test_accuracy, best_model
