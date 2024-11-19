from imports import *
from data_processing import load_data


def split_data(data):
    """
    Split the data into training and testing sets.

    Parameters:
    - data (pd.DataFrame): The dataset containing features and target variable.

    Returns:
    - X_train, X_test, y_train, y_test: Split data
    """
    # Splitting features and target variable
    X = data.drop(columns='class')
    y = data['class']

    # Split into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    return X_train, X_test, y_train, y_test


def print_data_shapes(X_train, X_test, y_train, y_test):
    """
    Print the shapes of the training and testing data.
    """
    print('Shape of X_train set: {}'.format(X_train.shape))
    print('Shape of y_train set: {}'.format(y_train.shape))
    print('_' * 50)
    print('Shape of X_test set: {}'.format(X_test.shape))
    print('Shape of y_test set: {}'.format(y_test.shape))
