import pandas as pd
import numpy as np

# Models from scikit-learn
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler  # Feature scaling
from sklearn.metrics import (accuracy_score,
                             confusion_matrix,
                             classification_report)

# Visualizations
import matplotlib.pyplot as plt
import seaborn as sns
