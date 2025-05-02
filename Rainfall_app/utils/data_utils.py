import pandas as pd
import pickle
import os

def load_feature_data():
    file_path = 'Data/feature_engineered_data.csv'
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    return pd.read_csv(file_path)

def load_reg_model():
    file_path = 'Data/best_random_forest_regressor_model.pkl'
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def load_clf_model():
    file_path = 'Data/best_random_forest_classifier_model.pkl'
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def load_nlp_results():
    file_path = 'Data/nlp_results.csv'
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    return pd.read_csv(file_path)

def load_lda_topics():
    file_path = 'Data/lda_topics.txt'
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(file_path, 'r') as f:
        return f.read()

def load_regional_performance_regression():
    file_path = 'Data/regional_performance_regression.csv'
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    return pd.read_csv(file_path, index_col='station_id')

def load_regional_performance_classification():
    file_path = 'Data/regional_performance_classification.csv'
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    return pd.read_csv(file_path, index_col='station_id')

def load_model_evaluation_results():
    file_path = "Data/model_evaluation_results.csv"
    print(f"Checking file path: {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")