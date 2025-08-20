import pandas as pd
from sklearn.linear_model import LinearRegression

class DatasetManager:
    def __init__(self, path="data/dataset.csv"):
        # Load dataset
        self.df = pd.read_csv(path)
        self.models = {}  # store trained models

    def get_features(self):
        """Return all features except target column"""
        # Assuming last column is target (like 'price')
        return list(self.df.columns[:-1])

    def get_target(self):
        """Return the target column name"""
        return self.df.columns[-1]