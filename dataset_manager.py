import pandas as pd
from sklearn.impute import SimpleImputer
from eda import target_variable_distribution, correlation_heatmap, plots, dataset_statistics
import os

class DatasetManager:
    def __init__(self, path="data/dataset.csv"):
        # Load dataset
        self.df = pd.read_csv(path)

        # Save a copy of the original DataFrame without preprocessing
        df_copy = self.df.copy()

        # Preprocess dataset, readying it for EDA
        # Separate categorical and numerical
        numeric_features = self.df.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = self.df.select_dtypes(include=['object']).columns

        # Fill numeric NaN with median, categorical with the most frequent value
        imputer_num = SimpleImputer(strategy="median")
        imputer_cat = SimpleImputer(strategy="most_frequent")

        self.df[numeric_features] = imputer_num.fit_transform(self.df[numeric_features])
        self.df[categorical_features] = imputer_cat.fit_transform(self.df[categorical_features])

        # Compute basic statistics of the dataset
        self.statistics = dataset_statistics(self.df)

        # Create a directory for EDA images if it doesn't exist
        if not os.path.exists("static/eda"):
            os.makedirs("static/eda")

        # Generate analysis' images for EDA
        self.images_paths = {
            "target_distribution": target_variable_distribution(self.df),
            "correlation_heatmap": correlation_heatmap(self.df),
            "plots": plots(self.df)
        }

        # Restore the original DataFrame for the pipeline
        self.df = df_copy

    def get_features(self):
        """Return all features except target column.
        
        Returns:
        list: A list of feature column names."""
        # Target column is assumed to be the last one
        return list(self.df.columns[:-1])

    def get_target(self):
        """Return the target column name.
        
        Returns:
        str: The target column name."""
        return self.df.columns[-1]