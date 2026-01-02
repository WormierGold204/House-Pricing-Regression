import pandas as pd
from sklearn.impute import SimpleImputer
from eda import target_variable_distribution, correlation_heatmap, plots, dataset_statistics
import os

class DatasetManager:
    def __init__(self, path="data/dataset.csv"):

        # Create a folder for eda images, if it does not already exist
        if not os.path.exists("static/eda"):
            os.makedirs("static/eda")
        
        # Load dataset
        self.df = pd.read_csv(path)

        # Obtain general statistics and missing values statistics on dataset
        self.statistics = dataset_statistics(self.df)

        # Create a copy of dataset to apply preprocessing
        self.df_copy = self.df.copy()

        # Handling of NA values
        # Drop Electrical row (just one sample)
        self.df_copy = self.df_copy.dropna(subset=['Electrical'])

        # NA values here show lack of feature: fill with string None (categorical) or 0 (numeric) to represent absence
        categorical_absence = [
            'PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
            'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
            'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
            'MasVnrType'
        ]
        numeric_absence = [
            'GarageYrBlt', 'MasVnrArea'
        ]
        self.df_copy[categorical_absence] = self.df_copy[categorical_absence].fillna('None')
        self.df_copy[numeric_absence] = self.df_copy[numeric_absence].fillna(0)

        # LotFrontage NA values mean lack of information: fill with median value
        self.df_copy['LotFrontage'] = self.df_copy['LotFrontage'].fillna(self.df_copy['LotFrontage'].median())

        # Create new derived features to enrich SalePrice's correlations
        self.df_copy['TotalSF'] = self.df_copy['TotalBsmtSF'] + self.df_copy['1stFlrSF'] + self.df_copy['2ndFlrSF'] # Total House Square Feet
        self.df_copy['HouseAge'] = self.df_copy['YrSold'] - self.df_copy['YearBuilt']   # House Age when the house was sold
        self.df_copy['TotalBath'] = self.df_copy['BsmtFullBath'] + 0.5*self.df_copy['BsmtHalfBath'] + self.df_copy['FullBath'] + 0.5*self.df_copy['HalfBath']

        # Generate images for EDA
        self.images_paths = {
            "target_distribution": target_variable_distribution(self.df_copy),
            "correlation_heatmap": correlation_heatmap(self.df_copy),
            "plots": plots(self.df_copy)
        }

    def get_features(self):
        """Return all features except target column.
        
        Returns:
        list: A list of feature column names."""
        # Target column is assumed to be the last one
        return list(self.df.columns[:-1])
    
    def get_target(self):
        return self.df.columns[-1]
