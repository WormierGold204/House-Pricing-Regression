import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from eda import target_variable_distribution, correlation_heatmap, plots, dataset_statistics

class DatasetManager:
    def __init__(self, path="data/dataset.csv"):
        # Load dataset
        self.df = pd.read_csv(path)

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

        # Generate analysis' images for EDA
        self.images_paths = {
            "target_distribution": target_variable_distribution(self.df),
            "correlation_heatmap": correlation_heatmap(self.df),
            "plots": plots(self.df)
        }

    def get_features(self):
        """Return all features except target column"""
        # Target column is assumed to be the last one
        return list(self.df.columns[:-1])

    def get_target(self):
        """Return the target column name"""
        return self.df.columns[-1]

    

    def training_preprocessing(self):
        """ Preprocess the dataset for training """
        # Separate categorical and numerical features
        numeric_features = self.df.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = self.df.select_dtypes(include=['object']).columns

        # Encode Categorical Variables using OneHotEncoder and handling unknown categories by ignoring them
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        encoded = encoder.fit_transform(self.df[categorical_features])

        # Convert the encoded features back to a DataFrame
        encoded_df = pd.DataFrame(
            encoded, 
            columns=encoder.get_feature_names_out(categorical_features)
        )

        # Create a DataFrame with the encoded features merged with numeric ones
        df_processed = pd.concat(
            [self.df[numeric_features].reset_index(drop=True), encoded_df], 
            axis=1
        )

        # Separate target variable (SalePrice) and the other features
        y = df_processed["SalePrice"]
        X = df_processed.drop("SalePrice", axis=1)

        # Scale numeric features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

        # Save mapping: raw feature -> encoded columns
        # This is needed for feature selection in the training phase
        self.feature_map = {}
        for feature in categorical_features:
            self.feature_map[feature] = [
                col for col in encoded_df.columns if col.startswith(feature + "_")
            ]
        for feature in numeric_features:
            if feature != "SalePrice":
                self.feature_map[feature] = [feature]

        return X_scaled, y