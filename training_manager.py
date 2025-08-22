from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import joblib
import json
import os

class TrainingManager:
    def __init__(self, dataset_manager):
        self.dataset_manager = dataset_manager

    def get_models(self):
        """Return the list of trained models."""
        with open("models/models.json", "r") as f:
            return json.load(f)

    def save_model(self, model, model_name):
        """Save the trained model to disk."""
        if not os.path.exists('models'):
            os.makedirs('models')

        # Save the model
        joblib.dump(model, os.path.join('models', f"{model_name}.pkl"))

    def load_model(self, model_name):
        """Load a trained model from disk."""
        model_path = os.path.join('models', f"{model_name}.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model {model_name} not found")

        # Return the loaded model
        return joblib.load(model_path)

    def update_metadata(self, metadata):
        """Update the models.json file with new model metadata."""
        json_path = os.path.join('models', "models.json")

        # Load existing metadata to avoid overwriting
        models = []
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                models = json.load(f)

        # Overwrite existing model metadata if the name matches
        models = [m for m in models if m["name"] != metadata["name"]]
        models.append(metadata)

        # Save updated metadata
        with open(json_path, "w") as f:
            json.dump(models, f, indent=4)

    def train_model(self, model_type, model_name, selected_features):
        # Obtain preprocessed data and keep only selected features
        if not selected_features:
            raise ValueError("No features selected for training")

        X = self.dataset_manager.df[selected_features]
        y = self.dataset_manager.df[self.dataset_manager.get_target()]

        # Split Categorical and Numeric Features
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X.select_dtypes(include=['object']).columns

        # Preprocessing to add in the pipeline

        # Preprocessing to handle missing values and scale numeric features
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        # Preprocessing to handle missing values and encode categorical features
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ])

        # Create a preprocessor that applies the transformations to the respective features
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )

        # Choose the model based on the user's choice
        if model_type == 'linear_regression':
            model = LinearRegression()
        elif model_type == 'gradient_boosting':
            model = GradientBoostingRegressor(random_state=42)
        elif model_type == 'random_forest':
            model = RandomForestRegressor(random_state=42)
        else:
            raise ValueError("Unsupported model type")

        # Create a pipeline that first preprocesses the data and then applies the model
        # This allows to not worry about the order of operations
        # and ensures that the same preprocessing is applied during training and prediction
        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                ('model', model)])

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train the model based on the pipeline
        pipeline.fit(X, y)

        # Save the model
        self.save_model(pipeline, model_name)

        # Evaluate the model
        y_pred = pipeline.predict(X_test)

        # Calculate evaluation metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        # Prepare metadata for the model and update the JSON file
        metadata = {
            "name": model_name,
            "type": model_type,
            "features": selected_features,
            "rmse": rmse,
            "r2": r2,
            "file": os.path.join('models', f"{model_name}.pkl")
        }
        self.update_metadata(metadata)

        return {"model": pipeline, "rmse": rmse, "r2": r2}

    def train_linear_regression(self, X_train, y_train):
        """Train a Linear Regression model."""
        # Initialize and fit the Linear Regression model
        model = LinearRegression()
        model.fit(X_train, y_train)
        return model

    def train_random_forest(self, X_train, y_train):
        """Train a Random Forest Regressor model."""
        # Initialize and fit the Random Forest Regressor model
        model = RandomForestRegressor(random_state=42)
        model.fit(X_train, y_train)
        return model

    def train_gradient_boosting(self, X_train, y_train):
        """Train a Gradient Boosting Regressor model."""
        # Initialize and fit the Gradient Boosting Regressor model
        model = GradientBoostingRegressor(random_state=42)
        model.fit(X_train, y_train)
        return model

    def predict(self, model_name, features):
        """Make a prediction using a trained model."""
        # Load the model
        model = self.load_model(model_name)

        # Features must be passed as a DataFrame with one row
        X = pd.DataFrame([features])

        # Take just the first element of the prediction, that is the expected prediction
        prediction = model.predict(X)[0]
        return prediction