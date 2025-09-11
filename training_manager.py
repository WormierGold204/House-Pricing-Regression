from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
import joblib
import json
import os

class TrainingManager:
    def __init__(self, dataset_manager):
        self.dataset_manager = dataset_manager

        # Ensure the models.json file exists, creating it if necessary
        if not os.path.exists("models"):
            os.makedirs("models")
        if not os.path.exists("models/models.json"):
            with open("models/models.json", "w") as f:
                json.dump([], f)


    def get_models(self):
        """Return the list of trained models.
        
        Returns:
        list: A list of dictionaries containing model metadata."""
        with open("models/models.json", "r") as f:
            return json.load(f)

    def save_model(self, model, model_name):
        """Save the trained model to disk.
        
        Parameters:
        model: The trained model to be saved.
        model_name (str): The name to save the model under.
        """

        # Save the model
        joblib.dump(model, f"models/{model_name}.pkl")

    def load_model(self, model_name):
        """Load a trained model from disk.
        
        Parameters:
        model_name (str): The name of the model to be loaded.
        
        Returns:
        The loaded model."""

        model_path = f"models/{model_name}.pkl"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model {model_name} not found")

        # Return the loaded model
        return joblib.load(model_path)

    def update_metadata(self, metadata, delete = False):
        """Update the models.json file with new model metadata.
        If delete is True, the dumped metadata does not contain the model to be deleted;
        otherwise, it contains the new model to be added.
        
        Parameters:
        metadata (dict): The metadata of the model to be added or the list of models to keep.
        delete (bool): Whether to delete a model (True) or add/update a model"""

        # Load existing metadata to avoid overwriting
        models = self.get_models()

        if delete:
            with open("models/models.json", "w") as f:
                json.dump(metadata, f, indent=4)
            return

        # Overwrite existing model metadata if the name matches
        models = [m for m in models if m["name"] != metadata["name"]]
        models.append(metadata)

        # Save updated metadata
        with open("models/models.json", "w") as f:
            json.dump(models, f, indent=4)

    def delete_model(self, model_name):
        """Delete a trained model and its metadata.
        
        Parameters:
        model_name (str): The name of the model to be deleted."""

        # Get all models from the json file
        models = self.get_models()

        # Find the model indicated by user in metadata
        model_to_delete = next((m for m in models if m["name"] == model_name), None)
        if not model_to_delete:
            raise ValueError("Model not found")

        # Delete .pkl file of the model if it exists
        if os.path.exists(model_to_delete["file"]):
            os.remove(model_to_delete["file"])

        # Remove information about the model from metadata list
        models = [m for m in models if m["name"] != model_name]

        # Save json file again
        self.update_metadata(models, True)

    def train_model(self, model_type, model_name, selected_features):
        """Train a model based on the selected type and features.
        
        Parameters:
        model_type (str): The type of model to train (e.g., 'linear_regression', 'random_forest', etc.).
        model_name (str): The name to save the trained model under.
        selected_features (list): The list of features to use for training.
        
        Returns:
        dict: A dictionary containing the trained model and its evaluation metrics (RMSE, R^2)."""
        
        # Obtain preprocessed data and keep only selected features
        if not selected_features:
            raise ValueError("No features selected for training")

        X = self.dataset_manager.df[selected_features]
        y = self.dataset_manager.df[self.dataset_manager.get_target()]

        # Split Categorical and Numeric Features
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X.select_dtypes(include=['object']).columns

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
        elif model_type in ['gradient_boosting', 'gradient_boosting_tuned']:
            model = GradientBoostingRegressor(random_state=42)
        elif model_type in ['random_forest', 'random_forest_tuned']:
            model = RandomForestRegressor(random_state=42)
        else:
            raise ValueError("Unsupported model type")

        # If the user selected a tuned model, set up GridSearchCV with hyperparameters for tuning
        if model_type == 'gradient_boosting_tuned':
            pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                ('model', model)])

            # Define the hyperparameter grid for tuning
            param_grid = {
                'model__n_estimators': [100, 300],
                'model__learning_rate': [0.01, 0.1, 0.2],
                'model__max_depth': [2, 5]
            }

            # Optimize the pipeline, searching for the best hyperparameters
            processed_model = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, scoring='neg_mean_squared_error')
        elif model_type == 'random_forest_tuned':
            pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                ('model', model)])

            param_grid = {
                'model__n_estimators': [100, 200],
                'model__max_depth': [None, 10, 20],
                'model__min_samples_split': [2, 5]
            }

            processed_model = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, scoring='neg_mean_squared_error')
        else:
            # Create a pipeline that first preprocesses the data and then applies the model (for tuned model this is done before the the tuning)
            # This allows not to worry about the order of operations
            # and ensures that the same preprocessing is applied during training and prediction
            processed_model = Pipeline(steps=[('preprocessor', preprocessor),
                                    ('model', model)])

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train the model based on the pipeline
        processed_model.fit(X_train, y_train)

        # Save the model
        self.save_model(processed_model, model_name)

        # Evaluate the model
        y_pred = processed_model.predict(X_test)

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
            "file": f"models/{model_name}.pkl"
        }
        self.update_metadata(metadata)

        return {"model": processed_model, "rmse": rmse, "r2": r2}

    def predict(self, model_name, features):
        """Make a prediction using a trained model.
        
        Parameters:
        model_name (str): The name of the model to use for prediction.
        features (dict): A dictionary of feature values for prediction.
        
        Returns:
        The prediction result.
        """
        # Load the model
        model = self.load_model(model_name)

        # Features must be passed as a DataFrame with one row
        X = pd.DataFrame([features])

        # Load metadata of current model
        model_meta = next((m for m in self.get_models() if m["name"] == model_name), None)

        # Fill non compiled features' boxes with NaN value
        expected_features = model_meta["features"]
        for f in expected_features:
            if f not in X.columns:
                X[f] = pd.NA

        # Take just the first element of the prediction, that is the expected prediction
        prediction = model.predict(X)[0]
        return prediction
