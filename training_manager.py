from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, SGDRegressor
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

        # Check if the models.json file exists; if not, create it
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

        # Take just user selected features
        X = self.dataset_manager.df[selected_features]
        y = self.dataset_manager.df[self.dataset_manager.get_target()]

        # Split the data into training and testing sets (80%-20%)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size = 0.2, random_state = 42
        )

        # Split categorical and numeric features so to operate different actions on each category
        numeric_features = X_train.select_dtypes(include = ['int64', 'float64']).columns.tolist()
        categorical_features = X_train.select_dtypes(include = ['object']).columns.tolist()

        # Split LotFrontage numeric column (if user selected it) in order to apply specific NA filling
        lot_frontage_feature = ['LotFrontage'] if 'LotFrontage' in numeric_features else []

        # Remove LotFrontage (if user selected it) from other features list
        numeric_features = [f for f in numeric_features if f not in lot_frontage_feature]

        # Preprocessing for numeric features (no LotFrontage)
        numeric_pipeline_nlf = Pipeline([
            ('imputer', SimpleImputer(strategy = 'constant', fill_value = 0))  # needed for MasVnrArea and GarageYrBlt feature
        ])

        # Preprocessing for LotFrontage numeric feature
        numeric_pipeline_lf = Pipeline([
            ('imputer', SimpleImputer(strategy = 'median')) # needed for LotFrontage feature
        ])

        # In linear and gradient descent models apply scaling too
        if model_type in ['linear_regression', 'gradient_descent']:
            numeric_pipeline_nlf.steps.append(('scaler', StandardScaler()))
            numeric_pipeline_lf.steps.append(('scaler', StandardScaler()))

        # Preprocessing for categorical features
        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy = 'constant', fill_value = 'None')), # categorical features NAs mean lack of feature
            ('onehot', OneHotEncoder(handle_unknown = 'ignore'))    # separate categorical features in more columns
        ])

        # Add preprocessings to pipeline (if user seleceted related features)
        transformers = []
        if numeric_features:
            transformers.append(('num_zero', numeric_pipeline_nlf, numeric_features))
        if lot_frontage_feature:
            transformers.append(('num_median', numeric_pipeline_lf, lot_frontage_feature))
        if categorical_features:
            transformers.append(('cat', categorical_pipeline, categorical_features))

        preprocessor = ColumnTransformer(transformers)

        # Choose the model based on the user's choice
        if model_type in ['linear_regression']:
            model = LinearRegression()
        elif model_type in ['gradient_boosting']:
            model = GradientBoostingRegressor(random_state=42)
        elif model_type in ['random_forest']:
            model = RandomForestRegressor(random_state=42)
        elif model_type in ['gradient_descent']:
            model = SGDRegressor(max_iter=1000, tol=1e-3)
        else:
            raise ValueError("Unsupported model type")

        # Create a pipeline that first preprocesses the data and then applies the model (for tuned model this is done before the the tuning)
        # This allows not to worry about the order of operations
        # and ensures that the same preprocessing is applied during both training and prediction
        processed_model = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

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

        # Take the first element of the prediction, that is the expected prediction
        prediction = model.predict(X)[0]
        return prediction
