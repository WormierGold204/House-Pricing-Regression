from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import numpy as np
import joblib
import json
import os

class TrainingManager:
    def __init__(self, dataset_manager):
        self.dataset_manager = dataset_manager

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
        X, y = self.dataset_manager.training_preprocessing()

        # Expand selections
        # This is needed because OneHotEncoder creates multiple columns for categorical features and here we have only the raw feature names
        expanded_features = []
        for feat in selected_features:
            expanded_features.extend(self.dataset_manager.feature_map.get(feat, []))
        X = X[expanded_features]

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        if model_type == 'linear_regression':
            model = self.train_linear_regression(X_train, y_train)
        elif model_type == 'gradient_boosting':
            model = self.train_gradient_boosting(X_train, y_train)
        elif model_type == 'random_forest':
            model = self.train_random_forest(X_train, y_train)
        else:
            raise ValueError("Unsupported model type")

        # Evaluate the model on the test set
        y_pred = model.predict(X_test)

        # Calculate evaluation metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        # Save the trained model
        self.save_model(model, model_name)

        # Update model metadata
        metadata = {
            "name": model_name,
            "type": model_type,
            "features": selected_features,
            "rmse": rmse,
            "r2": r2,
            "file": os.path.join('models', f"{model_name}.pkl")
        }
        self.update_metadata(metadata)

        # Return the trained model and evaluation metrics
        return {
            'model': model,
            'rmse': rmse,
            'r2': r2,
        }

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