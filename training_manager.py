from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import numpy as np

class TrainingManager:
    def __init__(self, dataset_manager):
        self.dataset_manager = dataset_manager

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