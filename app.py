from flask import Flask, render_template, request, jsonify
from dataset_manager import DatasetManager
from training_manager import TrainingManager
import json, os

app = Flask(__name__)
dataset = DatasetManager()
training_manager = TrainingManager(dataset)

# Route to display the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route to display the EDA page
@app.route('/eda', methods=["GET"])
def eda():
    return render_template('eda.html', statistics=dataset.statistics)

# Route to display the training page
@app.route("/train", methods=["GET", "POST"])
def train():
    # Handle a POST request to train a model
    if request.method == "POST":
        # Get the regression type, model name, and selected features from the AJAX request
        data = request.get_json()
        regression_type = data.get("model_type")
        model_name = data.get("model_name")
        selected_features = data.get("features", [])

        # Validate the input
        if not regression_type or not model_name or not selected_features:
            return jsonify({"error": "All fields are required"}), 400

        # Train the model using the selected features
        try:
            model = training_manager.train_model(regression_type, model_name, selected_features)
            # Check if the model was trained successfully and return the evaluation metrics
            if model['model']:
                return jsonify({"message": f"Model {model_name} trained successfully!\nEvaluation metrics:\nRMSE = {model['rmse']}\nR^2 = {model['r2']}"}), 200
            else:
                return jsonify({"error": "Model training failed"}), 500
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # Handle a GET request to display the training page
    # Get all features from the dataset and pass them to the template
    features = dataset.get_features()
    return render_template("training.html", features=features)

# Route to display the prediction page
@app.route('/prediction')
def prediction():
    return render_template('prediction.html')

# Route to get the list of trained models
@app.route('/models', methods=["GET"])
def models():
    return jsonify(training_manager.get_models()), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

# Route to perform predictions
@app.route('/predict', methods=["POST"])
def predict():
    # Get the model name and features from the request
    data = request.get_json()
    model_name = data.get("model_name")
    features = data.get("features", {})

    # Validate the input
    if not model_name or not features:
        return jsonify({"error": "Model name and features are required"}), 400

    # Perform prediction using the selected model
    try:
        prediction = training_manager.predict(model_name, features)
        return jsonify({"prediction": prediction}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500