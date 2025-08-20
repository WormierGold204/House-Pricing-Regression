from flask import Flask, render_template, request, jsonify
from dataset_manager import DatasetManager

app = Flask(__name__)
dataset = DatasetManager()

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/train", methods=["GET", "POST"])
def train():
    if request.method == "POST":
        regression_type = request.form.get("model_type")
        model_name = request.form.get("model_name")
        selected_features = request.form.getlist("features")  # list of checked features

        # For now, just print them (later you’ll run training here)
        print("Training with:")
        print("Model Name:", model_name)
        print("Regression Type:", regression_type)
        print("Features:", selected_features)

    # Get all features from the dataset and pass them to the template
    features = dataset.get_features()
    return render_template("training.html", features=features)

@app.route('/prediction')
def prediction():
    return render_template('prediction.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)