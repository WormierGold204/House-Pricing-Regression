document.addEventListener('DOMContentLoaded', async function() {
    const predictButton = document.querySelector('#predict-button');

    // Disable the predict button until all information needed for a prediction is provided
    predictButton.disabled = true;
    
    // Fetch list of models from the server
    models_list = await fetch('/models');
    models = await models_list.json();

    // Section to diplay models
    let modelsDiv = document.querySelector('#models-list');

    // Create radio buttons for each model
    models.forEach((model, index) => {
        const modelDiv = document.createElement('div');

        // Radio button
        const radioInput = document.createElement('input');
        radioInput.type = 'radio';
        radioInput.name = 'model';
        radioInput.value = model.name;
        radioInput.id = `model-${index}`;
        modelDiv.appendChild(radioInput);

        // Label for the radio button
        const label = document.createElement('label');
        label.textContent = `${model.name} (RMSE: ${model.rmse.toFixed(2)}, Number of Features: ${model.features.length})`;
        label.htmlFor = `model-${index}`;
        modelDiv.appendChild(label);

        modelsDiv.appendChild(modelDiv);

        // Listener to show information and display boxes for input features when a model is selected
        radioInput.addEventListener('change', () => {
            showInfo(model);
            showBox(model);

            // Note that a radio button has been selected
            predictButton.disabled = false;
        });
    });

    predictButton.addEventListener('click', async () => {
        // Get the selected model
        const selectedModelRadio = document.querySelector('input[name="model"]:checked').value;

        // Get the values from the input boxes
        const inputBoxes = document.querySelectorAll('#prediction-form input[type="text"]');
        let inputData = {};
        inputBoxes.forEach((box) => {
            let value = box.value.trim();
            inputData[box.name] = value;
        });
        
        // Send the data to the server for prediction
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ model_name: selectedModelRadio, features: inputData })
        });

        // Display the prediction result, clearing previous results
        const result = await response.json();
        let predictionText = document.querySelector('#prediction-result');
        if(response.ok) {
            predictionText.textContent = `Predicted Price: $${result.prediction.toFixed(2)}`;
        } else {
            predictionText.textContent = `Error: ${result.error}`;
        }
    });

    // Clear the input box and info section on page load
    let inputDiv = document.querySelector('#prediction-form');
    inputDiv.innerHTML = '';
    let infoDiv = document.querySelector('#model-info');
    infoDiv.innerHTML = '';
});

// Function to show input box for the selected model
function showBox(model) {
    let inputDiv = document.querySelector('#prediction-form');

    // Clear previous input boxes list
    inputDiv.innerHTML = '';

    // Create input boxes for each feature
    model.features.forEach((feature) => {
        const featureDiv = document.createElement('div');

        // Label for the feature
        const label = document.createElement('label');
        label.textContent = feature;
        label.htmlFor = `feature-${feature}`;
        featureDiv.appendChild(label);

        // Textbox for the feature
        const textInput = document.createElement('input');
        textInput.type = 'text';
        textInput.name = feature;
        textInput.id = `feature-${feature}`;

        featureDiv.appendChild(textInput);

        inputDiv.appendChild(featureDiv);
    });
}

// Function to show information about the selected model
function showInfo(model) {
    let infoDiv = document.querySelector('#model-info');

    // Clear previous info
    infoDiv.innerHTML = '';

    // Display model information
    // Model's name
    const namePara = document.createElement('p');
    namePara.textContent = `Model Name: ${model.name}`;
    infoDiv.appendChild(namePara);

    // Model's type
    const typePara = document.createElement('p');
    if (model.type === "linear_regression") {
        typePara.textContent = `Model Type: Linear Regressor`;
    } else if (model.type === "gradient_boosting") {
        typePara.textContent = `Model Type: Gradient Boosting Regressor`;
    } else if (model.type === "random_forest") {
        typePara.textContent = `Model Type: Random Forest Regressor`;
    } else if (model.type === "gradient_descent") {
        typePara.textContent = `Model Type: Gradient Descent Regressor`
    } else {
        typePara.textContent = `Model Type: Unknown`;
    }
    infoDiv.appendChild(typePara);

    // Model's RMSE
    const rmsePara = document.createElement('p');
    rmsePara.textContent = `Model RMSE: ${model.rmse.toFixed(2)}`;
    infoDiv.appendChild(rmsePara);

    // Model's R2
    const r2Para = document.createElement('p');
    r2Para.textContent = `Model R2: ${model.r2.toFixed(2)}`;
    infoDiv.appendChild(r2Para);

    // Model's number of features
    const featuresPara = document.createElement('p');
    featuresPara.textContent = `Number of Features: ${model.features.length}`;
    infoDiv.appendChild(featuresPara);
}