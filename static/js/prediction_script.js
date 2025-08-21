document.addEventListener('DOMContentLoaded', async function() {
    const predictButton = document.querySelector('#predict-button');

    // Disable the predict button until all information needed for a prediction is provided
    predictButton.disabled = true
    
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
            radioSelected = true;
        });
    });

    predictButton.addEventListener('click', async () => {
        // Get the selected model
        const selectedModelRadio = document.querySelector('input[name="model"]:checked');
        
    });

    // Clear the input box and info section on page load
    let inputDiv = document.querySelector('#prediction-form');
    inputDiv.innerHTML = '';
    let infoDiv = document.querySelector('#model-info');
    infoDiv.innerHTML = '';
});

// Variable to know if a radio button has been selected
let radioSelected = false;

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

        // Add event listener to check if the button can be enabled
        textInput.addEventListener('input', checkButton);

        featureDiv.appendChild(textInput);

        inputDiv.appendChild(featureDiv);
    });
}

// Function to check if the predict button can be enabled
function checkButton() {
    const predictButton = document.querySelector('#predict-button');
    const allInputs = document.querySelectorAll('#prediction-form input[type="text"]');
    
    // Check if all text input boxes are filled
    const allFilled = Array.from(allInputs).every(input => input.value.trim() !== "");
    
    // Enable the predict button if a radio button is selected and all input boxes are filled
    if (radioSelected && allFilled) {
        predictButton.disabled = false;
    } else {
        predictButton.disabled = true;
    }
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
    typePara.textContent = `Model Type: ${model.type}`;
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