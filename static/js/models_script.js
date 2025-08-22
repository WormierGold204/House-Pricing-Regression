document.addEventListener("DOMContentLoaded", async () => {
    const container = document.querySelector("#models-container")

    // Function to load models and dispaly them
    loadModels(container)
});

async function loadModels(container) {
    // Get the list of all models
    const res = await fetch('/models')
    const models = await res.json()

    // Clear all previous information
    container.innerHTML=''

    models.forEach(model => {
        let div = document.createElement("div")

        // Label to display information about the model
        let model_label = document.createElement("label")
        model_label.innerText = `Name: ${model.name}\n Type: ${model.type}; RMSE: ${model.rmse}, R^2: ${model.r2}\n Number of Features: ${model.features.length}`

        // Button to delete the model
        let button = document.createElement("button")
        button.innerText = "Delete"

        // Listener to permanently delete the model on button click
        button.addEventListener("click", async () => {
            if(confirm(`Delete model ${model.name}?`)) {
                    await fetch(`/view_models/${model.name}`, { method: "DELETE" });

                    // Reaload page after deleting a model
                    loadModels(container);
            }
        });

        div.appendChild(model_label)
        div.appendChild(button)

        container.appendChild(div)
    });
}