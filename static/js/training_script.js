// Script for training page
document.addEventListener("DOMContentLoaded", function() {
    // AJAX request to train the model with given parameters
    document.querySelector("#training-button").addEventListener("click", async function(event) {
        const modelType = document.querySelector("#model-type").value;
        const modelName = document.getElementById("model-name").value;
        const features = Array.from(
            document.querySelectorAll("input[name=features]:checked")
        ).map(cb => cb.value);

        document.querySelector("#status").innerText = "Training in progress...";

        results = await fetch("/train", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                model_type: modelType,
                model_name: modelName,
                features: features
            })
        });
        if (results.ok) {
            const data = await results.json();
            document.querySelector("#status").innerText = data.message;
        } else {
            const errorData = await results.json();
            document.querySelector("#status").innerText = `Error: ${errorData.error}`;
        }
    });

    // Select all features in the checkbox list
    document.querySelector("#select-all").addEventListener("click", function() {
        document.querySelectorAll("input[name=features]").forEach(cb => {
            cb.checked = true;
        });
    });

    // Deselect all features in the checkbox list
    document.querySelector("#deselect-all").addEventListener("click", function() {
        document.querySelectorAll("input[name=features]").forEach(cb => {
            cb.checked = false;
        });
    });

    // Select high correlated features
    document.querySelector("#select-hc").addEventListener("click", function() {
        const highCorrelatedFeatures = [
            "MSSubClass", "MSZoning", "LotFrontage", "LotArea", "LotConfig", "Neighborhood", "OverallQual",
            "OverallCond", "YearBuilt", "RoofStyle", "Exterior1st", "MasVnrType", "MasVnrArea", "ExterQual",
            "ExterCond", "BsmtCond", "BsmtExposure", "BsmtFinSF1", "BsmtFinSF2", "TotalBsmtSF", "HeatingQC",
            "1stFlrSF", "2ndFlrSF", "GrLivArea", "FullBath", "GarageCars", "GarageArea", "WoodDeckSF",
            "OpenPorchSF", "SaleCondition"
        ];
        document.querySelectorAll("input[name=features]").forEach(cb => {
            cb.checked = highCorrelatedFeatures.includes(cb.value);
        });
    });
});