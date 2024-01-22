document.addEventListener("DOMContentLoaded", function () {
    const form = document.getElementById("uploadForm");
    const fileInput = document.getElementById("fileInput");
    const predictButton = document.getElementById("predictButton");
    const loadingIcon = document.getElementById("loadingIcon");

    form.addEventListener("submit", function (event) {
        event.preventDefault();

        // Show loading icon and change button text
        loadingIcon.style.display = "inline-block";
        predictButton.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Predicting...';

        const formData = new FormData(form);
        fetch("/predict_flower", {
            method: "POST",
            body: formData,
        })
        .then(response => response.json())
        .then(data => {
            // Hide loading icon and reset button text
            loadingIcon.style.display = "none";
            predictButton.innerHTML = 'Predict';

            displayPredictionResult(data);
        })
        .catch(error => {
            // Hide loading icon in case of error and reset button text
            loadingIcon.style.display = "none";
            predictButton.innerHTML = 'Predict';

            console.error("Error:", error);
        });
    });

    // Image preview on file input change
    fileInput.addEventListener("change", function () {
        const file = fileInput.files[0];
        if (file) {
            displayImagePreview(file);
        }
    });
});

function displayPredictionResult(data) {
    const resultDiv = document.getElementById("predictionResult");
    resultDiv.innerHTML = `
        <p>Predicted: ${data.class}</p>
        <p>Confidence: ${data.confidence}%</p>
    `;
}

function displayImagePreview(file) {
    const body = document.body;

    const reader = new FileReader();
    reader.onload = function (e) {
        // Set the background image of the body to the dropped image
        body.style.backgroundImage = `url(${e.target.result})`;
        body.style.backgroundSize = "cover";
        body.style.backgroundPosition = "center";
        body.style.backgroundRepeat = "no-repeat";
    };

    reader.readAsDataURL(file);
}
