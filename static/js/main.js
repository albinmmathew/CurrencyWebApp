
const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('fileInput');
const preview = document.getElementById('preview');
const predictBtn = document.getElementById('predictBtn');
const resultCard = document.getElementById('resultCard');
const predictionText = document.getElementById('predictionText');
const confidenceText = document.getElementById('confidenceText');
const confidenceBar = document.getElementById('confidenceBar');
const modelSelect = document.getElementById('modelSelect');
const loadingOverlay = document.getElementById('loadingOverlay');

let selectedFile = null;

// Drag and Drop
dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.classList.add('active');
});

dropzone.addEventListener('dragleave', () => {
    dropzone.classList.remove('active');
});

dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.classList.remove('active');
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
});

dropzone.addEventListener('click', () => {
    fileInput.click();
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFile(e.target.files[0]);
    }
});

function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        alert('Please upload an image file');
        return;
    }
    selectedFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
        preview.src = e.target.result;
        preview.style.display = 'block';
        dropzone.querySelector('i').style.display = 'none';
        dropzone.querySelector('p').style.display = 'none';
        predictBtn.disabled = false;
    };
    reader.readAsDataURL(file);
}

// Prediction
predictBtn.addEventListener('click', async () => {
    if (!selectedFile) return;

    const formData = new FormData();
    formData.append('image', selectedFile);
    formData.append('model', modelSelect.value);

    // UI State
    predictBtn.disabled = true;
    predictBtn.innerText = 'Recognizing...';
    resultCard.style.display = 'none';

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.error) {
            alert('Error: ' + data.error);
        } else {
            // Success
            predictionText.innerText = data.prediction;
            confidenceText.innerText = `Confidence: ${data.confidence}`;

            resultCard.style.display = 'block';
            setTimeout(() => {
                confidenceBar.style.width = data.confidence;
            }, 100);
        }
    } catch (error) {
        console.error('Error:', error);
        alert('An error occurred during prediction.');
    } finally {
        predictBtn.disabled = false;
        predictBtn.innerText = 'Identify Currency';
    }
});
