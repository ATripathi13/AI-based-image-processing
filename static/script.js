const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d', { willReadFrequently: true });
const video = document.getElementById('video');
const fileInput = document.getElementById('file-input');
const dropZone = document.getElementById('drop-zone');
const uploadBtn = document.getElementById('upload-btn');
const cameraBtn = document.getElementById('camera-btn');
const snapBtn = document.getElementById('snap-btn');
const clearBtn = document.getElementById('clear-btn');
const predictionValue = document.getElementById('prediction');
const confidenceValue = document.getElementById('confidence');
const confidenceBar = document.getElementById('confidence-bar');
const resultArea = document.getElementById('result-area');

let isCameraActive = false;

// Initialize canvas with black background
function initCanvas() {
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
}
initCanvas();

// Handle File Upload
uploadBtn.addEventListener('click', () => fileInput.click());

fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) handleImage(file);
});

// Drag and Drop
dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('active');
});

dropZone.addEventListener('dragleave', () => dropZone.classList.remove('active'));

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('active');
    const file = e.dataTransfer.files[0];
    if (file) handleImage(file);
});

function handleImage(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
        const img = new Image();
        img.onload = () => {
            stopCamera();
            initCanvas();
            // Draw image to fill square canvas
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
            predict();
        };
        img.src = e.target.result;
        dropZone.style.display = 'none';
    };
    reader.readAsDataURL(file);
}

// Camera Functions
cameraBtn.addEventListener('click', async () => {
    if (isCameraActive) {
        stopCamera();
    } else {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' } });
            video.srcObject = stream;
            video.style.display = 'block';
            canvas.style.display = 'none';
            dropZone.style.display = 'none';
            snapBtn.style.display = 'inline-block';
            cameraBtn.textContent = 'Stop Camera';
            isCameraActive = true;
        } catch (err) {
            alert('Error accessing camera: ' + err.message);
        }
    }
});

function stopCamera() {
    if (video.srcObject) {
        video.srcObject.getTracks().forEach(track => track.stop());
    }
    video.style.display = 'none';
    canvas.style.display = 'block';
    snapBtn.style.display = 'none';
    cameraBtn.textContent = 'Use Camera';
    isCameraActive = false;
}

snapBtn.addEventListener('click', () => {
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    stopCamera();
    predict();
});

// Clear Canvas
clearBtn.addEventListener('click', () => {
    stopCamera();
    initCanvas();
    dropZone.style.display = 'flex';
    resultArea.style.display = 'none';
    fileInput.value = '';
});

// Prediction API Call
async function predict() {
    const imageData = canvas.toDataURL('image/png');
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: imageData })
        });
        
        const result = await response.json();
        
        if (result.error) {
            alert(result.error);
            return;
        }

        displayResult(result.prediction, result.confidence);
    } catch (err) {
        console.error('Prediction failed:', err);
    }
}

function displayResult(prediction, confidence) {
    resultArea.style.display = 'flex';
    predictionValue.textContent = prediction;
    const confPercent = (confidence * 100).toFixed(1) + '%';
    confidenceValue.textContent = confPercent;
    confidenceBar.style.width = confPercent;
}
