// ===============================
//  Regression Inference Script
//  Fixed Version for GitHub Pages
// ===============================

// ---- 1. SCALER CONSTANTS (Replace with real values from Python StandardScaler)
const RATING_MEAN = 3.9;
const RATING_STD = 0.5;
const AGE_MEAN = 15.0;
const AGE_STD = 10.0;

// ---- 2. ONE-HOT ENCODING (Adjust as per your Python preprocessing)
function getOneHotState(jobState) {
    const states = ['CA', 'NY', 'VA', 'TX', 'MD', 'Others'];
    return states.map(s => (s === jobState ? 1.0 : 0.0));
}

// ---- 3. LOAD MODEL ----
let session = null;
async function loadModel() {
    const modelPath = 'regression_model.onnx';
    try {
        session = await ort.InferenceSession.create(modelPath);
        document.getElementById('output').innerText = "✅ Model loaded successfully. Ready for prediction.";
        console.log("Model loaded:", modelPath);
    } catch (e) {
        console.error("❌ Failed to load ONNX model:", e);
        document.getElementById('output').innerText = "❌ Error loading model. See console for details.";
    }
}
loadModel();

// ---- 4. PREPROCESS INPUTS ----
function preprocessInputs() {
    const rating = parseFloat(document.getElementById('rating').value);
    const age = parseFloat(document.getElementById('age').value);
    const jobState = document.getElementById('job_state').value;
    const python_yn = parseFloat(document.getElementById('python_yn').value);
    const R_yn = parseFloat(document.getElementById('R_yn').value);

    // Scale numeric features
    const scaledRating = (rating - RATING_MEAN) / RATING_STD;
    const scaledAge = (age - AGE_MEAN) / AGE_STD;

    // One-hot encode state
    const oneHot = getOneHotState(jobState);

    // Combine all features into a single flat array
    const inputArray = [
        scaledRating, scaledAge,
        ...oneHot,
        python_yn, R_yn
    ];

    console.log("🧮 Input vector length:", inputArray.length);
    return new Float32Array(inputArray);
}

// ---- 5. RUN INFERENCE ----
async function runInference() {
    if (!session) {
        document.getElementById('output').innerText = "⏳ Model not loaded yet. Please wait...";
        return;
    }

    try {
        const inputData = preprocessInputs();
        const inputTensor = new ort.Tensor('float32', inputData, [1, inputData.length]);

        // NOTE: Make sure your ONNX export used these names
        const feeds = { input: inputTensor };
        const results = await session.run(feeds);

        // If unsure of output name, log all keys
        console.log("Model outputs:", Object.keys(results));

        // Try reading first output dynamically
        const outputKey = Object.keys(results)[0];
        const predictedSalary = results[outputKey].data[0];

        document.getElementById('output').innerText =
            `💰 Predicted Average Salary: $${predictedSalary.toFixed(2)}K`;
    } catch (e) {
        console.error("Prediction error:", e);
        document.getElementById('output').innerText = `❌ Prediction Error: ${e.message}`;
    }
}




