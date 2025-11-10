// =========================================
// Data Scientist Salary Predictor - FINAL
// Matches Python ONNX Export Exactly
// =========================================

// Replace with the true scaler means/stds from Python
const RATING_MEAN = 3.9; // example
const RATING_STD = 0.5;  // example
const AGE_MEAN = 15.0;   // example
const AGE_STD = 10.0;    // example

// One-Hot Encode Job States (must match your dataset order!)
function getOneHotState(jobState) {
    const states = [
        'CA','NY','TX','VA','MD','FL','NJ','MA','PA','IL','GA','NC','WA',
        'CO','OH','MI','MN','AZ','TN','IN','MO','WI','CT','OR','WA','KY',
        'OK','SC','LA','NV','UT','AL','AR','IA','KS','ID','MS','NE','NM','SD','Others'
    ];
    return states.map(s => (s === jobState ? 1.0 : 0.0));
}

// ------------------------------
// Model Loading
// ------------------------------
let session = null;
async function loadModel() {
    try {
        session = await ort.InferenceSession.create('regression_model.onnx');
        document.getElementById('output').innerText = '✅ Model loaded successfully. Ready for prediction.';
    } catch (e) {
        console.error('❌ Model load error:', e);
        document.getElementById('output').innerText = '❌ Model failed to load.';
    }
}
loadModel();

// ------------------------------
// Preprocess Inputs
// ------------------------------
function preprocessInputs() {
    const rating = parseFloat(document.getElementById('rating').value);
    const age = parseFloat(document.getElementById('age').value);
    const jobState = document.getElementById('job_state').value;
    const python_yn = parseFloat(document.getElementById('python_yn').value);
    const R_yn = parseFloat(document.getElementById('R_yn').value);
    const spark = 0; // not in HTML form yet
    const aws = 0;
    const excel = 0;

    const scaledRating = (rating - RATING_MEAN) / RATING_STD;
    const scaledAge = (age - AGE_MEAN) / AGE_STD;

    const stateOHE = getOneHotState(jobState);

    // Combine features in same order as training pipeline
    const inputArray = [
        scaledRating, scaledAge,
        ...stateOHE,
        python_yn, R_yn, spark, aws, excel
    ];

    console.log('🧮 Feature vector length:', inputArray.length);
    return new Float32Array(inputArray);
}

// ------------------------------
// Run Inference
// ------------------------------
async function runInference() {
    if (!session) {
        document.getElementById('output').innerText = '⏳ Model not loaded yet. Please wait...';
        return;
    }

    try {
        const inputData = preprocessInputs();
        const inputTensor = new ort.Tensor('float32', inputData, [1, inputData.length]);

        const results = await session.run({ input: inputTensor });
        const predicted = results.output.data[0];

        document.getElementById('output').innerText = `💰 Predicted Average Salary: $${predicted.toFixed(2)}K`;
    } catch (e) {
        console.error('Prediction error:', e);
        document.getElementById('output').innerText = `❌ Prediction error: ${e.message}`;
    }
}
