// --- 1. CONFIGURATION: PRECISE VALUES CALCULATED FROM TRAINING DATA ---

// --- STANDARD SCALER CONSTANTS (2 features) ---
const RATING_MEAN = 3.6189;  // Calculated Mean
const RATING_STD = 0.8012;   // Calculated Standard Deviation
const AGE_MEAN = 46.5916;    // Calculated Mean
const AGE_STD = 53.7788;     // Calculated Standard Deviation

// --- ONE-HOT ENCODING MAPPING (38 features) ---
// Total Features = 2 (Scaled) + 5 (Binary) + 38 (OHE) = 45
function getOneHotState(jobState) {
    // Correct size for 38 OHE features.
    let stateArray = new Array(38).fill(0.0); 

    // OHE indices correspond to the alphabetically sorted unique job states.
    switch (jobState) {
        case 'AL': stateArray[0] = 1.0; break;
        case 'AZ': stateArray[1] = 1.0; break;
        case 'CA': stateArray[2] = 1.0; break; 
        case 'CO': stateArray[3] = 1.0; break; 
        case 'CT': stateArray[4] = 1.0; break;
        case 'DC': stateArray[5] = 1.0; break;
        case 'DE': stateArray[6] = 1.0; break;
        case 'FL': stateArray[7] = 1.0; break;
        case 'GA': stateArray[8] = 1.0; break;
        case 'IA': stateArray[9] = 1.0; break;
        case 'ID': stateArray[10] = 1.0; break;
        case 'IL': stateArray[11] = 1.0; break;
        case 'IN': stateArray[12] = 1.0; break;
        case 'KS': stateArray[13] = 1.0; break;
        case 'KY': stateArray[14] = 1.0; break;
        case 'LA': stateArray[15] = 1.0; break;
        case 'MA': stateArray[16] = 1.0; break;
        case 'MD': stateArray[17] = 1.0; break; 
        case 'MI': stateArray[18] = 1.0; break;
        case 'MN': stateArray[19] = 1.0; break;
        case 'MO': stateArray[20] = 1.0; break;
        case 'NC': stateArray[21] = 1.0; break;
        case 'NE': stateArray[22] = 1.0; break;
        case 'NJ': stateArray[23] = 1.0; break;
        case 'NM': stateArray[24] = 1.0; break;
        case 'NY': stateArray[25] = 1.0; break; 
        case 'OH': stateArray[26] = 1.0; break;
        case 'OR': stateArray[27] = 1.0; break;
        case 'PA': stateArray[28] = 1.0; break;
        case 'RI': stateArray[29] = 1.0; break;
        case 'SC': stateArray[30] = 1.0; break;
        case 'TN': stateArray[31] = 1.0; break;
        case 'TX': stateArray[32] = 1.0; break; 
        case 'UT': stateArray[33] = 1.0; break;
        case 'VA': stateArray[34] = 1.0; break; 
        case 'WA': stateArray[35] = 1.0; break;
        case 'WI': stateArray[36] = 1.0; break;
        
        // This case handles any unlisted state, mapping it to the last feature index (37)
        default: stateArray[37] = 1.0; break; 
    }
    return stateArray;
}


// --- 2. Model Loading ---
const modelPath = 'regression_model.onnx'; 
let session = null;
async function loadModel() {
    try {
        session = await ort.InferenceSession.create(modelPath);
        document.getElementById('output').innerText = "Model loaded. Ready for prediction.";
    } catch (e) {
        document.getElementById('output').innerText = `Error loading model: ${e.message}`;
        console.error("Failed to load ONNX model:", e);
    }
}
loadModel(); 

// --- 3. Preprocessing (Guarantees 45 features) ---
function preprocessInputs() {
    // 1. COLLECT INPUTS FROM HTML
    const rating = parseFloat(document.getElementById('rating').value);
    const age = parseFloat(document.getElementById('age').value);
    const jobState = document.getElementById('job_state').value;
    const python_yn = parseFloat(document.getElementById('python_yn').value);
    const R_yn = parseFloat(document.getElementById('R_yn').value);
    
    // FIX: Hardcode the 3 missing binary features to 0.0 for a 45-feature array
    const spark_yn = 0.0; 
    const aws_yn = 0.0;
    const excel_yn = 0.0;
    

    // 2. APPLY STANDARD SCALING
    const scaledRating = (rating - RATING_MEAN) / RATING_STD;
    const scaledAge = (age - AGE_MEAN) / AGE_STD;

    // 3. APPLY ONE-HOT ENCODING
    const oneHotStates = getOneHotState(jobState); 

    // 4. CONSTRUCT FINAL ARRAY (MUST BE 45 ELEMENTS IN PYTHON ORDER)
    const processedArray = [
        // 1-2 Scaled Numeric Features
        scaledRating, scaledAge, 
        
        // 3-40 OHE State Features
        ...oneHotStates, 
        
        // 41-45 Binary Features (in the exact order of your Python ColumnTransformer)
        python_yn, R_yn, spark_yn, aws_yn, excel_yn, 
    ]; 

    if (processedArray.length !== 45) {
         throw new Error(`Feature count mismatch: Expected 45, got ${processedArray.length}.`);
    }
    
    return new Float32Array(processedArray);
}

// --- 4. Run Inference ---
async function runInference() {
    if (!session) {
        document.getElementById('output').innerText = "Model not loaded yet. Please wait.";
        return;
    }
    
    try {
        const inputTensorData = preprocessInputs();
        const inputTensor = new ort.Tensor('float32', inputTensorData, [1, inputTensorData.length]);
        
        const feeds = { input: inputTensor };
        const results = await session.run(feeds);
        
        const predictedSalary = results.output.data[0]; 

        document.getElementById('output').innerText = 
            `Predicted Average Salary: $${predictedSalary.toFixed(2)}K`;
            
    } catch (e) {
        document.getElementById('output').innerText = `Prediction Error: ${e.message}`;
    }
}


