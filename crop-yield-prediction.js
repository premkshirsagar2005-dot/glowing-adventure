/**
 * Crop Yield Prediction Model
 * Uses Multiple Linear Regression algorithm
 */

// Crop coefficients (base yield in kg/hectare for each crop)
const CROP_COEFFICIENTS = {
    wheat: { baseYield: 3500, tempOptimal: 20, rainOptimal: 800 },
    rice: { baseYield: 4500, tempOptimal: 25, rainOptimal: 1500 },
    corn: { baseYield: 9000, tempOptimal: 25, rainOptimal: 800 },
    cotton: { baseYield: 1800, tempOptimal: 25, rainOptimal: 600 },
    soybean: { baseYield: 2800, tempOptimal: 24, rainOptimal: 700 },
    potato: { baseYield: 25000, tempOptimal: 18, rainOptimal: 600 }
};

// Soil type multipliers
const SOIL_MULTIPLIERS = {
    clay: 1.0,
    sandy: 0.75,
    loamy: 1.15,
    silty: 1.1,
    peaty: 1.05
};

// Pre-trained model coefficients (learned from historical data)
// These are simplified coefficients for demonstration
const MODEL_COEFFICIENTS = {
    intercept: 1000,
    rainfall: 0.5,
    temperature: 20,
    humidity: 10,
    fertilizer: 2.5,
    pesticides: 5,
    ph: 100,
    area: 0
};

class CropYieldPredictor {
    constructor() {
        this.trained = false;
        this.trainingData = this.generateTrainingData();
        this.train();
    }

    /**
     * Generate synthetic training data based on agricultural research
     */
    generateTrainingData() {
        const crops = ['wheat', 'rice', 'corn', 'cotton', 'soybean', 'potato'];
        const soils = ['clay', 'sandy', 'loamy', 'silty', 'peaty'];
        const data = [];

        // Generate 500 sample data points
        for (let i = 0; i < 500; i++) {
            const crop = crops[Math.floor(Math.random() * crops.length)];
            const soil = soils[Math.floor(Math.random() * soils.length)];
            
            const rainfall = Math.random() * 3000 + 200;
            const temperature = Math.random() * 35 + 5;
            const humidity = Math.random() * 80 + 20;
            const fertilizer = Math.random() * 300 + 20;
            const pesticides = Math.random() * 20 + 0.5;
            const ph = Math.random() * 5 + 4;
            const area = Math.random() * 50 + 1;

            const yield_ = this.calculateYield(
                crop, soil, rainfall, temperature, humidity,
                fertilizer, pesticides, ph, area
            );

            data.push({
                crop, soil, rainfall, temperature, humidity,
                fertilizer, pesticides, ph, area, yield_
            });
        }

        return data;
    }

    /**
     * Calculate yield based on all factors
     */
    calculateYield(crop, soil, rainfall, temperature, humidity, fertilizer, pesticides, ph, area) {
        const cropData = CROP_COEFFICIENTS[crop];
        const soilMultiplier = SOIL_MULTIPLIERS[soil];

        // Base yield for the crop
        let yield_ = cropData.baseYield;

        // Temperature factor (optimal temperature gives best yield)
        const tempDiff = Math.abs(temperature - cropData.tempOptimal);
        const tempFactor = Math.max(0, 1 - (tempDiff / 30));
        yield_ *= (0.7 + 0.3 * tempFactor);

        // Rainfall factor (optimal rainfall)
        const rainDiff = Math.abs(rainfall - cropData.rainOptimal);
        const rainFactor = Math.max(0, 1 - (rainDiff / 2000));
        yield_ *= (0.6 + 0.4 * rainFactor);

        // Humidity factor (moderate humidity is best)
        const humidityFactor = Math.max(0, 1 - Math.abs(humidity - 60) / 80);
        yield_ *= (0.8 + 0.2 * humidityFactor);

        // Fertilizer factor (diminishing returns)
        const fertilizerFactor = Math.min(1.3, 1 + Math.log10(fertilizer + 1) * 0.15);
        yield_ *= fertilizerFactor;

        // Pesticides factor (protects yield)
        const pesticideFactor = Math.min(1.2, 1 + pesticides * 0.005);
        yield_ *= pesticideFactor;

        // pH factor (optimal around 6-7)
        const phFactor = Math.max(0, 1 - Math.abs(ph - 6.5) / 5);
        yield_ *= (0.7 + 0.3 * phFactor);

        // Apply soil multiplier
        yield_ *= soilMultiplier;

        // Add some randomness (noise)
        yield_ *= (0.95 + Math.random() * 0.1);

        return Math.round(yield_);
    }

    /**
     * Train the linear regression model using gradient descent
     */
    train() {
        const features = ['rainfall', 'temperature', 'humidity', 'fertilizer', 'pesticides', 'ph'];
        const learningRate = 0.0001;
        const iterations = 10000;

        // Initialize coefficients
        let coefficients = {
            intercept: 0,
            rainfall: 0,
            temperature: 0,
            humidity: 0,
            fertilizer: 0,
            pesticides: 0,
            ph: 0
        };

        const n = this.trainingData.length;

        for (let iter = 0; iter < iterations; iter++) {
            let gradients = {
                intercept: 0,
                rainfall: 0,
                temperature: 0,
                humidity: 0,
                fertilizer: 0,
                pesticides: 0,
                ph: 0
            };

            // Calculate gradients
            for (const data of this.trainingData) {
                const prediction = this.predictLinear(
                    data.rainfall, data.temperature, data.humidity,
                    data.fertilizer, data.pesticides, data.ph,
                    coefficients
                );
                const error = prediction - data.yield_;

                gradients.intercept += (2 / n) * error;
                gradients.rainfall += (2 / n) * error * (data.rainfall / 1000);
                gradients.temperature += (2 / n) * error * (data.temperature / 30);
                gradients.humidity += (2 / n) * error * (data.humidity / 100);
                gradients.fertilizer += (2 / n) * error * (data.fertilizer / 200);
                gradients.pesticides += (2 / n) * error * (data.pesticides / 20);
                gradients.ph += (2 / n) * error * (data.ph / 10);
            }

            // Update coefficients
            coefficients.intercept -= learningRate * gradients.intercept;
            coefficients.rainfall -= learningRate * gradients.rainfall;
            coefficients.temperature -= learningRate * gradients.temperature;
            coefficients.humidity -= learningRate * gradients.humidity;
            coefficients.fertilizer -= learningRate * gradients.fertilizer;
            coefficients.pesticides -= learningRate * gradients.pesticides;
            coefficients.ph -= learningRate * gradients.ph;
        }

        this.coefficients = coefficients;
        this.trained = true;
        console.log('Model trained successfully!');
    }

    /**
     * Linear prediction using trained coefficients
     */
    predictLinear(rainfall, temperature, humidity, fertilizer, pesticides, ph, coef) {
        return coef.intercept +
            coef.rainfall * (rainfall / 1000) +
            coef.temperature * (temperature / 30) +
            coef.humidity * (humidity / 100) +
            coef.fertilizer * (fertilizer / 200) +
            coef.pesticides * (pesticides / 20) +
            coef.ph * (ph / 10);
    }

    /**
     * Main prediction method
     */
    predict(inputData) {
        const { cropType, soilType, area, rainfall, temperature, humidity, fertilizer, pesticides, ph } = inputData;

        // Use the physics-based model for more accurate results
        let yieldPerHectare = this.calculateYield(
            cropType, soilType, rainfall, temperature, humidity,
            fertilizer, pesticides, ph, area
        );

        // Adjust with linear model correction
        const linearPrediction = this.predictLinear(
            rainfall, temperature, humidity, fertilizer, pesticides, ph,
            this.coefficients
        );

        // Blend the two predictions
        const finalYield = Math.round(yieldPerHectare * 0.7 + linearPrediction * 0.3);

        // Calculate confidence based on how optimal the conditions are
        const cropData = CROP_COEFFICIENTS[cropType];
        const tempOptimality = 1 - Math.abs(temperature - cropData.tempOptimal) / 30;
        const rainOptimality = 1 - Math.abs(rainfall - cropData.rainOptimal) / 2000;
        const confidence = Math.round((tempOptimality + rainOptimality) / 2 * 100);

        // Calculate total yield
        const totalYield = Math.round(finalYield * area);

        return {
            yieldPerHectare: finalYield,
            totalYield: totalYield,
            confidence: Math.min(95, Math.max(50, confidence)),
            factors: {
                cropType,
                soilType,
                area,
                rainfall,
                temperature,
                humidity,
                fertilizer,
                pesticides,
                ph
            }
        };
    }
}

// Initialize the predictor
const predictor = new CropYieldPredictor();

// Handle form submission
document.getElementById('predictionForm').addEventListener('submit', function(e) {
    e.preventDefault();

    const predictBtn = document.getElementById('predictBtn');
    const resultContainer = document.getElementById('resultContainer');

    // Show loading state
    predictBtn.innerHTML = '<span class="loading"></span> Analyzing...';
    predictBtn.disabled = true;

    // Get form values
    const inputData = {
        cropType: document.getElementById('cropType').value,
        soilType: document.getElementById('soilType').value,
        area: parseFloat(document.getElementById('area').value),
        rainfall: parseFloat(document.getElementById('rainfall').value),
        temperature: parseFloat(document.getElementById('temperature').value),
        humidity: parseFloat(document.getElementById('humidity').value),
        fertilizer: parseFloat(document.getElementById('fertilizer').value),
        pesticides: parseFloat(document.getElementById('pesticides').value),
        ph: parseFloat(document.getElementById('ph').value)
    };

    // Simulate processing time for better UX
    setTimeout(() => {
        try {
            const result = predictor.predict(inputData);

            // Display results
            resultContainer.className = 'result-container result-success show';
            resultContainer.innerHTML = `
                <div class="result-header">
                    <h2>🎯 Prediction Results</h2>
                </div>
                <div class="result-value">
                    ${result.yieldPerHectare.toLocaleString()}
                    <span class="result-unit">kg/hectare</span>
                </div>
                <div style="font-size: 1.5rem; margin-top: 0.5rem;">
                    Total: <strong>${result.totalYield.toLocaleString()} kg</strong>
                </div>
                <div style="margin-top: 1rem; opacity: 0.9;">
                    Model Confidence: ${result.confidence}%
                </div>
                <div class="result-details">
                    <div class="result-item">
                        <div class="result-item-label">Crop</div>
                        <div class="result-item-value">${result.factors.cropType.charAt(0).toUpperCase() + result.factors.cropType.slice(1)}</div>
                    </div>
                    <div class="result-item">
                        <div class="result-item-label">Soil</div>
                        <div class="result-item-value">${result.factors.soilType.charAt(0).toUpperCase() + result.factors.soilType.slice(1)}</div>
                    </div>
                    <div class="result-item">
                        <div class="result-item-label">Area</div>
                        <div class="result-item-value">${result.factors.area} ha</div>
                    </div>
                    <div class="result-item">
                        <div class="result-item-label">Rainfall</div>
                        <div class="result-item-value">${result.factors.rainfall} mm</div>
                    </div>
                    <div class="result-item">
                        <div class="result-item-label">Temperature</div>
                        <div class="result-item-value">${result.factors.temperature}°C</div>
                    </div>
                    <div class="result-item">
                        <div class="result-item-label">Humidity</div>
                        <div class="result-item-value">${result.factors.humidity}%</div>
                    </div>
                </div>
            `;
        } catch (error) {
            resultContainer.className = 'result-container result-error show';
            resultContainer.innerHTML = `
                <h2>⚠️ Error</h2>
                <p>Something went wrong: ${error.message}</p>
            `;
        }

        // Reset button
        predictBtn.innerHTML = '🔮 Predict Yield';
        predictBtn.disabled = false;
    }, 1500);
});

// Add input validation feedback
document.querySelectorAll('.form-group input, .form-group select').forEach(input => {
    input.addEventListener('input', function() {
        const resultContainer = document.getElementById('resultContainer');
        resultContainer.classList.remove('show');
    });
});

