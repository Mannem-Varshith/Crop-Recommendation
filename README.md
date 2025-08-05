# Crop Recommendation System

A machine learning-based crop recommendation system that predicts the best crops to cultivate based on soil and climate parameters. The system recommends the top 3 crops with yield predictions and determines the appropriate cultivation season (Kharif/Rabi).

## Features

- **Multi-Model Approach**: Uses separate models for crop classification, yield prediction, and season classification
- **Top 3 Recommendations**: Provides the 3 best crop options with confidence scores
- **Yield Prediction**: Estimates yield in quintals per acre
- **Season Classification**: Determines whether crops are suitable for Kharif or Rabi season
- **Easy-to-Use Interface**: Simple command-line interface for getting recommendations

## Input Parameters

The system takes the following parameters as input:

1. **N (Nitrogen)** - Nitrogen content in kg/ha
2. **P (Phosphorus)** - Phosphorus content in kg/ha  
3. **K (Potassium)** - Potassium content in kg/ha
4. **pH** - Soil pH level (0-14)
5. **rainfall** - Rainfall in mm
6. **temperature** - Temperature in °C

## Output

For each recommended crop, the system provides:

- **Crop Name**: The recommended crop
- **Confidence Score**: Probability of the crop being suitable
- **Expected Yield**: Predicted yield in quintals per acre
- **Cultivation Season**: Kharif or Rabi season
- **Season Information**: Sowing and harvesting periods
- **Yield Status**: High/Moderate/Low yield expectation

## Season Information

### Kharif Crops
- **Sowing Period**: June-July
- **Harvesting Period**: September-October
- **Examples**: Rice, Cotton, Maize, Jowar, Ragi

### Rabi Crops  
- **Sowing Period**: October-November
- **Harvesting Period**: March-April
- **Examples**: Wheat, Barley, Mustard, Peas

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Train the Model
First, run the training script to create the model:

```bash
python crop_recommendation_model.py
```

This will:
- Load and preprocess the data
- Train the machine learning models
- Save the trained model as `crop_recommendation_model.pkl`
- Display model performance metrics
- Run test predictions

### Step 2: Use the Web Interface (Recommended)
Launch the beautiful Streamlit web application:

```bash
# Option 1: Using the launcher script
python run_streamlit.py

# Option 2: Direct Streamlit command
streamlit run streamlit_app.py
```

This will open a web browser with an interactive interface where you can:
- Adjust soil and climate parameters using sliders
- Get instant crop recommendations
- View visualizations and charts
- See detailed analysis results

### Step 3: Programmatic Usage
You can also use the model programmatically:

```python
from crop_recommendation_model import CropRecommendationModel

model = CropRecommendationModel()
model.load_model('crop_recommendation_model.pkl')

recommendations = model.predict_crops(
    N=80, P=40, K=40, pH=5.5, rainfall=650, temperature=29
)
```

## Example Usage

```
CROP RECOMMENDATION SYSTEM
============================================================
Please enter the following parameters:
Nitrogen (N) content (kg/ha): 80
Phosphorus (P) content (kg/ha): 40
Potassium (K) content (kg/ha): 40
Soil pH (0-14): 5.5
Rainfall (mm): 650
Temperature (°C): 29

============================================================
CROP RECOMMENDATIONS
============================================================

1. RICE
   Confidence: 85.2%
   Expected Yield: 8.57 quintals per acre
   Cultivation Season: KHARIF
   Season Info: Kharif crops are sown in June-July and harvested in September-October
   Yield Status: High yield expected

2. MAIZE
   Confidence: 12.1%
   Expected Yield: 7.08 quintals per acre
   Cultivation Season: KHARIF
   Season Info: Kharif crops are sown in June-July and harvested in September-October
   Yield Status: High yield expected

3. RAGI
   Confidence: 2.7%
   Expected Yield: 7.13 quintals per acre
   Cultivation Season: KHARIF
   Season Info: Kharif crops are sown in June-July and harvested in September-October
   Yield Status: High yield expected
```

## Model Architecture

The system uses three separate machine learning models:

1. **Crop Classifier**: Random Forest Classifier to predict crop suitability
2. **Yield Predictor**: Random Forest Regressor to predict yield
3. **Season Classifier**: Random Forest Classifier to determine cultivation season

### Features Used
- N, P, K content
- pH level
- Rainfall
- Temperature
- NPK ratio (N/(P+K))
- Total nutrients (N+P+K)

## Data Source

The model is trained on Andhra Pradesh crop production data including:
- Soil nutrient levels (N, P, K)
- Environmental factors (pH, rainfall, temperature)
- Crop production and yield data
- Seasonal information (Kharif/Rabi)

## Files Description

- `crop_recommendation_model.py`: Main training script
- `streamlit_app.py`: Beautiful web interface using Streamlit
- `run_streamlit.py`: Launcher script for the web app
- `crop_recommendation_model.pkl`: Trained model file (generated after training)
- `requirements.txt`: Python dependencies
- `README.md`: This documentation file

## Model Performance

The system typically achieves:
- Crop Classification Accuracy: >90%
- Season Classification Accuracy: >95%
- Yield Prediction RMSE: <2.0 quintals/acre

## Notes

- The model is trained on Andhra Pradesh data and may be most suitable for similar climatic conditions
- Yield predictions are estimates and actual yields may vary based on farming practices and other factors
- Always consult with local agricultural experts for final crop decisions
- The system provides recommendations based on historical data patterns 