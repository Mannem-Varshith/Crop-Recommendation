import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
import pickle
import warnings
warnings.filterwarnings('ignore')

class CropRecommendationModel:
    def __init__(self):
        self.yield_model = None
        self.crop_classifier = None
        self.season_classifier = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_data(self):
        """Load and preprocess the crop data"""
        try:
            # Load the main dataset
            df = pd.read_csv('Andhra_Pradesh_Crop_Production_with_Yield.csv')
            
            # Clean column names
            df.columns = df.columns.str.strip()
            
            # Remove any unnamed columns
            df = df.drop(columns=[col for col in df.columns if 'Unnamed' in col], errors='ignore')
            
            # Convert yield to numeric, handling any non-numeric values
            df['Yield_quintals_per_acre'] = pd.to_numeric(df['Yield_quintals_per_acre'], errors='coerce')
            
            # Drop rows with missing values
            df = df.dropna()
            
            # Create additional features
            df['NPK_ratio'] = df['N'] / (df['P'] + df['K'])
            df['total_nutrients'] = df['N'] + df['P'] + df['K']
            
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def prepare_data(self, df):
        """Prepare data for modeling"""
        # Features for prediction
        feature_columns = ['N', 'P', 'K', 'pH', 'rainfall', 'temperature', 'NPK_ratio', 'total_nutrients']
        
        # Encode categorical variables
        le_crop = LabelEncoder()
        le_season = LabelEncoder()
        
        df['crop_encoded'] = le_crop.fit_transform(df['Crop'])
        df['season_encoded'] = le_season.fit_transform(df['Crop_Type'])
        
        self.label_encoders['crop'] = le_crop
        self.label_encoders['season'] = le_season
        
        # Prepare features and targets
        X = df[feature_columns]
        y_yield = df['Yield_quintals_per_acre']
        y_crop = df['crop_encoded']
        y_season = df['season_encoded']
        
        return X, y_yield, y_crop, y_season, feature_columns
    
    def train_models(self, X, y_yield, y_crop, y_season):
        """Train the prediction models"""
        # Split data
        X_train, X_test, y_yield_train, y_yield_test, y_crop_train, y_crop_test, y_season_train, y_season_test = train_test_split(
            X, y_yield, y_crop, y_season, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train yield prediction model
        self.yield_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.yield_model.fit(X_train_scaled, y_yield_train)
        
        # Train crop classification model
        self.crop_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.crop_classifier.fit(X_train_scaled, y_crop_train)
        
        # Train season classification model
        self.season_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.season_classifier.fit(X_train_scaled, y_season_train)
        
        # Evaluate models
        yield_pred = self.yield_model.predict(X_test_scaled)
        crop_pred = self.crop_classifier.predict(X_test_scaled)
        season_pred = self.season_classifier.predict(X_test_scaled)
        
        print("Model Performance:")
        print(f"Yield Prediction RMSE: {np.sqrt(mean_squared_error(y_yield_test, yield_pred)):.4f}")
        print(f"Crop Classification Accuracy: {accuracy_score(y_crop_test, crop_pred):.4f}")
        print(f"Season Classification Accuracy: {accuracy_score(y_season_test, season_pred):.4f}")
        
        return X_train_scaled, X_test_scaled
    
    def predict_crops(self, N, P, K, pH, rainfall, temperature):
        """Predict top 3 crops with yield and season"""
        # Prepare input features
        NPK_ratio = N / (P + K) if (P + K) > 0 else 0
        total_nutrients = N + P + K
        
        features = np.array([[N, P, K, pH, rainfall, temperature, NPK_ratio, total_nutrients]])
        features_scaled = self.scaler.transform(features)
        
        # Get crop probabilities
        crop_probs = self.crop_classifier.predict_proba(features_scaled)[0]
        crop_names = self.label_encoders['crop'].classes_
        
        # Get season prediction
        season_pred = self.season_classifier.predict(features_scaled)[0]
        season_name = self.label_encoders['season'].classes_[season_pred]
        
        # Get yield prediction
        yield_pred = self.yield_model.predict(features_scaled)[0]
        
        # Get top 3 crops with highest probabilities
        top_3_indices = np.argsort(crop_probs)[-3:][::-1]
        
        recommendations = []
        for idx in top_3_indices:
            crop_name = crop_names[idx]
            probability = crop_probs[idx]
            
            # For each crop, we need to predict its specific yield
            # We'll use the general yield prediction but adjust based on crop type
            crop_specific_yield = yield_pred
            
            # Adjust yield based on crop type (some crops typically have different yields)
            if crop_name.lower() in ['rice', 'wheat']:
                crop_specific_yield = yield_pred * 1.2  # Higher yielding crops
            elif crop_name.lower() in ['cotton', 'sunflower']:
                crop_specific_yield = yield_pred * 0.8  # Lower yielding crops
            elif crop_name.lower() in ['horsegram', 'moong']:
                crop_specific_yield = yield_pred * 0.6  # Pulses typically have lower yields
            
            recommendations.append({
                'crop': crop_name,
                'probability': probability,
                'yield_quintals_per_acre': crop_specific_yield,
                'estimated_yield_acres': crop_specific_yield,  # For 1 acre
                'season': season_name
            })
        
        return recommendations
    
    def save_model(self, filename='crop_recommendation_model.pkl'):
        """Save the trained model"""
        model_data = {
            'yield_model': self.yield_model,
            'crop_classifier': self.crop_classifier,
            'season_classifier': self.season_classifier,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved as {filename}")
    
    def load_model(self, filename='crop_recommendation_model.pkl'):
        """Load the trained model"""
        with open(filename, 'rb') as f:
            model_data = pickle.load(f)
        
        self.yield_model = model_data['yield_model']
        self.crop_classifier = model_data['crop_classifier']
        self.season_classifier = model_data['season_classifier']
        self.scaler = model_data['scaler']
        self.label_encoders = model_data['label_encoders']
        print(f"Model loaded from {filename}")

def main():
    """Main function to train and test the model"""
    print("Crop Recommendation Model Training...")
    
    # Initialize model
    model = CropRecommendationModel()
    
    # Load data
    df = model.load_data()
    if df is None:
        print("Failed to load data")
        return
    
    print(f"Loaded {len(df)} records")
    print(f"Available crops: {df['Crop'].unique()}")
    print(f"Available seasons: {df['Crop_Type'].unique()}")
    
    # Prepare data
    X, y_yield, y_crop, y_season, feature_columns = model.prepare_data(df)
    
    # Train models
    model.train_models(X, y_yield, y_crop, y_season)
    
    # Save model
    model.save_model()
    
    # Test predictions
    print("\n" + "="*50)
    print("TESTING THE MODEL")
    print("="*50)
    
    # Test case 1: High N, moderate P, K, good rainfall
    print("\nTest Case 1 - Rice-like conditions:")
    test_inputs = {
        'N': 80, 'P': 40, 'K': 40, 'pH': 5.5, 
        'rainfall': 650, 'temperature': 29
    }
    
    recommendations = model.predict_crops(**test_inputs)
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec['crop']} (Probability: {rec['probability']:.3f})")
        print(f"   Yield: {rec['yield_quintals_per_acre']:.2f} quintals/acre")
        print(f"   Season: {rec['season']}")
    
    # Test case 2: Low N, high P, moderate K, low rainfall
    print("\nTest Case 2 - Drought-like conditions:")
    test_inputs = {
        'N': 20, 'P': 60, 'K': 20, 'pH': 6.0, 
        'rainfall': 300, 'temperature': 25
    }
    
    recommendations = model.predict_crops(**test_inputs)
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec['crop']} (Probability: {rec['probability']:.3f})")
        print(f"   Yield: {rec['yield_quintals_per_acre']:.2f} quintals/acre")
        print(f"   Season: {rec['season']}")

if __name__ == "__main__":
    main() 