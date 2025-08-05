import streamlit as st
import pandas as pd
import numpy as np
import pickle
from crop_recommendation_model import CropRecommendationModel
import plotly.express as px
import plotly.graph_objects as go
import os

# Page configuration
st.set_page_config(
    page_title="Crop Recommendation System",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2E8B57;
    }
    .recommendation-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .season-info {
        background-color: #e8f5e8;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained model with error handling"""
    try:
        # Check if model file exists
        model_path = 'crop_recommendation_model.pkl'
        if not os.path.exists(model_path):
            st.error(f"Model file not found: {model_path}")
            return None
        
        model = CropRecommendationModel()
        model.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def main():
    # Header
    st.markdown('<h1 class="main-header">🌾 Crop Recommendation System</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Load model with progress indicator
    with st.spinner("Loading crop recommendation model..."):
        model = load_model()
    
    if model is None:
        st.error("Failed to load the model. Please check the model file and try again.")
        st.stop()
    
    # Sidebar for input parameters
    st.sidebar.markdown("## 📊 Input Parameters")
    st.sidebar.markdown("Enter your soil and climate parameters:")
    
    # Input form
    with st.sidebar.form("crop_recommendation_form"):
        st.markdown("### Soil Nutrients (kg/ha)")
        N = st.slider("Nitrogen (N)", 0, 200, 80, help="Nitrogen content in kg/ha")
        P = st.slider("Phosphorus (P)", 0, 200, 40, help="Phosphorus content in kg/ha")
        K = st.slider("Potassium (K)", 0, 200, 40, help="Potassium content in kg/ha")
        
        st.markdown("### Environmental Factors")
        pH = st.slider("Soil pH", 0.0, 14.0, 5.5, 0.1, help="Soil pH level (0-14)")
        rainfall = st.slider("Rainfall (mm)", 0, 2000, 650, help="Rainfall in mm")
        temperature = st.slider("Temperature (°C)", 0, 50, 29, help="Temperature in °C")
        
        submitted = st.form_submit_button("🌱 Get Recommendations", use_container_width=True)
    
    # Main content area
    if submitted:
        st.markdown("## 📋 Analysis Results")
        
        # Create two columns for layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Get recommendations with error handling
            try:
                with st.spinner("Analyzing soil and climate conditions..."):
                    recommendations = model.predict_crops(N=N, P=P, K=K, pH=pH, rainfall=rainfall, temperature=temperature)
                
                # Display recommendations
                st.markdown("### 🎯 Top 3 Crop Recommendations")
                
                for i, rec in enumerate(recommendations, 1):
                    with st.container():
                        st.markdown(f"""
                        <div class="recommendation-card">
                            <h3>{i}. {rec['crop'].upper()}</h3>
                            <p><strong>Confidence:</strong> {rec['probability']:.1%}</p>
                            <p><strong>Expected Yield:</strong> {rec['yield_quintals_per_acre']:.2f} quintals per acre</p>
                            <p><strong>Cultivation Season:</strong> {rec['season'].upper()}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Season information
                        if rec['season'].lower() == 'kharif':
                            st.markdown("""
                            <div class="season-info">
                                <strong>🌧️ Kharif Season:</strong> Sown in June-July, harvested in September-October
                            </div>
                            """, unsafe_allow_html=True)
                        elif rec['season'].lower() == 'rabi':
                            st.markdown("""
                            <div class="season-info">
                                <strong>❄️ Rabi Season:</strong> Sown in October-November, harvested in March-April
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Yield interpretation
                        if rec['yield_quintals_per_acre'] > 5:
                            yield_status = "🟢 High yield expected"
                        elif rec['yield_quintals_per_acre'] > 3:
                            yield_status = "🟡 Moderate yield expected"
                        else:
                            yield_status = "🔴 Low yield expected"
                        
                        st.markdown(f"**Yield Status:** {yield_status}")
                        st.markdown("---")
                
            except Exception as e:
                st.error(f"Error getting recommendations: {str(e)}")
                st.info("Please try adjusting the input parameters.")
        
        with col2:
            # Display input parameters summary
            st.markdown("### 📊 Input Summary")
            st.markdown(f"""
            <div class="metric-card">
                <p><strong>N:</strong> {N} kg/ha</p>
                <p><strong>P:</strong> {P} kg/ha</p>
                <p><strong>K:</strong> {K} kg/ha</p>
                <p><strong>pH:</strong> {pH}</p>
                <p><strong>Rainfall:</strong> {rainfall} mm</p>
                <p><strong>Temperature:</strong> {temperature}°C</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Create visualizations with error handling
            try:
                st.markdown("### 📈 Nutrient Balance")
                
                # NPK ratio chart
                npk_ratio = N / (P + K) if (P + K) > 0 else 0
                total_nutrients = N + P + K
                
                # Create a gauge chart for NPK ratio
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = npk_ratio,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "NPK Ratio"},
                    delta = {'reference': 1.0},
                    gauge = {
                        'axis': {'range': [None, 3]},
                        'bar': {'color': "darkgreen"},
                        'steps': [
                            {'range': [0, 0.5], 'color': "lightgray"},
                            {'range': [0.5, 1.5], 'color': "lightgreen"},
                            {'range': [1.5, 3], 'color': "green"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 1.5
                        }
                    }
                ))
                fig_gauge.update_layout(height=300)
                st.plotly_chart(fig_gauge, use_container_width=True)
                
                # Nutrient composition pie chart
                fig_pie = px.pie(
                    values=[N, P, K],
                    names=['Nitrogen', 'Phosphorus', 'Potassium'],
                    title="Nutrient Composition",
                    color_discrete_sequence=['#2E8B57', '#3CB371', '#90EE90']
                )
                fig_pie.update_layout(height=300)
                st.plotly_chart(fig_pie, use_container_width=True)
                
            except Exception as e:
                st.warning(f"Could not generate charts: {str(e)}")
    
    # Information section
    st.markdown("---")
    st.markdown("## ℹ️ About the System")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🎯 Model Performance")
        st.markdown("""
        - **Crop Classification Accuracy:** 90.04%
        - **Season Classification Accuracy:** 100%
        - **Yield Prediction RMSE:** 8.72 quintals/acre
        """)
    
    with col2:
        st.markdown("### 🌾 Supported Crops")
        st.markdown("""
        - Rice, Wheat, Cotton, Maize
        - Jowar, Ragi, Sunflower
        - Horsegram, Moong, Sesamum
        - Onion, Potato, Sweet Potato
        - And many more...
        """)
    
    with col3:
        st.markdown("### 📅 Seasons")
        st.markdown("""
        **🌧️ Kharif:** June-July sowing
        **❄️ Rabi:** October-November sowing
        **🌱 Whole Year:** Year-round crops
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🌾 Crop Recommendation System | Powered by Machine Learning</p>
        <p>Based on Andhra Pradesh crop production data</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 