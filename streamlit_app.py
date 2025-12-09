import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="🔥 Forest Fire FWI Prediction",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .header {
        text-align: center;
        color: #FF6B35;
        font-size: 2.5em;
        margin-bottom: 10px;
    }
    .subheader {
        text-align: center;
        color: #666;
        font-size: 1.1em;
        margin-bottom: 30px;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Get script directory
script_dir = Path(__file__).parent.absolute()
models_dir = script_dir / "Models"

# Load models with caching
@st.cache_resource
def load_models():
    try:
        ridge_model = joblib.load(models_dir / "ridge_model.pkl")
        scaler = joblib.load(models_dir / "scaler.pkl")
        return ridge_model, scaler
    except FileNotFoundError:
        st.error("❌ Models not found. Ensure Models/ridge_model.pkl and Models/scaler.pkl exist.")
        return None, None

@st.cache_resource
def load_data():
    try:
        csv_path = script_dir / "Algerian_forest_fires_cleaned.csv"
        df = pd.read_csv(csv_path)
        return df
    except FileNotFoundError:
        return None

# Load SHAP module
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# Header
st.markdown('<div class="header">🔥 Forest Fire FWI Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader">Predict Forest Fire Weather Index with AI & SHAP Explainability</div>', unsafe_allow_html=True)

# Load models and data
ridge_model, scaler = load_models()
dataset = load_data()

if ridge_model is None or scaler is None:
    st.stop()

# Sidebar navigation
st.sidebar.title("📋 Navigation")
page = st.sidebar.radio("Select a page:", [
    "🏠 Home",
    "🔮 Make Prediction",
    "📊 Dataset Overview",
    "📈 Feature Insights",
    "ℹ️ About"
])

# ==================== HOME PAGE ====================
if page == "🏠 Home":
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🚀 Quick Start")
        st.info("""
        Welcome to the ML Lifecycle Forest Fire Prediction system!
        
        **Features:**
        - 🔮 Predict FWI (Fire Weather Index) in real-time
        - 🧠 SHAP-based model explainability
        - 📊 Interactive dataset visualization
        - 📈 Feature importance analysis
        """)
    
    with col2:
        st.subheader("📌 Key Metrics")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Model Type", "Ridge Regression")
        with col_b:
            st.metric("Input Features", "9")
        with col_c:
            st.metric("Target", "FWI Index")
    
    st.markdown("---")
    st.subheader("📖 Understanding the Features")
    
    feature_info = {
        "Weather Conditions": {
            "Temperature (°C)": "Current air temperature",
            "Relative Humidity (%)": "Moisture in air (0-100%)",
            "Wind Speed (km/h)": "Speed of wind movement",
            "Rain (mm)": "Recent rainfall amount"
        },
        "Fire Weather Index Codes": {
            "FFMC": "Fine Fuel Moisture Code - moisture of surface litter",
            "DMC": "Duff Moisture Code - moisture in organic matter",
            "DC": "Drought Code - moisture in deep organic layers",
            "ISI": "Initial Spread Index - fire spread rate potential",
            "BUI": "Buildup Index - fuel accumulation indicator"
        }
    }
    
    for category, features in feature_info.items():
        with st.expander(f"📚 {category}"):
            for feature, description in features.items():
                st.write(f"**{feature}**: {description}")

# ==================== PREDICTION PAGE ====================
elif page == "🔮 Make Prediction":
    st.subheader("🔮 Predict Forest Fire Weather Index")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Weather Conditions**")
        temperature = st.slider("Temperature (°C)", -10.0, 50.0, 25.0, 0.1)
        rh = st.slider("Relative Humidity (%)", 0.0, 100.0, 50.0, 1.0)
        ws = st.slider("Wind Speed (km/h)", 0.0, 40.0, 10.0, 0.1)
        rain = st.slider("Rain (mm)", 0.0, 100.0, 0.0, 0.1)
    
    with col2:
        st.write("**Fire Weather Index Codes**")
        ffmc = st.slider("FFMC (Fine Fuel Moisture Code)", 0.0, 100.0, 65.0, 0.1)
        dmc = st.slider("DMC (Duff Moisture Code)", 0.0, 200.0, 20.0, 0.1)
        dc = st.slider("DC (Drought Code)", 0.0, 800.0, 50.0, 0.1)
        isi = st.slider("ISI (Initial Spread Index)", 0.0, 40.0, 5.0, 0.1)
        bui = st.slider("BUI (Buildup Index)", 0.0, 400.0, 50.0, 0.1)
    
    # Prepare features
    features = np.array([[temperature, rh, ws, rain, ffmc, dmc, dc, isi, bui]])
    scaled_features = scaler.transform(features)
    
    # Make prediction
    prediction = ridge_model.predict(scaled_features)[0]
    
    # Display prediction
    st.markdown("---")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📊 Prediction Result")
        
        # Color-coded FWI index
        if prediction < 5:
            color = "🟢"
            risk_level = "Low Risk"
        elif prediction < 15:
            color = "🟡"
            risk_level = "Moderate Risk"
        elif prediction < 30:
            color = "🟠"
            risk_level = "High Risk"
        else:
            color = "🔴"
            risk_level = "Very High Risk"
        
        st.metric("Forest Fire Weather Index (FWI)", f"{prediction:.2f}", delta=None)
        st.write(f"**Risk Level:** {color} {risk_level}")
        
        # FWI scale explanation
        with st.expander("📖 Understanding FWI Scale"):
            st.write("""
            - **0-5**: Low fire risk
            - **5-15**: Moderate fire risk
            - **15-30**: High fire risk
            - **30+**: Very high fire risk (extreme danger)
            """)
    
    with col2:
        # Visualization of input features
        st.markdown("### 📈 Input Features Summary")
        feature_names = ['Temp', 'RH', 'Ws', 'Rain', 'FFMC', 'DMC', 'DC', 'ISI', 'BUI']
        feature_values = [temperature, rh, ws, rain, ffmc, dmc, dc, isi, bui]
        
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(feature_names)))
        ax.barh(feature_names, feature_values, color=colors)
        ax.set_xlabel("Value (normalized scale)")
        ax.set_title("Input Feature Values")
        plt.tight_layout()
        st.pyplot(fig)
    
    # SHAP Explanation
    if SHAP_AVAILABLE:
        st.markdown("---")
        if st.button("🧭 Generate SHAP Explanation"):
            st.info("Computing SHAP values... This may take a moment.")
            
            try:
                # Load background data for SHAP
                if dataset is not None:
                    feature_cols = ['Temperature','RH','Ws','Rain','FFMC','DMC','DC','ISI','BUI']
                    X_bg = dataset[feature_cols].head(50)
                    
                    # Create explainer
                    explainer = shap.KernelExplainer(ridge_model.predict, X_bg)
                    shap_values = explainer.shap_values(features)
                    
                    # Plot
                    fig, ax = plt.subplots(figsize=(10, 6))
                    shap.summary_plot(shap_values, features, feature_names=feature_names, plot_type="bar", show=False)
                    st.pyplot(fig)
                    
                    st.success("✅ SHAP explanation generated!")
                    st.write("""
                    **What this shows:**
                    - Features with longer bars have more impact on the FWI prediction
                    - Red = positive impact on FWI, Blue = negative impact
                    """)
            except Exception as e:
                st.error(f"❌ Error computing SHAP: {str(e)}")

# ==================== DATASET OVERVIEW ====================
elif page == "📊 Dataset Overview":
    st.subheader("📊 Algerian Forest Fires Dataset Overview")
    
    if dataset is not None:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", len(dataset))
        with col2:
            st.metric("Total Features", len(dataset.columns))
        with col3:
            st.metric("Date Range", f"{dataset['year'].min()}-{dataset['year'].max()}")
        with col4:
            st.metric("Regions", dataset['Region'].nunique() if 'Region' in dataset.columns else "N/A")
        
        st.markdown("---")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.write("**Dataset Preview**")
            st.dataframe(dataset.head(10))
        
        with col2:
            st.write("**Statistical Summary**")
            st.dataframe(dataset[['Temperature','RH','Ws','Rain','FFMC','DMC','DC','ISI','BUI','FWI']].describe())
        
        st.markdown("---")
        st.write("**Distribution of Key Features**")
        
        feature_cols = ['Temperature', 'RH', 'Ws', 'Rain', 'FFMC', 'DMC', 'DC', 'ISI', 'BUI', 'FWI']
        selected_features = st.multiselect("Select features to visualize:", feature_cols, default=['Temperature', 'RH', 'FWI'])
        
        if selected_features:
            fig, axes = plt.subplots(1, len(selected_features), figsize=(15, 4))
            if len(selected_features) == 1:
                axes = [axes]
            
            for i, feature in enumerate(selected_features):
                axes[i].hist(dataset[feature], bins=30, color='steelblue', edgecolor='black', alpha=0.7)
                axes[i].set_title(f"Distribution of {feature}")
                axes[i].set_xlabel(feature)
                axes[i].set_ylabel("Frequency")
            
            plt.tight_layout()
            st.pyplot(fig)
    else:
        st.error("❌ Dataset not available")

# ==================== FEATURE INSIGHTS ====================
elif page == "📈 Feature Insights":
    st.subheader("📈 Feature Analysis & Correlations")
    
    if dataset is not None:
        feature_cols = ['Temperature','RH','Ws','Rain','FFMC','DMC','DC','ISI','BUI','FWI']
        X = dataset[feature_cols]
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.write("**Correlation Matrix**")
            fig, ax = plt.subplots(figsize=(10, 8))
            corr = X.corr()
            sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', center=0, square=True, ax=ax, cbar_kws={'label': 'Correlation'})
            ax.set_title("Feature Correlation Matrix")
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            st.write("**Correlation with FWI**")
            fwi_corr = X.corr()['FWI'].sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = ['green' if x > 0 else 'red' for x in fwi_corr.values]
            ax.barh(fwi_corr.index, fwi_corr.values, color=colors, alpha=0.7)
            ax.set_xlabel("Correlation Coefficient")
            ax.set_title("Feature Correlation with FWI")
            ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
            plt.tight_layout()
            st.pyplot(fig)
        
        st.markdown("---")
        st.write("**Top Features Most Correlated with FWI**")
        top_features = fwi_corr.abs().sort_values(ascending=False)[1:6]
        st.bar_chart(top_features)

# ==================== ABOUT PAGE ====================
elif page == "ℹ️ About":
    st.subheader("ℹ️ About This Project")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        ### 🎯 Project Overview
        
        This is a machine learning application for predicting Forest Fire Weather Index (FWI) 
        based on meteorological data and fire-weather indicators from the Algerian forest fires dataset.
        
        ### 🏗️ Technology Stack
        - **Frontend**: Streamlit (interactive web UI)
        - **Backend**: Python, scikit-learn
        - **Model**: Ridge Regression
        - **Explainability**: SHAP (SHapley Additive exPlanations)
        - **Deployment**: Streamlit Cloud
        
        ### 📚 Dataset
        - **Source**: Algerian Forest Fires Dataset
        - **Records**: 245+ fire incidents
        - **Time Period**: 2012-2013
        - **Features**: 9 meteorological and fire-index variables
        """)
    
    with col2:
        st.markdown("""
        ### 🔬 Model Details
        
        **Algorithm**: Ridge Regression (L2 Regularization)
        - Linear regression with penalty for large coefficients
        - Prevents overfitting
        - Provides interpretable feature weights
        
        **Training Process**:
        1. Data normalization using StandardScaler
        2. Ridge regression with optimal alpha
        3. Cross-validation for model evaluation
        4. SHAP integration for explainability
        
        ### 🚀 Features
        - Real-time FWI prediction
        - SHAP-based model explanations
        - Interactive dataset exploration
        - Feature correlation analysis
        - Risk level assessment
        
        ### 👨‍💻 Author
        Shubham Maurya
        
        ### 📄 License
        MIT License
        """)
    
    st.markdown("---")
    st.info("""
    **GitHub Repository**: [ml-lifecycle](https://github.com/ShubhamMaurya2001/ml-lifecycle)
    
    For questions, issues, or contributions, visit the GitHub repository.
    """)
