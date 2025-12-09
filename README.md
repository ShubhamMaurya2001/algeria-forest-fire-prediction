# ML Lifecycle - Algerian Forest Fire Prediction

A Flask-based machine learning web application for predicting Forest Fire Weather Index (FWI) using ridge regression and providing SHAP-based model explainability.

## Features

- **Prediction API**: Predict FWI values based on 9 input features (Temperature, RH, Ws, Rain, FFMC, DMC, DC, ISI, BUI)
- **SHAP Explainability**: Generate feature importance summaries to understand model predictions
- **Interactive Web UI**: User-friendly interface built with Flask and HTML/CSS
- **Pre-trained Model**: Ridge regression model trained on Algerian forest fires dataset

## Project Structure

```
├── application.py                      # Flask app with prediction & SHAP routes
├── explainability.py                   # SHAP computation utilities
├── ridge_and_lasso_regression.ipynb    # Model training & analysis notebook
├── Templates/
│   ├── index.html                      # Home page
│   └── home.html                       # Prediction & explanation form
├── Models/
│   ├── ridge_model.pkl                 # Pre-trained ridge regression model
│   └── scaler.pkl                      # StandardScaler for feature normalization
├── Algerian_forest_fires_cleaned.csv   # Dataset used for training
└── README.md                           # This file
```

## Installation

### Prerequisites
- Python 3.10 or higher
- Git and Git LFS (for model/data files)
- pip or conda package manager

### Steps

1. **Clone the repository**:
```bash
git clone https://github.com/ShubhamMaurya2001/ml-lifecycle.git
cd ml-lifecycle
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```
Required packages: `flask`, `streamlit`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `shap`, `joblib`

3. **Run the Streamlit application** (Recommended):
```bash
streamlit run streamlit_app.py
```

   OR **Run the Flask application**:
```bash
python application.py
```

4. **Access the web interface**:
- **Streamlit**: Open your browser and navigate to `http://localhost:8501`
- **Flask**: Open your browser and navigate to `http://localhost:5000`

## Usage

### Web Interface Navigation

1. **Home Page** (`/`): 
   - Overview of the ML Lifecycle project
   - Links to prediction and explanation pages

2. **Prediction & Explanation Page** (`/predictdata`):
   - **Weather Conditions Section**:
     - Temperature (°C): Current air temperature
     - Relative Humidity (%): Moisture in air (0-100)
     - Wind Speed (km/h): Speed of wind
     - Rain (mm): Recent rainfall amount
   
   - **Fire Weather Index Codes Section**:
     - **FFMC** (Fine Fuel Moisture Code): Indicates moisture of litter and other cured fine fuels
     - **DMC** (Duff Moisture Code): Represents moisture in decomposing organic matter
     - **DC** (Drought Code): Reflects moisture in deep organic layers
     - **ISI** (Initial Spread Index): Describes rate of fire spread potential
     - **BUI** (Buildup Index): Combines DMC and DC to indicate fuel accumulation
   
   - **Action Buttons**:
     - "🔮 Predict FWI": Calculates Forest Fire Weather Index (0-high risk)
     - "🧭 Explain (SHAP)": Shows which features influence the prediction most

## Model Details

- **Algorithm**: Ridge Regression (linear model)
- **Features**: 9 meteorological and fire-index inputs
- **Target**: Forest Fire Weather Index (FWI)
- **Dataset**: Algerian forest fires (cleaned)
- **Data Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Algerian+Forest+Fires+Dataset) - Algerian Forest Fires Dataset

## SHAP Explainability

The SHAP (SHapley Additive exPlanations) module provides model transparency:

### Features
- **Summary Plot**: Bar chart showing average feature importance across all predictions
- **Feature Attribution**: Quantifies how each input contributes to the FWI prediction
- **Model Trustworthiness**: Understand which weather conditions drive fire risk assessments

### How to Use
1. Fill in the 9 input fields on the prediction page
2. Click "🧭 Explain (SHAP)" button
3. View the SHAP summary plot showing feature impact rankings

### Interpretation
- Longer bars = stronger influence on predictions
- Features at the top are most important for decision-making

## Requirements

- Python 3.10+
- Flask
- scikit-learn
- pandas, numpy
- matplotlib, seaborn
- shap
- joblib

## Notes & Large Files

### Dataset Attribution
- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Algerian+Forest+Fires+Dataset)
- **Citation**: Sidi Mohammed Tahir, Abdelaziz Ramdane-Cherif
- **Description**: Real forest fire data from Bejaia and Sidi Bel-Abbes regions in Algeria (2012)
- **Files Included**: 
  - `Algerian_forest_fires_cleaned.csv` - Cleaned dataset for training/testing
  - `Algerian_forest_fires_dataset_UPDATE.csv` - Original dataset

### Model Files
- **ridge_model.pkl**: Pre-trained Ridge Regression model
- **scaler.pkl**: StandardScaler for feature normalization
- Both files are tracked in this repository for easy deployment

## Deployment

### Streamlit Cloud (Recommended - FREE)
1. Push your code to GitHub (already done!)
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Click "New app" → Select repo `ml-lifecycle`, branch `main`, file `streamlit_app.py`
4. Your app will be live in 2-3 minutes at: `https://ml-lifecycle.streamlit.app`

### Flask on Render (FREE)
1. Create account on [Render](https://render.com)
2. Create new Web Service, connect GitHub repo
3. Set build command: `pip install -r requirements.txt`
4. Set start command: `python application.py`

## Project Timeline

1. **Data Preparation**: Cleaned Algerian forest fire dataset from UCI
2. **Model Training**: Ridge Regression with StandardScaler normalization (see `ridge_and_lasso_regression.ipynb`)
3. **Web Development**: Both Flask and Streamlit applications
4. **Explainability**: SHAP integration for model interpretability
5. **Deployment**: Ready for cloud deployment on Streamlit Cloud or Render

## Future Enhancements

- Add more ML algorithms (XGBoost, Random Forest) for comparison
- Implement user authentication and prediction history
- Add batch prediction capability
- Create REST API documentation (Swagger/OpenAPI)
- Add mobile app interface
- Implement database for storing predictions

## Author

Shubham Maurya

## License

MIT (or your preferred license)
