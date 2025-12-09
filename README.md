# ML Lifecycle - Algerian Forest Fire Prediction

A machine learning web application for predicting Forest Fire Weather Index (FWI) using ridge regression and providing SHAP-based model explainability.

🌐 **[Live Demo](https://ml-lifecycle-ridge.streamlit.app/)** - Try the app online now!

📦 **GitHub**: [algeria-forest-fire-prediction](https://github.com/ShubhamMaurya2001/algeria-forest-fire-prediction)

## Features

- **Real-time Prediction**: Predict FWI values based on 9 input features (Temperature, RH, Ws, Rain, FFMC, DMC, DC, ISI, BUI)
- **SHAP Explainability**: Generate feature importance charts to understand model predictions
- **Interactive Web UI**: User-friendly interface built with Streamlit
- **Pre-trained Model**: Ridge regression model trained on Algerian forest fires dataset
- **Data Visualization**: Explore correlations and feature distributions

## Project Structure

```
├── streamlit_app.py                    # Main Streamlit application
├── application.py                      # Flask app (alternative)
├── explainability.py                   # SHAP computation utilities
├── ridge_and_lasso_regression.ipynb    # Model training & analysis notebook
├── Templates/                          # Flask templates (optional)
├── Models/
│   ├── ridge_model.pkl                 # Pre-trained ridge regression model
│   └── scaler.pkl                      # StandardScaler for feature normalization
├── Algerian_forest_fires_cleaned.csv   # Dataset used for training
├── requirements.txt                    # Python dependencies
└── README.md                           # This file
```

## Installation

### Prerequisites
- Python 3.10 or higher
- pip or conda package manager

### Quick Start (5 minutes)

1. **Clone the repository**:
```bash
git clone https://github.com/ShubhamMaurya2001/algeria-forest-fire-prediction.git
cd algeria-forest-fire-prediction
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Run the Streamlit application**:
```bash
streamlit run streamlit_app.py
```

4. **Open in browser**:
```
http://localhost:8501
```

**That's it!** Your app is now running locally.

## Usage

### 🏠 Home Page
- Project overview
- Feature explanations
- Quick links

### 🔮 Make Prediction
- Adjust 9 input parameters with sliders
- Get instant FWI prediction
- View color-coded risk levels:
  - 🟢 Low Risk (0-5)
  - 🟡 Moderate Risk (5-15)
  - 🟠 High Risk (15-30)
  - 🔴 Very High Risk (30+)
- Click "Generate SHAP Explanation" to see which features matter most

### 📊 Dataset Overview
- View dataset statistics
- Explore data distributions
- See sample data

### 📈 Feature Insights
- Correlation heatmap
- Feature importance analysis
- Identify top predictors

### ℹ️ About
- Project information
- Technology stack
- GitHub link

## Model Details

- **Algorithm**: Ridge Regression
- **Input Features**: 9 (Temperature, Humidity, Wind Speed, Rain, FFMC, DMC, DC, ISI, BUI)
- **Prediction Target**: Forest Fire Weather Index (FWI)
- **Dataset**: Algerian Forest Fires (UCI Machine Learning Repository)
- **Explainability**: SHAP (shows which features affect predictions)

## SHAP Explainability

**What is SHAP?** It explains which features (Temperature, Humidity, etc.) have the most impact on the prediction.

**Why it matters:**
- Understand why the model made a prediction
- Build trust in the model
- Identify key fire risk factors

**How to use:**
1. Enter your input values
2. Click "Generate SHAP Explanation"
3. See a chart showing feature importance

## Requirements

- Python 3.10+
- Streamlit
- scikit-learn
- pandas, numpy
- matplotlib, seaborn
- shap
- joblib

## Notes & Large Files

### Dataset
- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Algerian+Forest+Fires+Dataset)
- **Citation**: Sidi Mohammed Tahir, Abdelaziz Ramdane-Cherif
- **Location**: Bejaia and Sidi Bel-Abbes regions, Algeria (2012)

### Model Files
- Pre-trained Ridge Regression model and scaler included in the repository
- No need to train the model yourself!

## Deployment

**Already deployed!** 🚀 Try the live app here: https://ml-lifecycle-ridge.streamlit.app/

### Deploy Your Own Copy (Free)

1. Fork this repository on GitHub
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Click "New app"
4. Select your forked repository and `streamlit_app.py`
5. Click "Deploy"

Your app will be live in 2-3 minutes!

## Project Timeline

1. Data collection from UCI
2. Ridge Regression model training
3. Streamlit web application
4. SHAP explainability integration
5. Deployed on Streamlit Cloud

## Future Ideas

- Add more ML models (Random Forest, XGBoost)
- User login and prediction history
- Batch predictions (upload CSV)
- Mobile app
- Database to store predictions

## Author

Shubham Maurya

## License

MIT (or your preferred license)
