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

1. Clone the repository:
```bash
git clone https://github.com/ShubhamMaurya2001/ml-lifecycle.git
cd ml-lifecycle
```

2. Install dependencies:
```bash
pip install -r ../requirements.txt
```
(Ensure `shap`, `flask`, `scikit-learn`, `pandas`, `numpy`, `matplotlib` are installed)

3. Run the Flask application:
```bash
python application.py
```

4. Open your browser and navigate to:
```
http://localhost:5000
```

## Usage

1. **Home Page** (`/`): Overview and navigation
2. **Prediction Page** (`/predictdata`):
   - Enter weather conditions (Temperature, RH, Ws, Rain) and FWI codes (FFMC, DMC, DC, ISI, BUI)
   - Click "🔮 Predict FWI" to see the predicted value
   - Click "🧭 Explain (SHAP)" to see feature importance visualization

## Model Details

- **Algorithm**: Ridge Regression (linear model)
- **Features**: 9 meteorological and fire-index inputs
- **Target**: Forest Fire Weather Index (FWI)
- **Dataset**: Algerian forest fires (cleaned)

## SHAP Explainability

The SHAP summary plot shows:
- Which features have the most impact on FWI predictions
- The direction and magnitude of each feature's contribution
- Helps understand model behavior and build trust in predictions

## Requirements

- Python 3.10+
- Flask
- scikit-learn
- pandas, numpy
- matplotlib, seaborn
- shap
- joblib

## Notes

- Large model files (`.pkl`) are excluded from version control
- The scaler and model are loaded from the `Models/` folder
- For large-scale deployments, consider using MLflow or containerizing with Docker

## Author

Shubham Maurya

## License

MIT (or your preferred license)
