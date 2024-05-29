import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import numpy as np
import warnings
import logging

warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

def run_all_models():
    try:
        # Load the preprocessed data
        logger.info("Loading preprocessed data")
        data = pd.read_csv('preprocessed_data.csv')

        # Function to calculate RMSFE
        def rmsfe(y_true, y_pred):
            return np.sqrt(np.mean((y_true - y_pred)**2))

        # Function to perform k-fold cross-validation and collect RMSFE
        def kfold_cv_rmsfe(model, X, y, n_splits=5):
            kf = KFold(n_splits=n_splits)
            rmsfe_scores = []

            for train_index, test_index in kf.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                rmsfe_scores.append(rmsfe(y_test, y_pred))

            return np.mean(rmsfe_scores)

        # Prepare data
        X = data[['Low', 'High', 'Close']].values
        y_low = data['Low'].values
        y_high = data['High'].values

        models = {
            'Random Forest': RandomForestRegressor(n_estimators=100),
            'XGBoost': XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, verbosity=0),
            'SVR': SVR(kernel='rbf', C=1.0, epsilon=0.1),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1)
        }

        best_model_low = None
        best_rmsfe_low = float('inf')
        best_pred_low = None

        best_model_high = None
        best_rmsfe_high = float('inf')
        best_pred_high = None

        for model_name, model in models.items():
            logger.info(f"Training {model_name} model with cross-validation for low prices")
            rmsfe_low = kfold_cv_rmsfe(model, X, y_low)
            if rmsfe_low < best_rmsfe_low:
                best_rmsfe_low = rmsfe_low
                best_model_low = model_name
                model.fit(X, y_low)  # Fit model on full data
                best_pred_low = model.predict(X[-8:])  # Get the 8-step prediction

            logger.info(f"Training {model_name} model with cross-validation for high prices")
            rmsfe_high = kfold_cv_rmsfe(model, X, y_high)
            if rmsfe_high < best_rmsfe_high:
                best_rmsfe_high = rmsfe_high
                best_model_high = model_name
                model.fit(X, y_high)  # Fit model on full data
                best_pred_high = model.predict(X[-8:])  # Get the 8-step prediction

        pred_rmsfe_low = f"{best_pred_low} ({best_rmsfe_low:.2f})"
        pred_rmsfe_high = f"{best_pred_high} ({best_rmsfe_high:.2f})"

        result_str = (
            f"Best model for low price prediction: {best_model_low}\n"
            f"Low Prediction: {pred_rmsfe_low}\n\n"
            f"Best model for high price prediction: {best_model_high}\n"
            f"High Prediction: {pred_rmsfe_high}"
        )

        logger.info("Optimal model results generated")
        return result_str

    except Exception as e:
        logger.error(f"Error in run_all_models: {str(e)}")
        raise

if __name__ == "__main__":
    result = run_all_models()
    print(result)
