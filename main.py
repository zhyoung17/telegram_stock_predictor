import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import numpy as np
import warnings
import logging
import pandas as pd
from datetime import datetime
from collect_data import collect_data
from preprocess_data import preprocess_data
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Function to run all models and get predictions
# Function to run all models and get predictions
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
                best_pred_low = model.predict(X[-8:]).tolist()  # Get the 8-step prediction

            logger.info(f"Training {model_name} model with cross-validation for high prices")
            rmsfe_high = kfold_cv_rmsfe(model, X, y_high)
            if rmsfe_high < best_rmsfe_high:
                best_rmsfe_high = rmsfe_high
                best_model_high = model_name
                model.fit(X, y_high)  # Fit model on full data
                best_pred_high = model.predict(X[-8:]).tolist()  # Get the 8-step prediction

        result_str = (
            f"Best model for low price prediction: {best_model_low}\n"
            f"Low Prediction: {best_pred_low} (RMSFE: {best_rmsfe_low:.2f})\n"
            f"Best model for high price prediction: {best_model_high}\n"
            f"High Prediction: {best_pred_high} (RMSFE: {best_rmsfe_high:.2f})"
        )

        logger.info("Optimal model results generated")
        return best_model_low, best_rmsfe_low, best_pred_low, best_model_high, best_rmsfe_high, best_pred_high

    except Exception as e:
        logger.error(f"Error in run_all_models: {str(e)}")
        raise
# Main function to get predictions and plot them
def main():
    ticker = input("Enter the stock ticker: ").strip().upper()
    start_date = "2023-01-01"
    end_date = datetime.today().strftime('%Y-%m-%d')
    
    # Collect and preprocess data
    logger.info(f"Collecting data for {ticker}")
    collect_data(ticker, start_date, end_date)
    input_file = 'combined_data.csv'
    output_file = 'preprocessed_data.csv'
    preprocess_data(input_file, output_file)

    # Get predictions
    logger.info(f"Running models for {ticker}")
    best_model_low, rmsfe_low, low_predictions, best_model_high, rmsfe_high, high_predictions = run_all_models()

    # Plot predictions
    plt.figure(figsize=(12, 6))
    plt.plot(low_predictions, label=f'Low Predictions ({best_model_low}, RMSFE: {rmsfe_low:.2f})', color='blue', marker='o')
    plt.plot(high_predictions, label=f'High Predictions ({best_model_high}, RMSFE: {rmsfe_high:.2f})', color='red', marker='o')
    
    # Annotate points
    for i, (low, high) in enumerate(zip(low_predictions, high_predictions)):
        plt.annotate(f'{low:.2f}', (i, low), textcoords="offset points", xytext=(0,-10), ha='center', color='blue')
        plt.annotate(f'{high:.2f}', (i, high), textcoords="offset points", xytext=(0,10), ha='center', color='red')
    
    plt.title(f'High and Low Price Predictions for {ticker}')
    plt.xlabel('Hour')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()

