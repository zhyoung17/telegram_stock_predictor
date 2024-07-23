import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, GridSearchCV
from neuralprophet import NeuralProphet
import numpy as np
import warnings
import logging
import matplotlib.pyplot as plt
from datetime import datetime
from collect_data import collect_data
from preprocess_data import preprocess_data

warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

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

# Function to perform hyperparameter tuning
def optimize_model(model, param_grid, X, y):
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X, y)
    return grid_search.best_estimator_

# Function to difference the data
def difference_data(data, interval=1):
    diff = []
    for i in range(interval, len(data)):
        value = data[i] - data[i - interval]
        diff.append(value)
    return np.array(diff)

# Function to inverse the differencing
def inverse_difference(last_ob, value):
    return value + last_ob

# Function to run all models and get predictions
def run_all_models():
    try:
        # Load the preprocessed data
        logger.info("Loading preprocessed data")
        data = pd.read_csv('preprocessed_data.csv')

        # Prepare data
        X = data[['Low', 'High', 'Close', 'Volume']].values
        y_low = data['Low'].values
        y_high = data['High'].values

        # Difference the data
        X_diff = difference_data(X)
        y_low_diff = difference_data(y_low)
        y_high_diff = difference_data(y_high)

        # Hyperparameter grids
        param_grids = {
            'Random Forest': {
                'n_estimators': [100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5]
            },
            'SVR': {
                'C': [0.1, 1],
                'epsilon': [0.01, 0.1],
                'kernel': ['rbf']
            },
            'Gradient Boosting': {
                'n_estimators': [100, 200],
                'max_depth': [3, 5],
                'learning_rate': [0.01, 0.1]
            }
        }

        models = {
            'Random Forest': RandomForestRegressor(),
            'SVR': SVR(),
            'Gradient Boosting': GradientBoostingRegressor()
        }

        best_model_low = None
        best_rmsfe_low = float('inf')
        best_pred_low = None

        best_model_high = None
        best_rmsfe_high = float('inf')
        best_pred_high = None

        for model_name, model in models.items():
            logger.info(f"Optimizing {model_name} model for low prices")
            optimized_model = optimize_model(model, param_grids[model_name], X_diff, y_low_diff)
            rmsfe_low = kfold_cv_rmsfe(optimized_model, X_diff, y_low_diff)
            if rmsfe_low < best_rmsfe_low:
                best_rmsfe_low = rmsfe_low
                best_model_low = optimized_model
                best_pred_low_diff = optimized_model.predict(X_diff[-8:]).tolist()  # Get the 8-step prediction
                best_pred_low = [inverse_difference(y_low[-1], pred) for pred in best_pred_low_diff]

            logger.info(f"Optimizing {model_name} model for high prices")
            optimized_model = optimize_model(model, param_grids[model_name], X_diff, y_high_diff)
            rmsfe_high = kfold_cv_rmsfe(optimized_model, X_diff, y_high_diff)
            if rmsfe_high < best_rmsfe_high:
                best_rmsfe_high = rmsfe_high
                best_model_high = optimized_model
                best_pred_high_diff = optimized_model.predict(X_diff[-8:]).tolist()  # Get the 8-step prediction
                best_pred_high = [inverse_difference(y_high[-1], pred) for pred in best_pred_high_diff]

        # Adding NeuralProphet
        logger.info("Training NeuralProphet model for low prices")
        np_model_low = NeuralProphet()
        df_low = data[['Date', 'Low']].rename(columns={'Date': 'ds', 'Low': 'y'})
        np_model_low.fit(df_low, freq='D')
        future_low = np_model_low.make_future_dataframe(df_low, periods=8)
        forecast_low = np_model_low.predict(future_low)
        best_pred_low_np = forecast_low['yhat1'].tail(8).tolist()
        rmsfe_low_np = rmsfe(y_low[-8:], best_pred_low_np)
        if rmsfe_low_np < best_rmsfe_low:
            best_rmsfe_low = rmsfe_low_np
            best_model_low = 'NeuralProphet'
            best_pred_low = best_pred_low_np

        logger.info("Training NeuralProphet model for high prices")
        np_model_high = NeuralProphet()
        df_high = data[['Date', 'High']].rename(columns={'Date': 'ds', 'High': 'y'})
        np_model_high.fit(df_high, freq='D')
        future_high = np_model_high.make_future_dataframe(df_high, periods=8)
        forecast_high = np_model_high.predict(future_high)
        best_pred_high_np = forecast_high['yhat1'].tail(8).tolist()
        rmsfe_high_np = rmsfe(y_high[-8:], best_pred_high_np)
        if rmsfe_high_np < best_rmsfe_high:
            best_rmsfe_high = rmsfe_high_np
            best_model_high = 'NeuralProphet'
            best_pred_high = best_pred_high_np

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
    start_date = "2024-01-01"
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
