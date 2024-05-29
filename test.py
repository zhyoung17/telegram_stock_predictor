import os
from dotenv import load_dotenv
import pandas as pd
import logging
from datetime import datetime
from collect_data import collect_data
from preprocess_data import preprocess_data
from main import run_all_models
import numpy as np

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Function to get a list of tickers (static list for simplicity)
def get_tickers():
    # Static list of tickers for testing purposes
    return ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN", "NIO", "SOFI", "GME", "BBBY", "AMC"]

# Function to test predictions for a list of tickers
def test_predictions(tickers):
    results = []

    for ticker in tickers:
        try:
            logger.info(f"Testing ticker: {ticker}")

            # Collect and preprocess data
            start_date = "2023-01-01"
            end_date = datetime.today().strftime('%Y-%m-%d')
            collect_data(ticker, start_date, end_date)

            input_file = 'combined_data.csv'
            output_file = 'preprocessed_data.csv'
            preprocess_data(input_file, output_file)

            # Run predictions and get the results
            result_str = run_all_models()

            # Parse the result string to extract the predicted low and high prices and RMSE
            lines = result_str.split('\n')
            low_line = lines[1]
            high_line = lines[3]

            # Extract the values from the parsed lines
            low_price, low_rmse = map(float, low_line.split(': ')[1].strip().strip(')').split(' ('))
            high_price, high_rmse = map(float, high_line.split(': ')[1].strip().strip(')').split(' ('))

            price_difference = abs(high_price - low_price)
            avg_rmse = (low_rmse + high_rmse) / 2

            # Save the result for evaluation
            results.append((ticker, price_difference, avg_rmse))

        except Exception as e:
            logger.error(f"Error testing ticker {ticker}: {str(e)}")
            results.append((ticker, None, None))

    return results

if __name__ == "__main__":
    # Fetch tickers from the static list
    tickers = get_tickers()

    # Test predictions for the tickers
    results = test_predictions(tickers)

    # Create a DataFrame from the results
    results_df = pd.DataFrame(results, columns=['Ticker', 'Price Difference', 'Average RMSE'])

    # Remove rows with None values
    results_df.dropna(inplace=True)

    if not results_df.empty:
        # Determine the stock with the biggest price difference and best RMSE
        max_diff_stock = results_df.loc[results_df['Price Difference'].idxmax()]
        best_rmse_stock = results_df.loc[results_df['Average RMSE'].idxmin()]

        print(f"Stock with the biggest price difference: {max_diff_stock['Ticker']} (Difference: {max_diff_stock['Price Difference']}, Average RMSE: {max_diff_stock['Average RMSE']})")
        print(f"Stock with the best RMSE: {best_rmse_stock['Ticker']} (Difference: {best_rmse_stock['Price Difference']}, Average RMSE: {best_rmse_stock['Average RMSE']})")
    else:
        print("No valid results to display.")

    # Save results to a CSV file
    results_df.to_csv('test_results.csv', index=False)
