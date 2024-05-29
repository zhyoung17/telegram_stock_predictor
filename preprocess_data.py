import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller

def check_stationarity(series):
    result = adfuller(series.dropna())
    is_stationary = result[1] < 0.05  # p-value < 0.05 indicates stationarity
    return is_stationary, result

def make_stationary(df, column):
    df[column + '_diff'] = df[column].diff().dropna()
    return df

def preprocess_data(input_file, output_file):
    df = pd.read_csv(input_file, index_col='Datetime', parse_dates=True)
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)

    columns_to_check = ['Close', 'High', 'Low']
    for col in columns_to_check:
        is_stationary, result = check_stationarity(df[col])
        if not is_stationary:
            df = make_stationary(df, col)

    # Adding hourly moving averages
    df['50_MA'] = df['Close'].rolling(window=50).mean()  # 50-period moving average (50 hours)
    df['200_MA'] = df['Close'].rolling(window=200).mean()  # 200-period moving average (200 hours)

    df.to_csv(output_file)

if __name__ == "__main__":
    import sys
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    preprocess_data(input_file, output_file)
