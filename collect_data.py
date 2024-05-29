import yfinance as yf
import pandas as pd
from datetime import datetime

def fetch_stock_data(ticker, start_date, end_date):
    stock = yf.Ticker(ticker)
    df = stock.history(start=start_date, end=end_date, interval="1h")[['Low', 'High', 'Close', 'Volume']]
    df['50_MA'] = df['Close'].rolling(window=50).mean()  # 50-hour Moving Average
    df['200_MA'] = df['Close'].rolling(window=200).mean()  # 200-hour Moving Average
    return df

def fetch_quarterly_financials(ticker):
    stock = yf.Ticker(ticker)
    financials = stock.quarterly_financials.T
    financials = financials.resample('Q').last()  # Ensure financials are quarterly
    return financials

def fetch_balance_sheet(ticker):
    stock = yf.Ticker(ticker)
    balance_sheet = stock.quarterly_balance_sheet.T
    return balance_sheet

def fetch_cash_flow(ticker):
    stock = yf.Ticker(ticker)
    cash_flow = stock.quarterly_cashflow.T
    return cash_flow

def fetch_stats(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    stats = {
        'eps': info.get('trailingEps'),
        'pe_ratio': info.get('trailingPE'),
        'dividend_yield': info.get('dividendYield'),
        'free_cash_flow': info.get('freeCashflow'),
        'debt_to_equity': info.get('debtToEquity'),
    }
    return stats

def collect_data(ticker, start_date, end_date):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    if pd.isna(start_date) or pd.isna(end_date):
        raise ValueError("Both start_date and end_date must be valid dates.")
    
    stock_data = fetch_stock_data(ticker, start_date, end_date)
    financials = fetch_quarterly_financials(ticker)
    balance_sheet = fetch_balance_sheet(ticker)
    cash_flow = fetch_cash_flow(ticker)
    stats = fetch_stats(ticker)

    # Create a date range for merging
    stock_data['Date'] = stock_data.index
    financials['Date'] = financials.index
    balance_sheet['Date'] = balance_sheet.index
    cash_flow['Date'] = cash_flow.index

    # Remove timezone information
    stock_data.index = stock_data.index.tz_localize(None)
    financials.index = financials.index.tz_localize(None)
    balance_sheet.index = balance_sheet.index.tz_localize(None)
    cash_flow.index = cash_flow.index.tz_localize(None)

    # Merge the stock data with financials, repeating financials until new forecast data
    financials_expanded = financials.reindex(pd.date_range(start=financials.index.min(), end=end_date, freq='H')).fillna(method='ffill')
    balance_sheet_expanded = balance_sheet.reindex(pd.date_range(start=balance_sheet.index.min(), end=end_date, freq='H')).fillna(method='ffill')
    cash_flow_expanded = cash_flow.reindex(pd.date_range(start=cash_flow.index.min(), end=end_date, freq='H')).fillna(method='ffill')

    # Rename columns to avoid conflicts
    financials_expanded = financials_expanded.add_suffix('_financial')
    balance_sheet_expanded = balance_sheet_expanded.add_suffix('_balance')
    cash_flow_expanded = cash_flow_expanded.add_suffix('_cashflow')

    combined_data = stock_data.merge(financials_expanded, left_index=True, right_index=True, how='left')
    combined_data = combined_data.merge(balance_sheet_expanded, left_index=True, right_index=True, how='left')
    combined_data = combined_data.merge(cash_flow_expanded, left_index=True, right_index=True, how='left')

    # Add stats to each row
    for key, value in stats.items():
        combined_data[key] = value

    # Remove rows with no stock data
    combined_data = combined_data.dropna(subset=['Low', 'High', 'Close'])

    # Save to CSV
    combined_data.to_csv("combined_data.csv")

    return combined_data

if __name__ == "__main__":
    ticker = "AAPL"
    start_date = "2023-01-01"
    end_date = datetime.today().strftime('%Y-%m-%d')
    
    combined_data = collect_data(ticker, start_date, end_date)
    print(combined_data.head())
