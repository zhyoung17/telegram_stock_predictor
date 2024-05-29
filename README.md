# Telegram Stock Predictor
A simple Telegram bot that provides 8-step ahead predictions for stock prices using machine learning models.

## Features
Analyzes stock data from January 2023 to the present.
Delivers daily hourly stock predictions using various machine learning models.
Supports multiple stock tickers.

## How To Use
1. **Clone the Git repository:**
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```
2. **Create a Telegram bot using BotFather and obtain a secret key.**
3. **Create a `.env` file and save the secret key:**
    ```makefile
    TELEGRAM_BOT_TOKEN=<your_secret_key>
    ```
4. **Install the necessary packages:**
    ```bash
    pip install -r requirements.txt
    ```
5. **Run the Telegram bot and link it to your bot:**
    ```bash
    python telegram_bot.py
    ```
6. **Input the stock ticker to get predictions.**

## Code Overview

### Machine Learning Models
The bot uses several machine learning models to predict stock prices, including RandomForestRegressor, XGBRegressor, SVR, and GradientBoostingRegressor.

### Data Collection
The bot fetches stock data, quarterly financials, balance sheets, and cash flow statements using the yfinance library.

### Bot Functionality
The bot interacts with users via Telegram, handling commands such as /start, /stop, and stock ticker inputs.

## Contributing
1. Fork the repository.
2. Create a new branch: `git checkout -b feature/your-feature`.
3. Make your changes and commit them: `git commit -m 'Add some feature'`.
4. Push to the branch: `git push origin feature/your-feature`.
5. Submit a pull request.
