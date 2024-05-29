import os
from dotenv import load_dotenv
import pandas as pd
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters, ContextTypes, CallbackContext
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
import logging
import asyncio
import base64
import io
from PIL import Image
from typing import Final

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# State management
is_running = False

# Get the bot token from the environment variable
TOKEN: Final = os.getenv('TELEGRAM_BOT_TOKEN')
BOT_USERNAME: Final = '@ZHStockBot'

# Function to get the previous business day
def previous_business_day(date):
    date -= timedelta(days=1)
    while date.weekday() > 4:  # Monday to Friday are 0-4
        date -= timedelta(days=1)
    return date

# Function to get the predictions
def get_predictions():
    try:
        import main
        result_str = main.run_all_models()
        return result_str
    except Exception as e:
        logger.error(f"Error in get_predictions: {str(e)}")
        raise

# Function to collect and preprocess data for the given ticker
def collect_and_preprocess_data(ticker):
    try:
        import collect_data
        import preprocess_data
        
        logger.info(f"Collecting data for ticker: {ticker}")
        # Collect data
        start_date = "2023-01-01"
        end_date = datetime.today().strftime('%Y-%m-%d')
        collect_data.collect_data(ticker, start_date, end_date)
        
        logger.info("Preprocessing data")
        # Preprocess data
        input_file = 'combined_data.csv'
        output_file = 'preprocessed_data.csv'
        preprocess_data.preprocess_data(input_file, output_file)
    except AttributeError as e:
        logger.error(f"Module attribute error: {e}")
        raise
    except ImportError as e:
        logger.error(f"Module import error: {e}")
        raise

# Asynchronous function to handle the stock prediction process
async def predict_stock(ticker: str):
    try:
        logger.info(f"Starting prediction process for ticker: {ticker}")
        # Collect and preprocess data
        collect_and_preprocess_data(ticker)
        
        # Get the predictions
        result_str = get_predictions()
        logger.info("Prediction process completed")
        return result_str
    except Exception as e:
        logger.error(f"Error in predict_stock: {str(e)}")
        raise

# Asynchronous command handler for /start
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global is_running
    is_running = True
    logger.info("Received /start command")
    await update.message.reply_text("Bot started! Please enter a stock ticker symbol (e.g., AAPL) to get predictions.")

# Asynchronous command handler for /stop
async def stop_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global is_running
    is_running = False
    logger.info("Received /stop command")
    await update.message.reply_text("Bot stopped!")

# Asynchronous command handler for /yes
async def yes_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global is_running
    if is_running:
        await update.message.reply_text("Please enter a new stock ticker symbol (e.g., AAPL) to get predictions.")

# Asynchronous command handler for /help
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    help_text = (
        "Welcome to the Stock Prediction Bot!\n\n"
        "Here is an explanation of the values and those in brackets:\n\n"
        "*Model*: The machine learning model used for the prediction.\n"
        "*Low Prediction*: The predicted lowest stock price for the next day. The value in brackets is the Root Mean Squared Forecast Error (RMSFE), which indicates the prediction error.\n"
        "*High Prediction*: The predicted highest stock price for the next day. The value in brackets is the Root Mean Squared Forecast Error (RMSFE), which indicates the prediction error.\n\n"
        "Commands you can use:\n"
        "/start - Start the bot and enter a stock ticker symbol to get predictions.\n"
        "/stop - Stop the bot.\n"
        "/help - Show this help message.\n"
    )
    await update.message.reply_text(help_text, parse_mode=ParseMode.MARKDOWN)

# Asynchronous function to handle the stock ticker input
async def handle_message(update: Update, context: CallbackContext) -> None:
    global is_running
    if is_running:
        ticker = update.message.text.upper().strip()
        logger.info(f"Received ticker: {ticker}")
        await update.message.reply_text(f"Fetching predictions for {ticker}...")

        try:
            result_str = await predict_stock(ticker)
            await update.message.reply_text(result_str, parse_mode=ParseMode.MARKDOWN)
            # Prompt for a new prediction
            keyboard = [
                [InlineKeyboardButton("Yes", callback_data='yes')],
                [InlineKeyboardButton("No", callback_data='no')]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await update.message.reply_text("Do you want to make another prediction?", reply_markup=reply_markup)
        except Exception as e:
            await update.message.reply_text(f"Error: {str(e)}")
    else:
        await update.message.reply_text("The bot is currently stopped. Please use /start to start the bot.")

# Callback function for the inline buttons
async def button(update: Update, context: CallbackContext) -> None:
    query = update.callback_query
    await query.answer()
    
    if query.data == 'yes':
        await yes_command(query, context)
    elif query.data == 'no':
        global is_running
        is_running = False
        await query.edit_message_text(text="Bot stopped! Use /start to start again.")

if __name__ == '__main__':
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler('start', start_command))
    app.add_handler(CommandHandler('stop', stop_command))
    app.add_handler(CommandHandler('yes', yes_command))
    app.add_handler(CommandHandler('help', help_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(CallbackQueryHandler(button))

    logger.info("Bot is starting... Polling for updates.")
    app.run_polling(poll_interval=3)
