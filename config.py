"""
Configuration Management for Delta Exchange Trading Bot
"""
import os
from typing import Dict, Any

class Config:
    """Centralized configuration for the trading system"""

    # Delta Exchange API Configuration
    # DEMO ENVIRONMENT (for testing)
    DELTA_API_KEY = "X0hXz0ovm7TNahwksM7z2YzRpoCOXR"
    DELTA_API_SECRET = "UelavyXzxDVve0hqoBhTBUQasWL3FEdApbgEu9FW98SlOWAWbqP4XzIB0pUP"
    DELTA_BASE_URL = "https://cdn-ind.testnet.deltaex.org"  # TESTNET India (demo keys)

    # For PRODUCTION (uncomment when ready):
    # DELTA_BASE_URL = "https://api.india.delta.exchange"

    # Trading Configuration
    SYMBOL = "BTCUSD"  # Main trading pair
    TIMEFRAME = "5m"  # 5-minute candles for analysis

    # Risk Management Parameters
    MAX_POSITION_SIZE_PCT = 0.15  # Maximum 15% of capital per trade
    MAX_LEVERAGE = 5  # Conservative leverage (can go up to 20x on Delta)
    STOP_LOSS_PCT = 0.02  # 2% stop loss
    TAKE_PROFIT_LEVELS = [0.03, 0.05, 0.08]  # 3%, 5%, 8% take profit levels
    TAKE_PROFIT_SIZES = [0.4, 0.4, 0.2]  # Partial exits at each level

    # Trading Frequency Control
    MIN_TRADE_INTERVAL = 300  # 5 minutes minimum between trades
    MAX_DAILY_TRADES = 12  # Maximum 12 trades per day
    SIGNAL_THRESHOLD = 0.65  # ML model confidence threshold (0-1)

    # Fee Structure (Delta Exchange typical fees)
    MAKER_FEE = 0.0005  # 0.05%
    TAKER_FEE = 0.001   # 0.1%

    # ML Model Configuration
    LSTM_LOOKBACK = 60  # Use 60 periods for LSTM
    FEATURE_WINDOW = 100  # Use 100 periods for feature calculation
    MODEL_RETRAIN_HOURS = 24  # Retrain models every 24 hours

    # Data Collection
    CANDLES_TO_FETCH = 500  # Fetch 500 candles for analysis

    # Telegram Configuration
    TELEGRAM_BOT_TOKEN = "8325573196:AAEI1UTia5uCgSmsoxmO3aHD3O3fV-WWF0U"
    TELEGRAM_CHAT_ID = ""  # Will be auto-detected on first message

    # Trade Monitoring
    TRADE_CHECK_INTERVAL = 30  # Check trades every 30 seconds
    WEAKNESS_EXIT_THRESHOLD = 0.35  # Exit if signal strength drops below 35%

    # Database and Logging
    DB_PATH = "trading_data.db"
    LOG_LEVEL = "INFO"
    LOG_FILE = "trading_bot.log"

    # Model Paths
    LSTM_MODEL_PATH = "models/lstm_model.h5"
    RF_MODEL_PATH = "models/rf_model.pkl"
    SCALER_PATH = "models/scaler.pkl"

    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        """Return configuration as dictionary"""
        return {
            key: value for key, value in cls.__dict__.items()
            if not key.startswith('_') and not callable(value)
        }

    @classmethod
    def validate_config(cls) -> bool:
        """Validate critical configuration parameters"""
        if not cls.DELTA_API_KEY or not cls.DELTA_API_SECRET:
            raise ValueError("Delta Exchange API credentials not configured")

        if cls.MAX_POSITION_SIZE_PCT <= 0 or cls.MAX_POSITION_SIZE_PCT > 1:
            raise ValueError("MAX_POSITION_SIZE_PCT must be between 0 and 1")

        if cls.SIGNAL_THRESHOLD < 0.5 or cls.SIGNAL_THRESHOLD > 0.95:
            raise ValueError("SIGNAL_THRESHOLD should be between 0.5 and 0.95")

        return True
