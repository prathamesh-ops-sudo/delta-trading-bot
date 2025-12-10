"""
Configuration Management for Crypto Alert System
"""
import os
from typing import Dict, Any

class Config:
    """Centralized configuration for the alerting system"""

    # ============================================================
    # DATA SOURCE CONFIGURATION (Binance Public API - No Auth)
    # ============================================================
    # Multiple symbols to monitor
    SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"]  # 4 trading pairs
    INTERVAL = "5m"     # Kline interval: 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M

    # Legacy single symbol support (for backward compatibility)
    SYMBOL = SYMBOLS[0]

    # ============================================================
    # TELEGRAM CONFIGURATION
    # ============================================================
    TELEGRAM_BOT_TOKEN = "8325573196:AAEI1UTia5uCgSmsoxmO3aHD3O3fV-WWF0U"
    TELEGRAM_CHAT_ID = ""  # Will be auto-detected on first /start message

    # ============================================================
    # ML MODEL CONFIGURATION
    # ============================================================
    LSTM_LOOKBACK = 60          # Use 60 periods for LSTM
    FEATURE_WINDOW = 100        # Use 100 periods for feature calculation
    MODEL_RETRAIN_HOURS = 24    # Retrain models every 24 hours
    SIGNAL_THRESHOLD = 0.65     # ML model confidence threshold (65%)

    # Minimum signal strength for alerts
    BUY_SIGNAL_THRESHOLD = 0.70   # 70% confidence for BUY alerts
    SELL_SIGNAL_THRESHOLD = 0.70  # 70% confidence for SELL alerts

    # ============================================================
    # DATA COLLECTION
    # ============================================================
    CANDLES_TO_FETCH = 500      # Fetch 500 candles for analysis

    # ============================================================
    # ALERT FREQUENCY CONTROL
    # ============================================================
    MIN_ALERT_INTERVAL = 300    # Minimum 5 minutes between alerts
    MAX_DAILY_ALERTS = 20       # Maximum 20 alerts per day
    CHECK_INTERVAL = 60         # Check for signals every 60 seconds

    # ============================================================
    # PRICE MOVEMENT ALERTS
    # ============================================================
    ENABLE_PRICE_ALERTS = True
    PRICE_CHANGE_THRESHOLD_1H = 2.0   # Alert if price moves >2% in 1 hour
    PRICE_CHANGE_THRESHOLD_4H = 5.0   # Alert if price moves >5% in 4 hours
    PRICE_CHANGE_THRESHOLD_24H = 10.0 # Alert if price moves >10% in 24 hours

    # ============================================================
    # TECHNICAL INDICATOR ALERTS
    # ============================================================
    ENABLE_INDICATOR_ALERTS = True
    RSI_OVERSOLD = 30           # RSI below 30 = oversold
    RSI_OVERBOUGHT = 70         # RSI above 70 = overbought

    # ============================================================
    # DATABASE AND LOGGING
    # ============================================================
    DB_PATH = "alert_data.db"
    LOG_LEVEL = "INFO"
    LOG_FILE = "alert_bot.log"

    # ============================================================
    # MODEL PATHS
    # ============================================================
    LSTM_MODEL_PATH = "models/lstm_model.h5"
    RF_MODEL_PATH = "models/rf_model.pkl"
    SCALER_PATH = "models/scaler.pkl"

    # ============================================================
    # ALERT MESSAGE TEMPLATES
    # ============================================================
    ALERT_EMOJI = {
        'buy': '🟢',
        'sell': '🔴',
        'strong_buy': '💚',
        'strong_sell': '❤️',
        'neutral': '⚪',
        'info': 'ℹ️',
        'warning': '⚠️',
        'rocket': '🚀',
        'chart': '📊',
        'money': '💰'
    }

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
        if not cls.TELEGRAM_BOT_TOKEN:
            raise ValueError("Telegram bot token not configured")

        if cls.SIGNAL_THRESHOLD < 0.5 or cls.SIGNAL_THRESHOLD > 0.95:
            raise ValueError("SIGNAL_THRESHOLD should be between 0.5 and 0.95")

        if cls.BUY_SIGNAL_THRESHOLD < 0.5 or cls.BUY_SIGNAL_THRESHOLD > 1.0:
            raise ValueError("BUY_SIGNAL_THRESHOLD should be between 0.5 and 1.0")

        if cls.SELL_SIGNAL_THRESHOLD < 0.5 or cls.SELL_SIGNAL_THRESHOLD > 1.0:
            raise ValueError("SELL_SIGNAL_THRESHOLD should be between 0.5 and 1.0")

        return True
