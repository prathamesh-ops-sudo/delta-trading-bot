"""
Configuration Management for Professional Crypto Alert System
Built like a veteran trader - Warren Buffett + BlackRock Aladdin style
"""
import os
from typing import Dict, Any, List

class Config:
    """Centralized configuration for professional trading alerts"""

    # ============================================================
    # DATA SOURCE - Using Binance for data (mirrors Delta pricing)
    # ============================================================
    # Delta Exchange symbols mapping
    DELTA_SYMBOLS = {
        "BTCUSD": "BTCUSDT",    # Delta uses BTCUSD, Binance has BTCUSDT
        "ETHUSD": "ETHUSDT",    # Delta uses ETHUSD, Binance has ETHUSDT
        "BNBUSD": "BNBUSDT",    # Delta uses BNBUSD, Binance has BNBUSDT
        "SOLUSD": "SOLUSDT"     # Delta uses SOLUSD, Binance has SOLUSDT
    }

    # Symbols to monitor (Delta Exchange format)
    SYMBOLS = list(DELTA_SYMBOLS.keys())  # ["BTCUSD", "ETHUSD", "BNBUSD", "SOLUSD"]

    # Binance equivalents for data fetching
    BINANCE_SYMBOLS = list(DELTA_SYMBOLS.values())

    # Timeframes for multi-timeframe analysis (like professional traders)
    PRIMARY_TIMEFRAME = "15m"    # Main trading timeframe
    HIGHER_TIMEFRAME = "1h"      # For trend context
    LOWER_TIMEFRAME = "5m"       # For precise entry

    INTERVAL = PRIMARY_TIMEFRAME  # Legacy support

    # Legacy single symbol support
    SYMBOL = SYMBOLS[0]

    # ============================================================
    # TELEGRAM CONFIGURATION
    # ============================================================
    TELEGRAM_BOT_TOKEN = "8325573196:AAEI1UTia5uCgSmsoxmO3aHD3O3fV-WWF0U"
    TELEGRAM_CHAT_ID = ""  # Auto-detected on /start

    # ============================================================
    # PROFESSIONAL TRADING PARAMETERS
    # ============================================================

    # Signal Quality (High bar like Warren Buffett)
    MIN_SIGNAL_CONFIDENCE = 0.75      # 75% minimum (higher quality, fewer signals)
    CONFLUENCE_REQUIRED = 3            # Need 3+ confirmations

    # Balance Buy/Sell signals
    SIGNAL_BIAS_CORRECTION = True     # Prevent sell-heavy bias
    MIN_SIGNALS_BALANCE = 0.4         # 40% minimum for each direction

    # Price Action Requirements
    REQUIRE_PRICE_ACTION_CONFIRMATION = True
    REQUIRE_STRUCTURE_BREAK = False    # Don't force structure breaks
    REQUIRE_TREND_ALIGNMENT = True     # Must align with higher TF trend

    # Risk Management (Single solid TP like pros)
    SINGLE_TP_ONLY = True             # One target, make it count
    RISK_REWARD_RATIO = 2.0           # Minimum 2:1 R/R
    MAX_RISK_PER_TRADE_PCT = 1.0      # 1% max risk

    # Stop Loss Strategy
    STOP_LOSS_METHOD = "STRUCTURE"    # Options: ATR, STRUCTURE, FIXED
    STOP_LOSS_ATR_MULTIPLIER = 1.5    # Tighter stops (not 2x)

    # Take Profit Strategy
    TAKE_PROFIT_METHOD = "STRUCTURE"   # Options: ATR, STRUCTURE, FIXED
    TAKE_PROFIT_ATR_MULTIPLIER = 3.0   # 2:1 R/R minimum

    # ============================================================
    # PRICE ACTION ANALYSIS
    # ============================================================

    # Support/Resistance Detection
    SR_LOOKBACK_PERIODS = 100         # Look back 100 candles
    SR_TOUCH_TOLERANCE = 0.002        # 0.2% tolerance for S/R
    SR_STRENGTH_MIN_TOUCHES = 2       # Minimum 2 touches

    # Confluence Zones
    CONFLUENCE_ZONE_SIZE = 0.005      # 0.5% zone size

    # Trend Detection (Higher TF)
    TREND_SMA_FAST = 20              # 20-period SMA
    TREND_SMA_SLOW = 50              # 50-period SMA
    TREND_STRENGTH_MIN = 0.6         # 60% trend strength minimum

    # Market Structure
    SWING_HIGH_LOW_PERIODS = 10      # 10 candles for swing detection
    STRUCTURE_BREAK_CONFIRMATION = 2  # Need 2 candle closes

    # ============================================================
    # ML MODEL CONFIGURATION (Supporting role, not primary)
    # ============================================================
    LSTM_LOOKBACK = 60
    FEATURE_WINDOW = 100
    MODEL_RETRAIN_HOURS = 24

    # ML acts as FILTER, not signal generator
    ML_AS_FILTER = True               # ML confirms, doesn't generate
    ML_FILTER_THRESHOLD = 0.60        # 60% ML agreement needed

    # ============================================================
    # ALERT FREQUENCY CONTROL
    # ============================================================
    MIN_ALERT_INTERVAL = 900          # 15 minutes (not 5 - quality over quantity)
    MAX_DAILY_ALERTS = 12             # Max 12 alerts/day (3 per symbol)
    CHECK_INTERVAL = 180              # Check every 3 minutes (not 1 - reduce noise)

    # Alert Quality
    ALERT_ONLY_HIGH_PROBABILITY = True
    ALERT_ONLY_WITH_CONFLUENCE = True
    ALERT_REQUIRE_TREND_ALIGNMENT = True

    # ============================================================
    # DATA COLLECTION
    # ============================================================
    CANDLES_TO_FETCH = 500            # Need more for proper S/R detection

    # ============================================================
    # PRICE MOVEMENT ALERTS (Disabled - too noisy)
    # ============================================================
    ENABLE_PRICE_ALERTS = False       # Focus on quality setups only
    ENABLE_INDICATOR_ALERTS = False    # No RSI spam

    # ============================================================
    # DATABASE AND LOGGING
    # ============================================================
    DB_PATH = "professional_alerts.db"
    LOG_LEVEL = "INFO"
    LOG_FILE = "professional_bot.log"

    # ============================================================
    # MODEL PATHS
    # ============================================================
    LSTM_MODEL_PATH = "models/lstm_model.h5"
    RF_MODEL_PATH = "models/rf_model.pkl"
    SCALER_PATH = "models/scaler.pkl"

    # ============================================================
    # ALERT MESSAGE STYLE (Professional, not bot-like)
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
        'money': '💰',
        'target': '🎯',
        'stop': '🛑',
        'confluence': '✨',
        'structure': '🏗️'
    }

    # Professional messaging
    USE_PROFESSIONAL_LANGUAGE = True
    INCLUDE_RATIONALE = True           # Explain WHY
    INCLUDE_CONFLUENCE_DETAILS = True  # Show what aligned
    INCLUDE_MARKET_CONTEXT = True      # Show bigger picture

    @classmethod
    def get_binance_symbol(cls, delta_symbol: str) -> str:
        """Convert Delta Exchange symbol to Binance symbol"""
        return cls.DELTA_SYMBOLS.get(delta_symbol, delta_symbol)

    @classmethod
    def get_delta_symbol(cls, binance_symbol: str) -> str:
        """Convert Binance symbol to Delta Exchange symbol"""
        reverse_map = {v: k for k, v in cls.DELTA_SYMBOLS.items()}
        return reverse_map.get(binance_symbol, binance_symbol)

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

        if cls.RISK_REWARD_RATIO < 1.5:
            raise ValueError("Risk/Reward ratio must be at least 1.5:1")

        if cls.MIN_SIGNAL_CONFIDENCE < 0.7:
            raise ValueError("Minimum signal confidence should be at least 70%")

        return True
