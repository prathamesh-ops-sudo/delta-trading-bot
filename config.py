"""
Configuration module for Forex Trading System
All credentials and settings are loaded from environment variables for security
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum


class TradingMode(Enum):
    DEMO = "demo"
    LIVE = "live"
    BACKTEST = "backtest"


@dataclass
class MT5Config:
    """MetaTrader 5 configuration"""
    login: int = int(os.getenv("MT5_LOGIN", "10008855997"))
    password: str = os.getenv("MT5_PASSWORD", "W_4uOgLp")
    server: str = os.getenv("MT5_SERVER", "MetaQuotes-Demo")
    investor_password: str = os.getenv("MT5_INVESTOR_PASSWORD", "RbS!Oo4f")
    timeout: int = int(os.getenv("MT5_TIMEOUT", "60000"))
    

@dataclass
class TradingConfig:
    """Trading parameters configuration"""
    # Currency pairs to trade
    symbols: List[str] = field(default_factory=lambda: [
        "EURUSD", "GBPUSD", "USDJPY", "USDCHF", 
        "AUDUSD", "USDCAD", "NZDUSD", "EURGBP"
    ])
    
    # Risk management
    max_risk_per_trade: float = 0.02  # 2% max risk per trade
    min_risk_per_trade: float = 0.01  # 1% min risk per trade
    max_daily_drawdown: float = 0.10  # 10% max daily drawdown
    max_total_drawdown: float = 0.25  # 25% max total drawdown
    max_concurrent_trades: int = 5
    max_trades_per_hour: int = 3
    
    # Position sizing
    initial_balance: float = 100.0
    max_leverage: int = 500  # 1:500
    default_leverage: int = 100  # 1:100 default
    
    # Take profit and stop loss
    default_sl_pips: float = 20.0
    default_tp_pips: float = 40.0  # 1:2 risk-reward
    min_rr_ratio: float = 1.5  # Minimum risk-reward ratio
    trailing_sl_pips: float = 10.0
    
    # Trading frequency
    min_trades_per_hour: int = 2
    trade_check_interval: int = 60  # seconds
    
    # Timeframes for analysis
    timeframes: List[str] = field(default_factory=lambda: [
        "M1", "M5", "M15", "H1", "H4", "D1"
    ])
    
    # Trading hours (UTC)
    trading_start_hour: int = 0
    trading_end_hour: int = 24
    avoid_news_minutes: int = 30  # Avoid trading 30 min before/after major news


@dataclass
class MLConfig:
    """Machine Learning configuration"""
    # Model paths
    model_dir: str = os.getenv("MODEL_DIR", "./models")
    
    # LSTM configuration
    lstm_sequence_length: int = 60
    lstm_hidden_units: int = 128
    lstm_layers: int = 3
    lstm_dropout: float = 0.2
    
    # Transformer configuration
    transformer_d_model: int = 64
    transformer_nhead: int = 4
    transformer_num_layers: int = 2
    transformer_dropout: float = 0.1
    
    # Reinforcement Learning (DQN)
    dqn_state_size: int = 50
    dqn_action_size: int = 3  # Buy, Sell, Hold
    dqn_memory_size: int = 10000
    dqn_batch_size: int = 64
    dqn_gamma: float = 0.99  # Discount factor
    dqn_epsilon_start: float = 1.0
    dqn_epsilon_end: float = 0.01
    dqn_epsilon_decay: float = 0.995
    dqn_learning_rate: float = 0.001
    dqn_target_update: int = 10
    
    # Training
    training_epochs: int = 100
    batch_size: int = 32
    validation_split: float = 0.2
    early_stopping_patience: int = 10
    
    # Retraining schedule
    retrain_hour_utc: int = 0  # Midnight UTC
    min_samples_for_retrain: int = 1000
    
    # Feature engineering
    lookback_periods: List[int] = field(default_factory=lambda: [5, 10, 20, 50, 100, 200])
    

@dataclass
class DataConfig:
    """Data configuration"""
    # Database
    db_path: str = os.getenv("DB_PATH", "./data/forex_data.db")
    
    # Data sources
    use_mt5_data: bool = True
    use_yahoo_finance: bool = True
    use_news_api: bool = True
    
    # API keys
    news_api_key: str = os.getenv("NEWS_API_KEY", "")
    alpha_vantage_key: str = os.getenv("ALPHA_VANTAGE_KEY", "")
    
    # Historical data
    historical_years: int = 5
    
    # Data storage
    s3_bucket: str = os.getenv("S3_BUCKET", "forex-trading-data")
    use_s3: bool = os.getenv("USE_S3", "false").lower() == "true"


@dataclass
class AWSConfig:
    """AWS configuration"""
    region: str = os.getenv("AWS_REGION", "us-east-1")
    
    # S3
    s3_bucket: str = os.getenv("S3_BUCKET", "forex-trading-data")
    s3_model_prefix: str = "models/"
    s3_data_prefix: str = "data/"
    s3_logs_prefix: str = "logs/"
    
    # SNS for alerts
    sns_topic_arn: str = os.getenv("SNS_TOPIC_ARN", "")
    alert_email: str = os.getenv("ALERT_EMAIL", "")
    
    # CloudWatch
    cloudwatch_log_group: str = "/forex-trading/logs"
    
    # Lambda
    lambda_retrain_function: str = "forex-model-retrain"


@dataclass
class LoggingConfig:
    """Logging configuration"""
    log_dir: str = os.getenv("LOG_DIR", "./logs")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_to_file: bool = True
    log_to_console: bool = True
    log_to_cloudwatch: bool = os.getenv("LOG_TO_CLOUDWATCH", "false").lower() == "true"
    max_log_size_mb: int = 100
    backup_count: int = 5


@dataclass
class SystemConfig:
    """Main system configuration"""
    mode: TradingMode = TradingMode.DEMO
    mt5: MT5Config = field(default_factory=MT5Config)
    trading: TradingConfig = field(default_factory=TradingConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    data: DataConfig = field(default_factory=DataConfig)
    aws: AWSConfig = field(default_factory=AWSConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # System settings
    heartbeat_interval: int = 60  # seconds
    max_retries: int = 3
    retry_delay: int = 5  # seconds
    
    @classmethod
    def load_from_env(cls) -> 'SystemConfig':
        """Load configuration from environment variables"""
        mode_str = os.getenv("TRADING_MODE", "demo").lower()
        mode = TradingMode(mode_str) if mode_str in [m.value for m in TradingMode] else TradingMode.DEMO
        
        return cls(
            mode=mode,
            mt5=MT5Config(),
            trading=TradingConfig(),
            ml=MLConfig(),
            data=DataConfig(),
            aws=AWSConfig(),
            logging=LoggingConfig()
        )


# Global configuration instance
config = SystemConfig.load_from_env()


# Technical indicator settings
INDICATOR_SETTINGS = {
    "RSI": {"period": 14, "overbought": 70, "oversold": 30},
    "MACD": {"fast": 12, "slow": 26, "signal": 9},
    "BB": {"period": 20, "std_dev": 2},
    "ATR": {"period": 14},
    "EMA": {"periods": [9, 21, 50, 100, 200]},
    "SMA": {"periods": [20, 50, 100, 200]},
    "STOCH": {"k_period": 14, "d_period": 3, "slowing": 3},
    "ADX": {"period": 14, "threshold": 25},
    "CCI": {"period": 20},
    "WILLIAMS_R": {"period": 14},
}


# News impact levels
NEWS_IMPACT = {
    "high": 3,
    "medium": 2,
    "low": 1,
}


# Trading session times (UTC)
TRADING_SESSIONS = {
    "sydney": {"start": 21, "end": 6},
    "tokyo": {"start": 0, "end": 9},
    "london": {"start": 7, "end": 16},
    "new_york": {"start": 12, "end": 21},
}


# Risk disclaimer
DISCLAIMER = """
RISK DISCLAIMER:
Trading foreign exchange (Forex) carries a high level of risk and may not be 
suitable for all investors. The high degree of leverage can work against you 
as well as for you. Before deciding to trade foreign exchange, you should 
carefully consider your investment objectives, level of experience, and risk 
appetite. The possibility exists that you could sustain a loss of some or all 
of your initial investment and therefore you should not invest money that you 
cannot afford to lose. You should be aware of all the risks associated with 
foreign exchange trading and seek advice from an independent financial advisor 
if you have any doubts.

Past performance is not indicative of future results.
"""

print(DISCLAIMER)
