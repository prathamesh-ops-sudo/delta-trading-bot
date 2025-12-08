"""
Logging Configuration
Sets up comprehensive logging for the trading system
"""
import logging
import sys
from logging.handlers import RotatingFileHandler
from datetime import datetime
import os

from config import Config


def setup_logging(log_file: str = None, log_level: str = None) -> logging.Logger:
    """
    Configure logging for the trading bot

    Args:
        log_file: Path to log file (optional, uses config default)
        log_level: Logging level (optional, uses config default)

    Returns:
        Configured logger instance
    """
    if log_file is None:
        log_file = Config.LOG_FILE

    if log_level is None:
        log_level = Config.LOG_LEVEL

    # Create logs directory if it doesn't exist
    log_dir = os.path.dirname(log_file) if os.path.dirname(log_file) else "logs"
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create formatters
    detailed_formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    simple_formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )

    # File handler with rotation
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    root_logger.addHandler(console_handler)

    # Suppress noisy third-party loggers
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    logging.getLogger('h5py').setLevel(logging.WARNING)

    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info(f"Trading Bot Logging Initialized - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Log Level: {log_level.upper()}")
    logger.info(f"Log File: {log_file}")
    logger.info("=" * 80)

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class TradingLogger:
    """Custom logger for trading-specific events"""

    def __init__(self, name: str = "TradingBot"):
        self.logger = logging.getLogger(name)

    def log_trade_signal(self, signal: dict):
        """Log trade signal generation"""
        self.logger.info(
            f"Signal: {signal['signal']} | "
            f"Confidence: {signal['confidence']:.2%} | "
            f"LSTM: {signal.get('lstm_probability', 0):.2%} | "
            f"RF: {signal.get('rf_probability', 0):.2%}"
        )

    def log_trade_execution(self, trade: dict):
        """Log trade execution"""
        self.logger.info(
            f"TRADE EXECUTED | {trade['side'].upper()} | "
            f"Size: {trade['size']} | "
            f"Price: ${trade['entry_price']:,.2f} | "
            f"Leverage: {trade['leverage']}x | "
            f"SL: ${trade['stop_loss']:,.2f}"
        )

    def log_position_closed(self, position: dict):
        """Log position closure"""
        pnl = position.get('pnl', 0)
        roi = position.get('roi', 0)
        self.logger.info(
            f"POSITION CLOSED | "
            f"PnL: ${pnl:+,.2f} | "
            f"ROI: {roi:+.2f}% | "
            f"Reason: {position.get('close_reason', 'Unknown')}"
        )

    def log_performance_stats(self, stats: dict):
        """Log performance statistics"""
        self.logger.info("=" * 60)
        self.logger.info("PERFORMANCE STATISTICS")
        self.logger.info("-" * 60)
        self.logger.info(f"Total Trades: {stats['total_trades']}")
        self.logger.info(f"Win Rate: {stats['win_rate']:.1f}%")
        self.logger.info(f"Total PnL: ${stats['total_pnl']:+,.2f}")
        self.logger.info(f"Profit Factor: {stats['profit_factor']:.2f}")
        self.logger.info(f"Max Drawdown: {stats['max_drawdown']:.2f}%")
        self.logger.info("=" * 60)

    def log_error(self, error_msg: str, exc_info: bool = False):
        """Log error with optional exception info"""
        self.logger.error(error_msg, exc_info=exc_info)

    def log_warning(self, warning_msg: str):
        """Log warning"""
        self.logger.warning(warning_msg)

    def log_system_event(self, event_type: str, details: str):
        """Log system events"""
        self.logger.info(f"[{event_type}] {details}")
