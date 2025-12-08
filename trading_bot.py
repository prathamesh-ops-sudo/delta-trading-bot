"""
Main Trading Bot Orchestrator
Coordinates all components for automated cryptocurrency trading
"""
import time
import signal
import sys
from datetime import datetime, timedelta
from typing import Optional
import schedule
import logging

from config import Config
from logger_config import setup_logging, TradingLogger
from delta_exchange_api import DeltaExchangeAPI
from feature_engineering import FeatureEngine
from ml_models import TradingMLModels
from risk_manager import RiskManager
from order_executor import OrderExecutor
from trade_monitor import TradeMonitor
from telegram_notifier import TelegramNotifier

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)
trading_logger = TradingLogger()


class TradingBot:
    """Main trading bot orchestrating all components"""

    def __init__(self):
        self.running = False
        self.last_model_training = None

        # Initialize components
        logger.info("Initializing Trading Bot components...")

        try:
            # API and data processing
            self.api = DeltaExchangeAPI()
            self.feature_engine = FeatureEngine()
            self.ml_models = TradingMLModels()

            # Trading components
            self.risk_manager = RiskManager()
            self.order_executor = OrderExecutor(self.api, self.risk_manager)
            self.trade_monitor = TradeMonitor(
                self.api, self.ml_models, self.feature_engine, self.order_executor
            )

            # Notifications
            self.telegram = TelegramNotifier()

            # Load existing state
            self.risk_manager.load_state()

            logger.info("✓ All components initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize components: {e}", exc_info=True)
            raise

    def initialize_models(self) -> bool:
        """Initialize or train ML models"""
        logger.info("Initializing ML models...")

        try:
            # Try to load existing models
            if self.ml_models.load_models():
                logger.info("✓ Pre-trained models loaded successfully")
                return True

            # No existing models - train new ones
            logger.info("No existing models found - training new models...")
            logger.info("This may take several minutes...")

            # Fetch historical data
            logger.info(f"Fetching {Config.CANDLES_TO_FETCH} historical candles...")
            candles = self.api.get_candles(
                symbol=Config.SYMBOL,
                resolution=Config.TIMEFRAME,
                count=Config.CANDLES_TO_FETCH
            )

            if not candles or len(candles) < Config.LSTM_LOOKBACK:
                logger.error("Insufficient historical data for training")
                return False

            # Prepare data
            df = self.feature_engine.prepare_data(candles)
            df_with_features = self.feature_engine.calculate_all_features(df)

            logger.info(f"Prepared {len(df_with_features)} candles with features")

            # Train models
            results = self.ml_models.train_all_models(df_with_features)

            logger.info("✓ Model training completed")
            logger.info(f"LSTM Validation Accuracy: {results['lstm']['val_accuracy']:.2%}")
            logger.info(f"Random Forest Validation Accuracy: {results['random_forest']['val_accuracy']:.2%}")

            # Notify via Telegram
            self.telegram.notify_model_retrained(results)

            self.last_model_training = datetime.now()

            return True

        except Exception as e:
            logger.error(f"Model initialization failed: {e}", exc_info=True)
            return False

    def check_and_retrain_models(self):
        """Check if models need retraining and retrain if necessary"""
        if self.last_model_training is None:
            self.last_model_training = datetime.now()
            return

        hours_since_training = (datetime.now() - self.last_model_training).total_seconds() / 3600

        if hours_since_training >= Config.MODEL_RETRAIN_HOURS:
            logger.info(f"Models are {hours_since_training:.1f} hours old - retraining...")
            self.initialize_models()

    def analyze_market(self) -> Optional[dict]:
        """
        Analyze current market conditions and generate trading signal

        Returns:
            Dictionary with signal, confidence, and market data
        """
        try:
            logger.debug("Analyzing market conditions...")

            # Fetch recent candles
            candles = self.api.get_candles(
                symbol=Config.SYMBOL,
                resolution=Config.TIMEFRAME,
                count=Config.CANDLES_TO_FETCH
            )

            if not candles:
                logger.error("Failed to fetch market data")
                return None

            # Prepare data
            df = self.feature_engine.prepare_data(candles)
            df_with_features = self.feature_engine.calculate_all_features(df)

            # Prepare prediction input
            X = self.ml_models.prepare_prediction_data(df_with_features)

            if X is None:
                logger.error("Failed to prepare prediction data")
                return None

            # Get ML prediction
            signal = self.ml_models.ensemble_predict(X)

            # Get current market data
            current_price = self.api.get_current_price(Config.SYMBOL)
            atr = df_with_features['atr_14'].iloc[-1] if 'atr_14' in df_with_features.columns else None
            volatility = df_with_features['volatility_ratio'].iloc[-1] if 'volatility_ratio' in df_with_features.columns else 0

            market_data = {
                'current_price': current_price,
                'atr': atr,
                'volatility': volatility,
                'df_with_features': df_with_features
            }

            logger.debug(f"Market analysis complete: {signal['signal']} @ {signal['confidence']:.2%}")

            return {
                'signal': signal,
                'market_data': market_data,
                'timestamp': datetime.now()
            }

        except Exception as e:
            logger.error(f"Market analysis failed: {e}", exc_info=True)
            return None

    def process_signal(self, analysis: dict):
        """
        Process trading signal and execute if conditions are met

        Args:
            analysis: Market analysis result
        """
        try:
            signal = analysis['signal']
            market_data = analysis['market_data']

            # Log signal
            trading_logger.log_trade_signal(signal)

            # Check if signal is actionable
            if signal['signal'] == 'NEUTRAL':
                logger.debug("Signal is neutral - no action")
                return

            if signal['confidence'] < Config.SIGNAL_THRESHOLD:
                logger.debug(f"Signal confidence {signal['confidence']:.2%} below threshold {Config.SIGNAL_THRESHOLD:.2%}")
                return

            # Notify about signal
            self.telegram.notify_trade_signal(signal, market_data)

            # Execute trade
            logger.info(f"Executing {signal['signal']} trade with {signal['confidence']:.2%} confidence")

            trade_result = self.order_executor.execute_trade(signal, market_data)

            if trade_result:
                # Log successful execution
                trading_logger.log_trade_execution(trade_result)

                # Notify via Telegram
                self.telegram.notify_trade_execution(trade_result)

                logger.info("✓ Trade executed successfully")
            else:
                logger.warning("Trade execution failed or was blocked")

        except Exception as e:
            logger.error(f"Signal processing failed: {e}", exc_info=True)
            self.telegram.notify_error("Signal processing failed", str(e))

    def monitor_positions(self):
        """Monitor and manage open positions"""
        try:
            actions = self.trade_monitor.monitor_all_positions()

            for action in actions:
                if action['action'] == 'close':
                    # Position was closed
                    position_id = action['position_id']

                    # Find position in trade history
                    for trade in self.risk_manager.trade_history:
                        if trade.get('order_id') == position_id and trade.get('status') == 'closed':
                            trading_logger.log_position_closed(trade)
                            self.telegram.notify_position_closed(trade, action['reason'])
                            break

            # Cleanup closed positions
            self.trade_monitor.cleanup_closed_positions()

        except Exception as e:
            logger.error(f"Position monitoring failed: {e}", exc_info=True)

    def run_trading_cycle(self):
        """Run a single trading cycle"""
        try:
            logger.debug("=" * 60)
            logger.debug(f"Trading cycle started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

            # Monitor existing positions first
            self.monitor_positions()

            # Check if we should look for new trades
            active_positions = self.order_executor.get_active_positions()

            if len(active_positions) > 0:
                logger.debug(f"Already have {len(active_positions)} active position(s) - skipping new signals")
            else:
                # Analyze market and process signals
                analysis = self.analyze_market()

                if analysis:
                    self.process_signal(analysis)

            logger.debug("Trading cycle completed")

        except Exception as e:
            logger.error(f"Trading cycle error: {e}", exc_info=True)
            self.telegram.notify_error("Trading cycle error", str(e))

    def send_daily_report(self):
        """Send daily performance report"""
        try:
            stats = self.risk_manager.get_performance_stats()
            balance = self.api.get_account_balance()

            # Update drawdown
            self.risk_manager.update_drawdown(balance)

            # Log statistics
            trading_logger.log_performance_stats(stats)

            # Send to Telegram
            self.telegram.notify_daily_summary(stats, balance)

        except Exception as e:
            logger.error(f"Daily report failed: {e}", exc_info=True)

    def start(self):
        """Start the trading bot"""
        logger.info("=" * 80)
        logger.info("STARTING TRADING BOT")
        logger.info("=" * 80)

        try:
            # Validate configuration
            Config.validate_config()
            logger.info("✓ Configuration validated")

            # Initialize models
            if not self.initialize_models():
                logger.error("Failed to initialize models - cannot start bot")
                return

            # Send startup notification
            self.telegram.notify_bot_started()

            # Schedule tasks
            logger.info("Setting up scheduled tasks...")

            # Main trading cycle - every 5 minutes (adjust based on timeframe)
            schedule.every(5).minutes.do(self.run_trading_cycle)

            # Position monitoring - every 30 seconds
            schedule.every(30).seconds.do(self.monitor_positions)

            # Model retraining check - every hour
            schedule.every(1).hours.do(self.check_and_retrain_models)

            # Daily report - at midnight
            schedule.every().day.at("00:00").do(self.send_daily_report)

            # Save state - every hour
            schedule.every(1).hours.do(self.risk_manager.save_state)

            logger.info("✓ Scheduled tasks configured")

            # Set running flag
            self.running = True

            logger.info("=" * 80)
            logger.info("BOT IS NOW RUNNING - Press Ctrl+C to stop")
            logger.info("=" * 80)

            # Main loop
            while self.running:
                schedule.run_pending()
                time.sleep(1)

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received - shutting down...")
            self.stop()
        except Exception as e:
            logger.error(f"Fatal error: {e}", exc_info=True)
            self.telegram.notify_error("Fatal bot error", str(e))
            self.stop()

    def stop(self):
        """Stop the trading bot gracefully"""
        logger.info("=" * 80)
        logger.info("STOPPING TRADING BOT")
        logger.info("=" * 80)

        self.running = False

        try:
            # Close all open positions (optional - comment out if you want to keep positions open)
            # active_positions = self.order_executor.get_active_positions()
            # if active_positions:
            #     logger.info(f"Closing {len(active_positions)} open position(s)...")
            #     self.order_executor.emergency_close_all()

            # Save state
            logger.info("Saving state...")
            self.risk_manager.save_state()

            # Send final report
            logger.info("Sending final report...")
            self.send_daily_report()

            # Send shutdown notification
            self.telegram.notify_bot_stopped("Manual shutdown")

            logger.info("=" * 80)
            logger.info("BOT STOPPED SUCCESSFULLY")
            logger.info("=" * 80)

        except Exception as e:
            logger.error(f"Error during shutdown: {e}", exc_info=True)

        sys.exit(0)


def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info(f"Received signal {signum}")
    if 'bot' in globals():
        bot.stop()
    else:
        sys.exit(0)


def main():
    """Main entry point"""
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Create and start bot
    global bot
    bot = TradingBot()
    bot.start()


if __name__ == "__main__":
    main()
