"""
Crypto Alert Bot - ML-Powered Telegram Alerts
Monitors crypto markets and sends intelligent buy/sell signals
"""
import time
import signal
import sys
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional

from config import Config
from binance_data_fetcher import BinanceDataFetcher
from feature_engineering import FeatureEngine
from signal_generator import SignalGenerator
from telegram_notifier import TelegramNotifier
from logger_config import setup_logging

logger = logging.getLogger(__name__)


class AlertBot:
    """Main alert bot orchestrator"""

    def __init__(self):
        self.data_fetcher = BinanceDataFetcher()
        self.feature_engine = FeatureEngine()
        self.signal_generator = SignalGenerator()
        self.telegram = TelegramNotifier()

        self.last_alert_time = None
        self.last_signal = None
        self.alerts_today = 0
        self.alert_reset_time = datetime.now().date()

        self.running = False

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info("Shutdown signal received. Stopping bot...")
        self.running = False

    def initialize(self):
        """Initialize bot - load or train models"""
        # Setup logging first
        setup_logging()

        logger.info("Initializing Alert Bot...")

        # Validate configuration
        Config.validate_config()
        logger.info("✓ Configuration validated")

        # Test Binance connection
        if not self.data_fetcher.test_connection():
            raise ConnectionError("Failed to connect to Binance API")

        # Test Telegram connection
        if not self.telegram.test_connection():
            raise ConnectionError("Failed to connect to Telegram")

        # Load or train models
        if self.signal_generator.models_exist():
            logger.info("Loading existing models...")
            self.signal_generator.load_models()
            logger.info("✓ Models loaded successfully")
        else:
            logger.info("No existing models found. Training new models...")
            self._train_models()
            logger.info("✓ Models trained successfully")

        # Send startup message
        self._send_startup_message()

        logger.info("✓ Alert Bot initialized successfully!")

    def _train_models(self):
        """Train ML models with historical data"""
        logger.info("Fetching historical data for training...")

        # Fetch maximum historical data
        klines = self.data_fetcher.get_klines(Config.SYMBOL, Config.INTERVAL, 1000)
        df = self.data_fetcher.klines_to_dataframe(klines)

        # Calculate features
        logger.info("Calculating features...")
        df = self.feature_engine.prepare_data_from_binance(df)
        df = self.feature_engine.calculate_all_features(df)

        # Train models
        feature_cols = self.feature_engine.get_feature_names()
        self.signal_generator.train(df, feature_cols, epochs=30, batch_size=32)

        # Save models
        self.signal_generator.save_models()

    def _send_startup_message(self):
        """Send startup notification"""
        message = f"""
🤖 <b>Alert Bot Started</b>

📊 <b>Monitoring:</b> {Config.SYMBOL}
⏰ <b>Interval:</b> {Config.INTERVAL}
🎯 <b>Signal Threshold:</b> {Config.SIGNAL_THRESHOLD:.0%}
📈 <b>Buy Alert:</b> {Config.BUY_SIGNAL_THRESHOLD:.0%}+
📉 <b>Sell Alert:</b> {Config.SELL_SIGNAL_THRESHOLD:.0%}+
⏱️ <b>Check Interval:</b> {Config.CHECK_INTERVAL}s

🟢 System ready and monitoring markets...
"""
        self.telegram.send_message(message)

    def _should_send_alert(self, signal: str) -> bool:
        """Check if alert should be sent based on frequency limits"""
        now = datetime.now()

        # Reset daily counter
        if now.date() > self.alert_reset_time:
            self.alerts_today = 0
            self.alert_reset_time = now.date()
            logger.info("Daily alert counter reset")

        # Check daily limit
        if self.alerts_today >= Config.MAX_DAILY_ALERTS:
            logger.warning(f"Daily alert limit ({Config.MAX_DAILY_ALERTS}) reached")
            return False

        # Check time since last alert
        if self.last_alert_time:
            time_since_last = (now - self.last_alert_time).total_seconds()
            if time_since_last < Config.MIN_ALERT_INTERVAL:
                logger.debug(f"Alert suppressed (too soon: {time_since_last:.0f}s < {Config.MIN_ALERT_INTERVAL}s)")
                return False

        # Don't send repeated neutral signals
        if signal == 'NEUTRAL' and self.last_signal == 'NEUTRAL':
            return False

        return True

    def _send_signal_alert(self, signal_data: Dict):
        """Send trading signal alert via Telegram"""
        signal = signal_data['signal']
        confidence = signal_data['confidence']
        price = signal_data['price']
        timestamp = signal_data['timestamp']

        # Choose emoji based on signal
        if signal == 'BUY':
            if confidence >= 0.85:
                emoji = Config.ALERT_EMOJI['strong_buy']
                strength = "STRONG BUY"
            else:
                emoji = Config.ALERT_EMOJI['buy']
                strength = "BUY"
        elif signal == 'SELL':
            if confidence <= 0.15:
                emoji = Config.ALERT_EMOJI['strong_sell']
                strength = "STRONG SELL"
            else:
                emoji = Config.ALERT_EMOJI['sell']
                strength = "SELL"
        else:
            emoji = Config.ALERT_EMOJI['neutral']
            strength = "NEUTRAL"

        # Build message
        message = f"""
{emoji} <b>{strength} SIGNAL</b>

{Config.ALERT_EMOJI['chart']} <b>Symbol:</b> {Config.SYMBOL}
{Config.ALERT_EMOJI['money']} <b>Price:</b> ${price:,.2f}
📊 <b>Confidence:</b> {confidence:.1%}

<b>Technical Indicators:</b>
• RSI(14): {signal_data['rsi_14']:.1f}
• MACD Histogram: {signal_data['macd_hist']:.4f}
• BB Position: {signal_data['bb_position']:.2f}

<b>Model Scores:</b>
• LSTM: {signal_data['lstm_score']:.1%}
• Random Forest: {signal_data['rf_score']:.1%}

⏰ <b>Time:</b> {timestamp.strftime('%Y-%m-%d %H:%M:%S')}
"""

        self.telegram.send_message(message)

        # Update counters
        self.last_alert_time = datetime.now()
        self.last_signal = signal
        self.alerts_today += 1

        logger.info(f"✓ Sent {signal} alert (#{self.alerts_today} today)")

    def _check_price_movements(self, df):
        """Check for significant price movements"""
        if not Config.ENABLE_PRICE_ALERTS:
            return

        try:
            current_price = df['close'].iloc[-1]

            # Check 1-hour movement (12 candles for 5m interval)
            if len(df) >= 12:
                price_1h_ago = df['close'].iloc[-12]
                change_1h = ((current_price - price_1h_ago) / price_1h_ago) * 100

                if abs(change_1h) >= Config.PRICE_CHANGE_THRESHOLD_1H:
                    direction = "📈" if change_1h > 0 else "📉"
                    message = f"""
{Config.ALERT_EMOJI['warning']} <b>Price Alert</b>

{direction} <b>{Config.SYMBOL}</b> moved <b>{change_1h:+.2f}%</b> in 1 hour

<b>Current Price:</b> ${current_price:,.2f}
<b>1h Ago:</b> ${price_1h_ago:,.2f}
"""
                    self.telegram.send_message(message)
                    logger.info(f"Sent 1-hour price movement alert: {change_1h:+.2f}%")

        except Exception as e:
            logger.error(f"Error checking price movements: {e}")

    def _check_indicator_alerts(self, signal_data: Dict):
        """Check for extreme indicator values"""
        if not Config.ENABLE_INDICATOR_ALERTS:
            return

        try:
            rsi = signal_data['rsi_14']

            # RSI oversold
            if rsi < Config.RSI_OVERSOLD:
                message = f"""
{Config.ALERT_EMOJI['info']} <b>RSI Alert</b>

📉 <b>{Config.SYMBOL}</b> is <b>OVERSOLD</b>

<b>RSI(14):</b> {rsi:.1f}
<b>Price:</b> ${signal_data['price']:,.2f}

This could indicate a potential buying opportunity.
"""
                self.telegram.send_message(message)
                logger.info(f"Sent RSI oversold alert: {rsi:.1f}")

            # RSI overbought
            elif rsi > Config.RSI_OVERBOUGHT:
                message = f"""
{Config.ALERT_EMOJI['info']} <b>RSI Alert</b>

📈 <b>{Config.SYMBOL}</b> is <b>OVERBOUGHT</b>

<b>RSI(14):</b> {rsi:.1f}
<b>Price:</b> ${signal_data['price']:,.2f}

This could indicate a potential selling opportunity.
"""
                self.telegram.send_message(message)
                logger.info(f"Sent RSI overbought alert: {rsi:.1f}")

        except Exception as e:
            logger.error(f"Error checking indicator alerts: {e}")

    def run_once(self) -> bool:
        """Run one cycle of market analysis"""
        try:
            # Fetch latest data
            logger.info(f"Fetching latest {Config.SYMBOL} data...")
            klines = self.data_fetcher.get_klines(Config.SYMBOL, Config.INTERVAL, Config.CANDLES_TO_FETCH)
            df = self.data_fetcher.klines_to_dataframe(klines)

            # Calculate features
            df = self.feature_engine.prepare_data_from_binance(df)
            df = self.feature_engine.calculate_all_features(df)

            # Generate signal
            signal_data = self.signal_generator.predict(df)

            # Check if alert should be sent
            if self._should_send_alert(signal_data['signal']):
                if signal_data['signal'] != 'NEUTRAL':
                    self._send_signal_alert(signal_data)
                    # Check for extreme indicators
                    self._check_indicator_alerts(signal_data)

            # Check price movements
            self._check_price_movements(df)

            return True

        except Exception as e:
            logger.error(f"Error in run cycle: {e}", exc_info=True)
            return False

    def run(self):
        """Main bot loop"""
        logger.info("Starting Alert Bot main loop...")
        self.running = True

        cycle_count = 0

        while self.running:
            try:
                cycle_count += 1
                logger.info(f"=== Cycle #{cycle_count} ===")

                success = self.run_once()

                if not success:
                    logger.warning("Cycle failed, retrying in 30 seconds...")
                    time.sleep(30)
                    continue

                # Wait for next check
                logger.info(f"Waiting {Config.CHECK_INTERVAL} seconds until next check...")
                time.sleep(Config.CHECK_INTERVAL)

            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received")
                break
            except Exception as e:
                logger.error(f"Unexpected error in main loop: {e}", exc_info=True)
                time.sleep(60)  # Wait 1 minute before retrying

        # Send shutdown message
        try:
            self.telegram.send_message("🔴 <b>Alert Bot Stopped</b>\n\nBot has been shut down.")
        except:
            pass

        logger.info("Alert Bot stopped")


def main():
    """Entry point"""
    print("=" * 70)
    print("  Crypto Alert Bot")
    print("  ML-Powered Telegram Alerts for Cryptocurrency Trading")
    print("=" * 70)
    print()

    try:
        bot = AlertBot()
        bot.initialize()
        bot.run()

    except KeyboardInterrupt:
        logger.info("\nShutdown requested by user")
        sys.exit(0)

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
