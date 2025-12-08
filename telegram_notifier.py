"""
Telegram Notification System
Sends real-time trading alerts and updates to Telegram
"""
import logging
from typing import Dict, Optional, List
from datetime import datetime
import requests
import json

from config import Config

logger = logging.getLogger(__name__)


class TelegramNotifier:
    """Send trading notifications via Telegram"""

    def __init__(self):
        self.bot_token = Config.TELEGRAM_BOT_TOKEN
        self.chat_id = Config.TELEGRAM_CHAT_ID
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"

        # Auto-detect chat ID if not configured
        if self.bot_token and not self.chat_id:
            logger.info("Chat ID not configured - attempting auto-detection...")
            detected_chat_id = self._detect_chat_id()
            if detected_chat_id:
                self.chat_id = detected_chat_id
                logger.info(f"✓ Chat ID auto-detected: {self.chat_id}")
                logger.info("Please add this to config.py: TELEGRAM_CHAT_ID = \"{self.chat_id}\"")
            else:
                logger.warning("Could not auto-detect chat ID. Please send any message to your bot first.")

        self.enabled = bool(self.bot_token and self.chat_id)

        if not self.enabled:
            if not self.bot_token:
                logger.warning("Telegram notifications disabled - bot token not configured")
            else:
                logger.warning("Telegram notifications disabled - chat ID not detected. Send any message to your bot.")
        else:
            logger.info("Telegram notifier initialized")

    def send_message(self, message: str, parse_mode: str = "HTML") -> bool:
        """
        Send a text message to Telegram

        Args:
            message: Message text (supports HTML or Markdown)
            parse_mode: "HTML" or "Markdown"

        Returns:
            True if sent successfully
        """
        if not self.enabled:
            logger.debug(f"Telegram disabled. Would have sent: {message}")
            return False

        try:
            url = f"{self.base_url}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": parse_mode,
                "disable_web_page_preview": True
            }

            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()

            logger.debug("Telegram message sent successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")
            return False

    def _detect_chat_id(self) -> Optional[str]:
        """
        Auto-detect chat ID from recent messages
        User must send any message to the bot first
        """
        try:
            url = f"{self.base_url}/getUpdates"
            response = requests.get(url, timeout=10)
            response.raise_for_status()

            data = response.json()

            if data.get('ok') and data.get('result'):
                updates = data['result']
                if updates:
                    # Get the most recent message
                    latest_update = updates[-1]
                    if 'message' in latest_update:
                        chat_id = str(latest_update['message']['chat']['id'])
                        return chat_id

            return None

        except Exception as e:
            logger.error(f"Failed to detect chat ID: {e}")
            return None

    def notify_trade_signal(self, signal: Dict, market_data: Dict) -> bool:
        """Notify about a new trade signal"""
        signal_type = signal['signal']
        confidence = signal['confidence']
        current_price = market_data['current_price']

        # Emoji indicators
        emoji = "🟢" if signal_type == "BUY" else "🔴" if signal_type == "SELL" else "⚪"

        message = f"""
{emoji} <b>TRADE SIGNAL DETECTED</b>

<b>Signal:</b> {signal_type}
<b>Confidence:</b> {confidence:.1%}
<b>Price:</b> ${current_price:,.2f}

<b>ML Model Breakdown:</b>
• LSTM Probability: {signal.get('lstm_probability', 0):.1%}
• Random Forest: {signal.get('rf_probability', 0):.1%}
• Ensemble: {signal.get('ensemble_probability', 0):.1%}

<b>Market Conditions:</b>
• Volatility: {market_data.get('volatility', 0):.2%}
• ATR: ${market_data.get('atr', 0):.2f}

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

        return self.send_message(message)

    def notify_trade_execution(self, trade: Dict) -> bool:
        """Notify about trade execution"""
        side = trade['side'].upper()
        emoji = "✅" if side == "BUY" else "⬇️"

        message = f"""
{emoji} <b>TRADE EXECUTED</b>

<b>Direction:</b> {side}
<b>Entry Price:</b> ${trade['entry_price']:,.2f}
<b>Size:</b> {trade['size']} contracts
<b>Leverage:</b> {trade['leverage']}x
<b>Position Value:</b> ${trade['position_info']['position_value']:,.2f}

<b>Risk Management:</b>
• Stop Loss: ${trade['stop_loss']:,.2f}
• Risk Amount: ${trade['position_info']['risk_amount']:,.2f} ({trade['position_info']['risk_percentage']}%)

<b>Take Profit Levels:</b>
"""

        for tp in trade['take_profit_levels']:
            message += f"• TP{tp['level']}: ${tp['price']:,.2f} ({tp['profit_percentage']:.1f}%) - {tp['size_percentage']:.0f}%\n"

        message += f"\n📊 Order ID: {trade['order_id']}"
        message += f"\n⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

        return self.send_message(message)

    def notify_position_closed(self, position: Dict, reason: str) -> bool:
        """Notify about position closure"""
        pnl = position.get('pnl', 0)
        roi = position.get('roi', 0)

        # Emoji based on PnL
        if pnl > 0:
            emoji = "💰"
            status = "PROFIT"
        elif pnl < 0:
            emoji = "⚠️"
            status = "LOSS"
        else:
            emoji = "➖"
            status = "BREAKEVEN"

        message = f"""
{emoji} <b>POSITION CLOSED - {status}</b>

<b>Reason:</b> {reason}

<b>Trade Summary:</b>
• Entry: ${position['entry_price']:,.2f}
• Exit: ${position.get('exit_price', 0):,.2f}
• Size: {position['size']} contracts
• Leverage: {position['leverage']}x

<b>Results:</b>
• PnL: ${pnl:,.2f}
• ROI: {roi:+.2f}%

📊 Order ID: {position['order_id']}
⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

        return self.send_message(message)

    def notify_risk_alert(self, risk_assessment: Dict) -> bool:
        """Notify about risk management alerts"""
        message = f"""
⚠️ <b>RISK ALERT</b>

<b>Risk Level:</b> {risk_assessment['risk_level']}
<b>Risk Score:</b> {risk_assessment['risk_score']}/100

<b>Risk Factors:</b>
"""

        for factor in risk_assessment.get('risk_factors', []):
            message += f"• {factor}\n"

        message += f"\n⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

        return self.send_message(message)

    def notify_daily_summary(self, stats: Dict, balance: float) -> bool:
        """Send daily performance summary"""
        emoji = "📈" if stats.get('total_pnl', 0) > 0 else "📉"

        message = f"""
{emoji} <b>DAILY PERFORMANCE SUMMARY</b>

<b>Account Status:</b>
• Balance: ${balance:,.2f}
• Total PnL: ${stats.get('total_pnl', 0):,.2f}
• Max Drawdown: {stats.get('max_drawdown', 0):.2f}%

<b>Trading Activity:</b>
• Total Trades: {stats.get('total_trades', 0)}
• Winning: {stats.get('winning_trades', 0)}
• Losing: {stats.get('losing_trades', 0)}
• Win Rate: {stats.get('win_rate', 0):.1f}%

<b>Performance Metrics:</b>
• Avg Win: ${stats.get('avg_win', 0):,.2f}
• Avg Loss: ${stats.get('avg_loss', 0):,.2f}
• Profit Factor: {stats.get('profit_factor', 0):.2f}

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

        return self.send_message(message)

    def notify_position_update(self, position_status: Dict) -> bool:
        """Send position status update"""
        pnl = position_status.get('unrealized_pnl', 0)
        roi = position_status.get('roi_percentage', 0)

        emoji = "📊"
        if roi > 5:
            emoji = "🚀"
        elif roi < -3:
            emoji = "⚠️"

        message = f"""
{emoji} <b>POSITION UPDATE</b>

<b>Current Status:</b>
• Entry: ${position_status['entry_price']:,.2f}
• Current: ${position_status['current_price']:,.2f}
• Unrealized PnL: ${pnl:+,.2f}
• ROI: {roi:+.2f}%

<b>Risk Levels:</b>
• Stop Loss: ${position_status['stop_loss']:,.2f}

📊 Position ID: {position_status['position_id']}
"""

        return self.send_message(message)

    def notify_error(self, error_message: str, details: Optional[str] = None) -> bool:
        """Notify about system errors"""
        message = f"""
🚨 <b>SYSTEM ERROR</b>

<b>Error:</b> {error_message}
"""

        if details:
            message += f"\n<b>Details:</b> {details}"

        message += f"\n⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

        return self.send_message(message)

    def notify_bot_started(self) -> bool:
        """Notify that bot has started"""
        message = f"""
🤖 <b>TRADING BOT STARTED</b>

<b>Configuration:</b>
• Symbol: {Config.SYMBOL}
• Timeframe: {Config.TIMEFRAME}
• Max Leverage: {Config.MAX_LEVERAGE}x
• Signal Threshold: {Config.SIGNAL_THRESHOLD:.0%}

<b>Risk Limits:</b>
• Max Position Size: {Config.MAX_POSITION_SIZE_PCT:.0%}
• Stop Loss: {Config.STOP_LOSS_PCT:.1%}
• Max Daily Trades: {Config.MAX_DAILY_TRADES}

✅ Bot is now monitoring markets...

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

        return self.send_message(message)

    def notify_bot_stopped(self, reason: str = "Manual stop") -> bool:
        """Notify that bot has stopped"""
        message = f"""
🛑 <b>TRADING BOT STOPPED</b>

<b>Reason:</b> {reason}

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

        return self.send_message(message)

    def notify_model_retrained(self, results: Dict) -> bool:
        """Notify about model retraining"""
        message = f"""
🧠 <b>ML MODELS RETRAINED</b>

<b>LSTM Model:</b>
• Accuracy: {results['lstm']['val_accuracy']:.2%}
• Precision: {results['lstm']['val_precision']:.2%}

<b>Random Forest:</b>
• Accuracy: {results['random_forest']['val_accuracy']:.2%}
• Precision: {results['random_forest']['val_precision']:.2%}

<b>Training Data:</b>
• Samples: {results['training_samples']}
• Validation: {results['validation_samples']}

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

        return self.send_message(message)

    def test_connection(self) -> bool:
        """Test Telegram connection"""
        if not self.enabled:
            logger.warning("Cannot test connection - Telegram not configured")
            return False

        test_message = "🔧 Telegram connection test - Bot is working!"
        return self.send_message(test_message)
