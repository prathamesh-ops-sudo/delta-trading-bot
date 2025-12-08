"""
Risk Management and Position Sizing System
Handles position sizing, stop-loss, take-profit, and risk calculations
"""
import logging
from typing import Dict, Optional, List, Tuple
from datetime import datetime, timedelta
import json

from config import Config

logger = logging.getLogger(__name__)


class RiskManager:
    """Comprehensive risk management for trading operations"""

    def __init__(self):
        self.trade_history = []
        self.daily_trades = []
        self.max_drawdown = 0
        self.peak_balance = 0

    def calculate_position_size(self, account_balance: float, current_price: float,
                               stop_loss_pct: float, signal_confidence: float) -> Dict[str, float]:
        """
        Calculate optimal position size based on risk parameters

        Args:
            account_balance: Available account balance in USD
            current_price: Current asset price
            stop_loss_pct: Stop loss percentage (e.g., 0.02 for 2%)
            signal_confidence: ML model confidence (0-1)

        Returns:
            Dictionary with position size, leverage, and risk metrics
        """
        # Base risk per trade: 1% to 3% of account based on confidence
        base_risk_pct = 0.01 + (0.02 * signal_confidence)  # 1% to 3%

        # Maximum position value
        max_position_value = account_balance * Config.MAX_POSITION_SIZE_PCT

        # Calculate position size based on stop loss
        risk_amount = account_balance * base_risk_pct
        position_value = risk_amount / stop_loss_pct

        # Cap position value
        position_value = min(position_value, max_position_value)

        # Calculate leverage needed
        leverage = position_value / account_balance
        leverage = min(leverage, Config.MAX_LEVERAGE)

        # Adjust position value based on leverage
        final_position_value = account_balance * leverage

        # Calculate contract size
        contract_size = final_position_value / current_price

        # Calculate margin required (assuming initial margin based on leverage)
        margin_required = final_position_value / leverage

        return {
            'contract_size': round(contract_size, 4),
            'position_value': round(final_position_value, 2),
            'leverage': round(leverage, 2),
            'margin_required': round(margin_required, 2),
            'risk_amount': round(risk_amount, 2),
            'risk_percentage': round(base_risk_pct * 100, 2)
        }

    def calculate_stop_loss(self, entry_price: float, side: str,
                           atr: Optional[float] = None) -> float:
        """
        Calculate stop loss price

        Args:
            entry_price: Entry price
            side: 'buy' or 'sell'
            atr: Average True Range (optional, for dynamic stop loss)

        Returns:
            Stop loss price
        """
        # Use ATR-based stop loss if available, otherwise use fixed percentage
        if atr is not None:
            # ATR-based: 1.5x ATR
            stop_distance = atr * 1.5
        else:
            # Fixed percentage stop loss
            stop_distance = entry_price * Config.STOP_LOSS_PCT

        if side == 'buy':
            stop_loss = entry_price - stop_distance
        else:
            stop_loss = entry_price + stop_distance

        return round(stop_loss, 2)

    def calculate_take_profit_levels(self, entry_price: float, side: str) -> List[Dict[str, float]]:
        """
        Calculate multiple take profit levels

        Returns:
            List of dictionaries with price, size_pct for each TP level
        """
        take_profits = []

        for i, (tp_pct, size_pct) in enumerate(zip(Config.TAKE_PROFIT_LEVELS, Config.TAKE_PROFIT_SIZES)):
            if side == 'buy':
                tp_price = entry_price * (1 + tp_pct)
            else:
                tp_price = entry_price * (1 - tp_pct)

            take_profits.append({
                'level': i + 1,
                'price': round(tp_price, 2),
                'size_percentage': size_pct * 100,
                'profit_percentage': tp_pct * 100
            })

        return take_profits

    def can_trade(self, current_timestamp: datetime) -> Tuple[bool, str]:
        """
        Check if trading is allowed based on frequency limits

        Returns:
            (can_trade, reason)
        """
        # Check daily trade limit
        today = current_timestamp.date()
        today_trades = [t for t in self.daily_trades if t.date() == today]

        if len(today_trades) >= Config.MAX_DAILY_TRADES:
            return False, f"Daily trade limit reached ({Config.MAX_DAILY_TRADES})"

        # Check minimum interval between trades
        if self.daily_trades:
            last_trade = self.daily_trades[-1]
            time_diff = (current_timestamp - last_trade).total_seconds()

            if time_diff < Config.MIN_TRADE_INTERVAL:
                wait_time = Config.MIN_TRADE_INTERVAL - time_diff
                return False, f"Wait {int(wait_time)}s before next trade"

        return True, "OK"

    def record_trade(self, trade_data: Dict):
        """Record a trade for tracking"""
        self.trade_history.append(trade_data)
        self.daily_trades.append(datetime.now())

        # Keep only last 30 days of daily trades
        cutoff = datetime.now() - timedelta(days=30)
        self.daily_trades = [t for t in self.daily_trades if t > cutoff]

    def calculate_trade_metrics(self, entry_price: float, exit_price: float,
                               size: float, side: str, leverage: float) -> Dict[str, float]:
        """
        Calculate comprehensive trade metrics

        Returns:
            Dictionary with PnL, ROI, fees, etc.
        """
        # Calculate gross PnL
        if side == 'buy':
            price_change = exit_price - entry_price
        else:
            price_change = entry_price - exit_price

        gross_pnl = price_change * size

        # Calculate fees
        entry_fee = entry_price * size * Config.TAKER_FEE
        exit_fee = exit_price * size * Config.TAKER_FEE
        total_fees = entry_fee + exit_fee

        # Net PnL
        net_pnl = gross_pnl - total_fees

        # ROI calculation (based on margin used, not full position)
        margin_used = (entry_price * size) / leverage
        roi = (net_pnl / margin_used) * 100 if margin_used > 0 else 0

        return {
            'gross_pnl': round(gross_pnl, 2),
            'fees': round(total_fees, 2),
            'net_pnl': round(net_pnl, 2),
            'roi_percentage': round(roi, 2),
            'entry_fee': round(entry_fee, 2),
            'exit_fee': round(exit_fee, 2)
        }

    def update_drawdown(self, current_balance: float):
        """Update maximum drawdown tracking"""
        if current_balance > self.peak_balance:
            self.peak_balance = current_balance

        if self.peak_balance > 0:
            current_drawdown = ((self.peak_balance - current_balance) / self.peak_balance) * 100
            self.max_drawdown = max(self.max_drawdown, current_drawdown)

    def get_performance_stats(self) -> Dict:
        """Calculate overall performance statistics"""
        if not self.trade_history:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'max_drawdown': 0
            }

        total_trades = len(self.trade_history)
        winning_trades = [t for t in self.trade_history if t.get('pnl', 0) > 0]
        losing_trades = [t for t in self.trade_history if t.get('pnl', 0) < 0]

        total_pnl = sum(t.get('pnl', 0) for t in self.trade_history)
        total_wins = sum(t.get('pnl', 0) for t in winning_trades)
        total_losses = abs(sum(t.get('pnl', 0) for t in losing_trades))

        avg_win = total_wins / len(winning_trades) if winning_trades else 0
        avg_loss = total_losses / len(losing_trades) if losing_trades else 0

        profit_factor = total_wins / total_losses if total_losses > 0 else 0

        return {
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': round((len(winning_trades) / total_trades) * 100, 2) if total_trades > 0 else 0,
            'total_pnl': round(total_pnl, 2),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'profit_factor': round(profit_factor, 2),
            'max_drawdown': round(self.max_drawdown, 2)
        }

    def assess_trade_risk(self, signal_confidence: float, market_volatility: float,
                         current_positions: int) -> Dict[str, any]:
        """
        Assess overall risk before taking a trade

        Args:
            signal_confidence: ML model confidence (0-1)
            market_volatility: Current market volatility measure
            current_positions: Number of open positions

        Returns:
            Risk assessment with approval status
        """
        risk_score = 0
        risk_factors = []

        # Signal confidence check
        if signal_confidence < Config.SIGNAL_THRESHOLD:
            risk_score += 30
            risk_factors.append(f"Low signal confidence: {signal_confidence:.2f}")

        # Volatility check (high volatility = higher risk)
        if market_volatility > 0.05:  # 5% volatility threshold
            risk_score += 20
            risk_factors.append(f"High market volatility: {market_volatility:.2%}")

        # Position concentration check
        if current_positions > 0:
            risk_score += 15
            risk_factors.append(f"Already have {current_positions} open position(s)")

        # Recent performance check
        stats = self.get_performance_stats()
        if stats['total_trades'] >= 5:
            recent_trades = self.trade_history[-5:]
            recent_losses = sum(1 for t in recent_trades if t.get('pnl', 0) < 0)

            if recent_losses >= 3:
                risk_score += 25
                risk_factors.append(f"Recent losing streak: {recent_losses}/5 losses")

        # Risk assessment
        if risk_score >= 50:
            approved = False
            risk_level = "HIGH"
        elif risk_score >= 30:
            approved = True
            risk_level = "MEDIUM"
        else:
            approved = True
            risk_level = "LOW"

        return {
            'approved': approved,
            'risk_score': risk_score,
            'risk_level': risk_level,
            'risk_factors': risk_factors,
            'signal_confidence': signal_confidence,
            'market_volatility': market_volatility
        }

    def save_state(self, filepath: str = 'risk_manager_state.json'):
        """Save risk manager state to file"""
        state = {
            'trade_history': self.trade_history,
            'daily_trades': [t.isoformat() for t in self.daily_trades],
            'max_drawdown': self.max_drawdown,
            'peak_balance': self.peak_balance
        }

        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)

        logger.info(f"Risk manager state saved to {filepath}")

    def load_state(self, filepath: str = 'risk_manager_state.json'):
        """Load risk manager state from file"""
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)

            self.trade_history = state.get('trade_history', [])
            self.daily_trades = [datetime.fromisoformat(t) for t in state.get('daily_trades', [])]
            self.max_drawdown = state.get('max_drawdown', 0)
            self.peak_balance = state.get('peak_balance', 0)

            logger.info(f"Risk manager state loaded from {filepath}")
            return True

        except FileNotFoundError:
            logger.info("No previous state found, starting fresh")
            return False
        except Exception as e:
            logger.error(f"Error loading state: {e}")
            return False
