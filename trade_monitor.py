"""
Trade Monitoring and Exit Strategy System
Monitors open trades and determines optimal exit points based on:
- Trade strength deterioration
- ML signal weakening
- Technical indicator changes
- Stop loss / Take profit hits
"""
import logging
from typing import Dict, List, Optional
from datetime import datetime
import pandas as pd
import numpy as np

from delta_exchange_api import DeltaExchangeAPI
from ml_models import TradingMLModels
from feature_engineering import FeatureEngine
from order_executor import OrderExecutor
from config import Config

logger = logging.getLogger(__name__)


class TradeMonitor:
    """Monitor and manage open trades with intelligent exit strategies"""

    def __init__(self, api: DeltaExchangeAPI, ml_models: TradingMLModels,
                 feature_engine: FeatureEngine, order_executor: OrderExecutor):
        self.api = api
        self.ml_models = ml_models
        self.feature_engine = feature_engine
        self.order_executor = order_executor
        self.position_signals = {}  # Store signal history for each position

    def monitor_all_positions(self) -> List[Dict]:
        """
        Monitor all active positions and make exit decisions

        Returns:
            List of actions taken
        """
        actions = []
        active_positions = self.order_executor.get_active_positions()

        if not active_positions:
            return actions

        logger.info(f"Monitoring {len(active_positions)} active position(s)")

        # Get current market data
        current_data = self._get_current_market_data()

        if not current_data:
            logger.error("Failed to fetch market data for monitoring")
            return actions

        for position in active_positions:
            action = self._monitor_position(position, current_data)
            if action:
                actions.append(action)

        return actions

    def _monitor_position(self, position: Dict, market_data: Dict) -> Optional[Dict]:
        """
        Monitor a single position and determine if exit is needed

        Returns:
            Action dictionary if position should be exited, None otherwise
        """
        position_id = position['order_id']
        current_price = market_data['current_price']

        logger.debug(f"Monitoring position {position_id}: "
                    f"Entry ${position['entry_price']}, Current ${current_price}")

        # Check stop loss
        if self._check_stop_loss(position, current_price):
            logger.warning(f"Stop loss triggered for position {position_id}")
            self.order_executor.close_position(position_id, reason="Stop loss triggered")
            return {
                'position_id': position_id,
                'action': 'close',
                'reason': 'stop_loss',
                'price': current_price
            }

        # Check take profit
        tp_hit = self._check_take_profit(position, current_price)
        if tp_hit:
            logger.info(f"Take profit level {tp_hit} hit for position {position_id}")
            # Take profit orders should execute automatically, just log it
            return {
                'position_id': position_id,
                'action': 'partial_close',
                'reason': f'take_profit_level_{tp_hit}',
                'price': current_price
            }

        # Check trade strength using ML model
        signal_strength = self._assess_trade_strength(position, market_data)

        if signal_strength is not None:
            # Store signal strength history
            if position_id not in self.position_signals:
                self.position_signals[position_id] = []

            self.position_signals[position_id].append({
                'timestamp': datetime.now().isoformat(),
                'strength': signal_strength,
                'price': current_price
            })

            # Exit if signal strength drops below threshold
            if signal_strength < Config.WEAKNESS_EXIT_THRESHOLD:
                logger.warning(f"Trade weakness detected for position {position_id}: "
                             f"Signal strength {signal_strength:.2%}")

                self.order_executor.close_position(position_id, reason="Trade weakness detected")
                return {
                    'position_id': position_id,
                    'action': 'close',
                    'reason': 'weak_signal',
                    'signal_strength': signal_strength,
                    'price': current_price
                }

            # Check for signal reversal
            if self._detect_signal_reversal(position, signal_strength):
                logger.warning(f"Signal reversal detected for position {position_id}")
                self.order_executor.close_position(position_id, reason="Signal reversal")
                return {
                    'position_id': position_id,
                    'action': 'close',
                    'reason': 'signal_reversal',
                    'price': current_price
                }

        # Check trailing stop
        trailing_stop = self._check_trailing_stop(position, current_price, market_data)
        if trailing_stop:
            logger.info(f"Trailing stop triggered for position {position_id}")
            self.order_executor.close_position(position_id, reason="Trailing stop")
            return {
                'position_id': position_id,
                'action': 'close',
                'reason': 'trailing_stop',
                'price': current_price
            }

        # No action needed
        return None

    def _check_stop_loss(self, position: Dict, current_price: float) -> bool:
        """Check if stop loss is hit"""
        stop_loss = position['stop_loss']
        side = position['side']

        if side == 'buy':
            return current_price <= stop_loss
        else:
            return current_price >= stop_loss

    def _check_take_profit(self, position: Dict, current_price: float) -> Optional[int]:
        """
        Check if any take profit level is hit

        Returns:
            Take profit level number if hit, None otherwise
        """
        tp_levels = position.get('take_profit_levels', [])
        side = position['side']

        for tp in tp_levels:
            tp_price = tp['price']

            if side == 'buy' and current_price >= tp_price:
                return tp['level']
            elif side == 'sell' and current_price <= tp_price:
                return tp['level']

        return None

    def _assess_trade_strength(self, position: Dict, market_data: Dict) -> Optional[float]:
        """
        Assess current trade strength using ML model

        Returns:
            Signal strength (0-1) or None if assessment fails
        """
        try:
            # Get historical data with features
            df = market_data.get('df_with_features')

            if df is None or len(df) < Config.LSTM_LOOKBACK:
                return None

            # Prepare prediction data
            X = self.ml_models.prepare_prediction_data(df)

            if X is None:
                return None

            # Get current signal
            current_signal = self.ml_models.ensemble_predict(X)

            # Determine if signal aligns with position
            position_side = position['side']

            if position_side == 'buy':
                # For long positions, we want BUY signal
                if current_signal['signal'] == 'BUY':
                    strength = current_signal['confidence']
                else:
                    # Signal is against position
                    strength = 1 - current_signal['confidence']
            else:
                # For short positions, we want SELL signal
                if current_signal['signal'] == 'SELL':
                    strength = current_signal['confidence']
                else:
                    # Signal is against position
                    strength = 1 - current_signal['confidence']

            return strength

        except Exception as e:
            logger.error(f"Error assessing trade strength: {e}")
            return None

    def _detect_signal_reversal(self, position: Dict, current_strength: float) -> bool:
        """
        Detect if signal has reversed against the position

        Returns:
            True if reversal detected
        """
        position_id = position['order_id']

        if position_id not in self.position_signals:
            return False

        signal_history = self.position_signals[position_id]

        # Need at least 3 data points
        if len(signal_history) < 3:
            return False

        # Get recent strengths
        recent_strengths = [s['strength'] for s in signal_history[-3:]]

        # Check for consistent decline
        if all(recent_strengths[i] > recent_strengths[i+1] for i in range(len(recent_strengths)-1)):
            # Strength declining consistently
            decline_rate = (recent_strengths[0] - recent_strengths[-1]) / recent_strengths[0]

            # If declined by more than 30%, consider it a reversal
            if decline_rate > 0.3:
                return True

        return False

    def _check_trailing_stop(self, position: Dict, current_price: float,
                            market_data: Dict) -> bool:
        """
        Check trailing stop based on highest price since entry

        Returns:
            True if trailing stop should trigger
        """
        entry_price = position['entry_price']
        side = position['side']

        # Get ATR for dynamic trailing
        atr = market_data.get('atr', None)

        if atr is None:
            return False

        # Calculate maximum favorable price movement
        if side == 'buy':
            # For long positions, track highest price
            max_favorable_price = position.get('max_price', entry_price)

            if current_price > max_favorable_price:
                position['max_price'] = current_price
                max_favorable_price = current_price

            # Trailing stop: if price drops by 2x ATR from peak
            trailing_stop = max_favorable_price - (2 * atr)

            if current_price <= trailing_stop:
                # Only trigger if we're in profit
                if max_favorable_price > entry_price * 1.01:  # At least 1% profit
                    return True

        else:
            # For short positions, track lowest price
            min_favorable_price = position.get('min_price', entry_price)

            if current_price < min_favorable_price:
                position['min_price'] = current_price
                min_favorable_price = current_price

            # Trailing stop: if price rises by 2x ATR from lowest
            trailing_stop = min_favorable_price + (2 * atr)

            if current_price >= trailing_stop:
                # Only trigger if we're in profit
                if min_favorable_price < entry_price * 0.99:  # At least 1% profit
                    return True

        return False

    def _get_current_market_data(self) -> Optional[Dict]:
        """
        Get current market data with all features calculated

        Returns:
            Dictionary with current price, ATR, features DataFrame, etc.
        """
        try:
            # Fetch recent candles
            candles = self.api.get_candles(
                symbol=Config.SYMBOL,
                resolution=Config.TIMEFRAME,
                count=Config.CANDLES_TO_FETCH
            )

            if not candles:
                return None

            # Prepare DataFrame
            df = self.feature_engine.prepare_data(candles)

            # Calculate all features
            df_with_features = self.feature_engine.calculate_all_features(df)

            # Get current values
            current_price = df['close'].iloc[-1]
            current_atr = df_with_features['atr_14'].iloc[-1] if 'atr_14' in df_with_features.columns else None
            current_volatility = df_with_features['volatility_ratio'].iloc[-1] if 'volatility_ratio' in df_with_features.columns else 0

            return {
                'current_price': current_price,
                'atr': current_atr,
                'volatility': current_volatility,
                'df_with_features': df_with_features,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error fetching market data: {e}")
            return None

    def get_position_status(self, position_id: str) -> Optional[Dict]:
        """
        Get detailed status of a position including current PnL

        Returns:
            Position status dictionary
        """
        try:
            active_positions = self.order_executor.get_active_positions()

            for position in active_positions:
                if position['order_id'] == position_id:
                    # Get current price
                    current_price = self.api.get_current_price(Config.SYMBOL)

                    # Calculate unrealized PnL
                    if position['side'] == 'buy':
                        price_diff = current_price - position['entry_price']
                    else:
                        price_diff = position['entry_price'] - current_price

                    unrealized_pnl = price_diff * position['size']

                    # Calculate ROI
                    margin = (position['entry_price'] * position['size']) / position['leverage']
                    roi = (unrealized_pnl / margin) * 100 if margin > 0 else 0

                    return {
                        'position_id': position_id,
                        'status': 'open',
                        'entry_price': position['entry_price'],
                        'current_price': current_price,
                        'unrealized_pnl': unrealized_pnl,
                        'roi_percentage': roi,
                        'size': position['size'],
                        'leverage': position['leverage'],
                        'stop_loss': position['stop_loss'],
                        'take_profit_levels': position['take_profit_levels']
                    }

            return None

        except Exception as e:
            logger.error(f"Error getting position status: {e}")
            return None

    def cleanup_closed_positions(self):
        """Remove signal history for closed positions"""
        active_position_ids = [p['order_id'] for p in self.order_executor.get_active_positions()]

        for position_id in list(self.position_signals.keys()):
            if position_id not in active_position_ids:
                del self.position_signals[position_id]
