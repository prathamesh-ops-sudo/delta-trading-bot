"""
Order Execution Engine
Handles order placement, management, and execution with robust error handling
"""
import logging
from typing import Dict, Optional, List
from datetime import datetime
import time

from delta_exchange_api import DeltaExchangeAPI
from risk_manager import RiskManager
from config import Config

logger = logging.getLogger(__name__)


class OrderExecutor:
    """Manages order execution and position management"""

    def __init__(self, api: DeltaExchangeAPI, risk_manager: RiskManager):
        self.api = api
        self.risk_manager = risk_manager
        self.active_orders = {}
        self.active_positions = {}

    def execute_trade(self, signal: Dict, market_data: Dict) -> Optional[Dict]:
        """
        Execute a trade based on ML signal

        Args:
            signal: ML model signal with confidence
            market_data: Current market data including price, ATR, etc.

        Returns:
            Trade execution result or None if failed
        """
        try:
            current_price = market_data['current_price']
            atr = market_data.get('atr', None)
            signal_type = signal['signal']
            confidence = signal['confidence']

            logger.info(f"Executing {signal_type} signal with confidence {confidence:.2%}")

            # Check if trading is allowed
            can_trade, reason = self.risk_manager.can_trade(datetime.now())
            if not can_trade:
                logger.warning(f"Trade blocked: {reason}")
                return None

            # Get account balance
            account_balance = self.api.get_account_balance()
            if account_balance <= 0:
                logger.error("Insufficient account balance")
                return None

            # Assess risk
            market_volatility = market_data.get('volatility', 0)
            open_positions = len(self.active_positions)
            risk_assessment = self.risk_manager.assess_trade_risk(
                confidence, market_volatility, open_positions
            )

            if not risk_assessment['approved']:
                logger.warning(f"Trade rejected by risk manager: {risk_assessment['risk_factors']}")
                return None

            # Calculate position size
            position_info = self.risk_manager.calculate_position_size(
                account_balance=account_balance,
                current_price=current_price,
                stop_loss_pct=Config.STOP_LOSS_PCT,
                signal_confidence=confidence
            )

            logger.info(f"Position size: {position_info['contract_size']} contracts, "
                       f"Leverage: {position_info['leverage']}x, "
                       f"Risk: ${position_info['risk_amount']} ({position_info['risk_percentage']}%)")

            # Determine trade side
            side = 'buy' if signal_type == 'BUY' else 'sell'

            # Calculate stop loss
            stop_loss = self.risk_manager.calculate_stop_loss(
                entry_price=current_price,
                side=side,
                atr=atr
            )

            # Calculate take profit levels
            take_profit_levels = self.risk_manager.calculate_take_profit_levels(
                entry_price=current_price,
                side=side
            )

            # Place main order
            order_result = self._place_market_order(
                side=side,
                size=position_info['contract_size']
            )

            if not order_result or 'error' in order_result:
                logger.error(f"Order placement failed: {order_result}")
                return None

            # Extract order info
            order_id = order_result.get('result', {}).get('id')
            actual_entry_price = float(order_result.get('result', {}).get('average_fill_price', current_price))

            # Place stop loss order
            stop_loss_order = self._place_stop_order(
                side='sell' if side == 'buy' else 'buy',
                size=position_info['contract_size'],
                stop_price=stop_loss
            )

            # Place take profit orders
            tp_orders = []
            for tp in take_profit_levels:
                tp_size = position_info['contract_size'] * (tp['size_percentage'] / 100)
                tp_order = self._place_limit_order(
                    side='sell' if side == 'buy' else 'buy',
                    size=tp_size,
                    price=tp['price'],
                    reduce_only=True
                )
                if tp_order:
                    tp_orders.append({
                        'order_id': tp_order.get('result', {}).get('id'),
                        'level': tp['level'],
                        'price': tp['price'],
                        'size': tp_size
                    })

            # Create trade record
            trade_record = {
                'timestamp': datetime.now().isoformat(),
                'order_id': order_id,
                'signal': signal_type,
                'side': side,
                'entry_price': actual_entry_price,
                'size': position_info['contract_size'],
                'leverage': position_info['leverage'],
                'stop_loss': stop_loss,
                'stop_loss_order_id': stop_loss_order.get('result', {}).get('id') if stop_loss_order else None,
                'take_profit_levels': take_profit_levels,
                'take_profit_orders': tp_orders,
                'confidence': confidence,
                'risk_assessment': risk_assessment,
                'position_info': position_info,
                'status': 'open'
            }

            # Store active position
            self.active_positions[order_id] = trade_record

            # Record trade
            self.risk_manager.record_trade(trade_record)

            logger.info(f"Trade executed successfully: {side.upper()} {position_info['contract_size']} @ {actual_entry_price}")

            return trade_record

        except Exception as e:
            logger.error(f"Error executing trade: {e}", exc_info=True)
            return None

    def _place_market_order(self, side: str, size: float) -> Optional[Dict]:
        """Place a market order"""
        try:
            result = self.api.place_order(
                symbol=Config.SYMBOL,
                side=side,
                order_type='market_order',
                size=size
            )
            logger.info(f"Market order placed: {side} {size} contracts")
            return result
        except Exception as e:
            logger.error(f"Error placing market order: {e}")
            return None

    def _place_limit_order(self, side: str, size: float, price: float,
                          reduce_only: bool = False) -> Optional[Dict]:
        """Place a limit order"""
        try:
            result = self.api.place_order(
                symbol=Config.SYMBOL,
                side=side,
                order_type='limit_order',
                size=size,
                limit_price=price,
                reduce_only=reduce_only,
                time_in_force='gtc'
            )
            logger.info(f"Limit order placed: {side} {size} @ {price}")
            return result
        except Exception as e:
            logger.error(f"Error placing limit order: {e}")
            return None

    def _place_stop_order(self, side: str, size: float, stop_price: float) -> Optional[Dict]:
        """Place a stop loss order"""
        try:
            result = self.api.place_order(
                symbol=Config.SYMBOL,
                side=side,
                order_type='market_order',
                size=size,
                stop_price=stop_price,
                reduce_only=True
            )
            logger.info(f"Stop loss order placed: {side} {size} @ {stop_price}")
            return result
        except Exception as e:
            logger.error(f"Error placing stop order: {e}")
            return None

    def close_position(self, position_id: str, reason: str = "Manual close") -> bool:
        """
        Close an open position

        Args:
            position_id: Position identifier (order_id)
            reason: Reason for closing

        Returns:
            True if successful
        """
        try:
            if position_id not in self.active_positions:
                logger.warning(f"Position {position_id} not found")
                return False

            position = self.active_positions[position_id]

            logger.info(f"Closing position {position_id}: {reason}")

            # Cancel all related orders (stop loss and take profit)
            if position.get('stop_loss_order_id'):
                self.api.cancel_order(position['stop_loss_order_id'], Config.SYMBOL)

            for tp_order in position.get('take_profit_orders', []):
                if tp_order.get('order_id'):
                    self.api.cancel_order(tp_order['order_id'], Config.SYMBOL)

            # Close position
            close_side = 'sell' if position['side'] == 'buy' else 'buy'
            close_result = self.api.place_order(
                symbol=Config.SYMBOL,
                side=close_side,
                order_type='market_order',
                size=position['size'],
                reduce_only=True
            )

            if close_result:
                exit_price = float(close_result.get('result', {}).get('average_fill_price', 0))

                # Calculate PnL
                trade_metrics = self.risk_manager.calculate_trade_metrics(
                    entry_price=position['entry_price'],
                    exit_price=exit_price,
                    size=position['size'],
                    side=position['side'],
                    leverage=position['leverage']
                )

                # Update position record
                position['exit_price'] = exit_price
                position['exit_timestamp'] = datetime.now().isoformat()
                position['pnl'] = trade_metrics['net_pnl']
                position['roi'] = trade_metrics['roi_percentage']
                position['close_reason'] = reason
                position['status'] = 'closed'

                logger.info(f"Position closed: PnL ${trade_metrics['net_pnl']:.2f} ({trade_metrics['roi_percentage']:.2f}%)")

                # Remove from active positions
                del self.active_positions[position_id]

                return True

        except Exception as e:
            logger.error(f"Error closing position: {e}", exc_info=True)
            return False

    def get_active_positions(self) -> List[Dict]:
        """Get all active positions"""
        return list(self.active_positions.values())

    def update_position_status(self, position_id: str) -> Optional[Dict]:
        """
        Update position status from exchange

        Returns:
            Updated position data
        """
        try:
            if position_id not in self.active_positions:
                return None

            position = self.active_positions[position_id]

            # Get current positions from exchange
            exchange_positions = self.api.get_positions(Config.SYMBOL)

            # Find matching position
            for ex_pos in exchange_positions:
                if str(ex_pos.get('entry_order_id')) == str(position_id):
                    # Update position data
                    position['current_size'] = float(ex_pos.get('size', 0))
                    position['unrealized_pnl'] = float(ex_pos.get('unrealized_pnl', 0))

                    # Check if position was partially closed
                    if position['current_size'] < position['size']:
                        logger.info(f"Position {position_id} partially closed")

                    return position

            # Position not found on exchange - might be closed
            logger.info(f"Position {position_id} not found on exchange - may be closed")
            return None

        except Exception as e:
            logger.error(f"Error updating position status: {e}")
            return None

    def emergency_close_all(self) -> bool:
        """
        Emergency function to close all open positions

        Returns:
            True if all positions closed successfully
        """
        logger.warning("EMERGENCY: Closing all positions")

        success = True
        for position_id in list(self.active_positions.keys()):
            if not self.close_position(position_id, reason="Emergency close"):
                success = False

        return success

    def get_order_status(self, order_id: str) -> Optional[Dict]:
        """Get status of a specific order"""
        try:
            orders = self.api.get_orders(Config.SYMBOL)
            for order in orders:
                if str(order.get('id')) == str(order_id):
                    return order
            return None
        except Exception as e:
            logger.error(f"Error getting order status: {e}")
            return None
