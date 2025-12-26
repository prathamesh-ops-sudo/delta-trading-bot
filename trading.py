"""
Main Trading Engine with MT5 Integration
Handles order execution, position management, and trade lifecycle
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import time
import threading
from queue import Queue
import json

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    print("MetaTrader5 not available - using simulation mode")

from config import config, TRADING_SESSIONS
from data import data_manager, DatabaseManager
from indicators import TechnicalIndicators
from risk_management import risk_manager, Position, DynamicPositionSizer
from regime_detection import regime_manager
from decisions import decision_engine, TradingSignal, TradeDirection
from execution import execution_engine, Order, OrderSide, OrderType, ExecutionAlgo
from agentic import agentic_system

logger = logging.getLogger(__name__)


class TradeStatus(Enum):
    PENDING = "pending"
    OPEN = "open"
    CLOSED = "closed"
    CANCELLED = "cancelled"


@dataclass
class Trade:
    """Trade record"""
    ticket: int
    symbol: str
    direction: TradeDirection
    volume: float
    entry_price: float
    stop_loss: float
    take_profit: float
    open_time: datetime
    close_time: Optional[datetime] = None
    close_price: Optional[float] = None
    profit: float = 0.0
    commission: float = 0.0
    swap: float = 0.0
    status: TradeStatus = TradeStatus.OPEN
    strategy: str = ""
    confidence: float = 0.0
    leverage: int = 100
    trailing_sl: bool = False
    trailing_distance: float = 0.0
    signal: Optional[TradingSignal] = None
    indicators_at_entry: Dict = field(default_factory=dict)
    regime_at_entry: str = ""


class MT5Broker:
    """MetaTrader 5 broker interface"""
    
    def __init__(self):
        self.connected = False
        self.config = config.mt5
        self.account_info = None
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 5
    
    def connect(self) -> bool:
        """Connect to MT5 terminal"""
        if not MT5_AVAILABLE:
            logger.warning("MT5 not available - running in simulation mode")
            return False
        
        try:
            if not mt5.initialize():
                logger.error(f"MT5 initialization failed: {mt5.last_error()}")
                return False
            
            authorized = mt5.login(
                login=self.config.login,
                password=self.config.password,
                server=self.config.server
            )
            
            if not authorized:
                logger.error(f"MT5 login failed: {mt5.last_error()}")
                mt5.shutdown()
                return False
            
            self.connected = True
            self.account_info = self._get_account_info()
            self._reconnect_attempts = 0
            
            logger.info(f"Connected to MT5: {self.config.server}, "
                       f"Balance: {self.account_info.get('balance', 0):.2f}")
            return True
            
        except Exception as e:
            logger.error(f"MT5 connection error: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from MT5"""
        if MT5_AVAILABLE and self.connected:
            mt5.shutdown()
            self.connected = False
            logger.info("Disconnected from MT5")
    
    def reconnect(self) -> bool:
        """Attempt to reconnect"""
        if self._reconnect_attempts >= self._max_reconnect_attempts:
            logger.error("Max reconnection attempts reached")
            return False
        
        self._reconnect_attempts += 1
        logger.info(f"Reconnection attempt {self._reconnect_attempts}")
        
        self.disconnect()
        time.sleep(5)
        return self.connect()
    
    def _get_account_info(self) -> Dict:
        """Get account information"""
        if not self.connected:
            return {}
        
        try:
            info = mt5.account_info()
            if info:
                return {
                    'login': info.login,
                    'balance': info.balance,
                    'equity': info.equity,
                    'margin': info.margin,
                    'free_margin': info.margin_free,
                    'margin_level': info.margin_level,
                    'leverage': info.leverage,
                    'currency': info.currency,
                    'profit': info.profit
                }
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
        
        return {}
    
    def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """Get symbol information"""
        if not self.connected:
            return self._get_default_symbol_info(symbol)
        
        try:
            info = mt5.symbol_info(symbol)
            if info:
                return {
                    'symbol': info.name,
                    'digits': info.digits,
                    'point': info.point,
                    'spread': info.spread,
                    'trade_contract_size': info.trade_contract_size,
                    'volume_min': info.volume_min,
                    'volume_max': info.volume_max,
                    'volume_step': info.volume_step,
                    'bid': info.bid,
                    'ask': info.ask
                }
        except Exception as e:
            logger.error(f"Error getting symbol info: {e}")
        
        return self._get_default_symbol_info(symbol)
    
    def _get_default_symbol_info(self, symbol: str) -> Dict:
        """Get default symbol info for simulation"""
        is_jpy = 'JPY' in symbol
        return {
            'symbol': symbol,
            'digits': 3 if is_jpy else 5,
            'point': 0.001 if is_jpy else 0.00001,
            'spread': 10 if is_jpy else 1,
            'trade_contract_size': 100000,
            'volume_min': 0.01,
            'volume_max': 100,
            'volume_step': 0.01,
            'bid': 150.0 if is_jpy else 1.1000,
            'ask': 150.01 if is_jpy else 1.10010
        }
    
    def place_order(self, symbol: str, direction: TradeDirection, volume: float,
                    sl: float, tp: float, comment: str = "") -> Optional[int]:
        """Place a market order"""
        if not self.connected:
            return self._simulate_order(symbol, direction, volume, sl, tp)
        
        try:
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                logger.error(f"Symbol {symbol} not found")
                return None
            
            if not symbol_info.visible:
                if not mt5.symbol_select(symbol, True):
                    logger.error(f"Failed to select symbol {symbol}")
                    return None
            
            # Get current price
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                logger.error(f"Failed to get tick for {symbol}")
                return None
            
            price = tick.ask if direction == TradeDirection.LONG else tick.bid
            order_type = mt5.ORDER_TYPE_BUY if direction == TradeDirection.LONG else mt5.ORDER_TYPE_SELL
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": order_type,
                "price": price,
                "sl": sl,
                "tp": tp,
                "deviation": 20,
                "magic": 123456,
                "comment": comment[:31] if comment else "AI_Trade",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"Order failed: {result.retcode} - {result.comment}")
                return None
            
            logger.info(f"Order placed: {symbol} {direction.name} {volume} lots, ticket: {result.order}")
            return result.order
            
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return None
    
    def _simulate_order(self, symbol: str, direction: TradeDirection, 
                        volume: float, sl: float, tp: float) -> int:
        """Simulate order for testing"""
        ticket = int(time.time() * 1000) % 1000000000
        logger.info(f"[SIMULATED] Order placed: {symbol} {direction.name} {volume} lots, ticket: {ticket}")
        return ticket
    
    def modify_position(self, ticket: int, sl: float = None, tp: float = None) -> bool:
        """Modify position SL/TP"""
        if not self.connected:
            logger.info(f"[SIMULATED] Position {ticket} modified: SL={sl}, TP={tp}")
            return True
        
        try:
            position = mt5.positions_get(ticket=ticket)
            if not position:
                logger.error(f"Position {ticket} not found")
                return False
            
            pos = position[0]
            
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "symbol": pos.symbol,
                "position": ticket,
                "sl": sl if sl else pos.sl,
                "tp": tp if tp else pos.tp,
            }
            
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"Modify failed: {result.retcode}")
                return False
            
            logger.info(f"Position {ticket} modified: SL={sl}, TP={tp}")
            return True
            
        except Exception as e:
            logger.error(f"Error modifying position: {e}")
            return False
    
    def close_position(self, ticket: int, volume: float = None) -> bool:
        """Close a position"""
        if not self.connected:
            logger.info(f"[SIMULATED] Position {ticket} closed")
            return True
        
        try:
            position = mt5.positions_get(ticket=ticket)
            if not position:
                logger.error(f"Position {ticket} not found")
                return False
            
            pos = position[0]
            close_volume = volume if volume else pos.volume
            
            tick = mt5.symbol_info_tick(pos.symbol)
            if tick is None:
                return False
            
            # Opposite order to close
            if pos.type == mt5.ORDER_TYPE_BUY:
                order_type = mt5.ORDER_TYPE_SELL
                price = tick.bid
            else:
                order_type = mt5.ORDER_TYPE_BUY
                price = tick.ask
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": pos.symbol,
                "volume": close_volume,
                "type": order_type,
                "position": ticket,
                "price": price,
                "deviation": 20,
                "magic": 123456,
                "comment": "AI_Close",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"Close failed: {result.retcode}")
                return False
            
            logger.info(f"Position {ticket} closed")
            return True
            
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return False
    
    def get_open_positions(self) -> List[Dict]:
        """Get all open positions"""
        if not self.connected:
            return []
        
        try:
            positions = mt5.positions_get()
            if positions is None:
                return []
            
            return [{
                'ticket': pos.ticket,
                'symbol': pos.symbol,
                'type': 'buy' if pos.type == mt5.ORDER_TYPE_BUY else 'sell',
                'volume': pos.volume,
                'price_open': pos.price_open,
                'price_current': pos.price_current,
                'sl': pos.sl,
                'tp': pos.tp,
                'profit': pos.profit,
                'swap': pos.swap,
                'time': datetime.fromtimestamp(pos.time)
            } for pos in positions]
            
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []
    
    def get_current_price(self, symbol: str) -> Optional[Dict]:
        """Get current bid/ask price"""
        if not self.connected:
            info = self._get_default_symbol_info(symbol)
            return {'bid': info['bid'], 'ask': info['ask']}
        
        try:
            tick = mt5.symbol_info_tick(symbol)
            if tick:
                return {
                    'bid': tick.bid,
                    'ask': tick.ask,
                    'last': tick.last,
                    'time': datetime.fromtimestamp(tick.time)
                }
        except Exception as e:
            logger.error(f"Error getting price: {e}")
        
        return None


class TradingEngine:
    """Main trading engine coordinating all components"""
    
    def __init__(self):
        self.broker = MT5Broker()
        self.db = DatabaseManager()
        self.position_sizer = DynamicPositionSizer()
        
        # Active trades
        self.active_trades: Dict[int, Trade] = {}
        self.trade_history: List[Trade] = []
        self.pending_signals: Queue = Queue()
        
        # State
        self.is_running = False
        self.is_paused = False
        self.last_trade_time: Dict[str, datetime] = {}
        self.trades_this_hour = 0
        self.hour_start = datetime.now().replace(minute=0, second=0, microsecond=0)
        
        # Threads
        self._trading_thread = None
        self._monitor_thread = None
        
        # Statistics
        self.session_stats = {
            'trades_opened': 0,
            'trades_closed': 0,
            'total_profit': 0.0,
            'winning_trades': 0,
            'losing_trades': 0
        }
    
    def start(self) -> bool:
        """Start the trading engine"""
        logger.info("Starting trading engine...")
        
        # Connect to broker
        if not self.broker.connect():
            logger.warning("Running in simulation mode")
        
        # Initialize regime detection
        self._initialize_regime_detection()
        
        self.is_running = True
        
        # Start trading thread
        self._trading_thread = threading.Thread(target=self._trading_loop, daemon=True)
        self._trading_thread.start()
        
        # Start position monitor thread
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        
        logger.info("Trading engine started")
        return True
    
    def stop(self):
        """Stop the trading engine"""
        logger.info("Stopping trading engine...")
        self.is_running = False
        
        if self._trading_thread:
            self._trading_thread.join(timeout=10)
        if self._monitor_thread:
            self._monitor_thread.join(timeout=10)
        
        self.broker.disconnect()
        logger.info("Trading engine stopped")
    
    def pause(self):
        """Pause trading"""
        self.is_paused = True
        logger.info("Trading paused")
    
    def resume(self):
        """Resume trading"""
        self.is_paused = False
        logger.info("Trading resumed")
    
    def _initialize_regime_detection(self):
        """Initialize regime detection with historical data"""
        try:
            # Get historical data for regime fitting
            df = data_manager.get_ohlcv('EURUSD', 'H1', count=1000)
            if not df.empty:
                regime_manager.fit(df)
                logger.info("Regime detection initialized")
        except Exception as e:
            logger.warning(f"Could not initialize regime detection: {e}")
    
    def _trading_loop(self):
        """Main trading loop"""
        while self.is_running:
            try:
                if self.is_paused:
                    time.sleep(1)
                    continue
                
                # Check if we should trade
                if not self._should_trade():
                    time.sleep(10)
                    continue
                
                # Analyze each symbol
                for symbol in config.trading.symbols:
                    if not self.is_running:
                        break
                    
                    try:
                        self._analyze_and_trade(symbol)
                    except Exception as e:
                        logger.error(f"Error analyzing {symbol}: {e}")
                
                # Wait before next iteration
                time.sleep(config.trading.trade_check_interval)
                
            except Exception as e:
                logger.error(f"Trading loop error: {e}")
                time.sleep(30)
    
    def _monitor_loop(self):
        """Position monitoring loop"""
        while self.is_running:
            try:
                self._update_positions()
                self._check_trailing_stops()
                self._check_exit_conditions()
                
                time.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
                time.sleep(10)
    
    def _should_trade(self) -> bool:
        """Check if trading conditions are met"""
        # Check risk limits
        account_info = self.broker._get_account_info()
        balance = account_info.get('balance', config.trading.initial_balance)
        
        can_trade, reason = risk_manager.check_risk_limits(balance)
        if not can_trade:
            logger.warning(f"Risk limit: {reason}")
            return False
        
        # Check trading mode from agentic system
        params = agentic_system.get_trading_parameters()
        if params.get('trading_mode') == 'halted':
            return False
        
        # Check hourly trade limit
        current_hour = datetime.now().replace(minute=0, second=0, microsecond=0)
        if current_hour != self.hour_start:
            self.hour_start = current_hour
            self.trades_this_hour = 0
        
        if self.trades_this_hour >= config.trading.max_trades_per_hour:
            return False
        
        # Check concurrent positions
        if len(self.active_trades) >= config.trading.max_concurrent_trades:
            return False
        
        return True
    
    def _analyze_and_trade(self, symbol: str):
        """Analyze symbol and execute trade if signal generated"""
        # Check if we recently traded this symbol
        last_trade = self.last_trade_time.get(symbol)
        if last_trade and (datetime.now() - last_trade).seconds < 300:  # 5 min cooldown
            return
        
        # Get multi-timeframe data
        mtf_data = data_manager.get_multi_timeframe_data(symbol)
        if not mtf_data:
            return
        
        # Get account balance
        account_info = self.broker._get_account_info()
        balance = account_info.get('balance', config.trading.initial_balance)
        
        # Generate signal using decision engine
        signal = decision_engine.analyze_market(symbol, mtf_data, balance)
        
        if signal is None:
            return
        
        # Check with agentic system
        should_trade, reason, adj_confidence = agentic_system.should_take_trade({
            'confidence': signal.confidence,
            'strategy': signal.strategy,
            'regime': signal.regime.name if signal.regime else 'unknown'
        })
        
        if not should_trade:
            logger.debug(f"{symbol}: Signal rejected - {reason}")
            return
        
        # Update signal confidence
        signal.confidence = adj_confidence
        
        # Execute trade
        self._execute_signal(signal, balance)
    
    def _execute_signal(self, signal: TradingSignal, balance: float):
        """Execute a trading signal"""
        symbol = signal.symbol
        
        # Get symbol info
        symbol_info = self.broker.get_symbol_info(symbol)
        if not symbol_info:
            logger.error(f"Could not get symbol info for {symbol}")
            return
        
        # Calculate position size
        risk_amount = agentic_system.calculate_position_size(
            {'confidence': signal.confidence, 'strategy': signal.strategy},
            balance
        )
        
        sl_distance = abs(signal.entry_price - signal.stop_loss)
        if sl_distance == 0:
            return
        
        # Calculate lots
        pip_value = symbol_info['point'] * symbol_info['trade_contract_size']
        lots = risk_amount / (sl_distance / symbol_info['point'] * pip_value)
        
        # Round to volume step
        volume_step = symbol_info['volume_step']
        lots = max(symbol_info['volume_min'], 
                   min(symbol_info['volume_max'],
                       round(lots / volume_step) * volume_step))
        
        # Place order
        comment = f"{signal.strategy[:10]}_{signal.confidence:.0%}"
        ticket = self.broker.place_order(
            symbol=symbol,
            direction=signal.direction,
            volume=lots,
            sl=signal.stop_loss,
            tp=signal.take_profit,
            comment=comment
        )
        
        if ticket is None:
            logger.error(f"Failed to place order for {symbol}")
            return
        
        # Create trade record
        trade = Trade(
            ticket=ticket,
            symbol=symbol,
            direction=signal.direction,
            volume=lots,
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            open_time=datetime.now(),
            strategy=signal.strategy,
            confidence=signal.confidence,
            leverage=signal.leverage,
            trailing_sl=signal.trailing_sl_enabled,
            trailing_distance=signal.trailing_sl_distance,
            signal=signal,
            indicators_at_entry={
                'rsi': signal.rsi,
                'adx': signal.adx,
                'atr': signal.atr,
                'macd': signal.macd_signal,
                'bb_position': signal.bb_position
            },
            regime_at_entry=signal.regime.name if signal.regime else 'unknown'
        )
        
        self.active_trades[ticket] = trade
        self.last_trade_time[symbol] = datetime.now()
        self.trades_this_hour += 1
        self.session_stats['trades_opened'] += 1
        
        # Add to risk manager
        risk_manager.add_position(Position(
            symbol=symbol,
            direction=1 if signal.direction == TradeDirection.LONG else -1,
            size=lots,
            entry_price=signal.entry_price,
            current_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            leverage=signal.leverage
        ))
        
        # Save to database
        self.db.save_trade({
            'ticket': ticket,
            'symbol': symbol,
            'type': 'buy' if signal.direction == TradeDirection.LONG else 'sell',
            'volume': lots,
            'open_price': signal.entry_price,
            'sl': signal.stop_loss,
            'tp': signal.take_profit,
            'open_time': datetime.now(),
            'status': 'open',
            'strategy': signal.strategy,
            'ml_confidence': signal.confidence
        })
        
        logger.info(f"Trade opened: {symbol} {signal.direction.name} {lots} lots @ {signal.entry_price:.5f}, "
                   f"SL: {signal.stop_loss:.5f}, TP: {signal.take_profit:.5f}, "
                   f"Strategy: {signal.strategy}, Confidence: {signal.confidence:.1%}")
    
    def _update_positions(self):
        """Update active position information"""
        positions = self.broker.get_open_positions()
        
        for pos in positions:
            ticket = pos['ticket']
            if ticket in self.active_trades:
                trade = self.active_trades[ticket]
                trade.profit = pos['profit']
                trade.swap = pos['swap']
    
    def _check_trailing_stops(self):
        """Check and update trailing stops"""
        for ticket, trade in list(self.active_trades.items()):
            if not trade.trailing_sl:
                continue
            
            # Get current price
            price_info = self.broker.get_current_price(trade.symbol)
            if not price_info:
                continue
            
            current_price = price_info['bid'] if trade.direction == TradeDirection.LONG else price_info['ask']
            
            # Calculate profit in pips
            symbol_info = self.broker.get_symbol_info(trade.symbol)
            pip_value = symbol_info['point']
            
            if trade.direction == TradeDirection.LONG:
                profit_pips = (current_price - trade.entry_price) / pip_value
                new_sl = current_price - trade.trailing_distance
                
                if profit_pips > 10 and new_sl > trade.stop_loss:
                    self.broker.modify_position(ticket, sl=new_sl)
                    trade.stop_loss = new_sl
                    logger.info(f"Trailing SL updated for {ticket}: {new_sl:.5f}")
            else:
                profit_pips = (trade.entry_price - current_price) / pip_value
                new_sl = current_price + trade.trailing_distance
                
                if profit_pips > 10 and new_sl < trade.stop_loss:
                    self.broker.modify_position(ticket, sl=new_sl)
                    trade.stop_loss = new_sl
                    logger.info(f"Trailing SL updated for {ticket}: {new_sl:.5f}")
    
    def _check_exit_conditions(self):
        """Check for early exit conditions"""
        for ticket, trade in list(self.active_trades.items()):
            if trade.signal is None:
                continue
            
            # Get current price
            price_info = self.broker.get_current_price(trade.symbol)
            if not price_info:
                continue
            
            current_price = price_info['bid'] if trade.direction == TradeDirection.LONG else price_info['ask']
            
            # Calculate profit in pips
            symbol_info = self.broker.get_symbol_info(trade.symbol)
            pip_value = symbol_info['point']
            
            if trade.direction == TradeDirection.LONG:
                profit_pips = (current_price - trade.entry_price) / pip_value
            else:
                profit_pips = (trade.entry_price - current_price) / pip_value
            
            # Check exit conditions
            should_exit, reason = decision_engine.should_exit_trade(
                trade.signal, current_price, profit_pips
            )
            
            if should_exit:
                self._close_trade(ticket, reason)
    
    def _close_trade(self, ticket: int, reason: str = ""):
        """Close a trade"""
        if ticket not in self.active_trades:
            return
        
        trade = self.active_trades[ticket]
        
        # Close position
        if not self.broker.close_position(ticket):
            logger.error(f"Failed to close position {ticket}")
            return
        
        # Get final price
        price_info = self.broker.get_current_price(trade.symbol)
        close_price = price_info['bid'] if trade.direction == TradeDirection.LONG else price_info['ask']
        
        # Update trade record
        trade.close_time = datetime.now()
        trade.close_price = close_price
        trade.status = TradeStatus.CLOSED
        
        # Calculate final profit
        if trade.direction == TradeDirection.LONG:
            trade.profit = (close_price - trade.entry_price) * trade.volume * 100000
        else:
            trade.profit = (trade.entry_price - close_price) * trade.volume * 100000
        
        # Update statistics
        self.session_stats['trades_closed'] += 1
        self.session_stats['total_profit'] += trade.profit
        
        if trade.profit > 0:
            self.session_stats['winning_trades'] += 1
        else:
            self.session_stats['losing_trades'] += 1
        
        # Move to history
        self.trade_history.append(trade)
        del self.active_trades[ticket]
        
        # Update risk manager
        risk_manager.close_position(trade.symbol, close_price)
        
        # Update database
        self.db.save_trade({
            'ticket': ticket,
            'symbol': trade.symbol,
            'type': 'buy' if trade.direction == TradeDirection.LONG else 'sell',
            'volume': trade.volume,
            'open_price': trade.entry_price,
            'close_price': close_price,
            'sl': trade.stop_loss,
            'tp': trade.take_profit,
            'profit': trade.profit,
            'open_time': trade.open_time,
            'close_time': trade.close_time,
            'status': 'closed',
            'strategy': trade.strategy,
            'ml_confidence': trade.confidence
        })
        
        # Record for agentic learning
        agentic_system.record_trade_outcome(
            str(ticket),
            trade.profit > 0,
            []  # insights applied
        )
        
        logger.info(f"Trade closed: {trade.symbol} {trade.direction.name}, "
                   f"Profit: ${trade.profit:.2f}, Reason: {reason}")
    
    def close_all_positions(self, reason: str = "Manual close"):
        """Close all open positions"""
        for ticket in list(self.active_trades.keys()):
            self._close_trade(ticket, reason)
    
    def get_session_summary(self) -> Dict:
        """Get current session summary"""
        account_info = self.broker._get_account_info()
        
        win_rate = 0
        if self.session_stats['trades_closed'] > 0:
            win_rate = self.session_stats['winning_trades'] / self.session_stats['trades_closed']
        
        return {
            'balance': account_info.get('balance', 0),
            'equity': account_info.get('equity', 0),
            'open_positions': len(self.active_trades),
            'trades_opened': self.session_stats['trades_opened'],
            'trades_closed': self.session_stats['trades_closed'],
            'total_profit': self.session_stats['total_profit'],
            'win_rate': win_rate,
            'is_running': self.is_running,
            'is_paused': self.is_paused,
            'trading_mode': agentic_system.trading_mode
        }
    
    def get_trades_for_learning(self) -> List[Dict]:
        """Get today's trades for agentic learning"""
        today = datetime.now().date()
        
        trades = []
        for trade in self.trade_history:
            if trade.close_time and trade.close_time.date() == today:
                trades.append({
                    'ticket': trade.ticket,
                    'symbol': trade.symbol,
                    'direction': trade.direction.name.lower(),
                    'entry_price': trade.entry_price,
                    'exit_price': trade.close_price,
                    'profit': trade.profit,
                    'duration_minutes': (trade.close_time - trade.open_time).seconds / 60,
                    'entry_reason': trade.strategy,
                    'exit_reason': 'closed',
                    'confidence': trade.confidence,
                    'indicators': trade.indicators_at_entry
                })
        
        return trades


# Singleton instance
trading_engine = TradingEngine()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Trading Engine...")
    
    # Test broker connection (simulation mode)
    broker = MT5Broker()
    connected = broker.connect()
    print(f"Broker connected: {connected}")
    
    # Test symbol info
    symbol_info = broker.get_symbol_info('EURUSD')
    print(f"EURUSD info: {symbol_info}")
    
    # Test simulated order
    ticket = broker.place_order(
        symbol='EURUSD',
        direction=TradeDirection.LONG,
        volume=0.01,
        sl=1.0900,
        tp=1.1100
    )
    print(f"Order ticket: {ticket}")
    
    # Test trading engine
    engine = TradingEngine()
    
    # Get session summary
    summary = engine.get_session_summary()
    print(f"\nSession summary: {summary}")
    
    print("\nTrading Engine test complete!")
