"""
MT5 Broker Adapter
Provides a broker interface that communicates with MT5 via the Bridge API
This replaces direct MT5 Python library calls with HTTP API calls to the bridge
"""

import logging
import time
import uuid
import requests
from typing import Dict, List, Optional
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class TradeDirection(Enum):
    LONG = 1
    SHORT = -1
    NEUTRAL = 0


class MT5BrokerAdapter:
    """
    Broker adapter that communicates with MT5 via the Bridge API
    The Bridge API is polled by an Expert Advisor running inside MT5
    """
    
    def __init__(self, api_url: str = "http://localhost:5000"):
        self.api_url = api_url.rstrip('/')
        self.connected = False
        self.account_info: Dict = {}
        self._last_account_update: Optional[datetime] = None
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 5
        
    def connect(self) -> bool:
        """Check connection to MT5 via bridge"""
        try:
            response = requests.get(f"{self.api_url}/api/status", timeout=10)
            if response.status_code == 200:
                data = response.json()
                self.connected = data.get('mt5_connected', False)
                
                if self.connected:
                    self.account_info = self._get_account_info()
                    logger.info(f"Connected to MT5 via bridge. Balance: ${self.account_info.get('balance', 0):.2f}")
                else:
                    logger.warning("Bridge API is running but MT5 EA is not connected yet")
                    logger.info("Please start the TradingBridgeEA in your MT5 terminal")
                
                return self.connected
            else:
                logger.error(f"Bridge API returned status {response.status_code}")
                return False
                
        except requests.exceptions.ConnectionError:
            logger.error(f"Cannot connect to Bridge API at {self.api_url}")
            logger.info("Make sure the API server is running: python -m mt5_bridge.api_server")
            return False
        except Exception as e:
            logger.error(f"Connection error: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from bridge"""
        self.connected = False
        logger.info("Disconnected from MT5 bridge")
    
    def reconnect(self) -> bool:
        """Attempt to reconnect"""
        if self._reconnect_attempts >= self._max_reconnect_attempts:
            logger.error("Max reconnection attempts reached")
            return False
        
        self._reconnect_attempts += 1
        logger.info(f"Reconnection attempt {self._reconnect_attempts}")
        
        time.sleep(5)
        result = self.connect()
        
        if result:
            self._reconnect_attempts = 0
        
        return result
    
    def _get_account_info(self) -> Dict:
        """Get account information from bridge"""
        try:
            response = requests.get(f"{self.api_url}/api/account", timeout=10)
            if response.status_code == 200:
                data = response.json()
                self._last_account_update = datetime.now()
                return data.get('account', {})
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
        
        return {}
    
    def get_balance(self) -> float:
        """Get current account balance from MT5"""
        info = self._get_account_info()
        return info.get('balance', 0.0)
    
    def get_equity(self) -> float:
        """Get current account equity from MT5"""
        info = self._get_account_info()
        return info.get('equity', 0.0)
    
    def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """Get symbol information (uses defaults since EA doesn't send this)"""
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
        """
        Place a market order via the bridge
        Returns the ticket number if successful, None otherwise
        """
        if not self.is_connected():
            logger.error("Not connected to MT5 - cannot place order")
            return None
        
        try:
            # Use .name comparison to handle cross-module enum types
            direction_name = getattr(direction, 'name', str(direction))
            action = "buy" if direction_name == "LONG" else "sell"
            signal_id = str(uuid.uuid4())
            
            # Get current price for sanity checking SL/TP
            quote, _ = self.get_market_data(symbol)
            current_price = quote.get('bid', 0) if quote else 0
            
            # SANITY CHECK: Validate SL/TP are reasonable price levels
            # SL/TP should be within 10% of current price for Forex
            if current_price > 0:
                max_distance = current_price * 0.10  # 10% max distance
                
                # Check if SL is valid
                if sl > 0 and abs(sl - current_price) > max_distance:
                    logger.error(f"Invalid SL: {sl} is too far from current price {current_price}")
                    # Calculate a reasonable SL based on direction
                    is_jpy = 'JPY' in symbol
                    default_sl_pips = 50
                    pip_size = 0.01 if is_jpy else 0.0001
                    if action == "buy":
                        sl = current_price - (default_sl_pips * pip_size)
                    else:
                        sl = current_price + (default_sl_pips * pip_size)
                    logger.info(f"Using corrected SL: {sl}")
                
                # Check if TP is valid
                if tp > 0 and abs(tp - current_price) > max_distance:
                    logger.error(f"Invalid TP: {tp} is too far from current price {current_price}")
                    # Calculate a reasonable TP based on direction
                    is_jpy = 'JPY' in symbol
                    default_tp_pips = 100
                    pip_size = 0.01 if is_jpy else 0.0001
                    if action == "buy":
                        tp = current_price + (default_tp_pips * pip_size)
                    else:
                        tp = current_price - (default_tp_pips * pip_size)
                    logger.info(f"Using corrected TP: {tp}")
            
            payload = {
                'id': signal_id,
                'symbol': symbol,
                'action': action,
                'volume': volume,
                'sl': sl,
                'tp': tp,
                'comment': comment[:31] if comment else "AI_Trade"
            }
            
            response = requests.post(
                f"{self.api_url}/api/signals",
                json=payload,
                timeout=10
            )
            
            if response.status_code != 200:
                logger.error(f"Failed to send signal: {response.status_code}")
                return None
            
            logger.info(f"Signal sent: {symbol} {action} {volume} lots, waiting for execution...")
            
            # Wait for execution result
            result = self._wait_for_signal_result(signal_id, timeout=30)
            
            if result and result.get('success'):
                ticket = result.get('ticket', 0)
                logger.info(f"Order executed: {symbol} {action} {volume} lots, ticket: {ticket}")
                return ticket
            else:
                error = result.get('error', 'Unknown error') if result else 'Timeout'
                logger.error(f"Order failed: {error}")
                return None
                
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return None
    
    def _wait_for_signal_result(self, signal_id: str, timeout: int = 30) -> Optional[Dict]:
        """Wait for signal execution result from EA"""
        start = time.time()
        
        while (time.time() - start) < timeout:
            try:
                # Check the signal result endpoint
                response = requests.get(
                    f"{self.api_url}/api/signal_status/{signal_id}", 
                    timeout=5
                )
                if response.status_code == 200:
                    data = response.json()
                    if data.get('executed'):
                        result = data.get('result', {})
                        return result
            except Exception as e:
                logger.debug(f"Waiting for signal result: {e}")
            
            time.sleep(1)
        
        # Timeout - signal was not executed by EA
        logger.warning(f"Signal {signal_id} timed out - EA did not execute within {timeout}s")
        return None
    
    def modify_position(self, ticket: int, sl: float = None, tp: float = None) -> bool:
        """Modify position SL/TP"""
        if not self.is_connected():
            logger.error("Not connected to MT5 - cannot modify position")
            return False
        
        try:
            signal_id = str(uuid.uuid4())
            
            payload = {
                'id': signal_id,
                'symbol': '',  # Not needed for modify
                'action': 'modify',
                'volume': 0,
                'sl': sl or 0,
                'tp': tp or 0,
                'ticket': ticket
            }
            
            response = requests.post(
                f"{self.api_url}/api/signals",
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info(f"Modify signal sent for position {ticket}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error modifying position: {e}")
            return False
    
    def close_position(self, ticket: int, volume: float = None) -> bool:
        """Close a position"""
        if not self.is_connected():
            logger.error("Not connected to MT5 - cannot close position")
            return False
        
        try:
            signal_id = str(uuid.uuid4())
            
            payload = {
                'id': signal_id,
                'symbol': '',  # Not needed for close
                'action': 'close',
                'volume': volume or 0,
                'sl': 0,
                'tp': 0,
                'ticket': ticket
            }
            
            response = requests.post(
                f"{self.api_url}/api/signals",
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info(f"Close signal sent for position {ticket}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return False
    
    def get_open_positions(self) -> List[Dict]:
        """Get all open positions from MT5"""
        try:
            response = requests.get(f"{self.api_url}/api/positions", timeout=10)
            if response.status_code == 200:
                data = response.json()
                return data.get('positions', [])
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
        
        return []
    
    def get_current_price(self, symbol: str) -> Optional[Dict]:
        """Get current bid/ask price from real market data"""
        quote, _ = self.get_market_data(symbol)
        if quote:
            return {'bid': quote.get('bid', 0), 'ask': quote.get('ask', 0)}
        # Fallback to defaults if no market data
        info = self.get_symbol_info(symbol)
        return {'bid': info['bid'], 'ask': info['ask']}
    
    def get_market_data(self, symbol: str, count: int = 200) -> tuple:
        """Get real market data (quote + OHLC bars) from MT5 via bridge"""
        try:
            response = requests.get(
                f"{self.api_url}/api/market_data",
                params={'symbol': symbol, 'count': count},
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                quote = data.get('quote', {})
                bars = data.get('bars', [])
                return quote, bars
        except Exception as e:
            logger.debug(f"Error getting market data for {symbol}: {e}")
        
        return None, []
    
    def get_spread(self, symbol: str) -> float:
        """Get current spread for a symbol in points"""
        quote, _ = self.get_market_data(symbol)
        return quote.get('spread', 0.0) if quote else 0.0
    
    def has_market_data(self, symbol: str) -> bool:
        """Check if we have real market data for a symbol"""
        quote, bars = self.get_market_data(symbol)
        return quote is not None and len(bars) > 0
    
    def get_closed_trades(self, count: int = 50) -> List[Dict]:
        """Get closed trades for learning feedback"""
        try:
            response = requests.get(
                f"{self.api_url}/api/closed_trades",
                params={'count': count},
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                return data.get('trades', [])
        except Exception as e:
            logger.debug(f"Error getting closed trades: {e}")
        
        return []
    
    def get_trade_stats(self) -> Dict:
        """Get trading statistics from closed trades"""
        try:
            response = requests.get(
                f"{self.api_url}/api/closed_trades",
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                return data.get('stats', {})
        except Exception as e:
            logger.debug(f"Error getting trade stats: {e}")
        
        return {}
    
    def is_connected(self) -> bool:
        """Check if connected to MT5 via bridge"""
        try:
            response = requests.get(f"{self.api_url}/api/status", timeout=5)
            if response.status_code == 200:
                data = response.json()
                self.connected = data.get('mt5_connected', False)
                return self.connected
        except:
            pass
        
        self.connected = False
        return False


# Convenience function to create adapter
def create_mt5_adapter(api_url: str = None) -> MT5BrokerAdapter:
    """Create an MT5 broker adapter"""
    url = api_url or "http://localhost:5000"
    return MT5BrokerAdapter(api_url=url)
