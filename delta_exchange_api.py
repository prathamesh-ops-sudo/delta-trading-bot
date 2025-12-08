"""
Delta Exchange API Integration Layer
Handles all communication with Delta Exchange India API
"""
import hmac
import hashlib
import time
import requests
from typing import Dict, List, Optional, Any
import json
from datetime import datetime
import logging

from config import Config

logger = logging.getLogger(__name__)


class DeltaExchangeAPI:
    """Delta Exchange API client with authentication and error handling"""

    def __init__(self):
        self.api_key = Config.DELTA_API_KEY
        self.api_secret = Config.DELTA_API_SECRET
        self.base_url = Config.DELTA_BASE_URL
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'api-key': self.api_key
        })

    def _generate_signature(self, method: str, endpoint: str, payload: str = "") -> Dict[str, str]:
        """Generate HMAC signature for authenticated requests"""
        timestamp = str(int(time.time()))
        signature_data = method + timestamp + endpoint + payload

        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            signature_data.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

        return {
            'api-key': self.api_key,
            'timestamp': timestamp,
            'signature': signature
        }

    def _make_request(self, method: str, endpoint: str, params: Optional[Dict] = None,
                     data: Optional[Dict] = None, authenticated: bool = True) -> Dict:
        """Make HTTP request with error handling"""
        url = f"{self.base_url}{endpoint}"
        payload = json.dumps(data) if data else ""

        headers = self.session.headers.copy()
        if authenticated:
            headers.update(self._generate_signature(method, endpoint, payload))

        try:
            if method == "GET":
                response = self.session.get(url, params=params, headers=headers, timeout=10)
            elif method == "POST":
                response = self.session.post(url, data=payload, headers=headers, timeout=10)
            elif method == "DELETE":
                response = self.session.delete(url, headers=headers, timeout=10)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {method} {endpoint} - {str(e)}")
            raise

    # Market Data Methods

    def get_product(self, symbol: str) -> Dict:
        """Get product details for a symbol"""
        endpoint = f"/v2/products/{symbol}"
        return self._make_request("GET", endpoint, authenticated=False)

    def get_orderbook(self, symbol: str) -> Dict:
        """Get current orderbook for a symbol"""
        endpoint = f"/v2/l2orderbook/{symbol}"
        return self._make_request("GET", endpoint, authenticated=False)

    def get_ticker(self, symbol: str) -> Dict:
        """Get current ticker data"""
        endpoint = f"/v2/tickers/{symbol}"
        return self._make_request("GET", endpoint, authenticated=False)

    def get_candles(self, symbol: str, resolution: str, count: int = 500,
                    start: Optional[int] = None, end: Optional[int] = None) -> List[Dict]:
        """
        Get historical candlestick data

        Args:
            symbol: Trading pair symbol (e.g., "BTCUSD")
            resolution: Candle resolution (1m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 1d, 1w, 1M)
            count: Number of candles to fetch (max 2000)
            start: Start timestamp in seconds (optional)
            end: End timestamp in seconds (optional)
        """
        import time

        endpoint = "/v2/history/candles"

        # Resolution to seconds mapping
        resolution_seconds = {
            "1m": 60, "5m": 300, "15m": 900, "30m": 1800,
            "1h": 3600, "2h": 7200, "4h": 14400, "6h": 21600,
            "1d": 86400, "1w": 604800, "1M": 2592000
        }

        # Delta Exchange REQUIRES start and end timestamps (count is not supported)
        if start is None or end is None:
            # Calculate from count
            seconds = resolution_seconds.get(resolution, 300)
            end_time = int(time.time())
            start_time = end_time - (count * seconds)
        else:
            start_time = start
            end_time = end

        params = {
            "symbol": symbol,
            "resolution": resolution,
            "start": start_time,
            "end": end_time
        }

        response = self._make_request("GET", endpoint, params=params, authenticated=False)
        return response.get("result", [])

    # Account Methods

    def get_wallet_balance(self) -> Dict:
        """Get wallet balance and margin information"""
        endpoint = "/v2/wallet/balances"
        return self._make_request("GET", endpoint)

    def get_positions(self, symbol: Optional[str] = None) -> List[Dict]:
        """Get current open positions"""
        endpoint = "/v2/positions"
        params = {"product_id": symbol} if symbol else {}
        response = self._make_request("GET", endpoint, params=params)
        return response.get("result", [])

    # Order Methods

    def place_order(self, symbol: str, side: str, order_type: str, size: float,
                   limit_price: Optional[float] = None, stop_price: Optional[float] = None,
                   reduce_only: bool = False, post_only: bool = False,
                   time_in_force: str = "gtc") -> Dict:
        """
        Place a new order

        Args:
            symbol: Trading pair symbol
            side: 'buy' or 'sell'
            order_type: 'limit_order' or 'market_order'
            size: Order size in contracts
            limit_price: Limit price (required for limit orders)
            stop_price: Stop price (for stop orders)
            reduce_only: Only reduce position
            post_only: Post-only order (maker only)
            time_in_force: 'gtc', 'ioc', or 'fok'
        """
        endpoint = "/v2/orders"

        data = {
            "product_id": symbol,
            "size": size,
            "side": side,
            "order_type": order_type,
            "time_in_force": time_in_force,
            "reduce_only": reduce_only,
            "post_only": post_only
        }

        if limit_price:
            data["limit_price"] = str(limit_price)
        if stop_price:
            data["stop_price"] = str(stop_price)

        logger.info(f"Placing {side} {order_type} order: {size} contracts @ {limit_price}")
        return self._make_request("POST", endpoint, data=data)

    def cancel_order(self, order_id: str, symbol: str) -> Dict:
        """Cancel an existing order"""
        endpoint = f"/v2/orders/{order_id}"
        params = {"product_id": symbol}
        logger.info(f"Cancelling order {order_id}")
        return self._make_request("DELETE", endpoint, params=params)

    def get_orders(self, symbol: Optional[str] = None, state: Optional[str] = None) -> List[Dict]:
        """
        Get orders

        Args:
            symbol: Filter by symbol
            state: Filter by state (open, closed, cancelled)
        """
        endpoint = "/v2/orders"
        params = {}
        if symbol:
            params["product_id"] = symbol
        if state:
            params["state"] = state

        response = self._make_request("GET", endpoint, params=params)
        return response.get("result", [])

    def get_order_history(self, symbol: Optional[str] = None, page_size: int = 100) -> List[Dict]:
        """Get order history"""
        endpoint = "/v2/orders/history"
        params = {"page_size": page_size}
        if symbol:
            params["product_id"] = symbol

        response = self._make_request("GET", endpoint, params=params)
        return response.get("result", [])

    # Trading Methods

    def close_position(self, symbol: str, size: Optional[float] = None) -> Dict:
        """
        Close an open position

        Args:
            symbol: Trading pair symbol
            size: Size to close (None = close entire position)
        """
        positions = self.get_positions(symbol)

        if not positions:
            logger.warning(f"No open position found for {symbol}")
            return {"status": "no_position"}

        position = positions[0]
        position_size = float(position.get("size", 0))

        if position_size == 0:
            return {"status": "no_position"}

        # Determine close direction
        close_side = "sell" if position_size > 0 else "buy"
        close_size = abs(size) if size else abs(position_size)

        logger.info(f"Closing position: {close_side} {close_size} contracts")

        return self.place_order(
            symbol=symbol,
            side=close_side,
            order_type="market_order",
            size=close_size,
            reduce_only=True
        )

    def get_current_price(self, symbol: str) -> float:
        """Get current market price"""
        ticker = self.get_ticker(symbol)
        return float(ticker.get("result", {}).get("mark_price", 0))

    def calculate_pnl(self, symbol: str, entry_price: float, current_price: float,
                     size: float, side: str) -> Dict[str, float]:
        """
        Calculate profit/loss for a position

        Returns:
            Dictionary with pnl, pnl_percentage, fees
        """
        if side == "buy":
            pnl = (current_price - entry_price) * size
        else:
            pnl = (entry_price - current_price) * size

        # Calculate fees (entry + exit)
        entry_fee = entry_price * size * Config.TAKER_FEE
        exit_fee = current_price * size * Config.TAKER_FEE
        total_fees = entry_fee + exit_fee

        net_pnl = pnl - total_fees
        pnl_percentage = (net_pnl / (entry_price * size)) * 100

        return {
            "gross_pnl": pnl,
            "fees": total_fees,
            "net_pnl": net_pnl,
            "pnl_percentage": pnl_percentage
        }

    def get_account_balance(self) -> float:
        """Get available account balance in USD"""
        try:
            wallet = self.get_wallet_balance()
            balances = wallet.get("result", [])

            for balance in balances:
                if balance.get("asset_symbol") == "USDT" or balance.get("asset_symbol") == "USD":
                    return float(balance.get("available_balance", 0))

            return 0.0
        except Exception as e:
            logger.error(f"Error fetching account balance: {e}")
            return 0.0

    def get_leverage_info(self, symbol: str) -> Dict:
        """Get leverage information for a symbol"""
        product = self.get_product(symbol)
        return {
            "max_leverage": product.get("result", {}).get("max_leverage_notional", 20),
            "initial_margin": product.get("result", {}).get("initial_margin", 0.05),
            "maintenance_margin": product.get("result", {}).get("maintenance_margin", 0.025)
        }
