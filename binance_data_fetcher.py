"""
Binance Public API Data Fetcher (No Authentication Required)
"""
import requests
import pandas as pd
import time
from typing import List, Dict, Optional
from datetime import datetime
from logger_config import logger

class BinanceDataFetcher:
    """Fetch cryptocurrency data from Binance public API"""

    def __init__(self):
        self.base_url = "https://api.binance.com"
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'Mozilla/5.0'
        })

    def get_klines(self, symbol: str = "BTCUSDT", interval: str = "5m",
                   limit: int = 500) -> List[List]:
        """
        Fetch candlestick/kline data from Binance public API

        Args:
            symbol: Trading pair (e.g., BTCUSDT, ETHUSDT)
            interval: Kline interval (1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M)
            limit: Number of candles to fetch (max 1000)

        Returns:
            List of klines: [
                [
                    open_time,
                    open,
                    high,
                    low,
                    close,
                    volume,
                    close_time,
                    quote_asset_volume,
                    number_of_trades,
                    taker_buy_base_asset_volume,
                    taker_buy_quote_asset_volume,
                    ignore
                ],
                ...
            ]
        """
        endpoint = f"{self.base_url}/api/v3/klines"

        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": min(limit, 1000)  # Binance max is 1000
        }

        try:
            response = self.session.get(endpoint, params=params, timeout=10)
            response.raise_for_status()

            klines = response.json()
            logger.info(f"✓ Fetched {len(klines)} candles for {symbol} ({interval})")
            return klines

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch klines: {e}")
            raise

    def get_current_price(self, symbol: str = "BTCUSDT") -> float:
        """Get current price for a symbol"""
        endpoint = f"{self.base_url}/api/v3/ticker/price"

        params = {"symbol": symbol}

        try:
            response = self.session.get(endpoint, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()
            price = float(data['price'])
            logger.info(f"Current {symbol} price: ${price:,.2f}")
            return price

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch current price: {e}")
            raise

    def get_24h_ticker(self, symbol: str = "BTCUSDT") -> Dict:
        """Get 24-hour price change statistics"""
        endpoint = f"{self.base_url}/api/v3/ticker/24hr"

        params = {"symbol": symbol}

        try:
            response = self.session.get(endpoint, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()
            logger.info(f"24h stats for {symbol}: Price change: {data['priceChangePercent']}%")
            return data

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch 24h ticker: {e}")
            raise

    def klines_to_dataframe(self, klines: List[List]) -> pd.DataFrame:
        """
        Convert Binance klines to pandas DataFrame with proper formatting

        Args:
            klines: Raw klines data from Binance API

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        if not klines:
            raise ValueError("No klines data provided")

        # Create DataFrame from klines
        df = pd.DataFrame(klines, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])

        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')

        # Convert price and volume columns to float64 (required for TA-Lib)
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('float64')

        # Select and reorder columns
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

        # Drop any rows with NaN values
        df = df.dropna()

        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)

        logger.info(f"✓ Converted {len(df)} candles to DataFrame")
        logger.info(f"  Date range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
        logger.info(f"  Price range: ${df['close'].min():,.2f} - ${df['close'].max():,.2f}")

        return df

    def get_exchange_info(self, symbol: str = "BTCUSDT") -> Dict:
        """Get trading rules and symbol information"""
        endpoint = f"{self.base_url}/api/v3/exchangeInfo"

        params = {"symbol": symbol}

        try:
            response = self.session.get(endpoint, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()
            if 'symbols' in data and len(data['symbols']) > 0:
                return data['symbols'][0]
            return {}

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch exchange info: {e}")
            raise

    def test_connection(self) -> bool:
        """Test connection to Binance API"""
        endpoint = f"{self.base_url}/api/v3/ping"

        try:
            response = self.session.get(endpoint, timeout=5)
            response.raise_for_status()
            logger.info("✓ Binance API connection successful")
            return True

        except requests.exceptions.RequestException as e:
            logger.error(f"✗ Binance API connection failed: {e}")
            return False


if __name__ == "__main__":
    # Test the data fetcher
    fetcher = BinanceDataFetcher()

    print("=" * 70)
    print("  Testing Binance Data Fetcher")
    print("=" * 70)

    # Test connection
    print("\n1. Testing connection...")
    if fetcher.test_connection():
        print("   ✓ Connection successful")

    # Get current price
    print("\n2. Getting current BTC price...")
    try:
        price = fetcher.get_current_price("BTCUSDT")
        print(f"   ✓ Current BTC price: ${price:,.2f}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")

    # Get 24h ticker
    print("\n3. Getting 24h statistics...")
    try:
        ticker = fetcher.get_24h_ticker("BTCUSDT")
        print(f"   ✓ 24h change: {ticker['priceChangePercent']}%")
        print(f"   ✓ 24h volume: {float(ticker['volume']):,.2f} BTC")
    except Exception as e:
        print(f"   ✗ Failed: {e}")

    # Get klines
    print("\n4. Fetching 100 5-minute candles...")
    try:
        klines = fetcher.get_klines("BTCUSDT", "5m", 100)
        print(f"   ✓ Fetched {len(klines)} candles")
    except Exception as e:
        print(f"   ✗ Failed: {e}")

    # Convert to DataFrame
    print("\n5. Converting to DataFrame...")
    try:
        df = fetcher.klines_to_dataframe(klines)
        print(f"   ✓ DataFrame created with {len(df)} rows")
        print(f"\n   Latest candle:")
        print(f"   Time:   {df['timestamp'].iloc[-1]}")
        print(f"   Open:   ${df['open'].iloc[-1]:,.2f}")
        print(f"   High:   ${df['high'].iloc[-1]:,.2f}")
        print(f"   Low:    ${df['low'].iloc[-1]:,.2f}")
        print(f"   Close:  ${df['close'].iloc[-1]:,.2f}")
        print(f"   Volume: {df['volume'].iloc[-1]:,.4f} BTC")
    except Exception as e:
        print(f"   ✗ Failed: {e}")

    print("\n" + "=" * 70)
    print("  All tests completed!")
    print("=" * 70)
