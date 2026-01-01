"""
Data fetching and storage module for Forex Trading System
Handles MT5 data, Yahoo Finance, news APIs, and database operations
"""

import os
import sqlite3
import logging
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import threading
import time

import numpy as np
import pandas as pd
import requests

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    print("MetaTrader5 not available - will use alternative data sources")

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

from config import config, INDICATOR_SETTINGS, NEWS_IMPACT

logger = logging.getLogger(__name__)


# Timeframe mapping for MT5
MT5_TIMEFRAMES = {
    "M1": mt5.TIMEFRAME_M1 if MT5_AVAILABLE else 1,
    "M5": mt5.TIMEFRAME_M5 if MT5_AVAILABLE else 5,
    "M15": mt5.TIMEFRAME_M15 if MT5_AVAILABLE else 15,
    "M30": mt5.TIMEFRAME_M30 if MT5_AVAILABLE else 30,
    "H1": mt5.TIMEFRAME_H1 if MT5_AVAILABLE else 60,
    "H4": mt5.TIMEFRAME_H4 if MT5_AVAILABLE else 240,
    "D1": mt5.TIMEFRAME_D1 if MT5_AVAILABLE else 1440,
    "W1": mt5.TIMEFRAME_W1 if MT5_AVAILABLE else 10080,
}

# Yahoo Finance symbol mapping
YAHOO_SYMBOLS = {
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
    "USDCHF": "USDCHF=X",
    "AUDUSD": "AUDUSD=X",
    "USDCAD": "USDCAD=X",
    "NZDUSD": "NZDUSD=X",
    "EURGBP": "EURGBP=X",
    "EURJPY": "EURJPY=X",
    "GBPJPY": "GBPJPY=X",
}


@dataclass
class OHLCV:
    """OHLCV data structure"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    symbol: str
    timeframe: str


class DatabaseManager:
    """SQLite database manager for storing forex data"""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or config.data.db_path
        self._ensure_db_dir()
        self._init_database()
        self._lock = threading.Lock()
    
    def _ensure_db_dir(self):
        """Ensure database directory exists"""
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir)
    
    def _init_database(self):
        """Initialize database tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # OHLCV data table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ohlcv (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL DEFAULT 0,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, timeframe, timestamp)
                )
            """)
            
            # Trades table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticket INTEGER UNIQUE,
                    symbol TEXT NOT NULL,
                    type TEXT NOT NULL,
                    volume REAL NOT NULL,
                    open_price REAL NOT NULL,
                    close_price REAL,
                    sl REAL,
                    tp REAL,
                    profit REAL,
                    commission REAL DEFAULT 0,
                    swap REAL DEFAULT 0,
                    open_time DATETIME NOT NULL,
                    close_time DATETIME,
                    status TEXT DEFAULT 'open',
                    strategy TEXT,
                    ml_confidence REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # News events table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS news_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    description TEXT,
                    source TEXT,
                    url TEXT,
                    published_at DATETIME,
                    sentiment_score REAL,
                    impact_level TEXT,
                    currencies TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Model performance table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    version TEXT,
                    accuracy REAL,
                    precision_score REAL,
                    recall REAL,
                    f1_score REAL,
                    sharpe_ratio REAL,
                    max_drawdown REAL,
                    total_trades INTEGER,
                    win_rate REAL,
                    profit_factor REAL,
                    trained_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    notes TEXT
                )
            """)
            
            # Account history table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS account_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    balance REAL NOT NULL,
                    equity REAL NOT NULL,
                    margin REAL,
                    free_margin REAL,
                    margin_level REAL,
                    open_positions INTEGER DEFAULT 0,
                    daily_pnl REAL DEFAULT 0,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_tf ON ohlcv(symbol, timeframe)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_ohlcv_timestamp ON ohlcv(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status)")
            
            conn.commit()
    
    def save_ohlcv(self, data: pd.DataFrame, symbol: str, timeframe: str):
        """Save OHLCV data to database"""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                for _, row in data.iterrows():
                    try:
                        conn.execute("""
                            INSERT OR REPLACE INTO ohlcv 
                            (symbol, timeframe, timestamp, open, high, low, close, volume)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            symbol, timeframe, 
                            row.get('timestamp', row.name) if hasattr(row, 'name') else row['timestamp'],
                            row['open'], row['high'], row['low'], row['close'],
                            row.get('volume', 0)
                        ))
                    except Exception as e:
                        logger.warning(f"Error saving OHLCV row: {e}")
                conn.commit()
    
    def get_ohlcv(self, symbol: str, timeframe: str, 
                  start_date: datetime = None, end_date: datetime = None,
                  limit: int = None) -> pd.DataFrame:
        """Get OHLCV data from database"""
        query = "SELECT timestamp, open, high, low, close, volume FROM ohlcv WHERE symbol = ? AND timeframe = ?"
        params = [symbol, timeframe]
        
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date)
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date)
        
        query += " ORDER BY timestamp DESC"
        
        if limit:
            query += f" LIMIT {limit}"
        
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn, params=params, parse_dates=['timestamp'])
        
        if not df.empty:
            df = df.sort_values('timestamp').reset_index(drop=True)
            df.set_index('timestamp', inplace=True)
        
        return df
    
    def save_trade(self, trade_data: Dict):
        """Save trade to database"""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                columns = ', '.join(trade_data.keys())
                placeholders = ', '.join(['?' for _ in trade_data])
                conn.execute(
                    f"INSERT OR REPLACE INTO trades ({columns}) VALUES ({placeholders})",
                    list(trade_data.values())
                )
                conn.commit()
    
    def get_trades(self, status: str = None, symbol: str = None, 
                   start_date: datetime = None, limit: int = 100) -> pd.DataFrame:
        """Get trades from database"""
        query = "SELECT * FROM trades WHERE 1=1"
        params = []
        
        if status:
            query += " AND status = ?"
            params.append(status)
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        if start_date:
            query += " AND open_time >= ?"
            params.append(start_date)
        
        query += f" ORDER BY open_time DESC LIMIT {limit}"
        
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(query, conn, params=params)
    
    def save_news_event(self, event: Dict):
        """Save news event to database"""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO news_events 
                    (title, description, source, url, published_at, sentiment_score, impact_level, currencies)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    event.get('title'), event.get('description'), event.get('source'),
                    event.get('url'), event.get('published_at'), event.get('sentiment_score'),
                    event.get('impact_level'), json.dumps(event.get('currencies', []))
                ))
                conn.commit()
    
    def save_account_snapshot(self, balance: float, equity: float, 
                              margin: float = 0, free_margin: float = 0,
                              margin_level: float = 0, open_positions: int = 0,
                              daily_pnl: float = 0):
        """Save account snapshot"""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO account_history 
                    (balance, equity, margin, free_margin, margin_level, open_positions, daily_pnl)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (balance, equity, margin, free_margin, margin_level, open_positions, daily_pnl))
                conn.commit()


class MT5DataFetcher:
    """MetaTrader 5 data fetcher"""
    
    def __init__(self):
        self.connected = False
        self.config = config.mt5
    
    def connect(self) -> bool:
        """Connect to MT5 terminal"""
        if not MT5_AVAILABLE:
            logger.warning("MT5 not available")
            return False
        
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
        logger.info(f"Connected to MT5: {self.config.server}")
        return True
    
    def disconnect(self):
        """Disconnect from MT5"""
        if MT5_AVAILABLE and self.connected:
            mt5.shutdown()
            self.connected = False
            logger.info("Disconnected from MT5")
    
    def get_ohlcv(self, symbol: str, timeframe: str, 
                  count: int = 1000, start_date: datetime = None) -> pd.DataFrame:
        """Get OHLCV data from MT5"""
        if not self.connected:
            if not self.connect():
                return pd.DataFrame()
        
        tf = MT5_TIMEFRAMES.get(timeframe, mt5.TIMEFRAME_H1 if MT5_AVAILABLE else 60)
        
        try:
            if start_date:
                rates = mt5.copy_rates_from(symbol, tf, start_date, count)
            else:
                rates = mt5.copy_rates_from_pos(symbol, tf, 0, count)
            
            if rates is None or len(rates) == 0:
                logger.warning(f"No data received for {symbol} {timeframe}")
                return pd.DataFrame()
            
            df = pd.DataFrame(rates)
            df['timestamp'] = pd.to_datetime(df['time'], unit='s')
            df = df.rename(columns={'tick_volume': 'volume'})
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching MT5 data: {e}")
            return pd.DataFrame()
    
    def get_tick(self, symbol: str) -> Optional[Dict]:
        """Get current tick data"""
        if not self.connected:
            if not self.connect():
                return None
        
        try:
            tick = mt5.symbol_info_tick(symbol)
            if tick:
                return {
                    'bid': tick.bid,
                    'ask': tick.ask,
                    'last': tick.last,
                    'volume': tick.volume,
                    'time': datetime.fromtimestamp(tick.time)
                }
        except Exception as e:
            logger.error(f"Error fetching tick: {e}")
        
        return None
    
    def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """Get symbol information"""
        if not self.connected:
            if not self.connect():
                return None
        
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
                    'trade_mode': info.trade_mode,
                    'margin_initial': info.margin_initial,
                }
        except Exception as e:
            logger.error(f"Error fetching symbol info: {e}")
        
        return None
    
    def get_account_info(self) -> Optional[Dict]:
        """Get account information"""
        if not self.connected:
            if not self.connect():
                return None
        
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
                    'server': info.server,
                    'trade_mode': info.trade_mode,
                }
        except Exception as e:
            logger.error(f"Error fetching account info: {e}")
        
        return None


class YahooFinanceDataFetcher:
    """Yahoo Finance data fetcher as backup"""
    
    def __init__(self):
        self.available = YFINANCE_AVAILABLE
    
    def get_ohlcv(self, symbol: str, timeframe: str = "H1",
                  period: str = "1mo", interval: str = None) -> pd.DataFrame:
        """Get OHLCV data from Yahoo Finance"""
        if not self.available:
            logger.warning("yfinance not available")
            return pd.DataFrame()
        
        yahoo_symbol = YAHOO_SYMBOLS.get(symbol, f"{symbol}=X")
        
        # Map timeframe to interval
        interval_map = {
            "M1": "1m", "M5": "5m", "M15": "15m", "M30": "30m",
            "H1": "1h", "H4": "4h", "D1": "1d", "W1": "1wk"
        }
        
        yf_interval = interval or interval_map.get(timeframe, "1h")
        
        try:
            ticker = yf.Ticker(yahoo_symbol)
            df = ticker.history(period=period, interval=yf_interval)
            
            if df.empty:
                return pd.DataFrame()
            
            df = df.rename(columns={
                'Open': 'open', 'High': 'high', 'Low': 'low',
                'Close': 'close', 'Volume': 'volume'
            })
            df.index.name = 'timestamp'
            df = df[['open', 'high', 'low', 'close', 'volume']]
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching Yahoo Finance data: {e}")
            return pd.DataFrame()
    
    def get_historical(self, symbol: str, start_date: str, 
                       end_date: str = None, interval: str = "1d") -> pd.DataFrame:
        """Get historical data"""
        if not self.available:
            return pd.DataFrame()
        
        yahoo_symbol = YAHOO_SYMBOLS.get(symbol, f"{symbol}=X")
        
        try:
            ticker = yf.Ticker(yahoo_symbol)
            df = ticker.history(start=start_date, end=end_date, interval=interval)
            
            if df.empty:
                return pd.DataFrame()
            
            df = df.rename(columns={
                'Open': 'open', 'High': 'high', 'Low': 'low',
                'Close': 'close', 'Volume': 'volume'
            })
            df.index.name = 'timestamp'
            
            return df[['open', 'high', 'low', 'close', 'volume']]
            
        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            return pd.DataFrame()


class NewsDataFetcher:
    """News and sentiment data fetcher"""
    
    def __init__(self):
        self.api_key = config.data.news_api_key
        self.base_url = "https://newsapi.org/v2"
    
    def get_forex_news(self, keywords: List[str] = None, 
                       from_date: datetime = None) -> List[Dict]:
        """Fetch forex-related news"""
        if not self.api_key:
            logger.warning("News API key not configured")
            return []
        
        keywords = keywords or ["forex", "currency", "EUR/USD", "central bank", "interest rate"]
        query = " OR ".join(keywords)
        
        params = {
            "apiKey": self.api_key,
            "q": query,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": 100
        }
        
        if from_date:
            params["from"] = from_date.isoformat()
        
        try:
            response = requests.get(f"{self.base_url}/everything", params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            articles = []
            for article in data.get("articles", []):
                articles.append({
                    "title": article.get("title"),
                    "description": article.get("description"),
                    "source": article.get("source", {}).get("name"),
                    "url": article.get("url"),
                    "published_at": article.get("publishedAt"),
                    "content": article.get("content")
                })
            
            return articles
            
        except Exception as e:
            logger.error(f"Error fetching news: {e}")
            return []
    
    def get_economic_calendar(self) -> List[Dict]:
        """Fetch economic calendar events (using free sources)"""
        # Using Forex Factory or similar free calendar
        # This is a simplified implementation
        events = []
        
        try:
            # Try to fetch from investing.com economic calendar API
            url = "https://www.investing.com/economic-calendar/Service/getCalendarFilteredData"
            headers = {
                "User-Agent": "Mozilla/5.0",
                "X-Requested-With": "XMLHttpRequest"
            }
            
            # This is a placeholder - actual implementation would need proper API
            logger.info("Economic calendar fetch - using cached/mock data")
            
        except Exception as e:
            logger.warning(f"Error fetching economic calendar: {e}")
        
        return events
    
    def analyze_sentiment(self, text: str) -> float:
        """Simple sentiment analysis (returns -1 to 1)"""
        # Simple keyword-based sentiment
        positive_words = [
            "bullish", "growth", "increase", "rise", "gain", "positive",
            "strong", "rally", "surge", "boost", "optimistic", "recovery"
        ]
        negative_words = [
            "bearish", "decline", "decrease", "fall", "loss", "negative",
            "weak", "crash", "drop", "cut", "pessimistic", "recession"
        ]
        
        text_lower = text.lower()
        
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        total = pos_count + neg_count
        if total == 0:
            return 0.0
        
        return (pos_count - neg_count) / total


class MacroDataFetcher:
    """Fetches macro indicators: DXY, VIX, Treasury yields as whale activity proxies"""
    
    def __init__(self):
        self.available = YFINANCE_AVAILABLE
        self._cache = {}
        self._cache_ttl = 300  # 5 minutes cache
        
        # Yahoo Finance symbols for macro indicators
        self.symbols = {
            'DXY': 'DX-Y.NYB',      # US Dollar Index
            'VIX': '^VIX',           # CBOE Volatility Index
            'TNX': '^TNX',           # 10-Year Treasury Yield
            'IRX': '^IRX',           # 13-Week Treasury Bill
            'TYX': '^TYX',           # 30-Year Treasury Yield
            'FVX': '^FVX',           # 5-Year Treasury Yield
        }
    
    def _get_cached(self, key: str) -> Optional[Dict]:
        """Get cached data if still valid"""
        if key in self._cache:
            data, timestamp = self._cache[key]
            if (datetime.now() - timestamp).total_seconds() < self._cache_ttl:
                return data
        return None
    
    def _set_cache(self, key: str, data: Dict):
        """Cache data with timestamp"""
        self._cache[key] = (data, datetime.now())
    
    def get_dxy(self) -> Dict:
        """Get US Dollar Index (DXY) data - measures USD strength"""
        cached = self._get_cached('dxy')
        if cached:
            return cached
        
        if not self.available:
            return {'value': None, 'change': None, 'change_pct': None, 'trend': 'unknown'}
        
        try:
            ticker = yf.Ticker(self.symbols['DXY'])
            hist = ticker.history(period='5d', interval='1h')
            
            if hist.empty:
                return {'value': None, 'change': None, 'change_pct': None, 'trend': 'unknown'}
            
            current = hist['Close'].iloc[-1]
            prev_day = hist['Close'].iloc[-24] if len(hist) > 24 else hist['Close'].iloc[0]
            change = current - prev_day
            change_pct = (change / prev_day) * 100
            
            # Determine trend
            if change_pct > 0.1:
                trend = 'bullish_usd'
            elif change_pct < -0.1:
                trend = 'bearish_usd'
            else:
                trend = 'neutral'
            
            result = {
                'value': round(current, 3),
                'change': round(change, 3),
                'change_pct': round(change_pct, 3),
                'trend': trend,
                'interpretation': 'Strong USD' if trend == 'bullish_usd' else ('Weak USD' if trend == 'bearish_usd' else 'Stable USD')
            }
            self._set_cache('dxy', result)
            return result
            
        except Exception as e:
            logger.error(f"Error fetching DXY: {e}")
            return {'value': None, 'change': None, 'change_pct': None, 'trend': 'unknown'}
    
    def get_vix(self) -> Dict:
        """Get VIX (fear index) - measures market risk sentiment"""
        cached = self._get_cached('vix')
        if cached:
            return cached
        
        if not self.available:
            return {'value': None, 'level': 'unknown', 'risk_sentiment': 'unknown'}
        
        try:
            ticker = yf.Ticker(self.symbols['VIX'])
            hist = ticker.history(period='5d', interval='1h')
            
            if hist.empty:
                return {'value': None, 'level': 'unknown', 'risk_sentiment': 'unknown'}
            
            current = hist['Close'].iloc[-1]
            prev_day = hist['Close'].iloc[-24] if len(hist) > 24 else hist['Close'].iloc[0]
            change = current - prev_day
            
            # VIX levels interpretation
            if current < 12:
                level = 'very_low'
                risk_sentiment = 'extreme_greed'
            elif current < 20:
                level = 'low'
                risk_sentiment = 'risk_on'
            elif current < 30:
                level = 'moderate'
                risk_sentiment = 'cautious'
            elif current < 40:
                level = 'high'
                risk_sentiment = 'risk_off'
            else:
                level = 'extreme'
                risk_sentiment = 'panic'
            
            result = {
                'value': round(current, 2),
                'change': round(change, 2),
                'level': level,
                'risk_sentiment': risk_sentiment,
                'interpretation': f"VIX at {current:.1f} indicates {risk_sentiment.replace('_', ' ')}"
            }
            self._set_cache('vix', result)
            return result
            
        except Exception as e:
            logger.error(f"Error fetching VIX: {e}")
            return {'value': None, 'level': 'unknown', 'risk_sentiment': 'unknown'}
    
    def get_treasury_yields(self) -> Dict:
        """Get Treasury yields and calculate 2Y/10Y spread (yield curve)"""
        cached = self._get_cached('treasury')
        if cached:
            return cached
        
        if not self.available:
            return {'spread_2y10y': None, 'yield_10y': None, 'curve_status': 'unknown'}
        
        try:
            # Get 10-year yield
            tnx = yf.Ticker(self.symbols['TNX'])
            tnx_hist = tnx.history(period='5d', interval='1d')
            
            # Get 5-year yield (proxy for 2-year since ^IRX is 13-week)
            fvx = yf.Ticker(self.symbols['FVX'])
            fvx_hist = fvx.history(period='5d', interval='1d')
            
            if tnx_hist.empty or fvx_hist.empty:
                return {'spread_2y10y': None, 'yield_10y': None, 'curve_status': 'unknown'}
            
            yield_10y = tnx_hist['Close'].iloc[-1]
            yield_5y = fvx_hist['Close'].iloc[-1]
            
            # Calculate spread (using 5Y as proxy for 2Y)
            spread = yield_10y - yield_5y
            
            # Yield curve interpretation
            if spread < -0.5:
                curve_status = 'deeply_inverted'
                interpretation = 'Recession signal - risk off'
            elif spread < 0:
                curve_status = 'inverted'
                interpretation = 'Economic slowdown expected'
            elif spread < 0.5:
                curve_status = 'flat'
                interpretation = 'Economic uncertainty'
            else:
                curve_status = 'normal'
                interpretation = 'Healthy economic outlook'
            
            result = {
                'yield_10y': round(yield_10y, 3),
                'yield_5y': round(yield_5y, 3),
                'spread_5y10y': round(spread, 3),
                'curve_status': curve_status,
                'interpretation': interpretation
            }
            self._set_cache('treasury', result)
            return result
            
        except Exception as e:
            logger.error(f"Error fetching Treasury yields: {e}")
            return {'spread_2y10y': None, 'yield_10y': None, 'curve_status': 'unknown'}
    
    def get_all_macro_data(self) -> Dict:
        """Get all macro indicators in one call"""
        return {
            'dxy': self.get_dxy(),
            'vix': self.get_vix(),
            'treasury': self.get_treasury_yields(),
            'timestamp': datetime.now().isoformat()
        }
    
    def get_market_regime(self) -> Dict:
        """Determine overall market regime from macro indicators"""
        dxy = self.get_dxy()
        vix = self.get_vix()
        treasury = self.get_treasury_yields()
        
        # Score-based regime detection
        risk_score = 0  # Positive = risk-on, Negative = risk-off
        
        # VIX contribution
        if vix.get('level') == 'very_low':
            risk_score += 2
        elif vix.get('level') == 'low':
            risk_score += 1
        elif vix.get('level') == 'high':
            risk_score -= 1
        elif vix.get('level') == 'extreme':
            risk_score -= 2
        
        # Yield curve contribution
        if treasury.get('curve_status') == 'normal':
            risk_score += 1
        elif treasury.get('curve_status') == 'inverted':
            risk_score -= 1
        elif treasury.get('curve_status') == 'deeply_inverted':
            risk_score -= 2
        
        # Determine regime
        if risk_score >= 2:
            regime = 'risk_on'
            recommendation = 'Favor risk currencies (AUD, NZD) over safe havens (JPY, CHF)'
        elif risk_score <= -2:
            regime = 'risk_off'
            recommendation = 'Favor safe havens (JPY, CHF, USD) over risk currencies'
        else:
            regime = 'neutral'
            recommendation = 'Mixed signals - focus on technical setups'
        
        return {
            'regime': regime,
            'risk_score': risk_score,
            'recommendation': recommendation,
            'dxy_trend': dxy.get('trend', 'unknown'),
            'vix_level': vix.get('level', 'unknown'),
            'yield_curve': treasury.get('curve_status', 'unknown')
        }


class DataManager:
    """Main data manager coordinating all data sources"""
    
    def __init__(self):
        self.db = DatabaseManager()
        self.mt5 = MT5DataFetcher()
        self.yahoo = YahooFinanceDataFetcher()
        self.news = NewsDataFetcher()
        self.macro = MacroDataFetcher()
        self._cache = {}
        self._cache_ttl = 60  # seconds
    
    def get_ohlcv(self, symbol: str, timeframe: str, 
                  count: int = 1000, use_cache: bool = True) -> pd.DataFrame:
        """Get OHLCV data from best available source"""
        cache_key = f"{symbol}_{timeframe}_{count}"
        
        # Check cache
        if use_cache and cache_key in self._cache:
            cached_data, cached_time = self._cache[cache_key]
            if (datetime.now() - cached_time).seconds < self._cache_ttl:
                return cached_data
        
        # Try MT5 first
        df = self.mt5.get_ohlcv(symbol, timeframe, count)
        
        # Fallback to Yahoo Finance
        if df.empty and self.yahoo.available:
            logger.info(f"Falling back to Yahoo Finance for {symbol}")
            df = self.yahoo.get_ohlcv(symbol, timeframe)
        
        # Try database if still empty
        if df.empty:
            logger.info(f"Trying database for {symbol}")
            df = self.db.get_ohlcv(symbol, timeframe, limit=count)
        
        # Cache result
        if not df.empty:
            self._cache[cache_key] = (df, datetime.now())
            # Also save to database
            self.db.save_ohlcv(df.reset_index(), symbol, timeframe)
        
        return df
    
    def get_multi_timeframe_data(self, symbol: str, 
                                  timeframes: List[str] = None) -> Dict[str, pd.DataFrame]:
        """Get data for multiple timeframes"""
        timeframes = timeframes or config.trading.timeframes
        data = {}
        
        for tf in timeframes:
            df = self.get_ohlcv(symbol, tf)
            if not df.empty:
                data[tf] = df
        
        return data
    
    def get_all_symbols_data(self, timeframe: str = "H1") -> Dict[str, pd.DataFrame]:
        """Get data for all configured symbols"""
        data = {}
        
        for symbol in config.trading.symbols:
            df = self.get_ohlcv(symbol, timeframe)
            if not df.empty:
                data[symbol] = df
        
        return data
    
    def fetch_historical_data(self, symbol: str, years: int = 5) -> pd.DataFrame:
        """Fetch historical data for backtesting"""
        start_date = (datetime.now() - timedelta(days=years * 365)).strftime("%Y-%m-%d")
        
        # Try Yahoo Finance for historical data
        df = self.yahoo.get_historical(symbol, start_date)
        
        if not df.empty:
            self.db.save_ohlcv(df.reset_index(), symbol, "D1")
        
        return df
    
    def get_current_price(self, symbol: str) -> Optional[Dict]:
        """Get current price for a symbol"""
        tick = self.mt5.get_tick(symbol)
        
        if tick:
            return tick
        
        # Fallback to latest OHLCV close
        df = self.get_ohlcv(symbol, "M1", count=1)
        if not df.empty:
            return {
                'bid': df['close'].iloc[-1],
                'ask': df['close'].iloc[-1],
                'time': df.index[-1]
            }
        
        return None
    
    def get_news_sentiment(self, currencies: List[str] = None) -> Dict:
        """Get aggregated news sentiment for currencies"""
        currencies = currencies or ["EUR", "USD", "GBP", "JPY"]
        
        articles = self.news.get_forex_news()
        
        sentiment_scores = {curr: [] for curr in currencies}
        
        for article in articles:
            text = f"{article.get('title', '')} {article.get('description', '')}"
            score = self.news.analyze_sentiment(text)
            
            for curr in currencies:
                if curr in text.upper():
                    sentiment_scores[curr].append(score)
        
        # Calculate average sentiment
        avg_sentiment = {}
        for curr, scores in sentiment_scores.items():
            avg_sentiment[curr] = np.mean(scores) if scores else 0.0
        
        return avg_sentiment
    
    def cleanup_old_data(self, days: int = 30):
        """Clean up old data from database"""
        cutoff = datetime.now() - timedelta(days=days)
        
        with sqlite3.connect(self.db.db_path) as conn:
            # Keep daily data longer, clean up minute data
            conn.execute("""
                DELETE FROM ohlcv 
                WHERE timeframe IN ('M1', 'M5') AND timestamp < ?
            """, (cutoff,))
            conn.commit()
        
        logger.info(f"Cleaned up data older than {days} days")


# Singleton instance
data_manager = DataManager()


if __name__ == "__main__":
    # Test data fetching
    logging.basicConfig(level=logging.INFO)
    
    dm = DataManager()
    
    # Test OHLCV fetch
    print("Fetching EURUSD H1 data...")
    df = dm.get_ohlcv("EURUSD", "H1", count=100)
    print(f"Got {len(df)} rows")
    if not df.empty:
        print(df.tail())
    
    # Test multi-timeframe
    print("\nFetching multi-timeframe data...")
    mtf_data = dm.get_multi_timeframe_data("EURUSD", ["M15", "H1", "H4"])
    for tf, data in mtf_data.items():
        print(f"{tf}: {len(data)} rows")
