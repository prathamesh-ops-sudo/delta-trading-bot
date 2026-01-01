"""
SQLite Persistence Layer for Forex Trading System
Persists bars, trades, patterns, and system state across restarts
"""

import sqlite3
import json
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from threading import Lock
from contextlib import contextmanager

logger = logging.getLogger(__name__)

DB_PATH = os.environ.get('FOREX_DB_PATH', '/home/ec2-user/forex_trading_system/data/forex_trading.db')


class PersistenceManager:
    """SQLite-based persistence for trading system state"""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or DB_PATH
        self.lock = Lock()
        self._ensure_db_directory()
        self._init_database()
        logger.info(f"PersistenceManager initialized with database: {self.db_path}")
    
    def _ensure_db_directory(self):
        """Ensure the database directory exists"""
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
    
    @contextmanager
    def _get_connection(self):
        """Get a database connection with proper cleanup"""
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def _init_database(self):
        """Initialize database tables"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # OHLC Bars table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS bars (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp INTEGER NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, timestamp)
                )
            ''')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_bars_symbol_ts ON bars(symbol, timestamp)')
            
            # Closed trades table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS closed_trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticket INTEGER UNIQUE NOT NULL,
                    symbol TEXT NOT NULL,
                    type TEXT NOT NULL,
                    volume REAL NOT NULL,
                    price REAL NOT NULL,
                    profit REAL NOT NULL,
                    commission REAL DEFAULT 0,
                    swap REAL DEFAULT 0,
                    position_id INTEGER,
                    comment TEXT,
                    close_time INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_symbol ON closed_trades(symbol)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_time ON closed_trades(close_time)')
            
            # Pattern miner state table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS pattern_state (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern_type TEXT NOT NULL,
                    pattern_key TEXT NOT NULL,
                    pattern_data TEXT NOT NULL,
                    confidence REAL DEFAULT 0,
                    occurrences INTEGER DEFAULT 1,
                    last_seen TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(pattern_type, pattern_key)
                )
            ''')
            
            # System state table (for misc state like last sync times)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS system_state (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Trade signals history
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS signal_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    signal_id TEXT UNIQUE NOT NULL,
                    symbol TEXT NOT NULL,
                    action TEXT NOT NULL,
                    volume REAL NOT NULL,
                    sl REAL,
                    tp REAL,
                    entry_price REAL,
                    exit_price REAL,
                    profit REAL,
                    strategy TEXT,
                    reasoning TEXT,
                    executed INTEGER DEFAULT 0,
                    success INTEGER DEFAULT 0,
                    error TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    executed_at TIMESTAMP
                )
            ''')
            
            # Alerts/events log
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    message TEXT NOT NULL,
                    data TEXT,
                    acknowledged INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            logger.info("Database tables initialized successfully")
    
    # ==================== BARS ====================
    
    def save_bars(self, symbol: str, bars: List[Dict]):
        """Save OHLC bars to database"""
        if not bars:
            return
        
        with self.lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                for bar in bars:
                    try:
                        cursor.execute('''
                            INSERT OR REPLACE INTO bars (symbol, timestamp, open, high, low, close, volume)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            symbol,
                            bar.get('t', 0),
                            bar.get('o', 0),
                            bar.get('h', 0),
                            bar.get('l', 0),
                            bar.get('c', 0),
                            bar.get('v', 0)
                        ))
                    except Exception as e:
                        logger.warning(f"Error saving bar for {symbol}: {e}")
    
    def load_bars(self, symbol: str, count: int = 500) -> List[Dict]:
        """Load OHLC bars from database"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT timestamp, open, high, low, close, volume
                FROM bars
                WHERE symbol = ?
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (symbol, count))
            
            rows = cursor.fetchall()
            bars = []
            for row in reversed(rows):  # Reverse to get chronological order
                bars.append({
                    't': row['timestamp'],
                    'o': row['open'],
                    'h': row['high'],
                    'l': row['low'],
                    'c': row['close'],
                    'v': row['volume']
                })
            return bars
    
    def get_bar_count(self, symbol: str) -> int:
        """Get count of bars for a symbol"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM bars WHERE symbol = ?', (symbol,))
            return cursor.fetchone()[0]
    
    def cleanup_old_bars(self, days: int = 7):
        """Remove bars older than specified days"""
        cutoff = int((datetime.now() - timedelta(days=days)).timestamp())
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM bars WHERE timestamp < ?', (cutoff,))
            deleted = cursor.rowcount
            if deleted > 0:
                logger.info(f"Cleaned up {deleted} old bars")
    
    # ==================== TRADES ====================
    
    def save_trade(self, trade: Dict):
        """Save a closed trade to database"""
        with self.lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                try:
                    cursor.execute('''
                        INSERT OR IGNORE INTO closed_trades 
                        (ticket, symbol, type, volume, price, profit, commission, swap, position_id, comment, close_time)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        trade.get('ticket'),
                        trade.get('symbol'),
                        trade.get('type'),
                        trade.get('volume', 0),
                        trade.get('price', 0),
                        trade.get('profit', 0),
                        trade.get('commission', 0),
                        trade.get('swap', 0),
                        trade.get('position_id', 0),
                        trade.get('comment', ''),
                        trade.get('time', 0)
                    ))
                except Exception as e:
                    logger.warning(f"Error saving trade: {e}")
    
    def save_trades(self, trades: List[Dict]):
        """Save multiple closed trades"""
        for trade in trades:
            self.save_trade(trade)
    
    def load_trades(self, count: int = 100, symbol: str = None) -> List[Dict]:
        """Load closed trades from database"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if symbol:
                cursor.execute('''
                    SELECT * FROM closed_trades
                    WHERE symbol = ?
                    ORDER BY close_time DESC
                    LIMIT ?
                ''', (symbol, count))
            else:
                cursor.execute('''
                    SELECT * FROM closed_trades
                    ORDER BY close_time DESC
                    LIMIT ?
                ''', (count,))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def get_trade_stats(self) -> Dict:
        """Get trading statistics"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT 
                    COUNT(*) as total_trades,
                    SUM(profit) as total_profit,
                    SUM(CASE WHEN profit > 0 THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN profit < 0 THEN 1 ELSE 0 END) as losses,
                    AVG(profit) as avg_profit,
                    MAX(profit) as max_profit,
                    MIN(profit) as max_loss
                FROM closed_trades
            ''')
            row = cursor.fetchone()
            total = row['total_trades'] or 0
            wins = row['wins'] or 0
            return {
                'total_trades': total,
                'total_profit': row['total_profit'] or 0,
                'wins': wins,
                'losses': row['losses'] or 0,
                'win_rate': wins / total if total > 0 else 0,
                'avg_profit': row['avg_profit'] or 0,
                'max_profit': row['max_profit'] or 0,
                'max_loss': row['max_loss'] or 0
            }
    
    # ==================== PATTERNS ====================
    
    def save_pattern(self, pattern_type: str, pattern_key: str, pattern_data: Dict, confidence: float = 0):
        """Save or update a pattern"""
        with self.lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO pattern_state (pattern_type, pattern_key, pattern_data, confidence, last_seen, updated_at)
                    VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                    ON CONFLICT(pattern_type, pattern_key) DO UPDATE SET
                        pattern_data = excluded.pattern_data,
                        confidence = excluded.confidence,
                        occurrences = occurrences + 1,
                        last_seen = CURRENT_TIMESTAMP,
                        updated_at = CURRENT_TIMESTAMP
                ''', (pattern_type, pattern_key, json.dumps(pattern_data), confidence))
    
    def load_patterns(self, pattern_type: str = None) -> List[Dict]:
        """Load patterns from database"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if pattern_type:
                cursor.execute('''
                    SELECT * FROM pattern_state
                    WHERE pattern_type = ?
                    ORDER BY confidence DESC
                ''', (pattern_type,))
            else:
                cursor.execute('SELECT * FROM pattern_state ORDER BY confidence DESC')
            
            patterns = []
            for row in cursor.fetchall():
                pattern = dict(row)
                pattern['pattern_data'] = json.loads(pattern['pattern_data'])
                patterns.append(pattern)
            return patterns
    
    # ==================== SYSTEM STATE ====================
    
    def set_state(self, key: str, value: Any):
        """Set a system state value"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO system_state (key, value, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
            ''', (key, json.dumps(value)))
    
    def get_state(self, key: str, default: Any = None) -> Any:
        """Get a system state value"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT value FROM system_state WHERE key = ?', (key,))
            row = cursor.fetchone()
            if row:
                return json.loads(row['value'])
            return default
    
    # ==================== SIGNALS ====================
    
    def save_signal(self, signal: Dict):
        """Save a trade signal to history"""
        with self.lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO signal_history 
                    (signal_id, symbol, action, volume, sl, tp, entry_price, strategy, reasoning)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    signal.get('id'),
                    signal.get('symbol'),
                    signal.get('action'),
                    signal.get('volume', 0),
                    signal.get('sl', 0),
                    signal.get('tp', 0),
                    signal.get('entry_price', 0),
                    signal.get('strategy', ''),
                    signal.get('reasoning', '')
                ))
    
    def update_signal_result(self, signal_id: str, success: bool, error: str = None, 
                             exit_price: float = None, profit: float = None):
        """Update signal with execution result"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE signal_history
                SET executed = 1, success = ?, error = ?, exit_price = ?, profit = ?, executed_at = CURRENT_TIMESTAMP
                WHERE signal_id = ?
            ''', (1 if success else 0, error, exit_price, profit, signal_id))
    
    # ==================== ALERTS ====================
    
    def log_alert(self, alert_type: str, severity: str, message: str, data: Dict = None):
        """Log an alert/event"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO alerts (alert_type, severity, message, data)
                VALUES (?, ?, ?, ?)
            ''', (alert_type, severity, message, json.dumps(data) if data else None))
    
    def get_recent_alerts(self, count: int = 50, severity: str = None) -> List[Dict]:
        """Get recent alerts"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if severity:
                cursor.execute('''
                    SELECT * FROM alerts
                    WHERE severity = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                ''', (severity, count))
            else:
                cursor.execute('''
                    SELECT * FROM alerts
                    ORDER BY created_at DESC
                    LIMIT ?
                ''', (count,))
            
            alerts = []
            for row in cursor.fetchall():
                alert = dict(row)
                if alert.get('data'):
                    alert['data'] = json.loads(alert['data'])
                alerts.append(alert)
            return alerts


# Global instance
persistence = PersistenceManager()


def get_persistence() -> PersistenceManager:
    """Get the global persistence manager"""
    return persistence
