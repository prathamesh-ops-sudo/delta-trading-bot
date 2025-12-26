"""
Technical Indicators and Feature Engineering Module
Implements all technical analysis indicators and ML feature generation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("TA-Lib not available - using custom implementations")

from config import INDICATOR_SETTINGS

logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """Technical analysis indicators calculator"""
    
    def __init__(self):
        self.settings = INDICATOR_SETTINGS
    
    # ==================== Trend Indicators ====================
    
    def sma(self, data: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average"""
        return data.rolling(window=period).mean()
    
    def ema(self, data: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average"""
        return data.ewm(span=period, adjust=False).mean()
    
    def wma(self, data: pd.Series, period: int) -> pd.Series:
        """Weighted Moving Average"""
        weights = np.arange(1, period + 1)
        return data.rolling(window=period).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    def dema(self, data: pd.Series, period: int) -> pd.Series:
        """Double Exponential Moving Average"""
        ema1 = self.ema(data, period)
        ema2 = self.ema(ema1, period)
        return 2 * ema1 - ema2
    
    def tema(self, data: pd.Series, period: int) -> pd.Series:
        """Triple Exponential Moving Average"""
        ema1 = self.ema(data, period)
        ema2 = self.ema(ema1, period)
        ema3 = self.ema(ema2, period)
        return 3 * ema1 - 3 * ema2 + ema3
    
    # ==================== Momentum Indicators ====================
    
    def rsi(self, data: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index"""
        if TALIB_AVAILABLE:
            return pd.Series(talib.RSI(data.values, timeperiod=period), index=data.index)
        
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def macd(self, data: pd.Series, fast: int = 12, slow: int = 26, 
             signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """MACD - Moving Average Convergence Divergence"""
        if TALIB_AVAILABLE:
            macd, signal_line, hist = talib.MACD(
                data.values, fastperiod=fast, slowperiod=slow, signalperiod=signal
            )
            return (
                pd.Series(macd, index=data.index),
                pd.Series(signal_line, index=data.index),
                pd.Series(hist, index=data.index)
            )
        
        ema_fast = self.ema(data, fast)
        ema_slow = self.ema(data, slow)
        macd_line = ema_fast - ema_slow
        signal_line = self.ema(macd_line, signal)
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series,
                   k_period: int = 14, d_period: int = 3, 
                   slowing: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Stochastic Oscillator"""
        if TALIB_AVAILABLE:
            slowk, slowd = talib.STOCH(
                high.values, low.values, close.values,
                fastk_period=k_period, slowk_period=slowing, 
                slowk_matype=0, slowd_period=d_period, slowd_matype=0
            )
            return pd.Series(slowk, index=close.index), pd.Series(slowd, index=close.index)
        
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        
        fast_k = 100 * (close - lowest_low) / (highest_high - lowest_low)
        slow_k = fast_k.rolling(window=slowing).mean()
        slow_d = slow_k.rolling(window=d_period).mean()
        
        return slow_k, slow_d
    
    def williams_r(self, high: pd.Series, low: pd.Series, 
                   close: pd.Series, period: int = 14) -> pd.Series:
        """Williams %R"""
        if TALIB_AVAILABLE:
            return pd.Series(
                talib.WILLR(high.values, low.values, close.values, timeperiod=period),
                index=close.index
            )
        
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        
        return -100 * (highest_high - close) / (highest_high - lowest_low)
    
    def cci(self, high: pd.Series, low: pd.Series, 
            close: pd.Series, period: int = 20) -> pd.Series:
        """Commodity Channel Index"""
        if TALIB_AVAILABLE:
            return pd.Series(
                talib.CCI(high.values, low.values, close.values, timeperiod=period),
                index=close.index
            )
        
        tp = (high + low + close) / 3
        sma_tp = tp.rolling(window=period).mean()
        mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
        
        return (tp - sma_tp) / (0.015 * mad)
    
    def momentum(self, data: pd.Series, period: int = 10) -> pd.Series:
        """Momentum indicator"""
        if TALIB_AVAILABLE:
            return pd.Series(talib.MOM(data.values, timeperiod=period), index=data.index)
        return data.diff(period)
    
    def roc(self, data: pd.Series, period: int = 10) -> pd.Series:
        """Rate of Change"""
        if TALIB_AVAILABLE:
            return pd.Series(talib.ROC(data.values, timeperiod=period), index=data.index)
        return ((data - data.shift(period)) / data.shift(period)) * 100
    
    # ==================== Volatility Indicators ====================
    
    def atr(self, high: pd.Series, low: pd.Series, 
            close: pd.Series, period: int = 14) -> pd.Series:
        """Average True Range"""
        if TALIB_AVAILABLE:
            return pd.Series(
                talib.ATR(high.values, low.values, close.values, timeperiod=period),
                index=close.index
            )
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()
    
    def bollinger_bands(self, data: pd.Series, period: int = 20, 
                        std_dev: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Bollinger Bands"""
        if TALIB_AVAILABLE:
            upper, middle, lower = talib.BBANDS(
                data.values, timeperiod=period, nbdevup=std_dev, nbdevdn=std_dev
            )
            return (
                pd.Series(upper, index=data.index),
                pd.Series(middle, index=data.index),
                pd.Series(lower, index=data.index)
            )
        
        middle = self.sma(data, period)
        std = data.rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        
        return upper, middle, lower
    
    def keltner_channel(self, high: pd.Series, low: pd.Series, 
                        close: pd.Series, period: int = 20, 
                        atr_mult: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Keltner Channel"""
        middle = self.ema(close, period)
        atr_val = self.atr(high, low, close, period)
        
        upper = middle + (atr_mult * atr_val)
        lower = middle - (atr_mult * atr_val)
        
        return upper, middle, lower
    
    def donchian_channel(self, high: pd.Series, low: pd.Series, 
                         period: int = 20) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Donchian Channel"""
        upper = high.rolling(window=period).max()
        lower = low.rolling(window=period).min()
        middle = (upper + lower) / 2
        
        return upper, middle, lower
    
    # ==================== Volume Indicators ====================
    
    def obv(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """On-Balance Volume"""
        if TALIB_AVAILABLE:
            return pd.Series(talib.OBV(close.values, volume.values), index=close.index)
        
        direction = np.where(close > close.shift(), 1, 
                            np.where(close < close.shift(), -1, 0))
        return (volume * direction).cumsum()
    
    def vwap(self, high: pd.Series, low: pd.Series, 
             close: pd.Series, volume: pd.Series) -> pd.Series:
        """Volume Weighted Average Price"""
        tp = (high + low + close) / 3
        return (tp * volume).cumsum() / volume.cumsum()
    
    def mfi(self, high: pd.Series, low: pd.Series, close: pd.Series,
            volume: pd.Series, period: int = 14) -> pd.Series:
        """Money Flow Index"""
        if TALIB_AVAILABLE:
            return pd.Series(
                talib.MFI(high.values, low.values, close.values, volume.values, timeperiod=period),
                index=close.index
            )
        
        tp = (high + low + close) / 3
        mf = tp * volume
        
        positive_mf = mf.where(tp > tp.shift(), 0).rolling(window=period).sum()
        negative_mf = mf.where(tp < tp.shift(), 0).rolling(window=period).sum()
        
        mfi = 100 - (100 / (1 + positive_mf / negative_mf))
        return mfi
    
    # ==================== Trend Strength ====================
    
    def adx(self, high: pd.Series, low: pd.Series, 
            close: pd.Series, period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Average Directional Index with +DI and -DI"""
        if TALIB_AVAILABLE:
            adx = talib.ADX(high.values, low.values, close.values, timeperiod=period)
            plus_di = talib.PLUS_DI(high.values, low.values, close.values, timeperiod=period)
            minus_di = talib.MINUS_DI(high.values, low.values, close.values, timeperiod=period)
            return (
                pd.Series(adx, index=close.index),
                pd.Series(plus_di, index=close.index),
                pd.Series(minus_di, index=close.index)
            )
        
        # Calculate +DM and -DM
        plus_dm = high.diff()
        minus_dm = -low.diff()
        
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        
        # Calculate TR
        tr = self.atr(high, low, close, 1) * period  # Simplified
        
        # Smooth
        plus_di = 100 * self.ema(plus_dm, period) / self.atr(high, low, close, period)
        minus_di = 100 * self.ema(minus_dm, period) / self.atr(high, low, close, period)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = self.ema(dx, period)
        
        return adx, plus_di, minus_di
    
    def aroon(self, high: pd.Series, low: pd.Series, 
              period: int = 25) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Aroon Indicator"""
        if TALIB_AVAILABLE:
            aroon_down, aroon_up = talib.AROON(high.values, low.values, timeperiod=period)
            return (
                pd.Series(aroon_up, index=high.index),
                pd.Series(aroon_down, index=high.index),
                pd.Series(aroon_up - aroon_down, index=high.index)
            )
        
        aroon_up = high.rolling(window=period + 1).apply(
            lambda x: 100 * (period - x.argmax()) / period, raw=True
        )
        aroon_down = low.rolling(window=period + 1).apply(
            lambda x: 100 * (period - x.argmin()) / period, raw=True
        )
        
        return aroon_up, aroon_down, aroon_up - aroon_down
    
    # ==================== Pattern Recognition ====================
    
    def pivot_points(self, high: pd.Series, low: pd.Series, 
                     close: pd.Series) -> Dict[str, pd.Series]:
        """Calculate pivot points"""
        pp = (high.shift() + low.shift() + close.shift()) / 3
        
        r1 = 2 * pp - low.shift()
        s1 = 2 * pp - high.shift()
        r2 = pp + (high.shift() - low.shift())
        s2 = pp - (high.shift() - low.shift())
        r3 = high.shift() + 2 * (pp - low.shift())
        s3 = low.shift() - 2 * (high.shift() - pp)
        
        return {
            'PP': pp, 'R1': r1, 'R2': r2, 'R3': r3,
            'S1': s1, 'S2': s2, 'S3': s3
        }
    
    def fibonacci_retracement(self, high: float, low: float) -> Dict[str, float]:
        """Calculate Fibonacci retracement levels"""
        diff = high - low
        
        return {
            '0.0': low,
            '23.6': low + 0.236 * diff,
            '38.2': low + 0.382 * diff,
            '50.0': low + 0.5 * diff,
            '61.8': low + 0.618 * diff,
            '78.6': low + 0.786 * diff,
            '100.0': high
        }
    
    # ==================== Candlestick Patterns ====================
    
    def detect_candlestick_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect candlestick patterns"""
        patterns = pd.DataFrame(index=df.index)
        
        o, h, l, c = df['open'], df['high'], df['low'], df['close']
        body = abs(c - o)
        upper_shadow = h - np.maximum(o, c)
        lower_shadow = np.minimum(o, c) - l
        
        # Doji
        patterns['doji'] = body < (h - l) * 0.1
        
        # Hammer
        patterns['hammer'] = (
            (lower_shadow > 2 * body) & 
            (upper_shadow < body * 0.3) &
            (c > o)
        )
        
        # Shooting Star
        patterns['shooting_star'] = (
            (upper_shadow > 2 * body) & 
            (lower_shadow < body * 0.3) &
            (c < o)
        )
        
        # Engulfing
        patterns['bullish_engulfing'] = (
            (c.shift() < o.shift()) &  # Previous bearish
            (c > o) &  # Current bullish
            (o < c.shift()) &  # Open below previous close
            (c > o.shift())  # Close above previous open
        )
        
        patterns['bearish_engulfing'] = (
            (c.shift() > o.shift()) &  # Previous bullish
            (c < o) &  # Current bearish
            (o > c.shift()) &  # Open above previous close
            (c < o.shift())  # Close below previous open
        )
        
        # Morning/Evening Star (simplified)
        patterns['morning_star'] = (
            (c.shift(2) < o.shift(2)) &  # First bearish
            (body.shift() < body.shift(2) * 0.3) &  # Small middle
            (c > o) &  # Third bullish
            (c > (o.shift(2) + c.shift(2)) / 2)  # Close above midpoint
        )
        
        patterns['evening_star'] = (
            (c.shift(2) > o.shift(2)) &  # First bullish
            (body.shift() < body.shift(2) * 0.3) &  # Small middle
            (c < o) &  # Third bearish
            (c < (o.shift(2) + c.shift(2)) / 2)  # Close below midpoint
        )
        
        return patterns


class FeatureEngineer:
    """Feature engineering for ML models"""
    
    def __init__(self):
        self.indicators = TechnicalIndicators()
    
    def create_features(self, df: pd.DataFrame, 
                        include_patterns: bool = True) -> pd.DataFrame:
        """Create comprehensive feature set for ML"""
        features = pd.DataFrame(index=df.index)
        
        o, h, l, c, v = df['open'], df['high'], df['low'], df['close'], df.get('volume', pd.Series(0, index=df.index))
        
        # Price-based features
        features['returns'] = c.pct_change()
        features['log_returns'] = np.log(c / c.shift())
        features['high_low_range'] = (h - l) / c
        features['close_open_range'] = (c - o) / o
        
        # Moving averages
        for period in [5, 10, 20, 50, 100, 200]:
            features[f'sma_{period}'] = self.indicators.sma(c, period)
            features[f'ema_{period}'] = self.indicators.ema(c, period)
            features[f'price_sma_{period}_ratio'] = c / features[f'sma_{period}']
        
        # RSI
        features['rsi_14'] = self.indicators.rsi(c, 14)
        features['rsi_7'] = self.indicators.rsi(c, 7)
        features['rsi_21'] = self.indicators.rsi(c, 21)
        
        # MACD
        macd, signal, hist = self.indicators.macd(c)
        features['macd'] = macd
        features['macd_signal'] = signal
        features['macd_hist'] = hist
        
        # Stochastic
        stoch_k, stoch_d = self.indicators.stochastic(h, l, c)
        features['stoch_k'] = stoch_k
        features['stoch_d'] = stoch_d
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = self.indicators.bollinger_bands(c)
        features['bb_upper'] = bb_upper
        features['bb_middle'] = bb_middle
        features['bb_lower'] = bb_lower
        features['bb_width'] = (bb_upper - bb_lower) / bb_middle
        features['bb_position'] = (c - bb_lower) / (bb_upper - bb_lower)
        
        # ATR
        features['atr_14'] = self.indicators.atr(h, l, c, 14)
        features['atr_7'] = self.indicators.atr(h, l, c, 7)
        features['atr_normalized'] = features['atr_14'] / c
        
        # ADX
        adx, plus_di, minus_di = self.indicators.adx(h, l, c)
        features['adx'] = adx
        features['plus_di'] = plus_di
        features['minus_di'] = minus_di
        features['di_diff'] = plus_di - minus_di
        
        # CCI
        features['cci'] = self.indicators.cci(h, l, c)
        
        # Williams %R
        features['williams_r'] = self.indicators.williams_r(h, l, c)
        
        # Momentum
        features['momentum_10'] = self.indicators.momentum(c, 10)
        features['roc_10'] = self.indicators.roc(c, 10)
        
        # Volume features (if available)
        if v.sum() > 0:
            features['volume_sma_20'] = self.indicators.sma(v, 20)
            features['volume_ratio'] = v / features['volume_sma_20']
            features['obv'] = self.indicators.obv(c, v)
            features['mfi'] = self.indicators.mfi(h, l, c, v)
        
        # Volatility features
        features['volatility_20'] = c.rolling(20).std() / c
        features['volatility_50'] = c.rolling(50).std() / c
        
        # Trend features
        features['trend_5'] = (c - c.shift(5)) / c.shift(5)
        features['trend_10'] = (c - c.shift(10)) / c.shift(10)
        features['trend_20'] = (c - c.shift(20)) / c.shift(20)
        
        # Higher highs / Lower lows
        features['higher_high'] = (h > h.shift()).astype(int)
        features['lower_low'] = (l < l.shift()).astype(int)
        
        # Support/Resistance proximity
        pivot = self.indicators.pivot_points(h, l, c)
        features['dist_to_r1'] = (pivot['R1'] - c) / c
        features['dist_to_s1'] = (c - pivot['S1']) / c
        
        # Candlestick patterns
        if include_patterns:
            patterns = self.indicators.detect_candlestick_patterns(df)
            for col in patterns.columns:
                features[f'pattern_{col}'] = patterns[col].astype(int)
        
        # Lagged features
        for lag in [1, 2, 3, 5, 10]:
            features[f'returns_lag_{lag}'] = features['returns'].shift(lag)
            features[f'rsi_lag_{lag}'] = features['rsi_14'].shift(lag)
        
        # Rolling statistics
        for window in [5, 10, 20]:
            features[f'returns_mean_{window}'] = features['returns'].rolling(window).mean()
            features[f'returns_std_{window}'] = features['returns'].rolling(window).std()
            features[f'returns_skew_{window}'] = features['returns'].rolling(window).skew()
            features[f'returns_kurt_{window}'] = features['returns'].rolling(window).kurt()
        
        return features
    
    def create_target(self, df: pd.DataFrame, 
                      lookahead: int = 1, 
                      threshold: float = 0.0) -> pd.Series:
        """Create target variable for classification"""
        future_returns = df['close'].shift(-lookahead) / df['close'] - 1
        
        # 0 = Sell, 1 = Hold, 2 = Buy
        target = pd.Series(1, index=df.index)  # Default Hold
        target[future_returns > threshold] = 2  # Buy
        target[future_returns < -threshold] = 0  # Sell
        
        return target
    
    def create_regression_target(self, df: pd.DataFrame, 
                                  lookahead: int = 1) -> pd.Series:
        """Create target for regression (future returns)"""
        return df['close'].shift(-lookahead) / df['close'] - 1
    
    def normalize_features(self, features: pd.DataFrame, 
                           method: str = 'zscore') -> Tuple[pd.DataFrame, Dict]:
        """Normalize features"""
        stats = {}
        normalized = features.copy()
        
        for col in features.columns:
            if method == 'zscore':
                mean = features[col].mean()
                std = features[col].std()
                if std > 0:
                    normalized[col] = (features[col] - mean) / std
                stats[col] = {'mean': mean, 'std': std}
            elif method == 'minmax':
                min_val = features[col].min()
                max_val = features[col].max()
                if max_val > min_val:
                    normalized[col] = (features[col] - min_val) / (max_val - min_val)
                stats[col] = {'min': min_val, 'max': max_val}
        
        return normalized, stats
    
    def prepare_sequences(self, features: pd.DataFrame, 
                          target: pd.Series,
                          sequence_length: int = 60) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for LSTM/Transformer"""
        X, y = [], []
        
        features_arr = features.values
        target_arr = target.values
        
        for i in range(sequence_length, len(features_arr)):
            X.append(features_arr[i-sequence_length:i])
            y.append(target_arr[i])
        
        return np.array(X), np.array(y)
    
    def get_feature_importance_names(self) -> List[str]:
        """Get list of feature names for interpretation"""
        return [
            'returns', 'log_returns', 'high_low_range', 'close_open_range',
            'sma_5', 'sma_10', 'sma_20', 'sma_50', 'sma_100', 'sma_200',
            'ema_5', 'ema_10', 'ema_20', 'ema_50', 'ema_100', 'ema_200',
            'rsi_14', 'rsi_7', 'rsi_21',
            'macd', 'macd_signal', 'macd_hist',
            'stoch_k', 'stoch_d',
            'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_position',
            'atr_14', 'atr_7', 'atr_normalized',
            'adx', 'plus_di', 'minus_di', 'di_diff',
            'cci', 'williams_r',
            'momentum_10', 'roc_10',
            'volatility_20', 'volatility_50',
            'trend_5', 'trend_10', 'trend_20'
        ]


# Singleton instances
indicators = TechnicalIndicators()
feature_engineer = FeatureEngineer()


if __name__ == "__main__":
    # Test indicators
    import yfinance as yf
    
    print("Testing indicators...")
    
    # Get sample data
    ticker = yf.Ticker("EURUSD=X")
    df = ticker.history(period="1mo", interval="1h")
    df.columns = [c.lower() for c in df.columns]
    
    if not df.empty:
        # Test feature creation
        features = feature_engineer.create_features(df)
        print(f"Created {len(features.columns)} features")
        print(f"Features shape: {features.shape}")
        print(f"\nSample features:\n{features.tail()}")
        
        # Test target creation
        target = feature_engineer.create_target(df, lookahead=1, threshold=0.0001)
        print(f"\nTarget distribution:\n{target.value_counts()}")
    else:
        print("Could not fetch sample data")
