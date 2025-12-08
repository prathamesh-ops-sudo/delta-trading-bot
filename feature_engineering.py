"""
Advanced Feature Engineering for Trading Signals
Generates 50+ technical indicators for ML model training
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import talib
from scipy.stats import linregress
import logging

logger = logging.getLogger(__name__)


class FeatureEngine:
    """Generate comprehensive technical indicators for ML models"""

    def __init__(self):
        self.feature_names = []

    def prepare_data(self, candles: List[Dict]) -> pd.DataFrame:
        """Convert raw candle data to pandas DataFrame"""
        df = pd.DataFrame(candles)

        # Ensure proper column names and types
        if 'time' in df.columns:
            df['timestamp'] = pd.to_datetime(df['time'], unit='s')
        else:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

        # Rename columns to standard OHLCV format
        column_mapping = {
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume'
        }

        for old_col, new_col in column_mapping.items():
            if old_col in df.columns:
                df[new_col] = pd.to_numeric(df[old_col], errors='coerce')

        df = df.sort_values('timestamp').reset_index(drop=True)
        return df

    def calculate_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators and features"""
        df = df.copy()

        # Price-based features
        df = self._add_price_features(df)

        # Trend indicators
        df = self._add_trend_indicators(df)

        # Momentum indicators
        df = self._add_momentum_indicators(df)

        # Volatility indicators
        df = self._add_volatility_indicators(df)

        # Volume indicators
        df = self._add_volume_indicators(df)

        # Support/Resistance levels
        df = self._add_support_resistance(df)

        # Pattern recognition
        df = self._add_patterns(df)

        # Statistical features
        df = self._add_statistical_features(df)

        # Market microstructure
        df = self._add_microstructure_features(df)

        # Drop NaN values
        df = df.dropna()

        return df

    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic price-based features"""
        # Price changes
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

        # Price ranges
        df['high_low_range'] = (df['high'] - df['low']) / df['close']
        df['close_open_range'] = (df['close'] - df['open']) / df['open']

        # Typical price
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3

        # Weighted close
        df['weighted_close'] = (df['high'] + df['low'] + 2 * df['close']) / 4

        return df

    def _add_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Trend-following indicators"""
        close = df['close'].values

        # Moving Averages
        for period in [7, 14, 21, 50, 100, 200]:
            df[f'sma_{period}'] = talib.SMA(close, timeperiod=period)
            df[f'ema_{period}'] = talib.EMA(close, timeperiod=period)

        # Moving Average Crossovers
        df['sma_7_14_cross'] = df['sma_7'] / df['sma_14'] - 1
        df['sma_14_50_cross'] = df['sma_14'] / df['sma_50'] - 1
        df['ema_7_21_cross'] = df['ema_7'] / df['ema_21'] - 1

        # MACD
        df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(
            close, fastperiod=12, slowperiod=26, signalperiod=9
        )

        # ADX (Average Directional Index)
        df['adx'] = talib.ADX(df['high'].values, df['low'].values, close, timeperiod=14)
        df['adx_trend'] = (df['adx'] > 25).astype(int)

        # Parabolic SAR
        df['sar'] = talib.SAR(df['high'].values, df['low'].values)
        df['sar_signal'] = (close > df['sar']).astype(int)

        # Ichimoku Cloud components
        high_9 = df['high'].rolling(window=9).max()
        low_9 = df['low'].rolling(window=9).min()
        df['tenkan_sen'] = (high_9 + low_9) / 2

        high_26 = df['high'].rolling(window=26).max()
        low_26 = df['low'].rolling(window=26).min()
        df['kijun_sen'] = (high_26 + low_26) / 2

        df['ichimoku_signal'] = (df['tenkan_sen'] > df['kijun_sen']).astype(int)

        return df

    def _add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Momentum and oscillator indicators"""
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values

        # RSI (Relative Strength Index)
        for period in [7, 14, 21]:
            df[f'rsi_{period}'] = talib.RSI(close, timeperiod=period)

        # Stochastic Oscillator
        df['stoch_k'], df['stoch_d'] = talib.STOCH(
            high, low, close,
            fastk_period=14, slowk_period=3, slowd_period=3
        )

        # Stochastic RSI
        df['stochrsi_k'], df['stochrsi_d'] = talib.STOCHRSI(
            close, timeperiod=14, fastk_period=3, fastd_period=3
        )

        # CCI (Commodity Channel Index)
        df['cci'] = talib.CCI(high, low, close, timeperiod=14)

        # Williams %R
        df['willr'] = talib.WILLR(high, low, close, timeperiod=14)

        # ROC (Rate of Change)
        for period in [5, 10, 20]:
            df[f'roc_{period}'] = talib.ROC(close, timeperiod=period)

        # Money Flow Index
        df['mfi'] = talib.MFI(high, low, close, df['volume'].values, timeperiod=14)

        # Ultimate Oscillator
        df['ultosc'] = talib.ULTOSC(high, low, close)

        return df

    def _add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Volatility and range indicators"""
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values

        # Bollinger Bands
        for period in [20, 50]:
            upper, middle, lower = talib.BBANDS(close, timeperiod=period, nbdevup=2, nbdevdn=2)
            df[f'bb_upper_{period}'] = upper
            df[f'bb_middle_{period}'] = middle
            df[f'bb_lower_{period}'] = lower
            df[f'bb_width_{period}'] = (upper - lower) / middle
            df[f'bb_position_{period}'] = (close - lower) / (upper - lower)

        # ATR (Average True Range)
        for period in [7, 14, 21]:
            df[f'atr_{period}'] = talib.ATR(high, low, close, timeperiod=period)
            df[f'atr_pct_{period}'] = df[f'atr_{period}'] / close

        # Keltner Channels
        ema_20 = talib.EMA(close, timeperiod=20)
        atr_20 = df['atr_14']
        df['keltner_upper'] = ema_20 + 2 * atr_20
        df['keltner_lower'] = ema_20 - 2 * atr_20

        # Standard Deviation
        df['std_20'] = df['close'].rolling(window=20).std()
        df['volatility_ratio'] = df['std_20'] / df['close']

        return df

    def _add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Volume-based indicators"""
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values

        # Volume moving averages
        df['volume_sma_20'] = talib.SMA(volume, timeperiod=20)
        df['volume_ratio'] = volume / df['volume_sma_20']

        # On-Balance Volume
        df['obv'] = talib.OBV(close, volume)
        df['obv_sma'] = talib.SMA(df['obv'].values, timeperiod=20)

        # Accumulation/Distribution
        df['ad'] = talib.AD(high, low, close, volume)

        # Chaikin Money Flow
        df['cmf'] = talib.ADOSC(high, low, close, volume, fastperiod=3, slowperiod=10)

        # Volume Price Trend
        df['vpt'] = (df['close'].pct_change() * volume).cumsum()

        # Force Index
        df['force_index'] = df['close'].diff() * volume
        df['force_index_13'] = df['force_index'].ewm(span=13).mean()

        return df

    def _add_support_resistance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Support and resistance levels"""
        # Pivot points
        df['pivot'] = (df['high'].shift(1) + df['low'].shift(1) + df['close'].shift(1)) / 3
        df['r1'] = 2 * df['pivot'] - df['low'].shift(1)
        df['s1'] = 2 * df['pivot'] - df['high'].shift(1)
        df['r2'] = df['pivot'] + (df['high'].shift(1) - df['low'].shift(1))
        df['s2'] = df['pivot'] - (df['high'].shift(1) - df['low'].shift(1))

        # Distance to pivot levels
        df['dist_to_pivot'] = (df['close'] - df['pivot']) / df['close']
        df['dist_to_r1'] = (df['r1'] - df['close']) / df['close']
        df['dist_to_s1'] = (df['close'] - df['s1']) / df['close']

        return df

    def _add_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Candlestick pattern recognition"""
        open_p = df['open'].values
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values

        # Major candlestick patterns
        df['cdl_doji'] = talib.CDLDOJI(open_p, high, low, close)
        df['cdl_hammer'] = talib.CDLHAMMER(open_p, high, low, close)
        df['cdl_shooting_star'] = talib.CDLSHOOTINGSTAR(open_p, high, low, close)
        df['cdl_engulfing'] = talib.CDLENGULFING(open_p, high, low, close)
        df['cdl_morning_star'] = talib.CDLMORNINGSTAR(open_p, high, low, close)
        df['cdl_evening_star'] = talib.CDLEVENINGSTAR(open_p, high, low, close)
        df['cdl_harami'] = talib.CDLHARAMI(open_p, high, low, close)

        return df

    def _add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Statistical and derived features"""
        # Linear regression slope
        def calculate_slope(series, window):
            slopes = []
            for i in range(len(series)):
                if i < window:
                    slopes.append(np.nan)
                else:
                    y = series[i-window:i].values
                    x = np.arange(window)
                    slope, _, _, _, _ = linregress(x, y)
                    slopes.append(slope)
            return slopes

        df['price_slope_10'] = calculate_slope(df['close'], 10)
        df['price_slope_20'] = calculate_slope(df['close'], 20)

        # Skewness and Kurtosis
        df['returns_skew_20'] = df['returns'].rolling(window=20).skew()
        df['returns_kurt_20'] = df['returns'].rolling(window=20).kurt()

        # Autocorrelation
        df['autocorr_5'] = df['returns'].rolling(window=20).apply(
            lambda x: x.autocorr(lag=5) if len(x) > 5 else np.nan
        )

        return df

    def _add_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Market microstructure features"""
        # Spread
        df['spread'] = df['high'] - df['low']
        df['spread_pct'] = df['spread'] / df['close']

        # Upper/Lower wick ratios
        df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
        df['upper_wick_ratio'] = df['upper_wick'] / df['spread']
        df['lower_wick_ratio'] = df['lower_wick'] / df['spread']

        # Body size
        df['body'] = abs(df['close'] - df['open'])
        df['body_ratio'] = df['body'] / df['spread']

        return df

    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Get list of feature columns (excluding OHLCV and timestamp)"""
        exclude_cols = ['timestamp', 'time', 'open', 'high', 'low', 'close', 'volume']
        return [col for col in df.columns if col not in exclude_cols]

    def create_sequences(self, df: pd.DataFrame, lookback: int = 60) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM model

        Returns:
            X: Input sequences (samples, lookback, features)
            y: Target values (samples,) - 1 if price goes up, 0 if down
        """
        feature_cols = self.get_feature_columns(df)
        data = df[feature_cols].values

        X, y = [], []

        for i in range(lookback, len(data)):
            X.append(data[i-lookback:i])

            # Target: 1 if next candle closes higher, 0 otherwise
            future_return = (df['close'].iloc[i] - df['close'].iloc[i-1]) / df['close'].iloc[i-1]
            y.append(1 if future_return > 0 else 0)

        return np.array(X), np.array(y)
