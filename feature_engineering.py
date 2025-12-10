"""
Advanced Feature Engineering for Trading Signals
Generates 50+ technical indicators for ML model training
Fixed for Binance data and proper error handling
"""
import numpy as np
import pandas as pd
from typing import List
import talib
from logger_config import logger


class FeatureEngine:
    """Generate comprehensive technical indicators for ML models"""

    def __init__(self):
        self.feature_names = []

    def prepare_data_from_binance(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare DataFrame from Binance data (already formatted)

        Args:
            df: DataFrame with columns: timestamp, open, high, low, close, volume

        Returns:
            Cleaned and validated DataFrame
        """
        df = df.copy()

        # Ensure all OHLCV columns are float64 (required by TA-Lib)
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('float64')

        # Drop any rows with NaN values in OHLCV columns
        df = df.dropna(subset=['open', 'high', 'low', 'close', 'volume'])

        # Remove any infinite values
        df = df.replace([np.inf, -np.inf], np.nan).dropna()

        df = df.sort_values('timestamp').reset_index(drop=True)

        logger.info(f"Prepared {len(df)} candles for feature engineering")
        return df

    def calculate_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators and features with error handling"""
        try:
            df = df.copy()

            # Validate minimum data length
            if len(df) < 100:
                logger.warning(f"Only {len(df)} candles available. Need at least 100 for accurate features.")

            logger.info("Calculating features...")

            # Price-based features
            df = self._add_price_features(df)
            logger.debug("✓ Price features added")

            # Trend indicators
            df = self._add_trend_indicators(df)
            logger.debug("✓ Trend indicators added")

            # Momentum indicators
            df = self._add_momentum_indicators(df)
            logger.debug("✓ Momentum indicators added")

            # Volatility indicators
            df = self._add_volatility_indicators(df)
            logger.debug("✓ Volatility indicators added")

            # Volume indicators
            df = self._add_volume_indicators(df)
            logger.debug("✓ Volume indicators added")

            # Support/Resistance levels
            df = self._add_support_resistance(df)
            logger.debug("✓ Support/Resistance added")

            # Pattern recognition
            df = self._add_patterns(df)
            logger.debug("✓ Patterns added")

            # Statistical features
            df = self._add_statistical_features(df)
            logger.debug("✓ Statistical features added")

            # Market microstructure
            df = self._add_microstructure_features(df)
            logger.debug("✓ Microstructure features added")

            # Drop NaN and infinite values
            original_len = len(df)
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.dropna()

            dropped = original_len - len(df)
            if dropped > 0:
                logger.info(f"Dropped {dropped} rows with NaN/Inf values")

            # Get list of feature columns (exclude OHLCV and timestamp)
            base_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            self.feature_names = [col for col in df.columns if col not in base_cols]

            logger.info(f"✓ Generated {len(self.feature_names)} features from {len(df)} candles")

            return df

        except Exception as e:
            logger.error(f"Error calculating features: {e}")
            raise

    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic price-based features with error handling"""
        try:
            # Price changes
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

            # Price ranges
            df['high_low_range'] = (df['high'] - df['low']) / df['close']
            df['close_open_range'] = (df['close'] - df['open']) / df['open']

            # Body and shadow ratios
            df['body_size'] = abs(df['close'] - df['open']) / df['close']
            df['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['close']
            df['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['close']

            # Price momentum (multiple periods)
            for period in [5, 10, 20, 50]:
                df[f'price_momentum_{period}'] = df['close'].pct_change(period)

            return df
        except Exception as e:
            logger.warning(f"Error in price features: {e}")
            return df

    def _add_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Moving averages and trend indicators"""
        try:
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values

            # Moving Averages
            for period in [7, 14, 21, 50, 100, 200]:
                if len(df) >= period:
                    df[f'sma_{period}'] = talib.SMA(close, timeperiod=period)
                    df[f'ema_{period}'] = talib.EMA(close, timeperiod=period)

            # MACD
            if len(df) >= 26:
                macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
                df['macd'] = macd
                df['macd_signal'] = macd_signal
                df['macd_hist'] = macd_hist

            # ADX (Trend Strength)
            if len(df) >= 14:
                df['adx'] = talib.ADX(high, low, close, timeperiod=14)
                df['plus_di'] = talib.PLUS_DI(high, low, close, timeperiod=14)
                df['minus_di'] = talib.MINUS_DI(high, low, close, timeperiod=14)

            # Parabolic SAR
            if len(df) >= 5:
                df['sar'] = talib.SAR(high, low, acceleration=0.02, maximum=0.2)

            return df
        except Exception as e:
            logger.warning(f"Error in trend indicators: {e}")
            return df

    def _add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """RSI, Stochastic, and other momentum indicators"""
        try:
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values

            # RSI (multiple periods)
            for period in [9, 14, 21]:
                if len(df) >= period:
                    df[f'rsi_{period}'] = talib.RSI(close, timeperiod=period)

            # Stochastic Oscillator
            if len(df) >= 14:
                slowk, slowd = talib.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowd_period=3)
                df['stoch_k'] = slowk
                df['stoch_d'] = slowd

            # Williams %R
            if len(df) >= 14:
                df['willr'] = talib.WILLR(high, low, close, timeperiod=14)

            # CCI (Commodity Channel Index)
            if len(df) >= 14:
                df['cci'] = talib.CCI(high, low, close, timeperiod=14)

            # MOM (Momentum)
            if len(df) >= 10:
                df['mom'] = talib.MOM(close, timeperiod=10)

            # ROC (Rate of Change)
            for period in [10, 20]:
                if len(df) >= period:
                    df[f'roc_{period}'] = talib.ROC(close, timeperiod=period)

            return df
        except Exception as e:
            logger.warning(f"Error in momentum indicators: {e}")
            return df

    def _add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Bollinger Bands, ATR, and volatility measures"""
        try:
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values

            # Bollinger Bands
            if len(df) >= 20:
                upper, middle, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
                df['bb_upper'] = upper
                df['bb_middle'] = middle
                df['bb_lower'] = lower
                df['bb_width'] = (upper - lower) / middle
                df['bb_position'] = (close - lower) / (upper - lower)

            # ATR (Average True Range)
            for period in [7, 14, 21]:
                if len(df) >= period:
                    df[f'atr_{period}'] = talib.ATR(high, low, close, timeperiod=period)

            # NATR (Normalized ATR)
            if len(df) >= 14:
                df['natr'] = talib.NATR(high, low, close, timeperiod=14)

            # Historical Volatility
            for period in [10, 20, 30]:
                if len(df) >= period:
                    df[f'volatility_{period}'] = df['returns'].rolling(period).std() * np.sqrt(period)

            return df
        except Exception as e:
            logger.warning(f"Error in volatility indicators: {e}")
            return df

    def _add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Volume-based indicators"""
        try:
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            volume = df['volume'].values

            # Volume SMA
            for period in [10, 20, 50]:
                if len(df) >= period:
                    df[f'volume_sma_{period}'] = talib.SMA(volume, timeperiod=period)
                    df[f'volume_ratio_{period}'] = volume / df[f'volume_sma_{period}']

            # OBV (On-Balance Volume)
            if len(df) >= 1:
                df['obv'] = talib.OBV(close, volume)

            # AD (Accumulation/Distribution)
            if len(df) >= 1:
                df['ad'] = talib.AD(high, low, close, volume)

            # ADOSC (Accumulation/Distribution Oscillator)
            if len(df) >= 10:
                df['adosc'] = talib.ADOSC(high, low, close, volume, fastperiod=3, slowperiod=10)

            # MFI (Money Flow Index)
            if len(df) >= 14:
                df['mfi'] = talib.MFI(high, low, close, volume, timeperiod=14)

            return df
        except Exception as e:
            logger.warning(f"Error in volume indicators: {e}")
            return df

    def _add_support_resistance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Support and resistance levels"""
        try:
            # Rolling highs and lows
            for period in [20, 50, 100]:
                if len(df) >= period:
                    df[f'rolling_high_{period}'] = df['high'].rolling(period).max()
                    df[f'rolling_low_{period}'] = df['low'].rolling(period).min()
                    df[f'distance_to_high_{period}'] = (df[f'rolling_high_{period}'] - df['close']) / df['close']
                    df[f'distance_to_low_{period}'] = (df['close'] - df[f'rolling_low_{period}']) / df['close']

            return df
        except Exception as e:
            logger.warning(f"Error in support/resistance: {e}")
            return df

    def _add_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Candlestick pattern recognition"""
        try:
            open_price = df['open'].values
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values

            # Major candlestick patterns
            patterns = {
                'doji': talib.CDLDOJI,
                'hammer': talib.CDLHAMMER,
                'shooting_star': talib.CDLSHOOTINGSTAR,
                'engulfing': talib.CDLENGULFING,
                'harami': talib.CDLHARAMI,
                'morning_star': talib.CDLMORNINGSTAR,
                'evening_star': talib.CDLEVENINGSTAR,
            }

            for name, func in patterns.items():
                try:
                    df[f'pattern_{name}'] = func(open_price, high, low, close)
                except:
                    pass

            return df
        except Exception as e:
            logger.warning(f"Error in pattern recognition: {e}")
            return df

    def _add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Statistical measures"""
        try:
            # Rolling statistics
            for period in [10, 20, 50]:
                if len(df) >= period:
                    df[f'skew_{period}'] = df['returns'].rolling(period).skew()
                    df[f'kurtosis_{period}'] = df['returns'].rolling(period).kurt()

            # Z-score
            for period in [20, 50]:
                if len(df) >= period:
                    mean = df['close'].rolling(period).mean()
                    std = df['close'].rolling(period).std()
                    df[f'zscore_{period}'] = (df['close'] - mean) / std

            return df
        except Exception as e:
            logger.warning(f"Error in statistical features: {e}")
            return df

    def _add_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Market microstructure features"""
        try:
            # Spread measures
            df['spread'] = (df['high'] - df['low']) / df['close']

            # Price pressure
            df['price_pressure'] = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-10)

            # Volume price trend
            if len(df) >= 2:
                price_change = df['close'].pct_change()
                df['vpt'] = (price_change * df['volume']).cumsum()

            return df
        except Exception as e:
            logger.warning(f"Error in microstructure features: {e}")
            return df

    def get_feature_names(self) -> List[str]:
        """Get list of all feature names"""
        return self.feature_names


if __name__ == "__main__":
    # Test feature engineering
    from binance_data_fetcher import BinanceDataFetcher

    print("=" * 70)
    print("  Testing Feature Engineering")
    print("=" * 70)

    # Fetch data
    print("\n1. Fetching data from Binance...")
    fetcher = BinanceDataFetcher()
    klines = fetcher.get_klines("BTCUSDT", "5m", 500)
    df = fetcher.klines_to_dataframe(klines)
    print(f"   ✓ Fetched {len(df)} candles")

    # Calculate features
    print("\n2. Calculating features...")
    engine = FeatureEngine()
    df_prepared = engine.prepare_data_from_binance(df)
    df_features = engine.calculate_all_features(df_prepared)

    print(f"   ✓ Generated {len(engine.get_feature_names())} features")
    print(f"   ✓ Final dataset: {len(df_features)} rows × {len(df_features.columns)} columns")

    # Show sample
    print("\n3. Sample feature values (latest candle):")
    features = engine.get_feature_names()[:10]  # Show first 10 features
    for feat in features:
        print(f"   {feat:.<30} {df_features[feat].iloc[-1]:.4f}")

    print("\n" + "=" * 70)
    print("  Feature engineering test completed!")
    print("=" * 70)
