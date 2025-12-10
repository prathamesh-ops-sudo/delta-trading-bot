"""
ML-based Signal Generator for Crypto Alerts
Uses LSTM + Random Forest ensemble for buy/sell signals
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple
import pickle
import os
import logging
from datetime import datetime

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from config import Config

logger = logging.getLogger(__name__)


class SignalGenerator:
    """Generate trading signals using LSTM + Random Forest ensemble"""

    def __init__(self):
        self.lstm_model = None
        self.rf_model = None
        self.scaler = None
        self.feature_names = None

    def build_lstm_model(self, input_shape: Tuple[int, int]) -> keras.Model:
        """
        Build Bidirectional LSTM model for time series prediction

        Args:
            input_shape: (lookback_period, num_features)

        Returns:
            Compiled Keras model
        """
        model = Sequential([
            Bidirectional(LSTM(128, return_sequences=True), input_shape=input_shape),
            Dropout(0.3),
            BatchNormalization(),

            Bidirectional(LSTM(64, return_sequences=True)),
            Dropout(0.3),
            BatchNormalization(),

            LSTM(32, return_sequences=False),
            Dropout(0.2),
            BatchNormalization(),

            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),

            Dense(1, activation='sigmoid')  # Output: 0 (sell) to 1 (buy)
        ])

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )

        logger.info(f"✓ Built LSTM model with input shape {input_shape}")
        return model

    def build_rf_model(self, n_estimators: int = 200) -> RandomForestClassifier:
        """Build Random Forest classifier"""
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=20,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )

        logger.info(f"✓ Built Random Forest model with {n_estimators} trees")
        return model

    def prepare_training_data(self, df: pd.DataFrame, feature_cols: list) -> Tuple:
        """
        Prepare data for training

        Args:
            df: DataFrame with features and price data
            feature_cols: List of feature column names

        Returns:
            X_lstm, X_rf, y: Training data for LSTM and RF models
        """
        # Create labels based on future returns
        future_periods = 12  # Look 12 periods (1 hour for 5m candles) ahead
        df['future_returns'] = df['close'].shift(-future_periods) / df['close'] - 1
        df['label'] = (df['future_returns'] > 0.01).astype(int)  # 1% threshold for buy signal

        # Drop rows with NaN labels
        df = df.dropna(subset=['label'])

        # Extract features
        X = df[feature_cols].values
        y = df['label'].values

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Prepare LSTM input (3D: samples, timesteps, features)
        X_lstm = self._create_sequences(X_scaled, Config.LSTM_LOOKBACK)

        # Align RF data with LSTM data
        X_rf = X_scaled[Config.LSTM_LOOKBACK:]
        y_aligned = y[Config.LSTM_LOOKBACK:]

        logger.info(f"✓ Prepared training data:")
        logger.info(f"  LSTM input shape: {X_lstm.shape}")
        logger.info(f"  RF input shape: {X_rf.shape}")
        logger.info(f"  Labels: {len(y_aligned)} samples, {y_aligned.sum()} positive")

        return X_lstm, X_rf, y_aligned

    def _create_sequences(self, data: np.ndarray, lookback: int) -> np.ndarray:
        """Create sequences for LSTM input"""
        sequences = []
        for i in range(lookback, len(data)):
            sequences.append(data[i - lookback:i])
        return np.array(sequences)

    def train(self, df: pd.DataFrame, feature_cols: list, epochs: int = 50, batch_size: int = 32):
        """
        Train both LSTM and Random Forest models

        Args:
            df: DataFrame with features
            feature_cols: List of feature column names
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        logger.info("Starting model training...")

        # Prepare data
        X_lstm, X_rf, y = self.prepare_training_data(df, feature_cols)

        # Split data (80% train, 20% validation)
        split_idx = int(0.8 * len(y))

        X_lstm_train, X_lstm_val = X_lstm[:split_idx], X_lstm[split_idx:]
        X_rf_train, X_rf_val = X_rf[:split_idx], X_rf[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # Train LSTM
        logger.info("Training LSTM model...")
        self.lstm_model = self.build_lstm_model((Config.LSTM_LOOKBACK, X_lstm.shape[2]))

        history = self.lstm_model.fit(
            X_lstm_train, y_train,
            validation_data=(X_lstm_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=0
        )

        lstm_val_acc = history.history['val_accuracy'][-1]
        logger.info(f"✓ LSTM training completed. Val accuracy: {lstm_val_acc:.4f}")

        # Train Random Forest
        logger.info("Training Random Forest model...")
        self.rf_model = self.build_rf_model()
        self.rf_model.fit(X_rf_train, y_train)

        rf_val_acc = self.rf_model.score(X_rf_val, y_val)
        logger.info(f"✓ Random Forest training completed. Val accuracy: {rf_val_acc:.4f}")

        # Store feature names
        self.feature_names = feature_cols

    def predict(self, df: pd.DataFrame) -> Dict:
        """
        Generate trading signal for the latest data point

        Args:
            df: DataFrame with features (should have at least LSTM_LOOKBACK rows)

        Returns:
            Dictionary with signal information
        """
        if self.lstm_model is None or self.rf_model is None:
            raise ValueError("Models not trained or loaded")

        # Get feature columns
        feature_cols = self.feature_names

        # Extract and scale features
        X = df[feature_cols].values
        X_scaled = self.scaler.transform(X)

        # Prepare LSTM input (last lookback periods)
        if len(X_scaled) < Config.LSTM_LOOKBACK:
            raise ValueError(f"Need at least {Config.LSTM_LOOKBACK} candles for prediction")

        X_lstm = X_scaled[-Config.LSTM_LOOKBACK:].reshape(1, Config.LSTM_LOOKBACK, -1)
        X_rf = X_scaled[-1:].reshape(1, -1)

        # Get predictions
        lstm_pred = self.lstm_model.predict(X_lstm, verbose=0)[0][0]
        rf_pred_proba = self.rf_model.predict_proba(X_rf)[0][1]

        # Ensemble prediction (weighted average)
        # Score represents probability of UPWARD movement (0 = strong down, 1 = strong up)
        ensemble_score = 0.6 * lstm_pred + 0.4 * rf_pred_proba

        # Determine signal with clear thresholds
        if ensemble_score >= Config.BUY_SIGNAL_THRESHOLD:
            # High score (>= 70%) = Strong upward probability = BUY
            signal = 'BUY'
            signal_confidence = ensemble_score  # 70-100%
        elif ensemble_score <= (1 - Config.SELL_SIGNAL_THRESHOLD):
            # Low score (<= 30%) = Strong downward probability = SELL
            signal = 'SELL'
            signal_confidence = 1 - ensemble_score  # Convert to downward confidence (70-100%)
        else:
            # Middle range (30-70%) = Uncertain = NEUTRAL
            signal = 'NEUTRAL'
            signal_confidence = 0.5  # Neutral confidence

        # Get current market data
        latest = df.iloc[-1]

        result = {
            'signal': signal,
            'confidence': float(signal_confidence),  # Now shows actual signal strength
            'ensemble_score': float(ensemble_score),  # Raw score (0-1)
            'lstm_score': float(lstm_pred),
            'rf_score': float(rf_pred_proba),
            'price': float(latest['close']),
            'timestamp': latest['timestamp'],
            'rsi_14': float(latest.get('rsi_14', 0)),
            'macd_hist': float(latest.get('macd_hist', 0)),
            'bb_position': float(latest.get('bb_position', 0.5))
        }

        logger.info(f"Signal: {signal} | Confidence: {signal_confidence:.2%} | Ensemble: {ensemble_score:.2%} | Price: ${result['price']:,.2f}")

        return result

    def save_models(self):
        """Save trained models to disk"""
        os.makedirs('models', exist_ok=True)

        # Save LSTM
        self.lstm_model.save(Config.LSTM_MODEL_PATH)
        logger.info(f"✓ Saved LSTM model to {Config.LSTM_MODEL_PATH}")

        # Save Random Forest
        with open(Config.RF_MODEL_PATH, 'wb') as f:
            pickle.dump(self.rf_model, f)
        logger.info(f"✓ Saved RF model to {Config.RF_MODEL_PATH}")

        # Save scaler and feature names
        with open(Config.SCALER_PATH, 'wb') as f:
            pickle.dump({'scaler': self.scaler, 'feature_names': self.feature_names}, f)
        logger.info(f"✓ Saved scaler to {Config.SCALER_PATH}")

    def load_models(self):
        """Load trained models from disk"""
        if not os.path.exists(Config.LSTM_MODEL_PATH):
            raise FileNotFoundError(f"LSTM model not found at {Config.LSTM_MODEL_PATH}")

        if not os.path.exists(Config.RF_MODEL_PATH):
            raise FileNotFoundError(f"RF model not found at {Config.RF_MODEL_PATH}")

        if not os.path.exists(Config.SCALER_PATH):
            raise FileNotFoundError(f"Scaler not found at {Config.SCALER_PATH}")

        # Load LSTM
        self.lstm_model = load_model(Config.LSTM_MODEL_PATH)
        logger.info(f"✓ Loaded LSTM model from {Config.LSTM_MODEL_PATH}")

        # Load Random Forest
        with open(Config.RF_MODEL_PATH, 'rb') as f:
            self.rf_model = pickle.load(f)
        logger.info(f"✓ Loaded RF model from {Config.RF_MODEL_PATH}")

        # Load scaler and feature names
        with open(Config.SCALER_PATH, 'rb') as f:
            data = pickle.load(f)
            self.scaler = data['scaler']
            self.feature_names = data['feature_names']
        logger.info(f"✓ Loaded scaler from {Config.SCALER_PATH}")

    def models_exist(self) -> bool:
        """Check if trained models exist"""
        return (
            os.path.exists(Config.LSTM_MODEL_PATH) and
            os.path.exists(Config.RF_MODEL_PATH) and
            os.path.exists(Config.SCALER_PATH)
        )


if __name__ == "__main__":
    # Setup basic logging for testing
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )

    # Test signal generator
    from binance_data_fetcher import BinanceDataFetcher
    from feature_engineering import FeatureEngine

    print("=" * 70)
    print("  Testing Signal Generator")
    print("=" * 70)

    # Fetch data
    print("\n1. Fetching data...")
    fetcher = BinanceDataFetcher()
    klines = fetcher.get_klines("BTCUSDT", "5m", 500)
    df = fetcher.klines_to_dataframe(klines)

    # Calculate features
    print("\n2. Calculating features...")
    engine = FeatureEngine()
    df = engine.prepare_data_from_binance(df)
    df = engine.calculate_all_features(df)
    feature_cols = engine.get_feature_names()

    # Train models
    print("\n3. Training models (this may take a few minutes)...")
    signal_gen = SignalGenerator()
    signal_gen.train(df, feature_cols, epochs=10, batch_size=32)

    # Save models
    print("\n4. Saving models...")
    signal_gen.save_models()

    # Test prediction
    print("\n5. Testing prediction...")
    signal = signal_gen.predict(df)
    print(f"\n   Signal: {signal['signal']}")
    print(f"   Confidence: {signal['confidence']:.2%}")
    print(f"   LSTM Score: {signal['lstm_score']:.2%}")
    print(f"   RF Score: {signal['rf_score']:.2%}")
    print(f"   Price: ${signal['price']:,.2f}")

    print("\n" + "=" * 70)
    print("  Signal generator test completed!")
    print("=" * 70)
