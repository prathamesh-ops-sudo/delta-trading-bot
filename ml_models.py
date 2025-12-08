"""
Machine Learning Models for Trade Signal Generation
- LSTM for price prediction
- Random Forest for signal classification
- Ensemble approach for robust predictions
"""
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
import pickle
import logging
from datetime import datetime
import os

# ML/DL libraries
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from config import Config
from feature_engineering import FeatureEngine

logger = logging.getLogger(__name__)


class TradingMLModels:
    """ML/DL models for generating trading signals"""

    def __init__(self):
        self.lstm_model = None
        self.rf_model = None
        self.gb_model = None
        self.scaler = StandardScaler()
        self.feature_engine = FeatureEngine()

        # Create models directory
        os.makedirs('models', exist_ok=True)

    def build_lstm_model(self, input_shape: Tuple[int, int]) -> keras.Model:
        """
        Build LSTM model for price movement prediction

        Args:
            input_shape: (lookback_periods, num_features)
        """
        model = Sequential([
            # First LSTM layer
            Bidirectional(LSTM(128, return_sequences=True), input_shape=input_shape),
            Dropout(0.3),
            BatchNormalization(),

            # Second LSTM layer
            Bidirectional(LSTM(64, return_sequences=True)),
            Dropout(0.3),
            BatchNormalization(),

            # Third LSTM layer
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            BatchNormalization(),

            # Dense layers
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),

            # Output layer - binary classification (up/down)
            Dense(1, activation='sigmoid')
        ])

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )

        logger.info(f"LSTM model built with input shape: {input_shape}")
        return model

    def train_lstm(self, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray,
                   epochs: int = 50, batch_size: int = 32) -> Dict:
        """Train LSTM model"""
        logger.info("Training LSTM model...")

        # Build model
        self.lstm_model = self.build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))

        # Callbacks
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )

        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001,
            verbose=1
        )

        # Train
        history = self.lstm_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )

        # Evaluate
        train_loss, train_acc, train_prec, train_rec = self.lstm_model.evaluate(X_train, y_train, verbose=0)
        val_loss, val_acc, val_prec, val_rec = self.lstm_model.evaluate(X_val, y_val, verbose=0)

        results = {
            'train_accuracy': train_acc,
            'train_precision': train_prec,
            'train_recall': train_rec,
            'val_accuracy': val_acc,
            'val_precision': val_prec,
            'val_recall': val_rec,
            'history': history.history
        }

        logger.info(f"LSTM Training Complete - Val Accuracy: {val_acc:.4f}, Precision: {val_prec:.4f}")

        # Save model
        self.lstm_model.save(Config.LSTM_MODEL_PATH)
        logger.info(f"LSTM model saved to {Config.LSTM_MODEL_PATH}")

        return results

    def train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray,
                           X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        """Train Random Forest classifier"""
        logger.info("Training Random Forest model...")

        # Flatten sequences if needed (RF doesn't handle sequences)
        if len(X_train.shape) == 3:
            X_train_flat = X_train.reshape(X_train.shape[0], -1)
            X_val_flat = X_val.reshape(X_val.shape[0], -1)
        else:
            X_train_flat = X_train
            X_val_flat = X_val

        # Train model
        self.rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )

        self.rf_model.fit(X_train_flat, y_train)

        # Evaluate
        train_pred = self.rf_model.predict(X_train_flat)
        val_pred = self.rf_model.predict(X_val_flat)

        results = {
            'train_accuracy': accuracy_score(y_train, train_pred),
            'train_precision': precision_score(y_train, train_pred, zero_division=0),
            'train_recall': recall_score(y_train, train_pred, zero_division=0),
            'val_accuracy': accuracy_score(y_val, val_pred),
            'val_precision': precision_score(y_val, val_pred, zero_division=0),
            'val_recall': recall_score(y_val, val_pred, zero_division=0),
            'feature_importance': self.rf_model.feature_importances_
        }

        logger.info(f"RF Training Complete - Val Accuracy: {results['val_accuracy']:.4f}")

        # Save model
        with open(Config.RF_MODEL_PATH, 'wb') as f:
            pickle.dump(self.rf_model, f)
        logger.info(f"Random Forest model saved to {Config.RF_MODEL_PATH}")

        return results

    def train_gradient_boosting(self, X_train: np.ndarray, y_train: np.ndarray,
                               X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        """Train Gradient Boosting classifier"""
        logger.info("Training Gradient Boosting model...")

        # Flatten sequences if needed
        if len(X_train.shape) == 3:
            X_train_flat = X_train.reshape(X_train.shape[0], -1)
            X_val_flat = X_val.reshape(X_val.shape[0], -1)
        else:
            X_train_flat = X_train
            X_val_flat = X_val

        # Train model
        self.gb_model = GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=5,
            subsample=0.8,
            random_state=42
        )

        self.gb_model.fit(X_train_flat, y_train)

        # Evaluate
        train_pred = self.gb_model.predict(X_train_flat)
        val_pred = self.gb_model.predict(X_val_flat)

        results = {
            'train_accuracy': accuracy_score(y_train, train_pred),
            'val_accuracy': accuracy_score(y_val, val_pred),
            'val_precision': precision_score(y_val, val_pred, zero_division=0),
            'val_recall': recall_score(y_val, val_pred, zero_division=0),
        }

        logger.info(f"GB Training Complete - Val Accuracy: {results['val_accuracy']:.4f}")

        return results

    def load_models(self) -> bool:
        """Load pre-trained models"""
        try:
            # Load LSTM
            if os.path.exists(Config.LSTM_MODEL_PATH):
                self.lstm_model = load_model(Config.LSTM_MODEL_PATH)
                logger.info("LSTM model loaded successfully")

            # Load Random Forest
            if os.path.exists(Config.RF_MODEL_PATH):
                with open(Config.RF_MODEL_PATH, 'rb') as f:
                    self.rf_model = pickle.load(f)
                logger.info("Random Forest model loaded successfully")

            # Load Scaler
            if os.path.exists(Config.SCALER_PATH):
                with open(Config.SCALER_PATH, 'rb') as f:
                    self.scaler = pickle.load(f)
                logger.info("Scaler loaded successfully")

            return True

        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False

    def save_scaler(self):
        """Save the fitted scaler"""
        with open(Config.SCALER_PATH, 'wb') as f:
            pickle.dump(self.scaler, f)
        logger.info(f"Scaler saved to {Config.SCALER_PATH}")

    def predict_lstm(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict using LSTM model

        Returns:
            predictions: Binary predictions (0/1)
            probabilities: Confidence scores (0-1)
        """
        if self.lstm_model is None:
            raise ValueError("LSTM model not loaded")

        probabilities = self.lstm_model.predict(X, verbose=0)
        predictions = (probabilities > 0.5).astype(int)

        return predictions.flatten(), probabilities.flatten()

    def predict_rf(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict using Random Forest

        Returns:
            predictions: Binary predictions (0/1)
            probabilities: Confidence scores (0-1)
        """
        if self.rf_model is None:
            raise ValueError("Random Forest model not loaded")

        # Flatten if needed
        if len(X.shape) == 3:
            X_flat = X.reshape(X.shape[0], -1)
        else:
            X_flat = X

        predictions = self.rf_model.predict(X_flat)
        probabilities = self.rf_model.predict_proba(X_flat)[:, 1]

        return predictions, probabilities

    def ensemble_predict(self, X: np.ndarray) -> Dict[str, float]:
        """
        Ensemble prediction combining LSTM and RF

        Returns:
            Dictionary with signal, confidence, and individual model predictions
        """
        if self.lstm_model is None or self.rf_model is None:
            raise ValueError("Models not loaded")

        # Get predictions from both models
        lstm_pred, lstm_prob = self.predict_lstm(X)
        rf_pred, rf_prob = self.predict_rf(X)

        # Ensemble: weighted average of probabilities
        # LSTM weight: 0.6, RF weight: 0.4
        ensemble_prob = 0.6 * lstm_prob[-1] + 0.4 * rf_prob[-1]

        # Determine signal
        if ensemble_prob >= Config.SIGNAL_THRESHOLD:
            signal = 'BUY'
            confidence = ensemble_prob
        elif ensemble_prob <= (1 - Config.SIGNAL_THRESHOLD):
            signal = 'SELL'
            confidence = 1 - ensemble_prob
        else:
            signal = 'NEUTRAL'
            confidence = 0.5

        return {
            'signal': signal,
            'confidence': confidence,
            'ensemble_probability': ensemble_prob,
            'lstm_probability': lstm_prob[-1],
            'rf_probability': rf_prob[-1],
            'lstm_prediction': int(lstm_pred[-1]),
            'rf_prediction': int(rf_pred[-1])
        }

    def train_all_models(self, df: pd.DataFrame) -> Dict:
        """
        Train all models on historical data

        Args:
            df: DataFrame with OHLCV and calculated features
        """
        logger.info("Starting full model training pipeline...")

        # Create sequences for LSTM
        X_seq, y_seq = self.feature_engine.create_sequences(df, lookback=Config.LSTM_LOOKBACK)

        # Split data
        split_idx = int(len(X_seq) * 0.8)
        X_train_seq = X_seq[:split_idx]
        y_train_seq = y_seq[:split_idx]
        X_val_seq = X_seq[split_idx:]
        y_val_seq = y_seq[split_idx:]

        logger.info(f"Training data: {len(X_train_seq)} samples, Validation: {len(X_val_seq)} samples")

        # Scale data
        X_train_scaled = self._scale_sequences(X_train_seq, fit=True)
        X_val_scaled = self._scale_sequences(X_val_seq, fit=False)

        # Train LSTM
        lstm_results = self.train_lstm(X_train_scaled, y_train_seq, X_val_scaled, y_val_seq)

        # Train Random Forest
        rf_results = self.train_random_forest(X_train_scaled, y_train_seq, X_val_scaled, y_val_seq)

        # Train Gradient Boosting
        gb_results = self.train_gradient_boosting(X_train_scaled, y_train_seq, X_val_scaled, y_val_seq)

        # Save scaler
        self.save_scaler()

        return {
            'lstm': lstm_results,
            'random_forest': rf_results,
            'gradient_boosting': gb_results,
            'training_samples': len(X_train_seq),
            'validation_samples': len(X_val_seq),
            'timestamp': datetime.now().isoformat()
        }

    def _scale_sequences(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """Scale sequence data"""
        original_shape = X.shape
        X_reshaped = X.reshape(-1, X.shape[-1])

        if fit:
            X_scaled = self.scaler.fit_transform(X_reshaped)
        else:
            X_scaled = self.scaler.transform(X_reshaped)

        return X_scaled.reshape(original_shape)

    def prepare_prediction_data(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare latest data for prediction"""
        # Create sequence
        X_seq, _ = self.feature_engine.create_sequences(df, lookback=Config.LSTM_LOOKBACK)

        # Scale
        X_scaled = self._scale_sequences(X_seq, fit=False)

        # Return only the latest sequence
        return X_scaled[-1:] if len(X_scaled) > 0 else None
