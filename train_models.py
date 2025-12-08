"""
Manual Model Training Script
Train ML models manually before running the bot
"""
import logging
from datetime import datetime

from logger_config import setup_logging
from delta_exchange_api import DeltaExchangeAPI
from feature_engineering import FeatureEngine
from ml_models import TradingMLModels
from config import Config

# Setup logging
setup_logging(log_file="model_training.log")
logger = logging.getLogger(__name__)


def main():
    """Train models on historical data"""
    logger.info("=" * 80)
    logger.info("MANUAL MODEL TRAINING")
    logger.info("=" * 80)

    try:
        # Initialize components
        logger.info("Initializing components...")
        api = DeltaExchangeAPI()
        feature_engine = FeatureEngine()
        ml_models = TradingMLModels()

        # Fetch historical data
        logger.info(f"Fetching {Config.CANDLES_TO_FETCH} historical candles for {Config.SYMBOL}...")
        candles = api.get_candles(
            symbol=Config.SYMBOL,
            resolution=Config.TIMEFRAME,
            count=Config.CANDLES_TO_FETCH
        )

        if not candles or len(candles) < Config.LSTM_LOOKBACK:
            logger.error(f"Insufficient data: got {len(candles) if candles else 0} candles, need at least {Config.LSTM_LOOKBACK}")
            return False

        logger.info(f"✓ Fetched {len(candles)} candles")

        # Prepare data
        logger.info("Preparing data and calculating features...")
        df = feature_engine.prepare_data(candles)
        df_with_features = feature_engine.calculate_all_features(df)

        feature_cols = feature_engine.get_feature_columns(df_with_features)
        logger.info(f"✓ Prepared {len(df_with_features)} candles with {len(feature_cols)} features")

        # Train models
        logger.info("=" * 80)
        logger.info("Starting model training (this will take several minutes)...")
        logger.info("=" * 80)

        results = ml_models.train_all_models(df_with_features)

        # Display results
        logger.info("=" * 80)
        logger.info("TRAINING RESULTS")
        logger.info("=" * 80)

        logger.info(f"\nLSTM Model:")
        logger.info(f"  Training Accuracy:   {results['lstm']['train_accuracy']:.4f}")
        logger.info(f"  Validation Accuracy: {results['lstm']['val_accuracy']:.4f}")
        logger.info(f"  Training Precision:  {results['lstm']['train_precision']:.4f}")
        logger.info(f"  Validation Precision: {results['lstm']['val_precision']:.4f}")
        logger.info(f"  Training Recall:     {results['lstm']['train_recall']:.4f}")
        logger.info(f"  Validation Recall:   {results['lstm']['val_recall']:.4f}")

        logger.info(f"\nRandom Forest Model:")
        logger.info(f"  Training Accuracy:   {results['random_forest']['train_accuracy']:.4f}")
        logger.info(f"  Validation Accuracy: {results['random_forest']['val_accuracy']:.4f}")
        logger.info(f"  Training Precision:  {results['random_forest']['train_precision']:.4f}")
        logger.info(f"  Validation Precision: {results['random_forest']['val_precision']:.4f}")
        logger.info(f"  Training Recall:     {results['random_forest']['train_recall']:.4f}")
        logger.info(f"  Validation Recall:   {results['random_forest']['val_recall']:.4f}")

        logger.info(f"\nGradient Boosting Model:")
        logger.info(f"  Training Accuracy:   {results['gradient_boosting']['train_accuracy']:.4f}")
        logger.info(f"  Validation Accuracy: {results['gradient_boosting']['val_accuracy']:.4f}")

        logger.info(f"\nTraining Info:")
        logger.info(f"  Training Samples:   {results['training_samples']}")
        logger.info(f"  Validation Samples: {results['validation_samples']}")
        logger.info(f"  Timestamp:          {results['timestamp']}")

        logger.info("=" * 80)
        logger.info("✓ MODEL TRAINING COMPLETE")
        logger.info("=" * 80)

        logger.info("\nModels saved to:")
        logger.info(f"  - {Config.LSTM_MODEL_PATH}")
        logger.info(f"  - {Config.RF_MODEL_PATH}")
        logger.info(f"  - {Config.SCALER_PATH}")

        logger.info("\nYou can now run the trading bot with:")
        logger.info("  python trading_bot.py")

        return True

    except Exception as e:
        logger.error(f"Model training failed: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
