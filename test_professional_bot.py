"""
Quick test script for professional trading bot
Tests all 4 Delta symbols with the new price action system
"""
import logging
from datetime import datetime

from config import Config
from binance_data_fetcher import BinanceDataFetcher
from feature_engineering import FeatureEngine
from professional_signal_generator import ProfessionalSignalGenerator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def test_professional_system():
    """Test the professional trading system"""
    print("=" * 80)
    print("  PROFESSIONAL TRADING BOT TEST")
    print("  Warren Buffett + BlackRock Aladdin Style")
    print("=" * 80)
    print()

    # Initialize components
    logger.info("Initializing components...")
    data_fetcher = BinanceDataFetcher()
    feature_engine = FeatureEngine()
    signal_gen = ProfessionalSignalGenerator()

    # Check if models exist
    if not signal_gen.models_exist():
        logger.warning("ML models not found - will need to train first")
        logger.info("Run alert_bot.py to train models on first startup")
        return False

    # Load models
    logger.info("Loading ML models (used as filters)...")
    signal_gen.load_models()
    logger.info("✓ ML models loaded")
    print()

    # Test each Delta symbol
    results = []
    for delta_symbol in Config.SYMBOLS:
        try:
            print("-" * 80)
            logger.info(f"Testing {delta_symbol} (Delta Exchange)")

            # Convert to Binance symbol
            binance_symbol = Config.get_binance_symbol(delta_symbol)
            logger.info(f"Fetching {binance_symbol} data (Binance proxy)...")

            # Fetch primary timeframe (15m)
            klines_primary = data_fetcher.get_klines(
                binance_symbol,
                Config.PRIMARY_TIMEFRAME,
                Config.CANDLES_TO_FETCH
            )
            df_primary = data_fetcher.klines_to_dataframe(klines_primary)
            logger.info(f"✓ Primary TF ({Config.PRIMARY_TIMEFRAME}): {len(df_primary)} candles")

            # Fetch higher timeframe (1h)
            klines_higher = data_fetcher.get_klines(
                binance_symbol,
                Config.HIGHER_TIMEFRAME,
                200
            )
            df_higher = data_fetcher.klines_to_dataframe(klines_higher)
            logger.info(f"✓ Higher TF ({Config.HIGHER_TIMEFRAME}): {len(df_higher)} candles")

            # Calculate features
            logger.info("Calculating features...")
            df_primary = feature_engine.prepare_data_from_binance(df_primary)
            df_primary = feature_engine.calculate_all_features(df_primary)

            df_higher = feature_engine.prepare_data_from_binance(df_higher)
            df_higher = feature_engine.calculate_all_features(df_higher)
            logger.info(f"✓ Features: {len(feature_engine.get_feature_names())} indicators")

            # Generate signal
            logger.info("Generating professional trading signal...")
            signal_data = signal_gen.predict(
                df_primary,
                higher_tf_df=df_higher,
                symbol=delta_symbol
            )

            # Display result
            print()
            print(f"📊 SIGNAL RESULT FOR {delta_symbol}:")
            print(f"   Signal:         {signal_data['signal']}")
            print(f"   Confidence:     {signal_data['confidence']:.0%}")
            print(f"   Current Price:  ${signal_data['price']:,.2f}")
            print(f"   Confluence:     {signal_data['confluence_count']} confirmations")
            print(f"   Market Structure: {signal_data['market_structure']}")
            print(f"   Higher TF Trend:  {signal_data['higher_tf_trend']}")

            if signal_data['signal'] != 'NEUTRAL':
                print()
                print(f"   TRADING SETUP:")
                print(f"   Entry:          ${signal_data['entry_price']:,.2f}")
                print(f"   Stop Loss:      ${signal_data['stop_loss']:,.2f}")
                print(f"   Take Profit:    ${signal_data['take_profit']:,.2f}")
                print(f"   Risk/Reward:    {signal_data['risk_reward']:.1f}:1")
                print()
                print(f"   RATIONALE:")
                for reason in signal_data.get('rationale', []):
                    print(f"   • {reason}")
                print()
                print(f"   CONFLUENCE FACTORS:")
                for factor in signal_data.get('confluence_details', []):
                    print(f"   • {factor}")

            results.append({
                'symbol': delta_symbol,
                'signal': signal_data['signal'],
                'confidence': signal_data['confidence'],
                'price': signal_data['price']
            })

            print()

        except Exception as e:
            logger.error(f"Error testing {delta_symbol}: {e}", exc_info=True)
            results.append({
                'symbol': delta_symbol,
                'signal': 'ERROR',
                'confidence': 0,
                'price': 0
            })

    # Summary
    print("=" * 80)
    print("  TEST SUMMARY")
    print("=" * 80)
    print()

    for result in results:
        status = "✅" if result['signal'] != 'ERROR' else "❌"
        print(f"{status} {result['symbol']:10} | {result['signal']:10} | {result['confidence']:.0%} | ${result['price']:,.2f}")

    print()

    # Signal stats
    stats = signal_gen.get_signal_stats()
    print(f"Signal Statistics:")
    print(f"  Total Signals:  {stats['total_signals']}")
    print(f"  Buy Signals:    {stats['buy_signals']} ({stats['buy_ratio']:.0%})")
    print(f"  Sell Signals:   {stats['sell_signals']} ({stats['sell_ratio']:.0%})")

    print()
    print("=" * 80)
    logger.info("✓ Test completed successfully!")
    print("=" * 80)

    return True


if __name__ == "__main__":
    try:
        success = test_professional_system()
        exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        exit(1)
