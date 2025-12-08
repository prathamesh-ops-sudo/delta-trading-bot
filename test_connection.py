"""
Test Script - Verify all connections and components
Run this before starting the trading bot
"""
import sys
from datetime import datetime

# Setup basic logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)


def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def test_imports():
    """Test if all required packages are installed"""
    print_header("Testing Package Imports")

    packages = {
        'numpy': 'numpy',
        'pandas': 'pandas',
        'sklearn': 'scikit-learn',
        'tensorflow': 'tensorflow',
        'talib': 'TA-Lib',
        'requests': 'requests',
        'schedule': 'schedule'
    }

    all_good = True

    for module, package in packages.items():
        try:
            __import__(module)
            print(f"✓ {package:20s} - OK")
        except ImportError as e:
            print(f"✗ {package:20s} - FAILED: {e}")
            all_good = False

    return all_good


def test_config():
    """Test configuration"""
    print_header("Testing Configuration")

    try:
        from config import Config

        print(f"✓ Configuration loaded")
        print(f"  - Symbol: {Config.SYMBOL}")
        print(f"  - Timeframe: {Config.TIMEFRAME}")
        print(f"  - Max Leverage: {Config.MAX_LEVERAGE}x")
        print(f"  - Signal Threshold: {Config.SIGNAL_THRESHOLD:.0%}")
        print(f"  - API Key: {Config.DELTA_API_KEY[:10]}...")

        Config.validate_config()
        print(f"✓ Configuration valid")

        return True

    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False


def test_delta_api():
    """Test Delta Exchange API connection"""
    print_header("Testing Delta Exchange API")

    try:
        from delta_exchange_api import DeltaExchangeAPI
        from config import Config

        api = DeltaExchangeAPI()

        # Test public endpoint (no auth required)
        print("Testing public API endpoint...")
        ticker = api.get_ticker(Config.SYMBOL)

        if ticker:
            price = ticker.get('result', {}).get('mark_price', 0)
            print(f"✓ Public API - OK")
            print(f"  - Current {Config.SYMBOL} price: ${float(price):,.2f}")
        else:
            print(f"✗ Public API - Failed to get ticker")
            return False

        # Test authenticated endpoint
        print("\nTesting authenticated API endpoint...")
        balance = api.get_account_balance()

        if balance >= 0:
            print(f"✓ Authenticated API - OK")
            print(f"  - Account balance: ${balance:,.2f}")
        else:
            print(f"⚠ Authenticated API - Warning: Balance is 0 or negative")

        # Test candle data
        print("\nTesting historical data...")
        candles = api.get_candles(Config.SYMBOL, Config.TIMEFRAME, count=10)

        if candles and len(candles) > 0:
            print(f"✓ Historical data - OK")
            print(f"  - Fetched {len(candles)} candles")
        else:
            print(f"✗ Historical data - Failed")
            return False

        return True

    except Exception as e:
        print(f"✗ Delta API test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_telegram():
    """Test Telegram bot connection"""
    print_header("Testing Telegram Integration")

    try:
        from telegram_notifier import TelegramNotifier

        telegram = TelegramNotifier()

        if not telegram.enabled:
            print("⚠ Telegram not configured (bot token or chat ID missing)")
            print("  This is optional but recommended for notifications")
            return True

        print("Testing Telegram connection...")
        result = telegram.test_connection()

        if result:
            print("✓ Telegram - OK")
            print("  Check your Telegram for test message")
            return True
        else:
            print("✗ Telegram - Failed to send message")
            return False

    except Exception as e:
        print(f"✗ Telegram test failed: {e}")
        return False


def test_ml_models():
    """Test ML model loading/creation"""
    print_header("Testing ML Models")

    try:
        from ml_models import TradingMLModels

        models = TradingMLModels()

        # Try to load existing models
        if models.load_models():
            print("✓ Pre-trained models loaded successfully")
            print("  - LSTM model: OK")
            print("  - Random Forest model: OK")
            print("  - Scaler: OK")
        else:
            print("⚠ No pre-trained models found")
            print("  Models will be trained on first bot run (this takes 5-10 minutes)")

        return True

    except Exception as e:
        print(f"✗ ML models test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_feature_engineering():
    """Test feature engineering"""
    print_header("Testing Feature Engineering")

    try:
        from feature_engineering import FeatureEngine
        from delta_exchange_api import DeltaExchangeAPI
        from config import Config

        api = DeltaExchangeAPI()
        feature_engine = FeatureEngine()

        print("Fetching sample data...")
        candles = api.get_candles(Config.SYMBOL, Config.TIMEFRAME, count=200)

        if not candles:
            print("✗ Failed to fetch candles")
            return False

        print(f"Processing {len(candles)} candles...")
        df = feature_engine.prepare_data(candles)
        df_features = feature_engine.calculate_all_features(df)

        feature_cols = feature_engine.get_feature_columns(df_features)

        print(f"✓ Feature engineering - OK")
        print(f"  - Input candles: {len(candles)}")
        print(f"  - After processing: {len(df_features)} rows")
        print(f"  - Features calculated: {len(feature_cols)}")

        return True

    except Exception as e:
        print(f"✗ Feature engineering test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_risk_manager():
    """Test risk management"""
    print_header("Testing Risk Manager")

    try:
        from risk_manager import RiskManager

        risk_mgr = RiskManager()

        # Test position sizing
        position_info = risk_mgr.calculate_position_size(
            account_balance=1000,
            current_price=50000,
            stop_loss_pct=0.02,
            signal_confidence=0.75
        )

        print("✓ Risk manager - OK")
        print(f"  Sample position (1000 USD balance, 75% confidence):")
        print(f"  - Contract size: {position_info['contract_size']}")
        print(f"  - Leverage: {position_info['leverage']}x")
        print(f"  - Risk: ${position_info['risk_amount']} ({position_info['risk_percentage']}%)")

        return True

    except Exception as e:
        print(f"✗ Risk manager test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("  DELTA EXCHANGE TRADING BOT - CONNECTION TEST")
    print("  " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print("=" * 70)

    results = {}

    # Run tests
    results['Imports'] = test_imports()
    results['Configuration'] = test_config()
    results['Delta API'] = test_delta_api()
    results['Telegram'] = test_telegram()
    results['ML Models'] = test_ml_models()
    results['Feature Engineering'] = test_feature_engineering()
    results['Risk Manager'] = test_risk_manager()

    # Summary
    print_header("Test Summary")

    all_passed = True
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:10s} - {test_name}")

        if not result and test_name not in ['Telegram', 'ML Models']:
            all_passed = False

    print("\n" + "=" * 70)

    if all_passed:
        print("✓ ALL CRITICAL TESTS PASSED")
        print("\nYou can now run the trading bot with:")
        print("  python trading_bot.py")
    else:
        print("✗ SOME TESTS FAILED")
        print("\nPlease fix the issues before running the trading bot")
        print("Check the error messages above for details")

    print("=" * 70 + "\n")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
