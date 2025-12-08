# Automated Cryptocurrency Trading Bot for Delta Exchange

## Overview
A professional, production-ready automated trading system for BTC/USD on Delta Exchange India. The system uses advanced machine learning (LSTM + Random Forest ensemble) for signal generation, comprehensive risk management, and intelligent trade monitoring.

## Features

### Machine Learning & Analytics
- **LSTM Neural Network**: Bidirectional LSTM for time-series price prediction
- **Random Forest Classifier**: Ensemble decision-making for robust signals
- **50+ Technical Indicators**: Comprehensive feature engineering including:
  - Trend indicators (SMA, EMA, MACD, ADX, Ichimoku)
  - Momentum oscillators (RSI, Stochastic, CCI, MFI)
  - Volatility measures (Bollinger Bands, ATR, Keltner Channels)
  - Volume analysis (OBV, CMF, Force Index)
  - Pattern recognition (candlestick patterns)

### Risk Management
- **Dynamic Position Sizing**: Based on signal confidence and account balance
- **Multi-Level Take Profit**: Partial exits at 3%, 5%, and 8% profit levels
- **Stop Loss Protection**: ATR-based or fixed percentage stop loss
- **Trailing Stop**: Locks in profits as price moves favorably
- **Daily Trade Limits**: Prevents overtrading
- **Risk Scoring System**: Evaluates trade quality before execution

### Trade Monitoring
- **Real-time Position Tracking**: Monitors all open positions every 30 seconds
- **Signal Strength Analysis**: Exits trades when ML signal weakens
- **Automatic Exit Management**: Handles stop loss, take profit, and weakness exits
- **PnL Tracking**: Real-time profit/loss calculation with ROI metrics

### Telegram Notifications
- Trade signal alerts with confidence scores
- Trade execution confirmations
- Position closure notifications with PnL
- Daily performance summaries
- Risk alerts and error notifications

## System Architecture

```
trading_bot.py              # Main orchestrator
├── delta_exchange_api.py   # API integration layer
├── feature_engineering.py  # Technical indicator calculation
├── ml_models.py            # LSTM + Random Forest models
├── risk_manager.py         # Position sizing & risk management
├── order_executor.py       # Order placement & management
├── trade_monitor.py        # Position monitoring & exit logic
├── telegram_notifier.py    # Telegram notifications
├── logger_config.py        # Logging configuration
└── config.py               # System configuration
```

## Configuration Parameters

### Trading Settings
- **Symbol**: BTCUSD
- **Timeframe**: 5 minutes
- **Max Leverage**: 5x (conservative, can go up to 20x)
- **Max Position Size**: 15% of account balance
- **Stop Loss**: 2% from entry
- **Take Profit**: 3%, 5%, 8% levels

### Frequency Control
- **Min Trade Interval**: 5 minutes between trades
- **Max Daily Trades**: 12 trades per day
- **Signal Threshold**: 65% minimum ML confidence

### ML Settings
- **LSTM Lookback**: 60 candles
- **Model Retraining**: Every 24 hours
- **Feature Window**: 100 candles for feature calculation

## Installation

### Prerequisites
- Python 3.8 or higher
- Delta Exchange India API credentials
- Telegram Bot Token (optional but recommended)

### Step 1: Install Dependencies

**Windows:**
```bash
pip install -r requirements.txt
```

**Important: TA-Lib Installation**

TA-Lib requires additional steps on Windows:

1. Download the appropriate `.whl` file from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
   - For Python 3.8: `TA_Lib‑0.4.28‑cp38‑cp38‑win_amd64.whl`
   - For Python 3.9: `TA_Lib‑0.4.28‑cp39‑cp39‑win_amd64.whl`
   - For Python 3.10: `TA_Lib‑0.4.28‑cp310‑cp310‑win_amd64.whl`

2. Install the downloaded wheel file:
```bash
pip install TA_Lib‑0.4.28‑cp38‑cp38‑win_amd64.whl
```

**Linux/Mac:**
```bash
# Install TA-Lib C library first
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install

# Then install Python dependencies
pip install -r requirements.txt
```

### Step 2: Configure API Credentials

The API credentials are already set in `config.py`. If you need to change them:

```python
# config.py
DELTA_API_KEY = "your_api_key_here"
DELTA_API_SECRET = "your_api_secret_here"
```

### Step 3: Set Up Telegram Bot (Optional but Recommended)

1. Create a Telegram bot:
   - Open Telegram and search for @BotFather
   - Send `/newbot` and follow instructions
   - Copy the bot token

2. Get your Chat ID:
   - Start a chat with your bot
   - Send any message
   - Visit: `https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates`
   - Find your chat ID in the response

3. Update `config.py`:
```python
TELEGRAM_BOT_TOKEN = "your_bot_token_here"
TELEGRAM_CHAT_ID = "your_chat_id_here"
```

### Step 4: Test the Bot

Before running the bot with real trading, test the connection:

```bash
python test_connection.py
```

## Running the Bot

### Start the Trading Bot

```bash
python trading_bot.py
```

### What Happens on First Run

1. **Model Training**: If no pre-trained models exist, the bot will:
   - Fetch 500 historical candles
   - Calculate 50+ technical indicators
   - Train LSTM and Random Forest models
   - This takes 5-10 minutes on first run

2. **Scheduled Tasks Setup**:
   - Trading cycle: Every 5 minutes
   - Position monitoring: Every 30 seconds
   - Model retraining check: Every hour
   - Daily report: At midnight
   - State saving: Every hour

3. **Continuous Operation**: The bot will:
   - Analyze market conditions
   - Generate ML-based trading signals
   - Execute trades when conditions are met
   - Monitor open positions
   - Send Telegram notifications

### Stopping the Bot

Press `Ctrl+C` to stop gracefully. The bot will:
- Save current state
- Send final performance report
- Optionally close open positions (commented out by default)

## Trading Strategy

### Signal Generation
1. **Feature Extraction**: Calculates 50+ technical indicators
2. **LSTM Prediction**: Analyzes price sequences for trend prediction
3. **Random Forest Classification**: Makes binary up/down predictions
4. **Ensemble Decision**: Combines both models with weighted average
5. **Confidence Filtering**: Only trades signals above 65% confidence

### Trade Execution
1. **Risk Assessment**: Evaluates signal strength, volatility, existing positions
2. **Position Sizing**: Calculates optimal size based on account balance and confidence
3. **Order Placement**: Places market order with calculated leverage
4. **Protection Orders**: Sets stop loss and multiple take profit levels

### Trade Monitoring
1. **Signal Strength Tracking**: Continuously reassesses ML signal alignment
2. **Weakness Detection**: Exits if signal drops below 35% strength
3. **Trailing Stop**: Locks in profits using ATR-based trailing
4. **Automatic Management**: Handles all stop loss and take profit executions

## Risk Parameters

### Conservative Settings (Default)
- Max Leverage: 5x
- Max Position: 15% of capital
- Stop Loss: 2%
- Min Confidence: 65%

### Moderate Settings
```python
# config.py
MAX_LEVERAGE = 7
MAX_POSITION_SIZE_PCT = 0.20  # 20%
SIGNAL_THRESHOLD = 0.60  # 60%
```

### Aggressive Settings (Higher Risk)
```python
# config.py
MAX_LEVERAGE = 10
MAX_POSITION_SIZE_PCT = 0.25  # 25%
SIGNAL_THRESHOLD = 0.55  # 55%
MAX_DAILY_TRADES = 20
```

## Cloud Deployment (Budget: 2,000 INR)

### Option 1: Google Cloud Free Tier (Recommended)
**Cost**: 0 INR/month (Free tier)

```bash
# Deploy on Google Cloud f1-micro instance
gcloud compute instances create trading-bot \
    --machine-type=f1-micro \
    --zone=asia-south1-a \
    --image-family=ubuntu-2004-lts \
    --image-project=ubuntu-os-cloud
```

### Option 2: Oracle Cloud Free Tier
**Cost**: 0 INR/month (Always free)
- 2 VM instances (1/8 OCPU, 1GB RAM each)

### Option 3: Lightsail
**Cost**: ~370 INR/month ($5)
- 512 MB RAM, 1 vCPU
- 20 GB SSD

### Deployment Steps

1. **Upload Files**:
```bash
scp -r trade_project/* user@your-server:/home/user/trading-bot/
```

2. **Install Dependencies**:
```bash
ssh user@your-server
cd trading-bot
pip install -r requirements.txt
```

3. **Run with Screen** (keeps running after disconnect):
```bash
screen -S trading-bot
python trading_bot.py
# Press Ctrl+A then D to detach
```

4. **Setup Systemd Service** (auto-restart):
```bash
sudo nano /etc/systemd/system/trading-bot.service
```

Add:
```ini
[Unit]
Description=Crypto Trading Bot
After=network.target

[Service]
Type=simple
User=your-user
WorkingDirectory=/home/user/trading-bot
ExecStart=/usr/bin/python3 trading_bot.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable:
```bash
sudo systemctl enable trading-bot
sudo systemctl start trading-bot
sudo systemctl status trading-bot
```

## Monitoring & Logs

### View Logs
```bash
tail -f trading_bot.log
```

### Check Performance
- Daily Telegram summaries
- Log file analysis
- Check `risk_manager_state.json` for trade history

## Safety Features

1. **Position Limits**: Maximum 1 open position at a time (configurable)
2. **Daily Trade Cap**: Prevents overtrading
3. **Stop Loss Protection**: Every trade has automatic stop loss
4. **Emergency Shutdown**: Ctrl+C gracefully closes everything
5. **State Persistence**: Saves state every hour
6. **Error Notifications**: Telegram alerts for critical errors

## Performance Metrics

The system tracks:
- Win rate
- Total PnL
- Average win/loss
- Profit factor
- Maximum drawdown
- Sharpe ratio (in logs)

## Troubleshooting

### Issue: TA-Lib import error
**Solution**: Follow TA-Lib installation steps in Installation section

### Issue: API connection failed
**Solution**: Check API credentials in config.py and network connectivity

### Issue: Models not training
**Solution**: Ensure enough historical data (500+ candles available)

### Issue: No trades executing
**Solution**:
- Check signal confidence threshold
- Verify account balance
- Check daily trade limits
- Review logs for risk assessment rejections

## Support Files

- `trading_bot.log`: Main application logs
- `risk_manager_state.json`: Trade history and state
- `models/`: Trained ML models
  - `lstm_model.h5`: LSTM neural network
  - `rf_model.pkl`: Random Forest classifier
  - `scaler.pkl`: Feature scaler

## Customization

### Adjust Trading Frequency
```python
# config.py
TIMEFRAME = "15m"  # Change to 15-minute candles
MIN_TRADE_INTERVAL = 900  # 15 minutes
```

### Modify Risk Parameters
```python
# config.py
STOP_LOSS_PCT = 0.015  # 1.5% stop loss
TAKE_PROFIT_LEVELS = [0.02, 0.04, 0.06]  # 2%, 4%, 6%
```

### Change Position Sizing Logic
Edit `risk_manager.py` in the `calculate_position_size()` method

## Disclaimer

This trading bot is for educational and research purposes. Cryptocurrency trading involves substantial risk of loss. Past performance does not guarantee future results. Only trade with capital you can afford to lose.

**Important Notes**:
- Always test with small amounts first
- Monitor the bot regularly
- Keep API credentials secure
- Understand the risks of leveraged trading
- Be aware of exchange fees and funding rates

## License

This project is provided as-is without warranty. Use at your own risk.

## Contact & Support

For issues with this bot:
- Check the logs first
- Review configuration settings
- Test components individually
- Monitor Telegram notifications

---

**Built with**: Python, TensorFlow, Scikit-learn, TA-Lib
**Exchange**: Delta Exchange India
**Strategy**: ML Ensemble (LSTM + Random Forest)
**Risk Management**: Multi-layered position sizing and protection
