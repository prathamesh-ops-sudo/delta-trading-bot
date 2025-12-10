# 🤖 Crypto Alert Bot

**ML-Powered Cryptocurrency Trading Signals via Telegram**

An intelligent cryptocurrency monitoring system that uses Machine Learning (LSTM + Random Forest) to analyze market conditions and send real-time trading alerts directly to your Telegram.

---

## 🌟 Features

### 🧠 Machine Learning Powered
- **Bidirectional LSTM** neural network for time-series pattern recognition
- **Random Forest** ensemble for robust signal generation
- **50+ Technical Indicators** including RSI, MACD, Bollinger Bands, ATR, volume analysis
- **Ensemble Predictions** combining multiple ML models for higher accuracy

### 📊 Market Analysis
- Real-time cryptocurrency price monitoring via **Binance Public API** (no authentication required)
- Comprehensive technical analysis with 100+ features
- Candlestick pattern recognition
- Support/resistance level detection
- Volatility and momentum indicators

### 📱 Telegram Alerts
- **BUY/SELL/NEUTRAL** signals with confidence scores
- Price movement alerts (1h, 4h, 24h thresholds)
- RSI oversold/overbought notifications
- Auto-detection of Telegram chat ID
- Beautiful formatted messages with emojis

### ⚙️ Smart Alert Management
- Configurable signal confidence thresholds
- Daily alert limits to prevent spam
- Minimum time interval between alerts
- No repeated neutral signals

---

## 📁 Project Structure

```
trade_project/
├── alert_bot.py              # Main alert bot orchestrator
├── binance_data_fetcher.py   # Binance API data retrieval
├── feature_engineering.py    # Technical indicator calculation
├── signal_generator.py       # ML model training & prediction
├── telegram_notifier.py      # Telegram message sending
├── config.py                 # Configuration management
├── logger_config.py          # Logging setup
├── requirements.txt          # Python dependencies
├── README.md                 # This file
└── models/                   # Trained ML models (generated)
    ├── lstm_model.h5
    ├── rf_model.pkl
    └── scaler.pkl
```

---

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.8+
- TA-Lib library
- Telegram Bot Token (get from [@BotFather](https://t.me/botfather))

### 2. Installation

#### On Ubuntu/EC2:

```bash
# Clone repository
git clone https://github.com/prathamesh-ops-sudo/delta-trading-bot.git
cd delta-trading-bot

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install system dependencies
sudo apt-get update
sudo apt-get install -y build-essential wget

# Install TA-Lib
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install
sudo ldconfig
cd ..

# Install Python packages (compatible NumPy first!)
pip install --no-cache-dir "numpy<2.0" "numpy>=1.23.0"
pip install --no-cache-dir TA-Lib
pip install --no-cache-dir -r requirements.txt
```

#### On Windows:

```bash
# Clone repository
git clone https://github.com/prathamesh-ops-sudo/delta-trading-bot.git
cd delta-trading-bot

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install TA-Lib from wheel
pip install TA_Lib-0.4.28-cp310-cp310-win_amd64.whl

# Install other packages
pip install -r requirements.txt
```

### 3. Configuration

Edit `config.py`:

```python
# Telegram Configuration
TELEGRAM_BOT_TOKEN = "YOUR_BOT_TOKEN_HERE"  # Get from @BotFather

# Trading pair to monitor
SYMBOL = "BTCUSDT"  # Bitcoin/USDT

# Timeframe
INTERVAL = "5m"  # 1m, 5m, 15m, 30m, 1h, 4h, 1d

# Signal thresholds
BUY_SIGNAL_THRESHOLD = 0.70   # 70% confidence for BUY
SELL_SIGNAL_THRESHOLD = 0.70  # 70% confidence for SELL

# Alert limits
MAX_DAILY_ALERTS = 20
MIN_ALERT_INTERVAL = 300  # 5 minutes
```

### 4. Get Telegram Chat ID

Send `/start` to your bot on Telegram. The bot will auto-detect and display your chat ID.

### 5. Run the Bot

```bash
# Activate virtual environment
source venv/bin/activate  # On Linux
# or
venv\Scripts\activate  # On Windows

# Run the alert bot
python alert_bot.py
```

On first run, the bot will:
1. ✅ Fetch historical data from Binance
2. ✅ Calculate 50+ technical indicators
3. ✅ Train LSTM and Random Forest models
4. ✅ Save models to disk
5. ✅ Start monitoring and send alerts

**Subsequent runs** load pre-trained models (much faster!).

---

## 🔧 Testing Components

### Test Binance Data Fetcher
```bash
python binance_data_fetcher.py
```
Expected output:
- ✓ Connection successful
- ✓ Current BTC price
- ✓ 24h statistics
- ✓ Fetched candles
- ✓ DataFrame created

### Test Feature Engineering
```bash
python feature_engineering.py
```
Expected output:
- ✓ Fetched 500 candles
- ✓ Generated 100+ features
- ✓ Sample feature values

### Test Signal Generator
```bash
python signal_generator.py
```
Expected output:
- ✓ Data fetched
- ✓ Features calculated
- ✓ Models trained
- ✓ Signal generated (BUY/SELL/NEUTRAL)

---

## 📈 How It Works

### 1. Data Collection
- Fetches OHLCV data from **Binance Public API** (no authentication)
- Supports all Binance spot trading pairs
- Retrieves up to 1000 historical candles

### 2. Feature Engineering
Calculates **50+ technical indicators**:

**Trend Indicators:**
- SMA (7, 14, 21, 50, 100, 200)
- EMA (7, 14, 21, 50, 100, 200)
- MACD (12, 26, 9)
- ADX, +DI, -DI
- Parabolic SAR

**Momentum Indicators:**
- RSI (9, 14, 21)
- Stochastic Oscillator
- Williams %R
- CCI (Commodity Channel Index)
- ROC (Rate of Change)

**Volatility Indicators:**
- Bollinger Bands
- ATR (Average True Range)
- NATR (Normalized ATR)
- Historical Volatility

**Volume Indicators:**
- OBV (On-Balance Volume)
- MFI (Money Flow Index)
- A/D (Accumulation/Distribution)
- Volume ratios

**Pattern Recognition:**
- Doji, Hammer, Shooting Star
- Engulfing, Harami
- Morning Star, Evening Star

### 3. ML Model Ensemble

#### LSTM (60% weight)
- Bidirectional LSTM architecture
- 3 LSTM layers (128 → 64 → 32 units)
- Dropout & Batch Normalization
- Analyzes 60-period sequences

#### Random Forest (40% weight)
- 200 decision trees
- Max depth: 20
- Parallel processing

### 4. Signal Generation

**Signal Logic:**
```python
if ensemble_score >= 0.70:
    signal = "BUY"
elif ensemble_score <= 0.30:
    signal = "SELL"
else:
    signal = "NEUTRAL"
```

**Alert Sent When:**
- ✅ Confidence meets threshold
- ✅ Not sent too recently (5 min cooldown)
- ✅ Daily limit not exceeded
- ✅ Signal changed from previous

---

## 🎯 Alert Examples

### Buy Signal
```
💚 STRONG BUY SIGNAL

📊 Symbol: BTCUSDT
💰 Price: $98,234.50
📊 Confidence: 87.3%

Technical Indicators:
• RSI(14): 42.3
• MACD Histogram: 0.0234
• BB Position: 0.35

Model Scores:
• LSTM: 88.5%
• Random Forest: 85.2%

⏰ Time: 2025-12-10 15:30:45
```

### Price Alert
```
⚠️ Price Alert

📈 BTCUSDT moved +3.45% in 1 hour

Current Price: $99,123.00
1h Ago: $95,850.00
```

### RSI Alert
```
ℹ️ RSI Alert

📉 BTCUSDT is OVERSOLD

RSI(14): 28.5
Price: $96,543.00

This could indicate a potential buying opportunity.
```

---

## ☁️ EC2 Deployment

### Recommended Instance
- **Type:** t3.micro or t3.small
- **vCPUs:** 2
- **RAM:** 1-2 GB
- **Storage:** 8-16 GB
- **Region:** Choose closest to you

### Cost Estimate
- **t3.micro:** ~$7.50/month (~₹563/month)
- **t3.small:** ~$15/month (~₹1,125/month)

### Setup on EC2

```bash
# Update system
sudo apt-get update && sudo apt-get upgrade -y

# Install dependencies
sudo apt-get install -y python3 python3-pip python3-venv git build-essential wget

# Clone and setup (see Installation section above)

# Create systemd service
sudo nano /etc/systemd/system/alert-bot.service
```

**Service file content:**
```ini
[Unit]
Description=Crypto Alert Bot
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/delta-trading-bot
Environment="PATH=/home/ubuntu/delta-trading-bot/venv/bin"
ExecStart=/home/ubuntu/delta-trading-bot/venv/bin/python alert_bot.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**Enable and start service:**
```bash
sudo systemctl daemon-reload
sudo systemctl enable alert-bot
sudo systemctl start alert-bot
sudo systemctl status alert-bot
```

**View logs:**
```bash
sudo journalctl -u alert-bot -f
```

---

## 📊 Configuration Options

### Alert Thresholds

```python
# Signal confidence thresholds
BUY_SIGNAL_THRESHOLD = 0.70   # Higher = fewer, stronger signals
SELL_SIGNAL_THRESHOLD = 0.70

# Price movement alerts
PRICE_CHANGE_THRESHOLD_1H = 2.0   # %
PRICE_CHANGE_THRESHOLD_4H = 5.0   # %
PRICE_CHANGE_THRESHOLD_24H = 10.0 # %

# RSI thresholds
RSI_OVERSOLD = 30   # Below = oversold
RSI_OVERBOUGHT = 70 # Above = overbought
```

### Alert Frequency

```python
CHECK_INTERVAL = 60          # Check every 60 seconds
MIN_ALERT_INTERVAL = 300     # 5 min between alerts
MAX_DAILY_ALERTS = 20        # Max 20 alerts per day
```

### Model Parameters

```python
LSTM_LOOKBACK = 60           # 60 periods for LSTM
FEATURE_WINDOW = 100         # 100 periods for features
CANDLES_TO_FETCH = 500       # Fetch 500 candles
MODEL_RETRAIN_HOURS = 24     # Retrain every 24 hours
```

---

## 🛠️ Troubleshooting

### TA-Lib Installation Issues

**Error:** `ModuleNotFoundError: No module named '_ta_lib'`

**Solution:**
```bash
# Ensure NumPy <2.0 is installed FIRST
pip install --force-reinstall "numpy<2.0"
pip install --force-reinstall TA-Lib
```

### Telegram Not Receiving Messages

1. Check bot token is correct
2. Send `/start` to bot to enable chat
3. Verify TELEGRAM_CHAT_ID is set in config
4. Test: `python telegram_notifier.py`

### Models Not Training

**Error:** `Failed to converge`

**Solution:**
- Fetch more historical data (increase CANDLES_TO_FETCH)
- Check internet connection to Binance
- Verify TA-Lib is properly installed

### Memory Issues on EC2

Use swap space:
```bash
sudo dd if=/dev/zero of=/swapfile bs=1M count=4096
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

---

## 📝 Logging

Logs are saved to `alert_bot.log` and console.

**View logs:**
```bash
tail -f alert_bot.log
```

**Log levels:**
- `INFO`: Normal operations
- `WARNING`: Non-critical issues
- `ERROR`: Critical errors

---

## 🔐 Security Notes

- Never commit API keys or bot tokens to Git
- Use environment variables for sensitive data
- Keep your EC2 instance updated
- Use security groups to restrict access

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📄 License

MIT License - feel free to use and modify!

---

## 📞 Support

- **Issues:** https://github.com/prathamesh-ops-sudo/delta-trading-bot/issues
- **Telegram:** Set up your own bot using [@BotFather](https://t.me/botfather)

---

## 🎓 Data Sources

- **Binance Public API:** https://api.binance.com
  - No authentication required
  - Free tier with generous rate limits
  - Real-time OHLCV data
  - 30 calls/minute for free

---

## ⚠️ Disclaimer

This bot provides **informational alerts only**. It does NOT execute trades automatically.

**Important:**
- This is NOT financial advice
- Always do your own research (DYOR)
- Cryptocurrency trading carries risk
- Past performance doesn't guarantee future results
- Only invest what you can afford to lose

---

## 🚀 Roadmap

- [ ] Multi-symbol monitoring
- [ ] Web dashboard
- [ ] Backtesting framework
- [ ] Model performance tracking
- [ ] Custom indicator support
- [ ] Discord integration
- [ ] Advanced chart analysis
- [ ] Portfolio tracking

---

**Made with ❤️ for crypto traders**

*Happy Trading! 🚀*
