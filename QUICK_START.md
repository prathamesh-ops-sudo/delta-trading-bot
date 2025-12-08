# Quick Start Guide

## 5-Minute Setup

### Step 1: Install Python Dependencies

```bash
pip install -r requirements.txt
```

**Important**: TA-Lib requires special installation on Windows. See README.md for details.

### Step 2: Configure Telegram (Optional)

1. Create bot with @BotFather on Telegram
2. Get your chat ID
3. Edit `config.py`:
```python
TELEGRAM_BOT_TOKEN = "your_bot_token"
TELEGRAM_CHAT_ID = "your_chat_id"
```

### Step 3: Test Everything

```bash
python test_connection.py
```

This will verify:
- ✓ All packages installed
- ✓ Delta Exchange API connection
- ✓ Account access
- ✓ Telegram bot (if configured)
- ✓ Feature engineering
- ✓ Risk management

### Step 4: Train Models (First Time Only)

```bash
python train_models.py
```

This takes 5-10 minutes and creates:
- LSTM neural network model
- Random Forest classifier
- Feature scaler

**You only need to do this once** - the bot will retrain automatically every 24 hours.

### Step 5: Start Trading Bot

```bash
python trading_bot.py
```

The bot will:
- 🔍 Monitor BTC/USD market every 5 minutes
- 🧠 Analyze with ML models
- 💰 Execute trades when signals are strong
- 📊 Monitor positions every 30 seconds
- 📱 Send Telegram updates

### Stop the Bot

Press `Ctrl+C` to stop gracefully.

## What to Expect

### First 24 Hours
- Bot will analyze market conditions
- Trades only when ML confidence > 65%
- Maximum 12 trades per day
- Maximum 1 open position at a time

### Telegram Notifications
You'll receive:
- 🟢 Trade signals with confidence scores
- ✅ Trade execution confirmations
- 💰 Position closure with PnL
- 📊 Daily performance summaries

### Example Trade Flow

1. **Signal Detected** (08:15 AM)
   ```
   🟢 TRADE SIGNAL DETECTED
   Signal: BUY
   Confidence: 72%
   Price: $45,230
   ```

2. **Trade Executed** (08:15 AM)
   ```
   ✅ TRADE EXECUTED
   Direction: BUY
   Entry: $45,230
   Size: 0.0331 contracts
   Leverage: 5x
   Stop Loss: $44,325 (-2%)
   TP1: $46,587 (+3%)
   TP2: $47,492 (+5%)
   TP3: $48,846 (+8%)
   ```

3. **Position Closed** (10:45 AM)
   ```
   💰 POSITION CLOSED - PROFIT
   Entry: $45,230
   Exit: $46,780
   PnL: $51.25
   ROI: +17.2%
   Reason: Take profit level 2 hit
   ```

## Key Files

- `trading_bot.py` - Main bot (run this)
- `config.py` - All settings
- `trading_bot.log` - Detailed logs
- `risk_manager_state.json` - Trade history
- `models/` - Trained ML models

## Adjusting Risk

Edit `config.py`:

```python
# More conservative
MAX_LEVERAGE = 3
MAX_POSITION_SIZE_PCT = 0.10  # 10% max
SIGNAL_THRESHOLD = 0.70  # 70% min confidence

# More aggressive
MAX_LEVERAGE = 8
MAX_POSITION_SIZE_PCT = 0.25  # 25% max
SIGNAL_THRESHOLD = 0.60  # 60% min confidence
```

## Monitoring

### View Logs
```bash
# Real-time logs
tail -f trading_bot.log

# Search for trades
grep "TRADE EXECUTED" trading_bot.log

# Check errors
grep "ERROR" trading_bot.log
```

### Check Performance
- Telegram daily summaries (sent at midnight)
- Review `risk_manager_state.json`
- Read `trading_bot.log`

## Cloud Deployment

### Google Cloud (Free Tier)

1. Create f1-micro instance:
```bash
gcloud compute instances create trading-bot \
    --machine-type=f1-micro \
    --zone=asia-south1-a
```

2. SSH and setup:
```bash
gcloud compute ssh trading-bot
sudo apt update
sudo apt install python3-pip
git clone <your-repo>
cd trading-bot
pip3 install -r requirements.txt
```

3. Run with screen:
```bash
screen -S bot
python3 trading_bot.py
# Ctrl+A then D to detach
```

## Troubleshooting

### "TA-Lib not found"
Download and install TA-Lib wheel from:
https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib

### "API authentication failed"
Check API credentials in `config.py`

### "No trades executing"
- Check signal confidence in logs
- Verify account balance > 0
- Review risk assessment logs
- Lower SIGNAL_THRESHOLD in config

### "Telegram not working"
- Verify bot token is correct
- Check chat ID
- Test with: `python -c "from telegram_notifier import TelegramNotifier; TelegramNotifier().test_connection()"`

## Support

1. Run `python test_connection.py` first
2. Check logs: `trading_bot.log`
3. Review error messages
4. Ensure account has sufficient balance

## Safety Reminders

⚠️ **Start with small amounts**
⚠️ **Monitor the bot regularly**
⚠️ **Understand the risks**
⚠️ **Test with paper trading first** (if available)

---

**Ready to trade?**

```bash
python trading_bot.py
```

Good luck! 🚀
