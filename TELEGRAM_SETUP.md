# Telegram Setup Guide

Your Telegram bot token is already configured! ✅

**Bot Token**: `8325573196:AAEI1UTia5uCgSmsoxmO3aHD3O3fV-WWF0U`

## Quick Setup (2 Steps)

### Step 1: Send a Message to Your Bot

1. Open Telegram
2. Search for your bot (use the bot username from @BotFather)
3. Send any message to it (like `hello` or `/start`)

### Step 2: Auto-Detect Chat ID

**Option A - Automatic (Recommended)**

Just run the trading bot - it will auto-detect your chat ID:

```bash
python trading_bot.py
```

The bot will automatically find your chat ID from the message you sent!

**Option B - Using Setup Script**

```bash
python setup_telegram.py
```

This will:
- ✓ Find your chat ID
- ✓ Send a test message
- ✓ Show you the chat ID

### Step 3: Test Everything

```bash
python test_connection.py
```

This verifies everything is working!

## Manual Configuration (Optional)

If you want to manually set your chat ID, edit `config.py`:

```python
TELEGRAM_CHAT_ID = "your_chat_id_here"
```

## How It Works

1. **You send a message** to your bot on Telegram
2. **Bot auto-detects** your chat ID from that message
3. **Bot starts sending** you trading notifications!

## What You'll Receive

Once configured, you'll get:

### 🟢 Trade Signals
```
🟢 TRADE SIGNAL DETECTED
Signal: BUY
Confidence: 72%
Price: $45,230
LSTM: 75% | RF: 68%
```

### ✅ Trade Executions
```
✅ TRADE EXECUTED
Direction: BUY
Entry: $45,230
Size: 0.0331 contracts
Leverage: 5x
Stop Loss: $44,325
```

### 💰 Position Closures
```
💰 POSITION CLOSED - PROFIT
PnL: $51.25
ROI: +17.2%
```

### 📊 Daily Summaries
```
📈 DAILY PERFORMANCE
Trades: 8 | Win Rate: 75%
PnL: $287.50
```

## Troubleshooting

### "Could not auto-detect chat ID"
**Solution**: Make sure you sent a message to your bot first

### "Telegram notifications disabled"
**Solution**:
1. Send a message to your bot
2. Restart the trading bot
3. It will auto-detect your chat ID

### "Failed to send message"
**Solution**: Check your internet connection and bot token

## Test Your Setup

Send a test message:

```bash
python -c "from telegram_notifier import TelegramNotifier; TelegramNotifier().test_connection()"
```

You should receive a test message on Telegram!

---

**You're all set!** 🚀

Your bot token is configured, and chat ID will be detected automatically.

Just:
1. Send a message to your bot
2. Run: `python trading_bot.py`
3. Start receiving notifications!
