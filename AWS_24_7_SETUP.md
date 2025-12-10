# 🚀 AWS EC2 24/7 Setup Guide for Alert Bot

Complete guide to run your crypto alert bot 24/7 on AWS with auto-restart and continuous learning.

---

## 📋 Table of Contents
1. [Initial Setup](#initial-setup)
2. [Systemd Service Setup](#systemd-service-setup)
3. [Auto-Start on Boot](#auto-start-on-boot)
4. [Monitoring & Logs](#monitoring--logs)
5. [Model Retraining Schedule](#model-retraining-schedule)
6. [Backup Strategy](#backup-strategy)
7. [Troubleshooting](#troubleshooting)

---

## 🎯 Initial Setup

### 1. EC2 Instance Recommendation

**For Budget (~₹563/month):**
- **Instance:** t3.micro
- **vCPUs:** 2
- **RAM:** 1 GB
- **Storage:** 8-16 GB gp3
- **Region:** ap-south-1 (Mumbai) for low latency

**For Better Performance (~₹1,125/month):**
- **Instance:** t3.small
- **vCPUs:** 2
- **RAM:** 2 GB
- **Storage:** 16 GB gp3

### 2. Security Group Settings

**Inbound Rules:**
- SSH (22) from Your IP only
- HTTPS (443) - Optional for updates

**Outbound Rules:**
- All traffic (needed for Binance API + Telegram)

### 3. Initial Installation

```bash
# Already done if you followed README.md
cd ~/delta-trading-bot
git pull
source venv/bin/activate
```

---

## ⚙️ Systemd Service Setup

### 1. Create Service File

```bash
sudo nano /etc/systemd/system/alert-bot.service
```

### 2. Service Configuration

```ini
[Unit]
Description=Crypto Alert Bot - ML-Powered Telegram Alerts
After=network.target
Wants=network-online.target
StartLimitIntervalSec=0

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/delta-trading-bot
Environment="PATH=/home/ubuntu/delta-trading-bot/venv/bin"
Environment="PYTHONUNBUFFERED=1"

# Main command
ExecStart=/home/ubuntu/delta-trading-bot/venv/bin/python alert_bot.py

# Auto-restart configuration
Restart=always
RestartSec=10
StartLimitBurst=5

# Logging
StandardOutput=append:/home/ubuntu/delta-trading-bot/systemd_output.log
StandardError=append:/home/ubuntu/delta-trading-bot/systemd_error.log

# Resource limits (prevents memory leaks)
MemoryLimit=800M
CPUQuota=150%

[Install]
WantedBy=multi-user.target
```

### 3. Enable and Start Service

```bash
# Reload systemd configuration
sudo systemctl daemon-reload

# Enable service (starts on boot)
sudo systemctl enable alert-bot

# Start service now
sudo systemctl start alert-bot

# Check status
sudo systemctl status alert-bot
```

**Expected Output:**
```
● alert-bot.service - Crypto Alert Bot - ML-Powered Telegram Alerts
     Loaded: loaded (/etc/systemd/system/alert-bot.service; enabled)
     Active: active (running) since Tue 2025-12-10 08:30:00 UTC; 5s ago
   Main PID: 12345 (python)
      Tasks: 3 (limit: 1131)
     Memory: 250.5M
        CPU: 2.145s
     CGroup: /system.slice/alert-bot.service
             └─12345 /home/ubuntu/delta-trading-bot/venv/bin/python alert_bot.py
```

---

## 🔄 Auto-Start on Boot

The service is already configured to auto-start, but verify:

```bash
# Check if enabled
sudo systemctl is-enabled alert-bot

# Should output: enabled

# Test auto-start by rebooting
sudo reboot

# After reboot, check if running
sudo systemctl status alert-bot
```

---

## 📊 Monitoring & Logs

### Real-Time Logs

```bash
# Follow systemd logs
sudo journalctl -u alert-bot -f

# Follow application logs
tail -f ~/delta-trading-bot/alert_bot.log

# Follow systemd output
tail -f ~/delta-trading-bot/systemd_output.log
```

### Log Rotation (Prevent Disk Fill)

```bash
# Create log rotation config
sudo nano /etc/logrotate.d/alert-bot
```

**Add this content:**
```
/home/ubuntu/delta-trading-bot/*.log {
    daily
    rotate 7
    compress
    delaycompress
    notifempty
    missingok
    create 0644 ubuntu ubuntu
}

/home/ubuntu/delta-trading-bot/systemd_*.log {
    daily
    rotate 7
    compress
    delaycompress
    notifempty
    missingok
    create 0644 ubuntu ubuntu
}
```

**Test log rotation:**
```bash
sudo logrotate -f /etc/logrotate.d/alert-bot
```

### Check System Resources

```bash
# Memory usage
free -h

# Disk usage
df -h

# Process info
ps aux | grep python

# Resource usage
top -p $(pgrep -f alert_bot.py)
```

---

## 🧠 Model Retraining Schedule

### Automatic Daily Retraining

Create a cron job to retrain models daily with new data:

```bash
# Open crontab
crontab -e
```

**Add this line:**
```bash
# Retrain models every day at 3 AM UTC
0 3 * * * cd /home/ubuntu/delta-trading-bot && /home/ubuntu/delta-trading-bot/venv/bin/python retrain_models.py >> /home/ubuntu/delta-trading-bot/retrain.log 2>&1
```

### Create Retraining Script

```bash
nano ~/delta-trading-bot/retrain_models.py
```

**Content:**
```python
#!/usr/bin/env python3
"""
Automatic Model Retraining Script
Runs daily to update models with latest market data
"""
import sys
import logging
from datetime import datetime

from binance_data_fetcher import BinanceDataFetcher
from feature_engineering import FeatureEngine
from signal_generator import SignalGenerator
from config import Config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def retrain_models():
    """Retrain models with latest data"""
    try:
        logger.info("=" * 70)
        logger.info("Starting automatic model retraining")
        logger.info(f"Time: {datetime.now()}")
        logger.info("=" * 70)

        # Fetch latest data (more data = better training)
        logger.info("Fetching latest market data...")
        fetcher = BinanceDataFetcher()
        klines = fetcher.get_klines(Config.SYMBOL, Config.INTERVAL, 1000)
        df = fetcher.klines_to_dataframe(klines)
        logger.info(f"✓ Fetched {len(df)} candles")

        # Calculate features
        logger.info("Calculating features...")
        engine = FeatureEngine()
        df = engine.prepare_data_from_binance(df)
        df = engine.calculate_all_features(df)
        logger.info(f"✓ Generated {len(engine.get_feature_names())} features")

        # Train models
        logger.info("Training models...")
        signal_gen = SignalGenerator()
        feature_cols = engine.get_feature_names()

        # More epochs for daily retraining
        signal_gen.train(df, feature_cols, epochs=50, batch_size=32)

        # Save models
        signal_gen.save_models()
        logger.info("✓ Models retrained and saved successfully")

        # Restart service to load new models
        logger.info("Restarting alert-bot service...")
        import subprocess
        subprocess.run(['sudo', 'systemctl', 'restart', 'alert-bot'], check=True)
        logger.info("✓ Service restarted")

        logger.info("=" * 70)
        logger.info("Retraining completed successfully")
        logger.info("=" * 70)

        return True

    except Exception as e:
        logger.error(f"Retraining failed: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    success = retrain_models()
    sys.exit(0 if success else 1)
```

**Make it executable:**
```bash
chmod +x ~/delta-trading-bot/retrain_models.py
```

**Test it:**
```bash
cd ~/delta-trading-bot
source venv/bin/activate
python retrain_models.py
```

---

## 💾 Backup Strategy

### 1. Backup Models Daily

```bash
# Create backup directory
mkdir -p ~/backups/models

# Add to crontab (daily at 4 AM)
crontab -e
```

**Add:**
```bash
# Backup models daily at 4 AM (after retraining)
0 4 * * * cp -r /home/ubuntu/delta-trading-bot/models /home/ubuntu/backups/models_$(date +\%Y\%m\%d) && find /home/ubuntu/backups -name "models_*" -mtime +7 -exec rm -rf {} \;
```

### 2. Backup Logs Weekly

```bash
# Add to crontab (weekly on Sunday)
0 5 * * 0 tar -czf /home/ubuntu/backups/logs_$(date +\%Y\%m\%d).tar.gz /home/ubuntu/delta-trading-bot/*.log && find /home/ubuntu/backups -name "logs_*.tar.gz" -mtime +30 -delete
```

### 3. Push to GitHub Daily

```bash
# Add to crontab (daily at 6 AM)
0 6 * * * cd /home/ubuntu/delta-trading-bot && git add models/*.pkl models/*.h5 && git commit -m "Auto-backup: Models $(date +\%Y-\%m-\%d)" && git push origin main
```

---

## 🔍 Monitoring Dashboard

### Create a Simple Status Check Script

```bash
nano ~/check_bot_status.sh
```

**Content:**
```bash
#!/bin/bash

echo "======================================"
echo "   Crypto Alert Bot Status"
echo "======================================"
echo ""

# Service status
echo "📊 Service Status:"
sudo systemctl is-active alert-bot
echo ""

# Process info
echo "🔧 Process Info:"
ps aux | grep -E "python.*alert_bot.py" | grep -v grep
echo ""

# Memory usage
echo "💾 Memory Usage:"
free -h | grep Mem
echo ""

# Disk usage
echo "💿 Disk Usage:"
df -h / | tail -1
echo ""

# Recent logs
echo "📝 Recent Logs (last 10 lines):"
tail -10 ~/delta-trading-bot/alert_bot.log
echo ""

# Model files
echo "🤖 Model Files:"
ls -lh ~/delta-trading-bot/models/
echo ""

echo "======================================"
```

**Make executable:**
```bash
chmod +x ~/check_bot_status.sh
```

**Run anytime:**
```bash
~/check_bot_status.sh
```

---

## 🛠️ Service Management Commands

```bash
# Start service
sudo systemctl start alert-bot

# Stop service
sudo systemctl stop alert-bot

# Restart service
sudo systemctl restart alert-bot

# Check status
sudo systemctl status alert-bot

# View logs
sudo journalctl -u alert-bot -f

# View last 100 lines
sudo journalctl -u alert-bot -n 100

# Disable auto-start
sudo systemctl disable alert-bot

# Enable auto-start
sudo systemctl enable alert-bot
```

---

## 🔧 Troubleshooting

### Bot Not Starting

```bash
# Check logs
sudo journalctl -u alert-bot -n 50

# Check for Python errors
tail -50 ~/delta-trading-bot/systemd_error.log

# Test manually
cd ~/delta-trading-bot
source venv/bin/activate
python alert_bot.py
```

### Memory Issues

```bash
# Check memory
free -h

# Add more swap
sudo dd if=/dev/zero of=/swapfile bs=1M count=4096
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### Service Crashes Frequently

```bash
# Check crash logs
sudo journalctl -u alert-bot --since "1 hour ago"

# Increase restart delay
sudo nano /etc/systemd/system/alert-bot.service
# Change: RestartSec=30

sudo systemctl daemon-reload
sudo systemctl restart alert-bot
```

### Telegram Not Working

```bash
# Check Telegram token
grep TELEGRAM_BOT_TOKEN ~/delta-trading-bot/config.py

# Send /start to bot
# Check logs
tail -f ~/delta-trading-bot/alert_bot.log | grep Telegram
```

---

## 📈 Continuous Improvement

The bot automatically improves through:

1. **Daily Model Retraining**
   - Fetches latest 1000 candles
   - Retrains LSTM + Random Forest
   - Adapts to market changes

2. **Increasing Training Data**
   - Every day adds more data
   - Models learn from recent patterns
   - Improves accuracy over time

3. **Performance Tracking**
   - All signals logged
   - Can analyze which signals worked
   - Future: Implement feedback loop

---

## 💰 Monthly Costs

**t3.micro:**
- EC2 Instance: ~₹563/month
- Data Transfer: ~₹50/month (minimal)
- **Total:** ~₹613/month

**t3.small (recommended):**
- EC2 Instance: ~₹1,125/month
- Data Transfer: ~₹50/month
- **Total:** ~₹1,175/month

---

## ✅ Final Checklist

- [ ] Service running: `sudo systemctl status alert-bot`
- [ ] Auto-start enabled: `sudo systemctl is-enabled alert-bot`
- [ ] Telegram working: Send `/start` to bot
- [ ] Logs rotating: Check `/etc/logrotate.d/alert-bot`
- [ ] Cron jobs set: `crontab -l`
- [ ] Backups configured
- [ ] Monitoring script ready: `~/check_bot_status.sh`

---

**Your bot is now running 24/7! 🚀**

Check status: `sudo systemctl status alert-bot`
View logs: `tail -f ~/delta-trading-bot/alert_bot.log`
Monitor: `~/check_bot_status.sh`
