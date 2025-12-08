# EC2 Deployment Guide - Complete Setup

Complete step-by-step guide to deploy your trading bot on AWS EC2.

## 📋 EC2 Instance Configuration

### Recommended Instance Type

**t2.micro (Free Tier Eligible)**
- **vCPUs**: 1
- **Memory**: 1 GB
- **Storage**: 8 GB (can increase to 30 GB on free tier)
- **Network**: Low to Moderate
- **Cost**: FREE for first 750 hours/month (12 months)
- **Perfect for**: This trading bot

**Alternative (if not on free tier):**
- **t3.micro**: ~$7.50/month (~563 INR)
- **t3a.micro**: ~$6.75/month (~506 INR)

### Instance Specifications

```
Instance Type: t2.micro
AMI: Ubuntu Server 22.04 LTS
Architecture: 64-bit (x86)
Storage: 12 GB GP3 SSD (free tier allows up to 30 GB)
Region: ap-south-1 (Mumbai) - Lowest latency for India
```

## 🚀 Step-by-Step EC2 Setup

### Step 1: Create EC2 Instance

1. **Go to AWS Console**
   - Navigate to: https://console.aws.amazon.com/ec2/
   - Select Region: **ap-south-1 (Mumbai)**

2. **Launch Instance**
   - Click "Launch Instance"
   - Name: `trading-bot-production`

3. **Choose AMI**
   - Select: **Ubuntu Server 22.04 LTS (HVM), SSD Volume Type**
   - Architecture: **64-bit (x86)**

4. **Choose Instance Type**
   - Select: **t2.micro** (Free tier eligible)
   - Click "Next: Configure Instance Details"

5. **Configure Instance**
   - Keep default settings
   - Click "Next: Add Storage"

6. **Add Storage**
   - Size: **12 GB** (can go up to 30 GB on free tier)
   - Volume Type: **gp3** (or gp2)
   - Click "Next: Add Tags"

7. **Add Tags** (Optional)
   - Key: `Name`, Value: `Trading Bot`
   - Key: `Purpose`, Value: `Crypto Trading`
   - Click "Next: Configure Security Group"

8. **Configure Security Group**
   - Create a **new security group**
   - Security group name: `trading-bot-sg`
   - Description: `Security group for trading bot`

   **Add Rules:**
   ```
   Type: SSH
   Protocol: TCP
   Port: 22
   Source: My IP (or 0.0.0.0/0 if your IP changes)
   Description: SSH access
   ```

   Click "Review and Launch"

9. **Review and Launch**
   - Review all settings
   - Click **"Launch"**

10. **Create Key Pair** (IMPORTANT!)
    - Select: **"Create a new key pair"**
    - Key pair name: `trading-bot-key`
    - Key pair type: **RSA**
    - Private key file format: **.pem**
    - Click **"Download Key Pair"**
    - **SAVE THIS FILE SAFELY!** You can't download it again
    - Click **"Launch Instances"**

### Step 2: Connect to EC2 Instance

**Windows (using PowerShell or CMD):**

1. **Move your key file to a safe location:**
   ```powershell
   mkdir C:\Users\YourName\.ssh
   move C:\Users\YourName\Downloads\trading-bot-key.pem C:\Users\YourName\.ssh\
   ```

2. **Get your instance's Public IP:**
   - Go to EC2 Console
   - Select your instance
   - Copy the **Public IPv4 address** (e.g., 13.233.123.45)

3. **Connect using SSH:**
   ```powershell
   ssh -i "C:\Users\YourName\.ssh\trading-bot-key.pem" ubuntu@YOUR_INSTANCE_IP
   ```

   Example:
   ```powershell
   ssh -i "C:\Users\prath\.ssh\trading-bot-key.pem" ubuntu@13.233.123.45
   ```

   If you get a "permissions too open" error:
   - Right-click key file → Properties → Security
   - Remove all users except yourself
   - Give yourself Full Control

**Alternative: Use PuTTY (Windows)**

1. Download PuTTY: https://www.putty.org/
2. Convert .pem to .ppk using PuTTYgen
3. Use PuTTY to connect with the .ppk key

### Step 3: Initial Server Setup

Once connected to your EC2 instance:

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install Python 3.10 and pip
sudo apt install python3.10 python3-pip python3-venv -y

# Install build tools (required for TA-Lib)
sudo apt install build-essential wget -y

# Install Git
sudo apt install git -y

# Verify installations
python3 --version  # Should show Python 3.10.x
pip3 --version
git --version
```

### Step 4: Install TA-Lib (Technical Analysis Library)

```bash
# Download TA-Lib source
cd /tmp
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz

# Extract
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/

# Compile and install
./configure --prefix=/usr
make
sudo make install

# Update library cache
sudo ldconfig

# Verify installation
ls -l /usr/lib/libta_lib*

# Clean up
cd ~
rm -rf /tmp/ta-lib*
```

### Step 5: Clone Your Repository

```bash
# Clone your private repository
git clone https://github.com/prathamesh-ops-sudo/delta-trading-bot.git

# Enter directory
cd delta-trading-bot

# Verify files
ls -la
```

**If repository is private, you'll need authentication:**

**Option A - Personal Access Token (Recommended):**
1. Go to: https://github.com/settings/tokens
2. Generate new token (classic)
3. Select scopes: `repo` (full control)
4. Copy the token
5. Use it when cloning:
   ```bash
   git clone https://YOUR_TOKEN@github.com/prathamesh-ops-sudo/delta-trading-bot.git
   ```

**Option B - SSH Key:**
```bash
# Generate SSH key
ssh-keygen -t ed25519 -C "your_email@example.com"

# Display public key
cat ~/.ssh/id_ed25519.pub

# Copy this and add to GitHub: https://github.com/settings/keys
```

### Step 6: Install Python Dependencies

```bash
cd delta-trading-bot

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Verify TA-Lib installation
python -c "import talib; print('TA-Lib OK')"
```

**If TA-Lib Python package fails:**
```bash
pip install --upgrade setuptools
pip install numpy
pip install TA-Lib
```

### Step 7: Test the Bot

```bash
# Make sure you're in the virtual environment
source venv/bin/activate

# Test connection
python test_connection.py
```

You should see:
```
✓ PASS - Imports
✓ PASS - Configuration
✓ PASS - Delta API
✓ PASS - Feature Engineering
✓ PASS - Risk Manager
```

### Step 8: Send /start to Telegram Bot

1. Open Telegram on your phone/computer
2. Find your bot (search for the bot name)
3. Send: `/start` or just `hello`
4. **Wait for confirmation message**

The bot will auto-detect your chat ID and send you a confirmation!

### Step 9: Train Models (First Time)

```bash
# This takes 5-10 minutes
python train_models.py
```

Wait for:
```
✓ MODEL TRAINING COMPLETE
Models saved to:
  - models/lstm_model.h5
  - models/rf_model.pkl
  - models/scaler.pkl
```

### Step 10: Run the Trading Bot

**Option A - Foreground (for testing):**
```bash
python trading_bot.py
```

Press `Ctrl+C` to stop.

**Option B - Background with Screen (Recommended):**
```bash
# Install screen
sudo apt install screen -y

# Start screen session
screen -S trading-bot

# Run the bot
python trading_bot.py

# Detach: Press Ctrl+A then D

# Reattach later
screen -r trading-bot

# List sessions
screen -ls

# Kill session
screen -X -S trading-bot quit
```

**Option C - Systemd Service (Production, Auto-restart):**

Create service file:
```bash
sudo nano /etc/systemd/system/trading-bot.service
```

Add this content:
```ini
[Unit]
Description=Delta Exchange Trading Bot
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/delta-trading-bot
Environment="PATH=/home/ubuntu/delta-trading-bot/venv/bin"
ExecStart=/home/ubuntu/delta-trading-bot/venv/bin/python trading_bot.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
# Reload systemd
sudo systemctl daemon-reload

# Enable service (start on boot)
sudo systemctl enable trading-bot

# Start service
sudo systemctl start trading-bot

# Check status
sudo systemctl status trading-bot

# View logs
sudo journalctl -u trading-bot -f

# Stop service
sudo systemctl stop trading-bot

# Restart service
sudo systemctl restart trading-bot
```

## 📊 Monitoring Your Bot

### View Logs

```bash
# Real-time log monitoring
tail -f trading_bot.log

# Last 100 lines
tail -n 100 trading_bot.log

# Search for trades
grep "TRADE EXECUTED" trading_bot.log

# Search for errors
grep "ERROR" trading_bot.log

# Check system logs (if using systemd)
sudo journalctl -u trading-bot -n 50
```

### Check Bot Status

```bash
# If using systemd
sudo systemctl status trading-bot

# If using screen
screen -ls

# Check if process is running
ps aux | grep trading_bot

# Check resource usage
htop
```

### Monitor Performance

```bash
# View trade history
cat risk_manager_state.json | python -m json.tool

# Check disk space
df -h

# Check memory usage
free -h

# Check CPU usage
top
```

## 🔧 Maintenance Tasks

### Update the Bot

```bash
# Stop the bot
sudo systemctl stop trading-bot  # or exit screen session

# Pull latest code
cd delta-trading-bot
git pull

# Activate venv and update dependencies
source venv/bin/activate
pip install -r requirements.txt --upgrade

# Restart bot
sudo systemctl start trading-bot  # or run in screen
```

### Backup Data

```bash
# Create backup directory
mkdir -p ~/backups

# Backup important files
cd delta-trading-bot
tar -czf ~/backups/trading-bot-backup-$(date +%Y%m%d).tar.gz \
    risk_manager_state.json \
    models/ \
    trading_bot.log

# Copy to local machine (run from your PC)
scp -i "C:\Users\prath\.ssh\trading-bot-key.pem" \
    ubuntu@YOUR_IP:~/backups/*.tar.gz \
    C:\Users\prath\Desktop\backups\
```

### Clean Up Logs

```bash
# Rotate logs (keep last 10 days)
cd delta-trading-bot
find . -name "*.log" -mtime +10 -delete

# Or manually
mv trading_bot.log trading_bot.log.old
```

## 💰 Cost Breakdown (Mumbai Region)

### Free Tier (First 12 Months)
```
EC2 t2.micro: FREE (750 hours/month)
Storage (30 GB): FREE
Data Transfer (15 GB out): FREE
Estimated Cost: 0 INR/month
```

### After Free Tier
```
t2.micro instance: ~$8/month (~600 INR)
Storage (12 GB): ~$1/month (~75 INR)
Data Transfer (minimal): ~$0.50/month (~38 INR)
Total: ~$9.50/month (~713 INR)
```

**Well within your 2,000 INR budget!** ✅

## 🔐 Security Best Practices

### 1. Update Regularly
```bash
sudo apt update && sudo apt upgrade -y
```

### 2. Enable Firewall
```bash
sudo ufw allow 22/tcp
sudo ufw enable
sudo ufw status
```

### 3. Disable Root Login
```bash
sudo nano /etc/ssh/sshd_config
# Set: PermitRootLogin no
sudo systemctl restart sshd
```

### 4. Keep Logs Private
```bash
chmod 600 *.log
chmod 600 risk_manager_state.json
```

### 5. Use Elastic IP (Optional)
- Prevents IP from changing on restart
- Free if instance is running
- Costs ~$3.60/month (~270 INR) if instance is stopped

## 🚨 Troubleshooting

### Bot Won't Start
```bash
# Check Python path
which python

# Check dependencies
pip list

# Check for errors
python trading_bot.py
```

### TA-Lib Import Error
```bash
# Reinstall TA-Lib
sudo ldconfig
pip uninstall TA-Lib
pip install TA-Lib
```

### Out of Memory
```bash
# Check memory
free -h

# Add swap space (if needed)
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### Connection Issues
```bash
# Check security group allows SSH from your IP
# Check instance is running
# Check key permissions are correct
```

## ✅ Verification Checklist

After deployment:

- [ ] EC2 instance running
- [ ] SSH connection works
- [ ] Python and dependencies installed
- [ ] TA-Lib working
- [ ] Repository cloned
- [ ] test_connection.py passes
- [ ] Telegram bot responds to /start
- [ ] Chat ID detected
- [ ] Models trained
- [ ] Bot running in background
- [ ] Logs being written
- [ ] Telegram notifications working

## 📱 Quick Commands Reference

```bash
# Connect to EC2
ssh -i ~/.ssh/trading-bot-key.pem ubuntu@YOUR_IP

# Activate environment
cd delta-trading-bot && source venv/bin/activate

# Run bot
python trading_bot.py

# View logs
tail -f trading_bot.log

# Check status (systemd)
sudo systemctl status trading-bot

# Restart bot (systemd)
sudo systemctl restart trading-bot

# Attach to screen
screen -r trading-bot
```

## 🎯 Next Steps

1. **Launch EC2 instance** (follow steps above)
2. **Connect via SSH**
3. **Install dependencies**
4. **Clone repository**
5. **Send /start to Telegram bot**
6. **Train models**
7. **Run bot**
8. **Monitor via Telegram**

---

**You're ready to deploy!** 🚀

**Estimated Setup Time**: 20-30 minutes
**Monthly Cost**: FREE (first year) or ~713 INR after
