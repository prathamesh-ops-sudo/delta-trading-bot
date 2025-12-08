# EC2 Memory Fix - Installation Getting Killed

## Problem
Installing TensorFlow on t2.micro (1GB RAM) gets killed due to insufficient memory.

## Quick Fix (Run These Commands on EC2)

### Step 1: Add Swap Space (Critical!)

```bash
# Create 2GB swap file
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Make swap permanent
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

# Verify swap is active
free -h
```

You should see:
```
              total        used        free      shared  buff/cache   available
Mem:          964Mi       180Mi       450Mi       1.0Mi       333Mi       641Mi
Swap:         2.0Gi          0B       2.0Gi
```

### Step 2: Clear Previous Installation Attempts

```bash
cd ~/delta-trading-bot
source venv/bin/activate

# Clear pip cache
pip cache purge

# Remove any partially installed packages
pip uninstall tensorflow tensorflow-cpu keras -y
```

### Step 3: Install Packages One by One (Avoid Memory Overload)

```bash
cd ~/delta-trading-bot
source venv/bin/activate

# Install packages individually with --no-cache-dir flag
pip install --no-cache-dir numpy==1.24.3
pip install --no-cache-dir pandas==2.0.3
pip install --no-cache-dir scipy==1.11.1
pip install --no-cache-dir scikit-learn==1.3.0

# Use tensorflow-cpu instead of full tensorflow (much smaller)
pip install --no-cache-dir tensorflow-cpu==2.13.0

pip install --no-cache-dir keras==2.13.1
pip install --no-cache-dir TA-Lib==0.4.28
pip install --no-cache-dir requests==2.31.0
pip install --no-cache-dir urllib3==2.0.4
pip install --no-cache-dir schedule==1.2.0
pip install --no-cache-dir python-dotenv==1.0.0
pip install --no-cache-dir python-dateutil==2.8.2
pip install --no-cache-dir pytz==2023.3
```

**Important:**
- Use `tensorflow-cpu` instead of `tensorflow` (3x smaller)
- Use `--no-cache-dir` to avoid storing cache (saves memory)
- Install one at a time to prevent memory spikes

### Step 4: Verify Installation

```bash
python test_connection.py
```

## Alternative Solution: Use Automated Script

```bash
cd ~/delta-trading-bot

# Download and run automated setup script
chmod +x setup_ec2.sh
./setup_ec2.sh
```

## If Still Getting Killed

### Option 1: Upgrade to t3.micro (Recommended)

**t3.micro specs:**
- 2 vCPUs (vs 1 on t2.micro)
- 1 GB RAM (same)
- Better burst performance
- Cost: ~$7.50/month (~563 INR)

**How to upgrade:**
1. Stop your instance
2. Change instance type to t3.micro
3. Start instance
4. Re-run installation

### Option 2: Use Pre-built Docker Image (Advanced)

Create a Docker image on your local machine and deploy:

```dockerfile
FROM python:3.10-slim
RUN apt-get update && apt-get install -y build-essential wget
# ... (full Dockerfile provided if needed)
```

### Option 3: Temporarily Use Larger Instance for Setup

1. Change to **t2.small** (2GB RAM)
2. Install all packages
3. Train models
4. Change back to **t2.micro**
5. Bot will run fine on 1GB after installation

**Cost for temporary upgrade:**
- t2.small: ~$0.023/hour (~1.7 INR/hour)
- Setup time: ~30 minutes
- Total cost: ~1 INR for setup

## Monitor Memory During Installation

```bash
# In another terminal, watch memory usage
watch -n 1 free -h

# Or
htop
```

## Alternative: Install Without TensorFlow

If you want to skip TensorFlow temporarily:

```bash
# Install everything except TensorFlow
pip install --no-cache-dir -r requirements.txt --ignore-installed tensorflow tensorflow-cpu
```

Then modify `ml_models.py` to use only Random Forest (no LSTM).

## Troubleshooting

### "Killed" appears during pip install
**Solution:** Add more swap space
```bash
sudo swapoff /swapfile
sudo rm /swapfile
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### Swap not activating
**Solution:** Check swap status
```bash
sudo swapon --show
free -h
```

### Still running out of memory
**Solutions:**
1. Upgrade to t3.micro
2. Temporarily use t2.small for installation
3. Use Docker with pre-built image
4. Install on local machine, copy models to EC2

## Recommended: Automated Full Setup

Just run this single command:

```bash
cd ~
wget https://raw.githubusercontent.com/prathamesh-ops-sudo/delta-trading-bot/main/setup_ec2.sh
chmod +x setup_ec2.sh
./setup_ec2.sh
```

This script:
- ✓ Adds swap automatically
- ✓ Installs all dependencies
- ✓ Installs packages one by one
- ✓ Uses --no-cache-dir flag
- ✓ Tests installation
- ✓ Handles errors gracefully

## Quick Reference Commands

```bash
# Check memory
free -h

# Check swap
sudo swapon --show

# Add swap
sudo fallocate -l 2G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile

# Clear pip cache
pip cache purge

# Install with no cache
pip install --no-cache-dir package_name

# Monitor installation
watch -n 1 free -h
```

## Expected Installation Time

With swap space on t2.micro:
- numpy, pandas, scipy: ~2 minutes each
- scikit-learn: ~3 minutes
- tensorflow-cpu: ~8-10 minutes
- Other packages: ~2 minutes
- **Total: ~20-25 minutes**

## Success Indicators

After successful installation:

```bash
python -c "import numpy; print('NumPy OK')"
python -c "import pandas; print('Pandas OK')"
python -c "import sklearn; print('Sklearn OK')"
python -c "import tensorflow; print('TensorFlow OK')"
python -c "import talib; print('TA-Lib OK')"
```

All should print "OK"!
