#!/bin/bash
# EC2 Setup Script for Trading Bot
# Optimized for t2.micro (1GB RAM)

set -e  # Exit on error

echo "=========================================="
echo "Trading Bot EC2 Setup Script"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 1. Add Swap Space (Critical for t2.micro!)
echo -e "${YELLOW}Step 1: Adding Swap Space (2GB)${NC}"
if [ ! -f /swapfile ]; then
    sudo fallocate -l 2G /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile

    # Make swap permanent
    echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

    echo -e "${GREEN}✓ Swap space added${NC}"
else
    echo -e "${GREEN}✓ Swap already exists${NC}"
fi

# Verify swap
echo "Current memory status:"
free -h
echo ""

# 2. Update System
echo -e "${YELLOW}Step 2: Updating system packages${NC}"
sudo apt update
sudo apt upgrade -y
echo -e "${GREEN}✓ System updated${NC}"
echo ""

# 3. Install Dependencies
echo -e "${YELLOW}Step 3: Installing system dependencies${NC}"
sudo apt install -y \
    python3.10 \
    python3-pip \
    python3-venv \
    build-essential \
    wget \
    git \
    htop \
    screen
echo -e "${GREEN}✓ System dependencies installed${NC}"
echo ""

# 4. Install TA-Lib
echo -e "${YELLOW}Step 4: Installing TA-Lib${NC}"
if [ ! -f /usr/lib/libta_lib.so.0 ]; then
    cd /tmp
    wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
    tar -xzf ta-lib-0.4.0-src.tar.gz
    cd ta-lib/
    ./configure --prefix=/usr
    make
    sudo make install
    sudo ldconfig
    cd ~
    rm -rf /tmp/ta-lib*
    echo -e "${GREEN}✓ TA-Lib installed${NC}"
else
    echo -e "${GREEN}✓ TA-Lib already installed${NC}"
fi
echo ""

# 5. Clone Repository
echo -e "${YELLOW}Step 5: Repository setup${NC}"
if [ -d "delta-trading-bot" ]; then
    echo "Repository already exists, pulling latest changes..."
    cd delta-trading-bot
    git pull
else
    echo "Cloning repository..."
    git clone https://github.com/prathamesh-ops-sudo/delta-trading-bot.git
    cd delta-trading-bot
fi
echo -e "${GREEN}✓ Repository ready${NC}"
echo ""

# 6. Create Virtual Environment
echo -e "${YELLOW}Step 6: Creating Python virtual environment${NC}"
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${GREEN}✓ Virtual environment already exists${NC}"
fi
echo ""

# 7. Install Python Packages (with low memory optimizations)
echo -e "${YELLOW}Step 7: Installing Python packages${NC}"
echo -e "${YELLOW}This may take 10-15 minutes on t2.micro...${NC}"
source venv/bin/activate

# Upgrade pip first
pip install --upgrade pip

# Install packages one by one to avoid memory issues
echo "Installing numpy..."
pip install --no-cache-dir numpy==1.24.3

echo "Installing pandas..."
pip install --no-cache-dir pandas==2.0.3

echo "Installing scipy..."
pip install --no-cache-dir scipy==1.11.1

echo "Installing scikit-learn..."
pip install --no-cache-dir scikit-learn==1.3.0

echo "Installing tensorflow-cpu (this takes the longest)..."
pip install --no-cache-dir tensorflow-cpu==2.13.0

echo "Installing keras..."
pip install --no-cache-dir keras==2.13.1

echo "Installing TA-Lib Python wrapper..."
pip install --no-cache-dir TA-Lib==0.4.28

echo "Installing remaining packages..."
pip install --no-cache-dir requests==2.31.0
pip install --no-cache-dir urllib3==2.0.4
pip install --no-cache-dir schedule==1.2.0
pip install --no-cache-dir python-dotenv==1.0.0
pip install --no-cache-dir python-dateutil==2.8.2
pip install --no-cache-dir pytz==2023.3

echo -e "${GREEN}✓ All Python packages installed${NC}"
echo ""

# 8. Test Installation
echo -e "${YELLOW}Step 8: Testing installation${NC}"
python test_connection.py

echo ""
echo "=========================================="
echo -e "${GREEN}Setup Complete!${NC}"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Send /start to your Telegram bot"
echo "2. Train models: python train_models.py"
echo "3. Run bot: python trading_bot.py"
echo ""
echo "Or run in background:"
echo "  screen -S trading-bot"
echo "  python trading_bot.py"
echo "  # Press Ctrl+A then D to detach"
echo ""
