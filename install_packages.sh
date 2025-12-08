#!/bin/bash
# Bulletproof Package Installation for EC2
# Works on both t2.micro and t3.micro

set -e

echo "=========================================="
echo "Installing Trading Bot Dependencies"
echo "Optimized for Low-Memory EC2 Instances"
echo "=========================================="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Check if running in virtual environment
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo -e "${RED}ERROR: Not in virtual environment!${NC}"
    echo "Run: source venv/bin/activate"
    exit 1
fi

# Step 1: Add Swap (Critical!)
echo -e "${YELLOW}Step 1: Setting up swap space${NC}"
if [ ! -f /swapfile ]; then
    echo "Creating 4GB swap file..."
    sudo fallocate -l 4G /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
    echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
    echo -e "${GREEN}✓ Swap created${NC}"
else
    sudo swapon /swapfile 2>/dev/null || true
    echo -e "${GREEN}✓ Swap already exists${NC}"
fi

# Verify swap
echo ""
echo "Memory status:"
free -h
echo ""

# Step 2: Clear any previous failed installations
echo -e "${YELLOW}Step 2: Cleaning up previous installations${NC}"
pip cache purge
pip uninstall -y tensorflow tensorflow-cpu keras 2>/dev/null || true
echo -e "${GREEN}✓ Cleanup complete${NC}"
echo ""

# Step 3: Upgrade pip
echo -e "${YELLOW}Step 3: Upgrading pip${NC}"
pip install --upgrade pip setuptools wheel
echo -e "${GREEN}✓ Pip upgraded${NC}"
echo ""

# Step 4: Install packages one by one with --no-cache-dir
echo -e "${YELLOW}Step 4: Installing packages (this takes 15-20 minutes)${NC}"
echo ""

echo "Installing numpy..."
pip install --no-cache-dir numpy==1.24.3
echo -e "${GREEN}✓ numpy installed${NC}"

echo "Installing pandas..."
pip install --no-cache-dir pandas==2.0.3
echo -e "${GREEN}✓ pandas installed${NC}"

echo "Installing scipy..."
pip install --no-cache-dir scipy==1.11.1
echo -e "${GREEN}✓ scipy installed${NC}"

echo "Installing scikit-learn..."
pip install --no-cache-dir scikit-learn==1.3.0
echo -e "${GREEN}✓ scikit-learn installed${NC}"

echo "Installing tensorflow-cpu (this is the big one, ~10 minutes)..."
echo "Watching memory usage..."

# Install tensorflow-cpu with extra flags to reduce memory
pip install --no-cache-dir --no-build-isolation tensorflow-cpu==2.13.0

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ tensorflow-cpu installed${NC}"
else
    echo -e "${RED}✗ tensorflow-cpu installation failed${NC}"
    echo "Trying alternative installation method..."

    # Try installing with even more conservative settings
    pip install --no-cache-dir --prefer-binary tensorflow-cpu==2.13.0

    if [ $? -ne 0 ]; then
        echo -e "${RED}TensorFlow installation failed. Trying older version...${NC}"
        pip install --no-cache-dir tensorflow-cpu==2.12.0
    fi
fi

echo "Installing keras..."
pip install --no-cache-dir keras==2.13.1
echo -e "${GREEN}✓ keras installed${NC}"

echo "Installing TA-Lib Python wrapper..."
pip install --no-cache-dir TA-Lib==0.4.28
echo -e "${GREEN}✓ TA-Lib installed${NC}"

echo "Installing requests..."
pip install --no-cache-dir requests==2.31.0
echo -e "${GREEN}✓ requests installed${NC}"

echo "Installing urllib3..."
pip install --no-cache-dir urllib3==2.0.4
echo -e "${GREEN}✓ urllib3 installed${NC}"

echo "Installing schedule..."
pip install --no-cache-dir schedule==1.2.0
echo -e "${GREEN}✓ schedule installed${NC}"

echo "Installing python-dotenv..."
pip install --no-cache-dir python-dotenv==1.0.0
echo -e "${GREEN}✓ python-dotenv installed${NC}"

echo "Installing python-dateutil..."
pip install --no-cache-dir python-dateutil==2.8.2
echo -e "${GREEN}✓ python-dateutil installed${NC}"

echo "Installing pytz..."
pip install --no-cache-dir pytz==2023.3
echo -e "${GREEN}✓ pytz installed${NC}"

echo ""
echo "=========================================="
echo -e "${GREEN}Installation Complete!${NC}"
echo "=========================================="
echo ""

# Step 5: Verify installations
echo -e "${YELLOW}Verifying installations...${NC}"
echo ""

python -c "import numpy; print('✓ NumPy:', numpy.__version__)"
python -c "import pandas; print('✓ Pandas:', pandas.__version__)"
python -c "import sklearn; print('✓ Scikit-learn:', sklearn.__version__)"
python -c "import tensorflow as tf; print('✓ TensorFlow:', tf.__version__)"
python -c "import keras; print('✓ Keras:', keras.__version__)"
python -c "import talib; print('✓ TA-Lib: OK')"
python -c "import requests; print('✓ Requests:', requests.__version__)"
python -c "import schedule; print('✓ Schedule: OK')"

echo ""
echo -e "${GREEN}All packages verified!${NC}"
echo ""
echo "Next steps:"
echo "1. python test_connection.py"
echo "2. python train_models.py"
echo "3. python trading_bot.py"
echo ""
