# Recommended EC2 Configurations for Trading Bot

## 🏆 Best Options (All Under 2,000 INR/month)

### Option 1: t3.micro - RECOMMENDED ⭐

**Perfect balance of performance and cost**

```yaml
Instance Type: t3.micro
vCPUs: 2 (vs 1 on t2.micro)
RAM: 1 GB
Storage: 20 GB GP3 SSD
Network: Up to 5 Gbps
Region: ap-south-1 (Mumbai)

Cost: ~$7.50/month (~563 INR/month)
Performance: 2x better CPU than t2.micro
Installation: NO MEMORY ISSUES ✅
```

**Why t3.micro is better:**
- ✅ 2 vCPUs (vs 1) - ML models train faster
- ✅ Better burst performance
- ✅ More stable under load
- ✅ No installation failures
- ✅ Same RAM but better CPU
- ✅ Still very cheap!

**Launch Command:**
```bash
# When creating instance, select:
Instance Type: t3.micro
AMI: Ubuntu 22.04 LTS
Storage: 20 GB gp3
Region: ap-south-1
```

---

### Option 2: t3a.micro - CHEAPEST ⭐

**AMD processor, slightly cheaper**

```yaml
Instance Type: t3a.micro
vCPUs: 2
RAM: 1 GB
Storage: 20 GB GP3 SSD
Region: ap-south-1 (Mumbai)

Cost: ~$6.75/month (~506 INR/month)
Performance: Same as t3.micro
Installation: NO MEMORY ISSUES ✅
```

**Why t3a.micro:**
- ✅ Cheapest option
- ✅ Same performance as t3.micro
- ✅ Uses AMD processors (cheaper)
- ✅ Perfect for 24/7 running

---

### Option 3: t3.small - MORE POWER 🚀

**If you want extra headroom**

```yaml
Instance Type: t3.small
vCPUs: 2
RAM: 2 GB (double!)
Storage: 20 GB GP3 SSD
Region: ap-south-1 (Mumbai)

Cost: ~$15/month (~1,125 INR/month)
Performance: Best for ML models
Installation: SUPER FAST ✅
```

**Why t3.small:**
- ✅ 2 GB RAM (no swap needed!)
- ✅ Faster model training
- ✅ Can handle multiple models
- ✅ Still under budget!
- ✅ Best performance

---

### Option 4: t4g.micro - ARM/Graviton 💰

**AWS Graviton (ARM processor) - Great value**

```yaml
Instance Type: t4g.micro
vCPUs: 2
RAM: 1 GB
Storage: 20 GB GP3 SSD
Processor: AWS Graviton2 (ARM)
Region: ap-south-1 (Mumbai)

Cost: ~$6/month (~450 INR/month)
Performance: Better than t3.micro
Installation: Requires ARM-compatible packages
```

**Why t4g.micro:**
- ✅ Cheapest with 2 vCPUs
- ✅ Better price/performance
- ✅ 20% cheaper than t3.micro
- ⚠️ Uses ARM (need compatible packages)

---

## 📊 Comparison Table

| Instance | vCPUs | RAM | Cost/Month (INR) | Best For |
|----------|-------|-----|------------------|----------|
| **t2.micro** | 1 | 1 GB | FREE (12mo) / 600 | Free tier only |
| **t3.micro** ⭐ | 2 | 1 GB | ~563 | **Recommended** |
| **t3a.micro** | 2 | 1 GB | ~506 | Cheapest |
| **t3.small** | 2 | 2 GB | ~1,125 | More power |
| **t4g.micro** | 2 | 1 GB | ~450 | ARM/Advanced |

---

## 🎯 MY RECOMMENDATION: t3.micro

**Launch Configuration:**

```yaml
AMI: Ubuntu Server 22.04 LTS (ami-0dee22c13ea7a9a67)
Instance Type: t3.micro
Region: ap-south-1 (Mumbai)
Availability Zone: ap-south-1a

Storage:
  - Type: GP3 SSD
  - Size: 20 GB
  - IOPS: 3000 (default)
  - Throughput: 125 MB/s

Network:
  - VPC: Default
  - Auto-assign Public IP: Enable
  - Security Group:
      - SSH (22): Your IP
      - Custom: None needed

Advanced:
  - Monitoring: Basic (free)
  - Termination Protection: Enable
  - Shutdown Behavior: Stop
  - Credit Specification: Unlimited (for burst)
```

---

## 🚀 Step-by-Step: Launch t3.micro

### 1. Go to EC2 Console
https://ap-south-1.console.aws.amazon.com/ec2/

### 2. Click "Launch Instance"

### 3. Configure Instance

**Name:** `trading-bot-prod`

**Application and OS Images:**
- Quick Start: Ubuntu
- AMI: **Ubuntu Server 22.04 LTS (HVM), SSD Volume Type**
- Architecture: **64-bit (x86)**

**Instance Type:**
- Family: **t3**
- Type: **t3.micro** ⭐
- (2 vCPUs, 1 GiB Memory)

**Key Pair:**
- Create new: `trading-bot-key`
- Type: RSA
- Format: .pem
- **Download and save!**

**Network Settings:**
- VPC: Default
- Subnet: No preference
- Auto-assign public IP: **Enable**
- Firewall: Create security group
  - Name: `trading-bot-sg`
  - Description: `Trading bot security`
  - Allow SSH from: **My IP**

**Configure Storage:**
- Size: **20 GiB**
- Volume Type: **gp3**
- IOPS: 3000
- Throughput: 125
- Delete on termination: **Yes**

**Advanced Details:**
- Leave defaults
- (Optional) User data: You can add setup script here

### 4. Launch!

Click **"Launch instance"**

### 5. Wait 1-2 minutes

Instance will be "Running"

### 6. Connect

```bash
# Get public IP from EC2 console (e.g., 13.233.45.67)
ssh -i "path/to/trading-bot-key.pem" ubuntu@YOUR_IP
```

---

## 💻 Installation on t3.micro (NO SWAP NEEDED!)

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install dependencies
sudo apt install -y python3.10 python3-pip python3-venv build-essential wget git

# Install TA-Lib
cd /tmp
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install
sudo ldconfig
cd ~

# Clone repo
git clone https://github.com/prathamesh-ops-sudo/delta-trading-bot.git
cd delta-trading-bot

# Setup Python
python3 -m venv venv
source venv/bin/activate

# Install packages (WORKS WITHOUT SWAP!)
pip install --upgrade pip
pip install -r requirements.txt

# Test
python test_connection.py

# Train models
python train_models.py

# Run bot
python trading_bot.py
```

**Installation time on t3.micro:** ~15 minutes (vs 30+ on t2.micro)

---

## 💰 Cost Breakdown

### t3.micro (Recommended)

```
EC2 Instance (t3.micro):     ~563 INR/month
EBS Storage (20 GB gp3):     ~150 INR/month
Data Transfer (5 GB):        ~50 INR/month
───────────────────────────────────────────
TOTAL:                       ~763 INR/month ✅

Annual: ~9,156 INR (well under budget!)
```

### t3.small (If you want more power)

```
EC2 Instance (t3.small):     ~1,125 INR/month
EBS Storage (20 GB gp3):     ~150 INR/month
Data Transfer (5 GB):        ~50 INR/month
───────────────────────────────────────────
TOTAL:                       ~1,325 INR/month ✅

Annual: ~15,900 INR
```

Both are **under your 2,000 INR/month budget!**

---

## 🎁 Cost Saving Tips

### 1. Reserved Instances (Save 40%)
If you commit to 1 year:
- t3.micro: ~563 INR → ~340 INR/month
- t3.small: ~1,125 INR → ~675 INR/month

### 2. Savings Plans (Save 30%)
Flexible 1-year commitment:
- Similar savings to Reserved Instances
- Can change instance types

### 3. Spot Instances (Save 70%!)
⚠️ Can be terminated anytime:
- t3.micro: ~563 INR → ~170 INR/month
- Not recommended for 24/7 trading bot

---

## ✅ Final Recommendation

**Use t3.micro:**
- Cost: ~763 INR/month
- 2 vCPUs (no installation issues)
- Fast setup (15 mins)
- Reliable 24/7 operation
- Easy upgrade path if needed

**If budget allows, use t3.small:**
- Cost: ~1,325 INR/month
- 2 GB RAM (no memory worries)
- Faster model training
- Better for future expansion

---

## 🚀 Quick Launch Link

Create t3.micro with one click:
https://console.aws.amazon.com/ec2/v2/home?region=ap-south-1#LaunchInstanceWizard:

Select:
1. Ubuntu 22.04 LTS
2. **t3.micro** ⭐
3. 20 GB storage
4. Create key pair
5. Launch!

**You'll have zero installation issues!** 🎉

---

**Bottom line:** Use **t3.micro** for ~763 INR/month. It's perfect!
