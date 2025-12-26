# Aladdin Forex Trading System

An institutional-grade autonomous Forex trading platform with AI/ML capabilities, inspired by BlackRock's Aladdin system.

## Features

### Core Trading Capabilities
- **High-Frequency Scalping**: Executes 2+ trades per hour on major Forex pairs
- **Multi-Pair Support**: EUR/USD, GBP/USD, USD/JPY, USD/CHF, AUD/USD, USD/CAD, NZD/USD, EUR/GBP
- **24/7 Autonomous Operation**: Runs continuously with automatic error recovery

### AI/ML Components
- **LSTM/Transformer Models**: Deep learning for price prediction
- **Reinforcement Learning**: DQN agent for optimal trading decisions
- **Ensemble Predictions**: Combines multiple models for robust signals
- **Daily Self-Training**: Automatic model retraining with fresh data

### Institutional Features
- **HMM Regime Detection**: Identifies market states (trending, ranging, volatile)
- **Monte Carlo VaR/CVaR**: Advanced risk analytics
- **VWAP/TWAP/POV Execution**: Institutional-grade order execution algorithms
- **Portfolio Optimization**: Mean-variance, risk parity, and max Sharpe optimization

### Agentic Learning System
- **Daily Self-Improvement**: Analyzes trades and learns from outcomes
- **Insight Generation**: Tracks patterns and generates trading insights
- **Adaptive Parameters**: Automatically adjusts risk, leverage, and aggression
- **Trading Mode Switching**: Conservative/Normal/Aggressive based on performance

### Risk Management
- **Dynamic Position Sizing**: Based on volatility and confidence
- **Drawdown Protection**: Automatic trading halt at 20% drawdown
- **Correlation Analysis**: Multi-pair exposure management
- **Stress Testing**: Scenario analysis for extreme market conditions

## Quick Start

### Prerequisites
- Python 3.9+
- MetaTrader 5 (for live trading)
- AWS Account (for deployment)

### Installation

```bash
# Clone the repository
cd /home/ubuntu/forex_trading_system

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install TA-Lib (required for technical indicators)
# On Ubuntu/Debian:
sudo apt-get install libta-lib0-dev
pip install ta-lib
```

### Configuration

Create a `.env` file with your credentials:

```bash
MT5_LOGIN=your_login
MT5_PASSWORD=your_password
MT5_SERVER=MetaQuotes-Demo
```

### Running the System

```bash
# Demo mode (simulated trading)
python main.py --demo

# Backtest mode
python main.py --backtest --symbol EURUSD --bars 5000

# Backtest with Monte Carlo simulation
python main.py --backtest --monte-carlo

# Backtest with walk-forward optimization
python main.py --backtest --walk-forward

# Live trading (requires MT5)
python main.py
```

## Project Structure

```
forex_trading_system/
├── main.py                 # Entry point
├── config.py               # Configuration management
├── data.py                 # Data fetching (MT5, Yahoo Finance, News)
├── indicators.py           # Technical indicators and features
├── models.py               # ML models (LSTM, Transformer, DQN)
├── risk_management.py      # VaR, CVaR, Monte Carlo, position sizing
├── regime_detection.py     # HMM regime detection
├── execution.py            # VWAP/TWAP/POV execution algorithms
├── decisions.py            # FVG, liquidity sweeps, trade logic
├── trading.py              # MT5 integration and trade execution
├── agentic.py              # Daily learning and self-improvement
├── sentiment.py            # News and social sentiment analysis
├── backtesting.py          # Backtesting framework
├── monitoring.py           # Logging and alerting
├── aladdin_core.py         # Unified platform coordinator
├── requirements.txt        # Python dependencies
├── deploy/
│   └── aws_deployment.py   # AWS deployment scripts
└── tests/
    └── test_trading_system.py  # Unit tests
```

## Module Overview

### Data Module (`data.py`)
- MT5 data fetching with automatic reconnection
- Yahoo Finance fallback for historical data
- News sentiment from RSS feeds
- Multi-timeframe data aggregation

### Indicators Module (`indicators.py`)
- 30+ technical indicators (RSI, MACD, Bollinger Bands, ADX, ATR, etc.)
- Feature engineering for ML models
- Sequence preparation for LSTM/Transformer

### Models Module (`models.py`)
- LSTM price predictor with attention
- Transformer-based predictor
- DQN reinforcement learning agent
- Ensemble predictor combining all models

### Risk Management (`risk_management.py`)
- Historical and parametric VaR
- Conditional VaR (Expected Shortfall)
- Monte Carlo simulation
- Portfolio optimization (mean-variance, risk parity)
- Dynamic position sizing

### Regime Detection (`regime_detection.py`)
- Hidden Markov Model for regime identification
- Clustering-based regime detection
- Strategy weight recommendations per regime
- Risk adjustment factors

### Execution (`execution.py`)
- TWAP (Time-Weighted Average Price)
- VWAP (Volume-Weighted Average Price)
- POV (Percentage of Volume)
- Iceberg orders
- Sniper execution

### Decisions (`decisions.py`)
- Fair Value Gap (FVG) detection
- Liquidity sweep detection
- Multi-timeframe analysis
- Dynamic leverage calculation
- Trailing stop management

### Agentic Learning (`agentic.py`)
- Trade journal with persistence
- Insight generation and tracking
- Performance analysis
- Adaptive parameter updates
- Trading mode management

### Sentiment Analysis (`sentiment.py`)
- VADER sentiment analysis
- Forex-specific lexicon
- News article analysis
- Social media sentiment
- Currency-specific sentiment scores

### Backtesting (`backtesting.py`)
- Realistic cost modeling (spread, commission, slippage)
- Walk-forward optimization
- Monte Carlo simulation
- Strategy and regime performance breakdown

## Trading Strategies

### Trend Following
- Uses ADX > 25 to confirm trend
- MACD histogram for direction
- RSI filter to avoid overbought/oversold entries

### Mean Reversion
- Bollinger Bands for range identification
- RSI extremes (< 30 or > 70)
- ADX < 20 to confirm ranging market

### FVG Entry
- Identifies fair value gaps (imbalances)
- Trades displacement moves
- Multi-timeframe confirmation

### Liquidity Sweep
- Detects stop hunts at recent highs/lows
- Waits for reversal confirmation
- Trades institutional order flow

## Risk Disclaimer

**IMPORTANT**: Trading foreign exchange on margin carries a high level of risk and may not be suitable for all investors. The high degree of leverage can work against you as well as for you. Before deciding to trade foreign exchange, you should carefully consider your investment objectives, level of experience, and risk appetite.

- Past performance is not indicative of future results
- Only trade with money you can afford to lose
- This system is for educational purposes
- Always test thoroughly before live trading
- The developers are not responsible for any losses

## AWS Deployment

See `deploy/aws_deployment.py` for detailed deployment instructions.

### Quick Deploy

```bash
# Print deployment guide
python deploy/aws_deployment.py --guide

# Deploy infrastructure (requires AWS credentials)
python deploy/aws_deployment.py --deploy --stack-name forex-trading

# Upload code to S3
python deploy/aws_deployment.py --upload --bucket your-bucket-name
```

### Estimated Costs
- EC2 t3.micro: ~$8.50/month (or free tier)
- S3 storage: ~$0.50/month
- CloudWatch: ~$1.00/month
- Lambda: Free tier
- **Total: ~$10/month**

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html

# Run specific test class
pytest tests/test_trading_system.py::TestTechnicalIndicators -v
```

## Performance Targets

- Win Rate: > 60%
- Monthly Return: > 30% (backtested)
- Max Drawdown: < 20%
- Sharpe Ratio: > 1.5
- Trades per Day: 48+

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## License

This project is for educational purposes only. Use at your own risk.

## Support

For questions or issues, please open a GitHub issue.
