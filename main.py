#!/usr/bin/env python3
"""
Aladdin Forex Trading System - Main Entry Point
Institutional-grade autonomous trading platform with AI/ML capabilities
"""

import os
import sys
import argparse
import logging
import signal
import time
from datetime import datetime, timedelta
import threading
import json

from config import config, DISCLAIMER
from monitoring import monitoring, LoggingSetup
from data import data_manager
from aladdin_core import aladdin
from trading import trading_engine
from agentic import agentic_system
from backtesting import BacktestEngine, create_sample_strategy, WalkForwardOptimizer, MonteCarloBacktest

logger = logging.getLogger(__name__)


class TradingSystem:
    """Main trading system controller"""
    
    def __init__(self):
        self.is_running = False
        self.start_time = None
        self.shutdown_event = threading.Event()
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, initiating shutdown...")
        self.shutdown()
    
    def start(self):
        """Start the trading system"""
        print("=" * 60)
        print("ALADDIN FOREX TRADING SYSTEM")
        print("Institutional-Grade Autonomous Trading Platform")
        print("=" * 60)
        print()
        print(DISCLAIMER)
        print()
        
        logger.info("Starting Aladdin Trading System...")
        
        self.is_running = True
        self.start_time = datetime.now()
        
        # Start monitoring
        monitoring.start()
        logger.info("Monitoring system started")
        
        # Start Aladdin core (includes all agents and trading engine)
        aladdin.start()
        logger.info("Aladdin core platform started")
        
        # Main loop
        self._main_loop()
    
    def _main_loop(self):
        """Main system loop"""
        last_status_time = datetime.now()
        last_daily_reset = datetime.now().date()
        
        while self.is_running and not self.shutdown_event.is_set():
            try:
                current_time = datetime.now()
                
                # Print status every 5 minutes
                if (current_time - last_status_time).seconds >= 300:
                    self._print_status()
                    last_status_time = current_time
                
                # Daily reset at midnight UTC
                if current_time.date() != last_daily_reset:
                    self._daily_reset()
                    last_daily_reset = current_time.date()
                
                # Check for daily learning cycle (at configured hour)
                if current_time.hour == config.ml.retrain_hour_utc:
                    self._run_daily_learning()
                
                # Sleep for a bit
                time.sleep(10)
                
            except Exception as e:
                logger.error(f"Main loop error: {e}")
                time.sleep(30)
    
    def _print_status(self):
        """Print current system status"""
        try:
            dashboard = aladdin.get_dashboard_data()
            
            print("\n" + "=" * 50)
            print(f"STATUS UPDATE - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 50)
            
            account = dashboard.get('account', {})
            print(f"Balance: ${account.get('balance', 0):.2f}")
            print(f"Equity: ${account.get('equity', 0):.2f}")
            
            session = dashboard.get('session', {})
            print(f"Open Positions: {session.get('open_positions', 0)}")
            print(f"Trades Today: {session.get('trades_opened', 0)}")
            print(f"Win Rate: {session.get('win_rate', 0):.1%}")
            print(f"Total Profit: ${session.get('total_profit', 0):.2f}")
            
            regime = dashboard.get('regime', {})
            print(f"Market Regime: {regime.get('regime', 'unknown')}")
            print(f"Trading Mode: {dashboard.get('trading_params', {}).get('trading_mode', 'normal')}")
            
            print("=" * 50 + "\n")
            
        except Exception as e:
            logger.error(f"Error printing status: {e}")
    
    def _daily_reset(self):
        """Perform daily reset tasks"""
        logger.info("Performing daily reset...")
        monitoring.trading_monitor.reset_daily()
    
    def _run_daily_learning(self):
        """Run daily learning cycle"""
        logger.info("Running daily learning cycle...")
        try:
            trades = trading_engine.get_trades_for_learning()
            account_info = trading_engine.broker._get_account_info()
            balance = account_info.get('balance', config.trading.initial_balance)
            
            report = agentic_system.run_daily_learning_cycle(trades, balance)
            
            logger.info(f"Daily learning complete: {report.total_trades} trades analyzed, "
                       f"Win rate: {report.win_rate:.1%}")
            
        except Exception as e:
            logger.error(f"Daily learning error: {e}")
    
    def shutdown(self):
        """Shutdown the trading system"""
        logger.info("Shutting down trading system...")
        
        self.is_running = False
        self.shutdown_event.set()
        
        # Stop Aladdin core
        aladdin.stop()
        
        # Stop monitoring
        monitoring.stop()
        
        # Print final summary
        self._print_final_summary()
        
        logger.info("Trading system shutdown complete")
    
    def _print_final_summary(self):
        """Print final session summary"""
        if self.start_time is None:
            return
        
        runtime = datetime.now() - self.start_time
        
        print("\n" + "=" * 60)
        print("SESSION SUMMARY")
        print("=" * 60)
        print(f"Runtime: {runtime}")
        
        try:
            summary = trading_engine.get_session_summary()
            print(f"Final Balance: ${summary.get('balance', 0):.2f}")
            print(f"Total Trades: {summary.get('trades_closed', 0)}")
            print(f"Win Rate: {summary.get('win_rate', 0):.1%}")
            print(f"Total Profit: ${summary.get('total_profit', 0):.2f}")
        except:
            pass
        
        print("=" * 60 + "\n")


def run_backtest(args):
    """Run backtesting mode"""
    print("=" * 60)
    print("BACKTEST MODE")
    print("=" * 60)
    
    # Setup logging
    LoggingSetup.setup(log_dir="./logs", log_level=logging.INFO)
    
    # Get historical data
    logger.info(f"Fetching historical data for {args.symbol}...")
    
    df = data_manager.get_ohlcv(
        symbol=args.symbol,
        timeframe=args.timeframe,
        count=args.bars
    )
    
    if df.empty:
        logger.error("No data available for backtesting")
        return
    
    logger.info(f"Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")
    
    # Create strategy
    strategy = create_sample_strategy(
        rsi_oversold=30,
        rsi_overbought=70,
        adx_threshold=25,
        sl_atr_mult=2.0,
        rr_ratio=2.0
    )
    
    # Run backtest
    logger.info("Running backtest...")
    engine = BacktestEngine(initial_balance=args.balance)
    result = engine.run(df, strategy, args.symbol)
    
    # Print results
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)
    print(f"Period: {result.start_date} to {result.end_date}")
    print(f"Initial Balance: ${result.initial_balance:.2f}")
    print(f"Final Balance: ${result.final_balance:.2f}")
    print(f"Total Return: {result.total_return_pct:.2%}")
    print(f"Total Trades: {result.total_trades}")
    print(f"Win Rate: {result.win_rate:.2%}")
    print(f"Profit Factor: {result.profit_factor:.2f}")
    print(f"Max Drawdown: {result.max_drawdown_pct:.2%}")
    print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
    print(f"Sortino Ratio: {result.sortino_ratio:.2f}")
    print(f"Calmar Ratio: {result.calmar_ratio:.2f}")
    print(f"Trades/Day: {result.trades_per_day:.2f}")
    print(f"Avg Trade Duration: {result.avg_trade_duration:.1f} minutes")
    print(f"Expectancy: ${result.expectancy:.2f}")
    
    # Strategy performance
    if result.strategy_performance:
        print("\nStrategy Performance:")
        for strategy, perf in result.strategy_performance.items():
            print(f"  {strategy}: {perf['trades']} trades, "
                  f"{perf['win_rate']:.1%} win rate, ${perf['profit']:.2f} profit")
    
    # Regime performance
    if result.regime_performance:
        print("\nRegime Performance:")
        for regime, perf in result.regime_performance.items():
            print(f"  {regime}: {perf['trades']} trades, "
                  f"{perf['win_rate']:.1%} win rate, ${perf['profit']:.2f} profit")
    
    # Monthly returns
    if result.monthly_returns:
        print("\nMonthly Returns:")
        for month, ret in sorted(result.monthly_returns.items())[-12:]:
            print(f"  {month}: ${ret:.2f}")
    
    # Run Monte Carlo simulation
    if args.monte_carlo and result.trades:
        print("\n" + "-" * 40)
        print("MONTE CARLO SIMULATION")
        print("-" * 40)
        
        mc = MonteCarloBacktest(num_simulations=1000)
        mc_results = mc.simulate(result.trades, args.balance)
        
        print(f"Mean Final Balance: ${mc_results['mean_final_balance']:.2f}")
        print(f"5th Percentile: ${mc_results['percentile_5']:.2f}")
        print(f"95th Percentile: ${mc_results['percentile_95']:.2f}")
        print(f"Probability of Profit: {mc_results['prob_profit']:.1%}")
        print(f"Probability of Ruin (<50%): {mc_results['prob_ruin']:.1%}")
        print(f"Mean Max Drawdown: {mc_results['mean_max_drawdown']:.1%}")
    
    # Run walk-forward optimization
    if args.walk_forward:
        print("\n" + "-" * 40)
        print("WALK-FORWARD OPTIMIZATION")
        print("-" * 40)
        
        wfo = WalkForwardOptimizer(train_ratio=0.7, num_folds=5)
        
        param_grid = {
            'rsi_oversold': [25, 30, 35],
            'rsi_overbought': [65, 70, 75],
            'adx_threshold': [20, 25, 30]
        }
        
        wfo_results = wfo.optimize(df, create_sample_strategy, param_grid, args.symbol, args.balance)
        
        print(f"Avg Test Return: {wfo_results['avg_test_return']:.2%}")
        print(f"Std Test Return: {wfo_results['std_test_return']:.2%}")
        print(f"Avg Sharpe: {wfo_results['avg_sharpe']:.2f}")
        print(f"Avg Win Rate: {wfo_results['avg_win_rate']:.1%}")
        print(f"Robustness Score: {wfo_results['robustness_score']:.2f}")
    
    print("\n" + "=" * 60)
    print("Backtest complete!")


def run_demo(args):
    """Run demo mode with continuous simulated trading (24/7)"""
    print("=" * 60)
    print("DEMO MODE - CONTINUOUS TRADING SIMULATION")
    print("=" * 60)
    print()
    print("Running in demo mode with simulated data...")
    print("This simulates 24/7 trading without live execution.")
    print()
    
    LoggingSetup.setup(log_dir="./logs", log_level=logging.INFO)
    
    from decisions import decision_engine, TradeDirection
    from regime_detection import regime_manager
    from risk_management import risk_manager
    from sentiment import sentiment_manager
    import numpy as np
    import pandas as pd
    
    try:
        from pattern_miner import pattern_miner
        PATTERN_MINER_AVAILABLE = True
    except ImportError:
        PATTERN_MINER_AVAILABLE = False
        pattern_miner = None
    
    try:
        from bedrock_ai import bedrock_ai
        BEDROCK_AVAILABLE = True
    except ImportError:
        BEDROCK_AVAILABLE = False
        bedrock_ai = None
    
    logger.info(f"Demo mode started - PatternMiner: {PATTERN_MINER_AVAILABLE}, BedrockAI: {BEDROCK_AVAILABLE}")
    
    symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF']
    account_balance = 100.0
    simulated_trades = []
    last_status_time = datetime.now()
    last_daily_reset = datetime.now().date()
    iteration = 0
    
    print("Starting continuous demo trading loop...")
    print("Press Ctrl+C to stop")
    print()
    
    while True:
        try:
            iteration += 1
            current_time = datetime.now()
            
            for symbol in symbols:
                np.random.seed(int(time.time()) + hash(symbol) % 1000)
                n_samples = 500
                
                base_price = {'EURUSD': 1.1000, 'GBPUSD': 1.2700, 'USDJPY': 150.00, 'USDCHF': 0.8800}.get(symbol, 1.0)
                prices = [base_price]
                for i in range(n_samples - 1):
                    change = np.random.normal(0.0001, 0.001)
                    prices.append(prices[-1] * (1 + change))
                
                df = pd.DataFrame({
                    'open': prices,
                    'high': [p * (1 + abs(np.random.normal(0, 0.002))) for p in prices],
                    'low': [p * (1 - abs(np.random.normal(0, 0.002))) for p in prices],
                    'close': prices,
                    'volume': np.random.randint(1000, 10000, n_samples)
                })
                df.index = pd.date_range(end=current_time, periods=n_samples, freq='H')
                df['datetime'] = df.index
                
                regime = regime_manager.detect_regime(df)
                
                mtf_data = {'H1': df, 'H4': df.resample('4H').agg({
                    'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
                }).dropna()}
                
                signal = decision_engine.analyze_market(symbol, mtf_data, account_balance=account_balance)
                
                if signal and signal.confidence > 0.6:
                    trade_result = np.random.choice(['win', 'loss'], p=[0.55, 0.45])
                    if trade_result == 'win':
                        profit = abs(signal.take_profit - signal.entry_price) * signal.position_size * 0.8
                    else:
                        profit = -abs(signal.stop_loss - signal.entry_price) * signal.position_size
                    
                    account_balance += profit
                    simulated_trades.append({
                        'symbol': symbol,
                        'direction': signal.direction.name,
                        'profit': profit,
                        'confidence': signal.confidence,
                        'strategy': signal.strategy,
                        'timestamp': current_time
                    })
                    
                    logger.info(f"[DEMO] {symbol} {signal.direction.name} | "
                               f"Confidence: {signal.confidence:.2f} | "
                               f"Strategy: {signal.strategy} | "
                               f"P/L: ${profit:.2f} | "
                               f"Balance: ${account_balance:.2f}")
                
                if PATTERN_MINER_AVAILABLE and pattern_miner and iteration % 100 == 0:
                    try:
                        patterns = pattern_miner.analyze_historical_data(df, symbol)
                        if patterns:
                            logger.info(f"[DEMO] Learned {len(patterns)} patterns for {symbol}")
                    except Exception as e:
                        logger.warning(f"Pattern learning error: {e}")
            
            if (current_time - last_status_time).seconds >= 300:
                wins = len([t for t in simulated_trades if t['profit'] > 0])
                total = len(simulated_trades)
                win_rate = wins / total if total > 0 else 0
                
                print("\n" + "=" * 50)
                print(f"DEMO STATUS - {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
                print("=" * 50)
                print(f"Account Balance: ${account_balance:.2f}")
                print(f"Total Trades: {total}")
                print(f"Win Rate: {win_rate:.1%}")
                print(f"Trading Mode: {agentic_system.get_trading_parameters()['trading_mode']}")
                print(f"PatternMiner: {PATTERN_MINER_AVAILABLE}")
                print(f"BedrockAI: {BEDROCK_AVAILABLE}")
                if PATTERN_MINER_AVAILABLE and pattern_miner:
                    active_patterns = pattern_miner.get_all_active_patterns()
                    print(f"Active Patterns: {len(active_patterns)}")
                print("=" * 50 + "\n")
                
                last_status_time = current_time
            
            if current_time.date() != last_daily_reset:
                logger.info("Running daily learning cycle...")
                try:
                    trades_today = [t for t in simulated_trades 
                                   if t['timestamp'].date() == last_daily_reset]
                    
                    historical_data = {}
                    for symbol in symbols:
                        np.random.seed(int(time.time()) + hash(symbol) % 1000)
                        n_samples = 1000
                        base_price = {'EURUSD': 1.1000, 'GBPUSD': 1.2700, 'USDJPY': 150.00, 'USDCHF': 0.8800}.get(symbol, 1.0)
                        prices = [base_price]
                        for i in range(n_samples - 1):
                            change = np.random.normal(0.0001, 0.001)
                            prices.append(prices[-1] * (1 + change))
                        
                        df = pd.DataFrame({
                            'open': prices,
                            'high': [p * (1 + abs(np.random.normal(0, 0.002))) for p in prices],
                            'low': [p * (1 - abs(np.random.normal(0, 0.002))) for p in prices],
                            'close': prices,
                            'volume': np.random.randint(1000, 10000, n_samples)
                        })
                        df.index = pd.date_range(end=current_time, periods=n_samples, freq='H')
                        df['datetime'] = df.index
                        historical_data[symbol] = df
                    
                    report = agentic_system.run_daily_learning_cycle(
                        trades_today, account_balance, historical_data
                    )
                    logger.info(f"Daily learning complete: {report.total_trades} trades analyzed")
                except Exception as e:
                    logger.error(f"Daily learning error: {e}")
                
                last_daily_reset = current_time.date()
            
            time.sleep(30)
            
        except KeyboardInterrupt:
            print("\n" + "=" * 60)
            print("Demo mode stopped by user")
            print(f"Final Balance: ${account_balance:.2f}")
            print(f"Total Trades: {len(simulated_trades)}")
            print("=" * 60)
            break
        except Exception as e:
            logger.error(f"Demo loop error: {e}")
            time.sleep(60)


def run_mt5_bridge(args):
    """Run with MT5 Bridge for real trading execution"""
    print("=" * 60)
    print("MT5 BRIDGE MODE - REAL TRADING")
    print("=" * 60)
    print()
    print("Connecting to MT5 via Bridge API...")
    print("Make sure the TradingBridgeEA is running in your MT5 terminal!")
    print()
    
    LoggingSetup.setup(log_dir="./logs", log_level=logging.INFO)
    
    from mt5_bridge.mt5_broker_adapter import MT5BrokerAdapter
    from mt5_bridge.api_server import app as bridge_app, bridge_state
    from decisions import decision_engine, TradeDirection
    from regime_detection import regime_manager
    from risk_management import risk_manager
    from trading_captain import trading_captain, TradingMode
    from self_reflection import reflection_engine
    import numpy as np
    import pandas as pd
    import threading
    
    try:
        from pattern_miner import pattern_miner
        PATTERN_MINER_AVAILABLE = True
    except ImportError:
        PATTERN_MINER_AVAILABLE = False
        pattern_miner = None
    
    try:
        from bedrock_ai import bedrock_ai
        BEDROCK_AVAILABLE = True
    except ImportError:
        BEDROCK_AVAILABLE = False
        bedrock_ai = None
    
    # Phoenix components - 24/7 learning and enhanced RL
    try:
        from knowledge_acquisition import knowledge_engine
        KNOWLEDGE_ENGINE_AVAILABLE = True
        # Start background knowledge collection (every 30 minutes)
        knowledge_engine.start_background_collection(interval_minutes=30)
        logger.info("Knowledge acquisition engine started - 24/7 learning enabled")
    except ImportError as e:
        KNOWLEDGE_ENGINE_AVAILABLE = False
        knowledge_engine = None
        logger.warning(f"Knowledge acquisition not available: {e}")
    
    # Advanced Knowledge - Economic Calendar, Topic Sentiment, Cross-Asset Regime
    try:
        from advanced_knowledge import get_advanced_knowledge
        advanced_knowledge = get_advanced_knowledge()
        ADVANCED_KNOWLEDGE_AVAILABLE = True
        # Start background updates (every 15 minutes)
        advanced_knowledge.start_background_updates(interval_minutes=15)
        logger.info("Advanced knowledge engine started - Economic calendar, topic sentiment, cross-asset regime enabled")
    except ImportError as e:
        ADVANCED_KNOWLEDGE_AVAILABLE = False
        advanced_knowledge = None
        logger.warning(f"Advanced knowledge not available: {e}")
    
    # Adaptive Learning - Online/Offline Learning System
    try:
        from adaptive_learning import get_adaptive_learning
        adaptive_learning = get_adaptive_learning()
        ADAPTIVE_LEARNING_AVAILABLE = True
        # Start background learning updates (every 60 seconds)
        adaptive_learning.start_background_updates(interval_seconds=60)
        logger.info(f"Adaptive learning engine started - Mode: {adaptive_learning.get_learning_mode().value}, Market open: {adaptive_learning.is_market_open()}")
    except ImportError as e:
        ADAPTIVE_LEARNING_AVAILABLE = False
        adaptive_learning = None
        logger.warning(f"Adaptive learning not available: {e}")
    
    # Trade Journaling - Human-like decision making with uncertainty governor
    try:
        from trade_journaling import get_journaling_engine
        journaling_engine = get_journaling_engine()
        TRADE_JOURNALING_AVAILABLE = True
        # Start background journal analysis (every 30 minutes)
        journaling_engine.start_background_analysis(interval_minutes=30)
        logger.info(f"Trade journaling engine started - Uncertainty governor active, {len(journaling_engine.journal.entries)} journal entries")
    except ImportError as e:
        TRADE_JOURNALING_AVAILABLE = False
        journaling_engine = None
        logger.warning(f"Trade journaling not available: {e}")
    
    try:
        from phoenix_brain import phoenix_brain, TradingState
        PHOENIX_BRAIN_AVAILABLE = True
        logger.info("Phoenix brain initialized - brutal reward function active")
    except ImportError as e:
        PHOENIX_BRAIN_AVAILABLE = False
        phoenix_brain = None
        TradingState = None
        logger.warning(f"Phoenix brain not available: {e}")
    
    try:
        from vector_memory import vector_memory, TradeExperience
        VECTOR_MEMORY_AVAILABLE = True
        logger.info("Vector memory initialized - FAISS experience replay enabled")
    except ImportError as e:
        VECTOR_MEMORY_AVAILABLE = False
        vector_memory = None
        TradeExperience = None
        logger.warning(f"Vector memory not available: {e}")
    
    logger.info(f"MT5 Bridge mode - PatternMiner: {PATTERN_MINER_AVAILABLE}, BedrockAI: {BEDROCK_AVAILABLE}")
    logger.info(f"Phoenix components - Knowledge: {KNOWLEDGE_ENGINE_AVAILABLE}, Brain: {PHOENIX_BRAIN_AVAILABLE}, Memory: {VECTOR_MEMORY_AVAILABLE}")
    logger.info(f"Advanced Knowledge - Calendar/Sentiment/CrossAsset: {ADVANCED_KNOWLEDGE_AVAILABLE}")
    logger.info(f"Adaptive Learning - Online/Offline: {ADAPTIVE_LEARNING_AVAILABLE}")
    logger.info(f"Trade Journaling - Uncertainty Governor: {TRADE_JOURNALING_AVAILABLE}")
    logger.info(f"Trading Captain initialized - Mode: {trading_captain.mode.value}")
    
    # Start the bridge API server in a background thread
    api_port = int(os.environ.get('MT5_BRIDGE_PORT', 5000))
    
    def run_api_server():
        bridge_app.run(host='0.0.0.0', port=api_port, debug=False, threaded=True, use_reloader=False)
    
    api_thread = threading.Thread(target=run_api_server, daemon=True)
    api_thread.start()
    logger.info(f"Bridge API server started on port {api_port}")
    
    # Wait for API to start
    time.sleep(2)
    
    # Create MT5 broker adapter
    broker = MT5BrokerAdapter(api_url=f"http://localhost:{api_port}")
    
    symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF']
    executed_trades = []
    last_status_time = datetime.now()
    last_daily_reset = datetime.now().date()
    iteration = 0
    
    print("Waiting for MT5 EA to connect...")
    print(f"API URL for EA: http://<your-ec2-ip>:{api_port}")
    print()
    print("Starting trading loop...")
    print("Press Ctrl+C to stop")
    print()
    
    while True:
        try:
            iteration += 1
            current_time = datetime.now()
            
            # Check MT5 connection
            mt5_connected = broker.is_connected()
            
            if mt5_connected:
                # Get real balance from MT5
                account_balance = broker.get_balance()
                account_equity = broker.get_equity()
                open_positions = broker.get_open_positions()
                
                # Update Trading Captain with real balance
                trading_captain.update_balance(account_balance)
                
                if account_balance <= 0:
                    logger.warning("MT5 balance is 0 - waiting for account data...")
                    time.sleep(5)
                    continue
                
                # Analyze each symbol using REAL market data from MT5
                for symbol in symbols:
                    try:
                        # Get REAL market data from MT5 via the bridge API
                        quote, bars = broker.get_market_data(symbol)
                        
                        if not bars or len(bars) < 50:
                            logger.debug(f"Insufficient market data for {symbol} - waiting for EA to send data")
                            continue
                        
                        # Convert bars to DataFrame for analysis
                        df = pd.DataFrame(bars)
                        # Bars from API are dicts with keys: t, o, h, l, c, v
                        # Rename to full names for indicator calculations
                        df = df.rename(columns={'t': 'timestamp', 'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'})
                        
                        # Ensure we only have the expected columns (drop timestamp from price data)
                        price_columns = ['open', 'high', 'low', 'close', 'volume']
                        for col in price_columns:
                            if col not in df.columns:
                                logger.warning(f"Missing column {col} in market data for {symbol}")
                                continue
                        
                        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
                        df = df.set_index('datetime')
                        df = df.sort_index()
                        
                        # Ensure price columns are numeric (not timestamps)
                        for col in ['open', 'high', 'low', 'close']:
                            if col in df.columns:
                                df[col] = pd.to_numeric(df[col], errors='coerce')
                        
                        # Get current spread for cost-aware trading
                        current_spread = quote.get('spread', 0.0) if quote else 0.0
                        stop_level = quote.get('stop_level', 0) if quote else 0
                        point = quote.get('point', 0.00001) if quote else 0.00001
                        
                        # Log that we're using real data
                        if iteration % 60 == 0:
                            logger.info(f"[REAL DATA] {symbol}: {len(df)} bars, spread={current_spread:.1f} pts, bid={quote.get('bid', 0):.5f}")
                        
                        # Detect regime
                        regime = regime_manager.detect_regime(df)
                        
                        # Create multi-timeframe data
                        mtf_data = {'H1': df, 'H4': df.resample('4H').agg({
                            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
                        }).dropna()}
                        
                        # Get trading signal
                        signal = decision_engine.analyze_market(symbol, mtf_data, account_balance=account_balance)
                        
                        # Use Trading Captain for intelligent decision-making
                        if signal:
                            # Check if Trading Captain approves this trade
                            should_trade, reason = trading_captain.should_take_trade(signal, len(open_positions))
                            
                            if should_trade:
                                # Check if we already have a position in this symbol
                                existing_position = any(p.get('symbol') == symbol for p in open_positions)
                                
                                if not existing_position:
                                    # Generate trade thesis (intelligent explanation)
                                    regime_name = regime.name if regime else 'unknown'
                                    indicators = {
                                        'rsi': signal.rsi,
                                        'adx': signal.adx,
                                        'atr': signal.atr
                                    }
                                    thesis = trading_captain.generate_trade_thesis(
                                        signal, {'close': signal.entry_price}, indicators, regime_name
                                    )
                                    
                                    # Log the trade thesis (intelligent reasoning)
                                    logger.info(thesis.to_readable())
                                    
                                    # Use Trading Captain's calculated position size
                                    position_size = thesis.position_size_lots
                                    
                                    direction = TradeDirection.LONG if signal.direction.name == 'LONG' else TradeDirection.SHORT
                                    
                                    # Place real order via MT5 bridge
                                    ticket = broker.place_order(
                                        symbol=symbol,
                                        direction=direction,
                                        volume=position_size,
                                        sl=signal.stop_loss,
                                        tp=signal.take_profit,
                                        comment=f"AI_{signal.strategy}_{trading_captain.mode.value}"
                                    )
                                    
                                    if ticket:
                                        trade_data = {
                                            'ticket': ticket,
                                            'symbol': symbol,
                                            'direction': signal.direction.name,
                                            'volume': position_size,
                                            'confidence': signal.confidence,
                                            'strategy': signal.strategy,
                                            'regime': regime_name,
                                            'risk_budget': thesis.risk_budget_percent,
                                            'thesis': thesis.entry_reason,
                                            'entry_price': signal.entry_price,
                                            'indicators': indicators,
                                            'timestamp': current_time
                                        }
                                        executed_trades.append(trade_data)
                                        
                                        # Record trade entry for self-reflection
                                        reflection_engine.record_trade_entry(str(ticket), trade_data)
                                        
                                        # Get pre-trade wisdom from past insights
                                        wisdom = reflection_engine.get_pre_trade_wisdom(symbol, signal.strategy, regime_name)
                                        if "No specific" not in wisdom:
                                            logger.info(f"[WISDOM] {wisdom}")
                                        
                                        logger.info(f"[MT5] INTELLIGENT ORDER: {symbol} {signal.direction.name} | "
                                                   f"Mode: {trading_captain.mode.value} | "
                                                   f"Risk: {thesis.risk_budget_percent:.1%} | "
                                                   f"Size: {position_size} lots | "
                                                   f"R:R: 1:{thesis.risk_reward_ratio:.1f} | "
                                                   f"Ticket: {ticket}")
                                    else:
                                        logger.warning(f"[MT5] Order failed for {symbol}")
                            else:
                                logger.debug(f"[Captain] Trade rejected for {symbol}: {reason}")
                        
                        # Pattern learning
                        if PATTERN_MINER_AVAILABLE and pattern_miner and iteration % 100 == 0:
                            try:
                                patterns = pattern_miner.analyze_historical_data(df, symbol)
                                if patterns:
                                    logger.info(f"[MT5] Learned {len(patterns)} patterns for {symbol}")
                            except Exception as e:
                                logger.warning(f"Pattern learning error: {e}")
                    
                    except Exception as e:
                        logger.error(f"Error analyzing {symbol}: {e}")
                
                # Status update every 5 minutes
                if (current_time - last_status_time).seconds >= 300:
                    print("\n" + "=" * 60)
                    print(f"PHOENIX TRADING SYSTEM - {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
                    print("=" * 60)
                    print(f"MT5 Connected: YES")
                    print(f"Account Balance: ${account_balance:.2f}")
                    print(f"Account Equity: ${account_equity:.2f}")
                    print(f"Peak Balance: ${trading_captain.memory.peak_balance:.2f}")
                    print(f"Drawdown: {trading_captain.memory.drawdown_percent:.1%}")
                    print(f"Open Positions: {len(open_positions)}")
                    print(f"Trades Executed: {len(executed_trades)}")
                    print("-" * 60)
                    print(f"TRADING CAPTAIN MODE: {trading_captain.mode.value.upper()}")
                    print(f"Win Streak: {trading_captain.memory.win_streak} | Loss Streak: {trading_captain.memory.loss_streak}")
                    print(f"Daily P&L: ${trading_captain.memory.daily_pnl:.2f}")
                    if trading_captain.memory.insights:
                        print(f"Latest Insight: {trading_captain.memory.insights[-1]}")
                    print("-" * 60)
                    print(f"PatternMiner: {PATTERN_MINER_AVAILABLE}")
                    print(f"BedrockAI: {BEDROCK_AVAILABLE}")
                    
                    # Phoenix components status
                    print("-" * 60)
                    print("PHOENIX INTELLIGENCE:")
                    if KNOWLEDGE_ENGINE_AVAILABLE and knowledge_engine:
                        state = knowledge_engine.get_current_state()
                        if state:
                            print(f"  Market Sentiment: {state.overall_sentiment:+.2f}")
                            print(f"  Insights Collected: {state.insights_count}")
                            if state.strategy_hints:
                                print(f"  Strategy Hint: {state.strategy_hints[0][:60]}...")
                    if PHOENIX_BRAIN_AVAILABLE and phoenix_brain:
                        metrics = phoenix_brain.get_performance_metrics()
                        print(f"  Sharpe Ratio: {metrics.get('sharpe', 0):.2f}")
                        print(f"  Profit Factor: {metrics.get('profit_factor', 1):.2f}")
                        print(f"  Regime Alignment: {metrics.get('regime_alignment_rate', 0):.0%}")
                    if VECTOR_MEMORY_AVAILABLE and vector_memory:
                        stats = vector_memory.get_statistics()
                        print(f"  Experiences Stored: {stats.get('total_experiences', 0)}")
                    print("=" * 60 + "\n")
                    
                    last_status_time = current_time
            
            else:
                # MT5 not connected yet
                if iteration % 10 == 0:
                    print(f"Waiting for MT5 EA connection... (iteration {iteration})")
                    print(f"Make sure to add http://<ec2-ip>:{api_port} to MT5 allowed URLs")
            
            # Daily learning cycle
            if current_time.date() != last_daily_reset:
                logger.info("Running daily learning cycle...")
                try:
                    # Generate daily self-reflection summary
                    daily_summary = reflection_engine.generate_daily_summary(
                        executed_trades, account_balance if mt5_connected else 100.0
                    )
                    logger.info(f"[REFLECTION] Daily summary generated - Mood: {daily_summary.mood}")
                    
                    # Phoenix brain session summary
                    if PHOENIX_BRAIN_AVAILABLE and phoenix_brain:
                        knowledge_state = None
                        if KNOWLEDGE_ENGINE_AVAILABLE and knowledge_engine:
                            ks = knowledge_engine.get_current_state()
                            if ks:
                                knowledge_state = ks.to_dict()
                        
                        session_summary = phoenix_brain.generate_session_summary(knowledge_state)
                        logger.info(f"[PHOENIX] Session Summary:\n{session_summary[:500]}...")
                    
                    # Generate knowledge acquisition report
                    if KNOWLEDGE_ENGINE_AVAILABLE and knowledge_engine:
                        knowledge_report = knowledge_engine.generate_learning_report()
                        logger.info(f"[PHOENIX] Knowledge Report:\n{knowledge_report[:500]}...")
                    
                    # Run agentic learning
                    report = agentic_system.run_daily_learning_cycle(
                        executed_trades, account_balance if mt5_connected else 100.0
                    )
                    logger.info(f"Daily learning complete")
                    
                    # Clear executed trades for new day
                    executed_trades.clear()
                except Exception as e:
                    logger.error(f"Daily learning error: {e}")
                
                last_daily_reset = current_time.date()
            
            time.sleep(30)
            
        except KeyboardInterrupt:
            print("\n" + "=" * 60)
            print("MT5 Bridge mode stopped by user")
            if mt5_connected:
                print(f"Final Balance: ${broker.get_balance():.2f}")
            print(f"Total Trades Executed: {len(executed_trades)}")
            print("=" * 60)
            break
        except Exception as e:
            logger.error(f"MT5 Bridge loop error: {e}")
            time.sleep(60)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Aladdin Forex Trading System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Start live trading
  python main.py --demo             # Run demo mode (simulated)
  python main.py --mt5-bridge       # Run with MT5 Bridge (real trading)
  python main.py --backtest         # Run backtest
  python main.py --backtest --walk-forward  # Run with walk-forward optimization
        """
    )
    
    # Mode selection
    parser.add_argument('--demo', action='store_true', help='Run in demo mode (simulated)')
    parser.add_argument('--mt5-bridge', action='store_true', help='Run with MT5 Bridge for real trading')
    parser.add_argument('--backtest', action='store_true', help='Run backtest mode')
    
    # Backtest options
    parser.add_argument('--symbol', type=str, default='EURUSD', help='Trading symbol')
    parser.add_argument('--timeframe', type=str, default='H1', help='Timeframe')
    parser.add_argument('--bars', type=int, default=5000, help='Number of bars')
    parser.add_argument('--balance', type=float, default=100.0, help='Initial balance')
    parser.add_argument('--monte-carlo', action='store_true', help='Run Monte Carlo simulation')
    parser.add_argument('--walk-forward', action='store_true', help='Run walk-forward optimization')
    
    # General options
    parser.add_argument('--log-level', type=str, default='INFO', 
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = getattr(logging, args.log_level)
    LoggingSetup.setup(log_dir="./logs", log_level=log_level)
    
    # Run appropriate mode
    if args.demo:
        run_demo(args)
    elif args.mt5_bridge:
        run_mt5_bridge(args)
    elif args.backtest:
        run_backtest(args)
    else:
        # Live trading mode
        system = TradingSystem()
        try:
            system.start()
        except KeyboardInterrupt:
            system.shutdown()
        except Exception as e:
            logger.error(f"Fatal error: {e}")
            system.shutdown()
            sys.exit(1)


if __name__ == "__main__":
    main()
