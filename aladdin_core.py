"""
Aladdin Core - Unified Platform Coordinator
Institutional-grade investment management system inspired by BlackRock's Aladdin
Coordinates all AI agents, risk analytics, portfolio optimization, and execution
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import threading
import time
import json
import os
from collections import defaultdict
from queue import Queue, PriorityQueue

from config import config, DISCLAIMER
from data import data_manager, DatabaseManager
from indicators import TechnicalIndicators, FeatureEngineer
from models import ensemble_predictor, LSTMPricePredictor, DQNAgent, TradingEnvironment
from risk_management import risk_manager, RiskMetrics, MonteCarloSimulator, PortfolioOptimizer
from regime_detection import regime_manager, MarketRegime
from decisions import decision_engine, TradingSignal, TradeDirection
from execution import execution_engine, ExecutionAlgo
from agentic import agentic_system, DailyReport
from trading import trading_engine, Trade

logger = logging.getLogger(__name__)


class AgentType(Enum):
    RISK = "risk"
    STRATEGY = "strategy"
    EXECUTION = "execution"
    REGIME = "regime"
    LEARNING = "learning"


class AgentStatus(Enum):
    IDLE = "idle"
    RUNNING = "running"
    ERROR = "error"
    PAUSED = "paused"


@dataclass
class AgentMessage:
    """Message for inter-agent communication"""
    sender: AgentType
    receiver: AgentType
    message_type: str
    payload: Dict
    priority: int = 5
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class PortfolioState:
    """Current portfolio state"""
    timestamp: datetime
    balance: float
    equity: float
    margin_used: float
    free_margin: float
    positions: List[Dict]
    exposure_by_symbol: Dict[str, float]
    exposure_by_currency: Dict[str, float]
    total_risk: float
    var_95: float
    var_99: float
    regime: str
    strategy_allocation: Dict[str, float]


@dataclass
class SystemHealth:
    """System health metrics"""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    api_latency_ms: float
    broker_connected: bool
    data_feed_active: bool
    agents_status: Dict[str, AgentStatus]
    errors_last_hour: int
    warnings_last_hour: int


class BaseAgent:
    """Base class for all AI agents"""
    
    def __init__(self, agent_type: AgentType, name: str):
        self.agent_type = agent_type
        self.name = name
        self.status = AgentStatus.IDLE
        self.message_queue = Queue()
        self.last_run = None
        self.run_count = 0
        self.error_count = 0
        self._running = False
        self._thread = None
    
    def start(self):
        """Start the agent"""
        self._running = True
        self.status = AgentStatus.RUNNING
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info(f"Agent {self.name} started")
    
    def stop(self):
        """Stop the agent"""
        self._running = False
        self.status = AgentStatus.IDLE
        if self._thread:
            self._thread.join(timeout=5)
        logger.info(f"Agent {self.name} stopped")
    
    def _run_loop(self):
        """Main agent loop"""
        while self._running:
            try:
                # Process messages
                while not self.message_queue.empty():
                    msg = self.message_queue.get()
                    self._handle_message(msg)
                
                # Run agent logic
                self._execute()
                self.last_run = datetime.now()
                self.run_count += 1
                
                time.sleep(self._get_interval())
                
            except Exception as e:
                self.error_count += 1
                self.status = AgentStatus.ERROR
                logger.error(f"Agent {self.name} error: {e}")
                time.sleep(10)
                self.status = AgentStatus.RUNNING
    
    def _execute(self):
        """Execute agent logic - override in subclass"""
        pass
    
    def _handle_message(self, msg: AgentMessage):
        """Handle incoming message - override in subclass"""
        pass
    
    def _get_interval(self) -> float:
        """Get run interval in seconds - override in subclass"""
        return 60
    
    def send_message(self, receiver: AgentType, message_type: str, payload: Dict):
        """Send message to another agent"""
        msg = AgentMessage(
            sender=self.agent_type,
            receiver=receiver,
            message_type=message_type,
            payload=payload
        )
        return msg


class RiskAgent(BaseAgent):
    """Risk assessment and monitoring agent"""
    
    def __init__(self):
        super().__init__(AgentType.RISK, "RiskAgent")
        self.mc_simulator = MonteCarloSimulator()
        self.portfolio_optimizer = PortfolioOptimizer()
        self.risk_limits = {
            'max_var_95': 0.05,
            'max_drawdown': 0.20,
            'max_leverage': 500,
            'max_position_size': 0.10,
            'max_correlation': 0.8
        }
        self.current_risk_assessment = {}
        self.alerts = []
    
    def _execute(self):
        """Execute risk assessment"""
        # Get current positions
        positions = trading_engine.active_trades
        account_info = trading_engine.broker._get_account_info()
        balance = account_info.get('balance', config.trading.initial_balance)
        
        # Calculate portfolio risk
        self._assess_portfolio_risk(positions, balance)
        
        # Run stress tests
        self._run_stress_tests(balance)
        
        # Check risk limits
        self._check_risk_limits()
        
        # Update risk manager
        risk_manager.update_metrics(
            np.array([t.profit for t in trading_engine.trade_history[-100:]]) if trading_engine.trade_history else np.array([0]),
            balance
        )
    
    def _assess_portfolio_risk(self, positions: Dict, balance: float):
        """Assess current portfolio risk"""
        if not positions:
            self.current_risk_assessment = {
                'var_95': 0,
                'var_99': 0,
                'total_exposure': 0,
                'risk_level': 'low'
            }
            return
        
        # Calculate total exposure
        total_exposure = sum(t.volume * t.entry_price for t in positions.values())
        exposure_pct = total_exposure / balance if balance > 0 else 0
        
        # Estimate VaR from recent returns
        returns = np.array([t.profit / balance for t in trading_engine.trade_history[-50:]]) if trading_engine.trade_history else np.array([0])
        
        var_95 = np.percentile(returns, 5) if len(returns) > 10 else -0.02
        var_99 = np.percentile(returns, 1) if len(returns) > 10 else -0.05
        
        # Determine risk level
        if exposure_pct > 0.5 or abs(var_95) > 0.03:
            risk_level = 'high'
        elif exposure_pct > 0.3 or abs(var_95) > 0.02:
            risk_level = 'medium'
        else:
            risk_level = 'low'
        
        self.current_risk_assessment = {
            'var_95': var_95,
            'var_99': var_99,
            'total_exposure': total_exposure,
            'exposure_pct': exposure_pct,
            'risk_level': risk_level,
            'timestamp': datetime.now().isoformat()
        }
    
    def _run_stress_tests(self, balance: float):
        """Run stress test scenarios"""
        stress_results = risk_manager.run_stress_tests(balance)
        self.current_risk_assessment['stress_tests'] = stress_results
    
    def _check_risk_limits(self):
        """Check if any risk limits are breached"""
        assessment = self.current_risk_assessment
        
        # Check VaR limit
        if abs(assessment.get('var_95', 0)) > self.risk_limits['max_var_95']:
            self._create_alert('VaR limit exceeded', 'high')
        
        # Check exposure
        if assessment.get('exposure_pct', 0) > self.risk_limits['max_position_size'] * 5:
            self._create_alert('Total exposure too high', 'high')
    
    def _create_alert(self, message: str, severity: str):
        """Create risk alert"""
        alert = {
            'timestamp': datetime.now().isoformat(),
            'message': message,
            'severity': severity,
            'assessment': self.current_risk_assessment
        }
        self.alerts.append(alert)
        logger.warning(f"Risk Alert [{severity}]: {message}")
    
    def get_risk_budget(self) -> Dict:
        """Get current risk budget for trading"""
        risk_level = self.current_risk_assessment.get('risk_level', 'medium')
        
        if risk_level == 'high':
            return {'max_risk_per_trade': 0.005, 'max_new_positions': 0}
        elif risk_level == 'medium':
            return {'max_risk_per_trade': 0.01, 'max_new_positions': 2}
        else:
            return {'max_risk_per_trade': 0.02, 'max_new_positions': 5}
    
    def _get_interval(self) -> float:
        return 30  # Run every 30 seconds


class StrategyAgent(BaseAgent):
    """Strategy selection and allocation agent"""
    
    def __init__(self):
        super().__init__(AgentType.STRATEGY, "StrategyAgent")
        self.strategies = {
            'trend_following': {'weight': 0.3, 'active': True},
            'mean_reversion': {'weight': 0.25, 'active': True},
            'fvg_entry': {'weight': 0.25, 'active': True},
            'liquidity_sweep': {'weight': 0.2, 'active': True}
        }
        self.strategy_performance = defaultdict(lambda: {'trades': 0, 'wins': 0, 'profit': 0})
        self.current_allocation = {}
    
    def _execute(self):
        """Execute strategy selection logic"""
        # Get current regime
        regime = regime_manager.current_regime
        
        # Update strategy weights based on regime
        self._update_weights_for_regime(regime)
        
        # Update based on recent performance
        self._update_weights_from_performance()
        
        # Calculate final allocation
        self._calculate_allocation()
    
    def _update_weights_for_regime(self, regime: MarketRegime):
        """Update strategy weights based on market regime"""
        if regime is None:
            return
        
        regime_name = regime.name.lower()
        
        if 'trend' in regime_name:
            self.strategies['trend_following']['weight'] = 0.5
            self.strategies['mean_reversion']['weight'] = 0.15
            self.strategies['fvg_entry']['weight'] = 0.2
            self.strategies['liquidity_sweep']['weight'] = 0.15
        elif 'low_vol' in regime_name or 'rang' in regime_name:
            self.strategies['trend_following']['weight'] = 0.15
            self.strategies['mean_reversion']['weight'] = 0.45
            self.strategies['fvg_entry']['weight'] = 0.25
            self.strategies['liquidity_sweep']['weight'] = 0.15
        elif 'high_vol' in regime_name:
            self.strategies['trend_following']['weight'] = 0.2
            self.strategies['mean_reversion']['weight'] = 0.2
            self.strategies['fvg_entry']['weight'] = 0.3
            self.strategies['liquidity_sweep']['weight'] = 0.3
    
    def _update_weights_from_performance(self):
        """Update weights based on recent performance"""
        total_trades = sum(s['trades'] for s in self.strategy_performance.values())
        
        if total_trades < 20:
            return
        
        for strategy, perf in self.strategy_performance.items():
            if perf['trades'] >= 5:
                win_rate = perf['wins'] / perf['trades']
                
                # Adjust weight based on win rate
                current_weight = self.strategies.get(strategy, {}).get('weight', 0.25)
                
                if win_rate > 0.6:
                    new_weight = min(0.5, current_weight * 1.1)
                elif win_rate < 0.4:
                    new_weight = max(0.1, current_weight * 0.9)
                else:
                    new_weight = current_weight
                
                if strategy in self.strategies:
                    self.strategies[strategy]['weight'] = new_weight
        
        # Normalize weights
        total_weight = sum(s['weight'] for s in self.strategies.values())
        if total_weight > 0:
            for strategy in self.strategies:
                self.strategies[strategy]['weight'] /= total_weight
    
    def _calculate_allocation(self):
        """Calculate final strategy allocation"""
        self.current_allocation = {
            strategy: info['weight'] 
            for strategy, info in self.strategies.items() 
            if info['active']
        }
    
    def record_trade_result(self, strategy: str, profit: float):
        """Record trade result for strategy"""
        self.strategy_performance[strategy]['trades'] += 1
        if profit > 0:
            self.strategy_performance[strategy]['wins'] += 1
        self.strategy_performance[strategy]['profit'] += profit
    
    def get_strategy_weights(self) -> Dict[str, float]:
        """Get current strategy weights"""
        return self.current_allocation
    
    def _get_interval(self) -> float:
        return 60  # Run every minute


class RegimeAgent(BaseAgent):
    """Market regime detection and monitoring agent"""
    
    def __init__(self):
        super().__init__(AgentType.REGIME, "RegimeAgent")
        self.current_regime = None
        self.regime_history = []
        self.regime_change_callbacks = []
    
    def _execute(self):
        """Execute regime detection"""
        # Get latest data
        df = data_manager.get_ohlcv('EURUSD', 'H1', count=500)
        
        if df.empty:
            return
        
        # Detect regime
        new_regime = regime_manager.detect_regime(df)
        
        # Check for regime change
        if self.current_regime and new_regime:
            if self.current_regime.name != new_regime.name:
                self._handle_regime_change(self.current_regime, new_regime)
        
        self.current_regime = new_regime
        
        # Record history
        self.regime_history.append({
            'timestamp': datetime.now().isoformat(),
            'regime': new_regime.name if new_regime else 'unknown',
            'probability': new_regime.probability if new_regime else 0
        })
        
        # Keep only last 1000 entries
        self.regime_history = self.regime_history[-1000:]
    
    def _handle_regime_change(self, old_regime: MarketRegime, new_regime: MarketRegime):
        """Handle regime change event"""
        logger.info(f"Regime change: {old_regime.name} -> {new_regime.name}")
        
        # Notify callbacks
        for callback in self.regime_change_callbacks:
            try:
                callback(old_regime, new_regime)
            except Exception as e:
                logger.error(f"Regime change callback error: {e}")
    
    def register_regime_change_callback(self, callback: Callable):
        """Register callback for regime changes"""
        self.regime_change_callbacks.append(callback)
    
    def get_regime_summary(self) -> Dict:
        """Get regime summary"""
        if not self.current_regime:
            return {'regime': 'unknown', 'probability': 0}
        
        return {
            'regime': self.current_regime.name,
            'probability': self.current_regime.probability,
            'volatility': self.current_regime.volatility,
            'trend_strength': self.current_regime.trend_strength,
            'recommended_strategies': self.current_regime.recommended_strategies,
            'risk_adjustment': self.current_regime.risk_adjustment
        }
    
    def _get_interval(self) -> float:
        return 300  # Run every 5 minutes


class LearningAgent(BaseAgent):
    """Daily learning and model retraining agent"""
    
    def __init__(self):
        super().__init__(AgentType.LEARNING, "LearningAgent")
        self.last_training_date = None
        self.training_scheduled = False
        self.training_hour = config.ml.retrain_hour_utc
    
    def _execute(self):
        """Execute learning logic"""
        current_hour = datetime.utcnow().hour
        current_date = datetime.utcnow().date()
        
        # Check if it's time for daily learning
        if current_hour == self.training_hour and current_date != self.last_training_date:
            self._run_daily_learning()
            self.last_training_date = current_date
    
    def _run_daily_learning(self):
        """Run daily learning cycle"""
        logger.info("Starting daily learning cycle...")
        
        try:
            # Get today's trades
            trades = trading_engine.get_trades_for_learning()
            
            # Get account balance
            account_info = trading_engine.broker._get_account_info()
            balance = account_info.get('balance', config.trading.initial_balance)
            
            # Run agentic learning
            report = agentic_system.run_daily_learning_cycle(trades, balance)
            
            # Log report
            logger.info(f"Daily learning complete: {report.total_trades} trades, "
                       f"Win rate: {report.win_rate:.1%}, Profit: ${report.total_profit:.2f}")
            
            # Retrain models if enough data
            if len(trades) >= config.ml.min_samples_for_retrain:
                self._retrain_models()
            
        except Exception as e:
            logger.error(f"Daily learning error: {e}")
    
    def _retrain_models(self):
        """Retrain ML models"""
        logger.info("Retraining ML models...")
        
        try:
            # Get historical data
            df = data_manager.get_ohlcv('EURUSD', 'H1', count=5000)
            
            if df.empty or len(df) < 1000:
                logger.warning("Insufficient data for retraining")
                return
            
            # Create features
            feature_engineer = FeatureEngineer()
            features = feature_engineer.create_features(df)
            target = feature_engineer.create_target(df, lookahead=1, threshold=0.0001)
            
            # Prepare sequences
            X, y = feature_engineer.prepare_sequences(
                features.dropna(), 
                target[features.dropna().index],
                sequence_length=config.ml.lstm_sequence_length
            )
            
            if len(X) < 500:
                logger.warning("Insufficient sequences for training")
                return
            
            # Initialize and train ensemble
            ensemble_predictor.initialize_models(input_dim=X.shape[2])
            results = ensemble_predictor.train_all(X, y, df)
            
            # Save models
            ensemble_predictor.save_all(config.ml.model_dir)
            
            logger.info(f"Model retraining complete: {results}")
            
        except Exception as e:
            logger.error(f"Model retraining error: {e}")
    
    def _get_interval(self) -> float:
        return 3600  # Check every hour


class AladdinCore:
    """Main Aladdin platform coordinator"""
    
    def __init__(self):
        # Initialize agents
        self.risk_agent = RiskAgent()
        self.strategy_agent = StrategyAgent()
        self.regime_agent = RegimeAgent()
        self.learning_agent = LearningAgent()
        
        self.agents = {
            AgentType.RISK: self.risk_agent,
            AgentType.STRATEGY: self.strategy_agent,
            AgentType.REGIME: self.regime_agent,
            AgentType.LEARNING: self.learning_agent
        }
        
        # Message bus
        self.message_bus = Queue()
        
        # State
        self.is_running = False
        self.start_time = None
        self.portfolio_state = None
        
        # Event handlers
        self._setup_event_handlers()
        
        # Statistics
        self.stats = {
            'messages_processed': 0,
            'signals_generated': 0,
            'trades_executed': 0,
            'errors': 0
        }
    
    def _setup_event_handlers(self):
        """Setup event handlers between agents"""
        # Regime change affects strategy
        def on_regime_change(old_regime, new_regime):
            self.strategy_agent._update_weights_for_regime(new_regime)
            
            # If high volatility, reduce risk
            if 'high_vol' in new_regime.name.lower():
                risk_manager.adaptive_params = {
                    **risk_manager.position_sizer.__dict__,
                    'base_risk': 0.005
                }
        
        self.regime_agent.register_regime_change_callback(on_regime_change)
    
    def start(self):
        """Start the Aladdin platform"""
        logger.info("Starting Aladdin Core platform...")
        logger.info(DISCLAIMER)
        
        self.is_running = True
        self.start_time = datetime.now()
        
        # Start all agents
        for agent in self.agents.values():
            agent.start()
        
        # Start trading engine
        trading_engine.start()
        
        # Start execution engine
        execution_engine.start()
        
        # Start message processor
        self._message_thread = threading.Thread(target=self._process_messages, daemon=True)
        self._message_thread.start()
        
        # Start portfolio monitor
        self._portfolio_thread = threading.Thread(target=self._monitor_portfolio, daemon=True)
        self._portfolio_thread.start()
        
        logger.info("Aladdin Core platform started")
    
    def stop(self):
        """Stop the Aladdin platform"""
        logger.info("Stopping Aladdin Core platform...")
        
        self.is_running = False
        
        # Stop all agents
        for agent in self.agents.values():
            agent.stop()
        
        # Stop trading engine
        trading_engine.stop()
        
        # Stop execution engine
        execution_engine.stop()
        
        logger.info("Aladdin Core platform stopped")
    
    def _process_messages(self):
        """Process inter-agent messages"""
        while self.is_running:
            try:
                if not self.message_bus.empty():
                    msg = self.message_bus.get()
                    self._route_message(msg)
                    self.stats['messages_processed'] += 1
                else:
                    time.sleep(0.1)
            except Exception as e:
                logger.error(f"Message processing error: {e}")
                self.stats['errors'] += 1
    
    def _route_message(self, msg: AgentMessage):
        """Route message to appropriate agent"""
        if msg.receiver in self.agents:
            self.agents[msg.receiver].message_queue.put(msg)
    
    def _monitor_portfolio(self):
        """Monitor portfolio state"""
        while self.is_running:
            try:
                self._update_portfolio_state()
                time.sleep(10)
            except Exception as e:
                logger.error(f"Portfolio monitor error: {e}")
    
    def _update_portfolio_state(self):
        """Update current portfolio state"""
        account_info = trading_engine.broker._get_account_info()
        positions = list(trading_engine.active_trades.values())
        
        # Calculate exposures
        exposure_by_symbol = defaultdict(float)
        exposure_by_currency = defaultdict(float)
        
        for trade in positions:
            exposure = trade.volume * trade.entry_price
            exposure_by_symbol[trade.symbol] += exposure
            
            # Extract currencies
            base = trade.symbol[:3]
            quote = trade.symbol[3:]
            exposure_by_currency[base] += exposure
            exposure_by_currency[quote] -= exposure
        
        # Get risk metrics
        risk_assessment = self.risk_agent.current_risk_assessment
        
        self.portfolio_state = PortfolioState(
            timestamp=datetime.now(),
            balance=account_info.get('balance', 0),
            equity=account_info.get('equity', 0),
            margin_used=account_info.get('margin', 0),
            free_margin=account_info.get('free_margin', 0),
            positions=[{
                'ticket': t.ticket,
                'symbol': t.symbol,
                'direction': t.direction.name,
                'volume': t.volume,
                'profit': t.profit
            } for t in positions],
            exposure_by_symbol=dict(exposure_by_symbol),
            exposure_by_currency=dict(exposure_by_currency),
            total_risk=risk_assessment.get('exposure_pct', 0),
            var_95=risk_assessment.get('var_95', 0),
            var_99=risk_assessment.get('var_99', 0),
            regime=self.regime_agent.current_regime.name if self.regime_agent.current_regime else 'unknown',
            strategy_allocation=self.strategy_agent.get_strategy_weights()
        )
    
    def get_dashboard_data(self) -> Dict:
        """Get data for dashboard display"""
        account_info = trading_engine.broker._get_account_info()
        session_summary = trading_engine.get_session_summary()
        risk_assessment = self.risk_agent.current_risk_assessment
        regime_summary = self.regime_agent.get_regime_summary()
        strategy_weights = self.strategy_agent.get_strategy_weights()
        trading_params = agentic_system.get_trading_parameters()
        
        return {
            'account': {
                'balance': account_info.get('balance', 0),
                'equity': account_info.get('equity', 0),
                'margin': account_info.get('margin', 0),
                'free_margin': account_info.get('free_margin', 0),
                'leverage': account_info.get('leverage', 0)
            },
            'session': session_summary,
            'risk': risk_assessment,
            'regime': regime_summary,
            'strategies': strategy_weights,
            'trading_params': trading_params,
            'agents': {
                name.value: {
                    'status': agent.status.value,
                    'last_run': agent.last_run.isoformat() if agent.last_run else None,
                    'run_count': agent.run_count,
                    'error_count': agent.error_count
                }
                for name, agent in self.agents.items()
            },
            'system': {
                'uptime_seconds': (datetime.now() - self.start_time).seconds if self.start_time else 0,
                'messages_processed': self.stats['messages_processed'],
                'signals_generated': self.stats['signals_generated'],
                'trades_executed': self.stats['trades_executed'],
                'errors': self.stats['errors']
            },
            'timestamp': datetime.now().isoformat()
        }
    
    def get_system_health(self) -> SystemHealth:
        """Get system health metrics"""
        import psutil
        
        return SystemHealth(
            timestamp=datetime.now(),
            cpu_usage=psutil.cpu_percent() if 'psutil' in dir() else 0,
            memory_usage=psutil.virtual_memory().percent if 'psutil' in dir() else 0,
            api_latency_ms=0,  # Would measure actual API latency
            broker_connected=trading_engine.broker.connected,
            data_feed_active=True,  # Would check actual data feed
            agents_status={name.value: agent.status for name, agent in self.agents.items()},
            errors_last_hour=self.stats['errors'],
            warnings_last_hour=len(self.risk_agent.alerts)
        )
    
    def execute_command(self, command: str, params: Dict = None) -> Dict:
        """Execute a command on the platform"""
        params = params or {}
        
        commands = {
            'pause': lambda: trading_engine.pause(),
            'resume': lambda: trading_engine.resume(),
            'close_all': lambda: trading_engine.close_all_positions(params.get('reason', 'Manual')),
            'set_mode': lambda: setattr(agentic_system, 'trading_mode', params.get('mode', 'normal')),
            'retrain': lambda: self.learning_agent._retrain_models(),
            'status': lambda: self.get_dashboard_data()
        }
        
        if command in commands:
            try:
                result = commands[command]()
                return {'success': True, 'result': result}
            except Exception as e:
                return {'success': False, 'error': str(e)}
        
        return {'success': False, 'error': f'Unknown command: {command}'}


# Singleton instance
aladdin = AladdinCore()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Aladdin Core Platform...")
    print(DISCLAIMER)
    
    # Test individual agents
    print("\nTesting Risk Agent...")
    risk_agent = RiskAgent()
    risk_budget = risk_agent.get_risk_budget()
    print(f"Risk budget: {risk_budget}")
    
    print("\nTesting Strategy Agent...")
    strategy_agent = StrategyAgent()
    strategy_agent._execute()
    weights = strategy_agent.get_strategy_weights()
    print(f"Strategy weights: {weights}")
    
    print("\nTesting Regime Agent...")
    regime_agent = RegimeAgent()
    summary = regime_agent.get_regime_summary()
    print(f"Regime summary: {summary}")
    
    # Test dashboard data
    print("\nTesting Dashboard Data...")
    core = AladdinCore()
    # Don't start full system for test
    dashboard = core.get_dashboard_data()
    print(f"Dashboard keys: {list(dashboard.keys())}")
    
    print("\nAladdin Core Platform test complete!")
