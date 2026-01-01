"""
Adaptive Learning System for Forex Trading
Priority 2: Online/Offline Learning

Online Learning (Market Hours):
- Real-time calibration of confidence thresholds
- Adaptive position sizing based on recent performance
- Strategy weight adjustment
- Quick feedback loops (every few minutes)

Offline Learning (Off-Hours):
- Model retraining with accumulated data
- Walk-forward backtesting
- Regime analysis and pattern discovery
- Strategy optimization
"""

import logging
import json
import os
import threading
import time
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import statistics
import math

logger = logging.getLogger(__name__)


class MarketSession(Enum):
    """Forex market sessions"""
    SYDNEY = "sydney"
    TOKYO = "tokyo"
    LONDON = "london"
    NEW_YORK = "new_york"
    CLOSED = "closed"


class LearningMode(Enum):
    """Learning mode based on market status"""
    ONLINE = "online"      # Market open - lightweight calibration
    OFFLINE = "offline"    # Market closed - heavy computation
    TRANSITION = "transition"  # Switching between modes


@dataclass
class TradeOutcome:
    """Record of a completed trade for learning"""
    symbol: str
    direction: str  # BUY or SELL
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    pnl: float
    pnl_pips: float
    strategy: str
    confidence_at_entry: float
    regime_at_entry: str
    indicators_at_entry: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        d = asdict(self)
        d['entry_time'] = self.entry_time.isoformat()
        d['exit_time'] = self.exit_time.isoformat()
        return d
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'TradeOutcome':
        d['entry_time'] = datetime.fromisoformat(d['entry_time'])
        d['exit_time'] = datetime.fromisoformat(d['exit_time'])
        return cls(**d)


@dataclass
class PerformanceMetrics:
    """Performance metrics for a time period"""
    period_start: datetime
    period_end: datetime
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    total_pips: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    
    def calculate(self, trades: List[TradeOutcome]):
        """Calculate metrics from trade outcomes"""
        if not trades:
            return
        
        self.total_trades = len(trades)
        self.winning_trades = sum(1 for t in trades if t.pnl > 0)
        self.losing_trades = sum(1 for t in trades if t.pnl < 0)
        self.total_pnl = sum(t.pnl for t in trades)
        self.total_pips = sum(t.pnl_pips for t in trades)
        
        if self.total_trades > 0:
            self.win_rate = self.winning_trades / self.total_trades
        
        wins = [t.pnl for t in trades if t.pnl > 0]
        losses = [abs(t.pnl) for t in trades if t.pnl < 0]
        
        if wins:
            self.avg_win = statistics.mean(wins)
        if losses:
            self.avg_loss = statistics.mean(losses)
        
        total_wins = sum(wins) if wins else 0
        total_losses = sum(losses) if losses else 0
        
        if total_losses > 0:
            self.profit_factor = total_wins / total_losses
        
        if self.total_trades > 0:
            self.expectancy = (self.win_rate * self.avg_win) - ((1 - self.win_rate) * self.avg_loss)
        
        # Calculate max drawdown
        equity_curve = []
        running_equity = 0
        for t in sorted(trades, key=lambda x: x.exit_time):
            running_equity += t.pnl
            equity_curve.append(running_equity)
        
        if equity_curve:
            peak = equity_curve[0]
            max_dd = 0
            for equity in equity_curve:
                if equity > peak:
                    peak = equity
                dd = peak - equity
                if dd > max_dd:
                    max_dd = dd
            self.max_drawdown = max_dd
        
        # Simple Sharpe approximation
        if len(trades) > 1:
            returns = [t.pnl for t in trades]
            if statistics.stdev(returns) > 0:
                self.sharpe_ratio = (statistics.mean(returns) / statistics.stdev(returns)) * math.sqrt(252)


@dataclass
class AdaptiveParameters:
    """Parameters that can be adjusted by learning"""
    # Confidence thresholds
    min_confidence_threshold: float = 0.65
    high_confidence_threshold: float = 0.80
    
    # Position sizing
    base_risk_percent: float = 1.0
    max_risk_percent: float = 2.0
    risk_multiplier: float = 1.0
    
    # Strategy weights (sum to 1.0)
    trend_following_weight: float = 0.4
    mean_reversion_weight: float = 0.3
    breakout_weight: float = 0.3
    
    # Trade frequency
    max_trades_per_hour: int = 2
    min_time_between_trades: int = 300  # seconds
    
    # Stop loss / Take profit adjustments
    sl_atr_multiplier: float = 2.0
    tp_atr_multiplier: float = 4.0
    
    # Session preferences (0-1 weight for each session)
    session_weights: Dict[str, float] = field(default_factory=lambda: {
        'sydney': 0.6,
        'tokyo': 0.8,
        'london': 1.0,
        'new_york': 1.0
    })
    
    # Symbol preferences (0-1 weight)
    symbol_weights: Dict[str, float] = field(default_factory=lambda: {
        'EURUSD': 1.0,
        'GBPUSD': 1.0,
        'USDJPY': 1.0,
        'USDCHF': 0.8,
        'AUDUSD': 0.8,
        'USDCAD': 0.8,
        'NZDUSD': 0.7,
        'EURGBP': 0.7
    })
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'AdaptiveParameters':
        return cls(**d)


class MarketHoursDetector:
    """Detects forex market hours and sessions"""
    
    # Forex market hours (UTC)
    SESSIONS = {
        MarketSession.SYDNEY: (21, 6),    # 21:00 - 06:00 UTC
        MarketSession.TOKYO: (0, 9),      # 00:00 - 09:00 UTC
        MarketSession.LONDON: (7, 16),    # 07:00 - 16:00 UTC
        MarketSession.NEW_YORK: (12, 21), # 12:00 - 21:00 UTC
    }
    
    def __init__(self):
        self._last_check = None
        self._cached_status = None
    
    def get_current_session(self) -> MarketSession:
        """Get the current active market session"""
        now = datetime.utcnow()
        hour = now.hour
        weekday = now.weekday()
        
        # Forex market closed on weekends (Friday 21:00 UTC to Sunday 21:00 UTC)
        if weekday == 5:  # Saturday
            return MarketSession.CLOSED
        if weekday == 6 and hour < 21:  # Sunday before 21:00
            return MarketSession.CLOSED
        if weekday == 4 and hour >= 21:  # Friday after 21:00
            return MarketSession.CLOSED
        
        # Check which session is active
        active_sessions = []
        for session, (start, end) in self.SESSIONS.items():
            if start < end:
                if start <= hour < end:
                    active_sessions.append(session)
            else:  # Crosses midnight
                if hour >= start or hour < end:
                    active_sessions.append(session)
        
        if not active_sessions:
            return MarketSession.CLOSED
        
        # Return the most liquid session if multiple are active
        priority = [MarketSession.LONDON, MarketSession.NEW_YORK, MarketSession.TOKYO, MarketSession.SYDNEY]
        for session in priority:
            if session in active_sessions:
                return session
        
        return active_sessions[0]
    
    def is_market_open(self) -> bool:
        """Check if forex market is currently open"""
        return self.get_current_session() != MarketSession.CLOSED
    
    def get_learning_mode(self) -> LearningMode:
        """Determine the appropriate learning mode"""
        if self.is_market_open():
            return LearningMode.ONLINE
        return LearningMode.OFFLINE
    
    def time_until_market_open(self) -> timedelta:
        """Get time until market opens (if closed)"""
        if self.is_market_open():
            return timedelta(0)
        
        now = datetime.utcnow()
        weekday = now.weekday()
        
        # Calculate time until Sunday 21:00 UTC
        if weekday == 5:  # Saturday
            days_until_sunday = 1
        elif weekday == 6:  # Sunday
            if now.hour < 21:
                days_until_sunday = 0
            else:
                days_until_sunday = 7  # Next Sunday
        else:
            days_until_sunday = 6 - weekday
        
        next_open = now.replace(hour=21, minute=0, second=0, microsecond=0)
        next_open += timedelta(days=days_until_sunday)
        
        return next_open - now
    
    def time_until_market_close(self) -> timedelta:
        """Get time until market closes (if open)"""
        if not self.is_market_open():
            return timedelta(0)
        
        now = datetime.utcnow()
        weekday = now.weekday()
        
        # Market closes Friday 21:00 UTC
        days_until_friday = (4 - weekday) % 7
        if days_until_friday == 0 and now.hour >= 21:
            days_until_friday = 7
        
        next_close = now.replace(hour=21, minute=0, second=0, microsecond=0)
        next_close += timedelta(days=days_until_friday)
        
        return next_close - now
    
    def get_session_quality(self) -> float:
        """Get quality score for current session (0-1)"""
        session = self.get_current_session()
        
        quality_scores = {
            MarketSession.LONDON: 1.0,      # Most liquid
            MarketSession.NEW_YORK: 0.95,   # Very liquid
            MarketSession.TOKYO: 0.7,       # Good for JPY pairs
            MarketSession.SYDNEY: 0.5,      # Lower liquidity
            MarketSession.CLOSED: 0.0
        }
        
        return quality_scores.get(session, 0.5)


class OnlineLearner:
    """
    Real-time learning during market hours.
    Performs lightweight calibration without heavy computation.
    """
    
    def __init__(self, params: AdaptiveParameters):
        self.params = params
        self.recent_trades: List[TradeOutcome] = []
        self.performance_window = 20  # Number of recent trades to consider
        self.update_interval = 300  # 5 minutes
        self._last_update = None
        self._learning_rate = 0.1  # How fast to adapt
        
        # Track performance by strategy
        self.strategy_performance: Dict[str, List[float]] = {
            'trend_following': [],
            'mean_reversion': [],
            'breakout': []
        }
        
        # Track performance by symbol
        self.symbol_performance: Dict[str, List[float]] = {}
        
        # Track confidence calibration
        self.confidence_outcomes: List[Tuple[float, bool]] = []  # (confidence, was_winner)
        
        logger.info("OnlineLearner initialized")
    
    def record_trade(self, trade: TradeOutcome):
        """Record a completed trade for learning"""
        self.recent_trades.append(trade)
        
        # Keep only recent trades
        if len(self.recent_trades) > self.performance_window * 2:
            self.recent_trades = self.recent_trades[-self.performance_window:]
        
        # Update strategy performance
        strategy = trade.strategy.lower().replace(' ', '_')
        if strategy not in self.strategy_performance:
            self.strategy_performance[strategy] = []
        self.strategy_performance[strategy].append(trade.pnl)
        
        # Update symbol performance
        if trade.symbol not in self.symbol_performance:
            self.symbol_performance[trade.symbol] = []
        self.symbol_performance[trade.symbol].append(trade.pnl)
        
        # Update confidence calibration
        was_winner = trade.pnl > 0
        self.confidence_outcomes.append((trade.confidence_at_entry, was_winner))
        if len(self.confidence_outcomes) > 100:
            self.confidence_outcomes = self.confidence_outcomes[-100:]
        
        logger.info(f"OnlineLearner recorded trade: {trade.symbol} {trade.direction} PnL={trade.pnl:.2f}")
    
    def should_update(self) -> bool:
        """Check if it's time to update parameters"""
        if self._last_update is None:
            return True
        return (datetime.now() - self._last_update).total_seconds() >= self.update_interval
    
    def update_parameters(self) -> Dict[str, Any]:
        """
        Update adaptive parameters based on recent performance.
        Returns dict of changes made.
        """
        if not self.recent_trades:
            return {}
        
        changes = {}
        self._last_update = datetime.now()
        
        # 1. Calibrate confidence threshold
        confidence_change = self._calibrate_confidence()
        if confidence_change:
            changes['confidence'] = confidence_change
        
        # 2. Adjust strategy weights
        weight_changes = self._adjust_strategy_weights()
        if weight_changes:
            changes['strategy_weights'] = weight_changes
        
        # 3. Adjust symbol weights
        symbol_changes = self._adjust_symbol_weights()
        if symbol_changes:
            changes['symbol_weights'] = symbol_changes
        
        # 4. Adjust risk multiplier based on recent performance
        risk_change = self._adjust_risk()
        if risk_change:
            changes['risk'] = risk_change
        
        if changes:
            logger.info(f"OnlineLearner updated parameters: {changes}")
        
        return changes
    
    def _calibrate_confidence(self) -> Optional[Dict]:
        """Calibrate confidence thresholds based on outcomes"""
        if len(self.confidence_outcomes) < 10:
            return None
        
        # Group outcomes by confidence level
        high_conf = [(c, w) for c, w in self.confidence_outcomes if c >= 0.75]
        med_conf = [(c, w) for c, w in self.confidence_outcomes if 0.6 <= c < 0.75]
        low_conf = [(c, w) for c, w in self.confidence_outcomes if c < 0.6]
        
        changes = {}
        
        # If high confidence trades aren't winning enough, raise threshold
        if high_conf:
            high_win_rate = sum(1 for _, w in high_conf if w) / len(high_conf)
            if high_win_rate < 0.55:
                new_threshold = min(0.85, self.params.high_confidence_threshold + 0.02)
                if new_threshold != self.params.high_confidence_threshold:
                    self.params.high_confidence_threshold = new_threshold
                    changes['high_confidence_threshold'] = new_threshold
            elif high_win_rate > 0.70:
                new_threshold = max(0.70, self.params.high_confidence_threshold - 0.02)
                if new_threshold != self.params.high_confidence_threshold:
                    self.params.high_confidence_threshold = new_threshold
                    changes['high_confidence_threshold'] = new_threshold
        
        # Adjust minimum confidence threshold
        if low_conf:
            low_win_rate = sum(1 for _, w in low_conf if w) / len(low_conf)
            if low_win_rate < 0.40:
                new_threshold = min(0.75, self.params.min_confidence_threshold + 0.02)
                if new_threshold != self.params.min_confidence_threshold:
                    self.params.min_confidence_threshold = new_threshold
                    changes['min_confidence_threshold'] = new_threshold
        
        return changes if changes else None
    
    def _adjust_strategy_weights(self) -> Optional[Dict]:
        """Adjust strategy weights based on performance"""
        # Need enough data for each strategy
        min_trades = 5
        strategy_scores = {}
        
        for strategy, pnls in self.strategy_performance.items():
            recent_pnls = pnls[-self.performance_window:]
            if len(recent_pnls) >= min_trades:
                win_rate = sum(1 for p in recent_pnls if p > 0) / len(recent_pnls)
                avg_pnl = statistics.mean(recent_pnls)
                # Score combines win rate and average PnL
                strategy_scores[strategy] = win_rate * 0.5 + (1 if avg_pnl > 0 else 0) * 0.5
        
        if not strategy_scores:
            return None
        
        # Normalize scores to weights
        total_score = sum(strategy_scores.values())
        if total_score <= 0:
            return None
        
        changes = {}
        for strategy, score in strategy_scores.items():
            new_weight = score / total_score
            # Blend with current weight (slow adaptation)
            current_attr = f"{strategy}_weight"
            if hasattr(self.params, current_attr):
                current_weight = getattr(self.params, current_attr)
                blended_weight = current_weight * (1 - self._learning_rate) + new_weight * self._learning_rate
                # Ensure minimum weight of 0.1
                blended_weight = max(0.1, min(0.6, blended_weight))
                setattr(self.params, current_attr, blended_weight)
                changes[current_attr] = blended_weight
        
        # Normalize weights to sum to 1
        total = (self.params.trend_following_weight + 
                 self.params.mean_reversion_weight + 
                 self.params.breakout_weight)
        if total > 0:
            self.params.trend_following_weight /= total
            self.params.mean_reversion_weight /= total
            self.params.breakout_weight /= total
        
        return changes if changes else None
    
    def _adjust_symbol_weights(self) -> Optional[Dict]:
        """Adjust symbol weights based on performance"""
        min_trades = 3
        changes = {}
        
        for symbol, pnls in self.symbol_performance.items():
            recent_pnls = pnls[-self.performance_window:]
            if len(recent_pnls) >= min_trades:
                win_rate = sum(1 for p in recent_pnls if p > 0) / len(recent_pnls)
                
                current_weight = self.params.symbol_weights.get(symbol, 0.8)
                
                # Adjust weight based on win rate
                if win_rate > 0.6:
                    new_weight = min(1.0, current_weight + 0.05)
                elif win_rate < 0.4:
                    new_weight = max(0.3, current_weight - 0.05)
                else:
                    new_weight = current_weight
                
                if new_weight != current_weight:
                    self.params.symbol_weights[symbol] = new_weight
                    changes[symbol] = new_weight
        
        return changes if changes else None
    
    def _adjust_risk(self) -> Optional[Dict]:
        """Adjust risk multiplier based on recent performance"""
        if len(self.recent_trades) < 5:
            return None
        
        recent = self.recent_trades[-10:]
        total_pnl = sum(t.pnl for t in recent)
        win_rate = sum(1 for t in recent if t.pnl > 0) / len(recent)
        
        current_multiplier = self.params.risk_multiplier
        
        # Reduce risk if losing
        if total_pnl < 0 and win_rate < 0.4:
            new_multiplier = max(0.5, current_multiplier - 0.1)
        # Increase risk if winning consistently
        elif total_pnl > 0 and win_rate > 0.6:
            new_multiplier = min(1.5, current_multiplier + 0.05)
        else:
            new_multiplier = current_multiplier
        
        if new_multiplier != current_multiplier:
            self.params.risk_multiplier = new_multiplier
            return {'risk_multiplier': new_multiplier}
        
        return None
    
    def get_adjusted_confidence(self, base_confidence: float, symbol: str, strategy: str) -> float:
        """Get adjusted confidence based on learned parameters"""
        adjusted = base_confidence
        
        # Apply symbol weight
        symbol_weight = self.params.symbol_weights.get(symbol, 0.8)
        adjusted *= symbol_weight
        
        # Apply strategy weight
        strategy_key = strategy.lower().replace(' ', '_')
        strategy_weight = getattr(self.params, f"{strategy_key}_weight", 0.33)
        # Normalize strategy weight effect (centered around 0.33)
        strategy_factor = 0.7 + (strategy_weight / 0.33) * 0.3
        adjusted *= strategy_factor
        
        return min(1.0, max(0.0, adjusted))
    
    def get_adjusted_risk(self, base_risk: float) -> float:
        """Get adjusted risk based on learned parameters"""
        adjusted = base_risk * self.params.risk_multiplier
        return min(self.params.max_risk_percent, max(0.5, adjusted))


class OfflineLearner:
    """
    Heavy computation learning during market closed hours.
    Performs model retraining, backtesting, and optimization.
    """
    
    def __init__(self, params: AdaptiveParameters, data_dir: str = "data/learning"):
        self.params = params
        self.data_dir = data_dir
        self.all_trades: List[TradeOutcome] = []
        self._is_running = False
        self._last_run = None
        
        # Ensure data directory exists
        os.makedirs(data_dir, exist_ok=True)
        
        # Load historical trades
        self._load_trades()
        
        logger.info(f"OfflineLearner initialized with {len(self.all_trades)} historical trades")
    
    def _load_trades(self):
        """Load historical trades from disk"""
        trades_file = os.path.join(self.data_dir, "trade_history.json")
        if os.path.exists(trades_file):
            try:
                with open(trades_file, 'r') as f:
                    data = json.load(f)
                    self.all_trades = [TradeOutcome.from_dict(t) for t in data]
            except Exception as e:
                logger.warning(f"Failed to load trade history: {e}")
    
    def _save_trades(self):
        """Save trades to disk"""
        trades_file = os.path.join(self.data_dir, "trade_history.json")
        try:
            with open(trades_file, 'w') as f:
                json.dump([t.to_dict() for t in self.all_trades], f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save trade history: {e}")
    
    def add_trade(self, trade: TradeOutcome):
        """Add a trade to history"""
        self.all_trades.append(trade)
        self._save_trades()
    
    def run_offline_learning(self) -> Dict[str, Any]:
        """
        Run comprehensive offline learning.
        This is compute-intensive and should only run when market is closed.
        """
        if self._is_running:
            logger.warning("Offline learning already running")
            return {}
        
        self._is_running = True
        results = {}
        
        try:
            logger.info("Starting offline learning cycle...")
            
            # 1. Analyze overall performance
            results['performance'] = self._analyze_performance()
            
            # 2. Analyze by time period (regime analysis)
            results['regime_analysis'] = self._analyze_regimes()
            
            # 3. Optimize parameters
            results['optimization'] = self._optimize_parameters()
            
            # 4. Walk-forward analysis
            results['walk_forward'] = self._walk_forward_analysis()
            
            # 5. Pattern discovery
            results['patterns'] = self._discover_patterns()
            
            self._last_run = datetime.now()
            logger.info(f"Offline learning complete: {results}")
            
        except Exception as e:
            logger.error(f"Offline learning error: {e}")
            results['error'] = str(e)
        finally:
            self._is_running = False
        
        return results
    
    def _analyze_performance(self) -> Dict:
        """Analyze overall trading performance"""
        if not self.all_trades:
            return {'status': 'no_data'}
        
        metrics = PerformanceMetrics(
            period_start=min(t.entry_time for t in self.all_trades),
            period_end=max(t.exit_time for t in self.all_trades)
        )
        metrics.calculate(self.all_trades)
        
        return {
            'total_trades': metrics.total_trades,
            'win_rate': metrics.win_rate,
            'profit_factor': metrics.profit_factor,
            'expectancy': metrics.expectancy,
            'max_drawdown': metrics.max_drawdown,
            'sharpe_ratio': metrics.sharpe_ratio
        }
    
    def _analyze_regimes(self) -> Dict:
        """Analyze performance by market regime"""
        if not self.all_trades:
            return {'status': 'no_data'}
        
        # Group trades by regime
        regime_trades: Dict[str, List[TradeOutcome]] = {}
        for trade in self.all_trades:
            regime = trade.regime_at_entry or 'unknown'
            if regime not in regime_trades:
                regime_trades[regime] = []
            regime_trades[regime].append(trade)
        
        regime_analysis = {}
        for regime, trades in regime_trades.items():
            if len(trades) >= 3:
                win_rate = sum(1 for t in trades if t.pnl > 0) / len(trades)
                avg_pnl = statistics.mean([t.pnl for t in trades])
                regime_analysis[regime] = {
                    'trades': len(trades),
                    'win_rate': win_rate,
                    'avg_pnl': avg_pnl,
                    'recommendation': 'trade' if win_rate > 0.5 and avg_pnl > 0 else 'avoid'
                }
        
        return regime_analysis
    
    def _optimize_parameters(self) -> Dict:
        """Optimize trading parameters based on historical data"""
        if len(self.all_trades) < 20:
            return {'status': 'insufficient_data'}
        
        optimizations = {}
        
        # Optimize confidence threshold
        confidence_levels = [0.60, 0.65, 0.70, 0.75, 0.80]
        best_threshold = self.params.min_confidence_threshold
        best_expectancy = -float('inf')
        
        for threshold in confidence_levels:
            filtered_trades = [t for t in self.all_trades if t.confidence_at_entry >= threshold]
            if len(filtered_trades) >= 10:
                win_rate = sum(1 for t in filtered_trades if t.pnl > 0) / len(filtered_trades)
                avg_win = statistics.mean([t.pnl for t in filtered_trades if t.pnl > 0]) if any(t.pnl > 0 for t in filtered_trades) else 0
                avg_loss = statistics.mean([abs(t.pnl) for t in filtered_trades if t.pnl < 0]) if any(t.pnl < 0 for t in filtered_trades) else 0
                expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
                
                if expectancy > best_expectancy:
                    best_expectancy = expectancy
                    best_threshold = threshold
        
        if best_threshold != self.params.min_confidence_threshold:
            self.params.min_confidence_threshold = best_threshold
            optimizations['min_confidence_threshold'] = best_threshold
        
        # Optimize SL/TP multipliers
        sl_options = [1.5, 2.0, 2.5, 3.0]
        tp_options = [2.0, 3.0, 4.0, 5.0]
        
        # Simple grid search (in production, use more sophisticated optimization)
        best_sl = self.params.sl_atr_multiplier
        best_tp = self.params.tp_atr_multiplier
        best_pf = 0
        
        for sl in sl_options:
            for tp in tp_options:
                if tp > sl:  # TP should be greater than SL
                    # Simulate with these parameters
                    simulated_pf = self._simulate_sl_tp(sl, tp)
                    if simulated_pf > best_pf:
                        best_pf = simulated_pf
                        best_sl = sl
                        best_tp = tp
        
        if best_sl != self.params.sl_atr_multiplier:
            self.params.sl_atr_multiplier = best_sl
            optimizations['sl_atr_multiplier'] = best_sl
        if best_tp != self.params.tp_atr_multiplier:
            self.params.tp_atr_multiplier = best_tp
            optimizations['tp_atr_multiplier'] = best_tp
        
        return optimizations if optimizations else {'status': 'no_changes'}
    
    def _simulate_sl_tp(self, sl_mult: float, tp_mult: float) -> float:
        """Simulate profit factor with given SL/TP multipliers"""
        # Simplified simulation based on historical trade characteristics
        if not self.all_trades:
            return 0
        
        wins = 0
        losses = 0
        
        for trade in self.all_trades:
            # Estimate outcome based on actual trade direction
            if trade.pnl > 0:
                wins += tp_mult
            else:
                losses += sl_mult
        
        return wins / losses if losses > 0 else 0
    
    def _walk_forward_analysis(self) -> Dict:
        """Perform walk-forward analysis"""
        if len(self.all_trades) < 30:
            return {'status': 'insufficient_data'}
        
        # Sort trades by time
        sorted_trades = sorted(self.all_trades, key=lambda t: t.entry_time)
        
        # Split into training and testing periods
        split_idx = int(len(sorted_trades) * 0.7)
        train_trades = sorted_trades[:split_idx]
        test_trades = sorted_trades[split_idx:]
        
        # Calculate metrics for each period
        train_metrics = PerformanceMetrics(
            period_start=train_trades[0].entry_time,
            period_end=train_trades[-1].exit_time
        )
        train_metrics.calculate(train_trades)
        
        test_metrics = PerformanceMetrics(
            period_start=test_trades[0].entry_time,
            period_end=test_trades[-1].exit_time
        )
        test_metrics.calculate(test_trades)
        
        # Check for overfitting (test performance significantly worse than train)
        overfit_warning = False
        if train_metrics.win_rate > 0 and test_metrics.win_rate > 0:
            if test_metrics.win_rate < train_metrics.win_rate * 0.7:
                overfit_warning = True
        
        return {
            'train_win_rate': train_metrics.win_rate,
            'test_win_rate': test_metrics.win_rate,
            'train_expectancy': train_metrics.expectancy,
            'test_expectancy': test_metrics.expectancy,
            'overfit_warning': overfit_warning
        }
    
    def _discover_patterns(self) -> Dict:
        """Discover winning and losing patterns"""
        if len(self.all_trades) < 20:
            return {'status': 'insufficient_data'}
        
        patterns = {
            'winning_conditions': [],
            'losing_conditions': []
        }
        
        # Analyze by time of day
        hour_performance: Dict[int, List[float]] = {}
        for trade in self.all_trades:
            hour = trade.entry_time.hour
            if hour not in hour_performance:
                hour_performance[hour] = []
            hour_performance[hour].append(trade.pnl)
        
        best_hours = []
        worst_hours = []
        for hour, pnls in hour_performance.items():
            if len(pnls) >= 3:
                win_rate = sum(1 for p in pnls if p > 0) / len(pnls)
                if win_rate > 0.6:
                    best_hours.append(hour)
                elif win_rate < 0.4:
                    worst_hours.append(hour)
        
        if best_hours:
            patterns['winning_conditions'].append(f"Best hours (UTC): {best_hours}")
        if worst_hours:
            patterns['losing_conditions'].append(f"Avoid hours (UTC): {worst_hours}")
        
        # Analyze by day of week
        day_performance: Dict[int, List[float]] = {}
        for trade in self.all_trades:
            day = trade.entry_time.weekday()
            if day not in day_performance:
                day_performance[day] = []
            day_performance[day].append(trade.pnl)
        
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
        for day, pnls in day_performance.items():
            if len(pnls) >= 3:
                win_rate = sum(1 for p in pnls if p > 0) / len(pnls)
                if win_rate > 0.65:
                    patterns['winning_conditions'].append(f"Strong day: {day_names[day]}")
                elif win_rate < 0.35:
                    patterns['losing_conditions'].append(f"Weak day: {day_names[day]}")
        
        return patterns


class AdaptiveLearningEngine:
    """
    Unified interface for adaptive learning.
    Manages both online and offline learning modes.
    """
    
    def __init__(self, data_dir: str = "data/learning"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # Load or create adaptive parameters
        self.params = self._load_params()
        
        # Initialize components
        self.market_hours = MarketHoursDetector()
        self.online_learner = OnlineLearner(self.params)
        self.offline_learner = OfflineLearner(self.params, data_dir)
        
        # Background thread for learning
        self._running = False
        self._thread = None
        self._last_offline_run = None
        
        logger.info("AdaptiveLearningEngine initialized")
    
    def _load_params(self) -> AdaptiveParameters:
        """Load parameters from disk"""
        params_file = os.path.join(self.data_dir, "adaptive_params.json")
        if os.path.exists(params_file):
            try:
                with open(params_file, 'r') as f:
                    data = json.load(f)
                    return AdaptiveParameters.from_dict(data)
            except Exception as e:
                logger.warning(f"Failed to load params: {e}")
        return AdaptiveParameters()
    
    def _save_params(self):
        """Save parameters to disk"""
        params_file = os.path.join(self.data_dir, "adaptive_params.json")
        try:
            with open(params_file, 'w') as f:
                json.dump(self.params.to_dict(), f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save params: {e}")
    
    def get_learning_mode(self) -> LearningMode:
        """Get current learning mode"""
        return self.market_hours.get_learning_mode()
    
    def record_trade(self, trade: TradeOutcome):
        """Record a completed trade"""
        self.online_learner.record_trade(trade)
        self.offline_learner.add_trade(trade)
        logger.info(f"Trade recorded: {trade.symbol} {trade.direction} PnL={trade.pnl:.2f}")
    
    def update(self) -> Dict[str, Any]:
        """
        Perform learning update based on current mode.
        Call this periodically (e.g., every minute).
        """
        mode = self.get_learning_mode()
        results = {'mode': mode.value}
        
        if mode == LearningMode.ONLINE:
            # Online learning - lightweight updates
            if self.online_learner.should_update():
                changes = self.online_learner.update_parameters()
                if changes:
                    results['online_changes'] = changes
                    self._save_params()
        
        elif mode == LearningMode.OFFLINE:
            # Offline learning - run heavy computation if not done recently
            if self._should_run_offline():
                results['offline_results'] = self.offline_learner.run_offline_learning()
                self._last_offline_run = datetime.now()
                self._save_params()
        
        return results
    
    def _should_run_offline(self) -> bool:
        """Check if offline learning should run"""
        if self._last_offline_run is None:
            return True
        # Run offline learning once per market close period
        hours_since_last = (datetime.now() - self._last_offline_run).total_seconds() / 3600
        return hours_since_last >= 12  # At least 12 hours between runs
    
    def get_adjusted_confidence(self, base_confidence: float, symbol: str, strategy: str) -> float:
        """Get adjusted confidence from online learner"""
        return self.online_learner.get_adjusted_confidence(base_confidence, symbol, strategy)
    
    def get_adjusted_risk(self, base_risk: float) -> float:
        """Get adjusted risk from online learner"""
        return self.online_learner.get_adjusted_risk(base_risk)
    
    def get_parameters(self) -> AdaptiveParameters:
        """Get current adaptive parameters"""
        return self.params
    
    def get_session_quality(self) -> float:
        """Get current session quality"""
        return self.market_hours.get_session_quality()
    
    def is_market_open(self) -> bool:
        """Check if market is open"""
        return self.market_hours.is_market_open()
    
    def start_background_updates(self, interval_seconds: int = 60):
        """Start background learning updates"""
        if self._running:
            return
        
        self._running = True
        
        def update_loop():
            while self._running:
                try:
                    results = self.update()
                    if results.get('online_changes') or results.get('offline_results'):
                        logger.info(f"Learning update: {results}")
                except Exception as e:
                    logger.error(f"Learning update error: {e}")
                time.sleep(interval_seconds)
        
        self._thread = threading.Thread(target=update_loop, daemon=True)
        self._thread.start()
        logger.info(f"Background learning started (interval: {interval_seconds}s)")
    
    def stop_background_updates(self):
        """Stop background learning updates"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Background learning stopped")
    
    def get_state_summary(self) -> Dict[str, Any]:
        """Get summary of learning state"""
        return {
            'mode': self.get_learning_mode().value,
            'market_open': self.is_market_open(),
            'session': self.market_hours.get_current_session().value,
            'session_quality': self.get_session_quality(),
            'total_trades_recorded': len(self.offline_learner.all_trades),
            'recent_trades': len(self.online_learner.recent_trades),
            'parameters': {
                'min_confidence': self.params.min_confidence_threshold,
                'risk_multiplier': self.params.risk_multiplier,
                'strategy_weights': {
                    'trend_following': self.params.trend_following_weight,
                    'mean_reversion': self.params.mean_reversion_weight,
                    'breakout': self.params.breakout_weight
                }
            },
            'last_offline_run': self._last_offline_run.isoformat() if self._last_offline_run else None
        }


# Global instance
_adaptive_learning: Optional[AdaptiveLearningEngine] = None


def get_adaptive_learning() -> AdaptiveLearningEngine:
    """Get or create the global adaptive learning engine"""
    global _adaptive_learning
    if _adaptive_learning is None:
        _adaptive_learning = AdaptiveLearningEngine()
    return _adaptive_learning


def record_trade_outcome(
    symbol: str,
    direction: str,
    entry_price: float,
    exit_price: float,
    entry_time: datetime,
    exit_time: datetime,
    pnl: float,
    pnl_pips: float,
    strategy: str,
    confidence: float,
    regime: str,
    indicators: Dict[str, float] = None
):
    """Convenience function to record a trade outcome"""
    trade = TradeOutcome(
        symbol=symbol,
        direction=direction,
        entry_price=entry_price,
        exit_price=exit_price,
        entry_time=entry_time,
        exit_time=exit_time,
        pnl=pnl,
        pnl_pips=pnl_pips,
        strategy=strategy,
        confidence_at_entry=confidence,
        regime_at_entry=regime,
        indicators_at_entry=indicators or {}
    )
    get_adaptive_learning().record_trade(trade)


if __name__ == "__main__":
    # Test the adaptive learning system
    logging.basicConfig(level=logging.INFO)
    
    engine = get_adaptive_learning()
    
    print(f"Learning mode: {engine.get_learning_mode().value}")
    print(f"Market open: {engine.is_market_open()}")
    print(f"Session: {engine.market_hours.get_current_session().value}")
    print(f"Session quality: {engine.get_session_quality()}")
    
    # Simulate some trades
    for i in range(10):
        trade = TradeOutcome(
            symbol="EURUSD",
            direction="BUY" if i % 2 == 0 else "SELL",
            entry_price=1.1000 + i * 0.001,
            exit_price=1.1010 + i * 0.001 if i % 3 != 0 else 1.0990 + i * 0.001,
            entry_time=datetime.now() - timedelta(hours=i),
            exit_time=datetime.now() - timedelta(hours=i-1),
            pnl=10 if i % 3 != 0 else -15,
            pnl_pips=10 if i % 3 != 0 else -15,
            strategy="Trend Following",
            confidence_at_entry=0.7 + (i % 3) * 0.05,
            regime_at_entry="trending"
        )
        engine.record_trade(trade)
    
    # Run update
    results = engine.update()
    print(f"Update results: {results}")
    
    # Get state summary
    print(f"State summary: {engine.get_state_summary()}")
