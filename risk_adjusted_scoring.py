"""
Risk-Adjusted Scoring System - Priority 1
Implements institutional-grade performance evaluation like Warren Buffett/Aladdin.
Optimizes for risk-adjusted returns, not just P&L.

Key Metrics:
- Sharpe Ratio (risk-adjusted return)
- Sortino Ratio (downside risk-adjusted)
- Maximum Drawdown and Recovery
- Consistency Score (steady gains > volatile wins)
- Tail Risk (CVaR/Expected Shortfall)
- Turnover Penalty (overtrading hurts)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import json
import os
import statistics

logger = logging.getLogger(__name__)


class PerformanceGrade(Enum):
    """Performance grades like a report card"""
    EXCEPTIONAL = "A+"  # Top 5% - Buffett-level
    EXCELLENT = "A"     # Top 15%
    GOOD = "B"          # Top 35%
    AVERAGE = "C"       # Middle 30%
    BELOW_AVERAGE = "D" # Bottom 25%
    POOR = "F"          # Bottom 10%


@dataclass
class TradeRecord:
    """Individual trade record for analysis"""
    trade_id: str
    symbol: str
    direction: str
    entry_time: datetime
    exit_time: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    position_size: float
    pnl: float = 0.0
    pnl_percent: float = 0.0
    holding_period_hours: float = 0.0
    strategy: str = ""
    regime: str = ""
    
    # Risk metrics at entry
    stop_loss: float = 0.0
    take_profit: float = 0.0
    risk_reward_ratio: float = 0.0
    
    # Execution quality
    slippage: float = 0.0
    spread_cost: float = 0.0
    
    def __post_init__(self):
        if self.exit_time and self.entry_time:
            self.holding_period_hours = (self.exit_time - self.entry_time).total_seconds() / 3600


@dataclass
class RiskAdjustedScore:
    """Comprehensive risk-adjusted performance score"""
    # Core metrics
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0  # Return / Max Drawdown
    
    # Drawdown metrics
    max_drawdown: float = 0.0
    avg_drawdown: float = 0.0
    drawdown_duration_days: float = 0.0
    recovery_factor: float = 0.0  # Total return / Max Drawdown
    
    # Consistency metrics
    win_rate: float = 0.0
    profit_factor: float = 0.0  # Gross profit / Gross loss
    avg_win_loss_ratio: float = 0.0
    consistency_score: float = 0.0  # Low variance in returns
    
    # Tail risk
    var_95: float = 0.0  # Value at Risk 95%
    cvar_95: float = 0.0  # Conditional VaR (Expected Shortfall)
    tail_ratio: float = 0.0  # Right tail / Left tail
    
    # Turnover and costs
    turnover_rate: float = 0.0  # Trades per day
    avg_holding_period: float = 0.0
    total_costs: float = 0.0
    cost_adjusted_return: float = 0.0
    
    # Composite scores
    raw_score: float = 0.0  # 0-100
    risk_adjusted_score: float = 0.0  # 0-100 with penalties
    grade: PerformanceGrade = PerformanceGrade.AVERAGE
    
    # Penalties applied
    overtrading_penalty: float = 0.0
    drawdown_penalty: float = 0.0
    inconsistency_penalty: float = 0.0
    tail_risk_penalty: float = 0.0
    
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict:
        return {
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'calmar_ratio': self.calmar_ratio,
            'max_drawdown': self.max_drawdown,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'consistency_score': self.consistency_score,
            'var_95': self.var_95,
            'cvar_95': self.cvar_95,
            'turnover_rate': self.turnover_rate,
            'raw_score': self.raw_score,
            'risk_adjusted_score': self.risk_adjusted_score,
            'grade': self.grade.value,
            'penalties': {
                'overtrading': self.overtrading_penalty,
                'drawdown': self.drawdown_penalty,
                'inconsistency': self.inconsistency_penalty,
                'tail_risk': self.tail_risk_penalty
            }
        }


class RiskAdjustedScorer:
    """
    Calculates risk-adjusted performance scores.
    This is what separates a "trading bot" from an "investor".
    
    Warren Buffett's approach:
    - Focus on risk-adjusted returns, not just returns
    - Penalize excessive trading (turnover)
    - Value consistency over volatility
    - Protect against tail risks
    """
    
    # Benchmark values for scoring (based on institutional standards)
    BENCHMARKS = {
        'sharpe_excellent': 2.0,      # Sharpe > 2 is excellent
        'sharpe_good': 1.0,           # Sharpe > 1 is good
        'sharpe_acceptable': 0.5,     # Sharpe > 0.5 is acceptable
        
        'sortino_excellent': 3.0,
        'sortino_good': 1.5,
        
        'max_dd_excellent': 0.05,     # < 5% max DD is excellent
        'max_dd_good': 0.10,          # < 10% is good
        'max_dd_acceptable': 0.20,    # < 20% is acceptable
        
        'win_rate_excellent': 0.60,
        'win_rate_good': 0.50,
        
        'profit_factor_excellent': 2.0,
        'profit_factor_good': 1.5,
        
        'turnover_max_daily': 3,      # Max 3 trades/day before penalty
        'min_holding_hours': 1,       # Min 1 hour holding before penalty
    }
    
    def __init__(self, risk_free_rate: float = 0.05):
        """
        Initialize scorer.
        
        Args:
            risk_free_rate: Annual risk-free rate (default 5%)
        """
        self.risk_free_rate = risk_free_rate
        self.daily_rf = risk_free_rate / 252  # Daily risk-free rate
        
        self.trade_history: List[TradeRecord] = []
        self.daily_returns: List[float] = []
        self.equity_curve: List[Tuple[datetime, float]] = []
        
        # Score history for tracking improvement
        self.score_history: List[RiskAdjustedScore] = []
        
        # Load historical data if exists
        self._load_history()
        
        logger.info("RiskAdjustedScorer initialized")
    
    def _load_history(self):
        """Load historical trade and score data"""
        history_file = "data/risk_adjusted_history.json"
        if os.path.exists(history_file):
            try:
                with open(history_file, 'r') as f:
                    data = json.load(f)
                    self.daily_returns = data.get('daily_returns', [])
                    logger.info(f"Loaded {len(self.daily_returns)} daily returns from history")
            except Exception as e:
                logger.warning(f"Could not load history: {e}")
    
    def _save_history(self):
        """Save historical data"""
        os.makedirs("data", exist_ok=True)
        history_file = "data/risk_adjusted_history.json"
        try:
            with open(history_file, 'w') as f:
                json.dump({
                    'daily_returns': self.daily_returns[-252:],  # Keep 1 year
                    'last_updated': datetime.utcnow().isoformat()
                }, f)
        except Exception as e:
            logger.warning(f"Could not save history: {e}")
    
    def record_trade(self, trade: TradeRecord):
        """Record a completed trade"""
        self.trade_history.append(trade)
        
        # Keep last 500 trades
        if len(self.trade_history) > 500:
            self.trade_history = self.trade_history[-500:]
    
    def record_daily_return(self, return_pct: float, date: datetime = None):
        """Record daily return for time-series analysis"""
        self.daily_returns.append(return_pct)
        
        # Keep last 252 trading days (1 year)
        if len(self.daily_returns) > 252:
            self.daily_returns = self.daily_returns[-252:]
        
        self._save_history()
    
    def update_equity_curve(self, equity: float, timestamp: datetime = None):
        """Update equity curve for drawdown analysis"""
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        self.equity_curve.append((timestamp, equity))
        
        # Keep last 1000 points
        if len(self.equity_curve) > 1000:
            self.equity_curve = self.equity_curve[-1000:]
    
    def calculate_sharpe_ratio(self, returns: List[float] = None) -> float:
        """
        Calculate annualized Sharpe Ratio.
        Sharpe = (Mean Return - Risk Free Rate) / Std Dev of Returns
        """
        if returns is None:
            returns = self.daily_returns
        
        if len(returns) < 5:
            return 0.0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)
        
        if std_return == 0:
            return 0.0
        
        # Annualize (assuming daily returns)
        daily_sharpe = (mean_return - self.daily_rf) / std_return
        annualized_sharpe = daily_sharpe * np.sqrt(252)
        
        return annualized_sharpe
    
    def calculate_sortino_ratio(self, returns: List[float] = None) -> float:
        """
        Calculate Sortino Ratio (only penalizes downside volatility).
        Better than Sharpe for asymmetric return distributions.
        """
        if returns is None:
            returns = self.daily_returns
        
        if len(returns) < 5:
            return 0.0
        
        mean_return = np.mean(returns)
        
        # Downside deviation (only negative returns)
        negative_returns = [r for r in returns if r < 0]
        if not negative_returns:
            return 10.0  # Very high if no negative returns
        
        downside_std = np.std(negative_returns, ddof=1)
        
        if downside_std == 0:
            return 10.0
        
        daily_sortino = (mean_return - self.daily_rf) / downside_std
        annualized_sortino = daily_sortino * np.sqrt(252)
        
        return annualized_sortino
    
    def calculate_max_drawdown(self) -> Tuple[float, float, float]:
        """
        Calculate maximum drawdown from equity curve.
        Returns: (max_dd, avg_dd, dd_duration_days)
        """
        if len(self.equity_curve) < 2:
            return 0.0, 0.0, 0.0
        
        equities = [e[1] for e in self.equity_curve]
        timestamps = [e[0] for e in self.equity_curve]
        
        peak = equities[0]
        max_dd = 0.0
        drawdowns = []
        dd_start = None
        max_dd_duration = timedelta(0)
        
        for i, equity in enumerate(equities):
            if equity > peak:
                peak = equity
                if dd_start is not None:
                    duration = timestamps[i] - dd_start
                    if duration > max_dd_duration:
                        max_dd_duration = duration
                dd_start = None
            else:
                dd = (peak - equity) / peak
                drawdowns.append(dd)
                if dd > max_dd:
                    max_dd = dd
                if dd_start is None:
                    dd_start = timestamps[i]
        
        avg_dd = np.mean(drawdowns) if drawdowns else 0.0
        dd_duration_days = max_dd_duration.total_seconds() / 86400
        
        return max_dd, avg_dd, dd_duration_days
    
    def calculate_var_cvar(self, returns: List[float] = None, 
                           confidence: float = 0.95) -> Tuple[float, float]:
        """
        Calculate Value at Risk and Conditional VaR (Expected Shortfall).
        VaR: Maximum expected loss at confidence level
        CVaR: Average loss beyond VaR (tail risk)
        """
        if returns is None:
            returns = self.daily_returns
        
        if len(returns) < 10:
            return 0.0, 0.0
        
        sorted_returns = sorted(returns)
        var_index = int((1 - confidence) * len(sorted_returns))
        
        var = -sorted_returns[var_index] if var_index < len(sorted_returns) else 0.0
        
        # CVaR is average of returns worse than VaR
        tail_returns = sorted_returns[:var_index + 1]
        cvar = -np.mean(tail_returns) if tail_returns else var
        
        return var, cvar
    
    def calculate_consistency_score(self, returns: List[float] = None) -> float:
        """
        Calculate consistency score (0-100).
        Measures how steady the returns are.
        Buffett prefers steady 15% over volatile 30%.
        """
        if returns is None:
            returns = self.daily_returns
        
        if len(returns) < 10:
            return 50.0
        
        # Calculate coefficient of variation (lower is more consistent)
        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)
        
        if mean_return <= 0:
            return 0.0
        
        cv = std_return / abs(mean_return) if mean_return != 0 else float('inf')
        
        # Convert to score (lower CV = higher score)
        # CV of 0.5 = 100, CV of 2.0 = 50, CV of 5.0 = 0
        consistency = max(0, min(100, 100 - (cv - 0.5) * 33.33))
        
        # Bonus for positive skew (more big wins than big losses)
        if len(returns) >= 20:
            try:
                from scipy.stats import skew
                skewness = skew(returns)
                if skewness > 0:
                    consistency += min(10, skewness * 5)
            except ImportError:
                # Manual skewness calculation
                n = len(returns)
                m3 = sum((r - mean_return) ** 3 for r in returns) / n
                skewness = m3 / (std_return ** 3) if std_return > 0 else 0
                if skewness > 0:
                    consistency += min(10, skewness * 5)
        
        return min(100, consistency)
    
    def calculate_turnover_metrics(self, trades: List[TradeRecord] = None,
                                    days: int = 30) -> Tuple[float, float]:
        """
        Calculate turnover rate and average holding period.
        Returns: (trades_per_day, avg_holding_hours)
        """
        if trades is None:
            trades = self.trade_history
        
        if not trades:
            return 0.0, 0.0
        
        # Filter to recent trades
        cutoff = datetime.utcnow() - timedelta(days=days)
        recent_trades = [t for t in trades if t.entry_time > cutoff]
        
        if not recent_trades:
            return 0.0, 0.0
        
        trades_per_day = len(recent_trades) / days
        
        holding_periods = [t.holding_period_hours for t in recent_trades 
                          if t.holding_period_hours > 0]
        avg_holding = np.mean(holding_periods) if holding_periods else 0.0
        
        return trades_per_day, avg_holding
    
    def calculate_profit_factor(self, trades: List[TradeRecord] = None) -> float:
        """
        Calculate profit factor (gross profit / gross loss).
        PF > 1.5 is good, > 2.0 is excellent.
        """
        if trades is None:
            trades = self.trade_history
        
        if not trades:
            return 1.0
        
        gross_profit = sum(t.pnl for t in trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in trades if t.pnl < 0))
        
        if gross_loss == 0:
            return 10.0 if gross_profit > 0 else 1.0
        
        return gross_profit / gross_loss
    
    def calculate_win_rate(self, trades: List[TradeRecord] = None) -> float:
        """Calculate win rate"""
        if trades is None:
            trades = self.trade_history
        
        if not trades:
            return 0.5
        
        wins = sum(1 for t in trades if t.pnl > 0)
        return wins / len(trades)
    
    def calculate_penalties(self, turnover: float, max_dd: float,
                           consistency: float, cvar: float) -> Dict[str, float]:
        """
        Calculate penalties for poor behavior.
        These reduce the raw score to get risk-adjusted score.
        """
        penalties = {}
        
        # Overtrading penalty (> 3 trades/day)
        if turnover > self.BENCHMARKS['turnover_max_daily']:
            excess = turnover - self.BENCHMARKS['turnover_max_daily']
            penalties['overtrading'] = min(20, excess * 5)  # Up to 20 points
        else:
            penalties['overtrading'] = 0
        
        # Drawdown penalty (> 10% max DD)
        if max_dd > self.BENCHMARKS['max_dd_good']:
            excess = (max_dd - self.BENCHMARKS['max_dd_good']) * 100
            penalties['drawdown'] = min(25, excess * 2)  # Up to 25 points
        else:
            penalties['drawdown'] = 0
        
        # Inconsistency penalty (consistency < 50)
        if consistency < 50:
            penalties['inconsistency'] = (50 - consistency) * 0.3  # Up to 15 points
        else:
            penalties['inconsistency'] = 0
        
        # Tail risk penalty (CVaR > 3%)
        if cvar > 0.03:
            excess = (cvar - 0.03) * 100
            penalties['tail_risk'] = min(15, excess * 3)  # Up to 15 points
        else:
            penalties['tail_risk'] = 0
        
        return penalties
    
    def calculate_raw_score(self, sharpe: float, sortino: float, 
                           win_rate: float, profit_factor: float,
                           max_dd: float, consistency: float) -> float:
        """
        Calculate raw performance score (0-100).
        Weighted combination of all metrics.
        """
        score = 0.0
        
        # Sharpe contribution (25 points max)
        if sharpe >= self.BENCHMARKS['sharpe_excellent']:
            score += 25
        elif sharpe >= self.BENCHMARKS['sharpe_good']:
            score += 20
        elif sharpe >= self.BENCHMARKS['sharpe_acceptable']:
            score += 15
        elif sharpe > 0:
            score += 10
        
        # Sortino contribution (20 points max)
        if sortino >= self.BENCHMARKS['sortino_excellent']:
            score += 20
        elif sortino >= self.BENCHMARKS['sortino_good']:
            score += 15
        elif sortino > 0:
            score += 10
        
        # Win rate contribution (15 points max)
        if win_rate >= self.BENCHMARKS['win_rate_excellent']:
            score += 15
        elif win_rate >= self.BENCHMARKS['win_rate_good']:
            score += 10
        elif win_rate >= 0.4:
            score += 5
        
        # Profit factor contribution (15 points max)
        if profit_factor >= self.BENCHMARKS['profit_factor_excellent']:
            score += 15
        elif profit_factor >= self.BENCHMARKS['profit_factor_good']:
            score += 10
        elif profit_factor > 1.0:
            score += 5
        
        # Drawdown contribution (15 points max - lower is better)
        if max_dd <= self.BENCHMARKS['max_dd_excellent']:
            score += 15
        elif max_dd <= self.BENCHMARKS['max_dd_good']:
            score += 10
        elif max_dd <= self.BENCHMARKS['max_dd_acceptable']:
            score += 5
        
        # Consistency contribution (10 points max)
        score += consistency * 0.1
        
        return min(100, score)
    
    def determine_grade(self, risk_adjusted_score: float) -> PerformanceGrade:
        """Determine performance grade from score"""
        if risk_adjusted_score >= 90:
            return PerformanceGrade.EXCEPTIONAL
        elif risk_adjusted_score >= 75:
            return PerformanceGrade.EXCELLENT
        elif risk_adjusted_score >= 60:
            return PerformanceGrade.GOOD
        elif risk_adjusted_score >= 45:
            return PerformanceGrade.AVERAGE
        elif risk_adjusted_score >= 30:
            return PerformanceGrade.BELOW_AVERAGE
        else:
            return PerformanceGrade.POOR
    
    def calculate_score(self) -> RiskAdjustedScore:
        """
        Calculate comprehensive risk-adjusted score.
        This is the main method that evaluates overall performance.
        """
        # Calculate all metrics
        sharpe = self.calculate_sharpe_ratio()
        sortino = self.calculate_sortino_ratio()
        max_dd, avg_dd, dd_duration = self.calculate_max_drawdown()
        var_95, cvar_95 = self.calculate_var_cvar()
        consistency = self.calculate_consistency_score()
        turnover, avg_holding = self.calculate_turnover_metrics()
        profit_factor = self.calculate_profit_factor()
        win_rate = self.calculate_win_rate()
        
        # Calculate Calmar ratio
        total_return = sum(self.daily_returns) if self.daily_returns else 0
        calmar = total_return / max_dd if max_dd > 0 else 0
        
        # Recovery factor
        recovery_factor = total_return / max_dd if max_dd > 0 else 0
        
        # Tail ratio (right tail / left tail)
        if self.daily_returns:
            positive_returns = [r for r in self.daily_returns if r > 0]
            negative_returns = [abs(r) for r in self.daily_returns if r < 0]
            avg_win = np.mean(positive_returns) if positive_returns else 0
            avg_loss = np.mean(negative_returns) if negative_returns else 1
            tail_ratio = avg_win / avg_loss if avg_loss > 0 else 1
        else:
            tail_ratio = 1.0
        
        # Calculate raw score
        raw_score = self.calculate_raw_score(
            sharpe, sortino, win_rate, profit_factor, max_dd, consistency
        )
        
        # Calculate penalties
        penalties = self.calculate_penalties(turnover, max_dd, consistency, cvar_95)
        total_penalty = sum(penalties.values())
        
        # Risk-adjusted score
        risk_adjusted_score = max(0, raw_score - total_penalty)
        
        # Determine grade
        grade = self.determine_grade(risk_adjusted_score)
        
        # Create score object
        score = RiskAdjustedScore(
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            max_drawdown=max_dd,
            avg_drawdown=avg_dd,
            drawdown_duration_days=dd_duration,
            recovery_factor=recovery_factor,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win_loss_ratio=tail_ratio,
            consistency_score=consistency,
            var_95=var_95,
            cvar_95=cvar_95,
            tail_ratio=tail_ratio,
            turnover_rate=turnover,
            avg_holding_period=avg_holding,
            raw_score=raw_score,
            risk_adjusted_score=risk_adjusted_score,
            grade=grade,
            overtrading_penalty=penalties.get('overtrading', 0),
            drawdown_penalty=penalties.get('drawdown', 0),
            inconsistency_penalty=penalties.get('inconsistency', 0),
            tail_risk_penalty=penalties.get('tail_risk', 0)
        )
        
        # Store in history
        self.score_history.append(score)
        if len(self.score_history) > 100:
            self.score_history = self.score_history[-100:]
        
        logger.info(f"Risk-adjusted score: {risk_adjusted_score:.1f} (Grade: {grade.value})")
        
        return score
    
    def should_reduce_trading(self) -> Tuple[bool, str]:
        """
        Determine if system should reduce trading activity.
        Returns (should_reduce, reason)
        """
        if not self.score_history:
            return False, "Insufficient history"
        
        latest_score = self.score_history[-1]
        
        # Check for overtrading
        if latest_score.overtrading_penalty > 10:
            return True, f"Overtrading detected ({latest_score.turnover_rate:.1f} trades/day)"
        
        # Check for poor performance
        if latest_score.grade in [PerformanceGrade.POOR, PerformanceGrade.BELOW_AVERAGE]:
            return True, f"Poor performance (Grade: {latest_score.grade.value})"
        
        # Check for high drawdown
        if latest_score.max_drawdown > 0.15:
            return True, f"High drawdown ({latest_score.max_drawdown:.1%})"
        
        # Check for negative Sharpe
        if latest_score.sharpe_ratio < 0:
            return True, f"Negative Sharpe ratio ({latest_score.sharpe_ratio:.2f})"
        
        return False, "Performance acceptable"
    
    def get_improvement_suggestions(self) -> List[str]:
        """Get suggestions for improving performance"""
        if not self.score_history:
            return ["Need more trading history for analysis"]
        
        latest = self.score_history[-1]
        suggestions = []
        
        if latest.overtrading_penalty > 5:
            suggestions.append(f"Reduce trading frequency (currently {latest.turnover_rate:.1f}/day, target <3/day)")
        
        if latest.win_rate < 0.5:
            suggestions.append(f"Improve trade selection (win rate {latest.win_rate:.1%}, target >50%)")
        
        if latest.profit_factor < 1.5:
            suggestions.append(f"Improve risk/reward (profit factor {latest.profit_factor:.2f}, target >1.5)")
        
        if latest.max_drawdown > 0.10:
            suggestions.append(f"Reduce position sizes (max DD {latest.max_drawdown:.1%}, target <10%)")
        
        if latest.consistency_score < 50:
            suggestions.append("Focus on consistent small wins over volatile large wins")
        
        if latest.avg_holding_period < 1:
            suggestions.append(f"Hold positions longer (avg {latest.avg_holding_period:.1f}h, target >1h)")
        
        if latest.sharpe_ratio < 1.0:
            suggestions.append(f"Improve risk-adjusted returns (Sharpe {latest.sharpe_ratio:.2f}, target >1.0)")
        
        return suggestions if suggestions else ["Performance is good - maintain current approach"]
    
    def get_status_report(self) -> str:
        """Generate human-readable status report"""
        if not self.score_history:
            return "No performance data available yet"
        
        latest = self.score_history[-1]
        
        report = f"""
=== RISK-ADJUSTED PERFORMANCE REPORT ===
Grade: {latest.grade.value} (Score: {latest.risk_adjusted_score:.1f}/100)

Core Metrics:
  Sharpe Ratio: {latest.sharpe_ratio:.2f}
  Sortino Ratio: {latest.sortino_ratio:.2f}
  Win Rate: {latest.win_rate:.1%}
  Profit Factor: {latest.profit_factor:.2f}

Risk Metrics:
  Max Drawdown: {latest.max_drawdown:.1%}
  VaR (95%): {latest.var_95:.2%}
  CVaR (95%): {latest.cvar_95:.2%}

Behavior Metrics:
  Trades/Day: {latest.turnover_rate:.1f}
  Avg Holding: {latest.avg_holding_period:.1f} hours
  Consistency: {latest.consistency_score:.0f}/100

Penalties Applied:
  Overtrading: -{latest.overtrading_penalty:.1f}
  Drawdown: -{latest.drawdown_penalty:.1f}
  Inconsistency: -{latest.inconsistency_penalty:.1f}
  Tail Risk: -{latest.tail_risk_penalty:.1f}

Suggestions:
"""
        for suggestion in self.get_improvement_suggestions():
            report += f"  - {suggestion}\n"
        
        return report


# Singleton instance
_scorer_instance: Optional[RiskAdjustedScorer] = None


def get_risk_scorer() -> RiskAdjustedScorer:
    """Get singleton scorer instance"""
    global _scorer_instance
    if _scorer_instance is None:
        _scorer_instance = RiskAdjustedScorer()
    return _scorer_instance


def record_trade_for_scoring(trade_id: str, symbol: str, direction: str,
                             entry_time: datetime, exit_time: datetime,
                             entry_price: float, exit_price: float,
                             position_size: float, pnl: float,
                             strategy: str = "", regime: str = ""):
    """Convenience function to record a trade"""
    scorer = get_risk_scorer()
    
    trade = TradeRecord(
        trade_id=trade_id,
        symbol=symbol,
        direction=direction,
        entry_time=entry_time,
        exit_time=exit_time,
        entry_price=entry_price,
        exit_price=exit_price,
        position_size=position_size,
        pnl=pnl,
        pnl_percent=(pnl / (entry_price * position_size)) if entry_price > 0 else 0,
        strategy=strategy,
        regime=regime
    )
    
    scorer.record_trade(trade)


def get_current_score() -> RiskAdjustedScore:
    """Get current risk-adjusted score"""
    return get_risk_scorer().calculate_score()


def should_reduce_activity() -> Tuple[bool, str]:
    """Check if trading activity should be reduced"""
    return get_risk_scorer().should_reduce_trading()
