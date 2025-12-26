"""
Agentic Learning Module - Daily Self-Improvement System
Implements human-like learning, trade analysis, and continuous evolution
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import json
import os
import pickle
from collections import defaultdict

from config import config
from risk_management import risk_manager, RiskMetrics
from regime_detection import regime_manager, MarketRegime

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

logger = logging.getLogger(__name__)


@dataclass
class TradeAnalysis:
    """Analysis of a single trade"""
    ticket: int
    symbol: str
    direction: str
    entry_price: float
    exit_price: float
    profit: float
    profit_pips: float
    duration_minutes: float
    entry_reason: str
    exit_reason: str
    indicators_at_entry: Dict
    regime_at_entry: str
    confidence_at_entry: float
    lessons_learned: List[str] = field(default_factory=list)
    score: float = 0.0  # -1 to 1, how good was this trade decision


@dataclass
class DailyReport:
    """Daily trading report"""
    date: datetime
    total_trades: int
    winning_trades: int
    losing_trades: int
    total_profit: float
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    max_drawdown: float
    sharpe_ratio: float
    best_trade: Optional[TradeAnalysis]
    worst_trade: Optional[TradeAnalysis]
    regime_performance: Dict[str, Dict]
    strategy_performance: Dict[str, Dict]
    insights: List[str]
    recommendations: List[str]


@dataclass
class TradingInsight:
    """A learned insight from trading"""
    id: str
    created_at: datetime
    category: str  # 'entry', 'exit', 'risk', 'regime', 'indicator'
    condition: str  # e.g., "ADX > 40 AND trend_direction == position_direction"
    observation: str  # e.g., "Strong trends with ADX > 40 have 75% win rate"
    confidence: float  # 0 to 1
    sample_size: int
    success_rate: float
    avg_profit: float
    is_active: bool = True
    times_applied: int = 0
    times_successful: int = 0


class TradeJournal:
    """Trade journal for logging and analysis"""
    
    def __init__(self, journal_path: str = None):
        self.journal_path = journal_path or "./data/trade_journal.json"
        self.trades: List[TradeAnalysis] = []
        self.daily_reports: List[DailyReport] = []
        self._load_journal()
    
    def _load_journal(self):
        """Load journal from disk"""
        if os.path.exists(self.journal_path):
            try:
                with open(self.journal_path, 'r') as f:
                    data = json.load(f)
                    # Reconstruct trades (simplified)
                    self.trades = []
                    for t in data.get('trades', []):
                        self.trades.append(TradeAnalysis(**t))
            except Exception as e:
                logger.warning(f"Could not load journal: {e}")
    
    def _save_journal(self):
        """Save journal to disk"""
        os.makedirs(os.path.dirname(self.journal_path), exist_ok=True)
        try:
            data = {
                'trades': [vars(t) for t in self.trades[-1000:]],  # Keep last 1000
                'last_updated': datetime.now().isoformat()
            }
            with open(self.journal_path, 'w') as f:
                json.dump(data, f, default=str, indent=2)
        except Exception as e:
            logger.error(f"Could not save journal: {e}")
    
    def log_trade(self, trade: TradeAnalysis):
        """Log a completed trade"""
        self.trades.append(trade)
        self._save_journal()
        logger.info(f"Trade logged: {trade.symbol} {trade.direction} P/L: {trade.profit:.2f}")
    
    def get_trades_for_date(self, date: datetime) -> List[TradeAnalysis]:
        """Get all trades for a specific date"""
        return [t for t in self.trades 
                if hasattr(t, 'entry_time') and t.entry_time.date() == date.date()]
    
    def get_recent_trades(self, days: int = 7) -> List[TradeAnalysis]:
        """Get trades from recent days"""
        cutoff = datetime.now() - timedelta(days=days)
        return [t for t in self.trades 
                if hasattr(t, 'entry_time') and t.entry_time > cutoff]


class InsightEngine:
    """Engine for generating and managing trading insights"""
    
    def __init__(self, insights_path: str = None):
        self.insights_path = insights_path or "./data/insights.pkl"
        self.insights: Dict[str, TradingInsight] = {}
        self._load_insights()
    
    def _load_insights(self):
        """Load insights from disk"""
        if os.path.exists(self.insights_path):
            try:
                with open(self.insights_path, 'rb') as f:
                    self.insights = pickle.load(f)
            except Exception as e:
                logger.warning(f"Could not load insights: {e}")
    
    def _save_insights(self):
        """Save insights to disk"""
        os.makedirs(os.path.dirname(self.insights_path), exist_ok=True)
        try:
            with open(self.insights_path, 'wb') as f:
                pickle.dump(self.insights, f)
        except Exception as e:
            logger.error(f"Could not save insights: {e}")
    
    def add_insight(self, insight: TradingInsight):
        """Add or update an insight"""
        self.insights[insight.id] = insight
        self._save_insights()
        logger.info(f"Insight added: {insight.observation}")
    
    def get_active_insights(self, category: str = None) -> List[TradingInsight]:
        """Get active insights, optionally filtered by category"""
        insights = [i for i in self.insights.values() if i.is_active]
        if category:
            insights = [i for i in insights if i.category == category]
        return sorted(insights, key=lambda x: x.confidence, reverse=True)
    
    def update_insight_performance(self, insight_id: str, was_successful: bool):
        """Update insight performance after it was applied"""
        if insight_id in self.insights:
            insight = self.insights[insight_id]
            insight.times_applied += 1
            if was_successful:
                insight.times_successful += 1
            
            # Update confidence based on recent performance
            if insight.times_applied > 10:
                recent_success_rate = insight.times_successful / insight.times_applied
                insight.confidence = 0.7 * insight.confidence + 0.3 * recent_success_rate
            
            # Deactivate if performing poorly
            if insight.times_applied > 20 and insight.confidence < 0.4:
                insight.is_active = False
                logger.info(f"Insight deactivated due to poor performance: {insight.id}")
            
            self._save_insights()
    
    def generate_insights_from_trades(self, trades: List[TradeAnalysis]) -> List[TradingInsight]:
        """Generate new insights from trade analysis"""
        if len(trades) < 10:
            return []
        
        new_insights = []
        
        # Analyze by regime
        regime_stats = defaultdict(lambda: {'wins': 0, 'losses': 0, 'profit': 0})
        for trade in trades:
            regime = trade.regime_at_entry
            if trade.profit > 0:
                regime_stats[regime]['wins'] += 1
            else:
                regime_stats[regime]['losses'] += 1
            regime_stats[regime]['profit'] += trade.profit
        
        for regime, stats in regime_stats.items():
            total = stats['wins'] + stats['losses']
            if total >= 5:
                win_rate = stats['wins'] / total
                if win_rate > 0.6:
                    insight = TradingInsight(
                        id=f"regime_{regime}_{datetime.now().strftime('%Y%m%d')}",
                        created_at=datetime.now(),
                        category='regime',
                        condition=f"regime == '{regime}'",
                        observation=f"Trading in {regime} regime has {win_rate:.1%} win rate",
                        confidence=min(0.9, 0.5 + (win_rate - 0.5) * 2),
                        sample_size=total,
                        success_rate=win_rate,
                        avg_profit=stats['profit'] / total
                    )
                    new_insights.append(insight)
        
        # Analyze by indicator conditions
        indicator_stats = defaultdict(lambda: {'wins': 0, 'losses': 0, 'profit': 0})
        for trade in trades:
            indicators = trade.indicators_at_entry
            
            # ADX analysis
            adx = indicators.get('adx', 25)
            if adx > 40:
                key = 'adx_strong'
            elif adx > 25:
                key = 'adx_moderate'
            else:
                key = 'adx_weak'
            
            if trade.profit > 0:
                indicator_stats[key]['wins'] += 1
            else:
                indicator_stats[key]['losses'] += 1
            indicator_stats[key]['profit'] += trade.profit
        
        for condition, stats in indicator_stats.items():
            total = stats['wins'] + stats['losses']
            if total >= 5:
                win_rate = stats['wins'] / total
                if win_rate > 0.55 or win_rate < 0.35:
                    insight = TradingInsight(
                        id=f"indicator_{condition}_{datetime.now().strftime('%Y%m%d')}",
                        created_at=datetime.now(),
                        category='indicator',
                        condition=condition,
                        observation=f"Condition '{condition}' has {win_rate:.1%} win rate",
                        confidence=min(0.9, abs(win_rate - 0.5) * 2 + 0.5),
                        sample_size=total,
                        success_rate=win_rate,
                        avg_profit=stats['profit'] / total
                    )
                    new_insights.append(insight)
        
        # Add new insights
        for insight in new_insights:
            self.add_insight(insight)
        
        return new_insights


class PerformanceAnalyzer:
    """Analyzes trading performance and generates reports"""
    
    def __init__(self):
        self.metrics_history: List[Dict] = []
    
    def analyze_trade(self, trade_data: Dict, indicators: Dict, 
                      regime: MarketRegime) -> TradeAnalysis:
        """Analyze a single trade"""
        profit = trade_data.get('profit', 0)
        entry_price = trade_data.get('entry_price', 0)
        exit_price = trade_data.get('exit_price', 0)
        
        # Calculate profit in pips
        if 'JPY' in trade_data.get('symbol', ''):
            pip_value = 0.01
        else:
            pip_value = 0.0001
        
        profit_pips = (exit_price - entry_price) / pip_value
        if trade_data.get('direction') == 'sell':
            profit_pips = -profit_pips
        
        # Generate lessons learned
        lessons = []
        
        # Analyze entry quality
        rsi = indicators.get('rsi', 50)
        adx = indicators.get('adx', 25)
        
        if profit > 0:
            if adx > 40:
                lessons.append("Strong trend (ADX > 40) led to profitable trade")
            if rsi < 30 and trade_data.get('direction') == 'buy':
                lessons.append("Oversold RSI entry was successful")
            if rsi > 70 and trade_data.get('direction') == 'sell':
                lessons.append("Overbought RSI entry was successful")
        else:
            if adx < 20:
                lessons.append("Weak trend (ADX < 20) - should have avoided or used mean-reversion")
            if regime.name == 'high_vol':
                lessons.append("High volatility regime - consider smaller position size")
        
        # Score the trade decision
        score = 0.0
        if profit > 0:
            score += 0.5
            if trade_data.get('confidence', 0.5) > 0.7:
                score += 0.2  # High confidence was correct
        else:
            score -= 0.3
            if trade_data.get('confidence', 0.5) > 0.7:
                score -= 0.3  # High confidence was wrong - bigger penalty
        
        return TradeAnalysis(
            ticket=trade_data.get('ticket', 0),
            symbol=trade_data.get('symbol', ''),
            direction=trade_data.get('direction', ''),
            entry_price=entry_price,
            exit_price=exit_price,
            profit=profit,
            profit_pips=profit_pips,
            duration_minutes=trade_data.get('duration_minutes', 0),
            entry_reason=trade_data.get('entry_reason', ''),
            exit_reason=trade_data.get('exit_reason', ''),
            indicators_at_entry=indicators,
            regime_at_entry=regime.name if regime else 'unknown',
            confidence_at_entry=trade_data.get('confidence', 0.5),
            lessons_learned=lessons,
            score=score
        )
    
    def generate_daily_report(self, trades: List[TradeAnalysis], 
                              date: datetime) -> DailyReport:
        """Generate comprehensive daily report"""
        if not trades:
            return self._empty_report(date)
        
        # Basic statistics
        profits = [t.profit for t in trades]
        wins = [t for t in trades if t.profit > 0]
        losses = [t for t in trades if t.profit <= 0]
        
        total_profit = sum(profits)
        win_rate = len(wins) / len(trades) if trades else 0
        avg_win = np.mean([t.profit for t in wins]) if wins else 0
        avg_loss = np.mean([t.profit for t in losses]) if losses else 0
        
        # Profit factor
        gross_profit = sum(t.profit for t in wins)
        gross_loss = abs(sum(t.profit for t in losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Drawdown calculation
        cumulative = np.cumsum(profits)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = running_max - cumulative
        max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0
        
        # Sharpe ratio (simplified daily)
        if len(profits) > 1 and np.std(profits) > 0:
            sharpe_ratio = np.mean(profits) / np.std(profits) * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        # Best and worst trades
        best_trade = max(trades, key=lambda t: t.profit) if trades else None
        worst_trade = min(trades, key=lambda t: t.profit) if trades else None
        
        # Performance by regime
        regime_performance = defaultdict(lambda: {'trades': 0, 'wins': 0, 'profit': 0})
        for trade in trades:
            regime = trade.regime_at_entry
            regime_performance[regime]['trades'] += 1
            if trade.profit > 0:
                regime_performance[regime]['wins'] += 1
            regime_performance[regime]['profit'] += trade.profit
        
        # Performance by strategy/entry reason
        strategy_performance = defaultdict(lambda: {'trades': 0, 'wins': 0, 'profit': 0})
        for trade in trades:
            strategy = trade.entry_reason.split('_')[0] if trade.entry_reason else 'unknown'
            strategy_performance[strategy]['trades'] += 1
            if trade.profit > 0:
                strategy_performance[strategy]['wins'] += 1
            strategy_performance[strategy]['profit'] += trade.profit
        
        # Generate insights
        insights = self._generate_insights(trades, regime_performance, strategy_performance)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            win_rate, profit_factor, max_drawdown, regime_performance, strategy_performance
        )
        
        return DailyReport(
            date=date,
            total_trades=len(trades),
            winning_trades=len(wins),
            losing_trades=len(losses),
            total_profit=total_profit,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            best_trade=best_trade,
            worst_trade=worst_trade,
            regime_performance=dict(regime_performance),
            strategy_performance=dict(strategy_performance),
            insights=insights,
            recommendations=recommendations
        )
    
    def _empty_report(self, date: datetime) -> DailyReport:
        """Generate empty report for days with no trades"""
        return DailyReport(
            date=date,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            total_profit=0,
            win_rate=0,
            avg_win=0,
            avg_loss=0,
            profit_factor=0,
            max_drawdown=0,
            sharpe_ratio=0,
            best_trade=None,
            worst_trade=None,
            regime_performance={},
            strategy_performance={},
            insights=["No trades executed today"],
            recommendations=["Review market conditions for trading opportunities"]
        )
    
    def _generate_insights(self, trades: List[TradeAnalysis],
                           regime_perf: Dict, strategy_perf: Dict) -> List[str]:
        """Generate insights from daily performance"""
        insights = []
        
        # Overall performance insight
        wins = len([t for t in trades if t.profit > 0])
        total = len(trades)
        if total > 0:
            win_rate = wins / total
            if win_rate > 0.7:
                insights.append(f"Excellent day with {win_rate:.0%} win rate")
            elif win_rate < 0.4:
                insights.append(f"Challenging day with {win_rate:.0%} win rate - review setups")
        
        # Regime insights
        for regime, stats in regime_perf.items():
            if stats['trades'] >= 3:
                regime_wr = stats['wins'] / stats['trades']
                if regime_wr > 0.7:
                    insights.append(f"Strong performance in {regime} regime ({regime_wr:.0%} WR)")
                elif regime_wr < 0.3:
                    insights.append(f"Poor performance in {regime} regime - consider avoiding")
        
        # Strategy insights
        for strategy, stats in strategy_perf.items():
            if stats['trades'] >= 3:
                strat_wr = stats['wins'] / stats['trades']
                if strat_wr > 0.7:
                    insights.append(f"{strategy} strategy performing well ({strat_wr:.0%} WR)")
                elif strat_wr < 0.3:
                    insights.append(f"{strategy} strategy underperforming - review parameters")
        
        # Trade quality insights
        high_conf_trades = [t for t in trades if t.confidence_at_entry > 0.7]
        if high_conf_trades:
            high_conf_wr = len([t for t in high_conf_trades if t.profit > 0]) / len(high_conf_trades)
            if high_conf_wr > 0.6:
                insights.append(f"High confidence trades accurate ({high_conf_wr:.0%} WR)")
            else:
                insights.append(f"High confidence trades underperforming - recalibrate model")
        
        return insights
    
    def _generate_recommendations(self, win_rate: float, profit_factor: float,
                                   max_drawdown: float, regime_perf: Dict,
                                   strategy_perf: Dict) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Win rate recommendations
        if win_rate < 0.5:
            recommendations.append("Consider increasing entry threshold confidence to >70%")
            recommendations.append("Review losing trades for common patterns")
        
        # Profit factor recommendations
        if profit_factor < 1.5:
            recommendations.append("Improve risk-reward ratio - target 1:2 minimum")
            recommendations.append("Consider tighter stop losses or wider take profits")
        
        # Drawdown recommendations
        if max_drawdown > 0.05:  # 5% daily drawdown
            recommendations.append("Reduce position sizes to limit drawdown")
            recommendations.append("Consider implementing daily loss limit")
        
        # Regime-specific recommendations
        for regime, stats in regime_perf.items():
            if stats['trades'] >= 3:
                regime_wr = stats['wins'] / stats['trades']
                if regime_wr < 0.4:
                    recommendations.append(f"Reduce trading in {regime} regime or switch strategy")
        
        # Strategy recommendations
        best_strategy = None
        best_wr = 0
        for strategy, stats in strategy_perf.items():
            if stats['trades'] >= 3:
                strat_wr = stats['wins'] / stats['trades']
                if strat_wr > best_wr:
                    best_wr = strat_wr
                    best_strategy = strategy
        
        if best_strategy and best_wr > 0.6:
            recommendations.append(f"Increase allocation to {best_strategy} strategy")
        
        return recommendations


class AgenticLearningSystem:
    """Main agentic learning system that coordinates daily improvement"""
    
    def __init__(self):
        self.journal = TradeJournal()
        self.insight_engine = InsightEngine()
        self.performance_analyzer = PerformanceAnalyzer()
        
        # Learning state
        self.learning_rate = 0.1
        self.exploration_rate = 0.2
        self.confidence_threshold = 0.6
        
        # Adaptive parameters
        self.adaptive_params = {
            'base_risk': 0.01,
            'leverage_multiplier': 1.0,
            'aggression_level': 0.5,  # 0 = conservative, 1 = aggressive
            'regime_weights': {},
            'strategy_weights': {},
            'indicator_thresholds': {
                'rsi_oversold': 30,
                'rsi_overbought': 70,
                'adx_strong_trend': 40,
                'adx_weak_trend': 20
            }
        }
        
        # Performance tracking
        self.daily_performance: List[Dict] = []
        self.weekly_performance: List[Dict] = []
        self.monthly_performance: List[Dict] = []
        
        # Mode
        self.trading_mode = 'normal'  # 'conservative', 'normal', 'aggressive'
        
        self._load_state()
    
    def _load_state(self):
        """Load learning state from disk"""
        state_path = "./data/learning_state.pkl"
        if os.path.exists(state_path):
            try:
                with open(state_path, 'rb') as f:
                    state = pickle.load(f)
                    self.adaptive_params = state.get('adaptive_params', self.adaptive_params)
                    self.trading_mode = state.get('trading_mode', 'normal')
                    self.daily_performance = state.get('daily_performance', [])
            except Exception as e:
                logger.warning(f"Could not load learning state: {e}")
    
    def _save_state(self):
        """Save learning state to disk"""
        state_path = "./data/learning_state.pkl"
        os.makedirs(os.path.dirname(state_path), exist_ok=True)
        try:
            state = {
                'adaptive_params': self.adaptive_params,
                'trading_mode': self.trading_mode,
                'daily_performance': self.daily_performance[-365:],  # Keep 1 year
                'last_updated': datetime.now().isoformat()
            }
            with open(state_path, 'wb') as f:
                pickle.dump(state, f)
        except Exception as e:
            logger.error(f"Could not save learning state: {e}")
    
    def run_daily_learning_cycle(self, trades_today: List[Dict],
                                  account_balance: float,
                                  historical_data: Dict = None) -> DailyReport:
        """Run the daily learning cycle - called at end of each trading day"""
        logger.info("Starting daily learning cycle...")
        
        today = datetime.now()
        
        # 1. Analyze today's trades
        analyzed_trades = []
        for trade_data in trades_today:
            indicators = trade_data.get('indicators', {})
            regime = regime_manager.current_regime or MarketRegime(
                name='unknown', probability=0.5, volatility=0.01,
                trend_strength=0, mean_return=0, characteristics={},
                recommended_strategies=[], risk_adjustment=1.0
            )
            
            analysis = self.performance_analyzer.analyze_trade(trade_data, indicators, regime)
            analyzed_trades.append(analysis)
            self.journal.log_trade(analysis)
        
        # 2. Generate daily report
        daily_report = self.performance_analyzer.generate_daily_report(analyzed_trades, today)
        
        # 3. Generate new insights
        recent_trades = self.journal.get_recent_trades(days=7)
        analyzed_recent = [t for t in self.journal.trades[-100:]]  # Last 100 trades
        new_insights = self.insight_engine.generate_insights_from_trades(analyzed_recent)
        
        logger.info(f"Generated {len(new_insights)} new insights")
        
        # 4. Learn time-based patterns from historical data (human-like pattern recognition)
        if PATTERN_MINER_AVAILABLE and pattern_miner and historical_data:
            try:
                total_patterns = 0
                for symbol, df in historical_data.items():
                    if df is not None and not df.empty:
                        patterns = pattern_miner.analyze_historical_data(df, symbol)
                        total_patterns += len(patterns)
                logger.info(f"Pattern learning complete: discovered {total_patterns} time-based patterns")
                
                pattern_report = pattern_miner.generate_pattern_report()
                logger.info(f"Pattern Report:\n{pattern_report}")
            except Exception as e:
                logger.error(f"Pattern learning error: {e}")
        
        # 5. Get AI-powered daily insights from Bedrock
        ai_insights = []
        if BEDROCK_AVAILABLE and bedrock_ai:
            try:
                daily_stats = {
                    'total_trades': daily_report.total_trades,
                    'win_rate': daily_report.win_rate,
                    'total_profit': daily_report.total_profit,
                    'max_drawdown': daily_report.max_drawdown,
                    'best_trade': daily_report.best_trade.profit if daily_report.best_trade else 0,
                    'worst_trade': daily_report.worst_trade.profit if daily_report.worst_trade else 0
                }
                pattern_descriptions = []
                if PATTERN_MINER_AVAILABLE and pattern_miner:
                    active_patterns = pattern_miner.get_all_active_patterns()
                    pattern_descriptions = [p.description for p in active_patterns[:5]]
                
                ai_insights = bedrock_ai.generate_daily_insights(daily_stats, pattern_descriptions)
                logger.info(f"AI-generated insights: {ai_insights}")
            except Exception as e:
                logger.error(f"AI insights generation error: {e}")
        
        # 6. Update adaptive parameters based on performance
        self._update_adaptive_params(daily_report, analyzed_trades)
        
        # 7. Adjust trading mode based on drawdown
        self._adjust_trading_mode(account_balance)
        
        # 8. Update strategy weights
        self._update_strategy_weights(daily_report)
        
        # 9. Store daily performance
        self.daily_performance.append({
            'date': today.isoformat(),
            'profit': daily_report.total_profit,
            'win_rate': daily_report.win_rate,
            'trades': daily_report.total_trades,
            'mode': self.trading_mode,
            'ai_insights': ai_insights
        })
        
        # 10. Save state
        self._save_state()
        
        logger.info(f"Daily learning cycle complete. Mode: {self.trading_mode}, "
                   f"Insights: {len(self.insight_engine.insights)}, "
                   f"PatternMiner: {PATTERN_MINER_AVAILABLE}, BedrockAI: {BEDROCK_AVAILABLE}")
        
        return daily_report
    
    def _update_adaptive_params(self, report: DailyReport, trades: List[TradeAnalysis]):
        """Update adaptive parameters based on daily performance"""
        if not trades:
            return
        
        # Adjust risk based on recent performance
        if report.win_rate > 0.6 and report.profit_factor > 1.5:
            # Good performance - slightly increase risk
            self.adaptive_params['base_risk'] = min(0.02, 
                self.adaptive_params['base_risk'] * 1.05)
            self.adaptive_params['aggression_level'] = min(0.8,
                self.adaptive_params['aggression_level'] + 0.05)
        elif report.win_rate < 0.4 or report.profit_factor < 1.0:
            # Poor performance - reduce risk
            self.adaptive_params['base_risk'] = max(0.005,
                self.adaptive_params['base_risk'] * 0.9)
            self.adaptive_params['aggression_level'] = max(0.2,
                self.adaptive_params['aggression_level'] - 0.1)
        
        # Update indicator thresholds based on trade analysis
        adx_wins = [t for t in trades if t.profit > 0 and 
                    t.indicators_at_entry.get('adx', 0) > 40]
        adx_losses = [t for t in trades if t.profit <= 0 and 
                     t.indicators_at_entry.get('adx', 0) > 40]
        
        if len(adx_wins) + len(adx_losses) >= 5:
            adx_wr = len(adx_wins) / (len(adx_wins) + len(adx_losses))
            if adx_wr > 0.7:
                # Strong ADX is working well - lower threshold slightly
                self.adaptive_params['indicator_thresholds']['adx_strong_trend'] = max(35,
                    self.adaptive_params['indicator_thresholds']['adx_strong_trend'] - 1)
            elif adx_wr < 0.4:
                # Strong ADX not working - raise threshold
                self.adaptive_params['indicator_thresholds']['adx_strong_trend'] = min(50,
                    self.adaptive_params['indicator_thresholds']['adx_strong_trend'] + 2)
        
        logger.info(f"Updated adaptive params: risk={self.adaptive_params['base_risk']:.3f}, "
                   f"aggression={self.adaptive_params['aggression_level']:.2f}")
    
    def _adjust_trading_mode(self, account_balance: float):
        """Adjust trading mode based on equity curve"""
        if len(self.daily_performance) < 5:
            return
        
        # Calculate recent drawdown
        recent_profits = [d['profit'] for d in self.daily_performance[-20:]]
        cumulative = np.cumsum(recent_profits)
        if len(cumulative) > 0:
            peak = np.maximum.accumulate(cumulative)
            drawdown = (peak[-1] - cumulative[-1]) / (peak[-1] + account_balance) if peak[-1] > 0 else 0
        else:
            drawdown = 0
        
        # Adjust mode
        if drawdown > 0.15:  # 15% drawdown
            self.trading_mode = 'conservative'
            self.adaptive_params['base_risk'] = 0.005
            self.adaptive_params['aggression_level'] = 0.2
            logger.warning(f"Switched to CONSERVATIVE mode due to {drawdown:.1%} drawdown")
        elif drawdown > 0.10:  # 10% drawdown
            if self.trading_mode != 'conservative':
                self.trading_mode = 'conservative'
                self.adaptive_params['base_risk'] = 0.0075
                self.adaptive_params['aggression_level'] = 0.3
                logger.warning(f"Switched to CONSERVATIVE mode due to {drawdown:.1%} drawdown")
        elif drawdown < 0.05:  # Less than 5% drawdown
            # Check if we've been profitable recently
            recent_profit = sum(recent_profits[-5:])
            if recent_profit > 0 and self.trading_mode == 'conservative':
                self.trading_mode = 'normal'
                self.adaptive_params['base_risk'] = 0.01
                self.adaptive_params['aggression_level'] = 0.5
                logger.info("Switched back to NORMAL mode")
    
    def _update_strategy_weights(self, report: DailyReport):
        """Update strategy weights based on performance"""
        for strategy, stats in report.strategy_performance.items():
            if stats['trades'] >= 3:
                win_rate = stats['wins'] / stats['trades']
                
                # Update weight using exponential moving average
                current_weight = self.adaptive_params['strategy_weights'].get(strategy, 0.5)
                new_weight = 0.8 * current_weight + 0.2 * win_rate
                self.adaptive_params['strategy_weights'][strategy] = new_weight
        
        # Normalize weights
        total_weight = sum(self.adaptive_params['strategy_weights'].values())
        if total_weight > 0:
            for strategy in self.adaptive_params['strategy_weights']:
                self.adaptive_params['strategy_weights'][strategy] /= total_weight
    
    def get_trading_parameters(self) -> Dict:
        """Get current trading parameters for the trading engine"""
        return {
            'base_risk': self.adaptive_params['base_risk'],
            'leverage_multiplier': self.adaptive_params['leverage_multiplier'],
            'aggression_level': self.adaptive_params['aggression_level'],
            'confidence_threshold': self.confidence_threshold,
            'trading_mode': self.trading_mode,
            'indicator_thresholds': self.adaptive_params['indicator_thresholds'],
            'strategy_weights': self.adaptive_params['strategy_weights'],
            'active_insights': self.insight_engine.get_active_insights()
        }
    
    def should_take_trade(self, signal: Dict) -> Tuple[bool, str, float]:
        """Determine if a trade should be taken based on learned parameters"""
        confidence = signal.get('confidence', 0.5)
        strategy = signal.get('strategy', 'unknown')
        regime = signal.get('regime', 'unknown')
        
        # Check confidence threshold
        if confidence < self.confidence_threshold:
            return False, "Confidence below threshold", confidence
        
        # Check trading mode restrictions
        if self.trading_mode == 'conservative':
            if confidence < 0.75:
                return False, "Conservative mode requires high confidence", confidence
            if strategy not in ['trend_following', 'mean_reversion']:
                return False, "Conservative mode restricts to proven strategies", confidence
        
        # Check strategy weight
        strategy_weight = self.adaptive_params['strategy_weights'].get(strategy, 0.5)
        if strategy_weight < 0.3:
            return False, f"Strategy {strategy} has low weight ({strategy_weight:.2f})", confidence
        
        # Check regime compatibility
        regime_weight = self.adaptive_params['regime_weights'].get(regime, 0.5)
        if regime_weight < 0.3:
            return False, f"Regime {regime} has low weight ({regime_weight:.2f})", confidence
        
        # Apply insights
        for insight in self.insight_engine.get_active_insights():
            if insight.category == 'regime' and regime in insight.condition:
                if insight.success_rate < 0.4:
                    return False, f"Insight suggests avoiding {regime} regime", confidence
        
        # Adjust confidence based on factors
        adjusted_confidence = confidence * strategy_weight * (0.5 + regime_weight)
        
        return True, "Trade approved", adjusted_confidence
    
    def calculate_position_size(self, signal: Dict, account_balance: float) -> float:
        """Calculate position size based on learned parameters"""
        base_risk = self.adaptive_params['base_risk']
        aggression = self.adaptive_params['aggression_level']
        confidence = signal.get('confidence', 0.5)
        
        # Adjust risk based on confidence
        if confidence > 0.8:
            risk_multiplier = 1.0 + (aggression * 0.5)
        elif confidence > 0.6:
            risk_multiplier = 1.0
        else:
            risk_multiplier = 0.5
        
        # Apply trading mode
        if self.trading_mode == 'conservative':
            risk_multiplier *= 0.5
        elif self.trading_mode == 'aggressive':
            risk_multiplier *= 1.5
        
        final_risk = base_risk * risk_multiplier
        final_risk = max(0.005, min(0.02, final_risk))  # Cap between 0.5% and 2%
        
        return account_balance * final_risk
    
    def record_trade_outcome(self, trade_id: str, was_profitable: bool,
                             insights_applied: List[str]):
        """Record trade outcome for learning"""
        # Update insight performance
        for insight_id in insights_applied:
            self.insight_engine.update_insight_performance(insight_id, was_profitable)
        
        logger.debug(f"Trade {trade_id} outcome recorded: {'Win' if was_profitable else 'Loss'}")


# Singleton instance
agentic_system = AgenticLearningSystem()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Agentic Learning System...")
    
    # Create sample trades
    sample_trades = []
    for i in range(20):
        profit = np.random.normal(5, 20)
        sample_trades.append({
            'ticket': i,
            'symbol': 'EURUSD',
            'direction': 'buy' if np.random.random() > 0.5 else 'sell',
            'entry_price': 1.1000 + np.random.normal(0, 0.01),
            'exit_price': 1.1000 + np.random.normal(0, 0.01),
            'profit': profit,
            'duration_minutes': np.random.randint(5, 120),
            'entry_reason': np.random.choice(['trend_following', 'mean_reversion', 'breakout']),
            'exit_reason': 'tp_hit' if profit > 0 else 'sl_hit',
            'confidence': np.random.uniform(0.5, 0.9),
            'indicators': {
                'rsi': np.random.uniform(20, 80),
                'adx': np.random.uniform(15, 50),
                'atr': np.random.uniform(0.0005, 0.002)
            }
        })
    
    # Run daily learning cycle
    system = AgenticLearningSystem()
    report = system.run_daily_learning_cycle(sample_trades, account_balance=100)
    
    print(f"\nDaily Report:")
    print(f"  Total trades: {report.total_trades}")
    print(f"  Win rate: {report.win_rate:.1%}")
    print(f"  Total profit: ${report.total_profit:.2f}")
    print(f"  Profit factor: {report.profit_factor:.2f}")
    print(f"\nInsights:")
    for insight in report.insights[:5]:
        print(f"  - {insight}")
    print(f"\nRecommendations:")
    for rec in report.recommendations[:5]:
        print(f"  - {rec}")
    
    # Test trade decision
    test_signal = {
        'confidence': 0.75,
        'strategy': 'trend_following',
        'regime': 'trending'
    }
    should_trade, reason, adj_conf = system.should_take_trade(test_signal)
    print(f"\nTrade decision: {should_trade} - {reason} (conf: {adj_conf:.2f})")
    
    # Get trading parameters
    params = system.get_trading_parameters()
    print(f"\nCurrent trading parameters:")
    print(f"  Mode: {params['trading_mode']}")
    print(f"  Base risk: {params['base_risk']:.3f}")
    print(f"  Aggression: {params['aggression_level']:.2f}")
    
    print("\nAgentic Learning System test complete!")
