"""
Self-Reflection Module - Human-Like Trade Analysis and Learning
Generates natural language insights after each trade and daily summaries.
Stores insights in persistent memory for retrieval and learning.
"""

import json
import logging
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class TradeReflection:
    """Reflection on a single trade with human-like insights"""
    trade_id: str
    symbol: str
    direction: str
    entry_price: float
    exit_price: float
    profit_loss: float
    profit_pips: float
    
    # Context at entry
    entry_reason: str
    regime: str
    confidence: float
    strategy: str
    indicators: Dict[str, float]
    
    # Outcome analysis
    outcome: str  # "win", "loss", "breakeven"
    what_worked: List[str]
    what_failed: List[str]
    
    # Human-like insight
    insight: str
    lesson_learned: str
    adjustment_for_next_time: str
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    duration_minutes: int = 0
    
    def to_natural_language(self) -> str:
        """Generate human-readable reflection"""
        outcome_emoji = "profit" if self.outcome == "win" else "loss" if self.outcome == "loss" else "breakeven"
        
        reflection = f"""
Trade Reflection: {self.symbol} {self.direction}
{'='*50}
Outcome: {outcome_emoji.upper()} ({self.profit_loss:+.2f} USD, {self.profit_pips:+.1f} pips)
Duration: {self.duration_minutes} minutes

Why I Entered:
{self.entry_reason}

What Worked:
{chr(10).join(f'  - {w}' for w in self.what_worked) if self.what_worked else '  - Nothing notable'}

What Didn't Work:
{chr(10).join(f'  - {f}' for f in self.what_failed) if self.what_failed else '  - Nothing notable'}

My Insight:
"{self.insight}"

Lesson Learned:
{self.lesson_learned}

What I'll Do Differently:
{self.adjustment_for_next_time}
{'='*50}
"""
        return reflection


@dataclass
class DailySummary:
    """Daily trading summary with aggregate insights"""
    date: str
    total_trades: int
    wins: int
    losses: int
    breakevens: int
    total_pnl: float
    win_rate: float
    
    # Best and worst
    best_trade: Optional[str]
    worst_trade: Optional[str]
    best_strategy: Optional[str]
    worst_strategy: Optional[str]
    
    # Patterns observed
    patterns_observed: List[str]
    
    # Overall reflection
    daily_insight: str
    mood: str  # "confident", "cautious", "frustrated", "learning"
    tomorrow_plan: str
    
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_natural_language(self) -> str:
        """Generate human-readable daily summary"""
        return f"""
Daily Trading Summary - {self.date}
{'='*60}
Performance: {self.wins}W / {self.losses}L / {self.breakevens}BE ({self.win_rate:.0%} win rate)
P&L: ${self.total_pnl:+.2f}

Best Trade: {self.best_trade or 'None'}
Worst Trade: {self.worst_trade or 'None'}
Best Strategy: {self.best_strategy or 'N/A'}
Worst Strategy: {self.worst_strategy or 'N/A'}

Patterns I Noticed:
{chr(10).join(f'  - {p}' for p in self.patterns_observed) if self.patterns_observed else '  - No clear patterns today'}

My Reflection:
"{self.daily_insight}"

Current Mood: {self.mood.upper()}

Tomorrow's Plan:
{self.tomorrow_plan}
{'='*60}
"""


class InsightMemory:
    """Persistent memory for storing and retrieving insights"""
    
    def __init__(self, storage_path: str = "./data/insights"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.reflections_file = self.storage_path / "trade_reflections.json"
        self.summaries_file = self.storage_path / "daily_summaries.json"
        self.insights_index_file = self.storage_path / "insights_index.json"
        
        self.reflections: List[Dict] = self._load_json(self.reflections_file)
        self.summaries: List[Dict] = self._load_json(self.summaries_file)
        self.insights_index: Dict[str, List[str]] = self._load_json(self.insights_index_file) or {}
        
        logger.info(f"InsightMemory initialized with {len(self.reflections)} reflections, {len(self.summaries)} summaries")
    
    def _load_json(self, path: Path) -> Any:
        """Load JSON file if exists"""
        if path.exists():
            try:
                with open(path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error loading {path}: {e}")
        return []
    
    def _save_json(self, path: Path, data: Any):
        """Save data to JSON file"""
        try:
            with open(path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving {path}: {e}")
    
    def store_reflection(self, reflection: TradeReflection):
        """Store a trade reflection"""
        reflection_dict = asdict(reflection)
        self.reflections.append(reflection_dict)
        
        # Index by various keys for retrieval
        self._index_insight(reflection.symbol, reflection.insight)
        self._index_insight(reflection.strategy, reflection.insight)
        self._index_insight(reflection.regime, reflection.insight)
        self._index_insight(reflection.outcome, reflection.insight)
        
        # Keep only last 500 reflections
        if len(self.reflections) > 500:
            self.reflections = self.reflections[-500:]
        
        self._save_json(self.reflections_file, self.reflections)
        self._save_json(self.insights_index_file, self.insights_index)
        
        logger.info(f"Stored reflection for {reflection.symbol} {reflection.direction}")
    
    def store_summary(self, summary: DailySummary):
        """Store a daily summary"""
        summary_dict = asdict(summary)
        self.summaries.append(summary_dict)
        
        # Keep only last 90 days
        if len(self.summaries) > 90:
            self.summaries = self.summaries[-90:]
        
        self._save_json(self.summaries_file, self.summaries)
        logger.info(f"Stored daily summary for {summary.date}")
    
    def _index_insight(self, key: str, insight: str):
        """Index an insight by key for retrieval"""
        if key not in self.insights_index:
            self.insights_index[key] = []
        self.insights_index[key].append(insight)
        # Keep only last 50 insights per key
        if len(self.insights_index[key]) > 50:
            self.insights_index[key] = self.insights_index[key][-50:]
    
    def get_relevant_insights(self, symbol: str = None, strategy: str = None, 
                              regime: str = None, limit: int = 5) -> List[str]:
        """Retrieve relevant past insights"""
        insights = []
        
        if symbol and symbol in self.insights_index:
            insights.extend(self.insights_index[symbol][-limit:])
        if strategy and strategy in self.insights_index:
            insights.extend(self.insights_index[strategy][-limit:])
        if regime and regime in self.insights_index:
            insights.extend(self.insights_index[regime][-limit:])
        
        # Deduplicate and limit
        seen = set()
        unique_insights = []
        for insight in insights:
            if insight not in seen:
                seen.add(insight)
                unique_insights.append(insight)
        
        return unique_insights[:limit]
    
    def get_recent_performance(self, days: int = 7) -> Dict:
        """Get recent performance metrics"""
        cutoff = datetime.now() - timedelta(days=days)
        recent = [r for r in self.reflections 
                  if datetime.fromisoformat(str(r.get('timestamp', '')).replace('Z', '+00:00')) > cutoff]
        
        if not recent:
            return {'trades': 0, 'win_rate': 0, 'total_pnl': 0}
        
        wins = sum(1 for r in recent if r.get('outcome') == 'win')
        total_pnl = sum(r.get('profit_loss', 0) for r in recent)
        
        return {
            'trades': len(recent),
            'wins': wins,
            'losses': len(recent) - wins,
            'win_rate': wins / len(recent) if recent else 0,
            'total_pnl': total_pnl
        }


class SelfReflectionEngine:
    """
    Engine for generating human-like reflections and insights.
    Analyzes trades and generates natural language explanations.
    """
    
    def __init__(self):
        self.memory = InsightMemory()
        self.pending_trades: Dict[str, Dict] = {}  # Trades waiting for exit
        
        # Insight templates for different scenarios
        self.win_templates = [
            "The {strategy} setup worked well because {reason}. I should look for similar conditions.",
            "My patience paid off - waiting for {indicator} confirmation was the right call.",
            "The {regime} regime favored this trade. I'll be more aggressive in similar conditions.",
            "Entry timing was good - {indicator} aligned perfectly with the {strategy} signal.",
        ]
        
        self.loss_templates = [
            "I entered too early without waiting for {indicator} confirmation. Next time, be patient.",
            "The {regime} regime wasn't ideal for {strategy}. I should have recognized this.",
            "My stop was too tight given the {indicator} volatility. Need wider stops in similar conditions.",
            "I ignored the warning signs from {indicator}. Trust the indicators more.",
            "Market conditions changed mid-trade. I should have exited when {indicator} diverged.",
        ]
        
        self.lesson_templates = [
            "In {regime} regimes, {strategy} works best when {condition}.",
            "When {indicator} shows {value}, it's better to {action}.",
            "The {time_context} session tends to {behavior} - adjust accordingly.",
            "After {streak} consecutive {outcome}s, I should {adjustment}.",
        ]
        
        logger.info("SelfReflectionEngine initialized")
    
    def record_trade_entry(self, trade_id: str, trade_data: Dict):
        """Record a trade entry for later reflection"""
        self.pending_trades[trade_id] = {
            **trade_data,
            'entry_time': datetime.now()
        }
        logger.debug(f"Recorded trade entry: {trade_id}")
    
    def generate_reflection(self, trade_id: str, exit_data: Dict) -> TradeReflection:
        """Generate a reflection when a trade closes"""
        entry_data = self.pending_trades.pop(trade_id, {})
        
        # Calculate metrics
        profit_loss = exit_data.get('profit', 0)
        entry_price = entry_data.get('entry_price', exit_data.get('entry_price', 0))
        exit_price = exit_data.get('exit_price', entry_price)
        symbol = entry_data.get('symbol', exit_data.get('symbol', 'UNKNOWN'))
        direction = entry_data.get('direction', exit_data.get('direction', 'UNKNOWN'))
        
        # Calculate pips
        pip_size = 0.01 if 'JPY' in symbol else 0.0001
        if direction == 'LONG':
            profit_pips = (exit_price - entry_price) / pip_size
        else:
            profit_pips = (entry_price - exit_price) / pip_size
        
        # Determine outcome
        if profit_loss > 0.5:
            outcome = "win"
        elif profit_loss < -0.5:
            outcome = "loss"
        else:
            outcome = "breakeven"
        
        # Calculate duration
        entry_time = entry_data.get('entry_time', datetime.now())
        duration = int((datetime.now() - entry_time).total_seconds() / 60)
        
        # Get context
        strategy = entry_data.get('strategy', 'unknown')
        regime = entry_data.get('regime', 'unknown')
        confidence = entry_data.get('confidence', 0.5)
        indicators = entry_data.get('indicators', {})
        entry_reason = entry_data.get('thesis', entry_data.get('entry_reason', 'No reason recorded'))
        
        # Analyze what worked and what didn't
        what_worked, what_failed = self._analyze_trade(
            outcome, strategy, regime, indicators, profit_pips, duration
        )
        
        # Generate human-like insight
        insight = self._generate_insight(outcome, strategy, regime, indicators, profit_pips)
        lesson = self._generate_lesson(outcome, strategy, regime, indicators)
        adjustment = self._generate_adjustment(outcome, strategy, regime, indicators)
        
        reflection = TradeReflection(
            trade_id=trade_id,
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            exit_price=exit_price,
            profit_loss=profit_loss,
            profit_pips=profit_pips,
            entry_reason=entry_reason,
            regime=regime,
            confidence=confidence,
            strategy=strategy,
            indicators=indicators,
            outcome=outcome,
            what_worked=what_worked,
            what_failed=what_failed,
            insight=insight,
            lesson_learned=lesson,
            adjustment_for_next_time=adjustment,
            duration_minutes=duration
        )
        
        # Store in memory
        self.memory.store_reflection(reflection)
        
        # Log the reflection
        logger.info(reflection.to_natural_language())
        
        return reflection
    
    def _analyze_trade(self, outcome: str, strategy: str, regime: str,
                       indicators: Dict, profit_pips: float, duration: int) -> tuple:
        """Analyze what worked and what didn't in a trade"""
        what_worked = []
        what_failed = []
        
        rsi = indicators.get('rsi', 50)
        adx = indicators.get('adx', 25)
        atr = indicators.get('atr', 0)
        
        if outcome == "win":
            # Analyze winning factors
            if adx > 30:
                what_worked.append(f"Strong trend (ADX={adx:.0f}) supported the move")
            if 40 <= rsi <= 60:
                what_worked.append("RSI was in neutral zone, allowing room to move")
            if strategy == 'trend_following' and adx > 25:
                what_worked.append("Trend-following strategy matched trending conditions")
            if strategy == 'mean_reversion' and adx < 25:
                what_worked.append("Mean-reversion worked in ranging market")
            if duration < 60:
                what_worked.append("Quick execution captured the move efficiently")
            if not what_worked:
                what_worked.append("Entry timing was good")
        else:
            # Analyze losing factors
            if adx < 20 and strategy == 'trend_following':
                what_failed.append(f"Trend-following in weak trend (ADX={adx:.0f})")
            if adx > 35 and strategy == 'mean_reversion':
                what_failed.append(f"Mean-reversion against strong trend (ADX={adx:.0f})")
            if rsi > 70 or rsi < 30:
                what_failed.append(f"RSI was at extreme ({rsi:.0f}) - reversal risk")
            if duration > 120:
                what_failed.append("Held too long - should have cut earlier")
            if abs(profit_pips) > 50:
                what_failed.append("Large adverse move - stop was too wide")
            if not what_failed:
                what_failed.append("Market moved against the position unexpectedly")
        
        return what_worked, what_failed
    
    def _generate_insight(self, outcome: str, strategy: str, regime: str,
                          indicators: Dict, profit_pips: float) -> str:
        """Generate a human-like insight about the trade"""
        rsi = indicators.get('rsi', 50)
        adx = indicators.get('adx', 25)
        
        if outcome == "win":
            if adx > 35:
                return f"Strong momentum (ADX {adx:.0f}) made this {strategy} trade easy. I should be more aggressive when I see this setup again."
            elif profit_pips > 30:
                return f"Patience paid off with {profit_pips:.0f} pips. The {regime} regime was perfect for this strategy."
            else:
                return f"Small but clean win. The {strategy} setup was textbook - I should trust these signals more."
        else:
            if adx < 20:
                return f"I forced a trade in choppy conditions (ADX {adx:.0f}). Need to wait for clearer setups."
            elif rsi > 70 or rsi < 30:
                return f"RSI was screaming reversal at {rsi:.0f}. I ignored it and paid the price."
            else:
                return f"The {regime} regime wasn't ideal for {strategy}. I need to match strategy to conditions better."
    
    def _generate_lesson(self, outcome: str, strategy: str, regime: str,
                         indicators: Dict) -> str:
        """Generate a lesson learned from the trade"""
        adx = indicators.get('adx', 25)
        rsi = indicators.get('rsi', 50)
        
        if outcome == "win":
            if adx > 30:
                return f"In trending regimes with ADX > 30, {strategy} has high probability. Be confident."
            else:
                return f"Even in uncertain conditions, good risk management leads to profits."
        else:
            if strategy == 'trend_following' and adx < 25:
                return f"Don't use trend-following when ADX < 25. Wait for clearer trends."
            elif rsi > 70 or rsi < 30:
                return f"Extreme RSI ({rsi:.0f}) often precedes reversals. Use it as a warning, not a signal."
            else:
                return f"The {regime} regime requires different approach. Adapt strategy to conditions."
    
    def _generate_adjustment(self, outcome: str, strategy: str, regime: str,
                             indicators: Dict) -> str:
        """Generate what to do differently next time"""
        adx = indicators.get('adx', 25)
        
        if outcome == "win":
            return f"Keep using {strategy} in {regime} conditions. Consider slightly larger position size when confidence is high."
        else:
            if adx < 20:
                return f"In low-ADX conditions, either skip the trade or use tighter stops with smaller size."
            else:
                return f"Wait for more confirmation before entering {strategy} trades in {regime} regime. Add one more filter."
    
    def generate_daily_summary(self, trades: List[Dict], current_balance: float) -> DailySummary:
        """Generate end-of-day summary with insights"""
        today = datetime.now().strftime('%Y-%m-%d')
        
        if not trades:
            return DailySummary(
                date=today,
                total_trades=0,
                wins=0,
                losses=0,
                breakevens=0,
                total_pnl=0,
                win_rate=0,
                best_trade=None,
                worst_trade=None,
                best_strategy=None,
                worst_strategy=None,
                patterns_observed=["No trades today - market conditions may not have been favorable"],
                daily_insight="Sometimes the best trade is no trade. Patience is a virtue.",
                mood="cautious",
                tomorrow_plan="Look for high-probability setups with strong trend confirmation."
            )
        
        # Calculate metrics
        wins = sum(1 for t in trades if t.get('profit', 0) > 0)
        losses = sum(1 for t in trades if t.get('profit', 0) < 0)
        breakevens = len(trades) - wins - losses
        total_pnl = sum(t.get('profit', 0) for t in trades)
        win_rate = wins / len(trades) if trades else 0
        
        # Find best and worst
        sorted_trades = sorted(trades, key=lambda t: t.get('profit', 0))
        worst_trade = f"{sorted_trades[0].get('symbol')} ({sorted_trades[0].get('profit', 0):+.2f})" if sorted_trades else None
        best_trade = f"{sorted_trades[-1].get('symbol')} ({sorted_trades[-1].get('profit', 0):+.2f})" if sorted_trades else None
        
        # Analyze strategies
        strategy_pnl = {}
        for t in trades:
            strat = t.get('strategy', 'unknown')
            strategy_pnl[strat] = strategy_pnl.get(strat, 0) + t.get('profit', 0)
        
        best_strategy = max(strategy_pnl, key=strategy_pnl.get) if strategy_pnl else None
        worst_strategy = min(strategy_pnl, key=strategy_pnl.get) if strategy_pnl else None
        
        # Identify patterns
        patterns = self._identify_patterns(trades)
        
        # Generate mood and insight
        if win_rate > 0.6 and total_pnl > 0:
            mood = "confident"
            daily_insight = f"Great day with {win_rate:.0%} win rate. The strategies are working well in current conditions."
        elif win_rate < 0.4 or total_pnl < -2:
            mood = "frustrated"
            daily_insight = f"Tough day. Need to review what went wrong and adjust approach."
        elif total_pnl > 0:
            mood = "learning"
            daily_insight = f"Profitable but room for improvement. Win rate of {win_rate:.0%} could be better."
        else:
            mood = "cautious"
            daily_insight = f"Mixed results. Market conditions were challenging today."
        
        # Generate tomorrow's plan
        if mood == "confident":
            tomorrow_plan = f"Continue with current approach. Focus on {best_strategy} setups."
        elif mood == "frustrated":
            tomorrow_plan = f"Reduce position sizes and be more selective. Avoid {worst_strategy} until conditions improve."
        else:
            tomorrow_plan = f"Stay patient and wait for high-probability setups. Review {worst_strategy} performance."
        
        summary = DailySummary(
            date=today,
            total_trades=len(trades),
            wins=wins,
            losses=losses,
            breakevens=breakevens,
            total_pnl=total_pnl,
            win_rate=win_rate,
            best_trade=best_trade,
            worst_trade=worst_trade,
            best_strategy=best_strategy,
            worst_strategy=worst_strategy,
            patterns_observed=patterns,
            daily_insight=daily_insight,
            mood=mood,
            tomorrow_plan=tomorrow_plan
        )
        
        # Store in memory
        self.memory.store_summary(summary)
        
        # Log the summary
        logger.info(summary.to_natural_language())
        
        return summary
    
    def _identify_patterns(self, trades: List[Dict]) -> List[str]:
        """Identify patterns in today's trades"""
        patterns = []
        
        if not trades:
            return patterns
        
        # Time-based patterns
        morning_trades = [t for t in trades if 6 <= datetime.fromisoformat(str(t.get('timestamp', datetime.now()))).hour < 12]
        afternoon_trades = [t for t in trades if 12 <= datetime.fromisoformat(str(t.get('timestamp', datetime.now()))).hour < 18]
        
        if morning_trades:
            morning_pnl = sum(t.get('profit', 0) for t in morning_trades)
            if morning_pnl > 0:
                patterns.append(f"Morning session was profitable (${morning_pnl:.2f})")
            else:
                patterns.append(f"Morning session was challenging (${morning_pnl:.2f})")
        
        # Strategy patterns
        strategies = set(t.get('strategy', 'unknown') for t in trades)
        for strat in strategies:
            strat_trades = [t for t in trades if t.get('strategy') == strat]
            strat_wins = sum(1 for t in strat_trades if t.get('profit', 0) > 0)
            if len(strat_trades) >= 2:
                if strat_wins / len(strat_trades) > 0.7:
                    patterns.append(f"{strat} strategy performed well today ({strat_wins}/{len(strat_trades)} wins)")
                elif strat_wins / len(strat_trades) < 0.3:
                    patterns.append(f"{strat} strategy struggled today ({strat_wins}/{len(strat_trades)} wins)")
        
        # Streak patterns
        consecutive_wins = 0
        consecutive_losses = 0
        max_wins = 0
        max_losses = 0
        
        for t in trades:
            if t.get('profit', 0) > 0:
                consecutive_wins += 1
                consecutive_losses = 0
                max_wins = max(max_wins, consecutive_wins)
            else:
                consecutive_losses += 1
                consecutive_wins = 0
                max_losses = max(max_losses, consecutive_losses)
        
        if max_wins >= 3:
            patterns.append(f"Had a {max_wins}-trade winning streak - momentum was good")
        if max_losses >= 3:
            patterns.append(f"Had a {max_losses}-trade losing streak - should have paused")
        
        return patterns
    
    def get_pre_trade_wisdom(self, symbol: str, strategy: str, regime: str) -> str:
        """Get relevant wisdom before entering a trade"""
        insights = self.memory.get_relevant_insights(symbol, strategy, regime, limit=3)
        
        if not insights:
            return "No specific past insights for this setup. Proceed with standard risk management."
        
        wisdom = "Based on past experience:\n"
        for i, insight in enumerate(insights, 1):
            wisdom += f"  {i}. {insight}\n"
        
        return wisdom


# Global instance
reflection_engine = SelfReflectionEngine()
