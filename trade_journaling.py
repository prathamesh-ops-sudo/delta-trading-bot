"""
Trade Journaling and Uncertainty Governor System
Priority 1: Human-like Trading Decision Making

Trade Journaling:
- Pre-trade hypothesis documentation
- Entry reasoning with supporting/opposing factors
- Post-trade critique and lessons learned
- Pattern recognition from journal entries

Uncertainty Governor:
- Quantifies uncertainty from multiple sources
- Adjusts position sizing based on uncertainty
- Manages trade frequency based on confidence
- Implements "I don't know" as a valid trading decision
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


class UncertaintyLevel(Enum):
    """Uncertainty levels for trading decisions"""
    VERY_LOW = 1      # High confidence, full position
    LOW = 2           # Good confidence, normal position
    MODERATE = 3      # Some uncertainty, reduced position
    HIGH = 4          # Significant uncertainty, minimal position
    VERY_HIGH = 5     # Too uncertain, no trade


class TradeOutcomeType(Enum):
    """Types of trade outcomes"""
    WIN = "win"
    LOSS = "loss"
    BREAKEVEN = "breakeven"
    STOPPED_OUT = "stopped_out"
    TAKE_PROFIT = "take_profit"
    MANUAL_CLOSE = "manual_close"


@dataclass
class TradeHypothesis:
    """Pre-trade hypothesis - what we expect to happen and why"""
    timestamp: datetime
    symbol: str
    direction: str  # BUY or SELL
    
    # Core hypothesis
    primary_thesis: str  # Main reason for the trade
    expected_move: float  # Expected price movement in pips
    expected_duration: str  # How long we expect the trade to last
    
    # Supporting factors
    supporting_factors: List[str] = field(default_factory=list)
    opposing_factors: List[str] = field(default_factory=list)
    
    # Uncertainty assessment
    uncertainty_sources: Dict[str, float] = field(default_factory=dict)  # Source -> uncertainty score (0-1)
    overall_uncertainty: float = 0.5
    
    # Market context
    regime: str = "unknown"
    session: str = "unknown"
    key_levels: Dict[str, float] = field(default_factory=dict)  # support, resistance, etc.
    
    # Invalidation criteria
    invalidation_price: float = 0.0
    invalidation_reason: str = ""
    
    def to_dict(self) -> Dict:
        d = asdict(self)
        d['timestamp'] = self.timestamp.isoformat()
        return d
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'TradeHypothesis':
        d['timestamp'] = datetime.fromisoformat(d['timestamp'])
        return cls(**d)


@dataclass
class TradeCritique:
    """Post-trade critique - what actually happened and lessons learned"""
    timestamp: datetime
    symbol: str
    direction: str
    
    # Trade outcome
    outcome: TradeOutcomeType
    pnl: float
    pnl_pips: float
    duration_minutes: int
    
    # Analysis
    hypothesis_correct: bool  # Was the primary thesis correct?
    entry_quality: float  # 0-1 score for entry timing
    exit_quality: float  # 0-1 score for exit timing
    
    # What happened
    actual_move: float  # Actual price movement
    max_favorable: float  # Maximum favorable excursion (MFE)
    max_adverse: float  # Maximum adverse excursion (MAE)
    
    # Lessons learned
    what_worked: List[str] = field(default_factory=list)
    what_failed: List[str] = field(default_factory=list)
    lessons: List[str] = field(default_factory=list)
    
    # Would we take this trade again?
    would_repeat: bool = True
    repeat_reasoning: str = ""
    
    # Emotional state (self-awareness)
    emotional_state: str = "neutral"  # calm, anxious, overconfident, fearful
    emotional_impact: str = ""  # How emotions affected the trade
    
    def to_dict(self) -> Dict:
        d = asdict(self)
        d['timestamp'] = self.timestamp.isoformat()
        d['outcome'] = self.outcome.value
        return d
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'TradeCritique':
        d['timestamp'] = datetime.fromisoformat(d['timestamp'])
        d['outcome'] = TradeOutcomeType(d['outcome'])
        return cls(**d)


@dataclass
class JournalEntry:
    """Complete journal entry for a trade"""
    trade_id: str
    hypothesis: TradeHypothesis
    critique: Optional[TradeCritique] = None
    
    # Trade details
    entry_price: float = 0.0
    exit_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    position_size: float = 0.0
    
    # Status
    is_open: bool = True
    
    def to_dict(self) -> Dict:
        return {
            'trade_id': self.trade_id,
            'hypothesis': self.hypothesis.to_dict(),
            'critique': self.critique.to_dict() if self.critique else None,
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'position_size': self.position_size,
            'is_open': self.is_open
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'JournalEntry':
        return cls(
            trade_id=d['trade_id'],
            hypothesis=TradeHypothesis.from_dict(d['hypothesis']),
            critique=TradeCritique.from_dict(d['critique']) if d.get('critique') else None,
            entry_price=d.get('entry_price', 0.0),
            exit_price=d.get('exit_price', 0.0),
            stop_loss=d.get('stop_loss', 0.0),
            take_profit=d.get('take_profit', 0.0),
            position_size=d.get('position_size', 0.0),
            is_open=d.get('is_open', True)
        )


class UncertaintyGovernor:
    """
    Manages trading decisions based on uncertainty levels.
    Implements "I don't know" as a valid decision.
    """
    
    # Uncertainty thresholds
    THRESHOLDS = {
        UncertaintyLevel.VERY_LOW: 0.2,
        UncertaintyLevel.LOW: 0.35,
        UncertaintyLevel.MODERATE: 0.5,
        UncertaintyLevel.HIGH: 0.7,
        UncertaintyLevel.VERY_HIGH: 1.0
    }
    
    # Position size multipliers based on uncertainty
    POSITION_MULTIPLIERS = {
        UncertaintyLevel.VERY_LOW: 1.2,    # Can increase position slightly
        UncertaintyLevel.LOW: 1.0,          # Normal position
        UncertaintyLevel.MODERATE: 0.7,     # Reduced position
        UncertaintyLevel.HIGH: 0.4,         # Minimal position
        UncertaintyLevel.VERY_HIGH: 0.0     # No trade
    }
    
    def __init__(self):
        self.uncertainty_history: List[Tuple[datetime, float]] = []
        self.decision_history: List[Dict] = []
        
        # Uncertainty sources and their weights
        self.source_weights = {
            'technical': 0.25,      # Technical indicator disagreement
            'fundamental': 0.20,    # Economic data uncertainty
            'sentiment': 0.15,      # News/sentiment uncertainty
            'regime': 0.15,         # Market regime uncertainty
            'correlation': 0.10,    # Cross-asset correlation breakdown
            'volatility': 0.10,     # Volatility regime uncertainty
            'model': 0.05           # Model confidence uncertainty
        }
        
        # Recent performance affects uncertainty
        self.recent_trades: List[Dict] = []
        self.losing_streak = 0
        self.winning_streak = 0
        
        logger.info("UncertaintyGovernor initialized")
    
    def calculate_uncertainty(self, factors: Dict[str, float]) -> Tuple[float, UncertaintyLevel, Dict]:
        """
        Calculate overall uncertainty from multiple sources.
        
        Args:
            factors: Dict of uncertainty source -> uncertainty value (0-1)
        
        Returns:
            Tuple of (overall_uncertainty, uncertainty_level, breakdown)
        """
        breakdown = {}
        weighted_sum = 0.0
        total_weight = 0.0
        
        for source, weight in self.source_weights.items():
            if source in factors:
                value = factors[source]
                weighted_sum += value * weight
                total_weight += weight
                breakdown[source] = {
                    'value': value,
                    'weight': weight,
                    'contribution': value * weight
                }
        
        # Calculate base uncertainty
        if total_weight > 0:
            base_uncertainty = weighted_sum / total_weight
        else:
            base_uncertainty = 0.5  # Default to moderate uncertainty
        
        # Adjust for recent performance
        performance_adjustment = self._get_performance_adjustment()
        adjusted_uncertainty = base_uncertainty * performance_adjustment
        
        # Cap at 1.0
        overall_uncertainty = min(1.0, max(0.0, adjusted_uncertainty))
        
        # Determine level
        level = self._get_uncertainty_level(overall_uncertainty)
        
        # Record history
        self.uncertainty_history.append((datetime.now(), overall_uncertainty))
        if len(self.uncertainty_history) > 1000:
            self.uncertainty_history = self.uncertainty_history[-500:]
        
        return overall_uncertainty, level, breakdown
    
    def _get_performance_adjustment(self) -> float:
        """Adjust uncertainty based on recent trading performance"""
        adjustment = 1.0
        
        # Increase uncertainty after losing streak
        if self.losing_streak >= 3:
            adjustment *= 1.0 + (self.losing_streak - 2) * 0.1  # +10% per loss after 3
            adjustment = min(adjustment, 1.5)  # Cap at 50% increase
        
        # Slightly decrease uncertainty after winning streak (but be careful of overconfidence)
        if self.winning_streak >= 3:
            adjustment *= 0.95  # Only 5% decrease to avoid overconfidence
        
        return adjustment
    
    def _get_uncertainty_level(self, uncertainty: float) -> UncertaintyLevel:
        """Convert uncertainty score to level"""
        for level, threshold in sorted(self.THRESHOLDS.items(), key=lambda x: x[1]):
            if uncertainty <= threshold:
                return level
        return UncertaintyLevel.VERY_HIGH
    
    def get_position_multiplier(self, uncertainty_level: UncertaintyLevel) -> float:
        """Get position size multiplier based on uncertainty"""
        return self.POSITION_MULTIPLIERS.get(uncertainty_level, 0.5)
    
    def should_trade(self, uncertainty: float, min_confidence: float = 0.65) -> Tuple[bool, str]:
        """
        Determine if we should trade given the uncertainty level.
        Implements "I don't know" as a valid decision.
        
        Returns:
            Tuple of (should_trade, reason)
        """
        level = self._get_uncertainty_level(uncertainty)
        
        if level == UncertaintyLevel.VERY_HIGH:
            return False, "Uncertainty too high - 'I don't know' is the right answer here"
        
        if level == UncertaintyLevel.HIGH:
            return False, "High uncertainty - better to wait for clearer setup"
        
        # Check if confidence meets minimum threshold
        confidence = 1.0 - uncertainty
        if confidence < min_confidence:
            return False, f"Confidence {confidence:.1%} below minimum {min_confidence:.1%}"
        
        return True, f"Uncertainty acceptable ({level.name}), proceeding with trade"
    
    def record_trade_outcome(self, won: bool, pnl: float):
        """Record trade outcome to adjust future uncertainty"""
        self.recent_trades.append({
            'timestamp': datetime.now(),
            'won': won,
            'pnl': pnl
        })
        
        # Keep only recent trades
        if len(self.recent_trades) > 50:
            self.recent_trades = self.recent_trades[-50:]
        
        # Update streaks
        if won:
            self.winning_streak += 1
            self.losing_streak = 0
        else:
            self.losing_streak += 1
            self.winning_streak = 0
        
        logger.info(f"Trade outcome recorded: {'Win' if won else 'Loss'}, "
                   f"Winning streak: {self.winning_streak}, Losing streak: {self.losing_streak}")
    
    def get_trading_recommendation(self, uncertainty: float) -> Dict[str, Any]:
        """Get comprehensive trading recommendation based on uncertainty"""
        level = self._get_uncertainty_level(uncertainty)
        should_trade, reason = self.should_trade(uncertainty)
        multiplier = self.get_position_multiplier(level)
        
        return {
            'uncertainty': uncertainty,
            'level': level.name,
            'should_trade': should_trade,
            'reason': reason,
            'position_multiplier': multiplier,
            'losing_streak': self.losing_streak,
            'winning_streak': self.winning_streak,
            'recommendation': self._get_recommendation_text(level, should_trade)
        }
    
    def _get_recommendation_text(self, level: UncertaintyLevel, should_trade: bool) -> str:
        """Generate human-readable recommendation"""
        if level == UncertaintyLevel.VERY_HIGH:
            return "STAND ASIDE - Too much uncertainty. Wait for clarity."
        elif level == UncertaintyLevel.HIGH:
            return "CAUTION - High uncertainty. Consider waiting or paper trading."
        elif level == UncertaintyLevel.MODERATE:
            return "PROCEED WITH CAUTION - Reduce position size, tighten stops."
        elif level == UncertaintyLevel.LOW:
            return "NORMAL TRADING - Good setup, standard position size."
        else:
            return "HIGH CONVICTION - Strong setup, can increase position slightly."


class TradeJournal:
    """
    Maintains a trading journal with pre-trade hypotheses and post-trade critiques.
    Enables learning from past trades like a human trader.
    """
    
    def __init__(self, data_dir: str = "data/journal"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        self.entries: Dict[str, JournalEntry] = {}
        self.open_trades: Dict[str, JournalEntry] = {}
        
        # Load existing journal
        self._load_journal()
        
        # Statistics
        self.total_entries = len(self.entries)
        self.hypothesis_accuracy = 0.0
        
        logger.info(f"TradeJournal initialized with {self.total_entries} entries")
    
    def _load_journal(self):
        """Load journal entries from disk"""
        journal_file = os.path.join(self.data_dir, "journal.json")
        if os.path.exists(journal_file):
            try:
                with open(journal_file, 'r') as f:
                    data = json.load(f)
                    for entry_data in data:
                        entry = JournalEntry.from_dict(entry_data)
                        self.entries[entry.trade_id] = entry
                        if entry.is_open:
                            self.open_trades[entry.trade_id] = entry
            except Exception as e:
                logger.warning(f"Failed to load journal: {e}")
    
    def _save_journal(self):
        """Save journal entries to disk"""
        journal_file = os.path.join(self.data_dir, "journal.json")
        try:
            with open(journal_file, 'w') as f:
                json.dump([e.to_dict() for e in self.entries.values()], f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save journal: {e}")
    
    def create_hypothesis(
        self,
        trade_id: str,
        symbol: str,
        direction: str,
        primary_thesis: str,
        expected_move: float,
        expected_duration: str,
        supporting_factors: List[str],
        opposing_factors: List[str],
        uncertainty_sources: Dict[str, float],
        regime: str = "unknown",
        session: str = "unknown",
        key_levels: Dict[str, float] = None,
        invalidation_price: float = 0.0,
        invalidation_reason: str = ""
    ) -> TradeHypothesis:
        """Create a pre-trade hypothesis"""
        
        # Calculate overall uncertainty
        if uncertainty_sources:
            overall_uncertainty = statistics.mean(uncertainty_sources.values())
        else:
            overall_uncertainty = 0.5
        
        hypothesis = TradeHypothesis(
            timestamp=datetime.now(),
            symbol=symbol,
            direction=direction,
            primary_thesis=primary_thesis,
            expected_move=expected_move,
            expected_duration=expected_duration,
            supporting_factors=supporting_factors,
            opposing_factors=opposing_factors,
            uncertainty_sources=uncertainty_sources,
            overall_uncertainty=overall_uncertainty,
            regime=regime,
            session=session,
            key_levels=key_levels or {},
            invalidation_price=invalidation_price,
            invalidation_reason=invalidation_reason
        )
        
        logger.info(f"Hypothesis created for {symbol} {direction}: {primary_thesis[:50]}...")
        return hypothesis
    
    def open_trade(
        self,
        trade_id: str,
        hypothesis: TradeHypothesis,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        position_size: float
    ) -> JournalEntry:
        """Record a new trade with its hypothesis"""
        
        entry = JournalEntry(
            trade_id=trade_id,
            hypothesis=hypothesis,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=position_size,
            is_open=True
        )
        
        self.entries[trade_id] = entry
        self.open_trades[trade_id] = entry
        self._save_journal()
        
        logger.info(f"Trade opened and journaled: {trade_id} {hypothesis.symbol} {hypothesis.direction}")
        return entry
    
    def close_trade(
        self,
        trade_id: str,
        exit_price: float,
        outcome: TradeOutcomeType,
        pnl: float,
        pnl_pips: float,
        max_favorable: float = 0.0,
        max_adverse: float = 0.0
    ) -> Optional[TradeCritique]:
        """Close a trade and generate critique"""
        
        if trade_id not in self.entries:
            logger.warning(f"Trade {trade_id} not found in journal")
            return None
        
        entry = self.entries[trade_id]
        hypothesis = entry.hypothesis
        
        # Calculate duration
        duration_minutes = int((datetime.now() - hypothesis.timestamp).total_seconds() / 60)
        
        # Determine if hypothesis was correct
        actual_move = exit_price - entry.entry_price
        if hypothesis.direction == "SELL":
            actual_move = -actual_move
        
        hypothesis_correct = (
            (hypothesis.direction == "BUY" and actual_move > 0) or
            (hypothesis.direction == "SELL" and actual_move < 0)
        )
        
        # Calculate entry/exit quality
        entry_quality = self._calculate_entry_quality(entry, max_favorable, max_adverse)
        exit_quality = self._calculate_exit_quality(entry, exit_price, max_favorable, outcome)
        
        # Generate lessons
        what_worked, what_failed, lessons = self._generate_lessons(
            hypothesis, outcome, pnl, actual_move, hypothesis_correct
        )
        
        # Determine if we would repeat this trade
        would_repeat = hypothesis_correct or (not hypothesis_correct and pnl > -entry.position_size * 0.02)
        repeat_reasoning = self._generate_repeat_reasoning(hypothesis_correct, outcome, lessons)
        
        critique = TradeCritique(
            timestamp=datetime.now(),
            symbol=hypothesis.symbol,
            direction=hypothesis.direction,
            outcome=outcome,
            pnl=pnl,
            pnl_pips=pnl_pips,
            duration_minutes=duration_minutes,
            hypothesis_correct=hypothesis_correct,
            entry_quality=entry_quality,
            exit_quality=exit_quality,
            actual_move=actual_move,
            max_favorable=max_favorable,
            max_adverse=max_adverse,
            what_worked=what_worked,
            what_failed=what_failed,
            lessons=lessons,
            would_repeat=would_repeat,
            repeat_reasoning=repeat_reasoning
        )
        
        # Update entry
        entry.critique = critique
        entry.exit_price = exit_price
        entry.is_open = False
        
        # Remove from open trades
        if trade_id in self.open_trades:
            del self.open_trades[trade_id]
        
        self._save_journal()
        
        logger.info(f"Trade closed and critiqued: {trade_id} {outcome.value} PnL={pnl:.2f}")
        return critique
    
    def _calculate_entry_quality(self, entry: JournalEntry, mfe: float, mae: float) -> float:
        """Calculate entry quality score (0-1)"""
        if mfe == 0 and mae == 0:
            return 0.5  # No data
        
        # Good entry = low MAE relative to MFE
        if mfe > 0:
            quality = 1.0 - (mae / (mfe + mae))
        else:
            quality = 0.3  # Trade never went in our favor
        
        return max(0.0, min(1.0, quality))
    
    def _calculate_exit_quality(self, entry: JournalEntry, exit_price: float, mfe: float, outcome: TradeOutcomeType) -> float:
        """Calculate exit quality score (0-1)"""
        if outcome == TradeOutcomeType.TAKE_PROFIT:
            return 0.9  # Hit target
        elif outcome == TradeOutcomeType.STOPPED_OUT:
            return 0.4  # Stopped out is not great but acceptable
        
        # For manual closes, compare to MFE
        if mfe > 0:
            actual_capture = abs(exit_price - entry.entry_price)
            quality = actual_capture / mfe
            return max(0.0, min(1.0, quality))
        
        return 0.5
    
    def _generate_lessons(
        self,
        hypothesis: TradeHypothesis,
        outcome: TradeOutcomeType,
        pnl: float,
        actual_move: float,
        hypothesis_correct: bool
    ) -> Tuple[List[str], List[str], List[str]]:
        """Generate lessons learned from the trade"""
        what_worked = []
        what_failed = []
        lessons = []
        
        if hypothesis_correct:
            what_worked.append(f"Primary thesis was correct: {hypothesis.primary_thesis[:50]}")
            for factor in hypothesis.supporting_factors[:2]:
                what_worked.append(f"Supporting factor validated: {factor[:40]}")
        else:
            what_failed.append(f"Primary thesis was wrong: {hypothesis.primary_thesis[:50]}")
            for factor in hypothesis.opposing_factors[:2]:
                what_failed.append(f"Opposing factor materialized: {factor[:40]}")
        
        # Lessons based on outcome
        if outcome == TradeOutcomeType.WIN:
            lessons.append("Trade management was effective")
            if hypothesis.overall_uncertainty < 0.4:
                lessons.append("Low uncertainty correlated with winning trade")
        elif outcome == TradeOutcomeType.LOSS:
            if hypothesis.overall_uncertainty > 0.5:
                lessons.append("High uncertainty should have been a warning sign")
            if len(hypothesis.opposing_factors) > len(hypothesis.supporting_factors):
                lessons.append("More opposing than supporting factors - should have passed")
        elif outcome == TradeOutcomeType.STOPPED_OUT:
            lessons.append("Review stop loss placement - was it too tight?")
        
        return what_worked, what_failed, lessons
    
    def _generate_repeat_reasoning(self, hypothesis_correct: bool, outcome: TradeOutcomeType, lessons: List[str]) -> str:
        """Generate reasoning for whether to repeat this trade"""
        if hypothesis_correct and outcome in [TradeOutcomeType.WIN, TradeOutcomeType.TAKE_PROFIT]:
            return "Good setup, good execution, good outcome - repeat this pattern"
        elif hypothesis_correct and outcome == TradeOutcomeType.LOSS:
            return "Thesis was right but execution/timing was off - refine entry/exit"
        elif not hypothesis_correct and outcome == TradeOutcomeType.WIN:
            return "Lucky win - thesis was wrong, don't rely on this pattern"
        else:
            return "Review and improve - thesis and outcome both need work"
    
    def get_recent_lessons(self, count: int = 10) -> List[str]:
        """Get lessons from recent trades"""
        lessons = []
        sorted_entries = sorted(
            [e for e in self.entries.values() if e.critique],
            key=lambda x: x.critique.timestamp,
            reverse=True
        )
        
        for entry in sorted_entries[:count]:
            if entry.critique:
                lessons.extend(entry.critique.lessons)
        
        return lessons[:20]  # Cap at 20 lessons
    
    def get_hypothesis_accuracy(self) -> float:
        """Calculate accuracy of hypotheses"""
        critiqued = [e for e in self.entries.values() if e.critique]
        if not critiqued:
            return 0.0
        
        correct = sum(1 for e in critiqued if e.critique.hypothesis_correct)
        return correct / len(critiqued)
    
    def get_pattern_insights(self) -> Dict[str, Any]:
        """Analyze journal for patterns"""
        critiqued = [e for e in self.entries.values() if e.critique]
        if len(critiqued) < 5:
            return {'status': 'insufficient_data'}
        
        insights = {
            'total_trades': len(critiqued),
            'hypothesis_accuracy': self.get_hypothesis_accuracy(),
            'best_setups': [],
            'worst_setups': [],
            'common_mistakes': []
        }
        
        # Find best and worst setups by regime
        regime_performance: Dict[str, List[float]] = {}
        for entry in critiqued:
            regime = entry.hypothesis.regime
            if regime not in regime_performance:
                regime_performance[regime] = []
            regime_performance[regime].append(entry.critique.pnl)
        
        for regime, pnls in regime_performance.items():
            if len(pnls) >= 3:
                avg_pnl = statistics.mean(pnls)
                win_rate = sum(1 for p in pnls if p > 0) / len(pnls)
                if win_rate > 0.6:
                    insights['best_setups'].append(f"{regime}: {win_rate:.0%} win rate")
                elif win_rate < 0.4:
                    insights['worst_setups'].append(f"{regime}: {win_rate:.0%} win rate")
        
        # Find common mistakes
        stopped_out = [e for e in critiqued if e.critique.outcome == TradeOutcomeType.STOPPED_OUT]
        if len(stopped_out) > len(critiqued) * 0.3:
            insights['common_mistakes'].append("Too many stopped out - review stop placement")
        
        high_uncertainty_losses = [
            e for e in critiqued 
            if e.hypothesis.overall_uncertainty > 0.5 and e.critique.pnl < 0
        ]
        if len(high_uncertainty_losses) > 3:
            insights['common_mistakes'].append("Trading with high uncertainty leads to losses")
        
        return insights


class TradeJournalingEngine:
    """
    Unified interface for trade journaling and uncertainty management.
    Combines TradeJournal and UncertaintyGovernor.
    """
    
    def __init__(self, data_dir: str = "data/journal"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        self.journal = TradeJournal(data_dir)
        self.governor = UncertaintyGovernor()
        
        # Background thread for periodic analysis
        self._running = False
        self._thread = None
        
        logger.info("TradeJournalingEngine initialized")
    
    def evaluate_trade_opportunity(
        self,
        symbol: str,
        direction: str,
        confidence: float,
        uncertainty_factors: Dict[str, float],
        supporting_factors: List[str],
        opposing_factors: List[str],
        regime: str = "unknown",
        session: str = "unknown"
    ) -> Dict[str, Any]:
        """
        Evaluate a trade opportunity with uncertainty analysis.
        Returns recommendation on whether to trade and how.
        """
        
        # Calculate uncertainty
        uncertainty, level, breakdown = self.governor.calculate_uncertainty(uncertainty_factors)
        
        # Get trading recommendation
        recommendation = self.governor.get_trading_recommendation(uncertainty)
        
        # Check if we should trade
        should_trade, reason = self.governor.should_trade(uncertainty, min_confidence=0.65)
        
        # Factor in supporting vs opposing factors
        factor_balance = len(supporting_factors) - len(opposing_factors)
        if factor_balance < -1:
            should_trade = False
            reason = f"Too many opposing factors ({len(opposing_factors)}) vs supporting ({len(supporting_factors)})"
        
        # Get recent lessons that might apply
        recent_lessons = self.journal.get_recent_lessons(5)
        relevant_lessons = [l for l in recent_lessons if symbol in l or regime in l]
        
        return {
            'should_trade': should_trade,
            'reason': reason,
            'uncertainty': uncertainty,
            'uncertainty_level': level.name,
            'uncertainty_breakdown': breakdown,
            'position_multiplier': recommendation['position_multiplier'],
            'recommendation': recommendation['recommendation'],
            'factor_balance': factor_balance,
            'relevant_lessons': relevant_lessons,
            'losing_streak': self.governor.losing_streak,
            'winning_streak': self.governor.winning_streak
        }
    
    def create_trade_entry(
        self,
        trade_id: str,
        symbol: str,
        direction: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        position_size: float,
        primary_thesis: str,
        expected_move: float,
        expected_duration: str,
        supporting_factors: List[str],
        opposing_factors: List[str],
        uncertainty_factors: Dict[str, float],
        regime: str = "unknown",
        session: str = "unknown",
        invalidation_price: float = 0.0,
        invalidation_reason: str = ""
    ) -> JournalEntry:
        """Create a complete journal entry for a new trade"""
        
        # Create hypothesis
        hypothesis = self.journal.create_hypothesis(
            trade_id=trade_id,
            symbol=symbol,
            direction=direction,
            primary_thesis=primary_thesis,
            expected_move=expected_move,
            expected_duration=expected_duration,
            supporting_factors=supporting_factors,
            opposing_factors=opposing_factors,
            uncertainty_sources=uncertainty_factors,
            regime=regime,
            session=session,
            invalidation_price=invalidation_price,
            invalidation_reason=invalidation_reason
        )
        
        # Open trade in journal
        entry = self.journal.open_trade(
            trade_id=trade_id,
            hypothesis=hypothesis,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=position_size
        )
        
        return entry
    
    def close_trade_entry(
        self,
        trade_id: str,
        exit_price: float,
        outcome: TradeOutcomeType,
        pnl: float,
        pnl_pips: float,
        max_favorable: float = 0.0,
        max_adverse: float = 0.0
    ) -> Optional[TradeCritique]:
        """Close a trade and generate critique"""
        
        # Close in journal
        critique = self.journal.close_trade(
            trade_id=trade_id,
            exit_price=exit_price,
            outcome=outcome,
            pnl=pnl,
            pnl_pips=pnl_pips,
            max_favorable=max_favorable,
            max_adverse=max_adverse
        )
        
        # Record outcome in governor
        if critique:
            won = outcome in [TradeOutcomeType.WIN, TradeOutcomeType.TAKE_PROFIT]
            self.governor.record_trade_outcome(won, pnl)
        
        return critique
    
    def get_pre_trade_checklist(self, symbol: str, direction: str, regime: str) -> List[str]:
        """Generate a pre-trade checklist based on past lessons"""
        checklist = [
            f"Confirm {direction} bias aligns with higher timeframe trend",
            f"Check for upcoming high-impact news for {symbol[:3]} and {symbol[3:]}",
            f"Verify current regime ({regime}) is favorable for this setup",
            "Ensure risk/reward ratio is at least 1:2",
            "Confirm position size matches uncertainty level"
        ]
        
        # Add lessons from past trades
        insights = self.journal.get_pattern_insights()
        if insights.get('common_mistakes'):
            for mistake in insights['common_mistakes'][:2]:
                checklist.append(f"AVOID: {mistake}")
        
        return checklist
    
    def get_state_summary(self) -> Dict[str, Any]:
        """Get summary of journaling state"""
        return {
            'total_journal_entries': len(self.journal.entries),
            'open_trades': len(self.journal.open_trades),
            'hypothesis_accuracy': self.journal.get_hypothesis_accuracy(),
            'winning_streak': self.governor.winning_streak,
            'losing_streak': self.governor.losing_streak,
            'recent_lessons': self.journal.get_recent_lessons(5),
            'pattern_insights': self.journal.get_pattern_insights()
        }
    
    def start_background_analysis(self, interval_minutes: int = 30):
        """Start background analysis thread"""
        if self._running:
            return
        
        self._running = True
        
        def analysis_loop():
            while self._running:
                try:
                    # Periodic analysis
                    insights = self.journal.get_pattern_insights()
                    if insights.get('common_mistakes'):
                        logger.info(f"Journal analysis - Common mistakes: {insights['common_mistakes']}")
                    
                    accuracy = self.journal.get_hypothesis_accuracy()
                    logger.info(f"Journal analysis - Hypothesis accuracy: {accuracy:.1%}")
                    
                except Exception as e:
                    logger.error(f"Journal analysis error: {e}")
                
                time.sleep(interval_minutes * 60)
        
        self._thread = threading.Thread(target=analysis_loop, daemon=True)
        self._thread.start()
        logger.info(f"Background journal analysis started (interval: {interval_minutes} min)")
    
    def stop_background_analysis(self):
        """Stop background analysis"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)


# Global instance
_journaling_engine: Optional[TradeJournalingEngine] = None


def get_journaling_engine() -> TradeJournalingEngine:
    """Get or create the global journaling engine"""
    global _journaling_engine
    if _journaling_engine is None:
        _journaling_engine = TradeJournalingEngine()
    return _journaling_engine


def evaluate_trade(
    symbol: str,
    direction: str,
    confidence: float,
    uncertainty_factors: Dict[str, float],
    supporting_factors: List[str],
    opposing_factors: List[str],
    regime: str = "unknown",
    session: str = "unknown"
) -> Dict[str, Any]:
    """Convenience function to evaluate a trade opportunity"""
    return get_journaling_engine().evaluate_trade_opportunity(
        symbol=symbol,
        direction=direction,
        confidence=confidence,
        uncertainty_factors=uncertainty_factors,
        supporting_factors=supporting_factors,
        opposing_factors=opposing_factors,
        regime=regime,
        session=session
    )


if __name__ == "__main__":
    # Test the journaling system
    logging.basicConfig(level=logging.INFO)
    
    engine = get_journaling_engine()
    
    # Test uncertainty evaluation
    evaluation = engine.evaluate_trade_opportunity(
        symbol="EURUSD",
        direction="BUY",
        confidence=0.75,
        uncertainty_factors={
            'technical': 0.3,
            'fundamental': 0.4,
            'sentiment': 0.5,
            'regime': 0.3,
            'volatility': 0.4
        },
        supporting_factors=[
            "Strong uptrend on H4",
            "RSI showing bullish divergence",
            "Price above 200 EMA"
        ],
        opposing_factors=[
            "NFP release tomorrow",
            "DXY showing strength"
        ],
        regime="trending",
        session="london"
    )
    
    print(f"Trade evaluation: {evaluation}")
    
    # Test creating a trade entry
    if evaluation['should_trade']:
        entry = engine.create_trade_entry(
            trade_id="TEST001",
            symbol="EURUSD",
            direction="BUY",
            entry_price=1.1000,
            stop_loss=1.0970,
            take_profit=1.1060,
            position_size=0.01,
            primary_thesis="Bullish continuation in uptrend with RSI divergence",
            expected_move=60,
            expected_duration="4-8 hours",
            supporting_factors=evaluation.get('supporting_factors', []),
            opposing_factors=evaluation.get('opposing_factors', []),
            uncertainty_factors={'technical': 0.3, 'fundamental': 0.4},
            regime="trending",
            session="london",
            invalidation_price=1.0970,
            invalidation_reason="Break below swing low invalidates bullish thesis"
        )
        print(f"Trade entry created: {entry.trade_id}")
        
        # Simulate closing the trade
        critique = engine.close_trade_entry(
            trade_id="TEST001",
            exit_price=1.1045,
            outcome=TradeOutcomeType.WIN,
            pnl=4.50,
            pnl_pips=45,
            max_favorable=50,
            max_adverse=15
        )
        print(f"Trade critique: {critique}")
    
    # Get state summary
    print(f"State summary: {engine.get_state_summary()}")
