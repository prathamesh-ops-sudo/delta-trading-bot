"""
Trading Captain - Intelligent Meta-Controller
Makes holistic trading decisions like a veteran trader with 100x knowledge.
Decides risk budget, position sizing, trailing logic, and generates trade thesis.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import json

logger = logging.getLogger(__name__)


class TradingMode(Enum):
    CONSERVATIVE = "conservative"
    NORMAL = "normal"
    AGGRESSIVE = "aggressive"
    RECOVERY = "recovery"


@dataclass
class TradeThesis:
    """Complete trade thesis explaining the decision like a veteran trader"""
    symbol: str
    direction: str
    
    # The "Why" - Entry reasoning
    entry_reason: str
    supporting_factors: List[str]
    confidence_breakdown: Dict[str, float]
    
    # The "Invalidation" - Why SL is placed there
    invalidation_reason: str
    sl_logic: str
    
    # The "Management Plan" - How to manage the trade
    management_plan: str
    trailing_strategy: str
    partial_tp_levels: List[Tuple[float, float]]  # (price, percentage)
    exit_conditions: List[str]
    
    # Risk assessment
    risk_budget_percent: float
    position_size_lots: float
    max_loss_usd: float
    expected_profit_usd: float
    risk_reward_ratio: float
    
    # Context
    market_context: str
    regime: str
    time_context: str
    
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_readable(self) -> str:
        """Generate human-readable trade thesis"""
        lines = [
            f"\n{'='*60}",
            f"TRADE THESIS: {self.symbol} {self.direction.upper()}",
            f"{'='*60}",
            f"\n[ENTRY REASONING]",
            f"{self.entry_reason}",
            f"\nSupporting factors:",
        ]
        for factor in self.supporting_factors:
            lines.append(f"  - {factor}")
        
        lines.extend([
            f"\n[CONFIDENCE BREAKDOWN]",
        ])
        for factor, score in self.confidence_breakdown.items():
            lines.append(f"  {factor}: {score:.0%}")
        
        lines.extend([
            f"\n[INVALIDATION]",
            f"{self.invalidation_reason}",
            f"SL Logic: {self.sl_logic}",
            f"\n[MANAGEMENT PLAN]",
            f"{self.management_plan}",
            f"Trailing: {self.trailing_strategy}",
            f"\nPartial TP levels:",
        ])
        for price, pct in self.partial_tp_levels:
            lines.append(f"  - Close {pct:.0%} at {price:.5f}")
        
        lines.extend([
            f"\nExit conditions:",
        ])
        for condition in self.exit_conditions:
            lines.append(f"  - {condition}")
        
        lines.extend([
            f"\n[RISK ASSESSMENT]",
            f"Risk budget: {self.risk_budget_percent:.1%} of account",
            f"Position size: {self.position_size_lots:.2f} lots",
            f"Max loss: ${self.max_loss_usd:.2f}",
            f"Expected profit: ${self.expected_profit_usd:.2f}",
            f"Risk/Reward: 1:{self.risk_reward_ratio:.1f}",
            f"\n[CONTEXT]",
            f"Market: {self.market_context}",
            f"Regime: {self.regime}",
            f"Time: {self.time_context}",
            f"{'='*60}\n",
        ])
        
        return "\n".join(lines)


@dataclass
class PerformanceMemory:
    """Tracks recent performance for adaptive behavior"""
    recent_trades: List[Dict] = field(default_factory=list)
    win_streak: int = 0
    loss_streak: int = 0
    daily_pnl: float = 0.0
    weekly_pnl: float = 0.0
    drawdown_percent: float = 0.0
    peak_balance: float = 100.0
    current_balance: float = 100.0
    
    # Strategy performance
    strategy_wins: Dict[str, int] = field(default_factory=dict)
    strategy_losses: Dict[str, int] = field(default_factory=dict)
    
    # Regime performance
    regime_wins: Dict[str, int] = field(default_factory=dict)
    regime_losses: Dict[str, int] = field(default_factory=dict)
    
    # Time-based performance
    hour_performance: Dict[int, float] = field(default_factory=dict)
    
    # Insights learned
    insights: List[str] = field(default_factory=list)


class TradingCaptain:
    """
    The Trading Captain - Meta-controller that thinks like a veteran trader.
    Makes holistic decisions about risk, position sizing, and trade management.
    """
    
    def __init__(self, initial_balance: float = 100.0):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.mode = TradingMode.NORMAL
        self.memory = PerformanceMemory(
            peak_balance=initial_balance,
            current_balance=initial_balance
        )
        
        # Risk parameters by mode
        self.mode_params = {
            TradingMode.CONSERVATIVE: {
                'risk_percent': 0.5,
                'max_risk_percent': 1.0,
                'min_confidence': 0.70,
                'max_open_trades': 2,
                'leverage_multiplier': 0.5,
                'trailing_activation_pips': 30,
            },
            TradingMode.NORMAL: {
                'risk_percent': 1.0,
                'max_risk_percent': 2.0,
                'min_confidence': 0.60,
                'max_open_trades': 4,
                'leverage_multiplier': 1.0,
                'trailing_activation_pips': 20,
            },
            TradingMode.AGGRESSIVE: {
                'risk_percent': 1.5,
                'max_risk_percent': 2.5,
                'min_confidence': 0.55,
                'max_open_trades': 6,
                'leverage_multiplier': 1.5,
                'trailing_activation_pips': 15,
            },
            TradingMode.RECOVERY: {
                'risk_percent': 0.25,
                'max_risk_percent': 0.5,
                'min_confidence': 0.75,
                'max_open_trades': 1,
                'leverage_multiplier': 0.3,
                'trailing_activation_pips': 40,
            },
        }
        
        # Learned preferences
        self.preferred_strategies: Dict[str, float] = {}
        self.avoided_hours: List[int] = []
        self.preferred_pairs: List[str] = []
        
        logger.info(f"Trading Captain initialized - Mode: {self.mode.value}, Balance: ${initial_balance}")
    
    def update_balance(self, new_balance: float):
        """Update current balance and recalculate mode"""
        old_balance = self.current_balance
        self.current_balance = new_balance
        self.memory.current_balance = new_balance
        
        # Update peak balance
        if new_balance > self.memory.peak_balance:
            self.memory.peak_balance = new_balance
        
        # Calculate drawdown
        self.memory.drawdown_percent = (self.memory.peak_balance - new_balance) / self.memory.peak_balance
        
        # Update daily P&L
        self.memory.daily_pnl += (new_balance - old_balance)
        
        # Auto-adjust mode based on performance
        self._adjust_mode()
        
        logger.info(f"Balance updated: ${old_balance:.2f} -> ${new_balance:.2f}, "
                   f"Drawdown: {self.memory.drawdown_percent:.1%}, Mode: {self.mode.value}")
    
    def _adjust_mode(self):
        """Automatically adjust trading mode based on performance"""
        old_mode = self.mode
        
        # Recovery mode if drawdown > 15%
        if self.memory.drawdown_percent > 0.15:
            self.mode = TradingMode.RECOVERY
            if old_mode != self.mode:
                self._add_insight(f"Switched to RECOVERY mode - Drawdown at {self.memory.drawdown_percent:.1%}")
        
        # Conservative if drawdown > 10% or on loss streak
        elif self.memory.drawdown_percent > 0.10 or self.memory.loss_streak >= 3:
            self.mode = TradingMode.CONSERVATIVE
            if old_mode != self.mode:
                self._add_insight(f"Switched to CONSERVATIVE mode - Protecting capital")
        
        # Aggressive if on win streak and low drawdown
        elif self.memory.win_streak >= 3 and self.memory.drawdown_percent < 0.05:
            self.mode = TradingMode.AGGRESSIVE
            if old_mode != self.mode:
                self._add_insight(f"Switched to AGGRESSIVE mode - {self.memory.win_streak} win streak")
        
        # Normal otherwise
        else:
            self.mode = TradingMode.NORMAL
    
    def _add_insight(self, insight: str):
        """Add a learned insight"""
        timestamped = f"[{datetime.now().strftime('%Y-%m-%d %H:%M')}] {insight}"
        self.memory.insights.append(timestamped)
        logger.info(f"INSIGHT: {insight}")
        
        # Keep only last 100 insights
        if len(self.memory.insights) > 100:
            self.memory.insights = self.memory.insights[-100:]
    
    def record_trade_result(self, trade: Dict):
        """Record trade result and learn from it"""
        self.memory.recent_trades.append(trade)
        
        # Keep only last 50 trades
        if len(self.memory.recent_trades) > 50:
            self.memory.recent_trades = self.memory.recent_trades[-50:]
        
        is_win = trade.get('profit', 0) > 0
        strategy = trade.get('strategy', 'unknown')
        regime = trade.get('regime', 'unknown')
        hour = trade.get('hour', datetime.now().hour)
        
        # Update streaks
        if is_win:
            self.memory.win_streak += 1
            self.memory.loss_streak = 0
        else:
            self.memory.loss_streak += 1
            self.memory.win_streak = 0
        
        # Update strategy performance
        if is_win:
            self.memory.strategy_wins[strategy] = self.memory.strategy_wins.get(strategy, 0) + 1
        else:
            self.memory.strategy_losses[strategy] = self.memory.strategy_losses.get(strategy, 0) + 1
        
        # Update regime performance
        if is_win:
            self.memory.regime_wins[regime] = self.memory.regime_wins.get(regime, 0) + 1
        else:
            self.memory.regime_losses[regime] = self.memory.regime_losses.get(regime, 0) + 1
        
        # Update hour performance
        profit = trade.get('profit', 0)
        self.memory.hour_performance[hour] = self.memory.hour_performance.get(hour, 0) + profit
        
        # Generate insights
        self._analyze_and_learn()
    
    def _analyze_and_learn(self):
        """Analyze recent performance and generate insights"""
        # Check for strategy patterns
        for strategy, wins in self.memory.strategy_wins.items():
            losses = self.memory.strategy_losses.get(strategy, 0)
            total = wins + losses
            if total >= 5:
                win_rate = wins / total
                if win_rate > 0.7:
                    self._add_insight(f"Strategy '{strategy}' performing well ({win_rate:.0%} win rate)")
                    self.preferred_strategies[strategy] = win_rate
                elif win_rate < 0.3:
                    self._add_insight(f"Strategy '{strategy}' underperforming ({win_rate:.0%} win rate) - reducing weight")
                    self.preferred_strategies[strategy] = win_rate
        
        # Check for time patterns
        for hour, pnl in self.memory.hour_performance.items():
            if pnl < -5:  # Lost more than $5 at this hour
                if hour not in self.avoided_hours:
                    self.avoided_hours.append(hour)
                    self._add_insight(f"Hour {hour}:00 UTC has been unprofitable (${pnl:.2f}) - will be more cautious")
            elif pnl > 10:  # Made more than $10 at this hour
                self._add_insight(f"Hour {hour}:00 UTC has been profitable (${pnl:.2f}) - favorable trading time")
        
        # Check for regime patterns
        for regime, wins in self.memory.regime_wins.items():
            losses = self.memory.regime_losses.get(regime, 0)
            total = wins + losses
            if total >= 3:
                win_rate = wins / total
                if win_rate < 0.4:
                    self._add_insight(f"Struggling in '{regime}' regime ({win_rate:.0%} win rate) - will reduce aggression")
    
    def calculate_risk_budget(self, signal_confidence: float, strategy: str, 
                              regime: str, current_hour: int) -> float:
        """
        Calculate dynamic risk budget based on all factors.
        Returns risk as percentage of account (0.25% to 2.5%)
        """
        params = self.mode_params[self.mode]
        base_risk = params['risk_percent']
        
        # Confidence adjustment
        if signal_confidence > 0.8:
            base_risk *= 1.3
        elif signal_confidence > 0.7:
            base_risk *= 1.1
        elif signal_confidence < 0.6:
            base_risk *= 0.7
        
        # Strategy performance adjustment
        if strategy in self.preferred_strategies:
            strategy_win_rate = self.preferred_strategies[strategy]
            if strategy_win_rate > 0.6:
                base_risk *= 1.2
            elif strategy_win_rate < 0.4:
                base_risk *= 0.6
        
        # Time-based adjustment
        if current_hour in self.avoided_hours:
            base_risk *= 0.5
        
        # Win/loss streak adjustment
        if self.memory.win_streak >= 3:
            base_risk *= 1.2  # Compound wins
        elif self.memory.loss_streak >= 2:
            base_risk *= 0.7  # Cut losses
        
        # Drawdown adjustment
        if self.memory.drawdown_percent > 0.10:
            base_risk *= 0.5
        elif self.memory.drawdown_percent > 0.05:
            base_risk *= 0.8
        
        # Clamp to mode limits
        risk = max(0.25, min(base_risk, params['max_risk_percent']))
        
        return risk / 100  # Return as decimal
    
    def calculate_position_size(self, risk_budget_percent: float, 
                                 sl_pips: float, account_balance: float) -> float:
        """
        Calculate position size in lots based on risk budget.
        Uses proper money management formula.
        """
        risk_amount = account_balance * risk_budget_percent
        
        # Pip value per standard lot (approximately $10 for most USD pairs)
        pip_value_per_lot = 10.0
        
        # Position size = Risk Amount / (SL in pips * Pip value per lot)
        if sl_pips > 0:
            position_size = risk_amount / (sl_pips * pip_value_per_lot)
        else:
            position_size = 0.01
        
        # Clamp to reasonable limits for $100 account
        # Min: 0.01 lots (micro lot)
        # Max: Based on available margin and risk
        max_lots = min(1.0, account_balance / 100)  # Rough margin limit
        position_size = max(0.01, min(position_size, max_lots))
        
        return round(position_size, 2)
    
    def should_trail_stop(self, current_profit_pips: float, adx: float, 
                          rsi: float, regime: str) -> Tuple[bool, str]:
        """
        Decide whether to trail stop and explain why.
        Returns (should_trail, reason)
        """
        params = self.mode_params[self.mode]
        activation_pips = params['trailing_activation_pips']
        
        # Not enough profit yet
        if current_profit_pips < activation_pips:
            return False, f"Profit ({current_profit_pips:.1f} pips) below activation threshold ({activation_pips} pips)"
        
        # Strong trend - trail tightly
        if adx > 40:
            return True, f"Strong trend (ADX={adx:.0f}) - trailing to lock in profits"
        
        # Momentum weakening - stop trailing
        if adx < 20:
            return False, f"Weak momentum (ADX={adx:.0f}) - holding current stop"
        
        # RSI divergence - be cautious
        if rsi > 75 or rsi < 25:
            return False, f"RSI extreme ({rsi:.0f}) - potential reversal, holding stop"
        
        # Default: trail if in profit
        return True, f"In profit ({current_profit_pips:.1f} pips) - trailing stop"
    
    def should_take_partial_profit(self, current_profit_pips: float, 
                                    target_pips: float) -> Tuple[bool, float, str]:
        """
        Decide whether to take partial profit.
        Returns (should_take, percentage_to_close, reason)
        """
        profit_ratio = current_profit_pips / target_pips if target_pips > 0 else 0
        
        # At 50% of target - take 25%
        if 0.45 <= profit_ratio < 0.75:
            return True, 0.25, f"Reached 50% of target - securing 25% of position"
        
        # At 75% of target - take another 25%
        if 0.75 <= profit_ratio < 1.0:
            return True, 0.25, f"Reached 75% of target - securing another 25%"
        
        # Beyond target - let it run with trailing
        if profit_ratio >= 1.0:
            return False, 0, f"At target - letting remainder run with trailing stop"
        
        return False, 0, "Not at partial profit level yet"
    
    def generate_trade_thesis(self, signal: Any, market_data: Dict, 
                               indicators: Dict, regime: str) -> TradeThesis:
        """
        Generate a complete trade thesis explaining the decision.
        This is what makes the system feel intelligent - it explains its thinking.
        """
        symbol = signal.symbol
        direction = "LONG" if signal.direction.value == 1 else "SHORT"
        current_price = signal.entry_price
        
        # Calculate risk budget
        current_hour = datetime.now().hour
        risk_budget = self.calculate_risk_budget(
            signal.confidence, signal.strategy, regime, current_hour
        )
        
        # Calculate SL distance in pips
        pip_size = 0.01 if 'JPY' in symbol else 0.0001
        sl_pips = abs(signal.entry_price - signal.stop_loss) / pip_size
        tp_pips = abs(signal.take_profit - signal.entry_price) / pip_size
        
        # Calculate position size
        position_size = self.calculate_position_size(
            risk_budget, sl_pips, self.current_balance
        )
        
        # Calculate expected values
        max_loss = self.current_balance * risk_budget
        expected_profit = max_loss * signal.risk_reward if hasattr(signal, 'risk_reward') else max_loss * 2
        
        # Build supporting factors
        supporting_factors = []
        confidence_breakdown = {}
        
        if signal.adx > 30:
            supporting_factors.append(f"Strong trend confirmed (ADX={signal.adx:.0f})")
            confidence_breakdown['Trend Strength'] = min(signal.adx / 50, 1.0)
        
        if signal.mtf_alignment > 0.5:
            supporting_factors.append(f"Multi-timeframe alignment ({signal.mtf_alignment:.0%})")
            confidence_breakdown['MTF Alignment'] = signal.mtf_alignment
        
        if signal.fvg_signal:
            supporting_factors.append(f"Fair Value Gap detected ({signal.fvg_signal.size:.1f} pips)")
            confidence_breakdown['FVG Signal'] = 0.7
        
        if signal.liquidity_sweep:
            supporting_factors.append("Liquidity sweep with reversal confirmation")
            confidence_breakdown['Liquidity Sweep'] = 0.8
        
        # RSI context
        if 40 <= signal.rsi <= 60:
            supporting_factors.append(f"RSI neutral zone ({signal.rsi:.0f}) - room to move")
        elif (direction == "LONG" and signal.rsi < 40) or (direction == "SHORT" and signal.rsi > 60):
            supporting_factors.append(f"RSI supports direction ({signal.rsi:.0f})")
        
        confidence_breakdown['Base Strategy'] = signal.confidence
        
        # Build entry reason
        entry_reason = f"{signal.strategy.replace('_', ' ').title()} setup on {symbol}. "
        entry_reason += f"Market is in {regime} regime with {signal.adx:.0f} ADX. "
        if direction == "LONG":
            entry_reason += f"Looking for upside move from {current_price:.5f} to {signal.take_profit:.5f}."
        else:
            entry_reason += f"Looking for downside move from {current_price:.5f} to {signal.take_profit:.5f}."
        
        # Build invalidation reason
        if direction == "LONG":
            invalidation_reason = f"Trade invalidated if price breaks below {signal.stop_loss:.5f}. "
            invalidation_reason += f"This level represents {sl_pips:.0f} pips of risk."
        else:
            invalidation_reason = f"Trade invalidated if price breaks above {signal.stop_loss:.5f}. "
            invalidation_reason += f"This level represents {sl_pips:.0f} pips of risk."
        
        sl_logic = f"Stop placed at {sl_pips:.0f} pips ({signal.atr*2:.5f} = 2x ATR) to account for normal volatility."
        
        # Build management plan
        management_plan = f"Risk {risk_budget*100:.1f}% of account (${max_loss:.2f}) for potential ${expected_profit:.2f} gain. "
        management_plan += f"Position size: {position_size:.2f} lots. "
        
        if self.mode == TradingMode.AGGRESSIVE:
            management_plan += "AGGRESSIVE mode - will trail tightly to maximize gains."
        elif self.mode == TradingMode.CONSERVATIVE:
            management_plan += "CONSERVATIVE mode - will take partial profits early."
        elif self.mode == TradingMode.RECOVERY:
            management_plan += "RECOVERY mode - minimal risk, quick exits."
        else:
            management_plan += "NORMAL mode - balanced approach."
        
        # Trailing strategy
        params = self.mode_params[self.mode]
        trailing_strategy = f"Activate trailing after {params['trailing_activation_pips']} pips profit. "
        if signal.adx > 35:
            trailing_strategy += "Trail at 1.0x ATR (tight) due to strong trend."
        else:
            trailing_strategy += "Trail at 1.5x ATR (loose) to allow for pullbacks."
        
        # Partial TP levels
        partial_tp_levels = []
        if direction == "LONG":
            tp1 = current_price + (signal.take_profit - current_price) * 0.5
            tp2 = current_price + (signal.take_profit - current_price) * 0.75
            partial_tp_levels = [(tp1, 0.25), (tp2, 0.25), (signal.take_profit, 0.50)]
        else:
            tp1 = current_price - (current_price - signal.take_profit) * 0.5
            tp2 = current_price - (current_price - signal.take_profit) * 0.75
            partial_tp_levels = [(tp1, 0.25), (tp2, 0.25), (signal.take_profit, 0.50)]
        
        # Exit conditions
        exit_conditions = [
            f"Stop loss hit at {signal.stop_loss:.5f}",
            f"Take profit hit at {signal.take_profit:.5f}",
            "RSI divergence detected",
            "Regime change to unfavorable state",
            "Major news event approaching",
        ]
        
        # Market context
        market_context = f"{symbol} trading at {current_price:.5f}. "
        market_context += f"Volatility: {'High' if signal.atr > 0.001 else 'Normal' if signal.atr > 0.0005 else 'Low'}. "
        market_context += f"Spread conditions: Normal."
        
        # Time context
        time_context = f"Trading at {datetime.now().strftime('%H:%M')} UTC. "
        if current_hour in [8, 9, 10, 13, 14, 15]:
            time_context += "Active session (London/NY overlap)."
        elif current_hour in [0, 1, 2, 3]:
            time_context += "Asian session - lower volatility expected."
        else:
            time_context += "Standard session."
        
        thesis = TradeThesis(
            symbol=symbol,
            direction=direction,
            entry_reason=entry_reason,
            supporting_factors=supporting_factors,
            confidence_breakdown=confidence_breakdown,
            invalidation_reason=invalidation_reason,
            sl_logic=sl_logic,
            management_plan=management_plan,
            trailing_strategy=trailing_strategy,
            partial_tp_levels=partial_tp_levels,
            exit_conditions=exit_conditions,
            risk_budget_percent=risk_budget,
            position_size_lots=position_size,
            max_loss_usd=max_loss,
            expected_profit_usd=expected_profit,
            risk_reward_ratio=tp_pips / sl_pips if sl_pips > 0 else 2.0,
            market_context=market_context,
            regime=regime,
            time_context=time_context,
        )
        
        return thesis
    
    def should_take_trade(self, signal: Any, open_positions: int) -> Tuple[bool, str]:
        """
        Final decision on whether to take a trade.
        Returns (should_trade, reason)
        """
        params = self.mode_params[self.mode]
        
        # Check max open trades
        if open_positions >= params['max_open_trades']:
            return False, f"Max open trades ({params['max_open_trades']}) reached for {self.mode.value} mode"
        
        # Check minimum confidence
        if signal.confidence < params['min_confidence']:
            return False, f"Confidence ({signal.confidence:.0%}) below threshold ({params['min_confidence']:.0%}) for {self.mode.value} mode"
        
        # Check drawdown halt
        if self.memory.drawdown_percent > 0.20:
            return False, f"Trading halted - Drawdown ({self.memory.drawdown_percent:.0%}) exceeds 20% limit"
        
        # Check avoided hours
        current_hour = datetime.now().hour
        if current_hour in self.avoided_hours and self.mode != TradingMode.AGGRESSIVE:
            return False, f"Hour {current_hour}:00 has been unprofitable - skipping in {self.mode.value} mode"
        
        # Check strategy performance
        strategy = signal.strategy
        if strategy in self.preferred_strategies:
            if self.preferred_strategies[strategy] < 0.35:
                return False, f"Strategy '{strategy}' has poor performance ({self.preferred_strategies[strategy]:.0%} win rate)"
        
        return True, f"Trade approved in {self.mode.value} mode"
    
    def get_status_report(self) -> str:
        """Generate a status report of the trading captain"""
        lines = [
            f"\n{'='*50}",
            f"TRADING CAPTAIN STATUS",
            f"{'='*50}",
            f"Mode: {self.mode.value.upper()}",
            f"Balance: ${self.current_balance:.2f}",
            f"Peak: ${self.memory.peak_balance:.2f}",
            f"Drawdown: {self.memory.drawdown_percent:.1%}",
            f"Win Streak: {self.memory.win_streak}",
            f"Loss Streak: {self.memory.loss_streak}",
            f"Daily P&L: ${self.memory.daily_pnl:.2f}",
            f"\nRecent Insights:",
        ]
        
        for insight in self.memory.insights[-5:]:
            lines.append(f"  {insight}")
        
        lines.append(f"{'='*50}\n")
        
        return "\n".join(lines)


# Global instance
trading_captain = TradingCaptain(initial_balance=100.0)
