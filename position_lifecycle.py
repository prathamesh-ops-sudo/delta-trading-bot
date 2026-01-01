"""
Priority 3: Position Lifecycle Manager

Manages the complete lifecycle of trading positions with institutional-grade features:
- Trailing stops with regime-aware parameters
- Break-even moves after X pips profit
- Partial close rules (take 50% at 1R, let rest run)
- Averaging down rules (opt-in, regime-gated, risk-budgeted)
- Time-based exits (close before weekend, news events)

This is what separates amateur traders from professionals - managing positions
AFTER entry is often more important than the entry itself.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import json

logger = logging.getLogger(__name__)


class LifecycleAction(Enum):
    """Actions the lifecycle manager can recommend"""
    HOLD = "hold"
    TRAIL_STOP = "trail_stop"
    MOVE_TO_BREAKEVEN = "move_to_breakeven"
    PARTIAL_CLOSE = "partial_close"
    FULL_CLOSE = "full_close"
    ADD_TO_POSITION = "add_to_position"
    TIGHTEN_STOP = "tighten_stop"


class ExitReason(Enum):
    """Reasons for exiting a position"""
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    TRAILING_STOP = "trailing_stop"
    BREAKEVEN_STOP = "breakeven_stop"
    TIME_EXIT = "time_exit"
    WEEKEND_CLOSE = "weekend_close"
    NEWS_EVENT = "news_event"
    REGIME_CHANGE = "regime_change"
    CORRELATION_LIMIT = "correlation_limit"
    DRAWDOWN_LIMIT = "drawdown_limit"
    MANUAL = "manual"


class PositionPhase(Enum):
    """Phases of a position's lifecycle"""
    ENTRY = "entry"              # Just entered, initial stop
    EARLY = "early"              # < 0.5R profit
    DEVELOPING = "developing"    # 0.5R - 1R profit
    PROFITABLE = "profitable"    # 1R - 2R profit
    RUNNER = "runner"            # > 2R profit, let it run
    DEFENSIVE = "defensive"      # Protecting profits


@dataclass
class PositionState:
    """Current state of a position for lifecycle management"""
    symbol: str
    direction: str  # 'long' or 'short'
    entry_price: float
    current_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    entry_time: datetime
    
    # Calculated fields
    initial_risk_pips: float = 0.0
    current_pnl_pips: float = 0.0
    current_r_multiple: float = 0.0
    phase: PositionPhase = PositionPhase.ENTRY
    
    # Lifecycle tracking
    breakeven_moved: bool = False
    partial_closes: int = 0
    trailing_active: bool = False
    highest_r: float = 0.0
    lowest_r: float = 0.0
    
    # Averaging tracking
    average_downs: int = 0
    max_average_downs: int = 1
    
    def __post_init__(self):
        """Calculate derived fields"""
        self._calculate_metrics()
    
    def _calculate_metrics(self):
        """Calculate current position metrics"""
        # Calculate pip value based on symbol
        pip_multiplier = 100 if 'JPY' in self.symbol else 10000
        
        if self.direction == 'long':
            self.current_pnl_pips = (self.current_price - self.entry_price) * pip_multiplier
            self.initial_risk_pips = (self.entry_price - self.stop_loss) * pip_multiplier
        else:
            self.current_pnl_pips = (self.entry_price - self.current_price) * pip_multiplier
            self.initial_risk_pips = (self.stop_loss - self.entry_price) * pip_multiplier
        
        # Calculate R-multiple
        if self.initial_risk_pips > 0:
            self.current_r_multiple = self.current_pnl_pips / self.initial_risk_pips
        
        # Track highest/lowest R
        self.highest_r = max(self.highest_r, self.current_r_multiple)
        self.lowest_r = min(self.lowest_r, self.current_r_multiple)
        
        # Determine phase
        self._determine_phase()
    
    def _determine_phase(self):
        """Determine current position phase based on R-multiple"""
        if self.current_r_multiple < 0:
            self.phase = PositionPhase.ENTRY
        elif self.current_r_multiple < 0.5:
            self.phase = PositionPhase.EARLY
        elif self.current_r_multiple < 1.0:
            self.phase = PositionPhase.DEVELOPING
        elif self.current_r_multiple < 2.0:
            self.phase = PositionPhase.PROFITABLE
        else:
            self.phase = PositionPhase.RUNNER
        
        # Check if we should be defensive (gave back significant profits)
        if self.highest_r > 1.0 and self.current_r_multiple < self.highest_r * 0.5:
            self.phase = PositionPhase.DEFENSIVE
    
    def update_price(self, current_price: float):
        """Update current price and recalculate metrics"""
        self.current_price = current_price
        self._calculate_metrics()


@dataclass
class LifecycleConfig:
    """Configuration for position lifecycle management"""
    # Break-even settings
    breakeven_trigger_r: float = 0.5      # Move to BE after 0.5R profit
    breakeven_buffer_pips: float = 2.0    # Buffer above entry for BE
    
    # Partial close settings
    partial_close_enabled: bool = True
    partial_close_trigger_r: float = 1.0  # First partial at 1R
    partial_close_percent: float = 0.5    # Close 50% at first target
    second_partial_trigger_r: float = 2.0 # Second partial at 2R
    second_partial_percent: float = 0.25  # Close 25% more at 2R
    
    # Trailing stop settings
    trailing_enabled: bool = True
    trailing_trigger_r: float = 1.0       # Start trailing after 1R
    trailing_distance_r: float = 0.5      # Trail 0.5R behind price
    trailing_step_pips: float = 5.0       # Minimum step for trail updates
    
    # Regime-aware trailing
    trending_trail_distance_r: float = 0.75   # Wider trail in trends
    ranging_trail_distance_r: float = 0.3     # Tighter trail in ranges
    volatile_trail_distance_r: float = 1.0    # Very wide in volatile
    
    # Averaging down settings (DANGEROUS - opt-in only)
    averaging_enabled: bool = False       # Disabled by default
    averaging_max_adds: int = 1           # Maximum 1 add
    averaging_trigger_r: float = -0.5     # Add at -0.5R
    averaging_size_percent: float = 0.5   # Add 50% of original size
    averaging_regime_required: str = "trending"  # Only in trends
    
    # Time-based exits
    weekend_close_enabled: bool = True
    weekend_close_hours_before: int = 2   # Close 2 hours before weekend
    max_hold_hours: int = 168             # Max 1 week hold
    news_close_minutes_before: int = 30   # Close 30 min before major news
    
    # Risk limits
    max_drawdown_r: float = -1.5          # Close if drawdown exceeds 1.5R
    profit_protection_r: float = 2.0      # Protect profits after 2R


@dataclass
class LifecycleRecommendation:
    """Recommendation from the lifecycle manager"""
    action: LifecycleAction
    reason: str
    new_stop_loss: Optional[float] = None
    close_percent: Optional[float] = None
    add_size: Optional[float] = None
    urgency: str = "normal"  # 'low', 'normal', 'high', 'critical'
    details: Dict[str, Any] = field(default_factory=dict)


class PositionLifecycleManager:
    """
    Manages position lifecycle with institutional-grade features.
    
    Key principles:
    1. Protect capital first (move to breakeven early)
    2. Let winners run (trailing stops, not fixed TP)
    3. Cut losers quickly (respect stops)
    4. Scale out of winners (partial closes)
    5. Never average down without strict rules
    """
    
    def __init__(self, config: Optional[LifecycleConfig] = None):
        self.config = config or LifecycleConfig()
        self.positions: Dict[str, PositionState] = {}
        self.closed_positions: List[Dict] = []
        self.upcoming_events: List[Dict] = []
        self.current_regime: str = "unknown"
        
        logger.info("PositionLifecycleManager initialized")
    
    def register_position(self, position: PositionState) -> None:
        """Register a new position for lifecycle management"""
        key = f"{position.symbol}_{position.direction}"
        self.positions[key] = position
        logger.info(f"[LIFECYCLE] Registered position: {key}, entry={position.entry_price}, "
                   f"SL={position.stop_loss}, TP={position.take_profit}")
    
    def update_position(self, symbol: str, direction: str, current_price: float) -> Optional[LifecycleRecommendation]:
        """Update position price and get lifecycle recommendation"""
        key = f"{symbol}_{direction}"
        if key not in self.positions:
            return None
        
        position = self.positions[key]
        position.update_price(current_price)
        
        return self._evaluate_position(position)
    
    def set_regime(self, regime: str) -> None:
        """Update current market regime"""
        self.current_regime = regime.lower()
    
    def set_upcoming_events(self, events: List[Dict]) -> None:
        """Set upcoming economic events"""
        self.upcoming_events = events
    
    def _evaluate_position(self, position: PositionState) -> LifecycleRecommendation:
        """Evaluate position and return lifecycle recommendation"""
        recommendations = []
        
        # Check time-based exits first (highest priority)
        time_rec = self._check_time_exits(position)
        if time_rec and time_rec.urgency == 'critical':
            return time_rec
        if time_rec:
            recommendations.append(time_rec)
        
        # Check drawdown limit
        dd_rec = self._check_drawdown_limit(position)
        if dd_rec:
            return dd_rec  # Immediate action needed
        
        # Check break-even move
        be_rec = self._check_breakeven(position)
        if be_rec:
            recommendations.append(be_rec)
        
        # Check partial close
        pc_rec = self._check_partial_close(position)
        if pc_rec:
            recommendations.append(pc_rec)
        
        # Check trailing stop
        ts_rec = self._check_trailing_stop(position)
        if ts_rec:
            recommendations.append(ts_rec)
        
        # Check averaging opportunity (if enabled)
        if self.config.averaging_enabled:
            avg_rec = self._check_averaging(position)
            if avg_rec:
                recommendations.append(avg_rec)
        
        # Return highest priority recommendation
        if recommendations:
            # Sort by urgency
            urgency_order = {'critical': 0, 'high': 1, 'normal': 2, 'low': 3}
            recommendations.sort(key=lambda r: urgency_order.get(r.urgency, 3))
            return recommendations[0]
        
        # Default: hold
        return LifecycleRecommendation(
            action=LifecycleAction.HOLD,
            reason=f"Position in {position.phase.value} phase at {position.current_r_multiple:.2f}R",
            details={
                'phase': position.phase.value,
                'r_multiple': position.current_r_multiple,
                'pnl_pips': position.current_pnl_pips
            }
        )
    
    def _check_time_exits(self, position: PositionState) -> Optional[LifecycleRecommendation]:
        """Check for time-based exit conditions"""
        now = datetime.utcnow()
        
        # Weekend close check
        if self.config.weekend_close_enabled:
            # Friday 5pm EST = Friday 10pm UTC
            if now.weekday() == 4:  # Friday
                weekend_close_time = now.replace(hour=22, minute=0, second=0)
                close_threshold = weekend_close_time - timedelta(hours=self.config.weekend_close_hours_before)
                
                if now >= close_threshold:
                    return LifecycleRecommendation(
                        action=LifecycleAction.FULL_CLOSE,
                        reason="Weekend approaching - closing to avoid gap risk",
                        close_percent=1.0,
                        urgency='high',
                        details={'exit_reason': ExitReason.WEEKEND_CLOSE.value}
                    )
        
        # Max hold time check
        hold_duration = (now - position.entry_time).total_seconds() / 3600
        if hold_duration > self.config.max_hold_hours:
            return LifecycleRecommendation(
                action=LifecycleAction.FULL_CLOSE,
                reason=f"Max hold time exceeded ({hold_duration:.0f}h > {self.config.max_hold_hours}h)",
                close_percent=1.0,
                urgency='normal',
                details={'exit_reason': ExitReason.TIME_EXIT.value}
            )
        
        # News event check
        if self.upcoming_events:
            for event in self.upcoming_events:
                event_time = event.get('time')
                if event_time:
                    if isinstance(event_time, str):
                        try:
                            event_time = datetime.fromisoformat(event_time.replace('Z', '+00:00'))
                        except:
                            continue
                    
                    minutes_until = (event_time - now).total_seconds() / 60
                    if 0 < minutes_until < self.config.news_close_minutes_before:
                        impact = event.get('impact', 'medium')
                        if impact in ['high', 'critical']:
                            return LifecycleRecommendation(
                                action=LifecycleAction.FULL_CLOSE,
                                reason=f"High-impact news in {minutes_until:.0f} min: {event.get('title', 'Unknown')}",
                                close_percent=1.0,
                                urgency='critical',
                                details={'exit_reason': ExitReason.NEWS_EVENT.value, 'event': event}
                            )
        
        return None
    
    def _check_drawdown_limit(self, position: PositionState) -> Optional[LifecycleRecommendation]:
        """Check if position has exceeded max drawdown"""
        if position.current_r_multiple < self.config.max_drawdown_r:
            return LifecycleRecommendation(
                action=LifecycleAction.FULL_CLOSE,
                reason=f"Max drawdown exceeded ({position.current_r_multiple:.2f}R < {self.config.max_drawdown_r}R)",
                close_percent=1.0,
                urgency='critical',
                details={'exit_reason': ExitReason.DRAWDOWN_LIMIT.value}
            )
        return None
    
    def _check_breakeven(self, position: PositionState) -> Optional[LifecycleRecommendation]:
        """Check if we should move stop to breakeven"""
        if position.breakeven_moved:
            return None
        
        if position.current_r_multiple >= self.config.breakeven_trigger_r:
            # Calculate new stop at breakeven + buffer
            pip_multiplier = 100 if 'JPY' in position.symbol else 10000
            buffer = self.config.breakeven_buffer_pips / pip_multiplier
            
            if position.direction == 'long':
                new_stop = position.entry_price + buffer
            else:
                new_stop = position.entry_price - buffer
            
            return LifecycleRecommendation(
                action=LifecycleAction.MOVE_TO_BREAKEVEN,
                reason=f"Moving to breakeven at {position.current_r_multiple:.2f}R profit",
                new_stop_loss=new_stop,
                urgency='normal',
                details={'trigger_r': self.config.breakeven_trigger_r}
            )
        
        return None
    
    def _check_partial_close(self, position: PositionState) -> Optional[LifecycleRecommendation]:
        """Check if we should take partial profits"""
        if not self.config.partial_close_enabled:
            return None
        
        # First partial close at 1R
        if position.partial_closes == 0 and position.current_r_multiple >= self.config.partial_close_trigger_r:
            return LifecycleRecommendation(
                action=LifecycleAction.PARTIAL_CLOSE,
                reason=f"Taking {self.config.partial_close_percent*100:.0f}% profit at {position.current_r_multiple:.2f}R",
                close_percent=self.config.partial_close_percent,
                urgency='normal',
                details={'partial_number': 1, 'trigger_r': self.config.partial_close_trigger_r}
            )
        
        # Second partial close at 2R
        if position.partial_closes == 1 and position.current_r_multiple >= self.config.second_partial_trigger_r:
            return LifecycleRecommendation(
                action=LifecycleAction.PARTIAL_CLOSE,
                reason=f"Taking additional {self.config.second_partial_percent*100:.0f}% profit at {position.current_r_multiple:.2f}R",
                close_percent=self.config.second_partial_percent,
                urgency='normal',
                details={'partial_number': 2, 'trigger_r': self.config.second_partial_trigger_r}
            )
        
        return None
    
    def _check_trailing_stop(self, position: PositionState) -> Optional[LifecycleRecommendation]:
        """Check if we should update trailing stop"""
        if not self.config.trailing_enabled:
            return None
        
        if position.current_r_multiple < self.config.trailing_trigger_r:
            return None
        
        # Determine trail distance based on regime
        if self.current_regime == 'trending':
            trail_distance_r = self.config.trending_trail_distance_r
        elif self.current_regime == 'ranging':
            trail_distance_r = self.config.ranging_trail_distance_r
        elif self.current_regime in ['volatile', 'high_volatility']:
            trail_distance_r = self.config.volatile_trail_distance_r
        else:
            trail_distance_r = self.config.trailing_distance_r
        
        # Calculate new trailing stop
        pip_multiplier = 100 if 'JPY' in position.symbol else 10000
        trail_pips = trail_distance_r * position.initial_risk_pips
        trail_distance = trail_pips / pip_multiplier
        
        if position.direction == 'long':
            new_stop = position.current_price - trail_distance
            # Only move stop up, never down
            if new_stop <= position.stop_loss:
                return None
            # Check minimum step
            step_pips = (new_stop - position.stop_loss) * pip_multiplier
            if step_pips < self.config.trailing_step_pips:
                return None
        else:
            new_stop = position.current_price + trail_distance
            # Only move stop down, never up
            if new_stop >= position.stop_loss:
                return None
            # Check minimum step
            step_pips = (position.stop_loss - new_stop) * pip_multiplier
            if step_pips < self.config.trailing_step_pips:
                return None
        
        return LifecycleRecommendation(
            action=LifecycleAction.TRAIL_STOP,
            reason=f"Trailing stop update at {position.current_r_multiple:.2f}R (regime: {self.current_regime})",
            new_stop_loss=new_stop,
            urgency='normal',
            details={
                'trail_distance_r': trail_distance_r,
                'regime': self.current_regime,
                'old_stop': position.stop_loss
            }
        )
    
    def _check_averaging(self, position: PositionState) -> Optional[LifecycleRecommendation]:
        """
        Check if we should average down (ADD TO LOSING POSITION).
        
        WARNING: This is where most retail traders blow up their accounts.
        Only enabled with strict conditions:
        1. Must be in correct regime (trending)
        2. Must have risk budget available
        3. Maximum 1 add allowed
        4. Only at specific R-multiple
        """
        if not self.config.averaging_enabled:
            return None
        
        if position.average_downs >= self.config.averaging_max_adds:
            return None
        
        # Only average in the required regime
        if self.current_regime != self.config.averaging_regime_required:
            return None
        
        # Check if at averaging trigger
        if position.current_r_multiple <= self.config.averaging_trigger_r:
            add_size = position.position_size * self.config.averaging_size_percent
            
            return LifecycleRecommendation(
                action=LifecycleAction.ADD_TO_POSITION,
                reason=f"Averaging opportunity at {position.current_r_multiple:.2f}R in {self.current_regime} regime",
                add_size=add_size,
                urgency='low',  # Low urgency - optional action
                details={
                    'regime': self.current_regime,
                    'current_adds': position.average_downs,
                    'max_adds': self.config.averaging_max_adds
                }
            )
        
        return None
    
    def mark_breakeven_moved(self, symbol: str, direction: str) -> None:
        """Mark that breakeven has been moved for a position"""
        key = f"{symbol}_{direction}"
        if key in self.positions:
            self.positions[key].breakeven_moved = True
    
    def mark_partial_close(self, symbol: str, direction: str) -> None:
        """Mark that a partial close has been executed"""
        key = f"{symbol}_{direction}"
        if key in self.positions:
            self.positions[key].partial_closes += 1
    
    def mark_trailing_active(self, symbol: str, direction: str) -> None:
        """Mark that trailing stop is now active"""
        key = f"{symbol}_{direction}"
        if key in self.positions:
            self.positions[key].trailing_active = True
    
    def mark_average_down(self, symbol: str, direction: str) -> None:
        """Mark that an average down has been executed"""
        key = f"{symbol}_{direction}"
        if key in self.positions:
            self.positions[key].average_downs += 1
    
    def close_position(self, symbol: str, direction: str, exit_reason: ExitReason, 
                       exit_price: float) -> Optional[Dict]:
        """Close a position and record the result"""
        key = f"{symbol}_{direction}"
        if key not in self.positions:
            return None
        
        position = self.positions[key]
        position.update_price(exit_price)
        
        result = {
            'symbol': symbol,
            'direction': direction,
            'entry_price': position.entry_price,
            'exit_price': exit_price,
            'entry_time': position.entry_time.isoformat(),
            'exit_time': datetime.utcnow().isoformat(),
            'exit_reason': exit_reason.value,
            'pnl_pips': position.current_pnl_pips,
            'r_multiple': position.current_r_multiple,
            'highest_r': position.highest_r,
            'partial_closes': position.partial_closes,
            'breakeven_moved': position.breakeven_moved,
            'trailing_active': position.trailing_active,
            'average_downs': position.average_downs
        }
        
        self.closed_positions.append(result)
        del self.positions[key]
        
        logger.info(f"[LIFECYCLE] Closed position: {key}, R={position.current_r_multiple:.2f}, "
                   f"reason={exit_reason.value}")
        
        return result
    
    def get_position_summary(self) -> Dict:
        """Get summary of all managed positions"""
        return {
            'active_positions': len(self.positions),
            'closed_positions': len(self.closed_positions),
            'positions': {
                key: {
                    'symbol': pos.symbol,
                    'direction': pos.direction,
                    'phase': pos.phase.value,
                    'r_multiple': pos.current_r_multiple,
                    'pnl_pips': pos.current_pnl_pips,
                    'breakeven_moved': pos.breakeven_moved,
                    'partial_closes': pos.partial_closes,
                    'trailing_active': pos.trailing_active
                }
                for key, pos in self.positions.items()
            }
        }


# Singleton instance
_lifecycle_manager: Optional[PositionLifecycleManager] = None


def get_lifecycle_manager() -> PositionLifecycleManager:
    """Get singleton lifecycle manager instance"""
    global _lifecycle_manager
    if _lifecycle_manager is None:
        _lifecycle_manager = PositionLifecycleManager()
    return _lifecycle_manager


def evaluate_position(symbol: str, direction: str, current_price: float,
                     entry_price: float, stop_loss: float, take_profit: float,
                     position_size: float, entry_time: datetime) -> Optional[LifecycleRecommendation]:
    """
    Convenience function to evaluate a position.
    
    Returns a recommendation for what action to take.
    """
    manager = get_lifecycle_manager()
    
    key = f"{symbol}_{direction}"
    if key not in manager.positions:
        # Register new position
        position = PositionState(
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            current_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=position_size,
            entry_time=entry_time
        )
        manager.register_position(position)
    
    return manager.update_position(symbol, direction, current_price)


def should_close_for_weekend() -> bool:
    """Check if positions should be closed for weekend"""
    now = datetime.utcnow()
    if now.weekday() == 4:  # Friday
        # Friday 8pm UTC (3pm EST) - 2 hours before close
        if now.hour >= 20:
            return True
    return False


def should_close_for_news(upcoming_events: List[Dict], minutes_threshold: int = 30) -> Tuple[bool, Optional[Dict]]:
    """Check if positions should be closed for upcoming news"""
    now = datetime.utcnow()
    
    for event in upcoming_events:
        event_time = event.get('time')
        if event_time:
            if isinstance(event_time, str):
                try:
                    event_time = datetime.fromisoformat(event_time.replace('Z', '+00:00'))
                except:
                    continue
            
            minutes_until = (event_time - now).total_seconds() / 60
            if 0 < minutes_until < minutes_threshold:
                impact = event.get('impact', 'medium')
                if impact in ['high', 'critical']:
                    return True, event
    
    return False, None
