"""
Cost-Aware Trade Gating Module
Filters trades based on spread, session, news events, and edge requirements
"""

import logging
from datetime import datetime, timedelta, time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class TradingSession(Enum):
    """Major forex trading sessions"""
    SYDNEY = "sydney"
    TOKYO = "tokyo"
    LONDON = "london"
    NEW_YORK = "new_york"
    OVERLAP_LONDON_NY = "london_ny_overlap"  # Best liquidity
    OFF_HOURS = "off_hours"


@dataclass
class TradeGateResult:
    """Result of trade gating check"""
    allowed: bool
    reason: str
    gate_type: str  # spread, session, news, edge, etc.
    details: Dict = None


class SpreadGate:
    """Gate trades based on spread thresholds"""
    
    # Maximum spread in pips for each pair (conservative thresholds)
    DEFAULT_SPREAD_LIMITS = {
        'EURUSD': 2.0,
        'GBPUSD': 3.0,
        'USDJPY': 2.0,
        'USDCHF': 2.5,
        'AUDUSD': 2.5,
        'USDCAD': 2.5,
        'NZDUSD': 3.0,
        'EURGBP': 2.5,
    }
    
    def __init__(self, spread_limits: Dict[str, float] = None):
        self.spread_limits = spread_limits or self.DEFAULT_SPREAD_LIMITS
    
    def check(self, symbol: str, spread_pips: float) -> TradeGateResult:
        """Check if spread is acceptable for trading"""
        max_spread = self.spread_limits.get(symbol, 3.0)
        
        if spread_pips > max_spread:
            return TradeGateResult(
                allowed=False,
                reason=f"Spread too high: {spread_pips:.1f} pips > {max_spread:.1f} max",
                gate_type="spread",
                details={'current_spread': spread_pips, 'max_spread': max_spread}
            )
        
        return TradeGateResult(
            allowed=True,
            reason=f"Spread acceptable: {spread_pips:.1f} pips",
            gate_type="spread",
            details={'current_spread': spread_pips, 'max_spread': max_spread}
        )


class SessionGate:
    """Gate trades based on trading session (liquidity)"""
    
    # Session times in UTC
    SESSIONS = {
        TradingSession.SYDNEY: (time(22, 0), time(7, 0)),      # 22:00-07:00 UTC
        TradingSession.TOKYO: (time(0, 0), time(9, 0)),        # 00:00-09:00 UTC
        TradingSession.LONDON: (time(8, 0), time(17, 0)),      # 08:00-17:00 UTC
        TradingSession.NEW_YORK: (time(13, 0), time(22, 0)),   # 13:00-22:00 UTC
        TradingSession.OVERLAP_LONDON_NY: (time(13, 0), time(17, 0)),  # 13:00-17:00 UTC (best)
    }
    
    def __init__(self, require_overlap: bool = False, allowed_sessions: List[TradingSession] = None):
        self.require_overlap = require_overlap
        self.allowed_sessions = allowed_sessions or [
            TradingSession.LONDON,
            TradingSession.NEW_YORK,
            TradingSession.OVERLAP_LONDON_NY
        ]
    
    def get_current_session(self, utc_time: datetime = None) -> TradingSession:
        """Determine current trading session"""
        if utc_time is None:
            utc_time = datetime.utcnow()
        
        current_time = utc_time.time()
        
        # Check for London/NY overlap first (highest priority)
        overlap_start, overlap_end = self.SESSIONS[TradingSession.OVERLAP_LONDON_NY]
        if overlap_start <= current_time <= overlap_end:
            return TradingSession.OVERLAP_LONDON_NY
        
        # Check other sessions
        for session, (start, end) in self.SESSIONS.items():
            if session == TradingSession.OVERLAP_LONDON_NY:
                continue
            
            # Handle sessions that cross midnight
            if start > end:
                if current_time >= start or current_time <= end:
                    return session
            else:
                if start <= current_time <= end:
                    return session
        
        return TradingSession.OFF_HOURS
    
    def check(self, utc_time: datetime = None) -> TradeGateResult:
        """Check if current session is suitable for trading"""
        current_session = self.get_current_session(utc_time)
        
        if self.require_overlap and current_session != TradingSession.OVERLAP_LONDON_NY:
            return TradeGateResult(
                allowed=False,
                reason=f"Not in London/NY overlap session (current: {current_session.value})",
                gate_type="session",
                details={'current_session': current_session.value, 'required': 'overlap'}
            )
        
        if current_session not in self.allowed_sessions:
            return TradeGateResult(
                allowed=False,
                reason=f"Session {current_session.value} not in allowed sessions",
                gate_type="session",
                details={'current_session': current_session.value, 'allowed': [s.value for s in self.allowed_sessions]}
            )
        
        return TradeGateResult(
            allowed=True,
            reason=f"Session {current_session.value} is suitable for trading",
            gate_type="session",
            details={'current_session': current_session.value}
        )


class NewsGate:
    """Gate trades around high-impact news events"""
    
    # High-impact events to avoid
    HIGH_IMPACT_EVENTS = [
        'NFP', 'Non-Farm Payrolls', 'FOMC', 'Fed', 'Interest Rate',
        'CPI', 'Inflation', 'GDP', 'ECB', 'BOE', 'BOJ', 'RBA',
        'Unemployment', 'Retail Sales', 'PMI'
    ]
    
    def __init__(self, blackout_minutes: int = 30, calendar_events: List[Dict] = None):
        self.blackout_minutes = blackout_minutes
        self.calendar_events = calendar_events or []
        self.last_update = None
    
    def update_calendar(self, events: List[Dict]):
        """Update calendar events"""
        self.calendar_events = events
        self.last_update = datetime.utcnow()
    
    def is_high_impact(self, event: Dict) -> bool:
        """Check if event is high impact"""
        title = event.get('title', '').upper()
        impact = event.get('impact', '').lower()
        
        if impact == 'high':
            return True
        
        for keyword in self.HIGH_IMPACT_EVENTS:
            if keyword.upper() in title:
                return True
        
        return False
    
    def check(self, symbol: str = None, utc_time: datetime = None) -> TradeGateResult:
        """Check if we're in a news blackout period"""
        if utc_time is None:
            utc_time = datetime.utcnow()
        
        blackout_start = utc_time - timedelta(minutes=self.blackout_minutes)
        blackout_end = utc_time + timedelta(minutes=self.blackout_minutes)
        
        for event in self.calendar_events:
            if not self.is_high_impact(event):
                continue
            
            event_time = event.get('datetime')
            if isinstance(event_time, str):
                try:
                    event_time = datetime.fromisoformat(event_time.replace('Z', '+00:00'))
                except:
                    continue
            
            if event_time and blackout_start <= event_time <= blackout_end:
                # Check if event affects the symbol's currency
                currencies = event.get('currencies', [])
                if symbol:
                    symbol_currencies = [symbol[:3], symbol[3:]]
                    if currencies and not any(c in symbol_currencies for c in currencies):
                        continue
                
                return TradeGateResult(
                    allowed=False,
                    reason=f"News blackout: {event.get('title', 'Unknown event')} at {event_time}",
                    gate_type="news",
                    details={'event': event.get('title'), 'event_time': str(event_time), 'blackout_minutes': self.blackout_minutes}
                )
        
        return TradeGateResult(
            allowed=True,
            reason="No high-impact news in blackout window",
            gate_type="news",
            details={'blackout_minutes': self.blackout_minutes}
        )


class EdgeGate:
    """Gate trades based on expected edge vs costs"""
    
    def __init__(self, min_edge_multiplier: float = 2.0, avg_slippage_pips: float = 0.5):
        self.min_edge_multiplier = min_edge_multiplier
        self.avg_slippage_pips = avg_slippage_pips
    
    def check(self, expected_profit_pips: float, spread_pips: float, 
              confidence: float = 0.5) -> TradeGateResult:
        """Check if expected edge exceeds costs sufficiently"""
        total_cost = spread_pips + self.avg_slippage_pips
        required_edge = total_cost * self.min_edge_multiplier
        
        # Adjust for confidence
        adjusted_profit = expected_profit_pips * confidence
        
        if adjusted_profit < required_edge:
            return TradeGateResult(
                allowed=False,
                reason=f"Edge too small: {adjusted_profit:.1f} pips < {required_edge:.1f} required",
                gate_type="edge",
                details={
                    'expected_profit': expected_profit_pips,
                    'confidence': confidence,
                    'adjusted_profit': adjusted_profit,
                    'total_cost': total_cost,
                    'required_edge': required_edge
                }
            )
        
        return TradeGateResult(
            allowed=True,
            reason=f"Edge sufficient: {adjusted_profit:.1f} pips >= {required_edge:.1f} required",
            gate_type="edge",
            details={
                'expected_profit': expected_profit_pips,
                'adjusted_profit': adjusted_profit,
                'required_edge': required_edge,
                'edge_ratio': adjusted_profit / total_cost if total_cost > 0 else 0
            }
        )


class DrawdownGate:
    """Gate trades based on account drawdown"""
    
    def __init__(self, max_drawdown_percent: float = 10.0, halt_drawdown_percent: float = 18.0):
        self.max_drawdown_percent = max_drawdown_percent
        self.halt_drawdown_percent = halt_drawdown_percent
    
    def check(self, current_balance: float, peak_balance: float) -> TradeGateResult:
        """Check if drawdown is acceptable"""
        if peak_balance <= 0:
            return TradeGateResult(allowed=True, reason="No peak balance recorded", gate_type="drawdown")
        
        drawdown_percent = ((peak_balance - current_balance) / peak_balance) * 100
        
        if drawdown_percent >= self.halt_drawdown_percent:
            return TradeGateResult(
                allowed=False,
                reason=f"HALT: Drawdown {drawdown_percent:.1f}% >= {self.halt_drawdown_percent}% halt threshold",
                gate_type="drawdown",
                details={'drawdown_percent': drawdown_percent, 'halt_threshold': self.halt_drawdown_percent}
            )
        
        if drawdown_percent >= self.max_drawdown_percent:
            return TradeGateResult(
                allowed=False,
                reason=f"Drawdown {drawdown_percent:.1f}% >= {self.max_drawdown_percent}% max threshold",
                gate_type="drawdown",
                details={'drawdown_percent': drawdown_percent, 'max_threshold': self.max_drawdown_percent}
            )
        
        return TradeGateResult(
            allowed=True,
            reason=f"Drawdown acceptable: {drawdown_percent:.1f}%",
            gate_type="drawdown",
            details={'drawdown_percent': drawdown_percent}
        )


class TradeGatingSystem:
    """Unified trade gating system combining all gates"""
    
    def __init__(self, 
                 spread_limits: Dict[str, float] = None,
                 require_session_overlap: bool = False,
                 news_blackout_minutes: int = 30,
                 min_edge_multiplier: float = 2.0,
                 max_drawdown_percent: float = 10.0):
        
        self.spread_gate = SpreadGate(spread_limits)
        self.session_gate = SessionGate(require_overlap=require_session_overlap)
        self.news_gate = NewsGate(blackout_minutes=news_blackout_minutes)
        self.edge_gate = EdgeGate(min_edge_multiplier=min_edge_multiplier)
        self.drawdown_gate = DrawdownGate(max_drawdown_percent=max_drawdown_percent)
        
        self.peak_balance = 0.0
        self.enabled_gates = {
            'spread': True,
            'session': True,
            'news': True,
            'edge': True,
            'drawdown': True
        }
    
    def update_peak_balance(self, balance: float):
        """Update peak balance for drawdown calculation"""
        if balance > self.peak_balance:
            self.peak_balance = balance
    
    def update_calendar(self, events: List[Dict]):
        """Update news calendar"""
        self.news_gate.update_calendar(events)
    
    def enable_gate(self, gate_name: str, enabled: bool = True):
        """Enable or disable a specific gate"""
        if gate_name in self.enabled_gates:
            self.enabled_gates[gate_name] = enabled
    
    def check_all(self, 
                  symbol: str,
                  spread_pips: float,
                  expected_profit_pips: float = 0,
                  confidence: float = 0.5,
                  current_balance: float = 0) -> Tuple[bool, List[TradeGateResult]]:
        """
        Run all enabled gates and return combined result
        Returns: (allowed, list of gate results)
        """
        results = []
        all_passed = True
        
        # Spread gate
        if self.enabled_gates.get('spread', True):
            result = self.spread_gate.check(symbol, spread_pips)
            results.append(result)
            if not result.allowed:
                all_passed = False
        
        # Session gate
        if self.enabled_gates.get('session', True):
            result = self.session_gate.check()
            results.append(result)
            if not result.allowed:
                all_passed = False
        
        # News gate
        if self.enabled_gates.get('news', True):
            result = self.news_gate.check(symbol)
            results.append(result)
            if not result.allowed:
                all_passed = False
        
        # Edge gate
        if self.enabled_gates.get('edge', True) and expected_profit_pips > 0:
            result = self.edge_gate.check(expected_profit_pips, spread_pips, confidence)
            results.append(result)
            if not result.allowed:
                all_passed = False
        
        # Drawdown gate
        if self.enabled_gates.get('drawdown', True) and current_balance > 0:
            result = self.drawdown_gate.check(current_balance, self.peak_balance)
            results.append(result)
            if not result.allowed:
                all_passed = False
        
        return all_passed, results
    
    def get_gate_summary(self, results: List[TradeGateResult]) -> str:
        """Get a summary of gate results"""
        passed = [r for r in results if r.allowed]
        failed = [r for r in results if not r.allowed]
        
        summary = f"Gates: {len(passed)}/{len(results)} passed"
        if failed:
            summary += f" | Failed: {', '.join(r.gate_type for r in failed)}"
        
        return summary


# Global instance
trade_gating = TradeGatingSystem()


def get_trade_gating() -> TradeGatingSystem:
    """Get the global trade gating system"""
    return trade_gating
