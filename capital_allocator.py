"""
Capital Allocator + Conviction Engine - Priority 2
Implements institutional-grade position sizing and portfolio construction like Aladdin.

This is the "Buffett Brain" - separates "opinions" (signals) from "bet sizing" (allocation).

Key Features:
- Position sizing via constrained optimization
- Dynamic correlation matrix across positions
- Risk budget per theme (USD exposure, carry, risk-on/off)
- Concentration vs diversification decisions
- Conviction scoring for bet sizing
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import json
import os
import threading

logger = logging.getLogger(__name__)


class AllocationStrategy(Enum):
    """Portfolio allocation strategies"""
    EQUAL_WEIGHT = "equal_weight"           # Simple equal allocation
    RISK_PARITY = "risk_parity"             # Equal risk contribution
    MEAN_VARIANCE = "mean_variance"         # Markowitz optimization
    KELLY = "kelly"                         # Kelly criterion
    CONVICTION_WEIGHTED = "conviction"      # Weight by conviction score


class RiskTheme(Enum):
    """Risk themes for budget allocation"""
    USD_LONG = "usd_long"           # Long USD exposure
    USD_SHORT = "usd_short"         # Short USD exposure
    RISK_ON = "risk_on"             # Risk-on trades (AUD, NZD, EM)
    RISK_OFF = "risk_off"           # Risk-off trades (JPY, CHF)
    CARRY = "carry"                 # Carry trades (high yield vs low yield)
    MOMENTUM = "momentum"           # Trend following
    MEAN_REVERSION = "mean_reversion"  # Counter-trend


@dataclass
class ConvictionScore:
    """Conviction score for a trade signal"""
    signal_strength: float = 0.0      # Raw signal strength (0-1)
    regime_alignment: float = 0.0     # How well signal aligns with regime (0-1)
    historical_edge: float = 0.0      # Historical performance of similar setups (0-1)
    macro_support: float = 0.0        # Macro factors supporting trade (0-1)
    technical_confluence: float = 0.0 # Multiple technical confirmations (0-1)
    sentiment_alignment: float = 0.0  # News/sentiment support (0-1)
    
    # Negative factors (reduce conviction)
    crowding_risk: float = 0.0        # How crowded is this trade (0-1)
    event_risk: float = 0.0           # Upcoming event risk (0-1)
    correlation_risk: float = 0.0     # Correlation with existing positions (0-1)
    
    def total_score(self) -> float:
        """Calculate total conviction score (0-100)"""
        positive = (
            self.signal_strength * 20 +
            self.regime_alignment * 15 +
            self.historical_edge * 20 +
            self.macro_support * 15 +
            self.technical_confluence * 15 +
            self.sentiment_alignment * 15
        )
        
        negative = (
            self.crowding_risk * 10 +
            self.event_risk * 15 +
            self.correlation_risk * 10
        )
        
        return max(0, min(100, positive - negative))
    
    def to_dict(self) -> Dict:
        return {
            'signal_strength': self.signal_strength,
            'regime_alignment': self.regime_alignment,
            'historical_edge': self.historical_edge,
            'macro_support': self.macro_support,
            'technical_confluence': self.technical_confluence,
            'sentiment_alignment': self.sentiment_alignment,
            'crowding_risk': self.crowding_risk,
            'event_risk': self.event_risk,
            'correlation_risk': self.correlation_risk,
            'total': self.total_score()
        }


@dataclass
class PositionAllocation:
    """Recommended position allocation"""
    symbol: str
    direction: str  # 'long' or 'short'
    
    # Sizing
    recommended_size: float           # In lots
    max_size: float                   # Maximum allowed
    min_size: float                   # Minimum viable
    
    # Risk
    risk_budget_percent: float        # % of account risk
    expected_loss: float              # Expected loss if stopped out
    position_risk_contribution: float # Contribution to portfolio risk
    
    # Conviction
    conviction_score: float           # 0-100
    conviction_tier: str              # 'high', 'medium', 'low'
    
    # Theme exposure
    risk_themes: List[RiskTheme] = field(default_factory=list)
    theme_exposure: Dict[str, float] = field(default_factory=dict)
    
    # Correlation
    correlation_with_portfolio: float = 0.0
    diversification_benefit: float = 0.0
    
    # Reasoning
    sizing_rationale: str = ""
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'direction': self.direction,
            'recommended_size': self.recommended_size,
            'max_size': self.max_size,
            'risk_budget_percent': self.risk_budget_percent,
            'conviction_score': self.conviction_score,
            'conviction_tier': self.conviction_tier,
            'correlation_with_portfolio': self.correlation_with_portfolio,
            'sizing_rationale': self.sizing_rationale,
            'warnings': self.warnings
        }


@dataclass
class PortfolioState:
    """Current portfolio state for allocation decisions"""
    total_equity: float = 0.0
    available_margin: float = 0.0
    used_margin: float = 0.0
    
    # Current positions
    open_positions: List[Dict] = field(default_factory=list)
    position_count: int = 0
    
    # Exposure by theme
    theme_exposure: Dict[str, float] = field(default_factory=dict)
    
    # Risk metrics
    total_risk_percent: float = 0.0
    max_position_risk: float = 0.0
    portfolio_var: float = 0.0
    
    # Correlation matrix
    correlation_matrix: Dict[str, Dict[str, float]] = field(default_factory=dict)


class ConvictionEngine:
    """
    Calculates conviction scores for trade signals.
    Higher conviction = larger position size.
    """
    
    # Currency pair characteristics
    PAIR_THEMES = {
        'EURUSD': [RiskTheme.USD_SHORT],
        'GBPUSD': [RiskTheme.USD_SHORT],
        'USDJPY': [RiskTheme.USD_LONG, RiskTheme.RISK_ON],
        'USDCHF': [RiskTheme.USD_LONG, RiskTheme.RISK_OFF],
        'AUDUSD': [RiskTheme.USD_SHORT, RiskTheme.RISK_ON, RiskTheme.CARRY],
        'USDCAD': [RiskTheme.USD_LONG],
        'NZDUSD': [RiskTheme.USD_SHORT, RiskTheme.RISK_ON, RiskTheme.CARRY],
        'EURGBP': [],
        'EURJPY': [RiskTheme.RISK_ON, RiskTheme.CARRY],
        'GBPJPY': [RiskTheme.RISK_ON, RiskTheme.CARRY],
    }
    
    def __init__(self):
        self.historical_performance: Dict[str, Dict] = {}
        self._load_history()
        logger.info("ConvictionEngine initialized")
    
    def _load_history(self):
        """Load historical performance data"""
        history_file = "data/conviction_history.json"
        if os.path.exists(history_file):
            try:
                with open(history_file, 'r') as f:
                    self.historical_performance = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load conviction history: {e}")
    
    def _save_history(self):
        """Save historical performance"""
        os.makedirs("data", exist_ok=True)
        try:
            with open("data/conviction_history.json", 'w') as f:
                json.dump(self.historical_performance, f)
        except Exception as e:
            logger.warning(f"Could not save conviction history: {e}")
    
    def calculate_conviction(self, signal: Dict, market_context: Dict,
                            portfolio_state: PortfolioState) -> ConvictionScore:
        """
        Calculate conviction score for a trade signal.
        
        Args:
            signal: Trade signal with direction, confidence, strategy, etc.
            market_context: Current market conditions (regime, volatility, etc.)
            portfolio_state: Current portfolio state
        
        Returns:
            ConvictionScore with all components
        """
        conviction = ConvictionScore()
        
        symbol = signal.get('symbol', '')
        direction = signal.get('direction', '')
        strategy = signal.get('strategy', '')
        confidence = signal.get('confidence', 0.5)
        
        # 1. Signal strength (from raw confidence)
        conviction.signal_strength = min(1.0, confidence)
        
        # 2. Regime alignment
        regime = market_context.get('regime', 'unknown')
        conviction.regime_alignment = self._calculate_regime_alignment(
            direction, regime, strategy
        )
        
        # 3. Historical edge
        conviction.historical_edge = self._calculate_historical_edge(
            symbol, direction, strategy, regime
        )
        
        # 4. Macro support
        conviction.macro_support = self._calculate_macro_support(
            symbol, direction, market_context
        )
        
        # 5. Technical confluence
        conviction.technical_confluence = self._calculate_technical_confluence(signal)
        
        # 6. Sentiment alignment
        conviction.sentiment_alignment = self._calculate_sentiment_alignment(
            symbol, direction, market_context
        )
        
        # 7. Crowding risk
        conviction.crowding_risk = self._calculate_crowding_risk(
            symbol, direction, market_context
        )
        
        # 8. Event risk
        conviction.event_risk = self._calculate_event_risk(symbol, market_context)
        
        # 9. Correlation risk
        conviction.correlation_risk = self._calculate_correlation_risk(
            symbol, direction, portfolio_state
        )
        
        return conviction
    
    def _calculate_regime_alignment(self, direction: str, regime: str, 
                                    strategy: str) -> float:
        """Calculate how well trade aligns with current regime"""
        alignment = 0.5  # Neutral default
        
        # Trend following in trending regime
        if strategy in ['trend_following', 'momentum', 'breakout']:
            if regime in ['trending_up', 'trending_down', 'strong_trend']:
                alignment = 0.9
            elif regime in ['ranging', 'choppy']:
                alignment = 0.3
        
        # Mean reversion in ranging regime
        elif strategy in ['mean_reversion', 'range_trading']:
            if regime in ['ranging', 'consolidation']:
                alignment = 0.9
            elif regime in ['trending_up', 'trending_down']:
                alignment = 0.3
        
        # Breakout strategy
        elif strategy in ['breakout', 'fvg']:
            if regime in ['breakout', 'high_volatility']:
                alignment = 0.85
            elif regime in ['low_volatility']:
                alignment = 0.4
        
        return alignment
    
    def _calculate_historical_edge(self, symbol: str, direction: str,
                                   strategy: str, regime: str) -> float:
        """Calculate edge based on historical performance"""
        key = f"{symbol}_{direction}_{strategy}_{regime}"
        
        if key in self.historical_performance:
            perf = self.historical_performance[key]
            win_rate = perf.get('win_rate', 0.5)
            sample_size = perf.get('trades', 0)
            
            # Confidence increases with sample size
            confidence_factor = min(1.0, sample_size / 20)
            
            # Convert win rate to edge score
            if win_rate > 0.6:
                edge = 0.8 + (win_rate - 0.6) * 0.5
            elif win_rate > 0.5:
                edge = 0.5 + (win_rate - 0.5) * 3
            else:
                edge = win_rate
            
            return edge * confidence_factor
        
        return 0.5  # Neutral if no history
    
    def _calculate_macro_support(self, symbol: str, direction: str,
                                 market_context: Dict) -> float:
        """Calculate macro factor support"""
        support = 0.5
        
        # Get macro data
        dxy_trend = market_context.get('dxy_trend', 'neutral')
        vix_level = market_context.get('vix_level', 'normal')
        yield_curve = market_context.get('yield_curve', 'normal')
        
        # USD pairs
        if 'USD' in symbol:
            is_usd_long = (symbol.startswith('USD') and direction == 'long') or \
                         (symbol.endswith('USD') and direction == 'short')
            
            if dxy_trend == 'up' and is_usd_long:
                support += 0.2
            elif dxy_trend == 'down' and not is_usd_long:
                support += 0.2
            elif dxy_trend == 'up' and not is_usd_long:
                support -= 0.15
            elif dxy_trend == 'down' and is_usd_long:
                support -= 0.15
        
        # Risk sentiment
        themes = self.PAIR_THEMES.get(symbol, [])
        
        if RiskTheme.RISK_ON in themes:
            if vix_level == 'low':
                support += 0.15
            elif vix_level == 'high':
                support -= 0.2
        
        if RiskTheme.RISK_OFF in themes:
            if vix_level == 'high':
                support += 0.15
            elif vix_level == 'low':
                support -= 0.1
        
        return max(0, min(1, support))
    
    def _calculate_technical_confluence(self, signal: Dict) -> float:
        """Calculate technical indicator confluence"""
        confluence = 0.0
        factors = 0
        
        # RSI alignment
        rsi = signal.get('rsi', 50)
        direction = signal.get('direction', '')
        
        if direction == 'long' and rsi < 40:
            confluence += 0.2
            factors += 1
        elif direction == 'short' and rsi > 60:
            confluence += 0.2
            factors += 1
        
        # MACD alignment
        macd_signal = signal.get('macd_signal', 0)
        if (direction == 'long' and macd_signal > 0) or \
           (direction == 'short' and macd_signal < 0):
            confluence += 0.2
            factors += 1
        
        # ADX (trend strength)
        adx = signal.get('adx', 25)
        if adx > 25:
            confluence += 0.15
            factors += 1
        
        # MTF alignment
        mtf_alignment = signal.get('mtf_alignment', 0.5)
        if mtf_alignment > 0.6:
            confluence += 0.25
            factors += 1
        
        # FVG or liquidity sweep
        if signal.get('fvg_signal') or signal.get('liquidity_sweep'):
            confluence += 0.2
            factors += 1
        
        return min(1.0, confluence)
    
    def _calculate_sentiment_alignment(self, symbol: str, direction: str,
                                       market_context: Dict) -> float:
        """Calculate news/sentiment alignment"""
        sentiment = market_context.get('sentiment', {})
        
        if not sentiment:
            return 0.5
        
        pair_sentiment = sentiment.get(symbol, 0)  # -1 to 1
        
        if direction == 'long' and pair_sentiment > 0.2:
            return 0.5 + pair_sentiment * 0.5
        elif direction == 'short' and pair_sentiment < -0.2:
            return 0.5 + abs(pair_sentiment) * 0.5
        elif (direction == 'long' and pair_sentiment < -0.3) or \
             (direction == 'short' and pair_sentiment > 0.3):
            return 0.3  # Contrarian - lower conviction
        
        return 0.5
    
    def _calculate_crowding_risk(self, symbol: str, direction: str,
                                 market_context: Dict) -> float:
        """Calculate crowding/positioning risk"""
        positioning = market_context.get('positioning', {})
        
        if not positioning:
            return 0.3  # Assume moderate crowding
        
        pair_positioning = positioning.get(symbol, 0)  # -1 to 1 (net long/short)
        
        # If everyone is already positioned our way, crowding is high
        if direction == 'long' and pair_positioning > 0.5:
            return min(1.0, pair_positioning)
        elif direction == 'short' and pair_positioning < -0.5:
            return min(1.0, abs(pair_positioning))
        
        return 0.2  # Low crowding
    
    def _calculate_event_risk(self, symbol: str, market_context: Dict) -> float:
        """Calculate upcoming event risk"""
        events = market_context.get('upcoming_events', [])
        
        if not events:
            return 0.1
        
        # Check for high-impact events in next 24 hours
        high_impact_events = [e for e in events if e.get('impact', '') == 'high']
        
        # Check if events affect this pair
        currencies = [symbol[:3], symbol[3:6]] if len(symbol) >= 6 else []
        
        relevant_events = [e for e in high_impact_events 
                         if e.get('currency', '') in currencies]
        
        if relevant_events:
            # High event risk
            hours_until = min(e.get('hours_until', 24) for e in relevant_events)
            if hours_until < 2:
                return 0.9
            elif hours_until < 6:
                return 0.7
            elif hours_until < 24:
                return 0.5
        
        return 0.2
    
    def _calculate_correlation_risk(self, symbol: str, direction: str,
                                    portfolio_state: PortfolioState) -> float:
        """Calculate correlation with existing positions"""
        if not portfolio_state.open_positions:
            return 0.0  # No correlation risk if no positions
        
        correlations = portfolio_state.correlation_matrix.get(symbol, {})
        
        if not correlations:
            return 0.3  # Assume moderate correlation
        
        max_correlation = 0.0
        
        for pos in portfolio_state.open_positions:
            pos_symbol = pos.get('symbol', '')
            pos_direction = pos.get('direction', '')
            
            corr = correlations.get(pos_symbol, 0.5)
            
            # Same direction = correlation adds risk
            # Opposite direction = correlation reduces risk
            if pos_direction == direction:
                effective_corr = corr
            else:
                effective_corr = -corr
            
            max_correlation = max(max_correlation, effective_corr)
        
        return max(0, min(1, max_correlation))
    
    def update_historical_performance(self, symbol: str, direction: str,
                                      strategy: str, regime: str,
                                      won: bool, pnl: float):
        """Update historical performance for learning"""
        key = f"{symbol}_{direction}_{strategy}_{regime}"
        
        if key not in self.historical_performance:
            self.historical_performance[key] = {
                'trades': 0,
                'wins': 0,
                'total_pnl': 0,
                'win_rate': 0.5
            }
        
        perf = self.historical_performance[key]
        perf['trades'] += 1
        if won:
            perf['wins'] += 1
        perf['total_pnl'] += pnl
        perf['win_rate'] = perf['wins'] / perf['trades']
        
        self._save_history()


class CapitalAllocator:
    """
    Institutional-grade capital allocation engine.
    Decides HOW MUCH to bet, not WHAT to bet on.
    
    This is the Buffett/Aladdin approach:
    - Signals are "opinions"
    - Allocation is "how much do we bet"
    - Risk budgets constrain total exposure
    - Correlation matters for portfolio construction
    """
    
    # FX pair correlations (approximate)
    PAIR_CORRELATIONS = {
        ('EURUSD', 'GBPUSD'): 0.85,
        ('EURUSD', 'USDCHF'): -0.90,
        ('EURUSD', 'USDJPY'): -0.30,
        ('EURUSD', 'AUDUSD'): 0.70,
        ('GBPUSD', 'USDCHF'): -0.80,
        ('GBPUSD', 'USDJPY'): -0.25,
        ('USDJPY', 'USDCHF'): 0.60,
        ('AUDUSD', 'NZDUSD'): 0.90,
        ('AUDUSD', 'USDJPY'): 0.50,
        ('EURJPY', 'GBPJPY'): 0.90,
    }
    
    # Risk budget limits by theme
    THEME_LIMITS = {
        RiskTheme.USD_LONG: 0.40,      # Max 40% of risk in USD long
        RiskTheme.USD_SHORT: 0.40,
        RiskTheme.RISK_ON: 0.35,
        RiskTheme.RISK_OFF: 0.35,
        RiskTheme.CARRY: 0.30,
        RiskTheme.MOMENTUM: 0.50,
        RiskTheme.MEAN_REVERSION: 0.30,
    }
    
    def __init__(self, 
                 max_portfolio_risk: float = 0.05,
                 max_position_risk: float = 0.02,
                 max_positions: int = 6,
                 strategy: AllocationStrategy = AllocationStrategy.CONVICTION_WEIGHTED):
        """
        Initialize allocator.
        
        Args:
            max_portfolio_risk: Maximum total portfolio risk (default 5%)
            max_position_risk: Maximum single position risk (default 2%)
            max_positions: Maximum concurrent positions
            strategy: Allocation strategy to use
        """
        self.max_portfolio_risk = max_portfolio_risk
        self.max_position_risk = max_position_risk
        self.max_positions = max_positions
        self.strategy = strategy
        
        self.conviction_engine = ConvictionEngine()
        
        # Current state
        self.portfolio_state = PortfolioState()
        self._lock = threading.Lock()
        
        # Build correlation matrix
        self._build_correlation_matrix()
        
        logger.info(f"CapitalAllocator initialized - Strategy: {strategy.value}, "
                   f"Max risk: {max_portfolio_risk:.1%}")
    
    def _build_correlation_matrix(self):
        """Build full correlation matrix from pair correlations"""
        pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 
                'USDCAD', 'NZDUSD', 'EURGBP', 'EURJPY', 'GBPJPY']
        
        matrix = {}
        for p1 in pairs:
            matrix[p1] = {}
            for p2 in pairs:
                if p1 == p2:
                    matrix[p1][p2] = 1.0
                else:
                    # Look up correlation
                    key = (p1, p2) if (p1, p2) in self.PAIR_CORRELATIONS else (p2, p1)
                    matrix[p1][p2] = self.PAIR_CORRELATIONS.get(key, 0.3)
        
        self.portfolio_state.correlation_matrix = matrix
    
    def update_portfolio_state(self, equity: float, margin_used: float,
                               open_positions: List[Dict]):
        """Update current portfolio state"""
        with self._lock:
            self.portfolio_state.total_equity = equity
            self.portfolio_state.used_margin = margin_used
            self.portfolio_state.available_margin = equity - margin_used
            self.portfolio_state.open_positions = open_positions
            self.portfolio_state.position_count = len(open_positions)
            
            # Calculate theme exposure
            self._calculate_theme_exposure()
            
            # Calculate total risk
            self._calculate_portfolio_risk()
    
    def _calculate_theme_exposure(self):
        """Calculate current exposure by risk theme"""
        theme_exposure = {theme.value: 0.0 for theme in RiskTheme}
        
        for pos in self.portfolio_state.open_positions:
            symbol = pos.get('symbol', '')
            direction = pos.get('direction', '')
            risk = pos.get('risk_percent', 0.01)
            
            themes = ConvictionEngine.PAIR_THEMES.get(symbol, [])
            
            for theme in themes:
                # Adjust for direction
                if theme in [RiskTheme.USD_LONG, RiskTheme.USD_SHORT]:
                    is_usd_long = (symbol.startswith('USD') and direction == 'long') or \
                                 (symbol.endswith('USD') and direction == 'short')
                    if is_usd_long:
                        theme_exposure[RiskTheme.USD_LONG.value] += risk
                    else:
                        theme_exposure[RiskTheme.USD_SHORT.value] += risk
                else:
                    theme_exposure[theme.value] += risk
        
        self.portfolio_state.theme_exposure = theme_exposure
    
    def _calculate_portfolio_risk(self):
        """Calculate total portfolio risk considering correlations"""
        if not self.portfolio_state.open_positions:
            self.portfolio_state.total_risk_percent = 0.0
            return
        
        # Simple sum of individual risks (conservative)
        total_risk = sum(p.get('risk_percent', 0.01) 
                        for p in self.portfolio_state.open_positions)
        
        # Adjust for correlations (diversification benefit)
        # This is a simplified version - full implementation would use covariance matrix
        n = len(self.portfolio_state.open_positions)
        if n > 1:
            avg_correlation = 0.5  # Assume moderate correlation
            diversification_factor = np.sqrt(1 + (n - 1) * avg_correlation) / n
            total_risk *= diversification_factor
        
        self.portfolio_state.total_risk_percent = total_risk
        self.portfolio_state.max_position_risk = max(
            (p.get('risk_percent', 0) for p in self.portfolio_state.open_positions),
            default=0
        )
    
    def calculate_allocation(self, signal: Dict, market_context: Dict,
                            account_balance: float) -> PositionAllocation:
        """
        Calculate optimal position allocation for a signal.
        
        This is the core function - decides HOW MUCH to bet.
        
        Args:
            signal: Trade signal with symbol, direction, confidence, etc.
            market_context: Current market conditions
            account_balance: Current account balance
        
        Returns:
            PositionAllocation with sizing recommendation
        """
        symbol = signal.get('symbol', '')
        direction = signal.get('direction', 'long')
        sl_pips = signal.get('sl_pips', 30)
        
        # Calculate conviction score
        conviction = self.conviction_engine.calculate_conviction(
            signal, market_context, self.portfolio_state
        )
        conviction_score = conviction.total_score()
        
        # Determine conviction tier
        if conviction_score >= 70:
            conviction_tier = 'high'
        elif conviction_score >= 50:
            conviction_tier = 'medium'
        else:
            conviction_tier = 'low'
        
        # Check constraints
        warnings = []
        
        # 1. Position count limit
        if self.portfolio_state.position_count >= self.max_positions:
            warnings.append(f"At max positions ({self.max_positions})")
            return self._create_zero_allocation(symbol, direction, warnings)
        
        # 2. Portfolio risk limit
        remaining_risk = self.max_portfolio_risk - self.portfolio_state.total_risk_percent
        if remaining_risk <= 0.001:
            warnings.append("Portfolio risk budget exhausted")
            return self._create_zero_allocation(symbol, direction, warnings)
        
        # 3. Theme exposure limits
        themes = ConvictionEngine.PAIR_THEMES.get(symbol, [])
        for theme in themes:
            current_exposure = self.portfolio_state.theme_exposure.get(theme.value, 0)
            limit = self.THEME_LIMITS.get(theme, 0.5)
            if current_exposure >= limit:
                warnings.append(f"Theme {theme.value} at limit ({limit:.0%})")
        
        # 4. Correlation check
        correlation_risk = conviction.correlation_risk
        if correlation_risk > 0.7:
            warnings.append(f"High correlation with existing positions ({correlation_risk:.0%})")
        
        # Calculate base risk budget
        base_risk = self._calculate_base_risk(conviction_score, conviction_tier)
        
        # Apply constraints
        risk_budget = min(
            base_risk,
            self.max_position_risk,
            remaining_risk,
            self.max_portfolio_risk / self.max_positions
        )
        
        # Reduce for high correlation
        if correlation_risk > 0.5:
            risk_budget *= (1 - correlation_risk * 0.5)
        
        # Calculate position size
        position_size = self._calculate_position_size(
            risk_budget, sl_pips, account_balance
        )
        
        # Calculate expected loss
        expected_loss = account_balance * risk_budget
        
        # Calculate correlation with portfolio
        portfolio_correlation = self._calculate_portfolio_correlation(symbol, direction)
        diversification_benefit = 1 - portfolio_correlation
        
        # Generate rationale
        rationale = self._generate_sizing_rationale(
            conviction_score, conviction_tier, risk_budget, 
            position_size, warnings
        )
        
        allocation = PositionAllocation(
            symbol=symbol,
            direction=direction,
            recommended_size=position_size,
            max_size=position_size * 1.5,
            min_size=0.01,
            risk_budget_percent=risk_budget,
            expected_loss=expected_loss,
            position_risk_contribution=risk_budget / self.max_portfolio_risk,
            conviction_score=conviction_score,
            conviction_tier=conviction_tier,
            risk_themes=themes,
            theme_exposure={t.value: self.portfolio_state.theme_exposure.get(t.value, 0) 
                           for t in themes},
            correlation_with_portfolio=portfolio_correlation,
            diversification_benefit=diversification_benefit,
            sizing_rationale=rationale,
            warnings=warnings
        )
        
        return allocation
    
    def _calculate_base_risk(self, conviction_score: float, 
                            conviction_tier: str) -> float:
        """Calculate base risk budget from conviction"""
        if self.strategy == AllocationStrategy.CONVICTION_WEIGHTED:
            # High conviction = larger position
            if conviction_tier == 'high':
                return self.max_position_risk * 0.9
            elif conviction_tier == 'medium':
                return self.max_position_risk * 0.6
            else:
                return self.max_position_risk * 0.3
        
        elif self.strategy == AllocationStrategy.EQUAL_WEIGHT:
            return self.max_portfolio_risk / self.max_positions
        
        elif self.strategy == AllocationStrategy.KELLY:
            # Simplified Kelly: f = (bp - q) / b
            # where b = win/loss ratio, p = win prob, q = 1-p
            win_prob = conviction_score / 100
            win_loss_ratio = 2.0  # Assume 2:1 R:R
            
            kelly = (win_loss_ratio * win_prob - (1 - win_prob)) / win_loss_ratio
            kelly = max(0, kelly)
            
            # Use half-Kelly for safety
            return min(self.max_position_risk, kelly * 0.5)
        
        else:
            return self.max_position_risk * 0.5
    
    def _calculate_position_size(self, risk_budget: float, sl_pips: float,
                                 account_balance: float) -> float:
        """Calculate position size in lots"""
        risk_amount = account_balance * risk_budget
        
        # Pip value per standard lot (approximately $10 for most USD pairs)
        pip_value_per_lot = 10.0
        
        if sl_pips > 0:
            position_size = risk_amount / (sl_pips * pip_value_per_lot)
        else:
            position_size = 0.01
        
        # Clamp to reasonable limits
        max_lots = min(1.0, account_balance / 100)
        position_size = max(0.01, min(position_size, max_lots))
        
        return round(position_size, 2)
    
    def _calculate_portfolio_correlation(self, symbol: str, direction: str) -> float:
        """Calculate correlation of new position with existing portfolio"""
        if not self.portfolio_state.open_positions:
            return 0.0
        
        correlations = []
        for pos in self.portfolio_state.open_positions:
            pos_symbol = pos.get('symbol', '')
            pos_direction = pos.get('direction', '')
            
            corr = self.portfolio_state.correlation_matrix.get(symbol, {}).get(pos_symbol, 0.3)
            
            # Same direction = positive correlation contribution
            # Opposite direction = negative correlation contribution
            if pos_direction == direction:
                correlations.append(corr)
            else:
                correlations.append(-corr)
        
        return np.mean(correlations) if correlations else 0.0
    
    def _generate_sizing_rationale(self, conviction: float, tier: str,
                                   risk: float, size: float,
                                   warnings: List[str]) -> str:
        """Generate human-readable sizing rationale"""
        rationale = f"Conviction: {conviction:.0f}/100 ({tier}). "
        rationale += f"Risk budget: {risk:.2%} of account. "
        rationale += f"Position size: {size:.2f} lots. "
        
        if warnings:
            rationale += f"Warnings: {', '.join(warnings)}. "
        
        if tier == 'high':
            rationale += "High conviction setup - sizing at upper range."
        elif tier == 'medium':
            rationale += "Moderate conviction - standard sizing."
        else:
            rationale += "Low conviction - reduced sizing for risk management."
        
        return rationale
    
    def _create_zero_allocation(self, symbol: str, direction: str,
                               warnings: List[str]) -> PositionAllocation:
        """Create zero allocation when trade should be skipped"""
        return PositionAllocation(
            symbol=symbol,
            direction=direction,
            recommended_size=0.0,
            max_size=0.0,
            min_size=0.0,
            risk_budget_percent=0.0,
            expected_loss=0.0,
            position_risk_contribution=0.0,
            conviction_score=0.0,
            conviction_tier='none',
            sizing_rationale="Trade skipped due to constraints",
            warnings=warnings
        )
    
    def should_take_trade(self, allocation: PositionAllocation) -> Tuple[bool, str]:
        """
        Final decision on whether to take the trade.
        Returns (should_take, reason)
        """
        if allocation.recommended_size <= 0:
            return False, "Zero allocation - " + (allocation.warnings[0] if allocation.warnings else "constraints not met")
        
        if allocation.conviction_tier == 'low' and allocation.warnings:
            return False, f"Low conviction with warnings: {allocation.warnings[0]}"
        
        if allocation.correlation_with_portfolio > 0.8:
            return False, f"Too correlated with existing positions ({allocation.correlation_with_portfolio:.0%})"
        
        return True, f"Trade approved - {allocation.conviction_tier} conviction, {allocation.recommended_size:.2f} lots"
    
    def get_portfolio_summary(self) -> str:
        """Generate portfolio summary"""
        state = self.portfolio_state
        
        summary = f"""
=== PORTFOLIO ALLOCATION SUMMARY ===
Equity: ${state.total_equity:.2f}
Positions: {state.position_count}/{self.max_positions}
Total Risk: {state.total_risk_percent:.2%} / {self.max_portfolio_risk:.2%}
Available Risk Budget: {(self.max_portfolio_risk - state.total_risk_percent):.2%}

Theme Exposure:
"""
        for theme, exposure in state.theme_exposure.items():
            limit = self.THEME_LIMITS.get(RiskTheme(theme), 0.5) if theme in [t.value for t in RiskTheme] else 0.5
            summary += f"  {theme}: {exposure:.1%} / {limit:.0%}\n"
        
        return summary


# Singleton instances
_conviction_engine: Optional[ConvictionEngine] = None
_capital_allocator: Optional[CapitalAllocator] = None


def get_conviction_engine() -> ConvictionEngine:
    """Get singleton conviction engine"""
    global _conviction_engine
    if _conviction_engine is None:
        _conviction_engine = ConvictionEngine()
    return _conviction_engine


def get_capital_allocator() -> CapitalAllocator:
    """Get singleton capital allocator"""
    global _capital_allocator
    if _capital_allocator is None:
        _capital_allocator = CapitalAllocator()
    return _capital_allocator


def calculate_position_allocation(signal: Dict, market_context: Dict,
                                  account_balance: float) -> PositionAllocation:
    """Convenience function to calculate allocation"""
    allocator = get_capital_allocator()
    return allocator.calculate_allocation(signal, market_context, account_balance)


def update_portfolio(equity: float, margin_used: float, 
                    open_positions: List[Dict]):
    """Update portfolio state"""
    allocator = get_capital_allocator()
    allocator.update_portfolio_state(equity, margin_used, open_positions)


def should_take_trade(signal: Dict, market_context: Dict,
                     account_balance: float) -> Tuple[bool, str, PositionAllocation]:
    """
    Complete trade decision including allocation.
    Returns (should_take, reason, allocation)
    """
    allocator = get_capital_allocator()
    allocation = allocator.calculate_allocation(signal, market_context, account_balance)
    should_take, reason = allocator.should_take_trade(allocation)
    return should_take, reason, allocation
