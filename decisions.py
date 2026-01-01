"""
Trading Decisions Module - Veteran Trader Logic
Implements FVG detection, dynamic leverage, trailing SL, aggression levels,
and multi-timeframe analysis for human-like trading decisions
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging

from config import config, INDICATOR_SETTINGS
from indicators import TechnicalIndicators, FeatureEngineer
from regime_detection import regime_manager, MarketRegime
from agentic import agentic_system

try:
    from pattern_miner import pattern_miner, PatternMiner
    PATTERN_MINER_AVAILABLE = True
except ImportError:
    PATTERN_MINER_AVAILABLE = False
    pattern_miner = None

try:
    from bedrock_ai import bedrock_ai, BedrockAI
    BEDROCK_AVAILABLE = True
except ImportError:
    BEDROCK_AVAILABLE = False
    bedrock_ai = None

try:
    from trade_gating import get_trade_gating, TradeGatingSystem
    TRADE_GATING_AVAILABLE = True
except ImportError:
    TRADE_GATING_AVAILABLE = False

try:
    from data import MacroDataFetcher
    MACRO_DATA_AVAILABLE = True
except ImportError:
    MACRO_DATA_AVAILABLE = False
    MacroDataFetcher = None

try:
    from advanced_knowledge import (
        get_advanced_knowledge, 
        get_trading_context,
        should_avoid_trading as advanced_should_avoid,
        AdvancedKnowledgeEngine
    )
    ADVANCED_KNOWLEDGE_AVAILABLE = True
except ImportError:
    ADVANCED_KNOWLEDGE_AVAILABLE = False
    get_advanced_knowledge = None
    get_trading_context = None
    advanced_should_avoid = None

try:
    from adaptive_learning import (
        get_adaptive_learning,
        record_trade_outcome,
        AdaptiveLearningEngine,
        TradeOutcome
    )
    ADAPTIVE_LEARNING_AVAILABLE = True
except ImportError:
    ADAPTIVE_LEARNING_AVAILABLE = False
    get_adaptive_learning = None
    record_trade_outcome = None

try:
    from trade_journaling import (
        get_journaling_engine,
        evaluate_trade,
        TradeJournalingEngine,
        UncertaintyLevel,
        TradeOutcomeType
    )
    TRADE_JOURNALING_AVAILABLE = True
except ImportError:
    TRADE_JOURNALING_AVAILABLE = False
    get_journaling_engine = None
    evaluate_trade = None

try:
    from event_log import (
        get_event_system,
        log_decision_cycle,
        log_market_snapshot,
        log_signal,
        log_confidence_adjustment,
        log_gate_check,
        log_trade_decision,
        end_decision_cycle,
        MarketSnapshot,
        SignalEvent,
        ConfidenceAdjustment,
        GateCheckResult,
        InstitutionalEventSystem
    )
    EVENT_LOG_AVAILABLE = True
except ImportError:
    EVENT_LOG_AVAILABLE = False
    get_event_system = None
    log_decision_cycle = None

# Tier 2: Portfolio Risk Management
try:
    from portfolio_risk import (
        get_portfolio_risk_manager,
        check_trade_risk,
        get_portfolio_risk_state,
        is_kill_switch_active,
        PortfolioRiskManager,
        RiskLevel
    )
    PORTFOLIO_RISK_AVAILABLE = True
except ImportError:
    PORTFOLIO_RISK_AVAILABLE = False
    get_portfolio_risk_manager = None
    check_trade_risk = None
    is_kill_switch_active = None

# Tier 3: Research/Production Separation
try:
    from research_production import (
        get_research_production_manager,
        is_shadow_mode,
        process_shadow_signal,
        update_shadow_prices,
        ResearchProductionManager
    )
    RESEARCH_PRODUCTION_AVAILABLE = True
except ImportError:
    RESEARCH_PRODUCTION_AVAILABLE = False
    get_research_production_manager = None
    is_shadow_mode = None
    process_shadow_signal = None

logger = logging.getLogger(__name__)


class SignalStrength(Enum):
    VERY_WEAK = 1
    WEAK = 2
    MODERATE = 3
    STRONG = 4
    VERY_STRONG = 5


class TradeDirection(Enum):
    LONG = 1
    SHORT = -1
    NEUTRAL = 0


@dataclass
class FairValueGap:
    """Fair Value Gap (FVG) / Imbalance detection"""
    timestamp: datetime
    direction: TradeDirection
    high: float
    low: float
    size: float
    filled: bool = False
    fill_percentage: float = 0.0
    strength: SignalStrength = SignalStrength.MODERATE


@dataclass
class LiquiditySweep:
    """Liquidity sweep detection"""
    timestamp: datetime
    direction: TradeDirection
    sweep_level: float
    reversal_confirmed: bool = False
    strength: SignalStrength = SignalStrength.MODERATE


@dataclass
class TradingSignal:
    """Complete trading signal with all analysis"""
    symbol: str
    direction: TradeDirection
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    leverage: int
    
    # Analysis components
    fvg_signal: Optional[FairValueGap] = None
    liquidity_sweep: Optional[LiquiditySweep] = None
    regime: Optional[MarketRegime] = None
    
    # Indicator values
    rsi: float = 50.0
    adx: float = 25.0
    atr: float = 0.0
    macd_signal: float = 0.0
    bb_position: float = 0.5
    
    # Multi-timeframe bias
    mtf_bias: TradeDirection = TradeDirection.NEUTRAL
    mtf_alignment: float = 0.0
    
    # Strategy and reasoning
    strategy: str = ""
    entry_reason: str = ""
    risk_reward: float = 2.0
    
    # Aggression and risk
    aggression_level: float = 0.5
    trailing_sl_enabled: bool = False
    trailing_sl_distance: float = 0.0
    
    timestamp: datetime = field(default_factory=datetime.now)


class FVGDetector:
    """Fair Value Gap / Imbalance detector"""
    
    def __init__(self, min_gap_pips: float = 5.0):
        self.min_gap_pips = min_gap_pips
        self.active_fvgs: List[FairValueGap] = []
    
    def detect_fvg(self, df: pd.DataFrame, pip_value: float = 0.0001) -> List[FairValueGap]:
        """Detect Fair Value Gaps in price data"""
        fvgs = []
        
        if len(df) < 3:
            return fvgs
        
        for i in range(2, len(df)):
            # Bullish FVG: Gap between candle 1 high and candle 3 low
            candle_1_high = df['high'].iloc[i-2]
            candle_2 = df.iloc[i-1]
            candle_3_low = df['low'].iloc[i]
            
            # Bullish FVG
            if candle_3_low > candle_1_high:
                gap_size = (candle_3_low - candle_1_high) / pip_value
                if gap_size >= self.min_gap_pips:
                    fvg = FairValueGap(
                        timestamp=df.index[i] if hasattr(df.index[i], 'timestamp') else datetime.now(),
                        direction=TradeDirection.LONG,
                        high=candle_3_low,
                        low=candle_1_high,
                        size=gap_size,
                        strength=self._calculate_strength(gap_size, candle_2)
                    )
                    fvgs.append(fvg)
            
            # Bearish FVG: Gap between candle 1 low and candle 3 high
            candle_1_low = df['low'].iloc[i-2]
            candle_3_high = df['high'].iloc[i]
            
            if candle_1_low > candle_3_high:
                gap_size = (candle_1_low - candle_3_high) / pip_value
                if gap_size >= self.min_gap_pips:
                    fvg = FairValueGap(
                        timestamp=df.index[i] if hasattr(df.index[i], 'timestamp') else datetime.now(),
                        direction=TradeDirection.SHORT,
                        high=candle_1_low,
                        low=candle_3_high,
                        size=gap_size,
                        strength=self._calculate_strength(gap_size, candle_2)
                    )
                    fvgs.append(fvg)
        
        return fvgs
    
    def _calculate_strength(self, gap_size: float, displacement_candle: pd.Series) -> SignalStrength:
        """Calculate FVG strength based on gap size and displacement"""
        candle_body = abs(displacement_candle['close'] - displacement_candle['open'])
        candle_range = displacement_candle['high'] - displacement_candle['low']
        
        body_ratio = candle_body / candle_range if candle_range > 0 else 0
        
        if gap_size > 20 and body_ratio > 0.7:
            return SignalStrength.VERY_STRONG
        elif gap_size > 15 and body_ratio > 0.6:
            return SignalStrength.STRONG
        elif gap_size > 10:
            return SignalStrength.MODERATE
        elif gap_size > 5:
            return SignalStrength.WEAK
        else:
            return SignalStrength.VERY_WEAK
    
    def check_fvg_fill(self, fvg: FairValueGap, current_price: float) -> float:
        """Check how much of an FVG has been filled"""
        if fvg.direction == TradeDirection.LONG:
            if current_price <= fvg.low:
                return 1.0  # Fully filled
            elif current_price < fvg.high:
                return (fvg.high - current_price) / (fvg.high - fvg.low)
        else:
            if current_price >= fvg.high:
                return 1.0
            elif current_price > fvg.low:
                return (current_price - fvg.low) / (fvg.high - fvg.low)
        return 0.0
    
    def get_nearest_unfilled_fvg(self, current_price: float, 
                                  direction: TradeDirection = None) -> Optional[FairValueGap]:
        """Get nearest unfilled FVG"""
        unfilled = [f for f in self.active_fvgs if f.fill_percentage < 0.5]
        
        if direction:
            unfilled = [f for f in unfilled if f.direction == direction]
        
        if not unfilled:
            return None
        
        # Sort by distance to current price
        unfilled.sort(key=lambda f: abs((f.high + f.low) / 2 - current_price))
        return unfilled[0]


class LiquiditySweepDetector:
    """Detects liquidity sweeps (stop hunts)"""
    
    def __init__(self, lookback: int = 20, min_sweep_pips: float = 5.0):
        self.lookback = lookback
        self.min_sweep_pips = min_sweep_pips
    
    def detect_sweeps(self, df: pd.DataFrame, pip_value: float = 0.0001) -> List[LiquiditySweep]:
        """Detect liquidity sweeps"""
        sweeps = []
        
        if len(df) < self.lookback + 2:
            return sweeps
        
        for i in range(self.lookback, len(df) - 1):
            # Get recent highs and lows
            lookback_data = df.iloc[i-self.lookback:i]
            recent_high = lookback_data['high'].max()
            recent_low = lookback_data['low'].min()
            
            current = df.iloc[i]
            next_candle = df.iloc[i + 1]
            
            # Bullish sweep: Price sweeps below recent low then reverses
            if current['low'] < recent_low:
                sweep_distance = (recent_low - current['low']) / pip_value
                if sweep_distance >= self.min_sweep_pips:
                    # Check for reversal
                    if next_candle['close'] > current['close']:
                        sweep = LiquiditySweep(
                            timestamp=df.index[i] if hasattr(df.index[i], 'timestamp') else datetime.now(),
                            direction=TradeDirection.LONG,
                            sweep_level=recent_low,
                            reversal_confirmed=True,
                            strength=self._calculate_strength(sweep_distance, next_candle, current)
                        )
                        sweeps.append(sweep)
            
            # Bearish sweep: Price sweeps above recent high then reverses
            if current['high'] > recent_high:
                sweep_distance = (current['high'] - recent_high) / pip_value
                if sweep_distance >= self.min_sweep_pips:
                    if next_candle['close'] < current['close']:
                        sweep = LiquiditySweep(
                            timestamp=df.index[i] if hasattr(df.index[i], 'timestamp') else datetime.now(),
                            direction=TradeDirection.SHORT,
                            sweep_level=recent_high,
                            reversal_confirmed=True,
                            strength=self._calculate_strength(sweep_distance, next_candle, current)
                        )
                        sweeps.append(sweep)
        
        return sweeps
    
    def _calculate_strength(self, sweep_distance: float, 
                            reversal_candle: pd.Series,
                            sweep_candle: pd.Series) -> SignalStrength:
        """Calculate sweep strength"""
        reversal_body = abs(reversal_candle['close'] - reversal_candle['open'])
        sweep_body = abs(sweep_candle['close'] - sweep_candle['open'])
        
        if sweep_distance > 15 and reversal_body > sweep_body * 1.5:
            return SignalStrength.VERY_STRONG
        elif sweep_distance > 10 and reversal_body > sweep_body:
            return SignalStrength.STRONG
        elif sweep_distance > 5:
            return SignalStrength.MODERATE
        else:
            return SignalStrength.WEAK


class MultiTimeframeAnalyzer:
    """Multi-timeframe analysis for market bias"""
    
    def __init__(self):
        self.indicators = TechnicalIndicators()
        self.timeframe_weights = {
            'M1': 0.05,
            'M5': 0.10,
            'M15': 0.15,
            'H1': 0.25,
            'H4': 0.25,
            'D1': 0.20
        }
    
    def analyze_bias(self, mtf_data: Dict[str, pd.DataFrame]) -> Tuple[TradeDirection, float]:
        """Analyze multi-timeframe bias"""
        if not mtf_data:
            return TradeDirection.NEUTRAL, 0.0
        
        bias_scores = []
        total_weight = 0
        
        for tf, df in mtf_data.items():
            if df.empty or len(df) < 50:
                continue
            
            weight = self.timeframe_weights.get(tf, 0.1)
            bias = self._calculate_tf_bias(df)
            
            bias_scores.append(bias * weight)
            total_weight += weight
        
        if total_weight == 0:
            return TradeDirection.NEUTRAL, 0.0
        
        weighted_bias = sum(bias_scores) / total_weight
        
        if weighted_bias > 0.3:
            return TradeDirection.LONG, abs(weighted_bias)
        elif weighted_bias < -0.3:
            return TradeDirection.SHORT, abs(weighted_bias)
        else:
            return TradeDirection.NEUTRAL, abs(weighted_bias)
    
    def _calculate_tf_bias(self, df: pd.DataFrame) -> float:
        """Calculate bias for single timeframe (-1 to 1)"""
        close = df['close']
        
        # Trend indicators
        sma_20 = self.indicators.sma(close, 20)
        sma_50 = self.indicators.sma(close, 50)
        sma_200 = self.indicators.sma(close, 200)
        
        bias = 0.0
        
        # Price vs MAs
        if len(sma_20) > 0 and not pd.isna(sma_20.iloc[-1]):
            if close.iloc[-1] > sma_20.iloc[-1]:
                bias += 0.2
            else:
                bias -= 0.2
        
        if len(sma_50) > 0 and not pd.isna(sma_50.iloc[-1]):
            if close.iloc[-1] > sma_50.iloc[-1]:
                bias += 0.2
            else:
                bias -= 0.2
        
        if len(sma_200) > 0 and not pd.isna(sma_200.iloc[-1]):
            if close.iloc[-1] > sma_200.iloc[-1]:
                bias += 0.2
            else:
                bias -= 0.2
        
        # MA alignment
        if len(sma_20) > 0 and len(sma_50) > 0:
            if not pd.isna(sma_20.iloc[-1]) and not pd.isna(sma_50.iloc[-1]):
                if sma_20.iloc[-1] > sma_50.iloc[-1]:
                    bias += 0.2
                else:
                    bias -= 0.2
        
        # Recent momentum
        if len(close) >= 10:
            momentum = (close.iloc[-1] - close.iloc[-10]) / close.iloc[-10]
            bias += np.clip(momentum * 10, -0.2, 0.2)
        
        return np.clip(bias, -1, 1)
    
    def get_entry_timeframe_confirmation(self, mtf_data: Dict[str, pd.DataFrame],
                                          direction: TradeDirection) -> float:
        """Get confirmation score from entry timeframes (M1, M5)"""
        confirmation = 0.0
        
        for tf in ['M1', 'M5']:
            if tf in mtf_data and not mtf_data[tf].empty:
                df = mtf_data[tf]
                tf_bias = self._calculate_tf_bias(df)
                
                if direction == TradeDirection.LONG and tf_bias > 0:
                    confirmation += 0.5
                elif direction == TradeDirection.SHORT and tf_bias < 0:
                    confirmation += 0.5
        
        return confirmation


class DynamicLeverageCalculator:
    """Calculate dynamic leverage based on conditions"""
    
    def __init__(self, min_leverage: int = 50, max_leverage: int = 500):
        self.min_leverage = min_leverage
        self.max_leverage = max_leverage
    
    def calculate_leverage(self, confidence: float, adx: float, atr: float,
                           regime: MarketRegime, aggression: float) -> int:
        """Calculate optimal leverage"""
        base_leverage = 100
        
        # Confidence adjustment
        if confidence > 0.8:
            base_leverage *= 1.5
        elif confidence > 0.6:
            base_leverage *= 1.2
        elif confidence < 0.5:
            base_leverage *= 0.7
        
        # ADX adjustment (trend strength)
        if adx > 40:
            base_leverage *= 1.3  # Strong trend - can use more leverage
        elif adx < 20:
            base_leverage *= 0.7  # Weak trend - reduce leverage
        
        # Volatility adjustment (ATR)
        # Higher volatility = lower leverage
        if atr > 0.002:  # High volatility
            base_leverage *= 0.6
        elif atr > 0.001:
            base_leverage *= 0.8
        
        # Regime adjustment
        if regime:
            base_leverage *= regime.risk_adjustment
        
        # Aggression adjustment
        base_leverage *= (0.5 + aggression)
        
        # Clamp to limits
        leverage = int(np.clip(base_leverage, self.min_leverage, self.max_leverage))
        
        return leverage


class TrailingStopManager:
    """Manages trailing stop logic"""
    
    def __init__(self):
        self.active_trails: Dict[str, Dict] = {}
    
    def should_enable_trailing(self, signal: TradingSignal, 
                                current_profit_pips: float) -> bool:
        """Determine if trailing stop should be enabled"""
        # Enable trailing after minimum profit
        min_profit_pips = 10
        
        if current_profit_pips < min_profit_pips:
            return False
        
        # Enable if strong trend
        if signal.adx > 35:
            return True
        
        # Enable if high confidence
        if signal.confidence > 0.75:
            return True
        
        return current_profit_pips > 20
    
    def calculate_trailing_distance(self, atr: float, adx: float,
                                     aggression: float) -> float:
        """Calculate trailing stop distance in price"""
        # Base distance is 1.5 ATR
        base_distance = atr * 1.5
        
        # Tighter trail in strong trends
        if adx > 40:
            base_distance *= 0.8
        elif adx < 25:
            base_distance *= 1.2
        
        # Adjust for aggression
        base_distance *= (1.5 - aggression * 0.5)
        
        return base_distance
    
    def update_trailing_stop(self, ticket: str, direction: TradeDirection,
                              entry_price: float, current_price: float,
                              current_sl: float, trail_distance: float) -> float:
        """Update trailing stop level"""
        if direction == TradeDirection.LONG:
            new_sl = current_price - trail_distance
            if new_sl > current_sl:
                return new_sl
        else:
            new_sl = current_price + trail_distance
            if new_sl < current_sl:
                return new_sl
        
        return current_sl


class VeteranTraderDecisionEngine:
    """Main decision engine that thinks like a veteran trader"""
    
    def __init__(self):
        self.indicators = TechnicalIndicators()
        self.feature_engineer = FeatureEngineer()
        self.fvg_detector = FVGDetector()
        self.sweep_detector = LiquiditySweepDetector()
        self.mtf_analyzer = MultiTimeframeAnalyzer()
        self.leverage_calculator = DynamicLeverageCalculator()
        self.trailing_manager = TrailingStopManager()
        
        self.pattern_miner = pattern_miner if PATTERN_MINER_AVAILABLE else None
        self.bedrock_ai = bedrock_ai if BEDROCK_AVAILABLE else None
        
        # Trade gating system for cost-aware filtering
        self.trade_gating = get_trade_gating() if TRADE_GATING_AVAILABLE else None
        
        # Macro data fetcher for DXY, VIX, Treasury yields
        self.macro_fetcher = MacroDataFetcher() if MACRO_DATA_AVAILABLE else None
        self._last_macro_update = None
        self._cached_macro_regime = None
        
        # Advanced knowledge engine for economic calendar, topic sentiment, cross-asset regime
        self.advanced_knowledge = get_advanced_knowledge() if ADVANCED_KNOWLEDGE_AVAILABLE else None
        self._last_advanced_update = None
        self._cached_trading_context = {}
        
        # Adaptive learning engine for online/offline learning
        self.adaptive_learning = get_adaptive_learning() if ADAPTIVE_LEARNING_AVAILABLE else None
        
        # Trade journaling and uncertainty governor for human-like decision making
        self.journaling_engine = get_journaling_engine() if TRADE_JOURNALING_AVAILABLE else None
        
        # Institutional event logging system for measurement & auditability (Tier 1)
        self.event_system = get_event_system() if EVENT_LOG_AVAILABLE else None
        
        # Portfolio risk manager for currency exposure, concentration, kill switch (Tier 2)
        self.portfolio_risk = get_portfolio_risk_manager() if PORTFOLIO_RISK_AVAILABLE else None
        
        # Research/production manager for shadow mode, experiments (Tier 3)
        self.research_production = get_research_production_manager() if RESEARCH_PRODUCTION_AVAILABLE else None
        
        # Decision thresholds - INCREASED for higher-quality trades
        # Most retail forex systems lose money by overtrading
        # Better to take fewer, higher-quality trades with clear edge
        self.min_confidence = 0.70  # Increased from 0.55 to reduce trade frequency
        self.min_rr_ratio = 2.0     # Increased from 1.5 for better risk/reward
        
        # SIMPLIFIED STRATEGY STACK: Focus on 2 proven strategies
        # Session breakout (trend_following during London/NY) and mean reversion with news filter
        self.strategy_weights = {
            'session_breakout': 0.5,   # Primary: London/NY session breakout
            'mean_reversion': 0.5,     # Secondary: Mean reversion with news filter
        }
        
        # Per-strategy expectancy tracking
        self.strategy_stats = {
            'session_breakout': {'trades': 0, 'wins': 0, 'total_pips': 0.0},
            'mean_reversion': {'trades': 0, 'wins': 0, 'total_pips': 0.0},
        }
        
        logger.info(f"Decision engine initialized - PatternMiner: {PATTERN_MINER_AVAILABLE}, BedrockAI: {BEDROCK_AVAILABLE}, TradeGating: {TRADE_GATING_AVAILABLE}, MacroData: {MACRO_DATA_AVAILABLE}, AdvancedKnowledge: {ADVANCED_KNOWLEDGE_AVAILABLE}, AdaptiveLearning: {ADAPTIVE_LEARNING_AVAILABLE}, TradeJournaling: {TRADE_JOURNALING_AVAILABLE}, EventLog: {EVENT_LOG_AVAILABLE}, PortfolioRisk: {PORTFOLIO_RISK_AVAILABLE}, ResearchProduction: {RESEARCH_PRODUCTION_AVAILABLE}")
    
    def analyze_market(self, symbol: str, mtf_data: Dict[str, pd.DataFrame],
                       account_balance: float, spread_pips: float = 2.0) -> Optional[TradingSignal]:
        """Comprehensive market analysis like a veteran trader
        
        Args:
            symbol: Trading pair (e.g., 'EURUSD')
            mtf_data: Multi-timeframe OHLC data
            account_balance: Current account balance
            spread_pips: Current spread in pips (from real market data)
        """
        import time
        start_time = time.time()
        
        # Start decision cycle logging (Tier 1: Measurement & Auditability)
        cycle_id = None
        if self.event_system and EVENT_LOG_AVAILABLE:
            try:
                cycle_id = self.event_system.event_logger.start_decision_cycle(symbol)
            except Exception as e:
                logger.warning(f"Event logging error: {e}")
        
        # Tier 2: Check portfolio kill switch before any analysis
        if self.portfolio_risk and PORTFOLIO_RISK_AVAILABLE:
            try:
                if is_kill_switch_active():
                    reason = self.portfolio_risk.kill_switch.reason
                    logger.warning(f"[KILL SWITCH] Trading halted for {symbol}: {reason.value if reason else 'unknown'}")
                    if cycle_id and self.event_system:
                        self.event_system.event_logger.log_trade_decision(
                            cycle_id, "rejected", f"Kill switch active: {reason.value if reason else 'unknown'}"
                        )
                        self.event_system.event_logger.end_decision_cycle(cycle_id, (time.time() - start_time) * 1000)
                    return None
            except Exception as e:
                logger.warning(f"Portfolio risk check error: {e}")
        
        # Get primary timeframe data (H1 for analysis)
        primary_tf = 'H1'
        if primary_tf not in mtf_data or mtf_data[primary_tf].empty:
            primary_tf = list(mtf_data.keys())[0] if mtf_data else None
        
        if not primary_tf:
            if cycle_id and self.event_system:
                self.event_system.event_logger.log_trade_decision(cycle_id, "no_trade", "No data available")
                self.event_system.event_logger.end_decision_cycle(cycle_id, (time.time() - start_time) * 1000)
            return None
        
        df = mtf_data[primary_tf]
        if len(df) < 100:
            if cycle_id and self.event_system:
                self.event_system.event_logger.log_trade_decision(cycle_id, "no_trade", "Insufficient data")
                self.event_system.event_logger.end_decision_cycle(cycle_id, (time.time() - start_time) * 1000)
            return None
        
        # 1. Get regime context
        regime = regime_manager.detect_regime(df)
        
        # 1.5 Get macro regime (DXY, VIX, Treasury yields) - updates every 5 minutes
        macro_regime = None
        macro_confidence_adjustment = 1.0
        if self.macro_fetcher:
            try:
                now = datetime.now()
                if self._last_macro_update is None or (now - self._last_macro_update).total_seconds() > 300:
                    self._cached_macro_regime = self.macro_fetcher.get_market_regime()
                    self._last_macro_update = now
                    logger.info(f"Macro regime updated: {self._cached_macro_regime}")
                
                macro_regime = self._cached_macro_regime
                
                if macro_regime:
                    # Adjust confidence based on macro alignment with trade direction
                    # Risk-on favors AUD, NZD; Risk-off favors JPY, CHF, USD
                    risk_currencies = {'AUD', 'NZD', 'CAD'}
                    safe_currencies = {'JPY', 'CHF'}
                    
                    base_curr = symbol[:3]
                    quote_curr = symbol[3:]
                    
                    if macro_regime.get('regime') == 'risk_on':
                        if base_curr in risk_currencies or quote_curr in safe_currencies:
                            macro_confidence_adjustment = 1.15  # Boost confidence for risk-on trades
                        elif base_curr in safe_currencies or quote_curr in risk_currencies:
                            macro_confidence_adjustment = 0.85  # Reduce confidence for counter-trend
                    elif macro_regime.get('regime') == 'risk_off':
                        if base_curr in safe_currencies or quote_curr in risk_currencies:
                            macro_confidence_adjustment = 1.15  # Boost confidence for safe haven trades
                        elif base_curr in risk_currencies or quote_curr in safe_currencies:
                            macro_confidence_adjustment = 0.85  # Reduce confidence for counter-trend
                    
                    # VIX extreme levels affect all trades
                    if macro_regime.get('vix_level') == 'extreme':
                        macro_confidence_adjustment *= 0.7  # Reduce all trade confidence in panic
                    elif macro_regime.get('vix_level') == 'high':
                        macro_confidence_adjustment *= 0.85  # Cautious in high VIX
                    
                    logger.debug(f"Macro adjustment for {symbol}: {macro_confidence_adjustment:.2f}")
            except Exception as e:
                logger.warning(f"Macro regime error: {e}")
        
        # 1.6 Advanced Knowledge: Economic Calendar, Topic Sentiment, Cross-Asset Regime
        advanced_context = None
        advanced_confidence_adjustment = 1.0
        should_skip_trade = False
        skip_reason = ""
        
        if self.advanced_knowledge:
            try:
                now = datetime.now()
                # Update trading context every 5 minutes
                if (symbol not in self._cached_trading_context or 
                    self._last_advanced_update is None or 
                    (now - self._last_advanced_update).total_seconds() > 300):
                    
                    self._cached_trading_context[symbol] = self.advanced_knowledge.get_trading_context(symbol)
                    self._last_advanced_update = now
                    logger.info(f"Advanced knowledge updated for {symbol}: {self._cached_trading_context[symbol].get('direction')}, "
                               f"confidence={self._cached_trading_context[symbol].get('confidence', 0):.2f}")
                
                advanced_context = self._cached_trading_context.get(symbol)
                
                if advanced_context:
                    # Check if we should avoid trading (high-impact events, extreme volatility)
                    if advanced_context.get('should_reduce_risk'):
                        upcoming = advanced_context.get('upcoming_events', [])
                        if upcoming:
                            event_names = [e.get('name', 'Unknown') for e in upcoming[:2]]
                            skip_reason = f"High-impact events upcoming: {', '.join(event_names)}"
                            should_skip_trade = True
                            logger.info(f"Skipping {symbol} trade: {skip_reason}")
                    
                    # Apply confidence adjustment based on advanced knowledge
                    ak_confidence = advanced_context.get('confidence', 0.5)
                    ak_direction = advanced_context.get('direction', 'NEUTRAL')
                    
                    # If advanced knowledge has strong conviction, adjust our confidence
                    if ak_confidence > 0.6:
                        advanced_confidence_adjustment = 1.0 + (ak_confidence - 0.5) * 0.3  # Up to 1.15x boost
                    elif ak_confidence < 0.4:
                        advanced_confidence_adjustment = 0.85  # Reduce confidence when uncertain
                    
                    # Log the analysis breakdown
                    analysis = advanced_context.get('analysis', {})
                    logger.debug(f"Advanced knowledge for {symbol}: "
                                f"calendar_bias={analysis.get('calendar', {}).get('bias', 0):.2f}, "
                                f"sentiment_bias={analysis.get('sentiment', {}).get('bias', 0):.2f}, "
                                f"cross_asset_bias={analysis.get('cross_asset', {}).get('bias', 0):.2f}")
                    
            except Exception as e:
                logger.warning(f"Advanced knowledge error for {symbol}: {e}")
        
        # Skip trade if high-impact events are imminent
        if should_skip_trade:
            logger.info(f"Trade blocked for {symbol}: {skip_reason}")
            return None
        
        # 2. Multi-timeframe bias
        mtf_bias, mtf_alignment = self.mtf_analyzer.analyze_bias(mtf_data)
        
        # 3. Calculate indicators
        close = df['close']
        high = df['high']
        low = df['low']
        
        rsi = self.indicators.rsi(close, 14).iloc[-1]
        adx, plus_di, minus_di = self.indicators.adx(high, low, close)
        adx_value = adx.iloc[-1] if not pd.isna(adx.iloc[-1]) else 25
        atr = self.indicators.atr(high, low, close, 14).iloc[-1]
        
        macd, macd_signal, macd_hist = self.indicators.macd(close)
        macd_value = macd_hist.iloc[-1] if not pd.isna(macd_hist.iloc[-1]) else 0
        
        bb_upper, bb_middle, bb_lower = self.indicators.bollinger_bands(close)
        current_price = close.iloc[-1]
        bb_position = (current_price - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1]) \
            if (bb_upper.iloc[-1] - bb_lower.iloc[-1]) > 0 else 0.5
        
        # Log market snapshot (Tier 1: Measurement & Auditability)
        if cycle_id and self.event_system and EVENT_LOG_AVAILABLE:
            try:
                bid = current_price - (spread_pips * pip_value / 2) if 'pip_value' in dir() else current_price
                ask = current_price + (spread_pips * pip_value / 2) if 'pip_value' in dir() else current_price
                pip_value_temp = 0.01 if 'JPY' in symbol else 0.0001
                snapshot = MarketSnapshot(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    bid=current_price - (spread_pips * pip_value_temp / 2),
                    ask=current_price + (spread_pips * pip_value_temp / 2),
                    spread_pips=spread_pips,
                    rsi=float(rsi) if not pd.isna(rsi) else 50.0,
                    adx=float(adx_value),
                    atr=float(atr) if not pd.isna(atr) else 0.0,
                    macd=float(macd_value),
                    bb_position=float(bb_position),
                    regime=regime.value if hasattr(regime, 'value') else str(regime),
                    mtf_bias=mtf_bias.name if hasattr(mtf_bias, 'name') else str(mtf_bias),
                    mtf_alignment=float(mtf_alignment),
                    macro_regime=macro_regime.get('regime', 'unknown') if macro_regime else 'unknown'
                )
                self.event_system.event_logger.log_market_snapshot(cycle_id, snapshot)
            except Exception as e:
                logger.warning(f"Market snapshot logging error: {e}")
        
        # 4. Detect FVGs and liquidity sweeps
        pip_value = 0.01 if 'JPY' in symbol else 0.0001
        fvgs = self.fvg_detector.detect_fvg(df, pip_value)
        sweeps = self.sweep_detector.detect_sweeps(df, pip_value)
        
        # 4.5 Get time-based pattern signal (human-like pattern recognition)
        pattern_signal = None
        pattern_descriptions = []
        if self.pattern_miner:
            try:
                current_time = datetime.now()
                pattern_data = self.pattern_miner.get_pattern_signal(symbol, current_time)
                if pattern_data['pattern_count'] > 0:
                    pattern_descriptions = pattern_data.get('descriptions', [])
                    logger.info(f"Pattern signal for {symbol}: bias={pattern_data['bias']:.2f}, "
                               f"confidence={pattern_data['confidence']:.2f}, patterns={pattern_data['pattern_count']}")
            except Exception as e:
                logger.warning(f"Pattern mining error: {e}")
        
        # 4.6 Get AI analysis from Bedrock (if available)
        ai_analysis = None
        if self.bedrock_ai:
            try:
                indicators_dict = {
                    'rsi': float(rsi),
                    'adx': float(adx_value),
                    'atr': float(atr),
                    'macd': float(macd_value),
                    'bb_position': float(bb_position)
                }
                market_data = {
                    'close': float(current_price),
                    'open': float(df['open'].iloc[-1]),
                    'high': float(df['high'].iloc[-1]),
                    'low': float(df['low'].iloc[-1])
                }
                ai_analysis = self.bedrock_ai.analyze_market(
                    symbol, market_data, indicators_dict,
                    regime.name if regime else 'unknown',
                    pattern_descriptions
                )
                logger.info(f"AI analysis for {symbol}: sentiment={ai_analysis.sentiment}, "
                           f"recommendation={ai_analysis.recommendation}, confidence={ai_analysis.confidence:.2f}")
            except Exception as e:
                logger.warning(f"Bedrock AI analysis error: {e}")
        
        # 5. Generate signals from each strategy
        signals = []
        
        # Trend following signal
        tf_signal = self._trend_following_signal(
            df, rsi, adx_value, macd_value, mtf_bias, regime
        )
        if tf_signal:
            signals.append(('trend_following', tf_signal))
        
        # Mean reversion signal
        mr_signal = self._mean_reversion_signal(
            df, rsi, bb_position, adx_value, regime
        )
        if mr_signal:
            signals.append(('mean_reversion', mr_signal))
        
        # FVG entry signal
        fvg_signal = self._fvg_entry_signal(fvgs, current_price, mtf_bias)
        if fvg_signal:
            signals.append(('fvg_entry', fvg_signal))
        
        # Liquidity sweep signal
        sweep_signal = self._liquidity_sweep_signal(sweeps, current_price)
        if sweep_signal:
            signals.append(('liquidity_sweep', sweep_signal))
        
        # Pattern-based signal (time-of-day patterns like "9am dip")
        if self.pattern_miner and pattern_data and pattern_data['confidence'] > 0.6:
            pattern_signal = self._pattern_based_signal(pattern_data, current_price)
            if pattern_signal:
                signals.append(('pattern_based', pattern_signal))
        
        # AI-enhanced signal adjustment
        if ai_analysis and ai_analysis.confidence > 0.6:
            signals = self._apply_ai_analysis(signals, ai_analysis)
        
        if not signals:
            if cycle_id and self.event_system:
                self.event_system.event_logger.log_trade_decision(cycle_id, "no_trade", "No signals generated")
                self.event_system.event_logger.end_decision_cycle(cycle_id, (time.time() - start_time) * 1000)
            return None
        
        # Log all generated signals (Tier 1: Measurement & Auditability)
        if cycle_id and self.event_system and EVENT_LOG_AVAILABLE:
            try:
                for strategy_name, sig_data in signals:
                    direction_val, conf, reason = sig_data
                    signal_event = SignalEvent(
                        timestamp=datetime.now(),
                        symbol=symbol,
                        strategy=strategy_name,
                        direction="BUY" if direction_val == TradeDirection.LONG else "SELL",
                        base_confidence=conf,
                        entry_reason=reason
                    )
                    self.event_system.event_logger.log_signal(cycle_id, signal_event, is_selected=False)
            except Exception as e:
                logger.warning(f"Signal logging error: {e}")
        
        # 6. Select best signal based on weights and confidence
        best_signal = self._select_best_signal(signals)
        if not best_signal:
            if cycle_id and self.event_system:
                self.event_system.event_logger.log_trade_decision(cycle_id, "no_trade", "No best signal selected")
                self.event_system.event_logger.end_decision_cycle(cycle_id, (time.time() - start_time) * 1000)
            return None
        
        strategy, (direction, base_confidence, entry_reason) = best_signal
        
        # Log selected signal (Tier 1: Measurement & Auditability)
        if cycle_id and self.event_system and EVENT_LOG_AVAILABLE:
            try:
                selected_signal = SignalEvent(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    strategy=strategy,
                    direction="BUY" if direction == TradeDirection.LONG else "SELL",
                    base_confidence=base_confidence,
                    entry_reason=entry_reason
                )
                self.event_system.event_logger.log_signal(cycle_id, selected_signal, is_selected=True)
            except Exception as e:
                logger.warning(f"Selected signal logging error: {e}")
        
        # 7. Calculate final confidence with all factors
        confidence = self._calculate_final_confidence(
            base_confidence, mtf_alignment, regime, adx_value, rsi
        )
        
        # Apply macro regime adjustment (DXY, VIX, Treasury yields)
        prev_confidence = confidence
        confidence *= macro_confidence_adjustment
        
        # Log macro confidence adjustment (Tier 1: Measurement & Auditability)
        if cycle_id and self.event_system and EVENT_LOG_AVAILABLE and macro_confidence_adjustment != 1.0:
            try:
                adjustment = ConfidenceAdjustment(
                    source="macro",
                    original_confidence=prev_confidence,
                    adjusted_confidence=confidence,
                    adjustment_factor=macro_confidence_adjustment,
                    reason=f"Macro regime: {macro_regime.get('regime', 'unknown') if macro_regime else 'unknown'}, VIX: {macro_regime.get('vix_level', 'unknown') if macro_regime else 'unknown'}"
                )
                self.event_system.event_logger.log_confidence_adjustment(cycle_id, adjustment)
            except Exception as e:
                logger.warning(f"Macro adjustment logging error: {e}")
        
        # Apply advanced knowledge adjustment (economic calendar, topic sentiment, cross-asset regime)
        prev_confidence = confidence
        confidence *= advanced_confidence_adjustment
        
        # Log advanced knowledge confidence adjustment (Tier 1: Measurement & Auditability)
        if cycle_id and self.event_system and EVENT_LOG_AVAILABLE and advanced_confidence_adjustment != 1.0:
            try:
                adjustment = ConfidenceAdjustment(
                    source="advanced_knowledge",
                    original_confidence=prev_confidence,
                    adjusted_confidence=confidence,
                    adjustment_factor=advanced_confidence_adjustment,
                    reason=f"Advanced knowledge confidence: {advanced_context.get('confidence', 0.5) if advanced_context else 0.5:.2f}"
                )
                self.event_system.event_logger.log_confidence_adjustment(cycle_id, adjustment)
            except Exception as e:
                logger.warning(f"Advanced knowledge adjustment logging error: {e}")
        
        # Apply adaptive learning adjustment (online/offline learning)
        adaptive_confidence_adjustment = 1.0
        prev_confidence_adaptive = confidence
        if self.adaptive_learning:
            try:
                # Get adjusted confidence based on learned parameters
                adjusted_conf = self.adaptive_learning.get_adjusted_confidence(
                    confidence, symbol, strategy
                )
                # Calculate the adjustment factor
                if confidence > 0:
                    adaptive_confidence_adjustment = adjusted_conf / confidence
                confidence = adjusted_conf
                
                # Check session quality - reduce confidence during low-quality sessions
                session_quality = self.adaptive_learning.get_session_quality()
                if session_quality < 0.7:
                    confidence *= session_quality
                    logger.debug(f"Session quality adjustment: {session_quality:.2f}")
                
                logger.debug(f"Adaptive learning adjustment for {symbol}: {adaptive_confidence_adjustment:.2f}")
                
                # Log adaptive learning confidence adjustment (Tier 1: Measurement & Auditability)
                if cycle_id and self.event_system and EVENT_LOG_AVAILABLE and adaptive_confidence_adjustment != 1.0:
                    try:
                        adjustment = ConfidenceAdjustment(
                            source="adaptive_learning",
                            original_confidence=prev_confidence_adaptive,
                            adjusted_confidence=confidence,
                            adjustment_factor=adaptive_confidence_adjustment,
                            reason=f"Adaptive learning adjustment, session_quality={session_quality:.2f}"
                        )
                        self.event_system.event_logger.log_confidence_adjustment(cycle_id, adjustment)
                    except Exception as e:
                        logger.warning(f"Adaptive learning adjustment logging error: {e}")
            except Exception as e:
                logger.warning(f"Adaptive learning error: {e}")
        
        # Apply uncertainty governor - human-like "I don't know" decision making
        uncertainty_position_multiplier = 1.0
        if self.journaling_engine:
            try:
                # Build uncertainty factors from available data
                uncertainty_factors = {
                    'technical': 0.3 if adx_value > 25 else 0.6,  # Higher uncertainty when no clear trend
                    'regime': 0.3 if regime.value in ['trending', 'volatile'] else 0.5,
                    'volatility': 0.4 if atr > 0 else 0.6,
                }
                
                # Add macro uncertainty if available
                if macro_regime:
                    vix_level = macro_regime.get('vix_level', 'normal')
                    uncertainty_factors['fundamental'] = 0.3 if vix_level == 'low' else (0.7 if vix_level == 'extreme' else 0.5)
                
                # Add advanced knowledge uncertainty if available
                if advanced_context:
                    ak_confidence = advanced_context.get('confidence', 0.5)
                    uncertainty_factors['sentiment'] = 1.0 - ak_confidence
                
                # Evaluate trade opportunity with uncertainty
                direction_str = "BUY" if direction == TradeDirection.LONG else "SELL"
                evaluation = self.journaling_engine.evaluate_trade_opportunity(
                    symbol=symbol,
                    direction=direction_str,
                    confidence=confidence,
                    uncertainty_factors=uncertainty_factors,
                    supporting_factors=[entry_reason],
                    opposing_factors=[],
                    regime=regime.value if hasattr(regime, 'value') else str(regime),
                    session=self.adaptive_learning.market_hours.get_current_session().value if self.adaptive_learning else "unknown"
                )
                
                # Check if uncertainty governor says "I don't know"
                if not evaluation.get('should_trade', True):
                    logger.info(f"[UNCERTAINTY GOVERNOR] {symbol} rejected: {evaluation.get('reason', 'Too uncertain')}")
                    # Log uncertainty gate rejection (Tier 1: Measurement & Auditability)
                    if cycle_id and self.event_system and EVENT_LOG_AVAILABLE:
                        try:
                            gate_result = GateCheckResult(
                                gate_name="uncertainty_governor",
                                passed=False,
                                reason=evaluation.get('reason', 'Too uncertain'),
                                details={'uncertainty_level': evaluation.get('uncertainty_level', 'unknown')}
                            )
                            self.event_system.event_logger.log_gate_check(cycle_id, gate_result)
                            self.event_system.event_logger.log_trade_decision(cycle_id, "rejected", f"Uncertainty governor: {evaluation.get('reason', 'Too uncertain')}")
                            self.event_system.event_logger.end_decision_cycle(cycle_id, (time.time() - start_time) * 1000)
                        except Exception as e:
                            logger.warning(f"Uncertainty gate logging error: {e}")
                    return None
                
                # Apply position multiplier based on uncertainty
                uncertainty_position_multiplier = evaluation.get('position_multiplier', 1.0)
                
                # Log uncertainty analysis
                logger.debug(f"Uncertainty analysis for {symbol}: level={evaluation.get('uncertainty_level')}, "
                           f"multiplier={uncertainty_position_multiplier:.2f}")
                
            except Exception as e:
                logger.warning(f"Uncertainty governor error: {e}")
        
        confidence = min(confidence, 0.95)  # Cap at 95%
        
        if confidence < self.min_confidence:
            # Log confidence threshold rejection (Tier 1: Measurement & Auditability)
            if cycle_id and self.event_system and EVENT_LOG_AVAILABLE:
                try:
                    gate_result = GateCheckResult(
                        gate_name="confidence_threshold",
                        passed=False,
                        reason=f"Confidence {confidence:.2f} < min {self.min_confidence}",
                        details={'confidence': confidence, 'min_confidence': self.min_confidence}
                    )
                    self.event_system.event_logger.log_gate_check(cycle_id, gate_result)
                    self.event_system.event_logger.log_trade_decision(cycle_id, "rejected", f"Confidence below threshold: {confidence:.2f}")
                    self.event_system.event_logger.end_decision_cycle(cycle_id, (time.time() - start_time) * 1000)
                except Exception as e:
                    logger.warning(f"Confidence gate logging error: {e}")
            return None
        
        # 8. Get trading parameters from agentic system
        trading_params = agentic_system.get_trading_parameters()
        aggression = trading_params.get('aggression_level', 0.5)
        
        # 9. Calculate entry, SL, TP with minimum stop level enforcement
        entry_price = current_price
        
        # Calculate pip value and minimum stop distance
        pip_size = 0.01 if 'JPY' in symbol else 0.0001
        min_stop_pips = 30  # Minimum 30 pips for safety (broker usually requires 10-20)
        min_stop_distance = min_stop_pips * pip_size
        
        # Use ATR-based stop but enforce minimum
        sl_distance = max(atr * 2, min_stop_distance)
        tp_distance = sl_distance * max(self.min_rr_ratio, 2.0)  # At least 1:2 RR
        
        if direction == TradeDirection.LONG:
            stop_loss = entry_price - sl_distance
            take_profit = entry_price + tp_distance
        else:
            stop_loss = entry_price + sl_distance
            take_profit = entry_price - tp_distance
        
        # 10. Calculate leverage
        leverage = self.leverage_calculator.calculate_leverage(
            confidence, adx_value, atr, regime, aggression
        )
        
        # 11. Calculate position size (proper lot sizing for Forex)
        risk_amount = agentic_system.calculate_position_size(
            {'confidence': confidence, 'strategy': strategy},
            account_balance
        )
        # Convert risk amount to lots
        # For Forex: 1 standard lot = 100,000 units, pip value varies by pair
        # Simplified: risk_amount / (sl_distance_in_pips * pip_value_per_lot)
        # For most USD pairs, pip value is ~$10 per standard lot
        pip_value_per_lot = 10.0  # USD per pip per standard lot
        sl_pips = sl_distance * 10000 if 'JPY' not in symbol else sl_distance * 100
        position_size = risk_amount / (sl_pips * pip_value_per_lot) if sl_pips > 0 else 0
        # Apply uncertainty governor position multiplier (reduces size when uncertain)
        position_size *= uncertainty_position_multiplier
        # Ensure minimum lot size of 0.01 and maximum of 1.0 for $100 account
        position_size = max(0.01, min(position_size, 1.0))
        
        # 12. COST-AWARE TRADE GATING - Use TradeGatingSystem for comprehensive filtering
        # This checks: spread, session (London/NY), news blackouts, edge requirements, drawdown
        tp_pips = tp_distance / pip_size
        expected_profit_pips = tp_pips * confidence  # Expected profit adjusted for win probability
        
        if self.trade_gating:
            # Update peak balance for drawdown tracking
            self.trade_gating.update_peak_balance(account_balance)
            
            # Run all gates with real spread data
            gates_passed, gate_results = self.trade_gating.check_all(
                symbol=symbol,
                spread_pips=spread_pips,
                expected_profit_pips=expected_profit_pips,
                confidence=confidence,
                current_balance=account_balance
            )
            
            # Log all gate checks (Tier 1: Measurement & Auditability)
            if cycle_id and self.event_system and EVENT_LOG_AVAILABLE:
                try:
                    for gate in gate_results:
                        gate_result = GateCheckResult(
                            gate_name=gate.gate_type,
                            passed=gate.allowed,
                            reason=gate.reason,
                            details={'spread': spread_pips, 'expected_profit': expected_profit_pips}
                        )
                        self.event_system.event_logger.log_gate_check(cycle_id, gate_result)
                except Exception as e:
                    logger.warning(f"Gate check logging error: {e}")
            
            if not gates_passed:
                # Log which gates failed
                failed_gates = [r for r in gate_results if not r.allowed]
                for gate in failed_gates:
                    logger.info(f"[TRADE GATE] {symbol} rejected by {gate.gate_type}: {gate.reason}")
                
                # Log trade decision rejection (Tier 1: Measurement & Auditability)
                if cycle_id and self.event_system and EVENT_LOG_AVAILABLE:
                    try:
                        failed_gate_names = [g.gate_type for g in failed_gates]
                        self.event_system.event_logger.log_trade_decision(cycle_id, "rejected", f"Trade gates failed: {', '.join(failed_gate_names)}")
                        self.event_system.event_logger.end_decision_cycle(cycle_id, (time.time() - start_time) * 1000)
                    except Exception as e:
                        logger.warning(f"Gate rejection logging error: {e}")
                return None
            
            # Log gate summary for successful trades
            logger.debug(f"[TRADE GATE] {symbol} passed: {self.trade_gating.get_gate_summary(gate_results)}")
        else:
            # Fallback: basic edge check if trade gating not available
            slippage_buffer_pips = 0.5
            total_cost_pips = spread_pips + slippage_buffer_pips
            required_edge = total_cost_pips * 2.0
            
            if expected_profit_pips < required_edge:
                logger.debug(f"[EDGE GATE] {symbol} rejected: expected_edge={expected_profit_pips:.1f} pips < required={required_edge:.1f} pips")
                return None
        
        # 13. Determine trailing stop
        trailing_enabled = confidence > 0.7 and adx_value > 30
        trailing_distance = self.trailing_manager.calculate_trailing_distance(
            atr, adx_value, aggression
        ) if trailing_enabled else 0
        
        # 13.5 Tier 2: Check portfolio risk limits (currency exposure, concentration)
        if self.portfolio_risk and PORTFOLIO_RISK_AVAILABLE:
            try:
                direction_str = "BUY" if direction == TradeDirection.LONG else "SELL"
                # Note: current_positions would need to be passed in for full check
                # For now, we just check if kill switch is active (already done above)
                # Full position-level checks happen in main.py before order execution
                logger.debug(f"[PORTFOLIO RISK] Trade {direction_str} {symbol} passed initial risk checks")
            except Exception as e:
                logger.warning(f"Portfolio risk check error: {e}")
        
        # 13.6 Tier 3: Process signal in shadow mode if enabled
        if self.research_production and RESEARCH_PRODUCTION_AVAILABLE:
            try:
                if is_shadow_mode():
                    direction_str = "BUY" if direction == TradeDirection.LONG else "SELL"
                    shadow_trade = process_shadow_signal(
                        symbol=symbol,
                        direction=direction_str,
                        entry_price=entry_price,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        position_size=position_size,
                        strategy=strategy,
                        confidence=confidence
                    )
                    if shadow_trade:
                        logger.info(f"[SHADOW MODE] Recorded shadow trade: {shadow_trade.trade_id}")
            except Exception as e:
                logger.warning(f"Shadow mode processing error: {e}")
        
        # 14. Create final signal
        signal = TradingSignal(
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=position_size,
            leverage=leverage,
            fvg_signal=fvgs[-1] if fvgs and strategy == 'fvg_entry' else None,
            liquidity_sweep=sweeps[-1] if sweeps and strategy == 'liquidity_sweep' else None,
            regime=regime,
            rsi=rsi,
            adx=adx_value,
            atr=atr,
            macd_signal=macd_value,
            bb_position=bb_position,
            mtf_bias=mtf_bias,
            mtf_alignment=mtf_alignment,
            strategy=strategy,
            entry_reason=entry_reason,
            risk_reward=tp_distance / sl_distance if sl_distance > 0 else 0,
            aggression_level=aggression,
            trailing_sl_enabled=trailing_enabled,
            trailing_sl_distance=trailing_distance
        )
        
        # Log successful trade decision (Tier 1: Measurement & Auditability)
        if cycle_id and self.event_system and EVENT_LOG_AVAILABLE:
            try:
                direction_str = "BUY" if direction == TradeDirection.LONG else "SELL"
                self.event_system.event_logger.log_trade_decision(
                    cycle_id, 
                    "trade", 
                    f"{direction_str} {symbol} @ {entry_price:.5f}, SL={stop_loss:.5f}, TP={take_profit:.5f}, size={position_size:.2f}, conf={confidence:.2f}"
                )
                self.event_system.event_logger.end_decision_cycle(cycle_id, (time.time() - start_time) * 1000)
            except Exception as e:
                logger.warning(f"Trade decision logging error: {e}")
        
        return signal
    
    def _trend_following_signal(self, df: pd.DataFrame, rsi: float, adx: float,
                                 macd: float, mtf_bias: TradeDirection,
                                 regime: MarketRegime) -> Optional[Tuple]:
        """Generate trend following signal"""
        if adx < 25:  # No clear trend
            return None
        
        confidence = 0.5
        direction = None
        reasons = []
        
        # Strong uptrend
        if adx > 30 and macd > 0 and rsi > 50 and rsi < 70:
            direction = TradeDirection.LONG
            confidence += 0.1
            reasons.append("ADX confirms uptrend")
            
            if mtf_bias == TradeDirection.LONG:
                confidence += 0.15
                reasons.append("MTF aligned bullish")
        
        # Strong downtrend
        elif adx > 30 and macd < 0 and rsi < 50 and rsi > 30:
            direction = TradeDirection.SHORT
            confidence += 0.1
            reasons.append("ADX confirms downtrend")
            
            if mtf_bias == TradeDirection.SHORT:
                confidence += 0.15
                reasons.append("MTF aligned bearish")
        
        if direction is None:
            return None
        
        # Regime bonus
        if regime and 'trend' in regime.name.lower():
            confidence += 0.1
            reasons.append("Trending regime")
        
        return (direction, confidence, "trend_following: " + ", ".join(reasons))
    
    def _mean_reversion_signal(self, df: pd.DataFrame, rsi: float,
                                bb_position: float, adx: float,
                                regime: MarketRegime) -> Optional[Tuple]:
        """Generate mean reversion signal"""
        if adx > 35:  # Too trendy for mean reversion
            return None
        
        confidence = 0.5
        direction = None
        reasons = []
        
        # Oversold conditions
        if rsi < 30 and bb_position < 0.1:
            direction = TradeDirection.LONG
            confidence += 0.15
            reasons.append(f"RSI oversold ({rsi:.1f})")
            reasons.append("Price at lower BB")
        
        # Overbought conditions
        elif rsi > 70 and bb_position > 0.9:
            direction = TradeDirection.SHORT
            confidence += 0.15
            reasons.append(f"RSI overbought ({rsi:.1f})")
            reasons.append("Price at upper BB")
        
        if direction is None:
            return None
        
        # Regime bonus
        if regime and 'low_vol' in regime.name.lower():
            confidence += 0.1
            reasons.append("Low volatility regime favors MR")
        
        return (direction, confidence, "mean_reversion: " + ", ".join(reasons))
    
    def _fvg_entry_signal(self, fvgs: List[FairValueGap], current_price: float,
                          mtf_bias: TradeDirection) -> Optional[Tuple]:
        """Generate FVG entry signal"""
        if not fvgs:
            return None
        
        # Look for recent unfilled FVGs
        recent_fvgs = [f for f in fvgs[-5:] if f.fill_percentage < 0.5]
        
        if not recent_fvgs:
            return None
        
        # Find FVG aligned with MTF bias
        for fvg in reversed(recent_fvgs):
            if fvg.direction == mtf_bias or mtf_bias == TradeDirection.NEUTRAL:
                confidence = 0.5
                
                # Strength bonus
                if fvg.strength == SignalStrength.VERY_STRONG:
                    confidence += 0.2
                elif fvg.strength == SignalStrength.STRONG:
                    confidence += 0.15
                elif fvg.strength == SignalStrength.MODERATE:
                    confidence += 0.1
                
                # MTF alignment bonus
                if fvg.direction == mtf_bias:
                    confidence += 0.1
                
                # Check if price is near FVG
                fvg_mid = (fvg.high + fvg.low) / 2
                distance_pct = abs(current_price - fvg_mid) / current_price
                
                if distance_pct < 0.002:  # Within 0.2%
                    confidence += 0.1
                    return (fvg.direction, confidence, 
                           f"fvg_entry: {fvg.strength.name} FVG, size={fvg.size:.1f} pips")
        
        return None
    
    def _liquidity_sweep_signal(self, sweeps: List[LiquiditySweep],
                                 current_price: float) -> Optional[Tuple]:
        """Generate liquidity sweep signal"""
        if not sweeps:
            return None
        
        # Look for recent confirmed sweeps
        recent_sweeps = [s for s in sweeps[-3:] if s.reversal_confirmed]
        
        if not recent_sweeps:
            return None
        
        # Use most recent sweep
        sweep = recent_sweeps[-1]
        confidence = 0.55
        
        # Strength bonus
        if sweep.strength == SignalStrength.VERY_STRONG:
            confidence += 0.2
        elif sweep.strength == SignalStrength.STRONG:
            confidence += 0.15
        elif sweep.strength == SignalStrength.MODERATE:
            confidence += 0.1
        
        return (sweep.direction, confidence,
               f"liquidity_sweep: {sweep.strength.name} sweep at {sweep.sweep_level:.5f}")
    
    def _select_best_signal(self, signals: List[Tuple]) -> Optional[Tuple]:
        """Select best signal based on weights and confidence"""
        if not signals:
            return None
        
        # Get weights from agentic system
        params = agentic_system.get_trading_parameters()
        strategy_weights = params.get('strategy_weights', self.strategy_weights)
        
        best_score = 0
        best_signal = None
        
        for strategy, signal_data in signals:
            direction, confidence, reason = signal_data
            weight = strategy_weights.get(strategy, 0.25)
            score = confidence * weight
            
            if score > best_score:
                best_score = score
                best_signal = (strategy, signal_data)
        
        return best_signal
    
    def _calculate_final_confidence(self, base_confidence: float,
                                     mtf_alignment: float, regime: MarketRegime,
                                     adx: float, rsi: float) -> float:
        """Calculate final confidence with all factors"""
        confidence = base_confidence
        
        # MTF alignment bonus
        confidence += mtf_alignment * 0.1
        
        # Regime confidence
        if regime:
            confidence *= (0.8 + regime.probability * 0.4)
        
        # ADX confirmation
        if adx > 40:
            confidence += 0.05
        elif adx < 20:
            confidence -= 0.05
        
        # RSI extremes (potential reversal risk)
        if rsi > 80 or rsi < 20:
            confidence -= 0.05
        
        return np.clip(confidence, 0, 1)
    
    def _pattern_based_signal(self, pattern_data: Dict, current_price: float) -> Optional[Tuple]:
        """Generate signal based on discovered time-based patterns"""
        bias = pattern_data.get('bias', 0)
        confidence = pattern_data.get('confidence', 0)
        expected_move = pattern_data.get('expected_move', 0)
        descriptions = pattern_data.get('descriptions', [])
        
        if abs(bias) < 0.3 or confidence < 0.6:
            return None
        
        direction = TradeDirection.LONG if bias > 0 else TradeDirection.SHORT
        
        base_confidence = 0.5 + abs(bias) * 0.3
        base_confidence = min(0.85, base_confidence * confidence)
        
        reason_parts = ["pattern_based"]
        if descriptions:
            reason_parts.append(descriptions[0][:50])
        reason = ": ".join(reason_parts)
        
        logger.info(f"Pattern-based signal: {direction.name}, confidence={base_confidence:.2f}, "
                   f"expected_move={expected_move:.1f} pips")
        
        return (direction, base_confidence, reason)
    
    def _apply_ai_analysis(self, signals: List[Tuple], ai_analysis) -> List[Tuple]:
        """Apply AI analysis to adjust signal confidence"""
        if not signals or not ai_analysis:
            return signals
        
        adjusted_signals = []
        
        for strategy, (direction, confidence, reason) in signals:
            adjusted_confidence = confidence
            
            if ai_analysis.recommendation == 'buy' and direction == TradeDirection.LONG:
                adjusted_confidence *= (1 + ai_analysis.confidence * 0.2)
                reason += f" [AI: bullish {ai_analysis.confidence:.0%}]"
            elif ai_analysis.recommendation == 'sell' and direction == TradeDirection.SHORT:
                adjusted_confidence *= (1 + ai_analysis.confidence * 0.2)
                reason += f" [AI: bearish {ai_analysis.confidence:.0%}]"
            elif ai_analysis.recommendation == 'hold' or ai_analysis.recommendation == 'wait':
                adjusted_confidence *= 0.8
                reason += " [AI: caution]"
            elif (ai_analysis.recommendation == 'buy' and direction == TradeDirection.SHORT) or \
                 (ai_analysis.recommendation == 'sell' and direction == TradeDirection.LONG):
                adjusted_confidence *= 0.6
                reason += " [AI: conflicting]"
            
            adjusted_confidence *= ai_analysis.position_size_modifier
            adjusted_confidence = np.clip(adjusted_confidence, 0, 1)
            
            adjusted_signals.append((strategy, (direction, adjusted_confidence, reason)))
        
        return adjusted_signals
    
    def learn_patterns_from_data(self, symbol: str, df: pd.DataFrame):
        """Learn time-based patterns from historical data"""
        if self.pattern_miner and not df.empty:
            try:
                patterns = self.pattern_miner.analyze_historical_data(df, symbol)
                logger.info(f"Learned {len(patterns)} new patterns for {symbol}")
                return patterns
            except Exception as e:
                logger.error(f"Pattern learning error: {e}")
        return []
    
    def update_strategy_stats(self, strategy: str, profit_pips: float, was_win: bool):
        """Update per-strategy expectancy tracking
        
        Args:
            strategy: Strategy name (e.g., 'session_breakout', 'mean_reversion')
            profit_pips: Profit/loss in pips (positive for profit, negative for loss)
            was_win: Whether the trade was profitable
        """
        if strategy not in self.strategy_stats:
            self.strategy_stats[strategy] = {'trades': 0, 'wins': 0, 'total_pips': 0.0}
        
        stats = self.strategy_stats[strategy]
        stats['trades'] += 1
        stats['total_pips'] += profit_pips
        if was_win:
            stats['wins'] += 1
        
        # Calculate expectancy
        if stats['trades'] > 0:
            win_rate = stats['wins'] / stats['trades']
            avg_pips = stats['total_pips'] / stats['trades']
            logger.info(f"[STRATEGY STATS] {strategy}: trades={stats['trades']}, "
                       f"win_rate={win_rate:.1%}, avg_pips={avg_pips:.1f}, "
                       f"total_pips={stats['total_pips']:.1f}")
    
    def get_strategy_expectancy(self, strategy: str) -> Dict[str, float]:
        """Get expectancy metrics for a strategy"""
        if strategy not in self.strategy_stats:
            return {'trades': 0, 'win_rate': 0.0, 'avg_pips': 0.0, 'expectancy': 0.0}
        
        stats = self.strategy_stats[strategy]
        trades = stats['trades']
        if trades == 0:
            return {'trades': 0, 'win_rate': 0.0, 'avg_pips': 0.0, 'expectancy': 0.0}
        
        win_rate = stats['wins'] / trades
        avg_pips = stats['total_pips'] / trades
        
        return {
            'trades': trades,
            'win_rate': win_rate,
            'avg_pips': avg_pips,
            'expectancy': avg_pips  # Simplified: average pips per trade
        }
    
    def should_exit_trade(self, signal: TradingSignal, current_price: float,
                          current_profit_pips: float) -> Tuple[bool, str]:
        """Determine if trade should be exited early"""
        # Check stop loss
        if signal.direction == TradeDirection.LONG:
            if current_price <= signal.stop_loss:
                return True, "Stop loss hit"
        else:
            if current_price >= signal.stop_loss:
                return True, "Stop loss hit"
        
        # Check take profit
        if signal.direction == TradeDirection.LONG:
            if current_price >= signal.take_profit:
                return True, "Take profit hit"
        else:
            if current_price <= signal.take_profit:
                return True, "Take profit hit"
        
        # Early exit on regime change
        current_regime = regime_manager.current_regime
        if current_regime and signal.regime:
            if current_regime.name != signal.regime.name:
                if current_profit_pips > 5:  # Only if in profit
                    return True, "Regime changed - securing profit"
        
        # Early exit on RSI divergence
        # (Would need current RSI - simplified here)
        
        return False, ""


# Singleton instance
decision_engine = VeteranTraderDecisionEngine()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Trading Decisions Module...")
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 200
    
    prices = [1.1000]
    for i in range(n_samples - 1):
        change = np.random.normal(0.0001, 0.001)
        prices.append(prices[-1] * (1 + change))
    
    df = pd.DataFrame({
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.002))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.002))) for p in prices],
        'close': prices,
        'volume': np.random.randint(1000, 10000, n_samples)
    })
    df.index = pd.date_range(start='2024-01-01', periods=n_samples, freq='H')
    
    # Test FVG detection
    print("\nTesting FVG Detection...")
    fvg_detector = FVGDetector(min_gap_pips=3)
    fvgs = fvg_detector.detect_fvg(df)
    print(f"Detected {len(fvgs)} FVGs")
    if fvgs:
        print(f"Latest FVG: {fvgs[-1].direction.name}, size={fvgs[-1].size:.1f} pips")
    
    # Test liquidity sweep detection
    print("\nTesting Liquidity Sweep Detection...")
    sweep_detector = LiquiditySweepDetector()
    sweeps = sweep_detector.detect_sweeps(df)
    print(f"Detected {len(sweeps)} liquidity sweeps")
    
    # Test MTF analysis
    print("\nTesting Multi-Timeframe Analysis...")
    mtf_analyzer = MultiTimeframeAnalyzer()
    mtf_data = {'H1': df, 'H4': df.resample('4H').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    }).dropna()}
    bias, alignment = mtf_analyzer.analyze_bias(mtf_data)
    print(f"MTF Bias: {bias.name}, Alignment: {alignment:.2f}")
    
    # Test full decision engine
    print("\nTesting Decision Engine...")
    engine = VeteranTraderDecisionEngine()
    signal = engine.analyze_market('EURUSD', mtf_data, account_balance=100)
    
    if signal:
        print(f"\nGenerated Signal:")
        print(f"  Direction: {signal.direction.name}")
        print(f"  Confidence: {signal.confidence:.2f}")
        print(f"  Entry: {signal.entry_price:.5f}")
        print(f"  SL: {signal.stop_loss:.5f}")
        print(f"  TP: {signal.take_profit:.5f}")
        print(f"  Leverage: {signal.leverage}x")
        print(f"  Strategy: {signal.strategy}")
        print(f"  Reason: {signal.entry_reason}")
        print(f"  R:R: {signal.risk_reward:.2f}")
        print(f"  Trailing SL: {signal.trailing_sl_enabled}")
    else:
        print("No signal generated (conditions not met)")
    
    print("\nTrading Decisions Module test complete!")
