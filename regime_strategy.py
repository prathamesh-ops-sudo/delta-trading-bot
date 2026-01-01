"""
Priority 4: Regime-Aware Strategy Selector

Selects the optimal trading strategy based on current market regime.
Different market conditions require different approaches:
- Trending markets → Trend following strategies
- Ranging markets → Mean reversion strategies
- High volatility → Reduce size or sit out
- Low volatility → Breakout strategies

This is what separates systematic traders from gamblers - adapting
strategy to market conditions rather than forcing one approach.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime classifications"""
    STRONG_TREND_UP = "strong_trend_up"
    WEAK_TREND_UP = "weak_trend_up"
    RANGING = "ranging"
    WEAK_TREND_DOWN = "weak_trend_down"
    STRONG_TREND_DOWN = "strong_trend_down"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    BREAKOUT = "breakout"
    UNKNOWN = "unknown"


class TradingStrategy(Enum):
    """Available trading strategies"""
    TREND_FOLLOWING = "trend_following"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    MOMENTUM = "momentum"
    CARRY = "carry"
    RANGE_TRADING = "range_trading"
    VOLATILITY_BREAKOUT = "volatility_breakout"
    SIT_OUT = "sit_out"  # No trading - conditions unfavorable


@dataclass
class RegimeIndicators:
    """Indicators used to determine market regime"""
    # Trend indicators
    adx: float = 0.0              # Average Directional Index (0-100)
    adx_trend: str = "neutral"    # 'up', 'down', 'neutral'
    ma_alignment: float = 0.0     # MA alignment score (-1 to 1)
    price_vs_ma200: float = 0.0   # Price relative to 200 MA (%)
    
    # Volatility indicators
    atr_percentile: float = 50.0  # ATR percentile (0-100)
    bollinger_width: float = 0.0  # Bollinger Band width
    vix_level: float = 20.0       # VIX or equivalent
    
    # Range indicators
    range_bound_score: float = 0.0  # How range-bound (0-1)
    support_resistance_clarity: float = 0.0  # S/R clarity (0-1)
    
    # Momentum indicators
    rsi: float = 50.0
    macd_histogram: float = 0.0
    momentum_score: float = 0.0   # Overall momentum (-1 to 1)
    
    # Market structure
    higher_highs: int = 0         # Count of higher highs
    lower_lows: int = 0           # Count of lower lows
    swing_count: int = 0          # Number of swings in period


@dataclass
class StrategyConfig:
    """Configuration for a trading strategy"""
    name: TradingStrategy
    min_confidence: float = 0.6
    position_size_multiplier: float = 1.0
    max_trades_per_day: int = 3
    preferred_timeframes: List[str] = field(default_factory=lambda: ['H1', 'H4'])
    stop_loss_atr_multiplier: float = 2.0
    take_profit_atr_multiplier: float = 3.0
    trailing_enabled: bool = True
    partial_close_enabled: bool = True
    
    # Entry conditions
    required_indicators: List[str] = field(default_factory=list)
    entry_filters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyRecommendation:
    """Recommendation from the strategy selector"""
    primary_strategy: TradingStrategy
    secondary_strategy: Optional[TradingStrategy]
    regime: MarketRegime
    confidence: float
    position_size_multiplier: float
    reasoning: str
    indicators: Dict[str, float]
    warnings: List[str] = field(default_factory=list)


class RegimeDetector:
    """
    Detects current market regime using multiple indicators.
    
    Regime detection is crucial because:
    1. Trend-following in ranges = whipsaws and losses
    2. Mean reversion in trends = fighting the market
    3. Trading high volatility without adjustment = blown accounts
    """
    
    def __init__(self):
        self.regime_history: List[Tuple[datetime, MarketRegime]] = []
        self.regime_persistence_hours: int = 4  # Min hours before regime change
        
        # Thresholds for regime detection
        self.adx_strong_trend = 40
        self.adx_weak_trend = 25
        self.adx_no_trend = 20
        
        self.volatility_high_percentile = 80
        self.volatility_low_percentile = 20
        
        self.range_bound_threshold = 0.7
        
        logger.info("RegimeDetector initialized")
    
    def detect_regime(self, indicators: RegimeIndicators) -> Tuple[MarketRegime, float]:
        """
        Detect current market regime from indicators.
        
        Returns:
            Tuple of (regime, confidence)
        """
        # Check for high volatility first (overrides other regimes)
        if indicators.atr_percentile > self.volatility_high_percentile:
            if indicators.vix_level > 30:
                return MarketRegime.HIGH_VOLATILITY, 0.9
            return MarketRegime.HIGH_VOLATILITY, 0.7
        
        # Check for low volatility (potential breakout setup)
        if indicators.atr_percentile < self.volatility_low_percentile:
            if indicators.bollinger_width < 0.01:  # Very tight bands
                return MarketRegime.LOW_VOLATILITY, 0.8
            return MarketRegime.LOW_VOLATILITY, 0.6
        
        # Check for strong trends
        if indicators.adx > self.adx_strong_trend:
            if indicators.ma_alignment > 0.5 and indicators.momentum_score > 0.3:
                return MarketRegime.STRONG_TREND_UP, 0.85
            elif indicators.ma_alignment < -0.5 and indicators.momentum_score < -0.3:
                return MarketRegime.STRONG_TREND_DOWN, 0.85
        
        # Check for weak trends
        if indicators.adx > self.adx_weak_trend:
            if indicators.ma_alignment > 0.2:
                return MarketRegime.WEAK_TREND_UP, 0.65
            elif indicators.ma_alignment < -0.2:
                return MarketRegime.WEAK_TREND_DOWN, 0.65
        
        # Check for ranging market
        if indicators.adx < self.adx_no_trend:
            if indicators.range_bound_score > self.range_bound_threshold:
                return MarketRegime.RANGING, 0.75
            return MarketRegime.RANGING, 0.55
        
        # Check for breakout conditions
        if indicators.range_bound_score > 0.8 and indicators.atr_percentile < 30:
            # Tight range with low volatility = potential breakout
            return MarketRegime.BREAKOUT, 0.6
        
        return MarketRegime.UNKNOWN, 0.3
    
    def calculate_indicators(self, prices: np.ndarray, highs: np.ndarray, 
                            lows: np.ndarray, volumes: Optional[np.ndarray] = None) -> RegimeIndicators:
        """Calculate regime indicators from price data"""
        if len(prices) < 50:
            return RegimeIndicators()
        
        indicators = RegimeIndicators()
        
        # Calculate ADX
        indicators.adx = self._calculate_adx(highs, lows, prices)
        
        # Calculate MA alignment
        indicators.ma_alignment = self._calculate_ma_alignment(prices)
        
        # Calculate price vs MA200
        if len(prices) >= 200:
            ma200 = np.mean(prices[-200:])
            indicators.price_vs_ma200 = (prices[-1] - ma200) / ma200 * 100
        
        # Calculate ATR percentile
        atr = self._calculate_atr(highs, lows, prices)
        atr_history = [self._calculate_atr(highs[i:i+14], lows[i:i+14], prices[i:i+14]) 
                       for i in range(0, len(prices)-14, 14)]
        if atr_history:
            indicators.atr_percentile = np.percentile(atr_history, 
                                                       [i for i, x in enumerate(sorted(atr_history)) 
                                                        if x <= atr][-1] if atr_history else 50)
        
        # Calculate Bollinger width
        ma20 = np.mean(prices[-20:])
        std20 = np.std(prices[-20:])
        indicators.bollinger_width = (2 * std20) / ma20 if ma20 > 0 else 0
        
        # Calculate range-bound score
        indicators.range_bound_score = self._calculate_range_score(prices, highs, lows)
        
        # Calculate RSI
        indicators.rsi = self._calculate_rsi(prices)
        
        # Calculate momentum score
        indicators.momentum_score = self._calculate_momentum_score(prices)
        
        # Count higher highs and lower lows
        indicators.higher_highs, indicators.lower_lows = self._count_swing_points(highs, lows)
        
        return indicators
    
    def _calculate_adx(self, highs: np.ndarray, lows: np.ndarray, 
                       closes: np.ndarray, period: int = 14) -> float:
        """Calculate Average Directional Index"""
        if len(highs) < period + 1:
            return 0.0
        
        try:
            # True Range
            tr = np.maximum(
                highs[1:] - lows[1:],
                np.maximum(
                    np.abs(highs[1:] - closes[:-1]),
                    np.abs(lows[1:] - closes[:-1])
                )
            )
            
            # Directional Movement
            plus_dm = np.where(
                (highs[1:] - highs[:-1]) > (lows[:-1] - lows[1:]),
                np.maximum(highs[1:] - highs[:-1], 0),
                0
            )
            minus_dm = np.where(
                (lows[:-1] - lows[1:]) > (highs[1:] - highs[:-1]),
                np.maximum(lows[:-1] - lows[1:], 0),
                0
            )
            
            # Smoothed averages
            atr = np.mean(tr[-period:])
            plus_di = 100 * np.mean(plus_dm[-period:]) / atr if atr > 0 else 0
            minus_di = 100 * np.mean(minus_dm[-period:]) / atr if atr > 0 else 0
            
            # ADX
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di) if (plus_di + minus_di) > 0 else 0
            return dx
        except:
            return 0.0
    
    def _calculate_ma_alignment(self, prices: np.ndarray) -> float:
        """Calculate MA alignment score (-1 to 1)"""
        if len(prices) < 200:
            return 0.0
        
        ma20 = np.mean(prices[-20:])
        ma50 = np.mean(prices[-50:])
        ma100 = np.mean(prices[-100:])
        ma200 = np.mean(prices[-200:])
        
        # Perfect bullish alignment: price > ma20 > ma50 > ma100 > ma200
        # Perfect bearish alignment: price < ma20 < ma50 < ma100 < ma200
        
        score = 0.0
        current = prices[-1]
        
        if current > ma20:
            score += 0.25
        else:
            score -= 0.25
        
        if ma20 > ma50:
            score += 0.25
        else:
            score -= 0.25
        
        if ma50 > ma100:
            score += 0.25
        else:
            score -= 0.25
        
        if ma100 > ma200:
            score += 0.25
        else:
            score -= 0.25
        
        return score
    
    def _calculate_atr(self, highs: np.ndarray, lows: np.ndarray, 
                       closes: np.ndarray, period: int = 14) -> float:
        """Calculate Average True Range"""
        if len(highs) < period + 1:
            return 0.0
        
        tr = np.maximum(
            highs[1:] - lows[1:],
            np.maximum(
                np.abs(highs[1:] - closes[:-1]),
                np.abs(lows[1:] - closes[:-1])
            )
        )
        return np.mean(tr[-period:])
    
    def _calculate_range_score(self, prices: np.ndarray, highs: np.ndarray, 
                               lows: np.ndarray, lookback: int = 50) -> float:
        """Calculate how range-bound the market is (0-1)"""
        if len(prices) < lookback:
            return 0.0
        
        recent_high = np.max(highs[-lookback:])
        recent_low = np.min(lows[-lookback:])
        range_size = recent_high - recent_low
        
        if range_size == 0:
            return 1.0
        
        # Count how many times price touched the range boundaries
        upper_touches = np.sum(highs[-lookback:] > recent_high * 0.98)
        lower_touches = np.sum(lows[-lookback:] < recent_low * 1.02)
        
        # More touches = more range-bound
        touch_score = min(1.0, (upper_touches + lower_touches) / 10)
        
        # Calculate how much price stayed within the range
        mid_range = (recent_high + recent_low) / 2
        deviations = np.abs(prices[-lookback:] - mid_range) / (range_size / 2)
        containment_score = 1.0 - np.mean(np.minimum(deviations, 1.0))
        
        return (touch_score + containment_score) / 2
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate Relative Strength Index"""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_momentum_score(self, prices: np.ndarray) -> float:
        """Calculate overall momentum score (-1 to 1)"""
        if len(prices) < 20:
            return 0.0
        
        # Rate of change over different periods
        roc_5 = (prices[-1] - prices[-5]) / prices[-5] if prices[-5] > 0 else 0
        roc_10 = (prices[-1] - prices[-10]) / prices[-10] if prices[-10] > 0 else 0
        roc_20 = (prices[-1] - prices[-20]) / prices[-20] if prices[-20] > 0 else 0
        
        # Weighted average
        momentum = (roc_5 * 0.5 + roc_10 * 0.3 + roc_20 * 0.2) * 100
        
        # Normalize to -1 to 1
        return max(-1.0, min(1.0, momentum / 5))
    
    def _count_swing_points(self, highs: np.ndarray, lows: np.ndarray, 
                            lookback: int = 20) -> Tuple[int, int]:
        """Count higher highs and lower lows"""
        if len(highs) < lookback:
            return 0, 0
        
        higher_highs = 0
        lower_lows = 0
        
        for i in range(5, lookback):
            # Check for swing high
            if highs[-i] > highs[-i-1] and highs[-i] > highs[-i+1]:
                if i > 5 and highs[-i] > highs[-i-5]:
                    higher_highs += 1
            
            # Check for swing low
            if lows[-i] < lows[-i-1] and lows[-i] < lows[-i+1]:
                if i > 5 and lows[-i] < lows[-i-5]:
                    lower_lows += 1
        
        return higher_highs, lower_lows


class StrategySelector:
    """
    Selects optimal trading strategy based on market regime.
    
    Key principle: Match strategy to market conditions.
    """
    
    def __init__(self):
        self.regime_detector = RegimeDetector()
        self.strategy_configs = self._initialize_strategies()
        self.regime_strategy_map = self._initialize_regime_map()
        
        logger.info("StrategySelector initialized")
    
    def _initialize_strategies(self) -> Dict[TradingStrategy, StrategyConfig]:
        """Initialize strategy configurations"""
        return {
            TradingStrategy.TREND_FOLLOWING: StrategyConfig(
                name=TradingStrategy.TREND_FOLLOWING,
                min_confidence=0.65,
                position_size_multiplier=1.0,
                max_trades_per_day=2,
                preferred_timeframes=['H4', 'D1'],
                stop_loss_atr_multiplier=2.5,
                take_profit_atr_multiplier=4.0,
                trailing_enabled=True,
                partial_close_enabled=True,
                required_indicators=['adx', 'ma_alignment'],
                entry_filters={'adx_min': 25, 'ma_alignment_min': 0.5}
            ),
            TradingStrategy.MEAN_REVERSION: StrategyConfig(
                name=TradingStrategy.MEAN_REVERSION,
                min_confidence=0.70,
                position_size_multiplier=0.8,
                max_trades_per_day=3,
                preferred_timeframes=['H1', 'H4'],
                stop_loss_atr_multiplier=1.5,
                take_profit_atr_multiplier=2.0,
                trailing_enabled=False,
                partial_close_enabled=False,
                required_indicators=['rsi', 'bollinger'],
                entry_filters={'rsi_oversold': 30, 'rsi_overbought': 70}
            ),
            TradingStrategy.BREAKOUT: StrategyConfig(
                name=TradingStrategy.BREAKOUT,
                min_confidence=0.75,
                position_size_multiplier=0.7,
                max_trades_per_day=2,
                preferred_timeframes=['H1', 'H4'],
                stop_loss_atr_multiplier=1.5,
                take_profit_atr_multiplier=3.0,
                trailing_enabled=True,
                partial_close_enabled=True,
                required_indicators=['atr', 'range'],
                entry_filters={'consolidation_min_bars': 10}
            ),
            TradingStrategy.MOMENTUM: StrategyConfig(
                name=TradingStrategy.MOMENTUM,
                min_confidence=0.70,
                position_size_multiplier=0.9,
                max_trades_per_day=2,
                preferred_timeframes=['H4', 'D1'],
                stop_loss_atr_multiplier=2.0,
                take_profit_atr_multiplier=3.5,
                trailing_enabled=True,
                partial_close_enabled=True,
                required_indicators=['momentum', 'rsi'],
                entry_filters={'momentum_min': 0.3}
            ),
            TradingStrategy.RANGE_TRADING: StrategyConfig(
                name=TradingStrategy.RANGE_TRADING,
                min_confidence=0.70,
                position_size_multiplier=0.8,
                max_trades_per_day=4,
                preferred_timeframes=['M15', 'H1'],
                stop_loss_atr_multiplier=1.0,
                take_profit_atr_multiplier=1.5,
                trailing_enabled=False,
                partial_close_enabled=False,
                required_indicators=['support', 'resistance'],
                entry_filters={'range_score_min': 0.7}
            ),
            TradingStrategy.SIT_OUT: StrategyConfig(
                name=TradingStrategy.SIT_OUT,
                min_confidence=1.0,  # Never trade
                position_size_multiplier=0.0,
                max_trades_per_day=0,
                preferred_timeframes=[],
                stop_loss_atr_multiplier=0,
                take_profit_atr_multiplier=0,
                trailing_enabled=False,
                partial_close_enabled=False
            )
        }
    
    def _initialize_regime_map(self) -> Dict[MarketRegime, List[TradingStrategy]]:
        """Map regimes to appropriate strategies"""
        return {
            MarketRegime.STRONG_TREND_UP: [
                TradingStrategy.TREND_FOLLOWING,
                TradingStrategy.MOMENTUM
            ],
            MarketRegime.WEAK_TREND_UP: [
                TradingStrategy.TREND_FOLLOWING,
                TradingStrategy.BREAKOUT
            ],
            MarketRegime.RANGING: [
                TradingStrategy.MEAN_REVERSION,
                TradingStrategy.RANGE_TRADING
            ],
            MarketRegime.WEAK_TREND_DOWN: [
                TradingStrategy.TREND_FOLLOWING,
                TradingStrategy.BREAKOUT
            ],
            MarketRegime.STRONG_TREND_DOWN: [
                TradingStrategy.TREND_FOLLOWING,
                TradingStrategy.MOMENTUM
            ],
            MarketRegime.HIGH_VOLATILITY: [
                TradingStrategy.SIT_OUT  # Don't trade high volatility
            ],
            MarketRegime.LOW_VOLATILITY: [
                TradingStrategy.BREAKOUT,
                TradingStrategy.RANGE_TRADING
            ],
            MarketRegime.BREAKOUT: [
                TradingStrategy.BREAKOUT,
                TradingStrategy.MOMENTUM
            ],
            MarketRegime.UNKNOWN: [
                TradingStrategy.SIT_OUT
            ]
        }
    
    def select_strategy(self, indicators: RegimeIndicators, 
                       symbol: str = "") -> StrategyRecommendation:
        """
        Select optimal strategy based on current market conditions.
        
        Args:
            indicators: Current regime indicators
            symbol: Trading symbol (for symbol-specific adjustments)
            
        Returns:
            StrategyRecommendation with primary and secondary strategies
        """
        # Detect regime
        regime, regime_confidence = self.regime_detector.detect_regime(indicators)
        
        # Get strategies for this regime
        strategies = self.regime_strategy_map.get(regime, [TradingStrategy.SIT_OUT])
        
        primary_strategy = strategies[0]
        secondary_strategy = strategies[1] if len(strategies) > 1 else None
        
        # Get strategy config
        config = self.strategy_configs[primary_strategy]
        
        # Calculate position size multiplier based on regime confidence
        position_multiplier = config.position_size_multiplier * regime_confidence
        
        # Adjust for high volatility
        if indicators.atr_percentile > 70:
            position_multiplier *= 0.7
        
        # Build reasoning
        reasoning = self._build_reasoning(regime, indicators, primary_strategy)
        
        # Check for warnings
        warnings = self._check_warnings(indicators, regime)
        
        return StrategyRecommendation(
            primary_strategy=primary_strategy,
            secondary_strategy=secondary_strategy,
            regime=regime,
            confidence=regime_confidence,
            position_size_multiplier=position_multiplier,
            reasoning=reasoning,
            indicators={
                'adx': indicators.adx,
                'ma_alignment': indicators.ma_alignment,
                'atr_percentile': indicators.atr_percentile,
                'rsi': indicators.rsi,
                'momentum': indicators.momentum_score,
                'range_score': indicators.range_bound_score
            },
            warnings=warnings
        )
    
    def _build_reasoning(self, regime: MarketRegime, indicators: RegimeIndicators,
                        strategy: TradingStrategy) -> str:
        """Build human-readable reasoning for strategy selection"""
        parts = []
        
        # Regime explanation
        regime_explanations = {
            MarketRegime.STRONG_TREND_UP: f"Strong uptrend detected (ADX={indicators.adx:.1f}, MA alignment={indicators.ma_alignment:.2f})",
            MarketRegime.WEAK_TREND_UP: f"Weak uptrend (ADX={indicators.adx:.1f})",
            MarketRegime.RANGING: f"Range-bound market (range score={indicators.range_bound_score:.2f})",
            MarketRegime.WEAK_TREND_DOWN: f"Weak downtrend (ADX={indicators.adx:.1f})",
            MarketRegime.STRONG_TREND_DOWN: f"Strong downtrend detected (ADX={indicators.adx:.1f}, MA alignment={indicators.ma_alignment:.2f})",
            MarketRegime.HIGH_VOLATILITY: f"High volatility environment (ATR percentile={indicators.atr_percentile:.0f})",
            MarketRegime.LOW_VOLATILITY: f"Low volatility - potential breakout setup (ATR percentile={indicators.atr_percentile:.0f})",
            MarketRegime.BREAKOUT: "Breakout conditions forming",
            MarketRegime.UNKNOWN: "Unclear market conditions"
        }
        parts.append(regime_explanations.get(regime, "Unknown regime"))
        
        # Strategy explanation
        strategy_explanations = {
            TradingStrategy.TREND_FOLLOWING: "Using trend-following to ride the move",
            TradingStrategy.MEAN_REVERSION: "Using mean reversion to fade extremes",
            TradingStrategy.BREAKOUT: "Watching for breakout entry",
            TradingStrategy.MOMENTUM: "Following momentum",
            TradingStrategy.RANGE_TRADING: "Trading the range boundaries",
            TradingStrategy.SIT_OUT: "Sitting out - conditions unfavorable"
        }
        parts.append(strategy_explanations.get(strategy, ""))
        
        return ". ".join(parts)
    
    def _check_warnings(self, indicators: RegimeIndicators, 
                       regime: MarketRegime) -> List[str]:
        """Check for warning conditions"""
        warnings = []
        
        if indicators.atr_percentile > 80:
            warnings.append("Extreme volatility - reduce position size")
        
        if indicators.rsi > 80:
            warnings.append("RSI overbought - potential reversal")
        elif indicators.rsi < 20:
            warnings.append("RSI oversold - potential reversal")
        
        if regime == MarketRegime.UNKNOWN:
            warnings.append("Unclear regime - consider sitting out")
        
        if abs(indicators.momentum_score) > 0.8:
            warnings.append("Extreme momentum - late entry risk")
        
        return warnings
    
    def get_strategy_config(self, strategy: TradingStrategy) -> StrategyConfig:
        """Get configuration for a specific strategy"""
        return self.strategy_configs.get(strategy, self.strategy_configs[TradingStrategy.SIT_OUT])


# Singleton instance
_strategy_selector: Optional[StrategySelector] = None


def get_strategy_selector() -> StrategySelector:
    """Get singleton strategy selector instance"""
    global _strategy_selector
    if _strategy_selector is None:
        _strategy_selector = StrategySelector()
    return _strategy_selector


def select_strategy_for_market(prices: np.ndarray, highs: np.ndarray, 
                               lows: np.ndarray, symbol: str = "") -> StrategyRecommendation:
    """
    Convenience function to select strategy from price data.
    
    Args:
        prices: Close prices array
        highs: High prices array
        lows: Low prices array
        symbol: Trading symbol
        
    Returns:
        StrategyRecommendation
    """
    selector = get_strategy_selector()
    indicators = selector.regime_detector.calculate_indicators(prices, highs, lows)
    return selector.select_strategy(indicators, symbol)


def should_trade_in_regime(regime: MarketRegime) -> bool:
    """Check if trading is recommended in the given regime"""
    no_trade_regimes = [MarketRegime.HIGH_VOLATILITY, MarketRegime.UNKNOWN]
    return regime not in no_trade_regimes
