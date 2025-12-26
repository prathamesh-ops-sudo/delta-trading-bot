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
        
        # Decision thresholds
        self.min_confidence = 0.55
        self.min_rr_ratio = 1.5
        
        # Strategy weights (updated by agentic system)
        self.strategy_weights = {
            'trend_following': 0.3,
            'mean_reversion': 0.25,
            'fvg_entry': 0.25,
            'liquidity_sweep': 0.2
        }
    
    def analyze_market(self, symbol: str, mtf_data: Dict[str, pd.DataFrame],
                       account_balance: float) -> Optional[TradingSignal]:
        """Comprehensive market analysis like a veteran trader"""
        
        # Get primary timeframe data (H1 for analysis)
        primary_tf = 'H1'
        if primary_tf not in mtf_data or mtf_data[primary_tf].empty:
            primary_tf = list(mtf_data.keys())[0] if mtf_data else None
        
        if not primary_tf:
            return None
        
        df = mtf_data[primary_tf]
        if len(df) < 100:
            return None
        
        # 1. Get regime context
        regime = regime_manager.detect_regime(df)
        
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
        
        # 4. Detect FVGs and liquidity sweeps
        pip_value = 0.01 if 'JPY' in symbol else 0.0001
        fvgs = self.fvg_detector.detect_fvg(df, pip_value)
        sweeps = self.sweep_detector.detect_sweeps(df, pip_value)
        
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
        
        if not signals:
            return None
        
        # 6. Select best signal based on weights and confidence
        best_signal = self._select_best_signal(signals)
        if not best_signal:
            return None
        
        strategy, (direction, base_confidence, entry_reason) = best_signal
        
        # 7. Calculate final confidence with all factors
        confidence = self._calculate_final_confidence(
            base_confidence, mtf_alignment, regime, adx_value, rsi
        )
        
        if confidence < self.min_confidence:
            return None
        
        # 8. Get trading parameters from agentic system
        trading_params = agentic_system.get_trading_parameters()
        aggression = trading_params.get('aggression_level', 0.5)
        
        # 9. Calculate entry, SL, TP
        entry_price = current_price
        sl_distance = atr * 2  # 2 ATR stop loss
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
        
        # 11. Calculate position size
        risk_amount = agentic_system.calculate_position_size(
            {'confidence': confidence, 'strategy': strategy},
            account_balance
        )
        position_size = risk_amount / sl_distance if sl_distance > 0 else 0
        
        # 12. Determine trailing stop
        trailing_enabled = confidence > 0.7 and adx_value > 30
        trailing_distance = self.trailing_manager.calculate_trailing_distance(
            atr, adx_value, aggression
        ) if trailing_enabled else 0
        
        # 13. Create final signal
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
