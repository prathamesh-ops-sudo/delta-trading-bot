"""
Professional Price Action Analysis
Support/Resistance, Confluence Zones, Market Structure
Built like a veteran trader - Warren Buffett + BlackRock Aladdin style
"""
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional
from config import Config

logger = logging.getLogger(__name__)


class PriceActionAnalyzer:
    """Analyze price action like a professional trader"""

    def __init__(self):
        self.support_levels = []
        self.resistance_levels = []
        self.confluence_zones = []

    def analyze(self, df: pd.DataFrame, higher_tf_df: Optional[pd.DataFrame] = None) -> Dict:
        """
        Complete price action analysis

        Args:
            df: Primary timeframe DataFrame (15m)
            higher_tf_df: Higher timeframe DataFrame (1h) for trend context

        Returns:
            Dictionary with price action signals and context
        """
        try:
            current_price = float(df['close'].iloc[-1])

            # Detect support and resistance levels
            self.support_levels, self.resistance_levels = self._detect_support_resistance(df)

            # Find confluence zones
            self.confluence_zones = self._find_confluence_zones(df)

            # Analyze market structure
            structure = self._analyze_market_structure(df)

            # Determine higher timeframe trend
            higher_trend = self._analyze_higher_tf_trend(higher_tf_df if higher_tf_df is not None else df)

            # Check proximity to key levels
            nearest_support = self._find_nearest_level(current_price, self.support_levels, 'below')
            nearest_resistance = self._find_nearest_level(current_price, self.resistance_levels, 'above')

            # Confluence analysis at current price
            confluence_score = self._calculate_confluence_at_price(current_price, df)

            # Generate price action signal
            signal = self._generate_price_action_signal(
                df,
                current_price,
                nearest_support,
                nearest_resistance,
                structure,
                higher_trend,
                confluence_score
            )

            result = {
                'signal': signal['direction'],
                'confidence': signal['confidence'],
                'confluence_score': confluence_score,
                'confluence_count': signal['confluence_count'],
                'support_levels': self.support_levels[:3],  # Top 3
                'resistance_levels': self.resistance_levels[:3],  # Top 3
                'nearest_support': nearest_support,
                'nearest_resistance': nearest_resistance,
                'market_structure': structure,
                'higher_tf_trend': higher_trend,
                'rationale': signal['rationale'],
                'confluence_details': signal['confluence_details'],
                'entry_zone': signal['entry_zone'],
                'stop_loss': signal['stop_loss'],
                'take_profit': signal['take_profit'],
                'risk_reward': signal['risk_reward']
            }

            logger.info(f"Price Action: {signal['direction']} | Confluence: {confluence_score:.1f} | Trend: {higher_trend}")

            return result

        except Exception as e:
            logger.error(f"Error in price action analysis: {e}", exc_info=True)
            return self._get_neutral_result()

    def _detect_support_resistance(self, df: pd.DataFrame) -> Tuple[List[float], List[float]]:
        """
        Detect significant support and resistance levels

        Uses swing highs/lows with multiple touches confirmation
        """
        try:
            lookback = min(Config.SR_LOOKBACK_PERIODS, len(df))
            tolerance = Config.SR_TOUCH_TOLERANCE

            # Find swing highs and lows
            swing_highs = []
            swing_lows = []

            for i in range(Config.SWING_HIGH_LOW_PERIODS, lookback - Config.SWING_HIGH_LOW_PERIODS):
                # Swing high: higher than surrounding candles
                if df['high'].iloc[-i] == df['high'].iloc[-i - Config.SWING_HIGH_LOW_PERIODS:-i + Config.SWING_HIGH_LOW_PERIODS + 1].max():
                    swing_highs.append(float(df['high'].iloc[-i]))

                # Swing low: lower than surrounding candles
                if df['low'].iloc[-i] == df['low'].iloc[-i - Config.SWING_HIGH_LOW_PERIODS:-i + Config.SWING_HIGH_LOW_PERIODS + 1].min():
                    swing_lows.append(float(df['low'].iloc[-i]))

            # Cluster swing levels to find strong S/R
            resistance_levels = self._cluster_levels(swing_highs, tolerance)
            support_levels = self._cluster_levels(swing_lows, tolerance)

            # Filter by number of touches
            resistance_levels = self._filter_by_touches(resistance_levels, df, 'resistance', tolerance)
            support_levels = self._filter_by_touches(support_levels, df, 'support', tolerance)

            # Sort by strength (number of touches)
            resistance_levels = sorted(resistance_levels, key=lambda x: x['touches'], reverse=True)
            support_levels = sorted(support_levels, key=lambda x: x['touches'], reverse=True)

            # Extract just the price levels
            resistance_prices = [level['price'] for level in resistance_levels[:5]]
            support_prices = [level['price'] for level in support_levels[:5]]

            logger.debug(f"Detected {len(support_prices)} support and {len(resistance_prices)} resistance levels")

            return support_prices, resistance_prices

        except Exception as e:
            logger.warning(f"Error detecting S/R levels: {e}")
            return [], []

    def _cluster_levels(self, levels: List[float], tolerance: float) -> List[float]:
        """Cluster nearby levels into single levels"""
        if not levels:
            return []

        levels = sorted(levels)
        clustered = []
        current_cluster = [levels[0]]

        for level in levels[1:]:
            if abs(level - current_cluster[-1]) / current_cluster[-1] <= tolerance:
                current_cluster.append(level)
            else:
                clustered.append(np.mean(current_cluster))
                current_cluster = [level]

        clustered.append(np.mean(current_cluster))
        return clustered

    def _filter_by_touches(self, levels: List[float], df: pd.DataFrame, level_type: str, tolerance: float) -> List[Dict]:
        """Filter levels by number of touches"""
        result = []

        for level in levels:
            touches = 0

            for i in range(len(df)):
                high = df['high'].iloc[i]
                low = df['low'].iloc[i]

                # Check if price touched this level
                if level_type == 'resistance':
                    if abs(high - level) / level <= tolerance:
                        touches += 1
                else:  # support
                    if abs(low - level) / level <= tolerance:
                        touches += 1

            if touches >= Config.SR_STRENGTH_MIN_TOUCHES:
                result.append({'price': level, 'touches': touches})

        return result

    def _find_confluence_zones(self, df: pd.DataFrame) -> List[Dict]:
        """Find zones where multiple factors align"""
        zones = []
        zone_size = Config.CONFLUENCE_ZONE_SIZE

        # Combine all potential levels
        all_levels = self.support_levels + self.resistance_levels

        # Add Fibonacci levels (from recent swing)
        if len(df) >= 50:
            recent_high = df['high'].iloc[-50:].max()
            recent_low = df['low'].iloc[-50:].min()
            diff = recent_high - recent_low

            fib_levels = [
                recent_low + diff * 0.236,
                recent_low + diff * 0.382,
                recent_low + diff * 0.5,
                recent_low + diff * 0.618,
                recent_low + diff * 0.786
            ]
            all_levels.extend(fib_levels)

        # Find zones where multiple levels cluster
        for level in all_levels:
            nearby_count = sum(1 for l in all_levels if abs(l - level) / level <= zone_size)

            if nearby_count >= 2:
                zones.append({
                    'price': level,
                    'strength': nearby_count,
                    'type': 'confluence'
                })

        # Remove duplicates and sort by strength
        zones = sorted(zones, key=lambda x: x['strength'], reverse=True)

        return zones[:5]  # Top 5 confluence zones

    def _analyze_market_structure(self, df: pd.DataFrame) -> Dict:
        """Analyze market structure (higher highs, lower lows, etc.)"""
        try:
            lookback = min(50, len(df))

            # Find recent swing points
            swing_highs = []
            swing_lows = []

            for i in range(5, lookback - 5):
                if df['high'].iloc[-i] == df['high'].iloc[-i-5:-i+5].max():
                    swing_highs.append({'price': df['high'].iloc[-i], 'index': i})

                if df['low'].iloc[-i] == df['low'].iloc[-i-5:-i+5].min():
                    swing_lows.append({'price': df['low'].iloc[-i], 'index': i})

            # Determine structure
            structure_type = 'RANGE'

            if len(swing_highs) >= 2 and len(swing_lows) >= 2:
                # Check for higher highs and higher lows (uptrend)
                if swing_highs[0]['price'] > swing_highs[1]['price'] and \
                   swing_lows[0]['price'] > swing_lows[1]['price']:
                    structure_type = 'UPTREND'

                # Check for lower highs and lower lows (downtrend)
                elif swing_highs[0]['price'] < swing_highs[1]['price'] and \
                     swing_lows[0]['price'] < swing_lows[1]['price']:
                    structure_type = 'DOWNTREND'

            return {
                'type': structure_type,
                'swing_highs': swing_highs[:3],
                'swing_lows': swing_lows[:3]
            }

        except Exception as e:
            logger.warning(f"Error analyzing market structure: {e}")
            return {'type': 'UNKNOWN', 'swing_highs': [], 'swing_lows': []}

    def _analyze_higher_tf_trend(self, df: pd.DataFrame) -> str:
        """Determine higher timeframe trend using SMAs"""
        try:
            if len(df) < Config.TREND_SMA_SLOW:
                return 'NEUTRAL'

            close = df['close'].iloc[-1]
            sma_fast = df['close'].iloc[-Config.TREND_SMA_FAST:].mean()
            sma_slow = df['close'].iloc[-Config.TREND_SMA_SLOW:].mean()

            # Strong uptrend: price above both SMAs, fast above slow
            if close > sma_fast and sma_fast > sma_slow:
                trend_strength = (sma_fast - sma_slow) / sma_slow
                return 'BULLISH' if trend_strength >= Config.TREND_STRENGTH_MIN else 'NEUTRAL'

            # Strong downtrend: price below both SMAs, fast below slow
            elif close < sma_fast and sma_fast < sma_slow:
                trend_strength = (sma_slow - sma_fast) / sma_slow
                return 'BEARISH' if trend_strength >= Config.TREND_STRENGTH_MIN else 'NEUTRAL'

            else:
                return 'NEUTRAL'

        except Exception as e:
            logger.warning(f"Error analyzing higher TF trend: {e}")
            return 'NEUTRAL'

    def _find_nearest_level(self, price: float, levels: List[float], direction: str) -> Optional[float]:
        """Find nearest support/resistance level"""
        if not levels:
            return None

        if direction == 'below':
            below_levels = [l for l in levels if l < price]
            return max(below_levels) if below_levels else None
        else:  # above
            above_levels = [l for l in levels if l > price]
            return min(above_levels) if above_levels else None

    def _calculate_confluence_at_price(self, price: float, df: pd.DataFrame) -> float:
        """Calculate confluence score at current price"""
        score = 0.0
        tolerance = Config.CONFLUENCE_ZONE_SIZE

        # Check proximity to S/R levels
        for level in self.support_levels + self.resistance_levels:
            if abs(level - price) / price <= tolerance:
                score += 2.0

        # Check proximity to confluence zones
        for zone in self.confluence_zones:
            if abs(zone['price'] - price) / price <= tolerance:
                score += zone['strength']

        # Check moving averages
        if len(df) >= 50:
            sma_20 = df['close'].iloc[-20:].mean()
            sma_50 = df['close'].iloc[-50:].mean()

            if abs(sma_20 - price) / price <= tolerance:
                score += 1.0
            if abs(sma_50 - price) / price <= tolerance:
                score += 1.5

        return score

    def _generate_price_action_signal(
        self,
        df: pd.DataFrame,
        current_price: float,
        nearest_support: Optional[float],
        nearest_resistance: Optional[float],
        structure: Dict,
        higher_trend: str,
        confluence_score: float
    ) -> Dict:
        """
        Generate trading signal based on price action

        This is the core logic - trade like a veteran
        """
        try:
            signal = 'NEUTRAL'
            confidence = 0.5
            rationale = []
            confluence_details = []
            confluence_count = 0

            # Get technical indicators
            latest = df.iloc[-1]
            rsi = latest.get('rsi_14', 50)

            # --- BUY SETUP DETECTION ---
            if higher_trend == 'BULLISH' and structure['type'] in ['UPTREND', 'RANGE']:
                potential_buy = True
                buy_confidence = 0.0

                # 1. Confluence: Near support in uptrend
                if nearest_support and abs(current_price - nearest_support) / current_price <= 0.01:
                    buy_confidence += 0.25
                    confluence_count += 1
                    rationale.append("Price at key support level")
                    confluence_details.append(f"Support at ${nearest_support:,.2f}")

                # 2. Confluence: RSI oversold/neutral
                if rsi < 50:
                    buy_confidence += 0.20
                    confluence_count += 1
                    rationale.append(f"RSI showing buying opportunity ({rsi:.1f})")
                    confluence_details.append(f"RSI: {rsi:.1f} (bullish)")

                # 3. Confluence: Bullish candle pattern
                if latest['close'] > latest['open']:
                    buy_confidence += 0.15
                    confluence_count += 1
                    rationale.append("Bullish candle formation")
                    confluence_details.append("Green candle close")

                # 4. Confluence: Higher timeframe trend
                buy_confidence += 0.20
                confluence_count += 1
                rationale.append("Aligned with higher timeframe uptrend")
                confluence_details.append(f"1H Trend: {higher_trend}")

                # 5. Confluence: High confluence zone
                if confluence_score >= 3.0:
                    buy_confidence += 0.20
                    confluence_count += 1
                    rationale.append("Multiple factors converging at this level")
                    confluence_details.append(f"Confluence score: {confluence_score:.1f}")

                # Check if meets minimum requirements
                if buy_confidence >= Config.MIN_SIGNAL_CONFIDENCE and \
                   confluence_count >= Config.CONFLUENCE_REQUIRED:
                    signal = 'BUY'
                    confidence = min(buy_confidence, 0.95)

                    # Calculate entry, stop, and target
                    entry_zone = (current_price * 0.998, current_price * 1.002)
                    stop_loss = nearest_support * 0.995 if nearest_support else current_price * 0.98

                    # Target based on structure
                    if nearest_resistance:
                        take_profit = nearest_resistance * 0.995
                    else:
                        take_profit = current_price * 1.04  # 4% if no clear resistance

                    risk = current_price - stop_loss
                    reward = take_profit - current_price
                    risk_reward = reward / risk if risk > 0 else 0

                    # Only take trade if R/R >= 2:1
                    if risk_reward < Config.RISK_REWARD_RATIO:
                        signal = 'NEUTRAL'
                        confidence = 0.5
                        rationale.append(f"Rejected: R/R too low ({risk_reward:.1f}:1)")

            # --- SELL SETUP DETECTION ---
            elif higher_trend == 'BEARISH' and structure['type'] in ['DOWNTREND', 'RANGE']:
                potential_sell = True
                sell_confidence = 0.0

                # 1. Confluence: Near resistance in downtrend
                if nearest_resistance and abs(current_price - nearest_resistance) / current_price <= 0.01:
                    sell_confidence += 0.25
                    confluence_count += 1
                    rationale.append("Price at key resistance level")
                    confluence_details.append(f"Resistance at ${nearest_resistance:,.2f}")

                # 2. Confluence: RSI overbought/neutral
                if rsi > 50:
                    sell_confidence += 0.20
                    confluence_count += 1
                    rationale.append(f"RSI showing selling opportunity ({rsi:.1f})")
                    confluence_details.append(f"RSI: {rsi:.1f} (bearish)")

                # 3. Confluence: Bearish candle pattern
                if latest['close'] < latest['open']:
                    sell_confidence += 0.15
                    confluence_count += 1
                    rationale.append("Bearish candle formation")
                    confluence_details.append("Red candle close")

                # 4. Confluence: Higher timeframe trend
                sell_confidence += 0.20
                confluence_count += 1
                rationale.append("Aligned with higher timeframe downtrend")
                confluence_details.append(f"1H Trend: {higher_trend}")

                # 5. Confluence: High confluence zone
                if confluence_score >= 3.0:
                    sell_confidence += 0.20
                    confluence_count += 1
                    rationale.append("Multiple factors converging at this level")
                    confluence_details.append(f"Confluence score: {confluence_score:.1f}")

                # Check if meets minimum requirements
                if sell_confidence >= Config.MIN_SIGNAL_CONFIDENCE and \
                   confluence_count >= Config.CONFLUENCE_REQUIRED:
                    signal = 'SELL'
                    confidence = min(sell_confidence, 0.95)

                    # Calculate entry, stop, and target
                    entry_zone = (current_price * 0.998, current_price * 1.002)
                    stop_loss = nearest_resistance * 1.005 if nearest_resistance else current_price * 1.02

                    # Target based on structure
                    if nearest_support:
                        take_profit = nearest_support * 1.005
                    else:
                        take_profit = current_price * 0.96  # 4% if no clear support

                    risk = stop_loss - current_price
                    reward = current_price - take_profit
                    risk_reward = reward / risk if risk > 0 else 0

                    # Only take trade if R/R >= 2:1
                    if risk_reward < Config.RISK_REWARD_RATIO:
                        signal = 'NEUTRAL'
                        confidence = 0.5
                        rationale.append(f"Rejected: R/R too low ({risk_reward:.1f}:1)")

            # Build result
            if signal == 'NEUTRAL':
                return {
                    'direction': 'NEUTRAL',
                    'confidence': 0.5,
                    'confluence_count': 0,
                    'rationale': ["No high-quality setup detected"],
                    'confluence_details': [],
                    'entry_zone': (0, 0),
                    'stop_loss': 0,
                    'take_profit': 0,
                    'risk_reward': 0
                }

            return {
                'direction': signal,
                'confidence': confidence,
                'confluence_count': confluence_count,
                'rationale': rationale,
                'confluence_details': confluence_details,
                'entry_zone': entry_zone,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'risk_reward': risk_reward
            }

        except Exception as e:
            logger.error(f"Error generating price action signal: {e}", exc_info=True)
            return self._get_neutral_signal()

    def _get_neutral_result(self) -> Dict:
        """Return neutral result on error"""
        return {
            'signal': 'NEUTRAL',
            'confidence': 0.5,
            'confluence_score': 0,
            'confluence_count': 0,
            'support_levels': [],
            'resistance_levels': [],
            'nearest_support': None,
            'nearest_resistance': None,
            'market_structure': {'type': 'UNKNOWN'},
            'higher_tf_trend': 'NEUTRAL',
            'rationale': ['Analysis error'],
            'confluence_details': [],
            'entry_zone': (0, 0),
            'stop_loss': 0,
            'take_profit': 0,
            'risk_reward': 0
        }

    def _get_neutral_signal(self) -> Dict:
        """Return neutral signal"""
        return {
            'direction': 'NEUTRAL',
            'confidence': 0.5,
            'confluence_count': 0,
            'rationale': ['No setup'],
            'confluence_details': [],
            'entry_zone': (0, 0),
            'stop_loss': 0,
            'take_profit': 0,
            'risk_reward': 0
        }
