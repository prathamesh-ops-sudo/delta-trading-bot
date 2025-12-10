"""
Professional Signal Generator
Price action PRIMARY, ML as FILTER
Built like a veteran trader - Warren Buffett + BlackRock Aladdin style
"""
import numpy as np
import pandas as pd
import logging
from typing import Dict, Optional
from datetime import datetime

from config import Config
from price_action_analyzer import PriceActionAnalyzer
from signal_generator import SignalGenerator

logger = logging.getLogger(__name__)


class ProfessionalSignalGenerator:
    """
    Generate trading signals like a professional trader

    Philosophy:
    1. Price action is PRIMARY signal source (not ML)
    2. ML models act as FILTERS (confirm or reject)
    3. Require high confluence (3+ confirmations)
    4. One solid take-profit target
    5. Minimum 2:1 risk/reward
    6. Quality over quantity
    """

    def __init__(self):
        self.price_action = PriceActionAnalyzer()
        self.ml_signal_gen = SignalGenerator()

        self.last_buy_signal_time = None
        self.last_sell_signal_time = None
        self.buy_signal_count = 0
        self.sell_signal_count = 0

    def load_models(self):
        """Load ML models (used as filters)"""
        self.ml_signal_gen.load_models()
        logger.info("✓ ML models loaded (will be used as filters)")

    def models_exist(self) -> bool:
        """Check if ML models exist"""
        return self.ml_signal_gen.models_exist()

    def train_models(self, df: pd.DataFrame, feature_cols: list, epochs: int = 50, batch_size: int = 32):
        """Train ML models"""
        self.ml_signal_gen.train(df, feature_cols, epochs, batch_size)

    def save_models(self):
        """Save ML models"""
        self.ml_signal_gen.save_models()

    def predict(
        self,
        df: pd.DataFrame,
        higher_tf_df: Optional[pd.DataFrame] = None,
        symbol: str = "BTCUSD"
    ) -> Dict:
        """
        Generate professional trading signal

        Args:
            df: Primary timeframe DataFrame (15m)
            higher_tf_df: Higher timeframe DataFrame (1h) for trend context
            symbol: Delta Exchange symbol (BTCUSD, ETHUSD, etc.)

        Returns:
            Signal dictionary with all details
        """
        try:
            current_price = float(df['close'].iloc[-1])
            timestamp = df['timestamp'].iloc[-1]

            # ============================================================
            # STEP 1: PRICE ACTION ANALYSIS (PRIMARY)
            # ============================================================
            logger.debug(f"[{symbol}] Running price action analysis...")
            price_action_result = self.price_action.analyze(df, higher_tf_df)

            pa_signal = price_action_result['signal']
            pa_confidence = price_action_result['confidence']
            confluence_count = price_action_result['confluence_count']

            logger.info(f"[{symbol}] Price Action: {pa_signal} ({pa_confidence:.0%}) | Confluence: {confluence_count}")

            # If price action says NEUTRAL, we're done (no setup)
            if pa_signal == 'NEUTRAL':
                return self._build_neutral_signal(symbol, current_price, timestamp, price_action_result)

            # ============================================================
            # STEP 2: ML FILTER (CONFIRMATION)
            # ============================================================
            if Config.ML_AS_FILTER:
                logger.debug(f"[{symbol}] Running ML filter...")
                ml_result = self.ml_signal_gen.predict(df)

                ml_signal = ml_result['signal']
                ml_ensemble = ml_result['ensemble_score']

                # ML must AGREE with price action
                ml_agrees = False

                if pa_signal == 'BUY':
                    # For BUY, ML ensemble should be >= threshold (bullish)
                    if ml_ensemble >= Config.ML_FILTER_THRESHOLD:
                        ml_agrees = True
                        logger.info(f"[{symbol}] ML CONFIRMS buy signal ({ml_ensemble:.0%})")
                    else:
                        logger.warning(f"[{symbol}] ML REJECTS buy signal ({ml_ensemble:.0%} < {Config.ML_FILTER_THRESHOLD:.0%})")

                elif pa_signal == 'SELL':
                    # For SELL, ML ensemble should be <= (1 - threshold) (bearish)
                    if ml_ensemble <= (1 - Config.ML_FILTER_THRESHOLD):
                        ml_agrees = True
                        logger.info(f"[{symbol}] ML CONFIRMS sell signal ({ml_ensemble:.0%})")
                    else:
                        logger.warning(f"[{symbol}] ML REJECTS sell signal ({ml_ensemble:.0%} > {1 - Config.ML_FILTER_THRESHOLD:.0%})")

                # If ML doesn't agree, REJECT the signal
                if not ml_agrees:
                    return self._build_rejected_signal(
                        symbol,
                        current_price,
                        timestamp,
                        price_action_result,
                        ml_result,
                        "ML filter did not confirm price action signal"
                    )

            # ============================================================
            # STEP 3: BIAS CORRECTION (Balance buy/sell signals)
            # ============================================================
            if Config.SIGNAL_BIAS_CORRECTION:
                total_signals = self.buy_signal_count + self.sell_signal_count

                if total_signals >= 10:  # Need minimum sample
                    buy_ratio = self.buy_signal_count / total_signals
                    sell_ratio = self.sell_signal_count / total_signals

                    # If too many sells, raise bar for sell signals
                    if pa_signal == 'SELL' and sell_ratio > 0.6:
                        if pa_confidence < 0.85:
                            logger.warning(f"[{symbol}] REJECTED SELL - bias correction (too many sells: {sell_ratio:.0%})")
                            return self._build_neutral_signal(symbol, current_price, timestamp, price_action_result)

                    # If too many buys, raise bar for buy signals
                    elif pa_signal == 'BUY' and buy_ratio > 0.6:
                        if pa_confidence < 0.85:
                            logger.warning(f"[{symbol}] REJECTED BUY - bias correction (too many buys: {buy_ratio:.0%})")
                            return self._build_neutral_signal(symbol, current_price, timestamp, price_action_result)

            # ============================================================
            # STEP 4: BUILD FINAL SIGNAL
            # ============================================================

            # Update signal counters
            if pa_signal == 'BUY':
                self.buy_signal_count += 1
                self.last_buy_signal_time = datetime.now()
            elif pa_signal == 'SELL':
                self.sell_signal_count += 1
                self.last_sell_signal_time = datetime.now()

            # Get technical indicators for context
            latest = df.iloc[-1]
            rsi_14 = float(latest.get('rsi_14', 50))
            macd_hist = float(latest.get('macd_hist', 0))
            atr = float(latest.get('atr_14', current_price * 0.02))

            # Build comprehensive signal
            signal_data = {
                # Core signal
                'signal': pa_signal,
                'confidence': pa_confidence,
                'symbol': symbol,
                'price': current_price,
                'timestamp': timestamp,

                # Price action details
                'confluence_count': confluence_count,
                'confluence_score': price_action_result['confluence_score'],
                'rationale': price_action_result['rationale'],
                'confluence_details': price_action_result['confluence_details'],

                # Market context
                'market_structure': price_action_result['market_structure']['type'],
                'higher_tf_trend': price_action_result['higher_tf_trend'],
                'support_levels': price_action_result['support_levels'],
                'resistance_levels': price_action_result['resistance_levels'],
                'nearest_support': price_action_result['nearest_support'],
                'nearest_resistance': price_action_result['nearest_resistance'],

                # Entry and exits (SINGLE TP)
                'entry_price': current_price,
                'entry_zone_low': price_action_result['entry_zone'][0],
                'entry_zone_high': price_action_result['entry_zone'][1],
                'stop_loss': price_action_result['stop_loss'],
                'take_profit': price_action_result['take_profit'],  # Single TP only
                'risk_reward': price_action_result['risk_reward'],

                # Technical indicators (for context)
                'rsi_14': rsi_14,
                'macd_hist': macd_hist,
                'atr': atr,

                # ML scores (supporting info)
                'ml_ensemble': ml_result['ensemble_score'] if Config.ML_AS_FILTER else 0.5,
                'ml_lstm': ml_result['lstm_score'] if Config.ML_AS_FILTER else 0.5,
                'ml_rf': ml_result['rf_score'] if Config.ML_AS_FILTER else 0.5,
            }

            logger.info(f"✅ [{symbol}] {pa_signal} SIGNAL GENERATED | Confidence: {pa_confidence:.0%} | R/R: {price_action_result['risk_reward']:.1f}:1")

            return signal_data

        except Exception as e:
            logger.error(f"Error generating professional signal for {symbol}: {e}", exc_info=True)
            return self._build_error_signal(symbol, current_price if 'current_price' in locals() else 0, datetime.now())

    def _build_neutral_signal(
        self,
        symbol: str,
        current_price: float,
        timestamp: datetime,
        price_action_result: Dict
    ) -> Dict:
        """Build neutral signal (no setup detected)"""
        return {
            'signal': 'NEUTRAL',
            'confidence': 0.5,
            'symbol': symbol,
            'price': current_price,
            'timestamp': timestamp,
            'confluence_count': 0,
            'confluence_score': price_action_result.get('confluence_score', 0),
            'rationale': price_action_result.get('rationale', ['No high-quality setup']),
            'confluence_details': [],
            'market_structure': price_action_result.get('market_structure', {}).get('type', 'UNKNOWN'),
            'higher_tf_trend': price_action_result.get('higher_tf_trend', 'NEUTRAL'),
            'support_levels': price_action_result.get('support_levels', []),
            'resistance_levels': price_action_result.get('resistance_levels', []),
            'nearest_support': price_action_result.get('nearest_support'),
            'nearest_resistance': price_action_result.get('nearest_resistance'),
            'entry_price': 0,
            'entry_zone_low': 0,
            'entry_zone_high': 0,
            'stop_loss': 0,
            'take_profit': 0,
            'risk_reward': 0,
            'rsi_14': 0,
            'macd_hist': 0,
            'atr': 0,
            'ml_ensemble': 0.5,
            'ml_lstm': 0.5,
            'ml_rf': 0.5,
        }

    def _build_rejected_signal(
        self,
        symbol: str,
        current_price: float,
        timestamp: datetime,
        price_action_result: Dict,
        ml_result: Dict,
        reason: str
    ) -> Dict:
        """Build rejected signal (ML filter rejected)"""
        signal = self._build_neutral_signal(symbol, current_price, timestamp, price_action_result)
        signal['rationale'] = [reason]
        signal['ml_ensemble'] = ml_result['ensemble_score']
        signal['ml_lstm'] = ml_result['lstm_score']
        signal['ml_rf'] = ml_result['rf_score']
        return signal

    def _build_error_signal(self, symbol: str, current_price: float, timestamp: datetime) -> Dict:
        """Build error signal"""
        return {
            'signal': 'NEUTRAL',
            'confidence': 0.5,
            'symbol': symbol,
            'price': current_price,
            'timestamp': timestamp,
            'confluence_count': 0,
            'confluence_score': 0,
            'rationale': ['Analysis error'],
            'confluence_details': [],
            'market_structure': 'UNKNOWN',
            'higher_tf_trend': 'NEUTRAL',
            'support_levels': [],
            'resistance_levels': [],
            'nearest_support': None,
            'nearest_resistance': None,
            'entry_price': 0,
            'entry_zone_low': 0,
            'entry_zone_high': 0,
            'stop_loss': 0,
            'take_profit': 0,
            'risk_reward': 0,
            'rsi_14': 0,
            'macd_hist': 0,
            'atr': 0,
            'ml_ensemble': 0.5,
            'ml_lstm': 0.5,
            'ml_rf': 0.5,
        }

    def get_signal_stats(self) -> Dict:
        """Get signal generation statistics"""
        total = self.buy_signal_count + self.sell_signal_count

        return {
            'total_signals': total,
            'buy_signals': self.buy_signal_count,
            'sell_signals': self.sell_signal_count,
            'buy_ratio': self.buy_signal_count / total if total > 0 else 0,
            'sell_ratio': self.sell_signal_count / total if total > 0 else 0,
            'last_buy_time': self.last_buy_signal_time,
            'last_sell_time': self.last_sell_signal_time,
        }
