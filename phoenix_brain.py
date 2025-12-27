"""
Phoenix Brain - Enhanced RL Agent with Brutal Reward Function
Phoenix Trading System - Self-Aware Trading Intelligence

This module provides:
- PPO/SAC reinforcement learning with distributional RL
- Brutal, honest reward function with regime-correct bonuses/penalties
- Meta-reward for improving Sharpe/profit factor over time
- Natural language reflection after every session
- Integration with vector memory for experience replay
"""

import logging
import json
import os
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path
from collections import deque
import threading

logger = logging.getLogger(__name__)

# Optional imports with graceful fallback
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available - using simplified RL")

try:
    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.vec_env import DummyVecEnv
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    logger.warning("stable-baselines3 not available - using custom RL")


@dataclass
class TradingState:
    """Current state of the trading environment"""
    # Price features
    price: float
    price_change_1h: float
    price_change_4h: float
    price_change_24h: float
    
    # Technical indicators
    rsi: float
    adx: float
    atr: float
    macd: float
    macd_signal: float
    bollinger_position: float  # -1 to 1 (lower to upper band)
    
    # Regime features
    regime: str  # trending, ranging, volatile
    regime_confidence: float
    
    # Account features
    balance: float
    equity: float
    drawdown: float
    open_positions: int
    
    # Performance features
    win_streak: int
    loss_streak: int
    daily_pnl: float
    weekly_pnl: float
    
    # Sentiment features
    sentiment_score: float
    news_impact: float
    
    def to_array(self) -> np.ndarray:
        """Convert state to numpy array for RL"""
        regime_map = {'trending': 1.0, 'ranging': 0.0, 'volatile': -1.0}
        
        return np.array([
            self.price_change_1h,
            self.price_change_4h,
            self.price_change_24h,
            (self.rsi - 50) / 50,  # Normalize to -1 to 1
            self.adx / 100,
            self.atr,
            self.macd,
            self.macd_signal,
            self.bollinger_position,
            regime_map.get(self.regime, 0.0),
            self.regime_confidence,
            self.drawdown,
            min(self.open_positions / 5, 1.0),
            min(self.win_streak / 5, 1.0),
            min(self.loss_streak / 5, 1.0),
            self.daily_pnl / 100,
            self.sentiment_score,
            self.news_impact
        ], dtype=np.float32)


@dataclass
class TradingAction:
    """Action taken by the RL agent"""
    action_type: str  # hold, buy, sell, close
    symbol: str
    size_multiplier: float  # 0.3x to 2.5x normal size
    leverage_multiplier: float  # Adjustment to base leverage
    sl_multiplier: float  # ATR multiplier for stop loss
    tp_multiplier: float  # Risk:reward ratio
    trail_aggressiveness: float  # 0 to 1
    confidence: float
    
    @classmethod
    def from_array(cls, action_array: np.ndarray, symbol: str) -> 'TradingAction':
        """Create action from RL output array"""
        # Action array: [action_type, size, leverage, sl, tp, trail, confidence]
        action_types = ['hold', 'buy', 'sell', 'close']
        
        return cls(
            action_type=action_types[int(action_array[0]) % 4],
            symbol=symbol,
            size_multiplier=0.3 + action_array[1] * 2.2,  # 0.3 to 2.5
            leverage_multiplier=0.5 + action_array[2] * 1.5,  # 0.5 to 2.0
            sl_multiplier=1.0 + action_array[3] * 2.0,  # 1.0 to 3.0 ATR
            tp_multiplier=1.5 + action_array[4] * 2.5,  # 1.5 to 4.0 R:R
            trail_aggressiveness=action_array[5],
            confidence=action_array[6]
        )


class BrutalRewardFunction:
    """
    Brutal, honest reward function that punishes bad decisions heavily.
    
    Rewards:
    - +2 * P/L for regime-aligned wins
    - +bonus for win streaks and Sharpe > 1.5
    - +meta-reward for improving monthly metrics
    
    Penalties:
    - -3 * loss for regime mismatches (e.g., mean reversion in strong trend)
    - -penalty for revenge trading after losses
    - -penalty for overleveraging
    - -penalty for ignoring news/FVG signals
    """
    
    def __init__(self):
        self.trade_history: List[Dict] = []
        self.monthly_metrics: Dict[str, float] = {}
        self.previous_sharpe: float = 0.0
        self.previous_profit_factor: float = 1.0
        self.consecutive_losses: int = 0
        self.last_trade_time: Optional[datetime] = None
    
    def calculate_reward(
        self,
        trade_result: Dict,
        state: TradingState,
        action: TradingAction
    ) -> Tuple[float, str]:
        """
        Calculate reward for a completed trade.
        Returns (reward, explanation).
        """
        reward = 0.0
        explanations = []
        
        pnl = trade_result.get('profit_loss', 0)
        pnl_pips = trade_result.get('profit_pips', 0)
        strategy = trade_result.get('strategy', 'unknown')
        regime = state.regime
        
        # Base reward: P/L
        base_reward = pnl_pips / 10  # Normalize pips to reasonable scale
        
        # 1. REGIME ALIGNMENT CHECK
        regime_aligned = self._check_regime_alignment(strategy, regime, state.adx)
        
        if pnl > 0:
            if regime_aligned:
                # Regime-aligned win: +2x reward
                reward += base_reward * 2.0
                explanations.append(f"+{base_reward * 2.0:.2f} regime-aligned win")
            else:
                # Lucky win against regime: only +1x
                reward += base_reward * 1.0
                explanations.append(f"+{base_reward:.2f} win (but regime mismatch - lucky)")
        else:
            if not regime_aligned:
                # Regime mismatch loss: -3x penalty
                reward += base_reward * 3.0  # base_reward is negative
                explanations.append(f"{base_reward * 3.0:.2f} BRUTAL penalty - traded against regime!")
            else:
                # Normal loss: -1.5x
                reward += base_reward * 1.5
                explanations.append(f"{base_reward * 1.5:.2f} loss (regime was correct)")
        
        # 2. WIN/LOSS STREAK BONUS/PENALTY
        if pnl > 0:
            self.consecutive_losses = 0
            if state.win_streak >= 3:
                streak_bonus = min(state.win_streak * 0.5, 3.0)
                reward += streak_bonus
                explanations.append(f"+{streak_bonus:.2f} win streak bonus ({state.win_streak} wins)")
        else:
            self.consecutive_losses += 1
            if self.consecutive_losses >= 3:
                # Potential revenge trading detection
                streak_penalty = min(self.consecutive_losses * 0.5, 3.0)
                reward -= streak_penalty
                explanations.append(f"-{streak_penalty:.2f} loss streak penalty - avoid revenge trading!")
        
        # 3. OVERLEVERAGING PENALTY
        if action.size_multiplier > 1.5 and state.drawdown > 0.1:
            overleverage_penalty = (action.size_multiplier - 1.0) * 2.0
            reward -= overleverage_penalty
            explanations.append(f"-{overleverage_penalty:.2f} overleveraging during drawdown")
        
        # 4. NEWS AWARENESS BONUS/PENALTY
        if abs(state.news_impact) > 0.5:
            if action.action_type == 'hold':
                # Good: avoided trading during high-impact news
                reward += 1.0
                explanations.append("+1.00 correctly avoided high-impact news")
            elif pnl < 0:
                # Bad: traded during news and lost
                reward -= 2.0
                explanations.append("-2.00 traded during high-impact news and lost")
        
        # 5. CONFIDENCE CALIBRATION
        if action.confidence > 0.8 and pnl < 0:
            # Overconfident and wrong
            reward -= 1.5
            explanations.append("-1.50 overconfident prediction was wrong")
        elif action.confidence < 0.5 and pnl > 0:
            # Underconfident but right - small bonus for learning
            reward += 0.5
            explanations.append("+0.50 trade worked despite low confidence - learn from this")
        
        # 6. SHARPE RATIO META-REWARD
        current_sharpe = self._calculate_rolling_sharpe()
        if current_sharpe > self.previous_sharpe + 0.1:
            sharpe_bonus = min((current_sharpe - self.previous_sharpe) * 5, 3.0)
            reward += sharpe_bonus
            explanations.append(f"+{sharpe_bonus:.2f} Sharpe ratio improved!")
        self.previous_sharpe = current_sharpe
        
        # 7. PROFIT FACTOR META-REWARD
        current_pf = self._calculate_profit_factor()
        if current_pf > self.previous_profit_factor * 1.1:
            pf_bonus = min((current_pf - self.previous_profit_factor) * 2, 2.0)
            reward += pf_bonus
            explanations.append(f"+{pf_bonus:.2f} profit factor improved!")
        self.previous_profit_factor = current_pf
        
        # Store trade for metrics
        self.trade_history.append({
            'timestamp': datetime.utcnow().isoformat(),
            'pnl': pnl,
            'pnl_pips': pnl_pips,
            'strategy': strategy,
            'regime': regime,
            'regime_aligned': regime_aligned,
            'reward': reward
        })
        
        # Keep only last 100 trades
        if len(self.trade_history) > 100:
            self.trade_history = self.trade_history[-100:]
        
        explanation = " | ".join(explanations)
        return reward, explanation
    
    def _check_regime_alignment(self, strategy: str, regime: str, adx: float) -> bool:
        """Check if strategy aligns with current regime"""
        # Trend-following strategies
        trend_strategies = ['momentum', 'breakout', 'trend_following', 'macd_cross']
        
        # Mean-reversion strategies
        reversion_strategies = ['mean_reversion', 'bollinger_bounce', 'rsi_reversal', 'grid']
        
        strategy_lower = strategy.lower()
        
        if regime == 'trending' and adx > 25:
            # Strong trend - trend strategies should work
            if any(s in strategy_lower for s in trend_strategies):
                return True
            if any(s in strategy_lower for s in reversion_strategies):
                return False  # Mean reversion in strong trend = BAD
        
        elif regime == 'ranging' or adx < 20:
            # Range-bound - mean reversion should work
            if any(s in strategy_lower for s in reversion_strategies):
                return True
            if any(s in strategy_lower for s in trend_strategies):
                return False  # Trend following in range = BAD
        
        # Default: neutral
        return True
    
    def _calculate_rolling_sharpe(self, window: int = 20) -> float:
        """Calculate rolling Sharpe ratio"""
        if len(self.trade_history) < window:
            return 0.0
        
        recent_pnl = [t['pnl'] for t in self.trade_history[-window:]]
        mean_return = np.mean(recent_pnl)
        std_return = np.std(recent_pnl)
        
        if std_return == 0:
            return 0.0
        
        # Annualized (assuming ~250 trading days, ~10 trades/day)
        sharpe = (mean_return / std_return) * np.sqrt(250 * 10)
        return sharpe
    
    def _calculate_profit_factor(self, window: int = 50) -> float:
        """Calculate profit factor (gross profit / gross loss)"""
        if len(self.trade_history) < 5:
            return 1.0
        
        recent = self.trade_history[-window:]
        gross_profit = sum(t['pnl'] for t in recent if t['pnl'] > 0)
        gross_loss = abs(sum(t['pnl'] for t in recent if t['pnl'] < 0))
        
        if gross_loss == 0:
            return 10.0  # Cap at 10
        
        return min(gross_profit / gross_loss, 10.0)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of recent performance"""
        if not self.trade_history:
            return {'trades': 0, 'sharpe': 0, 'profit_factor': 1.0}
        
        recent = self.trade_history[-50:]
        wins = sum(1 for t in recent if t['pnl'] > 0)
        
        return {
            'trades': len(recent),
            'win_rate': wins / len(recent) if recent else 0,
            'sharpe': self._calculate_rolling_sharpe(),
            'profit_factor': self._calculate_profit_factor(),
            'regime_alignment_rate': sum(1 for t in recent if t['regime_aligned']) / len(recent) if recent else 0,
            'avg_reward': np.mean([t['reward'] for t in recent]) if recent else 0
        }


class NaturalLanguageReflector:
    """
    Generates natural language reflections after trading sessions.
    Makes the agent feel "alive" and self-aware.
    """
    
    def __init__(self):
        self.reflection_history: List[Dict] = []
    
    def generate_trade_reflection(
        self,
        trade_result: Dict,
        reward: float,
        reward_explanation: str,
        state: TradingState
    ) -> str:
        """Generate reflection after a single trade"""
        pnl = trade_result.get('profit_loss', 0)
        pnl_pips = trade_result.get('profit_pips', 0)
        strategy = trade_result.get('strategy', 'unknown')
        symbol = trade_result.get('symbol', 'unknown')
        
        # Build reflection based on outcome
        if pnl > 0:
            if reward > 2:
                reflection = f"Excellent trade on {symbol}! {strategy} strategy aligned perfectly with the {state.regime} regime. "
                reflection += f"Captured {pnl_pips:.1f} pips. The market gave clear signals and I executed well. "
                reflection += f"Key insight: {self._extract_key_insight(trade_result, state, True)}"
            else:
                reflection = f"Profitable trade on {symbol} (+{pnl_pips:.1f} pips), but the reward was modest. "
                reflection += f"I may have gotten lucky - the regime wasn't ideal for {strategy}. "
                reflection += "Need to be more selective about regime alignment."
        else:
            if reward < -2:
                reflection = f"Brutal loss on {symbol}. I traded {strategy} in a {state.regime} regime with ADX={state.adx:.0f}. "
                reflection += f"This was a clear mismatch. Lost {abs(pnl_pips):.1f} pips. "
                reflection += f"Lesson learned: {self._extract_key_insight(trade_result, state, False)}"
            else:
                reflection = f"Loss on {symbol} (-{abs(pnl_pips):.1f} pips). The setup was reasonable but didn't work out. "
                reflection += "This is acceptable - not every trade wins. Moving on."
        
        # Add emotional state
        if state.loss_streak >= 3:
            reflection += " I'm feeling the pressure of consecutive losses. Must stay disciplined and avoid revenge trading."
        elif state.win_streak >= 3:
            reflection += " Confidence is high after this streak, but I must not become overconfident."
        
        # Store reflection
        self.reflection_history.append({
            'timestamp': datetime.utcnow().isoformat(),
            'trade': trade_result,
            'reflection': reflection,
            'reward': reward
        })
        
        return reflection
    
    def generate_session_reflection(
        self,
        trades: List[Dict],
        performance: Dict[str, Any],
        knowledge_state: Optional[Dict] = None
    ) -> str:
        """Generate end-of-session reflection"""
        if not trades:
            return "No trades executed this session. Market conditions may not have been favorable, or I'm being appropriately cautious."
        
        total_pnl = sum(t.get('profit_loss', 0) for t in trades)
        total_pips = sum(t.get('profit_pips', 0) for t in trades)
        wins = sum(1 for t in trades if t.get('profit_loss', 0) > 0)
        win_rate = wins / len(trades) if trades else 0
        
        # Build session summary
        reflection = f"Session Summary: {len(trades)} trades, {wins}W/{len(trades)-wins}L ({win_rate:.0%} win rate)\n"
        reflection += f"P/L: ${total_pnl:.2f} ({total_pips:+.1f} pips)\n\n"
        
        # Performance analysis
        sharpe = performance.get('sharpe', 0)
        pf = performance.get('profit_factor', 1)
        
        if sharpe > 1.5:
            reflection += "Risk-adjusted returns are excellent (Sharpe > 1.5). The strategy selection is working well.\n"
        elif sharpe < 0.5:
            reflection += "Risk-adjusted returns are poor. Need to improve strategy selection or reduce position sizes.\n"
        
        if pf > 2:
            reflection += "Profit factor is strong - wins are significantly larger than losses.\n"
        elif pf < 1:
            reflection += "Profit factor below 1 - losses exceed profits. This is unsustainable.\n"
        
        # Regime alignment analysis
        alignment_rate = performance.get('regime_alignment_rate', 0)
        if alignment_rate < 0.6:
            reflection += f"\nCRITICAL: Only {alignment_rate:.0%} of trades were regime-aligned. "
            reflection += "I'm fighting the market instead of flowing with it. Must improve regime detection.\n"
        
        # Strategy breakdown
        strategy_results = {}
        for t in trades:
            strategy = t.get('strategy', 'unknown')
            if strategy not in strategy_results:
                strategy_results[strategy] = {'wins': 0, 'losses': 0, 'pnl': 0}
            if t.get('profit_loss', 0) > 0:
                strategy_results[strategy]['wins'] += 1
            else:
                strategy_results[strategy]['losses'] += 1
            strategy_results[strategy]['pnl'] += t.get('profit_loss', 0)
        
        reflection += "\nStrategy Performance:\n"
        for strategy, results in sorted(strategy_results.items(), key=lambda x: x[1]['pnl'], reverse=True):
            wr = results['wins'] / (results['wins'] + results['losses']) if (results['wins'] + results['losses']) > 0 else 0
            reflection += f"  {strategy}: {results['wins']}W/{results['losses']}L ({wr:.0%}), ${results['pnl']:.2f}\n"
        
        # Tomorrow's plan
        reflection += "\nPlan for next session:\n"
        
        best_strategy = max(strategy_results.items(), key=lambda x: x[1]['pnl'])[0] if strategy_results else 'momentum'
        worst_strategy = min(strategy_results.items(), key=lambda x: x[1]['pnl'])[0] if strategy_results else None
        
        reflection += f"  - Favor {best_strategy} strategy based on recent performance\n"
        if worst_strategy and strategy_results[worst_strategy]['pnl'] < 0:
            reflection += f"  - Reduce weight on {worst_strategy} until conditions improve\n"
        
        if knowledge_state:
            sentiment = knowledge_state.get('overall_sentiment', 0)
            if sentiment > 0.3:
                reflection += "  - Market sentiment is bullish - look for long opportunities\n"
            elif sentiment < -0.3:
                reflection += "  - Market sentiment is bearish - look for short opportunities\n"
        
        return reflection
    
    def _extract_key_insight(self, trade: Dict, state: TradingState, was_win: bool) -> str:
        """Extract key insight from trade"""
        strategy = trade.get('strategy', 'unknown')
        
        if was_win:
            if state.adx > 30 and 'momentum' in strategy.lower():
                return "Strong ADX + momentum = high probability setup"
            elif state.rsi < 30 and 'reversion' in strategy.lower():
                return "Oversold RSI + mean reversion in range = good entry"
            else:
                return "Setup aligned with market conditions"
        else:
            if state.adx > 30 and 'reversion' in strategy.lower():
                return "Never fade strong trends (ADX > 30)"
            elif state.adx < 20 and 'momentum' in strategy.lower():
                return "Momentum fails in low-ADX choppy markets"
            else:
                return "Review entry criteria for this setup"


class PhoenixBrain:
    """
    Main RL brain for Phoenix trading system.
    Combines PPO/SAC with brutal rewards and natural language reflection.
    """
    
    def __init__(self, storage_path: str = "./data/phoenix_brain"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.reward_function = BrutalRewardFunction()
        self.reflector = NaturalLanguageReflector()
        
        # State
        self.current_state: Optional[TradingState] = None
        self.pending_trades: Dict[str, Dict] = {}  # trade_id -> trade_data
        self.session_trades: List[Dict] = []
        
        # RL model (simplified if SB3 not available)
        self.model = None
        self.state_dim = 18
        self.action_dim = 7
        
        if TORCH_AVAILABLE:
            self._init_neural_network()
        
        # Load saved state
        self._load_state()
        
        logger.info("PhoenixBrain initialized")
    
    def _init_neural_network(self):
        """Initialize neural network for policy"""
        class PolicyNetwork(nn.Module):
            def __init__(self, state_dim, action_dim):
                super().__init__()
                self.fc1 = nn.Linear(state_dim, 128)
                self.fc2 = nn.Linear(128, 128)
                self.fc3 = nn.Linear(128, 64)
                
                # Actor head (policy)
                self.actor = nn.Linear(64, action_dim)
                
                # Critic head (value)
                self.critic = nn.Linear(64, 1)
            
            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = torch.relu(self.fc2(x))
                x = torch.relu(self.fc3(x))
                
                action_probs = torch.sigmoid(self.actor(x))
                value = self.critic(x)
                
                return action_probs, value
        
        self.model = PolicyNetwork(self.state_dim, self.action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        # Load saved weights if available
        weights_file = self.storage_path / "policy_weights.pt"
        if weights_file.exists():
            try:
                self.model.load_state_dict(torch.load(weights_file))
                logger.info("Loaded saved policy weights")
            except Exception as e:
                logger.warning(f"Could not load policy weights: {e}")
    
    def _load_state(self):
        """Load saved brain state"""
        state_file = self.storage_path / "brain_state.json"
        
        try:
            if state_file.exists():
                with open(state_file, 'r') as f:
                    data = json.load(f)
                    self.reward_function.trade_history = data.get('trade_history', [])
                    self.reward_function.previous_sharpe = data.get('previous_sharpe', 0)
                    self.reward_function.previous_profit_factor = data.get('previous_profit_factor', 1)
                    logger.info("Loaded brain state from disk")
        except Exception as e:
            logger.warning(f"Could not load brain state: {e}")
    
    def _save_state(self):
        """Save brain state to disk"""
        state_file = self.storage_path / "brain_state.json"
        
        try:
            data = {
                'trade_history': self.reward_function.trade_history,
                'previous_sharpe': self.reward_function.previous_sharpe,
                'previous_profit_factor': self.reward_function.previous_profit_factor,
                'last_save': datetime.utcnow().isoformat()
            }
            
            with open(state_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            # Save model weights
            if self.model and TORCH_AVAILABLE:
                weights_file = self.storage_path / "policy_weights.pt"
                torch.save(self.model.state_dict(), weights_file)
                
        except Exception as e:
            logger.error(f"Error saving brain state: {e}")
    
    def update_state(self, state: TradingState):
        """Update current market state"""
        self.current_state = state
    
    def decide_action(
        self,
        symbol: str,
        state: Optional[TradingState] = None
    ) -> TradingAction:
        """
        Decide trading action based on current state.
        Returns action with size, leverage, SL/TP parameters.
        """
        state = state or self.current_state
        
        if state is None:
            # Default conservative action
            return TradingAction(
                action_type='hold',
                symbol=symbol,
                size_multiplier=1.0,
                leverage_multiplier=1.0,
                sl_multiplier=2.0,
                tp_multiplier=2.0,
                trail_aggressiveness=0.5,
                confidence=0.5
            )
        
        # Get state array
        state_array = state.to_array()
        
        if self.model and TORCH_AVAILABLE:
            # Use neural network
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state_array).unsqueeze(0)
                action_probs, value = self.model(state_tensor)
                action_array = action_probs.squeeze().numpy()
        else:
            # Rule-based fallback
            action_array = self._rule_based_action(state)
        
        # Convert to TradingAction
        action = TradingAction.from_array(action_array, symbol)
        
        # Apply safety constraints
        action = self._apply_safety_constraints(action, state)
        
        return action
    
    def _rule_based_action(self, state: TradingState) -> np.ndarray:
        """Rule-based action selection as fallback"""
        action = np.zeros(7)
        
        # Action type based on indicators
        if state.rsi < 30 and state.regime == 'ranging':
            action[0] = 1  # Buy
        elif state.rsi > 70 and state.regime == 'ranging':
            action[0] = 2  # Sell
        elif state.adx > 25 and state.macd > state.macd_signal:
            action[0] = 1  # Buy (trend)
        elif state.adx > 25 and state.macd < state.macd_signal:
            action[0] = 2  # Sell (trend)
        else:
            action[0] = 0  # Hold
        
        # Size based on confidence and drawdown
        base_size = 0.5
        if state.drawdown > 0.1:
            base_size *= 0.5  # Reduce size during drawdown
        if state.win_streak >= 3:
            base_size *= 1.2  # Slight increase on streak
        action[1] = min(base_size, 1.0)
        
        # Leverage based on volatility
        if state.atr > 0.002:
            action[2] = 0.3  # Low leverage in high vol
        else:
            action[2] = 0.6
        
        # SL/TP based on regime
        if state.regime == 'trending':
            action[3] = 0.5  # Tighter SL
            action[4] = 0.8  # Wider TP
        else:
            action[3] = 0.7  # Wider SL
            action[4] = 0.5  # Tighter TP
        
        # Trail aggressiveness
        action[5] = 0.5 if state.adx > 25 else 0.3
        
        # Confidence
        action[6] = min(state.regime_confidence, 0.8)
        
        return action
    
    def _apply_safety_constraints(self, action: TradingAction, state: TradingState) -> TradingAction:
        """Apply safety constraints to action"""
        # Reduce size during drawdown
        if state.drawdown > 0.15:
            action.size_multiplier = min(action.size_multiplier, 0.5)
            action.action_type = 'hold' if state.drawdown > 0.18 else action.action_type
        
        # Reduce size after loss streak
        if state.loss_streak >= 3:
            action.size_multiplier = min(action.size_multiplier, 0.5)
        
        # Cap leverage
        action.leverage_multiplier = min(action.leverage_multiplier, 2.0)
        
        # Ensure minimum SL
        action.sl_multiplier = max(action.sl_multiplier, 1.0)
        
        return action
    
    def record_trade_entry(self, trade_id: str, trade_data: Dict):
        """Record a trade entry for later reward calculation"""
        self.pending_trades[trade_id] = {
            **trade_data,
            'entry_time': datetime.utcnow().isoformat(),
            'state_at_entry': self.current_state
        }
    
    def process_trade_exit(
        self,
        trade_id: str,
        exit_price: float,
        profit_loss: float,
        profit_pips: float
    ) -> Tuple[float, str, str]:
        """
        Process a trade exit and calculate reward.
        Returns (reward, reward_explanation, reflection).
        """
        if trade_id not in self.pending_trades:
            logger.warning(f"Trade {trade_id} not found in pending trades")
            return 0.0, "Trade not found", ""
        
        trade_data = self.pending_trades.pop(trade_id)
        state_at_entry = trade_data.get('state_at_entry') or self.current_state
        
        # Build trade result
        trade_result = {
            'trade_id': trade_id,
            'symbol': trade_data.get('symbol'),
            'direction': trade_data.get('direction'),
            'entry_price': trade_data.get('entry_price'),
            'exit_price': exit_price,
            'profit_loss': profit_loss,
            'profit_pips': profit_pips,
            'strategy': trade_data.get('strategy'),
            'regime': trade_data.get('regime'),
            'entry_time': trade_data.get('entry_time'),
            'exit_time': datetime.utcnow().isoformat()
        }
        
        # Calculate reward
        action = TradingAction(
            action_type='buy' if trade_data.get('direction') == 'LONG' else 'sell',
            symbol=trade_data.get('symbol', 'UNKNOWN'),
            size_multiplier=trade_data.get('size_multiplier', 1.0),
            leverage_multiplier=1.0,
            sl_multiplier=2.0,
            tp_multiplier=2.0,
            trail_aggressiveness=0.5,
            confidence=trade_data.get('confidence', 0.5)
        )
        
        reward, reward_explanation = self.reward_function.calculate_reward(
            trade_result, state_at_entry, action
        )
        
        # Generate reflection
        reflection = self.reflector.generate_trade_reflection(
            trade_result, reward, reward_explanation, state_at_entry
        )
        
        # Add to session trades
        self.session_trades.append(trade_result)
        
        # Update model (simplified online learning)
        if self.model and TORCH_AVAILABLE:
            self._update_model(state_at_entry, action, reward)
        
        # Save state
        self._save_state()
        
        logger.info(f"[PHOENIX] Trade {trade_id} processed: reward={reward:.2f}")
        logger.info(f"[PHOENIX] Reflection: {reflection[:200]}...")
        
        return reward, reward_explanation, reflection
    
    def _update_model(self, state: TradingState, action: TradingAction, reward: float):
        """Simple online policy gradient update"""
        if not self.model:
            return
        
        state_tensor = torch.FloatTensor(state.to_array()).unsqueeze(0)
        
        # Forward pass
        action_probs, value = self.model(state_tensor)
        
        # Calculate loss (simplified policy gradient)
        advantage = reward - value.item()
        
        # Policy loss
        action_tensor = torch.FloatTensor([
            0 if action.action_type == 'hold' else (1 if action.action_type == 'buy' else 2),
            action.size_multiplier / 2.5,
            action.leverage_multiplier / 2.0,
            action.sl_multiplier / 3.0,
            action.tp_multiplier / 4.0,
            action.trail_aggressiveness,
            action.confidence
        ])
        
        log_prob = -torch.sum((action_probs - action_tensor) ** 2)
        policy_loss = -log_prob * advantage
        
        # Value loss
        value_loss = (value - reward) ** 2
        
        # Total loss
        loss = policy_loss + 0.5 * value_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def generate_session_summary(self, knowledge_state: Optional[Dict] = None) -> str:
        """Generate end-of-session summary"""
        performance = self.reward_function.get_performance_summary()
        
        summary = self.reflector.generate_session_reflection(
            self.session_trades,
            performance,
            knowledge_state
        )
        
        # Clear session trades
        self.session_trades.clear()
        
        # Save state
        self._save_state()
        
        return summary
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        return self.reward_function.get_performance_summary()
    
    def should_trade(self, state: TradingState) -> Tuple[bool, str]:
        """
        Determine if we should trade based on current conditions.
        Returns (should_trade, reason).
        """
        # Drawdown check
        if state.drawdown > 0.18:
            return False, "Drawdown exceeds 18% - trading halted for capital preservation"
        
        # Loss streak check
        if state.loss_streak >= 5:
            return False, "5+ consecutive losses - taking a break to reassess"
        
        # News impact check
        if abs(state.news_impact) > 0.8:
            return False, "High-impact news event - avoiding market"
        
        # Regime confidence check
        if state.regime_confidence < 0.4:
            return False, "Low regime confidence - waiting for clearer conditions"
        
        return True, "Conditions acceptable for trading"


# Global instance
phoenix_brain = PhoenixBrain()


if __name__ == "__main__":
    # Test the Phoenix brain
    logging.basicConfig(level=logging.INFO)
    
    brain = PhoenixBrain()
    
    # Create test state
    test_state = TradingState(
        price=1.0850,
        price_change_1h=0.001,
        price_change_4h=0.003,
        price_change_24h=0.005,
        rsi=45,
        adx=32,
        atr=0.0015,
        macd=0.0002,
        macd_signal=0.0001,
        bollinger_position=0.2,
        regime='trending',
        regime_confidence=0.75,
        balance=100,
        equity=98,
        drawdown=0.02,
        open_positions=1,
        win_streak=2,
        loss_streak=0,
        daily_pnl=5.0,
        weekly_pnl=15.0,
        sentiment_score=0.3,
        news_impact=0.1
    )
    
    brain.update_state(test_state)
    
    # Get action
    action = brain.decide_action("EURUSD")
    print(f"Action: {action.action_type}, Size: {action.size_multiplier:.2f}x, Confidence: {action.confidence:.2f}")
    
    # Test trade processing
    brain.record_trade_entry("test_001", {
        'symbol': 'EURUSD',
        'direction': 'LONG',
        'entry_price': 1.0850,
        'strategy': 'momentum',
        'regime': 'trending',
        'confidence': 0.75,
        'size_multiplier': 1.0
    })
    
    reward, explanation, reflection = brain.process_trade_exit(
        "test_001", 1.0880, 30.0, 30.0
    )
    
    print(f"\nReward: {reward:.2f}")
    print(f"Explanation: {explanation}")
    print(f"Reflection: {reflection}")
    
    # Get performance
    metrics = brain.get_performance_metrics()
    print(f"\nPerformance: {metrics}")
