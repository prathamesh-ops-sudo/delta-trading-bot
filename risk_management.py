"""
Institutional-Grade Risk Management Module
Includes VaR, CVaR, Monte Carlo simulations, portfolio optimization, and dynamic position sizing
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from scipy import stats
from scipy.optimize import minimize
import warnings

warnings.filterwarnings('ignore')

from config import config

logger = logging.getLogger(__name__)


@dataclass
class RiskMetrics:
    """Container for risk metrics"""
    var_95: float = 0.0
    var_99: float = 0.0
    cvar_95: float = 0.0
    cvar_99: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    volatility: float = 0.0
    beta: float = 0.0
    calmar_ratio: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    expectancy: float = 0.0
    kelly_fraction: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class Position:
    """Position data structure"""
    symbol: str
    direction: int  # 1 = long, -1 = short
    size: float
    entry_price: float
    current_price: float
    stop_loss: float
    take_profit: float
    unrealized_pnl: float = 0.0
    risk_amount: float = 0.0
    leverage: float = 1.0
    open_time: datetime = field(default_factory=datetime.now)


class VaRCalculator:
    """Value at Risk calculator with multiple methods"""
    
    def __init__(self, confidence_levels: List[float] = None):
        self.confidence_levels = confidence_levels or [0.95, 0.99]
    
    def historical_var(self, returns: np.ndarray, confidence: float = 0.95) -> float:
        """Calculate Historical VaR"""
        if len(returns) == 0:
            return 0.0
        return -np.percentile(returns, (1 - confidence) * 100)
    
    def parametric_var(self, returns: np.ndarray, confidence: float = 0.95,
                       distribution: str = 'normal') -> float:
        """Calculate Parametric VaR (Normal or Student-t)"""
        if len(returns) == 0:
            return 0.0
        
        mean = np.mean(returns)
        std = np.std(returns)
        
        if distribution == 'normal':
            z_score = stats.norm.ppf(1 - confidence)
        elif distribution == 't':
            # Fit Student-t distribution
            params = stats.t.fit(returns)
            z_score = stats.t.ppf(1 - confidence, *params[:-2], loc=params[-2], scale=params[-1])
            return -z_score
        else:
            z_score = stats.norm.ppf(1 - confidence)
        
        return -(mean + z_score * std)
    
    def monte_carlo_var(self, returns: np.ndarray, confidence: float = 0.95,
                        num_simulations: int = 10000, horizon: int = 1) -> float:
        """Calculate Monte Carlo VaR"""
        if len(returns) == 0:
            return 0.0
        
        mean = np.mean(returns)
        std = np.std(returns)
        
        # Generate simulated returns
        simulated_returns = np.random.normal(mean, std, (num_simulations, horizon))
        portfolio_returns = np.sum(simulated_returns, axis=1)
        
        return -np.percentile(portfolio_returns, (1 - confidence) * 100)
    
    def cvar(self, returns: np.ndarray, confidence: float = 0.95) -> float:
        """Calculate Conditional VaR (Expected Shortfall)"""
        if len(returns) == 0:
            return 0.0
        
        var = self.historical_var(returns, confidence)
        tail_losses = returns[returns <= -var]
        
        if len(tail_losses) == 0:
            return var
        
        return -np.mean(tail_losses)
    
    def calculate_all(self, returns: np.ndarray) -> Dict[str, float]:
        """Calculate all VaR metrics"""
        results = {}
        
        for conf in self.confidence_levels:
            conf_str = str(int(conf * 100))
            results[f'historical_var_{conf_str}'] = self.historical_var(returns, conf)
            results[f'parametric_var_{conf_str}'] = self.parametric_var(returns, conf)
            results[f'parametric_var_t_{conf_str}'] = self.parametric_var(returns, conf, 't')
            results[f'monte_carlo_var_{conf_str}'] = self.monte_carlo_var(returns, conf)
            results[f'cvar_{conf_str}'] = self.cvar(returns, conf)
        
        return results


class MonteCarloSimulator:
    """Monte Carlo simulation for scenario analysis"""
    
    def __init__(self, num_simulations: int = 10000):
        self.num_simulations = num_simulations
    
    def simulate_price_paths(self, current_price: float, returns: np.ndarray,
                             horizon: int = 252, method: str = 'gbm') -> np.ndarray:
        """Simulate future price paths"""
        mean = np.mean(returns)
        std = np.std(returns)
        
        if method == 'gbm':
            # Geometric Brownian Motion
            dt = 1 / 252
            drift = (mean - 0.5 * std ** 2) * dt
            diffusion = std * np.sqrt(dt)
            
            random_shocks = np.random.normal(0, 1, (self.num_simulations, horizon))
            price_paths = np.zeros((self.num_simulations, horizon + 1))
            price_paths[:, 0] = current_price
            
            for t in range(1, horizon + 1):
                price_paths[:, t] = price_paths[:, t-1] * np.exp(drift + diffusion * random_shocks[:, t-1])
        
        elif method == 'bootstrap':
            # Historical bootstrap
            price_paths = np.zeros((self.num_simulations, horizon + 1))
            price_paths[:, 0] = current_price
            
            for t in range(1, horizon + 1):
                sampled_returns = np.random.choice(returns, self.num_simulations)
                price_paths[:, t] = price_paths[:, t-1] * (1 + sampled_returns)
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return price_paths
    
    def simulate_portfolio_returns(self, weights: np.ndarray, 
                                   returns_matrix: np.ndarray,
                                   horizon: int = 21) -> np.ndarray:
        """Simulate portfolio returns"""
        mean_returns = np.mean(returns_matrix, axis=0)
        cov_matrix = np.cov(returns_matrix.T)
        
        # Generate correlated random returns
        simulated_returns = np.random.multivariate_normal(
            mean_returns, cov_matrix, (self.num_simulations, horizon)
        )
        
        # Calculate portfolio returns
        portfolio_returns = np.sum(simulated_returns * weights, axis=2)
        cumulative_returns = np.cumprod(1 + portfolio_returns, axis=1) - 1
        
        return cumulative_returns
    
    def stress_test(self, portfolio_value: float, positions: List[Position],
                    scenarios: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Run stress tests on portfolio"""
        results = {}
        
        for scenario_name, shocks in scenarios.items():
            scenario_pnl = 0.0
            
            for pos in positions:
                if pos.symbol in shocks:
                    shock = shocks[pos.symbol]
                    position_pnl = pos.size * pos.current_price * shock * pos.direction
                    scenario_pnl += position_pnl
            
            results[scenario_name] = {
                'pnl': scenario_pnl,
                'pnl_pct': scenario_pnl / portfolio_value if portfolio_value > 0 else 0,
                'new_value': portfolio_value + scenario_pnl
            }
        
        return results
    
    def calculate_scenario_statistics(self, simulated_returns: np.ndarray) -> Dict[str, float]:
        """Calculate statistics from simulated returns"""
        final_returns = simulated_returns[:, -1]
        
        return {
            'mean_return': np.mean(final_returns),
            'median_return': np.median(final_returns),
            'std_return': np.std(final_returns),
            'skewness': stats.skew(final_returns),
            'kurtosis': stats.kurtosis(final_returns),
            'percentile_5': np.percentile(final_returns, 5),
            'percentile_25': np.percentile(final_returns, 25),
            'percentile_75': np.percentile(final_returns, 75),
            'percentile_95': np.percentile(final_returns, 95),
            'prob_positive': np.mean(final_returns > 0),
            'prob_loss_10pct': np.mean(final_returns < -0.10),
            'prob_gain_20pct': np.mean(final_returns > 0.20),
            'max_simulated_loss': np.min(final_returns),
            'max_simulated_gain': np.max(final_returns)
        }


class PortfolioOptimizer:
    """Portfolio optimization with multiple methods"""
    
    def __init__(self):
        self.risk_free_rate = 0.02  # 2% annual
    
    def mean_variance_optimization(self, returns: np.ndarray, 
                                   target_return: float = None,
                                   constraints: Dict = None) -> np.ndarray:
        """Mean-Variance (Markowitz) optimization"""
        n_assets = returns.shape[1]
        mean_returns = np.mean(returns, axis=0)
        cov_matrix = np.cov(returns.T)
        
        def portfolio_volatility(weights):
            return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        
        def portfolio_return(weights):
            return np.dot(weights, mean_returns)
        
        # Constraints
        cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]  # Weights sum to 1
        
        if target_return is not None:
            cons.append({'type': 'eq', 'fun': lambda x: portfolio_return(x) - target_return})
        
        # Bounds (0 to 1 for each weight, no shorting)
        bounds = tuple((0, 1) for _ in range(n_assets))
        
        # Initial guess
        init_weights = np.array([1/n_assets] * n_assets)
        
        # Optimize
        result = minimize(portfolio_volatility, init_weights, method='SLSQP',
                         bounds=bounds, constraints=cons)
        
        return result.x if result.success else init_weights
    
    def max_sharpe_optimization(self, returns: np.ndarray) -> np.ndarray:
        """Maximize Sharpe Ratio"""
        n_assets = returns.shape[1]
        mean_returns = np.mean(returns, axis=0) * 252  # Annualized
        cov_matrix = np.cov(returns.T) * 252  # Annualized
        
        def neg_sharpe(weights):
            port_return = np.dot(weights, mean_returns)
            port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            return -(port_return - self.risk_free_rate) / port_vol
        
        cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = tuple((0, 1) for _ in range(n_assets))
        init_weights = np.array([1/n_assets] * n_assets)
        
        result = minimize(neg_sharpe, init_weights, method='SLSQP',
                         bounds=bounds, constraints=cons)
        
        return result.x if result.success else init_weights
    
    def risk_parity_optimization(self, returns: np.ndarray) -> np.ndarray:
        """Risk Parity - equal risk contribution from each asset"""
        n_assets = returns.shape[1]
        cov_matrix = np.cov(returns.T)
        
        def risk_contribution(weights):
            port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            marginal_contrib = np.dot(cov_matrix, weights)
            risk_contrib = weights * marginal_contrib / port_vol
            return risk_contrib
        
        def risk_parity_objective(weights):
            rc = risk_contribution(weights)
            target_rc = np.mean(rc)
            return np.sum((rc - target_rc) ** 2)
        
        cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = tuple((0.01, 1) for _ in range(n_assets))
        init_weights = np.array([1/n_assets] * n_assets)
        
        result = minimize(risk_parity_objective, init_weights, method='SLSQP',
                         bounds=bounds, constraints=cons)
        
        return result.x if result.success else init_weights
    
    def min_cvar_optimization(self, returns: np.ndarray, 
                              confidence: float = 0.95) -> np.ndarray:
        """Minimize CVaR (Conditional Value at Risk)"""
        n_assets = returns.shape[1]
        n_samples = returns.shape[0]
        
        def portfolio_cvar(weights):
            port_returns = np.dot(returns, weights)
            var = np.percentile(port_returns, (1 - confidence) * 100)
            cvar = -np.mean(port_returns[port_returns <= var])
            return cvar
        
        cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = tuple((0, 1) for _ in range(n_assets))
        init_weights = np.array([1/n_assets] * n_assets)
        
        result = minimize(portfolio_cvar, init_weights, method='SLSQP',
                         bounds=bounds, constraints=cons)
        
        return result.x if result.success else init_weights


class DynamicPositionSizer:
    """Dynamic position sizing based on risk metrics and market conditions"""
    
    def __init__(self, base_risk: float = 0.01, max_risk: float = 0.02,
                 min_risk: float = 0.005):
        self.base_risk = base_risk
        self.max_risk = max_risk
        self.min_risk = min_risk
    
    def calculate_position_size(self, account_balance: float, entry_price: float,
                                stop_loss: float, symbol_info: Dict,
                                risk_adjustment: float = 1.0,
                                confidence: float = 0.5) -> Dict:
        """Calculate optimal position size"""
        # Adjust risk based on confidence
        adjusted_risk = self.base_risk * risk_adjustment
        adjusted_risk = max(self.min_risk, min(self.max_risk, adjusted_risk))
        
        # Further adjust based on ML confidence
        if confidence < 0.4:
            adjusted_risk *= 0.5
        elif confidence > 0.7:
            adjusted_risk *= 1.2
        
        adjusted_risk = min(adjusted_risk, self.max_risk)
        
        # Calculate risk amount
        risk_amount = account_balance * adjusted_risk
        
        # Calculate pip value and position size
        pip_distance = abs(entry_price - stop_loss)
        if pip_distance == 0:
            pip_distance = entry_price * 0.001  # Default 0.1%
        
        # Get symbol specifications
        contract_size = symbol_info.get('trade_contract_size', 100000)
        point = symbol_info.get('point', 0.00001)
        volume_min = symbol_info.get('volume_min', 0.01)
        volume_max = symbol_info.get('volume_max', 100)
        volume_step = symbol_info.get('volume_step', 0.01)
        
        # Calculate lots
        pip_value = contract_size * point
        lots = risk_amount / (pip_distance / point * pip_value)
        
        # Round to volume step
        lots = max(volume_min, min(volume_max, lots))
        lots = round(lots / volume_step) * volume_step
        
        return {
            'lots': lots,
            'risk_amount': risk_amount,
            'risk_percent': adjusted_risk,
            'pip_distance': pip_distance,
            'pip_value': pip_value * lots
        }
    
    def kelly_criterion(self, win_rate: float, avg_win: float, 
                        avg_loss: float) -> float:
        """Calculate Kelly Criterion for optimal bet sizing"""
        if avg_loss == 0 or win_rate == 0:
            return 0.0
        
        win_loss_ratio = avg_win / abs(avg_loss)
        kelly = win_rate - (1 - win_rate) / win_loss_ratio
        
        # Use fractional Kelly (half Kelly is common)
        return max(0, kelly * 0.5)
    
    def volatility_adjusted_size(self, base_size: float, current_atr: float,
                                 avg_atr: float) -> float:
        """Adjust position size based on volatility"""
        if current_atr == 0 or avg_atr == 0:
            return base_size
        
        vol_ratio = avg_atr / current_atr
        adjusted_size = base_size * vol_ratio
        
        # Limit adjustment to 50-150% of base
        return max(base_size * 0.5, min(base_size * 1.5, adjusted_size))
    
    def equity_curve_adjustment(self, equity_curve: List[float]) -> float:
        """Adjust risk based on equity curve (anti-martingale)"""
        if len(equity_curve) < 10:
            return 1.0
        
        recent = equity_curve[-10:]
        trend = (recent[-1] - recent[0]) / recent[0] if recent[0] > 0 else 0
        
        # Increase risk on winning streak, decrease on losing
        if trend > 0.05:  # 5% gain
            return 1.2
        elif trend < -0.05:  # 5% loss
            return 0.7
        else:
            return 1.0


class RiskManager:
    """Main risk management class coordinating all risk functions"""
    
    def __init__(self):
        self.var_calculator = VaRCalculator()
        self.mc_simulator = MonteCarloSimulator()
        self.portfolio_optimizer = PortfolioOptimizer()
        self.position_sizer = DynamicPositionSizer()
        
        self.positions: List[Position] = []
        self.equity_curve: List[float] = []
        self.trade_history: List[Dict] = []
        self.risk_metrics = RiskMetrics()
        
        # Risk limits
        self.max_daily_drawdown = config.trading.max_daily_drawdown
        self.max_total_drawdown = config.trading.max_total_drawdown
        self.max_concurrent_trades = config.trading.max_concurrent_trades
        self.max_correlation_exposure = 0.7
        
        # State
        self.daily_pnl = 0.0
        self.peak_equity = config.trading.initial_balance
        self.is_halted = False
        self.halt_reason = ""
    
    def update_metrics(self, returns: np.ndarray, account_balance: float):
        """Update all risk metrics"""
        if len(returns) < 10:
            return
        
        # VaR metrics
        var_metrics = self.var_calculator.calculate_all(returns)
        self.risk_metrics.var_95 = var_metrics.get('historical_var_95', 0)
        self.risk_metrics.var_99 = var_metrics.get('historical_var_99', 0)
        self.risk_metrics.cvar_95 = var_metrics.get('cvar_95', 0)
        self.risk_metrics.cvar_99 = var_metrics.get('cvar_99', 0)
        
        # Performance metrics
        self.risk_metrics.volatility = np.std(returns) * np.sqrt(252)
        
        mean_return = np.mean(returns) * 252
        if self.risk_metrics.volatility > 0:
            self.risk_metrics.sharpe_ratio = (mean_return - 0.02) / self.risk_metrics.volatility
        
        # Sortino ratio
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            downside_std = np.std(downside_returns) * np.sqrt(252)
            if downside_std > 0:
                self.risk_metrics.sortino_ratio = (mean_return - 0.02) / downside_std
        
        # Drawdown
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        self.risk_metrics.max_drawdown = abs(np.min(drawdowns))
        self.risk_metrics.current_drawdown = abs(drawdowns[-1]) if len(drawdowns) > 0 else 0
        
        # Calmar ratio
        if self.risk_metrics.max_drawdown > 0:
            self.risk_metrics.calmar_ratio = mean_return / self.risk_metrics.max_drawdown
        
        # Trade statistics
        if self.trade_history:
            profits = [t['profit'] for t in self.trade_history]
            wins = [p for p in profits if p > 0]
            losses = [p for p in profits if p < 0]
            
            self.risk_metrics.win_rate = len(wins) / len(profits) if profits else 0
            self.risk_metrics.avg_win = np.mean(wins) if wins else 0
            self.risk_metrics.avg_loss = np.mean(losses) if losses else 0
            
            total_wins = sum(wins)
            total_losses = abs(sum(losses))
            self.risk_metrics.profit_factor = total_wins / total_losses if total_losses > 0 else 0
            
            self.risk_metrics.expectancy = (
                self.risk_metrics.win_rate * self.risk_metrics.avg_win +
                (1 - self.risk_metrics.win_rate) * self.risk_metrics.avg_loss
            )
            
            # Kelly fraction
            self.risk_metrics.kelly_fraction = self.position_sizer.kelly_criterion(
                self.risk_metrics.win_rate,
                self.risk_metrics.avg_win,
                self.risk_metrics.avg_loss
            )
        
        self.risk_metrics.timestamp = datetime.now()
    
    def check_risk_limits(self, account_balance: float) -> Tuple[bool, str]:
        """Check if any risk limits are breached"""
        # Check daily drawdown
        if self.daily_pnl < -account_balance * self.max_daily_drawdown:
            return False, f"Daily drawdown limit breached: {self.daily_pnl:.2f}"
        
        # Check total drawdown
        if self.peak_equity > 0:
            total_dd = (self.peak_equity - account_balance) / self.peak_equity
            if total_dd > self.max_total_drawdown:
                return False, f"Total drawdown limit breached: {total_dd:.2%}"
        
        # Check concurrent positions
        if len(self.positions) >= self.max_concurrent_trades:
            return False, f"Max concurrent trades reached: {len(self.positions)}"
        
        return True, "OK"
    
    def can_open_trade(self, symbol: str, direction: int, 
                       account_balance: float) -> Tuple[bool, str]:
        """Check if a new trade can be opened"""
        # Check if halted
        if self.is_halted:
            return False, f"Trading halted: {self.halt_reason}"
        
        # Check risk limits
        can_trade, reason = self.check_risk_limits(account_balance)
        if not can_trade:
            return False, reason
        
        # Check for existing position in same symbol
        existing = [p for p in self.positions if p.symbol == symbol]
        if existing:
            # Allow hedging but not doubling down
            if existing[0].direction == direction:
                return False, f"Already have {symbol} position in same direction"
        
        return True, "OK"
    
    def calculate_stop_loss(self, entry_price: float, direction: int,
                            atr: float, risk_multiple: float = 2.0) -> float:
        """Calculate dynamic stop loss based on ATR"""
        sl_distance = atr * risk_multiple
        
        if direction == 1:  # Long
            return entry_price - sl_distance
        else:  # Short
            return entry_price + sl_distance
    
    def calculate_take_profit(self, entry_price: float, stop_loss: float,
                              direction: int, rr_ratio: float = 2.0) -> float:
        """Calculate take profit based on risk-reward ratio"""
        risk = abs(entry_price - stop_loss)
        reward = risk * rr_ratio
        
        if direction == 1:  # Long
            return entry_price + reward
        else:  # Short
            return entry_price - reward
    
    def add_position(self, position: Position):
        """Add a new position"""
        self.positions.append(position)
        logger.info(f"Added position: {position.symbol} {position.direction} {position.size}")
    
    def close_position(self, symbol: str, close_price: float) -> Optional[Dict]:
        """Close a position and record the trade"""
        for i, pos in enumerate(self.positions):
            if pos.symbol == symbol:
                # Calculate PnL
                if pos.direction == 1:  # Long
                    pnl = (close_price - pos.entry_price) * pos.size
                else:  # Short
                    pnl = (pos.entry_price - close_price) * pos.size
                
                trade_record = {
                    'symbol': symbol,
                    'direction': pos.direction,
                    'size': pos.size,
                    'entry_price': pos.entry_price,
                    'exit_price': close_price,
                    'profit': pnl,
                    'open_time': pos.open_time,
                    'close_time': datetime.now()
                }
                
                self.trade_history.append(trade_record)
                self.daily_pnl += pnl
                self.positions.pop(i)
                
                logger.info(f"Closed position: {symbol} PnL: {pnl:.2f}")
                return trade_record
        
        return None
    
    def update_equity(self, account_balance: float):
        """Update equity curve and peak"""
        self.equity_curve.append(account_balance)
        if account_balance > self.peak_equity:
            self.peak_equity = account_balance
    
    def reset_daily(self):
        """Reset daily metrics"""
        self.daily_pnl = 0.0
        logger.info("Daily risk metrics reset")
    
    def halt_trading(self, reason: str):
        """Halt all trading"""
        self.is_halted = True
        self.halt_reason = reason
        logger.warning(f"Trading halted: {reason}")
    
    def resume_trading(self):
        """Resume trading"""
        self.is_halted = False
        self.halt_reason = ""
        logger.info("Trading resumed")
    
    def get_portfolio_exposure(self) -> Dict[str, float]:
        """Get current portfolio exposure by symbol"""
        exposure = {}
        for pos in self.positions:
            symbol = pos.symbol
            value = pos.size * pos.current_price * pos.direction
            exposure[symbol] = exposure.get(symbol, 0) + value
        return exposure
    
    def run_stress_tests(self, account_balance: float) -> Dict:
        """Run predefined stress test scenarios"""
        scenarios = {
            'flash_crash': {sym: -0.05 for sym in ['EURUSD', 'GBPUSD', 'AUDUSD']},
            'usd_strength': {'EURUSD': -0.03, 'GBPUSD': -0.03, 'USDJPY': 0.03},
            'risk_off': {'EURUSD': -0.02, 'GBPUSD': -0.04, 'AUDUSD': -0.05, 'USDJPY': -0.02},
            'volatility_spike': {sym: -0.02 for sym in ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD']},
        }
        
        return self.mc_simulator.stress_test(account_balance, self.positions, scenarios)
    
    def get_risk_report(self) -> Dict:
        """Generate comprehensive risk report"""
        return {
            'metrics': {
                'var_95': self.risk_metrics.var_95,
                'var_99': self.risk_metrics.var_99,
                'cvar_95': self.risk_metrics.cvar_95,
                'sharpe_ratio': self.risk_metrics.sharpe_ratio,
                'sortino_ratio': self.risk_metrics.sortino_ratio,
                'max_drawdown': self.risk_metrics.max_drawdown,
                'current_drawdown': self.risk_metrics.current_drawdown,
                'volatility': self.risk_metrics.volatility,
                'win_rate': self.risk_metrics.win_rate,
                'profit_factor': self.risk_metrics.profit_factor,
                'kelly_fraction': self.risk_metrics.kelly_fraction
            },
            'positions': len(self.positions),
            'daily_pnl': self.daily_pnl,
            'is_halted': self.is_halted,
            'halt_reason': self.halt_reason,
            'exposure': self.get_portfolio_exposure(),
            'timestamp': datetime.now().isoformat()
        }


# Singleton instance
risk_manager = RiskManager()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Risk Management Module...")
    
    # Generate sample returns
    np.random.seed(42)
    returns = np.random.normal(0.0001, 0.01, 1000)
    
    # Test VaR
    var_calc = VaRCalculator()
    var_metrics = var_calc.calculate_all(returns)
    print(f"\nVaR Metrics:")
    for k, v in var_metrics.items():
        print(f"  {k}: {v:.4f}")
    
    # Test Monte Carlo
    mc = MonteCarloSimulator(num_simulations=1000)
    paths = mc.simulate_price_paths(1.1000, returns, horizon=21)
    stats = mc.calculate_scenario_statistics(paths[:, 1:] / paths[:, :-1] - 1)
    print(f"\nMonte Carlo Statistics:")
    for k, v in stats.items():
        print(f"  {k}: {v:.4f}")
    
    # Test Portfolio Optimization
    multi_returns = np.random.normal(0.0001, 0.01, (1000, 4))
    optimizer = PortfolioOptimizer()
    
    mv_weights = optimizer.mean_variance_optimization(multi_returns)
    print(f"\nMean-Variance Weights: {mv_weights}")
    
    sharpe_weights = optimizer.max_sharpe_optimization(multi_returns)
    print(f"Max Sharpe Weights: {sharpe_weights}")
    
    rp_weights = optimizer.risk_parity_optimization(multi_returns)
    print(f"Risk Parity Weights: {rp_weights}")
    
    # Test Position Sizing
    sizer = DynamicPositionSizer()
    size_result = sizer.calculate_position_size(
        account_balance=100,
        entry_price=1.1000,
        stop_loss=1.0980,
        symbol_info={'trade_contract_size': 100000, 'point': 0.00001,
                    'volume_min': 0.01, 'volume_max': 100, 'volume_step': 0.01}
    )
    print(f"\nPosition Size: {size_result}")
    
    print("\nRisk Management Module test complete!")
