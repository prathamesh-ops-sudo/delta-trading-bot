"""
Backtesting Framework with Walk-Forward Optimization
Implements realistic backtesting with slippage, spread, and commission simulation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import json
import os
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')

from config import config
from indicators import TechnicalIndicators, FeatureEngineer
from decisions import decision_engine, TradingSignal, TradeDirection
from regime_detection import regime_manager
from risk_management import risk_manager

logger = logging.getLogger(__name__)


@dataclass
class BacktestTrade:
    """Trade record for backtesting"""
    entry_time: datetime
    exit_time: datetime
    symbol: str
    direction: int  # 1 = long, -1 = short
    entry_price: float
    exit_price: float
    volume: float
    gross_profit: float
    commission: float
    slippage: float
    spread_cost: float
    net_profit: float
    strategy: str
    confidence: float
    regime: str
    indicators: Dict = field(default_factory=dict)


@dataclass
class BacktestResult:
    """Results from a backtest run"""
    start_date: datetime
    end_date: datetime
    initial_balance: float
    final_balance: float
    total_return: float
    total_return_pct: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    max_drawdown: float
    max_drawdown_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    avg_trade_duration: float
    trades_per_day: float
    expectancy: float
    equity_curve: List[float] = field(default_factory=list)
    drawdown_curve: List[float] = field(default_factory=list)
    trades: List[BacktestTrade] = field(default_factory=list)
    monthly_returns: Dict[str, float] = field(default_factory=dict)
    strategy_performance: Dict[str, Dict] = field(default_factory=dict)
    regime_performance: Dict[str, Dict] = field(default_factory=dict)


class CostModel:
    """Model for trading costs"""
    
    def __init__(self, spread_pips: float = 1.0, commission_per_lot: float = 7.0,
                 slippage_pips: float = 0.5):
        self.spread_pips = spread_pips
        self.commission_per_lot = commission_per_lot
        self.slippage_pips = slippage_pips
    
    def calculate_spread_cost(self, volume: float, pip_value: float) -> float:
        """Calculate spread cost"""
        return self.spread_pips * pip_value * volume
    
    def calculate_commission(self, volume: float) -> float:
        """Calculate commission"""
        return self.commission_per_lot * volume
    
    def calculate_slippage(self, volume: float, pip_value: float, 
                           volatility: float = 1.0) -> float:
        """Calculate slippage (increases with volatility)"""
        adjusted_slippage = self.slippage_pips * volatility
        return adjusted_slippage * pip_value * volume
    
    def get_total_cost(self, volume: float, pip_value: float, 
                       volatility: float = 1.0) -> Dict[str, float]:
        """Get total trading costs"""
        spread = self.calculate_spread_cost(volume, pip_value)
        commission = self.calculate_commission(volume)
        slippage = self.calculate_slippage(volume, pip_value, volatility)
        
        return {
            'spread': spread,
            'commission': commission,
            'slippage': slippage,
            'total': spread + commission + slippage
        }


class BacktestEngine:
    """Main backtesting engine"""
    
    def __init__(self, initial_balance: float = 100.0, cost_model: CostModel = None):
        self.initial_balance = initial_balance
        self.cost_model = cost_model or CostModel()
        self.indicators = TechnicalIndicators()
        self.feature_engineer = FeatureEngineer()
        
        # State
        self.balance = initial_balance
        self.equity = initial_balance
        self.positions: Dict[str, Dict] = {}
        self.trades: List[BacktestTrade] = []
        self.equity_curve: List[float] = []
        self.peak_equity = initial_balance
        
        # Settings
        self.max_positions = config.trading.max_concurrent_trades
        self.risk_per_trade = config.trading.max_risk_per_trade
        self.max_daily_drawdown = config.trading.max_daily_drawdown
    
    def run(self, data: pd.DataFrame, strategy: Callable, 
            symbol: str = 'EURUSD') -> BacktestResult:
        """Run backtest on historical data"""
        logger.info(f"Starting backtest: {len(data)} bars, {symbol}")
        
        # Reset state
        self._reset()
        
        # Prepare data
        data = data.copy()
        if 'datetime' not in data.columns and data.index.name != 'datetime':
            data['datetime'] = data.index
        
        # Calculate indicators
        data = self._add_indicators(data)
        
        # Get pip value
        pip_value = 0.01 if 'JPY' in symbol else 0.0001
        contract_size = 100000
        
        # Main loop
        for i in range(100, len(data)):  # Start after warmup period
            current_bar = data.iloc[i]
            historical_data = data.iloc[:i+1]
            
            # Update positions
            self._update_positions(current_bar, pip_value, contract_size)
            
            # Check for exits
            self._check_exits(current_bar, pip_value, contract_size)
            
            # Generate signals
            if len(self.positions) < self.max_positions:
                signal = strategy(historical_data, current_bar)
                
                if signal is not None:
                    self._execute_signal(signal, current_bar, symbol, 
                                        pip_value, contract_size)
            
            # Record equity
            self.equity_curve.append(self.equity)
            
            # Check daily drawdown
            if self._check_daily_drawdown():
                logger.warning("Daily drawdown limit hit - pausing trading")
        
        # Close remaining positions
        if data.shape[0] > 0:
            self._close_all_positions(data.iloc[-1], pip_value, contract_size)
        
        # Generate results
        return self._generate_results(data)
    
    def _reset(self):
        """Reset backtest state"""
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.positions = {}
        self.trades = []
        self.equity_curve = [self.initial_balance]
        self.peak_equity = self.initial_balance
    
    def _add_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to data"""
        close = data['close']
        high = data['high']
        low = data['low']
        
        # Moving averages
        data['sma_20'] = self.indicators.sma(close, 20)
        data['sma_50'] = self.indicators.sma(close, 50)
        data['ema_12'] = self.indicators.ema(close, 12)
        data['ema_26'] = self.indicators.ema(close, 26)
        
        # Momentum
        data['rsi'] = self.indicators.rsi(close, 14)
        macd, signal, hist = self.indicators.macd(close)
        data['macd'] = macd
        data['macd_signal'] = signal
        data['macd_hist'] = hist
        
        # Volatility
        data['atr'] = self.indicators.atr(high, low, close, 14)
        bb_upper, bb_middle, bb_lower = self.indicators.bollinger_bands(close)
        data['bb_upper'] = bb_upper
        data['bb_middle'] = bb_middle
        data['bb_lower'] = bb_lower
        
        # Trend
        adx, plus_di, minus_di = self.indicators.adx(high, low, close)
        data['adx'] = adx
        data['plus_di'] = plus_di
        data['minus_di'] = minus_di
        
        return data
    
    def _update_positions(self, bar: pd.Series, pip_value: float, 
                          contract_size: float):
        """Update position values"""
        for ticket, pos in self.positions.items():
            current_price = bar['close']
            
            if pos['direction'] == 1:  # Long
                unrealized = (current_price - pos['entry_price']) * pos['volume'] * contract_size
            else:  # Short
                unrealized = (pos['entry_price'] - current_price) * pos['volume'] * contract_size
            
            pos['unrealized_pnl'] = unrealized
            pos['current_price'] = current_price
        
        # Update equity
        total_unrealized = sum(p['unrealized_pnl'] for p in self.positions.values())
        self.equity = self.balance + total_unrealized
        
        # Update peak
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity
    
    def _check_exits(self, bar: pd.Series, pip_value: float, contract_size: float):
        """Check for stop loss and take profit exits"""
        to_close = []
        
        for ticket, pos in self.positions.items():
            current_price = bar['close']
            high = bar['high']
            low = bar['low']
            
            exit_price = None
            exit_reason = None
            
            if pos['direction'] == 1:  # Long
                # Check stop loss
                if low <= pos['stop_loss']:
                    exit_price = pos['stop_loss']
                    exit_reason = 'stop_loss'
                # Check take profit
                elif high >= pos['take_profit']:
                    exit_price = pos['take_profit']
                    exit_reason = 'take_profit'
            else:  # Short
                # Check stop loss
                if high >= pos['stop_loss']:
                    exit_price = pos['stop_loss']
                    exit_reason = 'stop_loss'
                # Check take profit
                elif low <= pos['take_profit']:
                    exit_price = pos['take_profit']
                    exit_reason = 'take_profit'
            
            if exit_price is not None:
                to_close.append((ticket, exit_price, exit_reason, bar))
        
        # Close positions
        for ticket, exit_price, reason, bar in to_close:
            self._close_position(ticket, exit_price, bar, pip_value, 
                               contract_size, reason)
    
    def _execute_signal(self, signal: Dict, bar: pd.Series, symbol: str,
                        pip_value: float, contract_size: float):
        """Execute a trading signal"""
        direction = signal.get('direction', 0)
        if direction == 0:
            return
        
        entry_price = bar['close']
        atr = bar.get('atr', entry_price * 0.001)
        
        # Calculate stop loss and take profit
        sl_distance = atr * signal.get('sl_atr_mult', 2.0)
        tp_distance = sl_distance * signal.get('rr_ratio', 2.0)
        
        if direction == 1:  # Long
            stop_loss = entry_price - sl_distance
            take_profit = entry_price + tp_distance
        else:  # Short
            stop_loss = entry_price + sl_distance
            take_profit = entry_price - tp_distance
        
        # Calculate position size
        risk_amount = self.balance * self.risk_per_trade
        pip_distance = sl_distance / pip_value
        volume = risk_amount / (pip_distance * pip_value * contract_size)
        volume = max(0.01, min(10.0, round(volume, 2)))
        
        # Calculate costs
        volatility = atr / entry_price * 100  # Normalized volatility
        costs = self.cost_model.get_total_cost(volume, pip_value * contract_size, volatility)
        
        # Adjust entry price for slippage
        if direction == 1:
            entry_price += costs['slippage'] / (volume * contract_size)
        else:
            entry_price -= costs['slippage'] / (volume * contract_size)
        
        # Create position
        ticket = len(self.trades) + len(self.positions) + 1
        
        self.positions[ticket] = {
            'symbol': symbol,
            'direction': direction,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'volume': volume,
            'entry_time': bar.get('datetime', bar.name),
            'strategy': signal.get('strategy', 'unknown'),
            'confidence': signal.get('confidence', 0.5),
            'regime': signal.get('regime', 'unknown'),
            'indicators': {
                'rsi': bar.get('rsi', 50),
                'adx': bar.get('adx', 25),
                'atr': atr
            },
            'costs': costs,
            'unrealized_pnl': 0,
            'current_price': entry_price
        }
        
        logger.debug(f"Opened position: {symbol} {'LONG' if direction == 1 else 'SHORT'} "
                    f"@ {entry_price:.5f}, SL: {stop_loss:.5f}, TP: {take_profit:.5f}")
    
    def _close_position(self, ticket: int, exit_price: float, bar: pd.Series,
                        pip_value: float, contract_size: float, reason: str = ''):
        """Close a position"""
        if ticket not in self.positions:
            return
        
        pos = self.positions[ticket]
        
        # Calculate gross profit
        if pos['direction'] == 1:  # Long
            gross_profit = (exit_price - pos['entry_price']) * pos['volume'] * contract_size
        else:  # Short
            gross_profit = (pos['entry_price'] - exit_price) * pos['volume'] * contract_size
        
        # Get costs
        costs = pos['costs']
        net_profit = gross_profit - costs['total']
        
        # Update balance
        self.balance += net_profit
        
        # Create trade record
        trade = BacktestTrade(
            entry_time=pos['entry_time'],
            exit_time=bar.get('datetime', bar.name),
            symbol=pos['symbol'],
            direction=pos['direction'],
            entry_price=pos['entry_price'],
            exit_price=exit_price,
            volume=pos['volume'],
            gross_profit=gross_profit,
            commission=costs['commission'],
            slippage=costs['slippage'],
            spread_cost=costs['spread'],
            net_profit=net_profit,
            strategy=pos['strategy'],
            confidence=pos['confidence'],
            regime=pos['regime'],
            indicators=pos['indicators']
        )
        
        self.trades.append(trade)
        del self.positions[ticket]
        
        logger.debug(f"Closed position: {pos['symbol']} P/L: ${net_profit:.2f} ({reason})")
    
    def _close_all_positions(self, bar: pd.Series, pip_value: float, 
                             contract_size: float):
        """Close all open positions"""
        for ticket in list(self.positions.keys()):
            self._close_position(ticket, bar['close'], bar, pip_value, 
                               contract_size, 'end_of_backtest')
    
    def _check_daily_drawdown(self) -> bool:
        """Check if daily drawdown limit is hit"""
        if len(self.equity_curve) < 2:
            return False
        
        # Simple check - compare to peak
        drawdown = (self.peak_equity - self.equity) / self.peak_equity
        return drawdown > self.max_daily_drawdown
    
    def _generate_results(self, data: pd.DataFrame) -> BacktestResult:
        """Generate backtest results"""
        if not self.trades:
            return self._empty_results(data)
        
        # Basic metrics
        profits = [t.net_profit for t in self.trades]
        wins = [t for t in self.trades if t.net_profit > 0]
        losses = [t for t in self.trades if t.net_profit <= 0]
        
        total_return = self.balance - self.initial_balance
        total_return_pct = total_return / self.initial_balance
        
        win_rate = len(wins) / len(self.trades) if self.trades else 0
        avg_win = np.mean([t.net_profit for t in wins]) if wins else 0
        avg_loss = np.mean([t.net_profit for t in losses]) if losses else 0
        
        # Profit factor
        gross_profit = sum(t.net_profit for t in wins)
        gross_loss = abs(sum(t.net_profit for t in losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Drawdown
        equity_array = np.array(self.equity_curve)
        running_max = np.maximum.accumulate(equity_array)
        drawdowns = (running_max - equity_array)
        drawdown_pct = drawdowns / running_max
        max_drawdown = np.max(drawdowns)
        max_drawdown_pct = np.max(drawdown_pct)
        
        # Risk-adjusted returns
        returns = np.diff(equity_array) / equity_array[:-1]
        
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0 and np.std(downside_returns) > 0:
            sortino_ratio = np.mean(returns) / np.std(downside_returns) * np.sqrt(252)
        else:
            sortino_ratio = 0
        
        annual_return = total_return_pct * 252 / len(data) if len(data) > 0 else 0
        calmar_ratio = annual_return / max_drawdown_pct if max_drawdown_pct > 0 else 0
        
        # Trade duration
        durations = []
        for t in self.trades:
            if hasattr(t.entry_time, 'timestamp') and hasattr(t.exit_time, 'timestamp'):
                duration = (t.exit_time - t.entry_time).total_seconds() / 60
                durations.append(duration)
        avg_duration = np.mean(durations) if durations else 0
        
        # Trades per day
        if len(data) > 0:
            days = (data.index[-1] - data.index[0]).days if hasattr(data.index[0], 'days') else len(data) / 24
            trades_per_day = len(self.trades) / max(1, days)
        else:
            trades_per_day = 0
        
        # Expectancy
        expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss
        
        # Monthly returns
        monthly_returns = self._calculate_monthly_returns(data)
        
        # Strategy performance
        strategy_perf = self._calculate_strategy_performance()
        
        # Regime performance
        regime_perf = self._calculate_regime_performance()
        
        return BacktestResult(
            start_date=data.index[0] if len(data) > 0 else datetime.now(),
            end_date=data.index[-1] if len(data) > 0 else datetime.now(),
            initial_balance=self.initial_balance,
            final_balance=self.balance,
            total_return=total_return,
            total_return_pct=total_return_pct,
            total_trades=len(self.trades),
            winning_trades=len(wins),
            losing_trades=len(losses),
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            max_drawdown=max_drawdown,
            max_drawdown_pct=max_drawdown_pct,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            avg_trade_duration=avg_duration,
            trades_per_day=trades_per_day,
            expectancy=expectancy,
            equity_curve=self.equity_curve,
            drawdown_curve=drawdown_pct.tolist(),
            trades=self.trades,
            monthly_returns=monthly_returns,
            strategy_performance=strategy_perf,
            regime_performance=regime_perf
        )
    
    def _empty_results(self, data: pd.DataFrame) -> BacktestResult:
        """Return empty results"""
        return BacktestResult(
            start_date=data.index[0] if len(data) > 0 else datetime.now(),
            end_date=data.index[-1] if len(data) > 0 else datetime.now(),
            initial_balance=self.initial_balance,
            final_balance=self.initial_balance,
            total_return=0,
            total_return_pct=0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0,
            profit_factor=0,
            avg_win=0,
            avg_loss=0,
            max_drawdown=0,
            max_drawdown_pct=0,
            sharpe_ratio=0,
            sortino_ratio=0,
            calmar_ratio=0,
            avg_trade_duration=0,
            trades_per_day=0,
            expectancy=0,
            equity_curve=[self.initial_balance],
            drawdown_curve=[0],
            trades=[],
            monthly_returns={},
            strategy_performance={},
            regime_performance={}
        )
    
    def _calculate_monthly_returns(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate monthly returns"""
        monthly = {}
        
        for trade in self.trades:
            if hasattr(trade.exit_time, 'strftime'):
                month_key = trade.exit_time.strftime('%Y-%m')
            else:
                month_key = str(trade.exit_time)[:7]
            
            if month_key not in monthly:
                monthly[month_key] = 0
            monthly[month_key] += trade.net_profit
        
        return monthly
    
    def _calculate_strategy_performance(self) -> Dict[str, Dict]:
        """Calculate performance by strategy"""
        perf = defaultdict(lambda: {'trades': 0, 'wins': 0, 'profit': 0})
        
        for trade in self.trades:
            strategy = trade.strategy
            perf[strategy]['trades'] += 1
            if trade.net_profit > 0:
                perf[strategy]['wins'] += 1
            perf[strategy]['profit'] += trade.net_profit
        
        # Calculate win rates
        for strategy in perf:
            if perf[strategy]['trades'] > 0:
                perf[strategy]['win_rate'] = perf[strategy]['wins'] / perf[strategy]['trades']
            else:
                perf[strategy]['win_rate'] = 0
        
        return dict(perf)
    
    def _calculate_regime_performance(self) -> Dict[str, Dict]:
        """Calculate performance by regime"""
        perf = defaultdict(lambda: {'trades': 0, 'wins': 0, 'profit': 0})
        
        for trade in self.trades:
            regime = trade.regime
            perf[regime]['trades'] += 1
            if trade.net_profit > 0:
                perf[regime]['wins'] += 1
            perf[regime]['profit'] += trade.net_profit
        
        # Calculate win rates
        for regime in perf:
            if perf[regime]['trades'] > 0:
                perf[regime]['win_rate'] = perf[regime]['wins'] / perf[regime]['trades']
            else:
                perf[regime]['win_rate'] = 0
        
        return dict(perf)


class WalkForwardOptimizer:
    """Walk-forward optimization for strategy parameters"""
    
    def __init__(self, train_ratio: float = 0.7, num_folds: int = 5):
        self.train_ratio = train_ratio
        self.num_folds = num_folds
        self.results: List[BacktestResult] = []
    
    def optimize(self, data: pd.DataFrame, strategy_factory: Callable,
                 param_grid: Dict[str, List], symbol: str = 'EURUSD',
                 initial_balance: float = 100.0) -> Dict:
        """Run walk-forward optimization"""
        logger.info(f"Starting walk-forward optimization with {self.num_folds} folds")
        
        # Split data into folds
        fold_size = len(data) // self.num_folds
        
        all_results = []
        best_params_per_fold = []
        
        for fold in range(self.num_folds - 1):
            logger.info(f"Processing fold {fold + 1}/{self.num_folds - 1}")
            
            # Define train and test periods
            train_start = fold * fold_size
            train_end = train_start + int(fold_size * self.train_ratio)
            test_start = train_end
            test_end = (fold + 2) * fold_size
            
            train_data = data.iloc[train_start:train_end]
            test_data = data.iloc[test_start:test_end]
            
            # Optimize on training data
            best_params, best_score = self._optimize_fold(
                train_data, strategy_factory, param_grid, symbol, initial_balance
            )
            
            best_params_per_fold.append(best_params)
            
            # Test on out-of-sample data
            strategy = strategy_factory(**best_params)
            engine = BacktestEngine(initial_balance)
            result = engine.run(test_data, strategy, symbol)
            
            all_results.append({
                'fold': fold,
                'train_period': (train_start, train_end),
                'test_period': (test_start, test_end),
                'best_params': best_params,
                'train_score': best_score,
                'test_result': result
            })
            
            logger.info(f"Fold {fold + 1}: Train score={best_score:.4f}, "
                       f"Test return={result.total_return_pct:.2%}")
        
        # Aggregate results
        return self._aggregate_results(all_results, best_params_per_fold)
    
    def _optimize_fold(self, train_data: pd.DataFrame, strategy_factory: Callable,
                       param_grid: Dict[str, List], symbol: str,
                       initial_balance: float) -> Tuple[Dict, float]:
        """Optimize parameters on a single fold"""
        best_params = None
        best_score = float('-inf')
        
        # Generate all parameter combinations
        param_combinations = self._generate_param_combinations(param_grid)
        
        for params in param_combinations:
            try:
                strategy = strategy_factory(**params)
                engine = BacktestEngine(initial_balance)
                result = engine.run(train_data, strategy, symbol)
                
                # Score based on Sharpe ratio and profit factor
                score = result.sharpe_ratio * 0.5 + min(result.profit_factor, 3) * 0.3 + result.win_rate * 0.2
                
                if score > best_score:
                    best_score = score
                    best_params = params
                    
            except Exception as e:
                logger.warning(f"Error with params {params}: {e}")
        
        return best_params or {}, best_score
    
    def _generate_param_combinations(self, param_grid: Dict[str, List]) -> List[Dict]:
        """Generate all parameter combinations"""
        if not param_grid:
            return [{}]
        
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        
        combinations = []
        
        def generate(index, current):
            if index == len(keys):
                combinations.append(current.copy())
                return
            
            for value in values[index]:
                current[keys[index]] = value
                generate(index + 1, current)
        
        generate(0, {})
        return combinations
    
    def _aggregate_results(self, all_results: List[Dict], 
                           best_params: List[Dict]) -> Dict:
        """Aggregate walk-forward results"""
        test_returns = [r['test_result'].total_return_pct for r in all_results]
        test_sharpes = [r['test_result'].sharpe_ratio for r in all_results]
        test_win_rates = [r['test_result'].win_rate for r in all_results]
        
        return {
            'num_folds': len(all_results),
            'avg_test_return': np.mean(test_returns),
            'std_test_return': np.std(test_returns),
            'avg_sharpe': np.mean(test_sharpes),
            'avg_win_rate': np.mean(test_win_rates),
            'best_params_per_fold': best_params,
            'fold_results': all_results,
            'robustness_score': np.mean(test_returns) / (np.std(test_returns) + 0.01)
        }


class MonteCarloBacktest:
    """Monte Carlo simulation for backtest robustness"""
    
    def __init__(self, num_simulations: int = 1000):
        self.num_simulations = num_simulations
    
    def simulate(self, trades: List[BacktestTrade], 
                 initial_balance: float = 100.0) -> Dict:
        """Run Monte Carlo simulation on trade sequence"""
        if not trades:
            return {'error': 'No trades to simulate'}
        
        profits = [t.net_profit for t in trades]
        
        final_balances = []
        max_drawdowns = []
        
        for _ in range(self.num_simulations):
            # Shuffle trade order
            shuffled = np.random.permutation(profits)
            
            # Calculate equity curve
            equity = [initial_balance]
            for profit in shuffled:
                equity.append(equity[-1] + profit)
            
            equity = np.array(equity)
            final_balances.append(equity[-1])
            
            # Calculate max drawdown
            running_max = np.maximum.accumulate(equity)
            drawdown = (running_max - equity) / running_max
            max_drawdowns.append(np.max(drawdown))
        
        return {
            'mean_final_balance': np.mean(final_balances),
            'std_final_balance': np.std(final_balances),
            'percentile_5': np.percentile(final_balances, 5),
            'percentile_25': np.percentile(final_balances, 25),
            'percentile_50': np.percentile(final_balances, 50),
            'percentile_75': np.percentile(final_balances, 75),
            'percentile_95': np.percentile(final_balances, 95),
            'mean_max_drawdown': np.mean(max_drawdowns),
            'worst_max_drawdown': np.max(max_drawdowns),
            'prob_profit': np.mean(np.array(final_balances) > initial_balance),
            'prob_double': np.mean(np.array(final_balances) > initial_balance * 2),
            'prob_ruin': np.mean(np.array(final_balances) < initial_balance * 0.5)
        }


def create_sample_strategy(rsi_oversold: float = 30, rsi_overbought: float = 70,
                           adx_threshold: float = 25, sl_atr_mult: float = 2.0,
                           rr_ratio: float = 2.0):
    """Create a sample strategy function"""
    
    def strategy(historical_data: pd.DataFrame, current_bar: pd.Series) -> Optional[Dict]:
        rsi = current_bar.get('rsi', 50)
        adx = current_bar.get('adx', 25)
        macd_hist = current_bar.get('macd_hist', 0)
        
        signal = None
        
        # Trend following with RSI filter
        if adx > adx_threshold:
            if rsi < rsi_oversold and macd_hist > 0:
                signal = {
                    'direction': 1,
                    'strategy': 'trend_following',
                    'confidence': min(0.9, 0.5 + (adx - adx_threshold) / 50),
                    'regime': 'trending',
                    'sl_atr_mult': sl_atr_mult,
                    'rr_ratio': rr_ratio
                }
            elif rsi > rsi_overbought and macd_hist < 0:
                signal = {
                    'direction': -1,
                    'strategy': 'trend_following',
                    'confidence': min(0.9, 0.5 + (adx - adx_threshold) / 50),
                    'regime': 'trending',
                    'sl_atr_mult': sl_atr_mult,
                    'rr_ratio': rr_ratio
                }
        
        # Mean reversion in ranging markets
        elif adx < 20:
            if rsi < rsi_oversold:
                signal = {
                    'direction': 1,
                    'strategy': 'mean_reversion',
                    'confidence': 0.6,
                    'regime': 'ranging',
                    'sl_atr_mult': sl_atr_mult * 0.8,
                    'rr_ratio': rr_ratio * 0.8
                }
            elif rsi > rsi_overbought:
                signal = {
                    'direction': -1,
                    'strategy': 'mean_reversion',
                    'confidence': 0.6,
                    'regime': 'ranging',
                    'sl_atr_mult': sl_atr_mult * 0.8,
                    'rr_ratio': rr_ratio * 0.8
                }
        
        return signal
    
    return strategy


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Backtesting Framework...")
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 5000
    
    prices = [1.1000]
    for i in range(n_samples - 1):
        change = np.random.normal(0.0001, 0.001)
        prices.append(prices[-1] * (1 + change))
    
    data = pd.DataFrame({
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.002))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.002))) for p in prices],
        'close': prices,
        'volume': np.random.randint(1000, 10000, n_samples)
    })
    data.index = pd.date_range(start='2020-01-01', periods=n_samples, freq='H')
    
    # Test basic backtest
    print("\nRunning basic backtest...")
    engine = BacktestEngine(initial_balance=100)
    strategy = create_sample_strategy()
    result = engine.run(data, strategy, 'EURUSD')
    
    print(f"\nBacktest Results:")
    print(f"  Initial Balance: ${result.initial_balance:.2f}")
    print(f"  Final Balance: ${result.final_balance:.2f}")
    print(f"  Total Return: {result.total_return_pct:.2%}")
    print(f"  Total Trades: {result.total_trades}")
    print(f"  Win Rate: {result.win_rate:.2%}")
    print(f"  Profit Factor: {result.profit_factor:.2f}")
    print(f"  Max Drawdown: {result.max_drawdown_pct:.2%}")
    print(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")
    print(f"  Trades/Day: {result.trades_per_day:.2f}")
    
    # Test Monte Carlo
    print("\nRunning Monte Carlo simulation...")
    mc = MonteCarloBacktest(num_simulations=1000)
    mc_results = mc.simulate(result.trades, 100)
    
    print(f"\nMonte Carlo Results:")
    print(f"  Mean Final Balance: ${mc_results['mean_final_balance']:.2f}")
    print(f"  5th Percentile: ${mc_results['percentile_5']:.2f}")
    print(f"  95th Percentile: ${mc_results['percentile_95']:.2f}")
    print(f"  Probability of Profit: {mc_results['prob_profit']:.2%}")
    print(f"  Probability of Ruin: {mc_results['prob_ruin']:.2%}")
    
    # Test Walk-Forward (simplified)
    print("\nRunning Walk-Forward Optimization (simplified)...")
    wfo = WalkForwardOptimizer(train_ratio=0.7, num_folds=3)
    
    param_grid = {
        'rsi_oversold': [25, 30],
        'rsi_overbought': [70, 75],
        'adx_threshold': [20, 25]
    }
    
    wfo_results = wfo.optimize(data, create_sample_strategy, param_grid, 'EURUSD', 100)
    
    print(f"\nWalk-Forward Results:")
    print(f"  Avg Test Return: {wfo_results['avg_test_return']:.2%}")
    print(f"  Avg Sharpe: {wfo_results['avg_sharpe']:.2f}")
    print(f"  Robustness Score: {wfo_results['robustness_score']:.2f}")
    
    print("\nBacktesting Framework test complete!")
