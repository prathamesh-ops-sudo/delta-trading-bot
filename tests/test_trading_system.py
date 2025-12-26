"""
Unit Tests for Aladdin Forex Trading System
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from indicators import TechnicalIndicators, FeatureEngineer
from risk_management import VaRCalculator, MonteCarloSimulator, DynamicPositionSizer, RiskManager
from regime_detection import RegimeFeatureExtractor, HMMRegimeDetector, RegimeManager
from decisions import FVGDetector, LiquiditySweepDetector, MultiTimeframeAnalyzer, TradeDirection
from execution import SlippageModel, TWAPExecutor, VWAPExecutor, Order, OrderSide, OrderType
from agentic import TradeJournal, InsightEngine, PerformanceAnalyzer, AgenticLearningSystem
from backtesting import BacktestEngine, CostModel, create_sample_strategy


@pytest.fixture
def sample_ohlcv_data():
    """Generate sample OHLCV data for testing"""
    np.random.seed(42)
    n_samples = 500
    
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
    
    return df


@pytest.fixture
def sample_returns():
    """Generate sample returns for testing"""
    np.random.seed(42)
    return np.random.normal(0.0001, 0.01, 1000)


class TestTechnicalIndicators:
    """Tests for technical indicators"""
    
    def test_sma(self, sample_ohlcv_data):
        indicators = TechnicalIndicators()
        sma = indicators.sma(sample_ohlcv_data['close'], 20)
        
        assert len(sma) == len(sample_ohlcv_data)
        assert not sma.iloc[20:].isna().any()
        assert sma.iloc[19:].mean() > 0
    
    def test_ema(self, sample_ohlcv_data):
        indicators = TechnicalIndicators()
        ema = indicators.ema(sample_ohlcv_data['close'], 12)
        
        assert len(ema) == len(sample_ohlcv_data)
        assert not ema.iloc[12:].isna().any()
    
    def test_rsi(self, sample_ohlcv_data):
        indicators = TechnicalIndicators()
        rsi = indicators.rsi(sample_ohlcv_data['close'], 14)
        
        assert len(rsi) == len(sample_ohlcv_data)
        # RSI should be between 0 and 100
        valid_rsi = rsi.dropna()
        assert (valid_rsi >= 0).all() and (valid_rsi <= 100).all()
    
    def test_macd(self, sample_ohlcv_data):
        indicators = TechnicalIndicators()
        macd, signal, hist = indicators.macd(sample_ohlcv_data['close'])
        
        assert len(macd) == len(sample_ohlcv_data)
        assert len(signal) == len(sample_ohlcv_data)
        assert len(hist) == len(sample_ohlcv_data)
    
    def test_bollinger_bands(self, sample_ohlcv_data):
        indicators = TechnicalIndicators()
        upper, middle, lower = indicators.bollinger_bands(sample_ohlcv_data['close'])
        
        assert len(upper) == len(sample_ohlcv_data)
        # Upper should be above middle, middle above lower
        valid_idx = ~(upper.isna() | middle.isna() | lower.isna())
        assert (upper[valid_idx] >= middle[valid_idx]).all()
        assert (middle[valid_idx] >= lower[valid_idx]).all()
    
    def test_atr(self, sample_ohlcv_data):
        indicators = TechnicalIndicators()
        atr = indicators.atr(
            sample_ohlcv_data['high'],
            sample_ohlcv_data['low'],
            sample_ohlcv_data['close'],
            14
        )
        
        assert len(atr) == len(sample_ohlcv_data)
        # ATR should be positive
        valid_atr = atr.dropna()
        assert (valid_atr >= 0).all()
    
    def test_adx(self, sample_ohlcv_data):
        indicators = TechnicalIndicators()
        adx, plus_di, minus_di = indicators.adx(
            sample_ohlcv_data['high'],
            sample_ohlcv_data['low'],
            sample_ohlcv_data['close']
        )
        
        assert len(adx) == len(sample_ohlcv_data)
        # ADX should be between 0 and 100
        valid_adx = adx.dropna()
        assert (valid_adx >= 0).all() and (valid_adx <= 100).all()


class TestFeatureEngineer:
    """Tests for feature engineering"""
    
    def test_create_features(self, sample_ohlcv_data):
        engineer = FeatureEngineer()
        features = engineer.create_features(sample_ohlcv_data)
        
        assert isinstance(features, pd.DataFrame)
        assert len(features) == len(sample_ohlcv_data)
        assert features.shape[1] > 10  # Should have multiple features
    
    def test_create_target(self, sample_ohlcv_data):
        engineer = FeatureEngineer()
        target = engineer.create_target(sample_ohlcv_data, lookahead=1, threshold=0.0001)
        
        assert len(target) == len(sample_ohlcv_data)
        # Target should be -1, 0, or 1
        valid_target = target.dropna()
        assert set(valid_target.unique()).issubset({-1, 0, 1})
    
    def test_normalize_features(self, sample_ohlcv_data):
        engineer = FeatureEngineer()
        features = engineer.create_features(sample_ohlcv_data)
        normalized = engineer.normalize_features(features)
        
        assert normalized.shape == features.shape
        # Normalized features should have reasonable range
        assert normalized.dropna().abs().max().max() < 100


class TestRiskManagement:
    """Tests for risk management"""
    
    def test_var_calculator(self, sample_returns):
        calculator = VaRCalculator()
        
        var_95 = calculator.historical_var(sample_returns, 0.95)
        var_99 = calculator.historical_var(sample_returns, 0.99)
        
        # VaR 99 should be more extreme than VaR 95
        assert var_99 <= var_95
        assert var_95 < 0  # Should be negative (loss)
    
    def test_cvar_calculator(self, sample_returns):
        calculator = VaRCalculator()
        
        cvar = calculator.cvar(sample_returns, 0.95)
        var = calculator.historical_var(sample_returns, 0.95)
        
        # CVaR should be more extreme than VaR
        assert cvar <= var
    
    def test_monte_carlo_simulator(self, sample_returns):
        simulator = MonteCarloSimulator(num_simulations=100)
        
        results = simulator.simulate_returns(sample_returns, horizon=10)
        
        assert 'mean_return' in results
        assert 'var_95' in results
        assert 'cvar_95' in results
        assert results['num_simulations'] == 100
    
    def test_position_sizer(self):
        sizer = DynamicPositionSizer(
            base_risk=0.01,
            max_risk=0.02,
            min_risk=0.005
        )
        
        # Test basic position sizing
        size = sizer.calculate_position_size(
            account_balance=1000,
            entry_price=1.1000,
            stop_loss=1.0950,
            confidence=0.7
        )
        
        assert size > 0
        assert size <= 1000 * 0.02 / 0.005  # Max risk constraint
    
    def test_risk_manager(self):
        manager = RiskManager()
        
        # Test risk limits check
        can_trade, reason = manager.check_risk_limits(account_balance=100)
        
        assert isinstance(can_trade, bool)
        assert isinstance(reason, str)


class TestRegimeDetection:
    """Tests for regime detection"""
    
    def test_feature_extractor(self, sample_ohlcv_data):
        extractor = RegimeFeatureExtractor()
        features = extractor.extract_features(sample_ohlcv_data)
        
        assert isinstance(features, pd.DataFrame)
        assert 'returns' in features.columns
        assert 'volatility' in features.columns
    
    def test_hmm_detector(self, sample_ohlcv_data):
        detector = HMMRegimeDetector(n_regimes=3)
        
        # Fit detector
        detector.fit(sample_ohlcv_data)
        
        # Predict regime
        regime = detector.predict(sample_ohlcv_data)
        
        assert regime is not None
        assert hasattr(regime, 'name')
        assert hasattr(regime, 'probability')
        assert 0 <= regime.probability <= 1
    
    def test_regime_manager(self, sample_ohlcv_data):
        manager = RegimeManager()
        
        # Fit and detect
        manager.fit(sample_ohlcv_data)
        regime = manager.detect_regime(sample_ohlcv_data)
        
        assert regime is not None
        
        # Get strategy weights
        weights = manager.get_strategy_weights()
        assert isinstance(weights, dict)
        assert sum(weights.values()) > 0


class TestDecisions:
    """Tests for trading decisions"""
    
    def test_fvg_detector(self, sample_ohlcv_data):
        detector = FVGDetector(min_gap_pips=3)
        fvgs = detector.detect_fvg(sample_ohlcv_data, pip_value=0.0001)
        
        assert isinstance(fvgs, list)
        for fvg in fvgs:
            assert hasattr(fvg, 'direction')
            assert hasattr(fvg, 'size')
    
    def test_liquidity_sweep_detector(self, sample_ohlcv_data):
        detector = LiquiditySweepDetector(lookback=20)
        sweeps = detector.detect_sweeps(sample_ohlcv_data, pip_value=0.0001)
        
        assert isinstance(sweeps, list)
        for sweep in sweeps:
            assert hasattr(sweep, 'direction')
            assert hasattr(sweep, 'sweep_level')
    
    def test_mtf_analyzer(self, sample_ohlcv_data):
        analyzer = MultiTimeframeAnalyzer()
        
        mtf_data = {
            'H1': sample_ohlcv_data,
            'H4': sample_ohlcv_data.resample('4H').agg({
                'open': 'first', 'high': 'max', 'low': 'min', 
                'close': 'last', 'volume': 'sum'
            }).dropna()
        }
        
        bias, alignment = analyzer.analyze_bias(mtf_data)
        
        assert bias in [TradeDirection.LONG, TradeDirection.SHORT, TradeDirection.NEUTRAL]
        assert 0 <= alignment <= 1


class TestExecution:
    """Tests for execution algorithms"""
    
    def test_slippage_model(self):
        model = SlippageModel(base_slippage_pips=0.5)
        
        slippage = model.calculate_slippage(
            volume=1.0,
            volatility=0.001,
            spread=0.0001
        )
        
        assert slippage >= 0
    
    def test_twap_executor(self):
        executor = TWAPExecutor(num_slices=5, interval_seconds=60)
        
        order = Order(
            symbol='EURUSD',
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1.0
        )
        
        schedule = executor.create_schedule(order)
        
        assert len(schedule) == 5
        assert sum(s['quantity'] for s in schedule) == pytest.approx(1.0, rel=0.01)
    
    def test_vwap_executor(self):
        executor = VWAPExecutor(num_slices=5)
        
        # Create volume profile
        volume_profile = [100, 200, 300, 200, 100]
        
        order = Order(
            symbol='EURUSD',
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1.0
        )
        
        schedule = executor.create_schedule(order, volume_profile)
        
        assert len(schedule) == 5
        # Higher volume periods should have larger slices
        quantities = [s['quantity'] for s in schedule]
        assert quantities[2] > quantities[0]  # Middle has highest volume


class TestAgentic:
    """Tests for agentic learning system"""
    
    def test_trade_journal(self):
        journal = TradeJournal()
        
        # Log a trade
        journal.log_trade(
            ticket='TEST001',
            symbol='EURUSD',
            direction='long',
            entry_price=1.1000,
            exit_price=1.1050,
            profit=50.0,
            entry_reason='trend_following',
            exit_reason='take_profit',
            confidence=0.75,
            indicators={'rsi': 45, 'adx': 35},
            regime='trending'
        )
        
        trades = journal.get_trades_by_date(datetime.now().date())
        assert len(trades) == 1
        assert trades[0]['profit'] == 50.0
    
    def test_insight_engine(self):
        engine = InsightEngine()
        
        # Generate insights from sample trades
        trades = [
            {
                'profit': 50, 'regime': 'trending', 'strategy': 'trend_following',
                'indicators': {'adx': 45}, 'confidence': 0.8
            },
            {
                'profit': 30, 'regime': 'trending', 'strategy': 'trend_following',
                'indicators': {'adx': 42}, 'confidence': 0.75
            },
            {
                'profit': -20, 'regime': 'ranging', 'strategy': 'trend_following',
                'indicators': {'adx': 18}, 'confidence': 0.6
            }
        ]
        
        insights = engine.generate_insights_from_trades(trades)
        
        assert isinstance(insights, list)
    
    def test_performance_analyzer(self):
        analyzer = PerformanceAnalyzer()
        
        trade = {
            'ticket': 'TEST001',
            'symbol': 'EURUSD',
            'direction': 'long',
            'entry_price': 1.1000,
            'exit_price': 1.1050,
            'profit': 50.0,
            'entry_reason': 'trend_following',
            'exit_reason': 'take_profit',
            'confidence': 0.75,
            'indicators': {'rsi': 45, 'adx': 35}
        }
        
        analysis = analyzer.analyze_trade(trade)
        
        assert analysis is not None
        assert analysis.profit == 50.0
        assert analysis.was_profitable == True
    
    def test_agentic_learning_system(self):
        system = AgenticLearningSystem()
        
        # Test trading parameters
        params = system.get_trading_parameters()
        
        assert 'trading_mode' in params
        assert 'base_risk' in params
        assert 'leverage_multiplier' in params
        
        # Test should_take_trade
        signal = {
            'confidence': 0.7,
            'strategy': 'trend_following',
            'regime': 'trending'
        }
        
        should_trade, reason, adj_confidence = system.should_take_trade(signal)
        
        assert isinstance(should_trade, bool)
        assert isinstance(reason, str)
        assert 0 <= adj_confidence <= 1


class TestBacktesting:
    """Tests for backtesting framework"""
    
    def test_cost_model(self):
        model = CostModel(spread_pips=1.0, commission_per_lot=7.0, slippage_pips=0.5)
        
        costs = model.get_total_cost(volume=1.0, pip_value=10.0, volatility=1.0)
        
        assert 'spread' in costs
        assert 'commission' in costs
        assert 'slippage' in costs
        assert 'total' in costs
        assert costs['total'] == costs['spread'] + costs['commission'] + costs['slippage']
    
    def test_backtest_engine(self, sample_ohlcv_data):
        engine = BacktestEngine(initial_balance=100)
        strategy = create_sample_strategy()
        
        result = engine.run(sample_ohlcv_data, strategy, 'EURUSD')
        
        assert result.initial_balance == 100
        assert result.total_trades >= 0
        assert 0 <= result.win_rate <= 1
        assert len(result.equity_curve) > 0
    
    def test_sample_strategy(self, sample_ohlcv_data):
        strategy = create_sample_strategy(
            rsi_oversold=30,
            rsi_overbought=70,
            adx_threshold=25
        )
        
        # Add indicators to data
        indicators = TechnicalIndicators()
        sample_ohlcv_data['rsi'] = indicators.rsi(sample_ohlcv_data['close'], 14)
        adx, _, _ = indicators.adx(
            sample_ohlcv_data['high'],
            sample_ohlcv_data['low'],
            sample_ohlcv_data['close']
        )
        sample_ohlcv_data['adx'] = adx
        macd, signal, hist = indicators.macd(sample_ohlcv_data['close'])
        sample_ohlcv_data['macd_hist'] = hist
        
        # Test strategy on a bar
        bar = sample_ohlcv_data.iloc[-1]
        signal = strategy(sample_ohlcv_data, bar)
        
        # Signal can be None or a dict
        if signal is not None:
            assert 'direction' in signal
            assert signal['direction'] in [-1, 0, 1]


class TestIntegration:
    """Integration tests"""
    
    def test_full_pipeline(self, sample_ohlcv_data):
        """Test full trading pipeline"""
        # 1. Create features
        engineer = FeatureEngineer()
        features = engineer.create_features(sample_ohlcv_data)
        
        # 2. Detect regime
        regime_manager = RegimeManager()
        regime_manager.fit(sample_ohlcv_data)
        regime = regime_manager.detect_regime(sample_ohlcv_data)
        
        # 3. Run backtest
        engine = BacktestEngine(initial_balance=100)
        strategy = create_sample_strategy()
        result = engine.run(sample_ohlcv_data, strategy, 'EURUSD')
        
        # 4. Analyze with agentic system
        system = AgenticLearningSystem()
        
        # Convert trades to dict format
        trades = [{
            'ticket': str(t.entry_time),
            'symbol': t.symbol,
            'direction': 'long' if t.direction == 1 else 'short',
            'entry_price': t.entry_price,
            'exit_price': t.exit_price,
            'profit': t.net_profit,
            'entry_reason': t.strategy,
            'exit_reason': 'closed',
            'confidence': t.confidence,
            'indicators': t.indicators
        } for t in result.trades[:10]]  # First 10 trades
        
        if trades:
            report = system.run_daily_learning_cycle(trades, 100)
            assert report is not None
    
    def test_risk_integration(self, sample_ohlcv_data):
        """Test risk management integration"""
        # Run backtest
        engine = BacktestEngine(initial_balance=100)
        strategy = create_sample_strategy()
        result = engine.run(sample_ohlcv_data, strategy, 'EURUSD')
        
        # Calculate returns from equity curve
        if len(result.equity_curve) > 1:
            equity = np.array(result.equity_curve)
            returns = np.diff(equity) / equity[:-1]
            
            # Calculate risk metrics
            calculator = VaRCalculator()
            var_95 = calculator.historical_var(returns, 0.95)
            cvar_95 = calculator.cvar(returns, 0.95)
            
            assert var_95 <= 0 or len(returns) < 10  # VaR should be negative (loss)
            assert cvar_95 <= var_95 or len(returns) < 10  # CVaR more extreme


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
