#!/usr/bin/env python3
"""
Tier 2: Portfolio Risk Management
Institutional-grade portfolio risk controls like BlackRock's Aladdin

Features:
1. Currency Exposure Limits - Max exposure per currency (e.g., max 30% USD exposure)
2. Concentration Limits - Max position size per symbol, max correlated positions
3. Portfolio Kill Switch - Auto-halt trading if drawdown exceeds threshold
4. Stress Testing - VaR, Expected Shortfall, scenario analysis
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import json
import sqlite3
import threading
import os

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Portfolio risk level classification"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    HALTED = "halted"


class KillSwitchReason(Enum):
    """Reasons for triggering portfolio kill switch"""
    DRAWDOWN_LIMIT = "drawdown_limit"
    DAILY_LOSS_LIMIT = "daily_loss_limit"
    CONCENTRATION_BREACH = "concentration_breach"
    CORRELATION_BREACH = "correlation_breach"
    MANUAL = "manual"
    STRESS_TEST_FAILURE = "stress_test_failure"


@dataclass
class CurrencyExposure:
    """Currency exposure tracking"""
    currency: str
    long_exposure: float  # Total long exposure in account currency
    short_exposure: float  # Total short exposure in account currency
    net_exposure: float  # Net exposure (long - short)
    gross_exposure: float  # Gross exposure (long + short)
    exposure_pct: float  # Exposure as % of account equity
    limit_pct: float  # Maximum allowed exposure %
    is_breached: bool  # Whether limit is breached
    positions: List[str] = field(default_factory=list)  # Position IDs contributing


@dataclass
class ConcentrationMetrics:
    """Position concentration metrics"""
    symbol: str
    position_size_lots: float
    position_value: float
    pct_of_equity: float
    max_allowed_pct: float
    is_breached: bool
    correlated_symbols: List[str] = field(default_factory=list)
    correlation_exposure: float = 0.0


@dataclass
class StressTestResult:
    """Result of a stress test scenario"""
    scenario_name: str
    description: str
    simulated_pnl: float
    simulated_pnl_pct: float
    var_95: float  # 95% Value at Risk
    var_99: float  # 99% Value at Risk
    expected_shortfall: float  # Expected Shortfall (CVaR)
    max_drawdown: float
    recovery_time_days: int
    passed: bool
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PortfolioRiskState:
    """Current portfolio risk state"""
    timestamp: datetime
    risk_level: RiskLevel
    total_equity: float
    total_exposure: float
    exposure_pct: float
    unrealized_pnl: float
    realized_pnl_today: float
    drawdown_pct: float
    peak_equity: float
    currency_exposures: List[CurrencyExposure] = field(default_factory=list)
    concentration_metrics: List[ConcentrationMetrics] = field(default_factory=list)
    stress_test_results: List[StressTestResult] = field(default_factory=list)
    kill_switch_active: bool = False
    kill_switch_reason: Optional[KillSwitchReason] = None
    warnings: List[str] = field(default_factory=list)
    breaches: List[str] = field(default_factory=list)


@dataclass
class RiskLimits:
    """Configurable risk limits"""
    # Drawdown limits
    max_drawdown_pct: float = 0.15  # 15% max drawdown triggers kill switch
    daily_loss_limit_pct: float = 0.05  # 5% daily loss limit
    
    # Currency exposure limits
    max_single_currency_exposure_pct: float = 0.40  # 40% max exposure to single currency
    max_usd_exposure_pct: float = 0.50  # 50% max USD exposure (since most pairs have USD)
    
    # Concentration limits
    max_position_size_pct: float = 0.10  # 10% max single position size
    max_correlated_exposure_pct: float = 0.25  # 25% max correlated positions
    max_positions: int = 5  # Maximum number of open positions
    
    # Stress test thresholds
    var_95_limit_pct: float = 0.08  # 8% VaR limit
    expected_shortfall_limit_pct: float = 0.12  # 12% ES limit
    
    # Warning thresholds (% of limit before warning)
    warning_threshold: float = 0.75  # Warn at 75% of limit


# Currency correlation matrix (simplified - in production would be dynamic)
CURRENCY_CORRELATIONS = {
    ('EUR', 'GBP'): 0.85,
    ('EUR', 'CHF'): 0.75,
    ('EUR', 'JPY'): 0.45,
    ('GBP', 'CHF'): 0.70,
    ('GBP', 'JPY'): 0.40,
    ('CHF', 'JPY'): 0.55,
    ('AUD', 'NZD'): 0.90,
    ('AUD', 'CAD'): 0.75,
    ('USD', 'CAD'): 0.65,
}


class CurrencyExposureManager:
    """Manages currency exposure limits"""
    
    def __init__(self, limits: RiskLimits):
        self.limits = limits
        self._exposure_cache: Dict[str, CurrencyExposure] = {}
    
    def calculate_exposures(self, positions: List[Dict], equity: float) -> List[CurrencyExposure]:
        """Calculate currency exposures from open positions"""
        currency_data: Dict[str, Dict] = {}
        
        for pos in positions:
            symbol = pos.get('symbol', '')
            volume = pos.get('volume', 0)
            direction = pos.get('type', 0)  # 0 = buy, 1 = sell
            price = pos.get('price_open', 0)
            
            # Extract currencies from symbol (e.g., EURUSD -> EUR, USD)
            if len(symbol) >= 6:
                base_currency = symbol[:3]
                quote_currency = symbol[3:6]
                
                # Calculate position value in account currency (assuming USD account)
                position_value = volume * 100000 * price  # Standard lot = 100,000 units
                
                # Initialize currency data if needed
                for currency in [base_currency, quote_currency]:
                    if currency not in currency_data:
                        currency_data[currency] = {
                            'long': 0.0,
                            'short': 0.0,
                            'positions': []
                        }
                
                # Add exposure based on direction
                if direction == 0:  # Buy - long base, short quote
                    currency_data[base_currency]['long'] += position_value
                    currency_data[quote_currency]['short'] += position_value
                else:  # Sell - short base, long quote
                    currency_data[base_currency]['short'] += position_value
                    currency_data[quote_currency]['long'] += position_value
                
                currency_data[base_currency]['positions'].append(str(pos.get('ticket', '')))
                currency_data[quote_currency]['positions'].append(str(pos.get('ticket', '')))
        
        # Build exposure objects
        exposures = []
        for currency, data in currency_data.items():
            net = data['long'] - data['short']
            gross = data['long'] + data['short']
            exposure_pct = (gross / equity * 100) if equity > 0 else 0
            
            # Determine limit based on currency
            if currency == 'USD':
                limit_pct = self.limits.max_usd_exposure_pct * 100
            else:
                limit_pct = self.limits.max_single_currency_exposure_pct * 100
            
            exposure = CurrencyExposure(
                currency=currency,
                long_exposure=data['long'],
                short_exposure=data['short'],
                net_exposure=net,
                gross_exposure=gross,
                exposure_pct=exposure_pct,
                limit_pct=limit_pct,
                is_breached=exposure_pct > limit_pct,
                positions=list(set(data['positions']))
            )
            exposures.append(exposure)
            self._exposure_cache[currency] = exposure
        
        return exposures
    
    def check_new_trade_exposure(self, symbol: str, direction: str, volume: float, 
                                  current_positions: List[Dict], equity: float) -> Tuple[bool, str]:
        """Check if a new trade would breach currency exposure limits"""
        # Simulate adding the new position
        simulated_positions = current_positions.copy()
        simulated_positions.append({
            'symbol': symbol,
            'volume': volume,
            'type': 0 if direction.upper() == 'BUY' else 1,
            'price_open': 1.0,  # Placeholder
            'ticket': 'simulated'
        })
        
        exposures = self.calculate_exposures(simulated_positions, equity)
        
        for exp in exposures:
            if exp.is_breached:
                return False, f"Trade would breach {exp.currency} exposure limit ({exp.exposure_pct:.1f}% > {exp.limit_pct:.1f}%)"
        
        return True, "Currency exposure within limits"


class ConcentrationManager:
    """Manages position concentration limits"""
    
    def __init__(self, limits: RiskLimits):
        self.limits = limits
    
    def calculate_concentration(self, positions: List[Dict], equity: float) -> List[ConcentrationMetrics]:
        """Calculate concentration metrics for all positions"""
        metrics = []
        
        # Group positions by symbol
        symbol_positions: Dict[str, List[Dict]] = {}
        for pos in positions:
            symbol = pos.get('symbol', '')
            if symbol not in symbol_positions:
                symbol_positions[symbol] = []
            symbol_positions[symbol].append(pos)
        
        for symbol, pos_list in symbol_positions.items():
            total_volume = sum(p.get('volume', 0) for p in pos_list)
            total_value = sum(p.get('volume', 0) * 100000 * p.get('price_open', 1) for p in pos_list)
            pct_of_equity = (total_value / equity * 100) if equity > 0 else 0
            max_allowed = self.limits.max_position_size_pct * 100
            
            # Find correlated symbols
            correlated = self._find_correlated_symbols(symbol, list(symbol_positions.keys()))
            correlation_exposure = sum(
                sum(p.get('volume', 0) * 100000 * p.get('price_open', 1) for p in symbol_positions.get(s, []))
                for s in correlated
            )
            
            metric = ConcentrationMetrics(
                symbol=symbol,
                position_size_lots=total_volume,
                position_value=total_value,
                pct_of_equity=pct_of_equity,
                max_allowed_pct=max_allowed,
                is_breached=pct_of_equity > max_allowed,
                correlated_symbols=correlated,
                correlation_exposure=correlation_exposure
            )
            metrics.append(metric)
        
        return metrics
    
    def _find_correlated_symbols(self, symbol: str, all_symbols: List[str]) -> List[str]:
        """Find symbols correlated with the given symbol"""
        if len(symbol) < 6:
            return []
        
        base = symbol[:3]
        quote = symbol[3:6]
        correlated = []
        
        for other in all_symbols:
            if other == symbol or len(other) < 6:
                continue
            
            other_base = other[:3]
            other_quote = other[3:6]
            
            # Check if currencies overlap
            if base in [other_base, other_quote] or quote in [other_base, other_quote]:
                correlated.append(other)
                continue
            
            # Check correlation matrix
            for (c1, c2), corr in CURRENCY_CORRELATIONS.items():
                if corr > 0.7:  # High correlation threshold
                    if (base == c1 and other_base == c2) or (base == c2 and other_base == c1):
                        correlated.append(other)
                        break
        
        return correlated
    
    def check_new_trade_concentration(self, symbol: str, volume: float, 
                                       current_positions: List[Dict], equity: float) -> Tuple[bool, str]:
        """Check if a new trade would breach concentration limits"""
        # Check max positions
        unique_symbols = set(p.get('symbol', '') for p in current_positions)
        if symbol not in unique_symbols and len(unique_symbols) >= self.limits.max_positions:
            return False, f"Maximum positions ({self.limits.max_positions}) reached"
        
        # Simulate adding the new position
        simulated_positions = current_positions.copy()
        simulated_positions.append({
            'symbol': symbol,
            'volume': volume,
            'price_open': 1.0,
            'ticket': 'simulated'
        })
        
        metrics = self.calculate_concentration(simulated_positions, equity)
        
        for m in metrics:
            if m.symbol == symbol and m.is_breached:
                return False, f"Trade would breach concentration limit for {symbol} ({m.pct_of_equity:.1f}% > {m.max_allowed_pct:.1f}%)"
        
        # Check correlated exposure
        total_correlated = sum(m.correlation_exposure for m in metrics if m.symbol == symbol)
        correlated_pct = (total_correlated / equity * 100) if equity > 0 else 0
        max_correlated = self.limits.max_correlated_exposure_pct * 100
        
        if correlated_pct > max_correlated:
            return False, f"Trade would breach correlated exposure limit ({correlated_pct:.1f}% > {max_correlated:.1f}%)"
        
        return True, "Concentration within limits"


class PortfolioKillSwitch:
    """Portfolio-level kill switch for emergency risk control"""
    
    def __init__(self, limits: RiskLimits, db_path: str = "portfolio_risk.db"):
        self.limits = limits
        self.db_path = db_path
        self._is_active = False
        self._reason: Optional[KillSwitchReason] = None
        self._activated_at: Optional[datetime] = None
        self._lock = threading.Lock()
        
        self._init_db()
        self._load_state()
    
    def _init_db(self):
        """Initialize SQLite database for kill switch state"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS kill_switch_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                action TEXT,
                reason TEXT,
                details TEXT
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS kill_switch_state (
                id INTEGER PRIMARY KEY,
                is_active INTEGER,
                reason TEXT,
                activated_at TEXT
            )
        ''')
        conn.commit()
        conn.close()
    
    def _load_state(self):
        """Load kill switch state from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT is_active, reason, activated_at FROM kill_switch_state WHERE id = 1')
            row = cursor.fetchone()
            if row:
                self._is_active = bool(row[0])
                self._reason = KillSwitchReason(row[1]) if row[1] else None
                self._activated_at = datetime.fromisoformat(row[2]) if row[2] else None
            conn.close()
        except Exception as e:
            logger.warning(f"Error loading kill switch state: {e}")
    
    def _save_state(self):
        """Save kill switch state to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO kill_switch_state (id, is_active, reason, activated_at)
                VALUES (1, ?, ?, ?)
            ''', (
                1 if self._is_active else 0,
                self._reason.value if self._reason else None,
                self._activated_at.isoformat() if self._activated_at else None
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning(f"Error saving kill switch state: {e}")
    
    def _log_action(self, action: str, reason: str, details: Dict):
        """Log kill switch action to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO kill_switch_history (timestamp, action, reason, details)
                VALUES (?, ?, ?, ?)
            ''', (datetime.now().isoformat(), action, reason, json.dumps(details)))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning(f"Error logging kill switch action: {e}")
    
    @property
    def is_active(self) -> bool:
        return self._is_active
    
    @property
    def reason(self) -> Optional[KillSwitchReason]:
        return self._reason
    
    def activate(self, reason: KillSwitchReason, details: Dict = None):
        """Activate the kill switch"""
        with self._lock:
            if not self._is_active:
                self._is_active = True
                self._reason = reason
                self._activated_at = datetime.now()
                self._save_state()
                self._log_action('ACTIVATE', reason.value, details or {})
                logger.critical(f"[KILL SWITCH] ACTIVATED - Reason: {reason.value}")
    
    def deactivate(self, authorized_by: str = "system"):
        """Deactivate the kill switch (requires authorization)"""
        with self._lock:
            if self._is_active:
                self._log_action('DEACTIVATE', f"Authorized by: {authorized_by}", {
                    'was_active_for': str(datetime.now() - self._activated_at) if self._activated_at else 'unknown'
                })
                self._is_active = False
                self._reason = None
                self._activated_at = None
                self._save_state()
                logger.warning(f"[KILL SWITCH] Deactivated by {authorized_by}")
    
    def check_drawdown(self, current_equity: float, peak_equity: float) -> bool:
        """Check if drawdown exceeds limit"""
        if peak_equity <= 0:
            return False
        
        drawdown_pct = (peak_equity - current_equity) / peak_equity
        
        if drawdown_pct >= self.limits.max_drawdown_pct:
            self.activate(KillSwitchReason.DRAWDOWN_LIMIT, {
                'current_equity': current_equity,
                'peak_equity': peak_equity,
                'drawdown_pct': drawdown_pct,
                'limit_pct': self.limits.max_drawdown_pct
            })
            return True
        
        return False
    
    def check_daily_loss(self, daily_pnl: float, starting_equity: float) -> bool:
        """Check if daily loss exceeds limit"""
        if starting_equity <= 0:
            return False
        
        daily_loss_pct = abs(daily_pnl) / starting_equity if daily_pnl < 0 else 0
        
        if daily_loss_pct >= self.limits.daily_loss_limit_pct:
            self.activate(KillSwitchReason.DAILY_LOSS_LIMIT, {
                'daily_pnl': daily_pnl,
                'starting_equity': starting_equity,
                'loss_pct': daily_loss_pct,
                'limit_pct': self.limits.daily_loss_limit_pct
            })
            return True
        
        return False


class StressTestEngine:
    """Portfolio stress testing engine"""
    
    # Predefined stress scenarios
    SCENARIOS = {
        'flash_crash': {
            'description': '2010-style flash crash - 5% adverse move in 10 minutes',
            'price_shock_pct': -0.05,
            'volatility_multiplier': 3.0,
            'liquidity_factor': 0.3  # 70% reduction in liquidity
        },
        'brexit_shock': {
            'description': 'Brexit-style GBP shock - 10% adverse move',
            'price_shock_pct': -0.10,
            'volatility_multiplier': 2.5,
            'liquidity_factor': 0.5
        },
        'snb_shock': {
            'description': 'SNB 2015-style CHF shock - 20% adverse move',
            'price_shock_pct': -0.20,
            'volatility_multiplier': 5.0,
            'liquidity_factor': 0.1
        },
        'moderate_adverse': {
            'description': 'Moderate adverse scenario - 2% move',
            'price_shock_pct': -0.02,
            'volatility_multiplier': 1.5,
            'liquidity_factor': 0.8
        },
        'correlation_breakdown': {
            'description': 'Correlation breakdown - all positions move against',
            'price_shock_pct': -0.03,
            'volatility_multiplier': 2.0,
            'liquidity_factor': 0.6,
            'correlation_shock': True
        }
    }
    
    def __init__(self, limits: RiskLimits):
        self.limits = limits
    
    def run_stress_test(self, positions: List[Dict], equity: float, 
                        scenario_name: str) -> StressTestResult:
        """Run a stress test scenario"""
        if scenario_name not in self.SCENARIOS:
            raise ValueError(f"Unknown scenario: {scenario_name}")
        
        scenario = self.SCENARIOS[scenario_name]
        
        # Calculate simulated P&L
        total_exposure = sum(
            p.get('volume', 0) * 100000 * p.get('price_open', 1)
            for p in positions
        )
        
        # Apply price shock
        simulated_pnl = total_exposure * scenario['price_shock_pct']
        
        # Adjust for liquidity (slippage)
        liquidity_cost = total_exposure * (1 - scenario['liquidity_factor']) * 0.01
        simulated_pnl -= liquidity_cost
        
        simulated_pnl_pct = (simulated_pnl / equity * 100) if equity > 0 else 0
        
        # Calculate VaR (simplified - in production would use historical simulation)
        var_95 = total_exposure * 0.02 * scenario['volatility_multiplier']  # ~2% daily VaR
        var_99 = total_exposure * 0.03 * scenario['volatility_multiplier']  # ~3% daily VaR
        expected_shortfall = var_99 * 1.2  # ES typically 20% higher than VaR
        
        # Check if passed
        var_95_pct = (var_95 / equity * 100) if equity > 0 else 0
        es_pct = (expected_shortfall / equity * 100) if equity > 0 else 0
        
        passed = (
            var_95_pct <= self.limits.var_95_limit_pct * 100 and
            es_pct <= self.limits.expected_shortfall_limit_pct * 100
        )
        
        return StressTestResult(
            scenario_name=scenario_name,
            description=scenario['description'],
            simulated_pnl=simulated_pnl,
            simulated_pnl_pct=simulated_pnl_pct,
            var_95=var_95,
            var_99=var_99,
            expected_shortfall=expected_shortfall,
            max_drawdown=abs(simulated_pnl_pct),
            recovery_time_days=int(abs(simulated_pnl) / (equity * 0.01)) if equity > 0 else 0,
            passed=passed,
            details={
                'total_exposure': total_exposure,
                'liquidity_cost': liquidity_cost,
                'var_95_pct': var_95_pct,
                'es_pct': es_pct
            }
        )
    
    def run_all_scenarios(self, positions: List[Dict], equity: float) -> List[StressTestResult]:
        """Run all stress test scenarios"""
        results = []
        for scenario_name in self.SCENARIOS:
            try:
                result = self.run_stress_test(positions, equity, scenario_name)
                results.append(result)
            except Exception as e:
                logger.warning(f"Stress test {scenario_name} failed: {e}")
        return results


class PortfolioRiskManager:
    """Main portfolio risk management class - Tier 2 implementation"""
    
    def __init__(self, limits: RiskLimits = None, db_path: str = "portfolio_risk.db"):
        self.limits = limits or RiskLimits()
        self.db_path = db_path
        
        # Initialize components
        self.exposure_manager = CurrencyExposureManager(self.limits)
        self.concentration_manager = ConcentrationManager(self.limits)
        self.kill_switch = PortfolioKillSwitch(self.limits, db_path)
        self.stress_engine = StressTestEngine(self.limits)
        
        # State tracking
        self._peak_equity = 0.0
        self._daily_starting_equity = 0.0
        self._last_daily_reset = datetime.now().date()
        self._lock = threading.Lock()
        
        logger.info(f"Portfolio Risk Manager initialized with limits: max_drawdown={self.limits.max_drawdown_pct:.0%}, "
                   f"daily_loss={self.limits.daily_loss_limit_pct:.0%}, max_positions={self.limits.max_positions}")
    
    def update_equity_tracking(self, current_equity: float):
        """Update equity tracking for drawdown calculations"""
        with self._lock:
            # Update peak equity
            if current_equity > self._peak_equity:
                self._peak_equity = current_equity
            
            # Reset daily tracking at midnight
            today = datetime.now().date()
            if today != self._last_daily_reset:
                self._daily_starting_equity = current_equity
                self._last_daily_reset = today
            
            # Initialize if first call
            if self._daily_starting_equity == 0:
                self._daily_starting_equity = current_equity
    
    def get_portfolio_risk_state(self, positions: List[Dict], equity: float, 
                                  unrealized_pnl: float = 0) -> PortfolioRiskState:
        """Get comprehensive portfolio risk state"""
        self.update_equity_tracking(equity)
        
        warnings = []
        breaches = []
        
        # Calculate currency exposures
        currency_exposures = self.exposure_manager.calculate_exposures(positions, equity)
        for exp in currency_exposures:
            if exp.is_breached:
                breaches.append(f"Currency exposure breach: {exp.currency} at {exp.exposure_pct:.1f}%")
            elif exp.exposure_pct > exp.limit_pct * self.limits.warning_threshold:
                warnings.append(f"Currency exposure warning: {exp.currency} at {exp.exposure_pct:.1f}%")
        
        # Calculate concentration metrics
        concentration_metrics = self.concentration_manager.calculate_concentration(positions, equity)
        for metric in concentration_metrics:
            if metric.is_breached:
                breaches.append(f"Concentration breach: {metric.symbol} at {metric.pct_of_equity:.1f}%")
            elif metric.pct_of_equity > metric.max_allowed_pct * self.limits.warning_threshold:
                warnings.append(f"Concentration warning: {metric.symbol} at {metric.pct_of_equity:.1f}%")
        
        # Check kill switch conditions
        drawdown_pct = (self._peak_equity - equity) / self._peak_equity if self._peak_equity > 0 else 0
        daily_pnl = equity - self._daily_starting_equity
        
        self.kill_switch.check_drawdown(equity, self._peak_equity)
        self.kill_switch.check_daily_loss(daily_pnl, self._daily_starting_equity)
        
        # Run stress tests
        stress_results = self.stress_engine.run_all_scenarios(positions, equity)
        for result in stress_results:
            if not result.passed:
                warnings.append(f"Stress test warning: {result.scenario_name} - VaR={result.var_95:.2f}")
        
        # Determine overall risk level
        if self.kill_switch.is_active:
            risk_level = RiskLevel.HALTED
        elif breaches:
            risk_level = RiskLevel.CRITICAL
        elif len(warnings) >= 3:
            risk_level = RiskLevel.HIGH
        elif warnings:
            risk_level = RiskLevel.MEDIUM
        else:
            risk_level = RiskLevel.LOW
        
        # Calculate total exposure
        total_exposure = sum(
            p.get('volume', 0) * 100000 * p.get('price_open', 1)
            for p in positions
        )
        exposure_pct = (total_exposure / equity * 100) if equity > 0 else 0
        
        return PortfolioRiskState(
            timestamp=datetime.now(),
            risk_level=risk_level,
            total_equity=equity,
            total_exposure=total_exposure,
            exposure_pct=exposure_pct,
            unrealized_pnl=unrealized_pnl,
            realized_pnl_today=daily_pnl,
            drawdown_pct=drawdown_pct,
            peak_equity=self._peak_equity,
            currency_exposures=currency_exposures,
            concentration_metrics=concentration_metrics,
            stress_test_results=stress_results,
            kill_switch_active=self.kill_switch.is_active,
            kill_switch_reason=self.kill_switch.reason,
            warnings=warnings,
            breaches=breaches
        )
    
    def can_open_trade(self, symbol: str, direction: str, volume: float,
                       current_positions: List[Dict], equity: float) -> Tuple[bool, str]:
        """Check if a new trade is allowed under risk limits"""
        # Check kill switch first
        if self.kill_switch.is_active:
            return False, f"Kill switch active: {self.kill_switch.reason.value if self.kill_switch.reason else 'unknown'}"
        
        # Check currency exposure
        allowed, reason = self.exposure_manager.check_new_trade_exposure(
            symbol, direction, volume, current_positions, equity
        )
        if not allowed:
            return False, reason
        
        # Check concentration
        allowed, reason = self.concentration_manager.check_new_trade_concentration(
            symbol, volume, current_positions, equity
        )
        if not allowed:
            return False, reason
        
        return True, "Trade allowed under portfolio risk limits"
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get a summary of current risk state"""
        return {
            'kill_switch_active': self.kill_switch.is_active,
            'kill_switch_reason': self.kill_switch.reason.value if self.kill_switch.reason else None,
            'peak_equity': self._peak_equity,
            'daily_starting_equity': self._daily_starting_equity,
            'limits': {
                'max_drawdown_pct': self.limits.max_drawdown_pct,
                'daily_loss_limit_pct': self.limits.daily_loss_limit_pct,
                'max_single_currency_exposure_pct': self.limits.max_single_currency_exposure_pct,
                'max_position_size_pct': self.limits.max_position_size_pct,
                'max_positions': self.limits.max_positions
            }
        }


# Global instance
_portfolio_risk_manager: Optional[PortfolioRiskManager] = None
_lock = threading.Lock()


def get_portfolio_risk_manager() -> PortfolioRiskManager:
    """Get or create the global portfolio risk manager instance"""
    global _portfolio_risk_manager
    with _lock:
        if _portfolio_risk_manager is None:
            _portfolio_risk_manager = PortfolioRiskManager()
        return _portfolio_risk_manager


# Convenience functions
def check_trade_risk(symbol: str, direction: str, volume: float,
                     current_positions: List[Dict], equity: float) -> Tuple[bool, str]:
    """Check if a trade is allowed under portfolio risk limits"""
    return get_portfolio_risk_manager().can_open_trade(
        symbol, direction, volume, current_positions, equity
    )


def get_portfolio_risk_state(positions: List[Dict], equity: float,
                              unrealized_pnl: float = 0) -> PortfolioRiskState:
    """Get current portfolio risk state"""
    return get_portfolio_risk_manager().get_portfolio_risk_state(
        positions, equity, unrealized_pnl
    )


def is_kill_switch_active() -> bool:
    """Check if kill switch is active"""
    return get_portfolio_risk_manager().kill_switch.is_active


def deactivate_kill_switch(authorized_by: str = "manual"):
    """Deactivate the kill switch"""
    get_portfolio_risk_manager().kill_switch.deactivate(authorized_by)


if __name__ == "__main__":
    # Test the portfolio risk manager
    logging.basicConfig(level=logging.INFO)
    
    manager = get_portfolio_risk_manager()
    
    # Test with sample positions
    test_positions = [
        {'symbol': 'EURUSD', 'volume': 0.1, 'type': 0, 'price_open': 1.1000, 'ticket': '1'},
        {'symbol': 'GBPUSD', 'volume': 0.1, 'type': 0, 'price_open': 1.2700, 'ticket': '2'},
    ]
    
    # Get risk state
    state = manager.get_portfolio_risk_state(test_positions, 100.0, 0)
    print(f"Risk Level: {state.risk_level.value}")
    print(f"Total Exposure: ${state.total_exposure:.2f} ({state.exposure_pct:.1f}%)")
    print(f"Kill Switch: {'ACTIVE' if state.kill_switch_active else 'inactive'}")
    print(f"Warnings: {state.warnings}")
    print(f"Breaches: {state.breaches}")
    
    # Test trade check
    allowed, reason = manager.can_open_trade('USDJPY', 'BUY', 0.1, test_positions, 100.0)
    print(f"New trade allowed: {allowed} - {reason}")
