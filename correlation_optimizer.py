"""
Priority 5: Correlation-Aware Portfolio Optimizer

Manages portfolio-level risk by considering correlations between positions.
Key features:
- Dynamic correlation matrix across currency pairs
- Avoid taking highly correlated positions
- Net exposure limits per currency
- Diversification scoring
- Portfolio-level VaR and risk metrics

This prevents the common mistake of thinking you're diversified when
you're actually just taking the same bet multiple times (e.g., long EURUSD
and long GBPUSD are essentially the same USD short bet).
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Set, Any
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)


class CurrencyExposure(Enum):
    """Types of currency exposure"""
    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"


@dataclass
class Position:
    """Represents a trading position"""
    symbol: str
    direction: str  # 'long' or 'short'
    size: float
    entry_price: float
    current_price: float
    pnl: float = 0.0
    
    @property
    def base_currency(self) -> str:
        """Get base currency (first 3 chars)"""
        return self.symbol[:3]
    
    @property
    def quote_currency(self) -> str:
        """Get quote currency (last 3 chars)"""
        return self.symbol[3:6] if len(self.symbol) >= 6 else self.symbol[3:]


@dataclass
class CorrelationConfig:
    """Configuration for correlation-aware optimization"""
    # Correlation thresholds
    high_correlation_threshold: float = 0.7    # Pairs above this are highly correlated
    max_correlated_positions: int = 2          # Max positions with correlation > threshold
    
    # Currency exposure limits
    max_single_currency_exposure: float = 0.4  # Max 40% exposure to any single currency
    max_usd_exposure: float = 0.5              # Max 50% USD exposure (it's in most pairs)
    
    # Diversification requirements
    min_diversification_score: float = 0.3     # Minimum diversification (0-1)
    target_diversification_score: float = 0.6  # Target diversification
    
    # Portfolio limits
    max_positions: int = 5                     # Maximum concurrent positions
    max_gross_exposure: float = 3.0            # Max gross exposure (sum of all positions)
    max_net_exposure: float = 1.5              # Max net exposure (long - short)
    
    # Risk limits
    max_portfolio_var: float = 0.02            # Max 2% portfolio VaR
    correlation_lookback_days: int = 30        # Days for correlation calculation


@dataclass
class CurrencyExposureReport:
    """Report of exposure to each currency"""
    currency: str
    long_exposure: float
    short_exposure: float
    net_exposure: float
    exposure_percent: float
    positions: List[str]


@dataclass
class CorrelationCheck:
    """Result of correlation check for a new position"""
    can_add: bool
    reason: str
    correlation_with_existing: Dict[str, float]
    highest_correlation: float
    highest_correlated_symbol: str
    diversification_impact: float
    warnings: List[str] = field(default_factory=list)


@dataclass
class PortfolioRiskReport:
    """Comprehensive portfolio risk report"""
    total_positions: int
    gross_exposure: float
    net_exposure: float
    diversification_score: float
    portfolio_var: float
    currency_exposures: Dict[str, CurrencyExposureReport]
    correlation_matrix: Dict[str, Dict[str, float]]
    risk_warnings: List[str]
    recommendations: List[str]


class CorrelationMatrix:
    """
    Manages dynamic correlation matrix for currency pairs.
    
    Correlations change over time, so we use rolling calculations.
    """
    
    # Static correlation estimates (used when insufficient data)
    STATIC_CORRELATIONS = {
        ('EURUSD', 'GBPUSD'): 0.85,
        ('EURUSD', 'USDCHF'): -0.90,
        ('EURUSD', 'USDJPY'): -0.30,
        ('EURUSD', 'AUDUSD'): 0.70,
        ('EURUSD', 'NZDUSD'): 0.65,
        ('EURUSD', 'USDCAD'): -0.75,
        ('GBPUSD', 'USDCHF'): -0.85,
        ('GBPUSD', 'USDJPY'): -0.25,
        ('GBPUSD', 'AUDUSD'): 0.65,
        ('GBPUSD', 'NZDUSD'): 0.60,
        ('GBPUSD', 'USDCAD'): -0.70,
        ('USDCHF', 'USDJPY'): 0.60,
        ('USDCHF', 'AUDUSD'): -0.65,
        ('USDCHF', 'NZDUSD'): -0.60,
        ('USDCHF', 'USDCAD'): 0.70,
        ('USDJPY', 'AUDUSD'): 0.55,
        ('USDJPY', 'NZDUSD'): 0.50,
        ('USDJPY', 'USDCAD'): 0.40,
        ('AUDUSD', 'NZDUSD'): 0.90,
        ('AUDUSD', 'USDCAD'): -0.60,
        ('NZDUSD', 'USDCAD'): -0.55,
        ('EURGBP', 'EURUSD'): 0.50,
        ('EURGBP', 'GBPUSD'): -0.50,
    }
    
    def __init__(self, lookback_days: int = 30):
        self.lookback_days = lookback_days
        self.price_history: Dict[str, List[Tuple[datetime, float]]] = defaultdict(list)
        self.correlation_cache: Dict[Tuple[str, str], Tuple[float, datetime]] = {}
        self.cache_duration = timedelta(hours=4)
        
        logger.info(f"CorrelationMatrix initialized with {lookback_days} day lookback")
    
    def update_price(self, symbol: str, price: float, timestamp: Optional[datetime] = None) -> None:
        """Update price history for a symbol"""
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        self.price_history[symbol].append((timestamp, price))
        
        # Keep only lookback period
        cutoff = timestamp - timedelta(days=self.lookback_days)
        self.price_history[symbol] = [
            (t, p) for t, p in self.price_history[symbol] if t > cutoff
        ]
    
    def get_correlation(self, symbol1: str, symbol2: str) -> float:
        """
        Get correlation between two symbols.
        
        Uses dynamic calculation if sufficient data, otherwise static estimates.
        """
        if symbol1 == symbol2:
            return 1.0
        
        # Normalize order for cache key
        key = tuple(sorted([symbol1, symbol2]))
        
        # Check cache
        if key in self.correlation_cache:
            corr, timestamp = self.correlation_cache[key]
            if datetime.utcnow() - timestamp < self.cache_duration:
                return corr
        
        # Try dynamic calculation
        corr = self._calculate_dynamic_correlation(symbol1, symbol2)
        
        if corr is not None:
            self.correlation_cache[key] = (corr, datetime.utcnow())
            return corr
        
        # Fall back to static correlation
        return self._get_static_correlation(symbol1, symbol2)
    
    def _calculate_dynamic_correlation(self, symbol1: str, symbol2: str) -> Optional[float]:
        """Calculate correlation from price history"""
        if symbol1 not in self.price_history or symbol2 not in self.price_history:
            return None
        
        prices1 = self.price_history[symbol1]
        prices2 = self.price_history[symbol2]
        
        if len(prices1) < 20 or len(prices2) < 20:
            return None
        
        # Align timestamps (simple approach - use last N prices)
        n = min(len(prices1), len(prices2), 100)
        
        returns1 = self._calculate_returns([p for _, p in prices1[-n:]])
        returns2 = self._calculate_returns([p for _, p in prices2[-n:]])
        
        if len(returns1) < 10 or len(returns2) < 10:
            return None
        
        # Calculate correlation
        min_len = min(len(returns1), len(returns2))
        returns1 = returns1[-min_len:]
        returns2 = returns2[-min_len:]
        
        try:
            corr = np.corrcoef(returns1, returns2)[0, 1]
            return corr if not np.isnan(corr) else None
        except:
            return None
    
    def _calculate_returns(self, prices: List[float]) -> List[float]:
        """Calculate returns from prices"""
        if len(prices) < 2:
            return []
        return [(prices[i] - prices[i-1]) / prices[i-1] 
                for i in range(1, len(prices)) if prices[i-1] != 0]
    
    def _get_static_correlation(self, symbol1: str, symbol2: str) -> float:
        """Get static correlation estimate"""
        key = tuple(sorted([symbol1, symbol2]))
        
        if key in self.STATIC_CORRELATIONS:
            return self.STATIC_CORRELATIONS[key]
        
        # Estimate based on shared currencies
        base1, quote1 = symbol1[:3], symbol1[3:6]
        base2, quote2 = symbol2[:3], symbol2[3:6]
        
        # Same base currency = positive correlation
        if base1 == base2:
            return 0.7
        
        # Same quote currency = positive correlation
        if quote1 == quote2:
            return 0.6
        
        # Base of one is quote of other = negative correlation
        if base1 == quote2 or base2 == quote1:
            return -0.5
        
        # No obvious relationship
        return 0.0
    
    def get_correlation_matrix(self, symbols: List[str]) -> Dict[str, Dict[str, float]]:
        """Get full correlation matrix for given symbols"""
        matrix = {}
        for s1 in symbols:
            matrix[s1] = {}
            for s2 in symbols:
                matrix[s1][s2] = self.get_correlation(s1, s2)
        return matrix


class CorrelationOptimizer:
    """
    Optimizes portfolio considering correlations between positions.
    
    Key responsibilities:
    1. Check if new position would create excessive correlation
    2. Calculate currency exposures
    3. Ensure diversification
    4. Manage portfolio-level risk
    """
    
    def __init__(self, config: Optional[CorrelationConfig] = None):
        self.config = config or CorrelationConfig()
        self.correlation_matrix = CorrelationMatrix(self.config.correlation_lookback_days)
        self.positions: Dict[str, Position] = {}
        
        logger.info("CorrelationOptimizer initialized")
    
    def add_position(self, position: Position) -> None:
        """Add a position to the portfolio"""
        key = f"{position.symbol}_{position.direction}"
        self.positions[key] = position
        logger.info(f"[CORRELATION] Added position: {key}")
    
    def remove_position(self, symbol: str, direction: str) -> None:
        """Remove a position from the portfolio"""
        key = f"{symbol}_{direction}"
        if key in self.positions:
            del self.positions[key]
            logger.info(f"[CORRELATION] Removed position: {key}")
    
    def update_price(self, symbol: str, price: float) -> None:
        """Update price for correlation calculation"""
        self.correlation_matrix.update_price(symbol, price)
    
    def check_new_position(self, symbol: str, direction: str, 
                          size: float) -> CorrelationCheck:
        """
        Check if a new position can be added without violating constraints.
        
        Args:
            symbol: Currency pair
            direction: 'long' or 'short'
            size: Position size
            
        Returns:
            CorrelationCheck with approval status and details
        """
        warnings = []
        correlations = {}
        highest_corr = 0.0
        highest_corr_symbol = ""
        
        # Check position count
        if len(self.positions) >= self.config.max_positions:
            return CorrelationCheck(
                can_add=False,
                reason=f"Maximum positions ({self.config.max_positions}) reached",
                correlation_with_existing={},
                highest_correlation=0.0,
                highest_correlated_symbol="",
                diversification_impact=0.0,
                warnings=["Portfolio at maximum capacity"]
            )
        
        # Check correlations with existing positions
        correlated_count = 0
        for key, pos in self.positions.items():
            corr = self.correlation_matrix.get_correlation(symbol, pos.symbol)
            
            # Adjust correlation based on direction
            # Same direction = correlation matters
            # Opposite direction = negative correlation matters
            if direction == pos.direction:
                effective_corr = corr
            else:
                effective_corr = -corr
            
            correlations[pos.symbol] = effective_corr
            
            if abs(effective_corr) > abs(highest_corr):
                highest_corr = effective_corr
                highest_corr_symbol = pos.symbol
            
            if effective_corr > self.config.high_correlation_threshold:
                correlated_count += 1
                warnings.append(f"High correlation ({effective_corr:.2f}) with {pos.symbol}")
        
        # Check if too many correlated positions
        if correlated_count >= self.config.max_correlated_positions:
            return CorrelationCheck(
                can_add=False,
                reason=f"Too many correlated positions ({correlated_count} >= {self.config.max_correlated_positions})",
                correlation_with_existing=correlations,
                highest_correlation=highest_corr,
                highest_correlated_symbol=highest_corr_symbol,
                diversification_impact=0.0,
                warnings=warnings
            )
        
        # Check currency exposure
        exposure_check = self._check_currency_exposure(symbol, direction, size)
        if not exposure_check[0]:
            return CorrelationCheck(
                can_add=False,
                reason=exposure_check[1],
                correlation_with_existing=correlations,
                highest_correlation=highest_corr,
                highest_correlated_symbol=highest_corr_symbol,
                diversification_impact=0.0,
                warnings=warnings + exposure_check[2]
            )
        warnings.extend(exposure_check[2])
        
        # Check gross/net exposure
        exposure_check = self._check_exposure_limits(direction, size)
        if not exposure_check[0]:
            return CorrelationCheck(
                can_add=False,
                reason=exposure_check[1],
                correlation_with_existing=correlations,
                highest_correlation=highest_corr,
                highest_correlated_symbol=highest_corr_symbol,
                diversification_impact=0.0,
                warnings=warnings
            )
        
        # Calculate diversification impact
        div_impact = self._calculate_diversification_impact(symbol, direction)
        
        if div_impact < 0 and self._get_diversification_score() < self.config.min_diversification_score:
            warnings.append("Position would reduce already low diversification")
        
        return CorrelationCheck(
            can_add=True,
            reason="Position approved",
            correlation_with_existing=correlations,
            highest_correlation=highest_corr,
            highest_correlated_symbol=highest_corr_symbol,
            diversification_impact=div_impact,
            warnings=warnings
        )
    
    def _check_currency_exposure(self, symbol: str, direction: str, 
                                 size: float) -> Tuple[bool, str, List[str]]:
        """Check if new position would exceed currency exposure limits"""
        warnings = []
        
        base = symbol[:3]
        quote = symbol[3:6] if len(symbol) >= 6 else symbol[3:]
        
        # Calculate current exposures
        exposures = self._calculate_currency_exposures()
        
        # Calculate new exposures
        total_size = sum(p.size for p in self.positions.values()) + size
        
        # For long position: long base, short quote
        # For short position: short base, long quote
        if direction == 'long':
            base_exposure = exposures.get(base, 0) + size / total_size
            quote_exposure = exposures.get(quote, 0) - size / total_size
        else:
            base_exposure = exposures.get(base, 0) - size / total_size
            quote_exposure = exposures.get(quote, 0) + size / total_size
        
        # Check USD exposure (special limit)
        if base == 'USD' or quote == 'USD':
            usd_exposure = abs(base_exposure if base == 'USD' else quote_exposure)
            if usd_exposure > self.config.max_usd_exposure:
                return False, f"USD exposure ({usd_exposure:.1%}) would exceed limit ({self.config.max_usd_exposure:.1%})", warnings
            if usd_exposure > self.config.max_usd_exposure * 0.8:
                warnings.append(f"USD exposure approaching limit ({usd_exposure:.1%})")
        
        # Check single currency exposure
        for currency, exposure in [(base, base_exposure), (quote, quote_exposure)]:
            if abs(exposure) > self.config.max_single_currency_exposure:
                return False, f"{currency} exposure ({abs(exposure):.1%}) would exceed limit ({self.config.max_single_currency_exposure:.1%})", warnings
        
        return True, "", warnings
    
    def _check_exposure_limits(self, direction: str, size: float) -> Tuple[bool, str]:
        """Check gross and net exposure limits"""
        current_gross = sum(p.size for p in self.positions.values())
        current_net = sum(
            p.size if p.direction == 'long' else -p.size 
            for p in self.positions.values()
        )
        
        new_gross = current_gross + size
        new_net = current_net + (size if direction == 'long' else -size)
        
        if new_gross > self.config.max_gross_exposure:
            return False, f"Gross exposure ({new_gross:.2f}) would exceed limit ({self.config.max_gross_exposure:.2f})"
        
        if abs(new_net) > self.config.max_net_exposure:
            return False, f"Net exposure ({new_net:.2f}) would exceed limit ({self.config.max_net_exposure:.2f})"
        
        return True, ""
    
    def _calculate_currency_exposures(self) -> Dict[str, float]:
        """Calculate exposure to each currency"""
        exposures = defaultdict(float)
        total_size = sum(p.size for p in self.positions.values())
        
        if total_size == 0:
            return dict(exposures)
        
        for pos in self.positions.values():
            base = pos.base_currency
            quote = pos.quote_currency
            weight = pos.size / total_size
            
            if pos.direction == 'long':
                exposures[base] += weight
                exposures[quote] -= weight
            else:
                exposures[base] -= weight
                exposures[quote] += weight
        
        return dict(exposures)
    
    def _calculate_diversification_impact(self, symbol: str, direction: str) -> float:
        """Calculate how adding this position would impact diversification"""
        if not self.positions:
            return 1.0  # First position always improves from 0
        
        current_div = self._get_diversification_score()
        
        # Simulate adding position
        temp_pos = Position(symbol, direction, 1.0, 0, 0)
        self.positions[f"{symbol}_{direction}_temp"] = temp_pos
        new_div = self._get_diversification_score()
        del self.positions[f"{symbol}_{direction}_temp"]
        
        return new_div - current_div
    
    def _get_diversification_score(self) -> float:
        """
        Calculate portfolio diversification score (0-1).
        
        Higher score = more diversified.
        Based on average pairwise correlation (lower = better).
        """
        if len(self.positions) < 2:
            return 0.0
        
        symbols = [p.symbol for p in self.positions.values()]
        directions = [p.direction for p in self.positions.values()]
        
        total_corr = 0.0
        count = 0
        
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                corr = self.correlation_matrix.get_correlation(symbols[i], symbols[j])
                
                # Adjust for direction
                if directions[i] == directions[j]:
                    effective_corr = corr
                else:
                    effective_corr = -corr
                
                total_corr += abs(effective_corr)
                count += 1
        
        if count == 0:
            return 0.0
        
        avg_corr = total_corr / count
        
        # Convert to diversification score (1 - avg_correlation)
        return max(0.0, min(1.0, 1.0 - avg_corr))
    
    def get_portfolio_risk_report(self) -> PortfolioRiskReport:
        """Generate comprehensive portfolio risk report"""
        # Calculate exposures
        currency_exposures = {}
        raw_exposures = self._calculate_currency_exposures()
        
        for currency, exposure in raw_exposures.items():
            positions_with_currency = [
                p.symbol for p in self.positions.values()
                if p.base_currency == currency or p.quote_currency == currency
            ]
            
            currency_exposures[currency] = CurrencyExposureReport(
                currency=currency,
                long_exposure=max(0, exposure),
                short_exposure=abs(min(0, exposure)),
                net_exposure=exposure,
                exposure_percent=abs(exposure),
                positions=positions_with_currency
            )
        
        # Calculate gross/net exposure
        gross_exposure = sum(p.size for p in self.positions.values())
        net_exposure = sum(
            p.size if p.direction == 'long' else -p.size 
            for p in self.positions.values()
        )
        
        # Get correlation matrix
        symbols = list(set(p.symbol for p in self.positions.values()))
        corr_matrix = self.correlation_matrix.get_correlation_matrix(symbols)
        
        # Calculate diversification
        div_score = self._get_diversification_score()
        
        # Calculate portfolio VaR (simplified)
        portfolio_var = self._calculate_portfolio_var()
        
        # Generate warnings
        warnings = []
        if gross_exposure > self.config.max_gross_exposure * 0.8:
            warnings.append(f"Gross exposure ({gross_exposure:.2f}) approaching limit")
        if abs(net_exposure) > self.config.max_net_exposure * 0.8:
            warnings.append(f"Net exposure ({net_exposure:.2f}) approaching limit")
        if div_score < self.config.min_diversification_score:
            warnings.append(f"Low diversification ({div_score:.2f})")
        if portfolio_var > self.config.max_portfolio_var * 0.8:
            warnings.append(f"Portfolio VaR ({portfolio_var:.2%}) approaching limit")
        
        # Check currency concentration
        for currency, exp in currency_exposures.items():
            if exp.exposure_percent > self.config.max_single_currency_exposure * 0.8:
                warnings.append(f"High {currency} exposure ({exp.exposure_percent:.1%})")
        
        # Generate recommendations
        recommendations = []
        if div_score < self.config.target_diversification_score:
            recommendations.append("Consider adding uncorrelated positions")
        if len(self.positions) < 3:
            recommendations.append("Portfolio may benefit from more positions")
        if abs(net_exposure) > gross_exposure * 0.5:
            recommendations.append("Consider balancing long/short exposure")
        
        return PortfolioRiskReport(
            total_positions=len(self.positions),
            gross_exposure=gross_exposure,
            net_exposure=net_exposure,
            diversification_score=div_score,
            portfolio_var=portfolio_var,
            currency_exposures=currency_exposures,
            correlation_matrix=corr_matrix,
            risk_warnings=warnings,
            recommendations=recommendations
        )
    
    def _calculate_portfolio_var(self, confidence: float = 0.95) -> float:
        """
        Calculate portfolio Value at Risk.
        
        Simplified calculation using correlation-adjusted volatility.
        """
        if not self.positions:
            return 0.0
        
        # Assume 1% daily volatility per position (simplified)
        base_vol = 0.01
        
        # Get weights
        total_size = sum(p.size for p in self.positions.values())
        if total_size == 0:
            return 0.0
        
        weights = [p.size / total_size for p in self.positions.values()]
        symbols = [p.symbol for p in self.positions.values()]
        
        # Build covariance matrix
        n = len(symbols)
        cov_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                corr = self.correlation_matrix.get_correlation(symbols[i], symbols[j])
                cov_matrix[i, j] = corr * base_vol * base_vol
        
        # Portfolio variance
        weights_array = np.array(weights)
        portfolio_var = np.sqrt(weights_array @ cov_matrix @ weights_array)
        
        # VaR at confidence level (assuming normal distribution)
        z_score = 1.645 if confidence == 0.95 else 2.326  # 95% or 99%
        var = portfolio_var * z_score
        
        return var
    
    def get_optimal_position_size(self, symbol: str, direction: str,
                                  base_size: float) -> float:
        """
        Get optimal position size considering correlations.
        
        Reduces size if position would increase portfolio risk.
        """
        check = self.check_new_position(symbol, direction, base_size)
        
        if not check.can_add:
            return 0.0
        
        # Reduce size based on correlation
        if check.highest_correlation > 0.5:
            reduction = (check.highest_correlation - 0.5) * 0.5
            base_size *= (1 - reduction)
        
        # Reduce size if diversification would decrease
        if check.diversification_impact < 0:
            base_size *= 0.8
        
        return base_size
    
    def suggest_hedge_positions(self) -> List[Dict[str, Any]]:
        """Suggest positions that would hedge current exposure"""
        suggestions = []
        exposures = self._calculate_currency_exposures()
        
        # Find currencies with high exposure
        for currency, exposure in exposures.items():
            if abs(exposure) > 0.3:
                # Suggest opposite exposure
                direction = 'short' if exposure > 0 else 'long'
                
                # Find pairs with this currency
                hedge_pairs = []
                for pair in ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD']:
                    if currency in pair:
                        hedge_pairs.append(pair)
                
                if hedge_pairs:
                    suggestions.append({
                        'reason': f"Hedge {currency} exposure ({exposure:.1%})",
                        'pairs': hedge_pairs,
                        'direction': direction,
                        'target_reduction': abs(exposure) * 0.5
                    })
        
        return suggestions


# Singleton instance
_correlation_optimizer: Optional[CorrelationOptimizer] = None


def get_correlation_optimizer() -> CorrelationOptimizer:
    """Get singleton correlation optimizer instance"""
    global _correlation_optimizer
    if _correlation_optimizer is None:
        _correlation_optimizer = CorrelationOptimizer()
    return _correlation_optimizer


def can_add_position(symbol: str, direction: str, size: float) -> Tuple[bool, str]:
    """
    Convenience function to check if position can be added.
    
    Returns:
        Tuple of (can_add, reason)
    """
    optimizer = get_correlation_optimizer()
    check = optimizer.check_new_position(symbol, direction, size)
    return check.can_add, check.reason


def get_adjusted_position_size(symbol: str, direction: str, base_size: float) -> float:
    """Get correlation-adjusted position size"""
    optimizer = get_correlation_optimizer()
    return optimizer.get_optimal_position_size(symbol, direction, base_size)


def get_portfolio_diversification() -> float:
    """Get current portfolio diversification score"""
    optimizer = get_correlation_optimizer()
    return optimizer._get_diversification_score()
