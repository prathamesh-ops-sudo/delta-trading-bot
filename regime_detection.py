"""
Market Regime Detection Module
Uses Hidden Markov Models (HMM) and clustering for market state identification
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging
import warnings
import pickle
import os

warnings.filterwarnings('ignore')

try:
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    print("hmmlearn not available - using simplified regime detection")

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

from config import config

logger = logging.getLogger(__name__)


@dataclass
class MarketRegime:
    """Market regime data structure"""
    name: str
    probability: float
    volatility: float
    trend_strength: float
    mean_return: float
    characteristics: Dict
    recommended_strategies: List[str]
    risk_adjustment: float


class RegimeFeatureExtractor:
    """Extract features for regime detection"""
    
    def __init__(self, lookback: int = 20):
        self.lookback = lookback
        self.scaler = StandardScaler()
        self.fitted = False
    
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract regime detection features from OHLCV data"""
        features = pd.DataFrame(index=df.index)
        
        close = df['close']
        high = df['high']
        low = df['low']
        
        # Returns
        features['returns'] = close.pct_change()
        features['log_returns'] = np.log(close / close.shift(1))
        
        # Realized volatility (rolling std of returns)
        features['volatility_5'] = features['returns'].rolling(5).std()
        features['volatility_10'] = features['returns'].rolling(10).std()
        features['volatility_20'] = features['returns'].rolling(20).std()
        
        # Volatility ratio (short-term vs long-term)
        features['vol_ratio'] = features['volatility_5'] / features['volatility_20']
        
        # Range/ATR normalized
        tr = pd.concat([
            high - low,
            abs(high - close.shift(1)),
            abs(low - close.shift(1))
        ], axis=1).max(axis=1)
        features['atr_normalized'] = tr.rolling(14).mean() / close
        
        # Trend strength (using price momentum)
        features['momentum_5'] = close / close.shift(5) - 1
        features['momentum_10'] = close / close.shift(10) - 1
        features['momentum_20'] = close / close.shift(20) - 1
        
        # Trend consistency (how often price moves in same direction)
        features['trend_consistency'] = features['returns'].rolling(10).apply(
            lambda x: abs(np.sum(np.sign(x))) / len(x)
        )
        
        # Mean reversion indicator
        sma_20 = close.rolling(20).mean()
        features['price_deviation'] = (close - sma_20) / sma_20
        
        # Skewness and kurtosis of returns
        features['skewness'] = features['returns'].rolling(20).skew()
        features['kurtosis'] = features['returns'].rolling(20).kurt()
        
        # Volume features (if available)
        if 'volume' in df.columns and df['volume'].sum() > 0:
            volume = df['volume']
            features['volume_ratio'] = volume / volume.rolling(20).mean()
            features['volume_volatility'] = volume.rolling(10).std() / volume.rolling(10).mean()
        
        # ADX-like trend strength (simplified)
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        
        atr = tr.rolling(14).mean()
        plus_di = 100 * plus_dm.rolling(14).mean() / atr
        minus_di = 100 * minus_dm.rolling(14).mean() / atr
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        features['adx'] = dx.rolling(14).mean()
        
        return features.dropna()
    
    def prepare_hmm_features(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare features specifically for HMM"""
        features = self.extract_features(df)
        
        # Select key features for HMM
        hmm_features = features[['returns', 'volatility_10', 'atr_normalized', 'momentum_10']].copy()
        hmm_features = hmm_features.dropna()
        
        # Standardize
        if not self.fitted:
            self.scaler.fit(hmm_features)
            self.fitted = True
        
        scaled_features = self.scaler.transform(hmm_features)
        return scaled_features, hmm_features.index


class HMMRegimeDetector:
    """Hidden Markov Model based regime detection"""
    
    def __init__(self, n_regimes: int = 3, covariance_type: str = 'full'):
        self.n_regimes = n_regimes
        self.covariance_type = covariance_type
        self.model = None
        self.feature_extractor = RegimeFeatureExtractor()
        self.regime_labels = {}
        self.fitted = False
    
    def fit(self, df: pd.DataFrame, max_iter: int = 100) -> bool:
        """Fit HMM model on historical data"""
        if not HMM_AVAILABLE:
            logger.warning("hmmlearn not available, using fallback")
            return self._fit_fallback(df)
        
        try:
            # Extract features
            features, index = self.feature_extractor.prepare_hmm_features(df)
            
            if len(features) < 100:
                logger.warning("Insufficient data for HMM fitting")
                return False
            
            # Fit HMM
            self.model = hmm.GaussianHMM(
                n_components=self.n_regimes,
                covariance_type=self.covariance_type,
                n_iter=max_iter,
                random_state=42
            )
            
            self.model.fit(features)
            
            # Label regimes based on characteristics
            self._label_regimes(features)
            
            self.fitted = True
            logger.info(f"HMM fitted with {self.n_regimes} regimes")
            return True
            
        except Exception as e:
            logger.error(f"Error fitting HMM: {e}")
            return self._fit_fallback(df)
    
    def _fit_fallback(self, df: pd.DataFrame) -> bool:
        """Fallback using GMM if HMM not available"""
        try:
            features, index = self.feature_extractor.prepare_hmm_features(df)
            
            self.model = GaussianMixture(
                n_components=self.n_regimes,
                covariance_type='full',
                random_state=42
            )
            
            self.model.fit(features)
            self._label_regimes_gmm(features)
            
            self.fitted = True
            logger.info(f"GMM fallback fitted with {self.n_regimes} regimes")
            return True
            
        except Exception as e:
            logger.error(f"Error fitting GMM fallback: {e}")
            return False
    
    def _label_regimes(self, features: np.ndarray):
        """Label regimes based on their characteristics"""
        if self.model is None:
            return
        
        # Get regime means
        means = self.model.means_
        
        # Identify regimes by volatility (feature index 1)
        vol_order = np.argsort(means[:, 1])
        
        # Label based on volatility level
        labels = ['low_vol', 'medium_vol', 'high_vol'][:self.n_regimes]
        
        for i, regime_idx in enumerate(vol_order):
            self.regime_labels[regime_idx] = {
                'name': labels[i] if i < len(labels) else f'regime_{regime_idx}',
                'mean_return': means[regime_idx, 0],
                'volatility': means[regime_idx, 1],
                'atr': means[regime_idx, 2] if means.shape[1] > 2 else 0,
                'momentum': means[regime_idx, 3] if means.shape[1] > 3 else 0
            }
    
    def _label_regimes_gmm(self, features: np.ndarray):
        """Label regimes for GMM model"""
        if self.model is None:
            return
        
        means = self.model.means_
        vol_order = np.argsort(means[:, 1])
        
        labels = ['low_vol', 'medium_vol', 'high_vol'][:self.n_regimes]
        
        for i, regime_idx in enumerate(vol_order):
            self.regime_labels[regime_idx] = {
                'name': labels[i] if i < len(labels) else f'regime_{regime_idx}',
                'mean_return': means[regime_idx, 0],
                'volatility': means[regime_idx, 1],
                'atr': means[regime_idx, 2] if means.shape[1] > 2 else 0,
                'momentum': means[regime_idx, 3] if means.shape[1] > 3 else 0
            }
    
    def predict(self, df: pd.DataFrame) -> Tuple[int, np.ndarray, MarketRegime]:
        """Predict current market regime"""
        if not self.fitted or self.model is None:
            return self._predict_fallback(df)
        
        try:
            features, index = self.feature_extractor.prepare_hmm_features(df)
            
            if len(features) == 0:
                return self._predict_fallback(df)
            
            # Get most likely regime
            if HMM_AVAILABLE and hasattr(self.model, 'predict_proba'):
                probs = self.model.predict_proba(features)
                regime = np.argmax(probs[-1])
                regime_probs = probs[-1]
            else:
                # GMM fallback
                probs = self.model.predict_proba(features)
                regime = np.argmax(probs[-1])
                regime_probs = probs[-1]
            
            # Create MarketRegime object
            regime_info = self.regime_labels.get(regime, {})
            market_regime = self._create_regime_object(regime, regime_probs[regime], regime_info)
            
            return regime, regime_probs, market_regime
            
        except Exception as e:
            logger.error(f"Error predicting regime: {e}")
            return self._predict_fallback(df)
    
    def _predict_fallback(self, df: pd.DataFrame) -> Tuple[int, np.ndarray, MarketRegime]:
        """Simple rule-based regime detection fallback"""
        features = self.feature_extractor.extract_features(df)
        
        if features.empty:
            return 1, np.array([0.33, 0.34, 0.33]), self._create_default_regime()
        
        # Get latest values
        vol = features['volatility_10'].iloc[-1] if 'volatility_10' in features else 0.01
        adx = features['adx'].iloc[-1] if 'adx' in features else 25
        momentum = features['momentum_10'].iloc[-1] if 'momentum_10' in features else 0
        
        # Simple rule-based classification
        if vol > 0.02:  # High volatility
            regime = 2
            probs = np.array([0.1, 0.2, 0.7])
        elif adx > 30 and abs(momentum) > 0.02:  # Strong trend
            regime = 1
            probs = np.array([0.2, 0.6, 0.2])
        else:  # Low volatility / ranging
            regime = 0
            probs = np.array([0.6, 0.2, 0.2])
        
        regime_info = {
            'name': ['low_vol', 'trending', 'high_vol'][regime],
            'volatility': vol,
            'momentum': momentum
        }
        
        return regime, probs, self._create_regime_object(regime, probs[regime], regime_info)
    
    def _create_regime_object(self, regime: int, probability: float, 
                              info: Dict) -> MarketRegime:
        """Create MarketRegime object from regime info"""
        name = info.get('name', f'regime_{regime}')
        volatility = info.get('volatility', 0.01)
        momentum = info.get('momentum', 0)
        
        # Determine characteristics and strategies based on regime
        if 'low_vol' in name or regime == 0:
            characteristics = {
                'type': 'ranging',
                'expected_moves': 'small',
                'mean_reversion': 'high'
            }
            strategies = ['mean_reversion', 'range_trading', 'grid']
            risk_adj = 0.8  # Lower risk in ranging markets
        elif 'high_vol' in name or regime == 2:
            characteristics = {
                'type': 'volatile',
                'expected_moves': 'large',
                'unpredictability': 'high'
            }
            strategies = ['breakout', 'momentum', 'reduced_exposure']
            risk_adj = 0.5  # Much lower risk in volatile markets
        else:
            characteristics = {
                'type': 'trending',
                'expected_moves': 'medium',
                'trend_following': 'high'
            }
            strategies = ['trend_following', 'momentum', 'breakout']
            risk_adj = 1.2  # Higher risk in trending markets
        
        return MarketRegime(
            name=name,
            probability=probability,
            volatility=volatility,
            trend_strength=abs(momentum),
            mean_return=info.get('mean_return', 0),
            characteristics=characteristics,
            recommended_strategies=strategies,
            risk_adjustment=risk_adj
        )
    
    def _create_default_regime(self) -> MarketRegime:
        """Create default regime when detection fails"""
        return MarketRegime(
            name='unknown',
            probability=0.5,
            volatility=0.01,
            trend_strength=0.0,
            mean_return=0.0,
            characteristics={'type': 'unknown'},
            recommended_strategies=['conservative'],
            risk_adjustment=0.5
        )
    
    def get_regime_history(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get regime history for entire dataset"""
        if not self.fitted or self.model is None:
            return pd.DataFrame()
        
        try:
            features, index = self.feature_extractor.prepare_hmm_features(df)
            
            if HMM_AVAILABLE and hasattr(self.model, 'predict'):
                regimes = self.model.predict(features)
                probs = self.model.predict_proba(features)
            else:
                regimes = self.model.predict(features)
                probs = self.model.predict_proba(features)
            
            result = pd.DataFrame({
                'regime': regimes,
                'prob_0': probs[:, 0],
                'prob_1': probs[:, 1],
                'prob_2': probs[:, 2] if probs.shape[1] > 2 else 0
            }, index=index)
            
            # Add regime names
            result['regime_name'] = result['regime'].map(
                lambda x: self.regime_labels.get(x, {}).get('name', f'regime_{x}')
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting regime history: {e}")
            return pd.DataFrame()
    
    def save(self, path: str):
        """Save model to disk"""
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'regime_labels': self.regime_labels,
                'scaler': self.feature_extractor.scaler,
                'fitted': self.fitted
            }, f)
        logger.info(f"Regime detector saved to {path}")
    
    def load(self, path: str):
        """Load model from disk"""
        if os.path.exists(path):
            with open(path, 'rb') as f:
                data = pickle.load(f)
                self.model = data['model']
                self.regime_labels = data['regime_labels']
                self.feature_extractor.scaler = data['scaler']
                self.feature_extractor.fitted = True
                self.fitted = data['fitted']
            logger.info(f"Regime detector loaded from {path}")


class ClusteringRegimeDetector:
    """K-Means clustering based regime detection"""
    
    def __init__(self, n_regimes: int = 3):
        self.n_regimes = n_regimes
        self.model = KMeans(n_clusters=n_regimes, random_state=42)
        self.feature_extractor = RegimeFeatureExtractor()
        self.scaler = StandardScaler()
        self.fitted = False
        self.cluster_centers = None
    
    def fit(self, df: pd.DataFrame) -> bool:
        """Fit clustering model"""
        try:
            features = self.feature_extractor.extract_features(df)
            
            # Select features for clustering
            cluster_features = features[['volatility_10', 'momentum_10', 'adx']].dropna()
            
            if len(cluster_features) < 100:
                return False
            
            # Scale and fit
            scaled = self.scaler.fit_transform(cluster_features)
            self.model.fit(scaled)
            self.cluster_centers = self.scaler.inverse_transform(self.model.cluster_centers_)
            
            self.fitted = True
            return True
            
        except Exception as e:
            logger.error(f"Error fitting clustering: {e}")
            return False
    
    def predict(self, df: pd.DataFrame) -> Tuple[int, np.ndarray]:
        """Predict regime using clustering"""
        if not self.fitted:
            return 1, np.array([0.33, 0.34, 0.33])
        
        try:
            features = self.feature_extractor.extract_features(df)
            cluster_features = features[['volatility_10', 'momentum_10', 'adx']].dropna()
            
            if cluster_features.empty:
                return 1, np.array([0.33, 0.34, 0.33])
            
            scaled = self.scaler.transform(cluster_features.iloc[[-1]])
            regime = self.model.predict(scaled)[0]
            
            # Calculate soft probabilities based on distance
            distances = np.linalg.norm(
                self.model.cluster_centers_ - scaled, axis=1
            )
            probs = 1 / (distances + 1e-10)
            probs = probs / probs.sum()
            
            return regime, probs
            
        except Exception as e:
            logger.error(f"Error predicting with clustering: {e}")
            return 1, np.array([0.33, 0.34, 0.33])


class RegimeManager:
    """Main regime management class"""
    
    def __init__(self, n_regimes: int = 3):
        self.hmm_detector = HMMRegimeDetector(n_regimes=n_regimes)
        self.cluster_detector = ClusteringRegimeDetector(n_regimes=n_regimes)
        self.current_regime = None
        self.regime_history = []
        self.strategy_weights = {
            'low_vol': {'mean_reversion': 0.6, 'trend_following': 0.2, 'breakout': 0.2},
            'medium_vol': {'mean_reversion': 0.3, 'trend_following': 0.5, 'breakout': 0.2},
            'high_vol': {'mean_reversion': 0.2, 'trend_following': 0.3, 'breakout': 0.5},
            'trending': {'mean_reversion': 0.1, 'trend_following': 0.7, 'breakout': 0.2},
            'unknown': {'mean_reversion': 0.33, 'trend_following': 0.34, 'breakout': 0.33}
        }
    
    def fit(self, df: pd.DataFrame) -> bool:
        """Fit all regime detectors"""
        hmm_success = self.hmm_detector.fit(df)
        cluster_success = self.cluster_detector.fit(df)
        
        return hmm_success or cluster_success
    
    def detect_regime(self, df: pd.DataFrame) -> MarketRegime:
        """Detect current market regime using ensemble"""
        # Get HMM prediction
        hmm_regime, hmm_probs, hmm_market_regime = self.hmm_detector.predict(df)
        
        # Get clustering prediction
        cluster_regime, cluster_probs = self.cluster_detector.predict(df)
        
        # Ensemble: weight HMM more if fitted
        if self.hmm_detector.fitted:
            ensemble_probs = 0.7 * hmm_probs + 0.3 * cluster_probs
        else:
            ensemble_probs = cluster_probs
        
        final_regime = np.argmax(ensemble_probs)
        
        # Update current regime
        self.current_regime = hmm_market_regime
        self.regime_history.append({
            'timestamp': datetime.now(),
            'regime': final_regime,
            'probabilities': ensemble_probs.tolist(),
            'market_regime': hmm_market_regime
        })
        
        return hmm_market_regime
    
    def get_strategy_weights(self) -> Dict[str, float]:
        """Get recommended strategy weights for current regime"""
        if self.current_regime is None:
            return self.strategy_weights['unknown']
        
        regime_name = self.current_regime.name
        
        # Find matching strategy weights
        for key in self.strategy_weights:
            if key in regime_name:
                return self.strategy_weights[key]
        
        return self.strategy_weights['unknown']
    
    def get_risk_adjustment(self) -> float:
        """Get risk adjustment factor for current regime"""
        if self.current_regime is None:
            return 0.5
        return self.current_regime.risk_adjustment
    
    def should_trade(self) -> Tuple[bool, str]:
        """Determine if trading is advisable in current regime"""
        if self.current_regime is None:
            return True, "No regime detected"
        
        # Don't trade in extremely volatile conditions
        if self.current_regime.volatility > 0.05:
            return False, "Extreme volatility detected"
        
        # Don't trade if regime probability is too low
        if self.current_regime.probability < 0.4:
            return True, "Low regime confidence - trade with caution"
        
        return True, f"Regime: {self.current_regime.name}"
    
    def save(self, directory: str):
        """Save all models"""
        os.makedirs(directory, exist_ok=True)
        self.hmm_detector.save(os.path.join(directory, 'hmm_regime.pkl'))
    
    def load(self, directory: str):
        """Load all models"""
        hmm_path = os.path.join(directory, 'hmm_regime.pkl')
        if os.path.exists(hmm_path):
            self.hmm_detector.load(hmm_path)


# Singleton instance
regime_manager = RegimeManager()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Regime Detection Module...")
    print(f"HMM available: {HMM_AVAILABLE}")
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    
    # Create synthetic price data with regime changes
    prices = [100]
    for i in range(n_samples - 1):
        if i < 300:  # Low volatility regime
            change = np.random.normal(0.0001, 0.005)
        elif i < 600:  # Trending regime
            change = np.random.normal(0.001, 0.01)
        else:  # High volatility regime
            change = np.random.normal(0, 0.02)
        prices.append(prices[-1] * (1 + change))
    
    df = pd.DataFrame({
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
        'close': prices,
        'volume': np.random.randint(1000, 10000, n_samples)
    })
    
    # Test feature extraction
    extractor = RegimeFeatureExtractor()
    features = extractor.extract_features(df)
    print(f"\nExtracted {len(features.columns)} features")
    print(f"Features: {list(features.columns)}")
    
    # Test HMM detector
    print("\nTesting HMM Regime Detector...")
    hmm_detector = HMMRegimeDetector(n_regimes=3)
    if hmm_detector.fit(df):
        regime, probs, market_regime = hmm_detector.predict(df)
        print(f"Current regime: {regime} ({market_regime.name})")
        print(f"Probabilities: {probs}")
        print(f"Recommended strategies: {market_regime.recommended_strategies}")
        print(f"Risk adjustment: {market_regime.risk_adjustment}")
    
    # Test regime manager
    print("\nTesting Regime Manager...")
    manager = RegimeManager()
    manager.fit(df)
    
    current_regime = manager.detect_regime(df)
    print(f"Detected regime: {current_regime.name}")
    print(f"Strategy weights: {manager.get_strategy_weights()}")
    print(f"Should trade: {manager.should_trade()}")
    
    print("\nRegime Detection Module test complete!")
