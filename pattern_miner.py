"""
Pattern Miner Module - Human-Like Time-Based Pattern Learning
Detects recurring patterns like "9am market dip" and learns from them
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import logging
import json
import os
from scipy import stats

from config import config

logger = logging.getLogger(__name__)


@dataclass
class TimePattern:
    """A discovered time-based pattern"""
    pattern_id: str
    symbol: str
    hour: int
    minute_bucket: int
    day_of_week: Optional[int]
    pattern_type: str
    avg_move_pips: float
    std_move_pips: float
    confidence: float
    z_score: float
    sample_size: int
    win_rate: float
    discovered_at: datetime
    last_updated: datetime
    times_traded: int = 0
    times_profitable: int = 0
    is_active: bool = True
    description: str = ""


@dataclass
class SessionPattern:
    """Pattern specific to trading sessions"""
    session: str
    symbol: str
    avg_volatility: float
    avg_direction: float
    best_entry_hour: int
    worst_entry_hour: int
    confidence: float
    sample_size: int


class PatternMiner:
    """
    Mines time-based patterns from historical data like a veteran trader.
    Learns patterns like:
    - "Every day at 9am London open, EURUSD drops 20 pips then reverses"
    - "Asian session has low volatility, avoid trading"
    - "NFP Fridays have high volatility at 8:30am EST"
    """
    
    def __init__(self, data_path: str = None):
        self.data_path = data_path or "./data/patterns.json"
        self.patterns: Dict[str, TimePattern] = {}
        self.session_patterns: Dict[str, SessionPattern] = {}
        self.hourly_stats: Dict[str, Dict[int, Dict]] = defaultdict(lambda: defaultdict(dict))
        self.minute_stats: Dict[str, Dict[str, Dict]] = defaultdict(lambda: defaultdict(dict))
        self.rolling_window_days = 20
        self.min_samples = 10
        self.confidence_threshold = 0.6
        self.z_score_threshold = 1.5
        self._load_patterns()
        
        self.sessions = {
            'asian': (0, 8),
            'london': (8, 16),
            'new_york': (13, 21),
            'overlap': (13, 16)
        }
    
    def _load_patterns(self):
        """Load discovered patterns from disk"""
        if os.path.exists(self.data_path):
            try:
                with open(self.data_path, 'r') as f:
                    data = json.load(f)
                    for p in data.get('patterns', []):
                        p['discovered_at'] = datetime.fromisoformat(p['discovered_at'])
                        p['last_updated'] = datetime.fromisoformat(p['last_updated'])
                        pattern = TimePattern(**p)
                        self.patterns[pattern.pattern_id] = pattern
                    logger.info(f"Loaded {len(self.patterns)} patterns")
            except Exception as e:
                logger.warning(f"Could not load patterns: {e}")
    
    def _save_patterns(self):
        """Save patterns to disk"""
        os.makedirs(os.path.dirname(self.data_path), exist_ok=True)
        try:
            data = {
                'patterns': [],
                'last_updated': datetime.now().isoformat()
            }
            for p in self.patterns.values():
                p_dict = {
                    'pattern_id': p.pattern_id,
                    'symbol': p.symbol,
                    'hour': p.hour,
                    'minute_bucket': p.minute_bucket,
                    'day_of_week': p.day_of_week,
                    'pattern_type': p.pattern_type,
                    'avg_move_pips': p.avg_move_pips,
                    'std_move_pips': p.std_move_pips,
                    'confidence': p.confidence,
                    'z_score': p.z_score,
                    'sample_size': p.sample_size,
                    'win_rate': p.win_rate,
                    'discovered_at': p.discovered_at.isoformat(),
                    'last_updated': p.last_updated.isoformat(),
                    'times_traded': p.times_traded,
                    'times_profitable': p.times_profitable,
                    'is_active': p.is_active,
                    'description': p.description
                }
                data['patterns'].append(p_dict)
            with open(self.data_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save patterns: {e}")
    
    def analyze_historical_data(self, df: pd.DataFrame, symbol: str) -> List[TimePattern]:
        """
        Analyze historical data to discover time-based patterns.
        Like a veteran trader noticing "market always dips at 9am"
        """
        if df.empty or len(df) < 100:
            return []
        
        discovered_patterns = []
        
        df = df.copy()
        if 'datetime' not in df.columns and df.index.name != 'datetime':
            if isinstance(df.index, pd.DatetimeIndex):
                df['datetime'] = df.index
            else:
                return []
        
        if 'datetime' in df.columns:
            df['hour'] = df['datetime'].dt.hour
            df['minute'] = df['datetime'].dt.minute
            df['day_of_week'] = df['datetime'].dt.dayofweek
            df['minute_bucket'] = (df['minute'] // 15) * 15
        
        if 'close' in df.columns and 'open' in df.columns:
            pip_multiplier = 100 if 'JPY' in symbol else 10000
            df['move_pips'] = (df['close'] - df['open']) * pip_multiplier
        else:
            return []
        
        hourly_patterns = self._analyze_hourly_patterns(df, symbol)
        discovered_patterns.extend(hourly_patterns)
        
        minute_patterns = self._analyze_minute_patterns(df, symbol)
        discovered_patterns.extend(minute_patterns)
        
        dow_patterns = self._analyze_day_of_week_patterns(df, symbol)
        discovered_patterns.extend(dow_patterns)
        
        reversal_patterns = self._analyze_reversal_patterns(df, symbol)
        discovered_patterns.extend(reversal_patterns)
        
        for pattern in discovered_patterns:
            self.patterns[pattern.pattern_id] = pattern
        
        self._save_patterns()
        
        logger.info(f"Discovered {len(discovered_patterns)} new patterns for {symbol}")
        return discovered_patterns
    
    def _analyze_hourly_patterns(self, df: pd.DataFrame, symbol: str) -> List[TimePattern]:
        """Analyze patterns by hour of day"""
        patterns = []
        
        hourly_stats = df.groupby('hour')['move_pips'].agg(['mean', 'std', 'count'])
        overall_mean = df['move_pips'].mean()
        overall_std = df['move_pips'].std()
        
        for hour, row in hourly_stats.iterrows():
            if row['count'] < self.min_samples:
                continue
            
            if overall_std > 0:
                z_score = (row['mean'] - overall_mean) / (overall_std / np.sqrt(row['count']))
            else:
                z_score = 0
            
            if abs(z_score) >= self.z_score_threshold:
                pattern_type = 'bullish_hour' if row['mean'] > 0 else 'bearish_hour'
                
                hour_trades = df[df['hour'] == hour]['move_pips']
                if row['mean'] > 0:
                    win_rate = (hour_trades > 0).mean()
                else:
                    win_rate = (hour_trades < 0).mean()
                
                confidence = min(0.95, 0.5 + abs(z_score) * 0.1)
                
                if abs(row['mean']) > 5:
                    pattern = TimePattern(
                        pattern_id=f"{symbol}_hour_{hour}",
                        symbol=symbol,
                        hour=int(hour),
                        minute_bucket=0,
                        day_of_week=None,
                        pattern_type=pattern_type,
                        avg_move_pips=float(row['mean']),
                        std_move_pips=float(row['std']),
                        confidence=confidence,
                        z_score=float(z_score),
                        sample_size=int(row['count']),
                        win_rate=float(win_rate),
                        discovered_at=datetime.now(),
                        last_updated=datetime.now(),
                        description=f"At {hour}:00 UTC, {symbol} tends to move {row['mean']:.1f} pips on average"
                    )
                    patterns.append(pattern)
        
        return patterns
    
    def _analyze_minute_patterns(self, df: pd.DataFrame, symbol: str) -> List[TimePattern]:
        """Analyze patterns by hour:minute bucket (e.g., 9:00, 9:15, 9:30, 9:45)"""
        patterns = []
        
        df['time_bucket'] = df['hour'].astype(str) + ':' + df['minute_bucket'].astype(str).str.zfill(2)
        
        bucket_stats = df.groupby('time_bucket')['move_pips'].agg(['mean', 'std', 'count'])
        overall_mean = df['move_pips'].mean()
        overall_std = df['move_pips'].std()
        
        for bucket, row in bucket_stats.iterrows():
            if row['count'] < self.min_samples:
                continue
            
            if overall_std > 0:
                z_score = (row['mean'] - overall_mean) / (overall_std / np.sqrt(row['count']))
            else:
                z_score = 0
            
            if abs(z_score) >= self.z_score_threshold * 1.2:
                hour, minute = map(int, bucket.split(':'))
                pattern_type = 'bullish_time' if row['mean'] > 0 else 'bearish_time'
                
                bucket_trades = df[df['time_bucket'] == bucket]['move_pips']
                if row['mean'] > 0:
                    win_rate = (bucket_trades > 0).mean()
                else:
                    win_rate = (bucket_trades < 0).mean()
                
                confidence = min(0.95, 0.5 + abs(z_score) * 0.08)
                
                if abs(row['mean']) > 8:
                    pattern = TimePattern(
                        pattern_id=f"{symbol}_time_{hour}_{minute}",
                        symbol=symbol,
                        hour=hour,
                        minute_bucket=minute,
                        day_of_week=None,
                        pattern_type=pattern_type,
                        avg_move_pips=float(row['mean']),
                        std_move_pips=float(row['std']),
                        confidence=confidence,
                        z_score=float(z_score),
                        sample_size=int(row['count']),
                        win_rate=float(win_rate),
                        discovered_at=datetime.now(),
                        last_updated=datetime.now(),
                        description=f"At {hour}:{minute:02d} UTC, {symbol} tends to move {row['mean']:.1f} pips"
                    )
                    patterns.append(pattern)
        
        return patterns
    
    def _analyze_day_of_week_patterns(self, df: pd.DataFrame, symbol: str) -> List[TimePattern]:
        """Analyze patterns by day of week combined with hour"""
        patterns = []
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        for dow in range(5):
            dow_df = df[df['day_of_week'] == dow]
            if len(dow_df) < 50:
                continue
            
            hourly_stats = dow_df.groupby('hour')['move_pips'].agg(['mean', 'std', 'count'])
            overall_mean = dow_df['move_pips'].mean()
            overall_std = dow_df['move_pips'].std()
            
            for hour, row in hourly_stats.iterrows():
                if row['count'] < 5:
                    continue
                
                if overall_std > 0:
                    z_score = (row['mean'] - overall_mean) / (overall_std / np.sqrt(row['count']))
                else:
                    z_score = 0
                
                if abs(z_score) >= self.z_score_threshold * 1.5:
                    pattern_type = f'bullish_{day_names[dow].lower()}' if row['mean'] > 0 else f'bearish_{day_names[dow].lower()}'
                    
                    hour_trades = dow_df[dow_df['hour'] == hour]['move_pips']
                    if row['mean'] > 0:
                        win_rate = (hour_trades > 0).mean()
                    else:
                        win_rate = (hour_trades < 0).mean()
                    
                    confidence = min(0.95, 0.5 + abs(z_score) * 0.07)
                    
                    if abs(row['mean']) > 10:
                        pattern = TimePattern(
                            pattern_id=f"{symbol}_dow_{dow}_hour_{hour}",
                            symbol=symbol,
                            hour=int(hour),
                            minute_bucket=0,
                            day_of_week=dow,
                            pattern_type=pattern_type,
                            avg_move_pips=float(row['mean']),
                            std_move_pips=float(row['std']),
                            confidence=confidence,
                            z_score=float(z_score),
                            sample_size=int(row['count']),
                            win_rate=float(win_rate),
                            discovered_at=datetime.now(),
                            last_updated=datetime.now(),
                            description=f"On {day_names[dow]}s at {hour}:00 UTC, {symbol} tends to move {row['mean']:.1f} pips"
                        )
                        patterns.append(pattern)
        
        return patterns
    
    def _analyze_reversal_patterns(self, df: pd.DataFrame, symbol: str) -> List[TimePattern]:
        """Analyze mean-reversion patterns (e.g., dip then recover)"""
        patterns = []
        
        if 'high' not in df.columns or 'low' not in df.columns:
            return patterns
        
        pip_multiplier = 100 if 'JPY' in symbol else 10000
        df['range_pips'] = (df['high'] - df['low']) * pip_multiplier
        df['body_pips'] = abs(df['close'] - df['open']) * pip_multiplier
        df['wick_ratio'] = 1 - (df['body_pips'] / df['range_pips'].replace(0, 1))
        
        for hour in range(24):
            hour_df = df[df['hour'] == hour]
            if len(hour_df) < self.min_samples:
                continue
            
            high_wick_ratio = (hour_df['wick_ratio'] > 0.6).mean()
            
            if high_wick_ratio > 0.5:
                next_hour_df = df[df['hour'] == (hour + 1) % 24]
                if len(next_hour_df) > 0:
                    reversal_rate = 0
                    for idx in hour_df.index:
                        if idx + 1 in df.index:
                            current_move = df.loc[idx, 'move_pips']
                            next_move = df.loc[idx + 1, 'move_pips']
                            if current_move * next_move < 0:
                                reversal_rate += 1
                    
                    if len(hour_df) > 0:
                        reversal_rate /= len(hour_df)
                    
                    if reversal_rate > 0.55:
                        pattern = TimePattern(
                            pattern_id=f"{symbol}_reversal_hour_{hour}",
                            symbol=symbol,
                            hour=hour,
                            minute_bucket=0,
                            day_of_week=None,
                            pattern_type='reversal',
                            avg_move_pips=float(hour_df['move_pips'].mean()),
                            std_move_pips=float(hour_df['move_pips'].std()),
                            confidence=min(0.9, reversal_rate),
                            z_score=0,
                            sample_size=len(hour_df),
                            win_rate=reversal_rate,
                            discovered_at=datetime.now(),
                            last_updated=datetime.now(),
                            description=f"At {hour}:00 UTC, {symbol} often reverses in the next hour ({reversal_rate:.0%} of the time)"
                        )
                        patterns.append(pattern)
        
        return patterns
    
    def get_pattern_signal(self, symbol: str, current_time: datetime) -> Dict[str, Any]:
        """
        Get trading signal based on discovered patterns for current time.
        Returns bias, confidence, and relevant patterns.
        """
        hour = current_time.hour
        minute_bucket = (current_time.minute // 15) * 15
        day_of_week = current_time.weekday()
        
        relevant_patterns = []
        total_bias = 0
        total_weight = 0
        
        for pattern in self.patterns.values():
            if pattern.symbol != symbol or not pattern.is_active:
                continue
            
            if pattern.hour == hour:
                if pattern.minute_bucket == 0 or pattern.minute_bucket == minute_bucket:
                    if pattern.day_of_week is None or pattern.day_of_week == day_of_week:
                        relevant_patterns.append(pattern)
                        
                        weight = pattern.confidence * pattern.sample_size / 100
                        if pattern.pattern_type in ['bullish_hour', 'bullish_time', 'bullish_monday', 
                                                     'bullish_tuesday', 'bullish_wednesday', 
                                                     'bullish_thursday', 'bullish_friday']:
                            total_bias += weight
                        elif pattern.pattern_type in ['bearish_hour', 'bearish_time', 'bearish_monday',
                                                       'bearish_tuesday', 'bearish_wednesday',
                                                       'bearish_thursday', 'bearish_friday']:
                            total_bias -= weight
                        
                        total_weight += weight
        
        if total_weight > 0:
            normalized_bias = total_bias / total_weight
        else:
            normalized_bias = 0
        
        avg_confidence = np.mean([p.confidence for p in relevant_patterns]) if relevant_patterns else 0
        
        return {
            'bias': normalized_bias,
            'confidence': avg_confidence,
            'patterns': relevant_patterns,
            'pattern_count': len(relevant_patterns),
            'expected_move': sum(p.avg_move_pips * p.confidence for p in relevant_patterns) / len(relevant_patterns) if relevant_patterns else 0,
            'descriptions': [p.description for p in relevant_patterns]
        }
    
    def update_pattern_performance(self, pattern_id: str, was_profitable: bool):
        """Update pattern performance after a trade"""
        if pattern_id in self.patterns:
            pattern = self.patterns[pattern_id]
            pattern.times_traded += 1
            if was_profitable:
                pattern.times_profitable += 1
            
            if pattern.times_traded > 10:
                actual_win_rate = pattern.times_profitable / pattern.times_traded
                pattern.confidence = 0.7 * pattern.confidence + 0.3 * actual_win_rate
            
            if pattern.times_traded > 20 and pattern.confidence < 0.4:
                pattern.is_active = False
                logger.info(f"Pattern deactivated due to poor performance: {pattern_id}")
            
            pattern.last_updated = datetime.now()
            self._save_patterns()
    
    def get_session_bias(self, symbol: str, current_time: datetime) -> Dict[str, Any]:
        """Get trading bias based on current session"""
        hour = current_time.hour
        
        current_session = None
        for session, (start, end) in self.sessions.items():
            if start <= hour < end:
                current_session = session
                break
        
        if current_session is None:
            current_session = 'asian'
        
        session_patterns = [p for p in self.patterns.values() 
                          if p.symbol == symbol and p.is_active and 
                          self.sessions.get(current_session, (0, 24))[0] <= p.hour < self.sessions.get(current_session, (0, 24))[1]]
        
        if session_patterns:
            avg_move = np.mean([p.avg_move_pips for p in session_patterns])
            avg_confidence = np.mean([p.confidence for p in session_patterns])
        else:
            avg_move = 0
            avg_confidence = 0.5
        
        return {
            'session': current_session,
            'avg_expected_move': avg_move,
            'confidence': avg_confidence,
            'pattern_count': len(session_patterns),
            'is_overlap': current_session == 'overlap'
        }
    
    def get_all_active_patterns(self, symbol: str = None) -> List[TimePattern]:
        """Get all active patterns, optionally filtered by symbol"""
        patterns = [p for p in self.patterns.values() if p.is_active]
        if symbol:
            patterns = [p for p in patterns if p.symbol == symbol]
        return sorted(patterns, key=lambda x: x.confidence, reverse=True)
    
    def generate_pattern_report(self) -> str:
        """Generate a human-readable report of discovered patterns"""
        report = ["=" * 60]
        report.append("PATTERN MINER REPORT - Discovered Time-Based Patterns")
        report.append("=" * 60)
        report.append("")
        
        active_patterns = [p for p in self.patterns.values() if p.is_active]
        
        if not active_patterns:
            report.append("No patterns discovered yet. Need more historical data.")
            return "\n".join(report)
        
        by_symbol = defaultdict(list)
        for p in active_patterns:
            by_symbol[p.symbol].append(p)
        
        for symbol, patterns in by_symbol.items():
            report.append(f"\n{symbol}:")
            report.append("-" * 40)
            
            patterns = sorted(patterns, key=lambda x: x.confidence, reverse=True)
            
            for p in patterns[:10]:
                report.append(f"  [{p.confidence:.0%}] {p.description}")
                report.append(f"       Win Rate: {p.win_rate:.0%} | Samples: {p.sample_size}")
        
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)


pattern_miner = PatternMiner()
