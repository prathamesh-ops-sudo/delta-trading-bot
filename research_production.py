#!/usr/bin/env python3
"""
Tier 3: Research/Production Separation
Institutional-grade research and production workflow like BlackRock's Aladdin

Features:
1. Shadow Mode - Run new strategies in parallel without real execution
2. Experiment Tracking - Track strategy experiments with versioning and metrics
3. Immutable Datasets - Snapshot market data for reproducible backtests
"""

import logging
import json
import sqlite3
import hashlib
import threading
import os
import pickle
import gzip
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import uuid

logger = logging.getLogger(__name__)


class ExperimentStatus(Enum):
    """Status of an experiment"""
    DRAFT = "draft"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PROMOTED = "promoted"  # Promoted to production
    ARCHIVED = "archived"


class ShadowTradeStatus(Enum):
    """Status of a shadow trade"""
    OPEN = "open"
    CLOSED = "closed"
    EXPIRED = "expired"


@dataclass
class ShadowTrade:
    """A simulated trade in shadow mode"""
    trade_id: str
    experiment_id: str
    symbol: str
    direction: str  # BUY or SELL
    entry_price: float
    entry_time: datetime
    stop_loss: float
    take_profit: float
    position_size: float
    strategy: str
    confidence: float
    status: ShadowTradeStatus = ShadowTradeStatus.OPEN
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    exit_reason: Optional[str] = None
    pnl: float = 0.0
    pnl_pips: float = 0.0
    max_favorable_excursion: float = 0.0  # Best unrealized P&L
    max_adverse_excursion: float = 0.0  # Worst unrealized P&L
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentConfig:
    """Configuration for an experiment"""
    name: str
    description: str
    strategy_name: str
    strategy_params: Dict[str, Any]
    symbols: List[str]
    start_date: datetime
    end_date: Optional[datetime] = None
    initial_balance: float = 10000.0
    risk_per_trade_pct: float = 0.01
    max_positions: int = 5
    tags: List[str] = field(default_factory=list)


@dataclass
class ExperimentMetrics:
    """Metrics for an experiment"""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    total_pnl_pips: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    avg_trade_duration_hours: float = 0.0
    expectancy: float = 0.0
    recovery_factor: float = 0.0


@dataclass
class Experiment:
    """An experiment tracking a strategy variation"""
    experiment_id: str
    config: ExperimentConfig
    status: ExperimentStatus
    created_at: datetime
    updated_at: datetime
    metrics: ExperimentMetrics = field(default_factory=ExperimentMetrics)
    shadow_trades: List[ShadowTrade] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    version: int = 1
    parent_experiment_id: Optional[str] = None  # For experiment lineage


@dataclass
class DatasetSnapshot:
    """An immutable snapshot of market data"""
    snapshot_id: str
    name: str
    description: str
    symbols: List[str]
    timeframe: str
    start_date: datetime
    end_date: datetime
    created_at: datetime
    data_hash: str  # SHA256 hash of the data for integrity
    row_count: int
    file_path: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class ShadowModeEngine:
    """Engine for running strategies in shadow mode"""
    
    def __init__(self, db_path: str = "shadow_mode.db"):
        self.db_path = db_path
        self._active_trades: Dict[str, ShadowTrade] = {}
        self._lock = threading.Lock()
        self._init_db()
    
    def _init_db(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS shadow_trades (
                trade_id TEXT PRIMARY KEY,
                experiment_id TEXT,
                symbol TEXT,
                direction TEXT,
                entry_price REAL,
                entry_time TEXT,
                stop_loss REAL,
                take_profit REAL,
                position_size REAL,
                strategy TEXT,
                confidence REAL,
                status TEXT,
                exit_price REAL,
                exit_time TEXT,
                exit_reason TEXT,
                pnl REAL,
                pnl_pips REAL,
                max_favorable_excursion REAL,
                max_adverse_excursion REAL,
                metadata TEXT
            )
        ''')
        conn.commit()
        conn.close()
    
    def open_shadow_trade(self, experiment_id: str, symbol: str, direction: str,
                          entry_price: float, stop_loss: float, take_profit: float,
                          position_size: float, strategy: str, confidence: float,
                          metadata: Dict = None) -> ShadowTrade:
        """Open a new shadow trade"""
        trade = ShadowTrade(
            trade_id=str(uuid.uuid4()),
            experiment_id=experiment_id,
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            entry_time=datetime.now(),
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=position_size,
            strategy=strategy,
            confidence=confidence,
            metadata=metadata or {}
        )
        
        with self._lock:
            self._active_trades[trade.trade_id] = trade
            self._save_trade(trade)
        
        logger.info(f"[SHADOW] Opened trade {trade.trade_id}: {direction} {symbol} @ {entry_price}")
        return trade
    
    def update_shadow_trades(self, current_prices: Dict[str, float]):
        """Update all active shadow trades with current prices"""
        with self._lock:
            for trade_id, trade in list(self._active_trades.items()):
                if trade.status != ShadowTradeStatus.OPEN:
                    continue
                
                current_price = current_prices.get(trade.symbol)
                if current_price is None:
                    continue
                
                # Calculate unrealized P&L
                pip_value = 0.01 if 'JPY' in trade.symbol else 0.0001
                if trade.direction == 'BUY':
                    pnl_pips = (current_price - trade.entry_price) / pip_value
                else:
                    pnl_pips = (trade.entry_price - current_price) / pip_value
                
                unrealized_pnl = pnl_pips * pip_value * trade.position_size * 100000
                
                # Update excursions
                if unrealized_pnl > trade.max_favorable_excursion:
                    trade.max_favorable_excursion = unrealized_pnl
                if unrealized_pnl < trade.max_adverse_excursion:
                    trade.max_adverse_excursion = unrealized_pnl
                
                # Check for stop loss or take profit
                if trade.direction == 'BUY':
                    if current_price <= trade.stop_loss:
                        self._close_trade(trade, current_price, "stop_loss")
                    elif current_price >= trade.take_profit:
                        self._close_trade(trade, current_price, "take_profit")
                else:
                    if current_price >= trade.stop_loss:
                        self._close_trade(trade, current_price, "stop_loss")
                    elif current_price <= trade.take_profit:
                        self._close_trade(trade, current_price, "take_profit")
    
    def _close_trade(self, trade: ShadowTrade, exit_price: float, reason: str):
        """Close a shadow trade"""
        trade.status = ShadowTradeStatus.CLOSED
        trade.exit_price = exit_price
        trade.exit_time = datetime.now()
        trade.exit_reason = reason
        
        # Calculate final P&L
        pip_value = 0.01 if 'JPY' in trade.symbol else 0.0001
        if trade.direction == 'BUY':
            trade.pnl_pips = (exit_price - trade.entry_price) / pip_value
        else:
            trade.pnl_pips = (trade.entry_price - exit_price) / pip_value
        
        trade.pnl = trade.pnl_pips * pip_value * trade.position_size * 100000
        
        self._save_trade(trade)
        del self._active_trades[trade.trade_id]
        
        logger.info(f"[SHADOW] Closed trade {trade.trade_id}: {reason}, P&L: {trade.pnl:.2f} ({trade.pnl_pips:.1f} pips)")
    
    def close_trade_manually(self, trade_id: str, exit_price: float, reason: str = "manual"):
        """Manually close a shadow trade"""
        with self._lock:
            if trade_id in self._active_trades:
                self._close_trade(self._active_trades[trade_id], exit_price, reason)
    
    def _save_trade(self, trade: ShadowTrade):
        """Save trade to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO shadow_trades 
            (trade_id, experiment_id, symbol, direction, entry_price, entry_time,
             stop_loss, take_profit, position_size, strategy, confidence, status,
             exit_price, exit_time, exit_reason, pnl, pnl_pips,
             max_favorable_excursion, max_adverse_excursion, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            trade.trade_id, trade.experiment_id, trade.symbol, trade.direction,
            trade.entry_price, trade.entry_time.isoformat(),
            trade.stop_loss, trade.take_profit, trade.position_size,
            trade.strategy, trade.confidence, trade.status.value,
            trade.exit_price, trade.exit_time.isoformat() if trade.exit_time else None,
            trade.exit_reason, trade.pnl, trade.pnl_pips,
            trade.max_favorable_excursion, trade.max_adverse_excursion,
            json.dumps(trade.metadata)
        ))
        conn.commit()
        conn.close()
    
    def get_active_trades(self, experiment_id: str = None) -> List[ShadowTrade]:
        """Get all active shadow trades"""
        with self._lock:
            if experiment_id:
                return [t for t in self._active_trades.values() if t.experiment_id == experiment_id]
            return list(self._active_trades.values())
    
    def get_trade_history(self, experiment_id: str = None, limit: int = 100) -> List[ShadowTrade]:
        """Get shadow trade history from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if experiment_id:
            cursor.execute('''
                SELECT * FROM shadow_trades WHERE experiment_id = ? 
                ORDER BY entry_time DESC LIMIT ?
            ''', (experiment_id, limit))
        else:
            cursor.execute('''
                SELECT * FROM shadow_trades ORDER BY entry_time DESC LIMIT ?
            ''', (limit,))
        
        trades = []
        for row in cursor.fetchall():
            trade = ShadowTrade(
                trade_id=row[0],
                experiment_id=row[1],
                symbol=row[2],
                direction=row[3],
                entry_price=row[4],
                entry_time=datetime.fromisoformat(row[5]),
                stop_loss=row[6],
                take_profit=row[7],
                position_size=row[8],
                strategy=row[9],
                confidence=row[10],
                status=ShadowTradeStatus(row[11]),
                exit_price=row[12],
                exit_time=datetime.fromisoformat(row[13]) if row[13] else None,
                exit_reason=row[14],
                pnl=row[15] or 0,
                pnl_pips=row[16] or 0,
                max_favorable_excursion=row[17] or 0,
                max_adverse_excursion=row[18] or 0,
                metadata=json.loads(row[19]) if row[19] else {}
            )
            trades.append(trade)
        
        conn.close()
        return trades


class ExperimentTracker:
    """Track and manage strategy experiments"""
    
    def __init__(self, db_path: str = "experiments.db"):
        self.db_path = db_path
        self._experiments: Dict[str, Experiment] = {}
        self._lock = threading.Lock()
        self._init_db()
        self._load_experiments()
    
    def _init_db(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS experiments (
                experiment_id TEXT PRIMARY KEY,
                config TEXT,
                status TEXT,
                created_at TEXT,
                updated_at TEXT,
                metrics TEXT,
                notes TEXT,
                version INTEGER,
                parent_experiment_id TEXT
            )
        ''')
        conn.commit()
        conn.close()
    
    def _load_experiments(self):
        """Load experiments from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM experiments')
        
        for row in cursor.fetchall():
            try:
                config_data = json.loads(row[1])
                config = ExperimentConfig(
                    name=config_data['name'],
                    description=config_data['description'],
                    strategy_name=config_data['strategy_name'],
                    strategy_params=config_data['strategy_params'],
                    symbols=config_data['symbols'],
                    start_date=datetime.fromisoformat(config_data['start_date']),
                    end_date=datetime.fromisoformat(config_data['end_date']) if config_data.get('end_date') else None,
                    initial_balance=config_data.get('initial_balance', 10000),
                    risk_per_trade_pct=config_data.get('risk_per_trade_pct', 0.01),
                    max_positions=config_data.get('max_positions', 5),
                    tags=config_data.get('tags', [])
                )
                
                metrics_data = json.loads(row[5]) if row[5] else {}
                metrics = ExperimentMetrics(**metrics_data)
                
                experiment = Experiment(
                    experiment_id=row[0],
                    config=config,
                    status=ExperimentStatus(row[2]),
                    created_at=datetime.fromisoformat(row[3]),
                    updated_at=datetime.fromisoformat(row[4]),
                    metrics=metrics,
                    notes=json.loads(row[6]) if row[6] else [],
                    version=row[7],
                    parent_experiment_id=row[8]
                )
                self._experiments[experiment.experiment_id] = experiment
            except Exception as e:
                logger.warning(f"Error loading experiment {row[0]}: {e}")
        
        conn.close()
        logger.info(f"Loaded {len(self._experiments)} experiments from database")
    
    def create_experiment(self, config: ExperimentConfig, 
                          parent_id: str = None) -> Experiment:
        """Create a new experiment"""
        experiment = Experiment(
            experiment_id=str(uuid.uuid4()),
            config=config,
            status=ExperimentStatus.DRAFT,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            parent_experiment_id=parent_id
        )
        
        with self._lock:
            self._experiments[experiment.experiment_id] = experiment
            self._save_experiment(experiment)
        
        logger.info(f"Created experiment {experiment.experiment_id}: {config.name}")
        return experiment
    
    def start_experiment(self, experiment_id: str) -> bool:
        """Start an experiment"""
        with self._lock:
            if experiment_id not in self._experiments:
                return False
            
            experiment = self._experiments[experiment_id]
            if experiment.status != ExperimentStatus.DRAFT:
                return False
            
            experiment.status = ExperimentStatus.RUNNING
            experiment.updated_at = datetime.now()
            self._save_experiment(experiment)
        
        logger.info(f"Started experiment {experiment_id}")
        return True
    
    def complete_experiment(self, experiment_id: str, metrics: ExperimentMetrics) -> bool:
        """Mark an experiment as completed with final metrics"""
        with self._lock:
            if experiment_id not in self._experiments:
                return False
            
            experiment = self._experiments[experiment_id]
            experiment.status = ExperimentStatus.COMPLETED
            experiment.metrics = metrics
            experiment.config.end_date = datetime.now()
            experiment.updated_at = datetime.now()
            self._save_experiment(experiment)
        
        logger.info(f"Completed experiment {experiment_id}: {metrics.total_trades} trades, "
                   f"Win rate: {metrics.win_rate:.1%}, P&L: ${metrics.total_pnl:.2f}")
        return True
    
    def promote_to_production(self, experiment_id: str) -> bool:
        """Promote an experiment to production"""
        with self._lock:
            if experiment_id not in self._experiments:
                return False
            
            experiment = self._experiments[experiment_id]
            if experiment.status != ExperimentStatus.COMPLETED:
                logger.warning(f"Cannot promote experiment {experiment_id}: not completed")
                return False
            
            # Check if metrics meet promotion criteria
            if experiment.metrics.win_rate < 0.45:
                logger.warning(f"Cannot promote experiment {experiment_id}: win rate too low")
                return False
            
            if experiment.metrics.profit_factor < 1.0:
                logger.warning(f"Cannot promote experiment {experiment_id}: profit factor < 1.0")
                return False
            
            experiment.status = ExperimentStatus.PROMOTED
            experiment.updated_at = datetime.now()
            self._save_experiment(experiment)
        
        logger.info(f"Promoted experiment {experiment_id} to production")
        return True
    
    def add_note(self, experiment_id: str, note: str):
        """Add a note to an experiment"""
        with self._lock:
            if experiment_id in self._experiments:
                self._experiments[experiment_id].notes.append(
                    f"[{datetime.now().isoformat()}] {note}"
                )
                self._experiments[experiment_id].updated_at = datetime.now()
                self._save_experiment(self._experiments[experiment_id])
    
    def _save_experiment(self, experiment: Experiment):
        """Save experiment to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        config_dict = {
            'name': experiment.config.name,
            'description': experiment.config.description,
            'strategy_name': experiment.config.strategy_name,
            'strategy_params': experiment.config.strategy_params,
            'symbols': experiment.config.symbols,
            'start_date': experiment.config.start_date.isoformat(),
            'end_date': experiment.config.end_date.isoformat() if experiment.config.end_date else None,
            'initial_balance': experiment.config.initial_balance,
            'risk_per_trade_pct': experiment.config.risk_per_trade_pct,
            'max_positions': experiment.config.max_positions,
            'tags': experiment.config.tags
        }
        
        metrics_dict = asdict(experiment.metrics)
        
        cursor.execute('''
            INSERT OR REPLACE INTO experiments 
            (experiment_id, config, status, created_at, updated_at, metrics, notes, version, parent_experiment_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            experiment.experiment_id,
            json.dumps(config_dict),
            experiment.status.value,
            experiment.created_at.isoformat(),
            experiment.updated_at.isoformat(),
            json.dumps(metrics_dict),
            json.dumps(experiment.notes),
            experiment.version,
            experiment.parent_experiment_id
        ))
        conn.commit()
        conn.close()
    
    def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """Get an experiment by ID"""
        return self._experiments.get(experiment_id)
    
    def list_experiments(self, status: ExperimentStatus = None, 
                         strategy: str = None) -> List[Experiment]:
        """List experiments with optional filters"""
        experiments = list(self._experiments.values())
        
        if status:
            experiments = [e for e in experiments if e.status == status]
        
        if strategy:
            experiments = [e for e in experiments if e.config.strategy_name == strategy]
        
        return sorted(experiments, key=lambda e: e.created_at, reverse=True)
    
    def get_best_experiments(self, metric: str = 'profit_factor', top_n: int = 5) -> List[Experiment]:
        """Get top performing experiments by a metric"""
        completed = [e for e in self._experiments.values() 
                    if e.status in [ExperimentStatus.COMPLETED, ExperimentStatus.PROMOTED]]
        
        return sorted(completed, key=lambda e: getattr(e.metrics, metric, 0), reverse=True)[:top_n]
    
    def calculate_metrics_from_trades(self, trades: List[ShadowTrade]) -> ExperimentMetrics:
        """Calculate experiment metrics from shadow trades"""
        if not trades:
            return ExperimentMetrics()
        
        closed_trades = [t for t in trades if t.status == ShadowTradeStatus.CLOSED]
        if not closed_trades:
            return ExperimentMetrics()
        
        winning = [t for t in closed_trades if t.pnl > 0]
        losing = [t for t in closed_trades if t.pnl <= 0]
        
        total_wins = sum(t.pnl for t in winning)
        total_losses = abs(sum(t.pnl for t in losing))
        
        # Calculate drawdown
        equity_curve = []
        running_equity = 0
        peak = 0
        max_dd = 0
        
        for trade in sorted(closed_trades, key=lambda t: t.exit_time or t.entry_time):
            running_equity += trade.pnl
            equity_curve.append(running_equity)
            if running_equity > peak:
                peak = running_equity
            dd = peak - running_equity
            if dd > max_dd:
                max_dd = dd
        
        # Calculate average trade duration
        durations = []
        for t in closed_trades:
            if t.exit_time and t.entry_time:
                durations.append((t.exit_time - t.entry_time).total_seconds() / 3600)
        
        avg_duration = sum(durations) / len(durations) if durations else 0
        
        return ExperimentMetrics(
            total_trades=len(closed_trades),
            winning_trades=len(winning),
            losing_trades=len(losing),
            win_rate=len(winning) / len(closed_trades) if closed_trades else 0,
            total_pnl=sum(t.pnl for t in closed_trades),
            total_pnl_pips=sum(t.pnl_pips for t in closed_trades),
            avg_win=total_wins / len(winning) if winning else 0,
            avg_loss=total_losses / len(losing) if losing else 0,
            profit_factor=total_wins / total_losses if total_losses > 0 else float('inf'),
            max_drawdown=max_dd,
            max_drawdown_pct=max_dd / peak if peak > 0 else 0,
            avg_trade_duration_hours=avg_duration,
            expectancy=(sum(t.pnl for t in closed_trades) / len(closed_trades)) if closed_trades else 0,
            recovery_factor=sum(t.pnl for t in closed_trades) / max_dd if max_dd > 0 else 0
        )


class ImmutableDatasetManager:
    """Manage immutable market data snapshots for reproducible backtests"""
    
    def __init__(self, storage_dir: str = "datasets", db_path: str = "datasets.db"):
        self.storage_dir = storage_dir
        self.db_path = db_path
        self._lock = threading.Lock()
        
        os.makedirs(storage_dir, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS dataset_snapshots (
                snapshot_id TEXT PRIMARY KEY,
                name TEXT,
                description TEXT,
                symbols TEXT,
                timeframe TEXT,
                start_date TEXT,
                end_date TEXT,
                created_at TEXT,
                data_hash TEXT,
                row_count INTEGER,
                file_path TEXT,
                metadata TEXT
            )
        ''')
        conn.commit()
        conn.close()
    
    def create_snapshot(self, name: str, description: str, data: Dict[str, Any],
                        symbols: List[str], timeframe: str, 
                        start_date: datetime, end_date: datetime,
                        metadata: Dict = None) -> DatasetSnapshot:
        """Create an immutable snapshot of market data"""
        snapshot_id = str(uuid.uuid4())
        
        # Serialize and compress data
        serialized = pickle.dumps(data)
        compressed = gzip.compress(serialized)
        
        # Calculate hash for integrity
        data_hash = hashlib.sha256(compressed).hexdigest()
        
        # Save to file
        file_path = os.path.join(self.storage_dir, f"{snapshot_id}.gz")
        with open(file_path, 'wb') as f:
            f.write(compressed)
        
        # Count rows
        row_count = 0
        for symbol_data in data.values():
            if hasattr(symbol_data, '__len__'):
                row_count += len(symbol_data)
        
        snapshot = DatasetSnapshot(
            snapshot_id=snapshot_id,
            name=name,
            description=description,
            symbols=symbols,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            created_at=datetime.now(),
            data_hash=data_hash,
            row_count=row_count,
            file_path=file_path,
            metadata=metadata or {}
        )
        
        self._save_snapshot_metadata(snapshot)
        logger.info(f"Created dataset snapshot {snapshot_id}: {name} ({row_count} rows)")
        
        return snapshot
    
    def _save_snapshot_metadata(self, snapshot: DatasetSnapshot):
        """Save snapshot metadata to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO dataset_snapshots 
            (snapshot_id, name, description, symbols, timeframe, start_date, end_date,
             created_at, data_hash, row_count, file_path, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            snapshot.snapshot_id, snapshot.name, snapshot.description,
            json.dumps(snapshot.symbols), snapshot.timeframe,
            snapshot.start_date.isoformat(), snapshot.end_date.isoformat(),
            snapshot.created_at.isoformat(), snapshot.data_hash,
            snapshot.row_count, snapshot.file_path, json.dumps(snapshot.metadata)
        ))
        conn.commit()
        conn.close()
    
    def load_snapshot(self, snapshot_id: str) -> Tuple[Optional[DatasetSnapshot], Optional[Dict]]:
        """Load a dataset snapshot with integrity verification"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM dataset_snapshots WHERE snapshot_id = ?', (snapshot_id,))
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return None, None
        
        snapshot = DatasetSnapshot(
            snapshot_id=row[0],
            name=row[1],
            description=row[2],
            symbols=json.loads(row[3]),
            timeframe=row[4],
            start_date=datetime.fromisoformat(row[5]),
            end_date=datetime.fromisoformat(row[6]),
            created_at=datetime.fromisoformat(row[7]),
            data_hash=row[8],
            row_count=row[9],
            file_path=row[10],
            metadata=json.loads(row[11]) if row[11] else {}
        )
        
        # Load and verify data
        if not os.path.exists(snapshot.file_path):
            logger.error(f"Dataset file not found: {snapshot.file_path}")
            return snapshot, None
        
        with open(snapshot.file_path, 'rb') as f:
            compressed = f.read()
        
        # Verify integrity
        actual_hash = hashlib.sha256(compressed).hexdigest()
        if actual_hash != snapshot.data_hash:
            logger.error(f"Dataset integrity check failed for {snapshot_id}")
            return snapshot, None
        
        # Decompress and deserialize
        serialized = gzip.decompress(compressed)
        data = pickle.loads(serialized)
        
        logger.info(f"Loaded dataset snapshot {snapshot_id}: {snapshot.name}")
        return snapshot, data
    
    def list_snapshots(self, symbol: str = None, timeframe: str = None) -> List[DatasetSnapshot]:
        """List available dataset snapshots"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM dataset_snapshots ORDER BY created_at DESC')
        
        snapshots = []
        for row in cursor.fetchall():
            snapshot = DatasetSnapshot(
                snapshot_id=row[0],
                name=row[1],
                description=row[2],
                symbols=json.loads(row[3]),
                timeframe=row[4],
                start_date=datetime.fromisoformat(row[5]),
                end_date=datetime.fromisoformat(row[6]),
                created_at=datetime.fromisoformat(row[7]),
                data_hash=row[8],
                row_count=row[9],
                file_path=row[10],
                metadata=json.loads(row[11]) if row[11] else {}
            )
            
            # Apply filters
            if symbol and symbol not in snapshot.symbols:
                continue
            if timeframe and snapshot.timeframe != timeframe:
                continue
            
            snapshots.append(snapshot)
        
        conn.close()
        return snapshots
    
    def delete_snapshot(self, snapshot_id: str) -> bool:
        """Delete a dataset snapshot"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT file_path FROM dataset_snapshots WHERE snapshot_id = ?', (snapshot_id,))
        row = cursor.fetchone()
        
        if not row:
            conn.close()
            return False
        
        file_path = row[0]
        
        # Delete from database
        cursor.execute('DELETE FROM dataset_snapshots WHERE snapshot_id = ?', (snapshot_id,))
        conn.commit()
        conn.close()
        
        # Delete file
        if os.path.exists(file_path):
            os.remove(file_path)
        
        logger.info(f"Deleted dataset snapshot {snapshot_id}")
        return True


class ResearchProductionManager:
    """Main manager for research/production separation - Tier 3 implementation"""
    
    def __init__(self, base_dir: str = "."):
        self.base_dir = base_dir
        
        # Initialize components
        self.shadow_engine = ShadowModeEngine(
            db_path=os.path.join(base_dir, "shadow_mode.db")
        )
        self.experiment_tracker = ExperimentTracker(
            db_path=os.path.join(base_dir, "experiments.db")
        )
        self.dataset_manager = ImmutableDatasetManager(
            storage_dir=os.path.join(base_dir, "datasets"),
            db_path=os.path.join(base_dir, "datasets.db")
        )
        
        # Shadow mode state
        self._shadow_mode_enabled = False
        self._active_experiment_id: Optional[str] = None
        
        logger.info("Research/Production Manager initialized")
    
    def enable_shadow_mode(self, experiment_id: str = None) -> str:
        """Enable shadow mode for testing strategies"""
        if experiment_id:
            experiment = self.experiment_tracker.get_experiment(experiment_id)
            if not experiment:
                raise ValueError(f"Experiment {experiment_id} not found")
            self._active_experiment_id = experiment_id
        else:
            # Create a default experiment
            config = ExperimentConfig(
                name=f"Shadow Mode {datetime.now().strftime('%Y%m%d_%H%M%S')}",
                description="Auto-created shadow mode experiment",
                strategy_name="shadow_default",
                strategy_params={},
                symbols=['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF'],
                start_date=datetime.now()
            )
            experiment = self.experiment_tracker.create_experiment(config)
            self._active_experiment_id = experiment.experiment_id
            self.experiment_tracker.start_experiment(experiment.experiment_id)
        
        self._shadow_mode_enabled = True
        logger.info(f"Shadow mode enabled with experiment {self._active_experiment_id}")
        return self._active_experiment_id
    
    def disable_shadow_mode(self) -> Optional[ExperimentMetrics]:
        """Disable shadow mode and return metrics"""
        if not self._shadow_mode_enabled:
            return None
        
        self._shadow_mode_enabled = False
        
        # Calculate final metrics
        if self._active_experiment_id:
            trades = self.shadow_engine.get_trade_history(self._active_experiment_id)
            metrics = self.experiment_tracker.calculate_metrics_from_trades(trades)
            self.experiment_tracker.complete_experiment(self._active_experiment_id, metrics)
            
            logger.info(f"Shadow mode disabled. Experiment {self._active_experiment_id} completed.")
            self._active_experiment_id = None
            return metrics
        
        return None
    
    @property
    def is_shadow_mode(self) -> bool:
        return self._shadow_mode_enabled
    
    @property
    def active_experiment_id(self) -> Optional[str]:
        return self._active_experiment_id
    
    def process_signal_in_shadow(self, symbol: str, direction: str, entry_price: float,
                                  stop_loss: float, take_profit: float, position_size: float,
                                  strategy: str, confidence: float, metadata: Dict = None) -> Optional[ShadowTrade]:
        """Process a trading signal in shadow mode"""
        if not self._shadow_mode_enabled or not self._active_experiment_id:
            return None
        
        return self.shadow_engine.open_shadow_trade(
            experiment_id=self._active_experiment_id,
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=position_size,
            strategy=strategy,
            confidence=confidence,
            metadata=metadata
        )
    
    def update_shadow_prices(self, current_prices: Dict[str, float]):
        """Update shadow trades with current prices"""
        if self._shadow_mode_enabled:
            self.shadow_engine.update_shadow_trades(current_prices)
    
    def get_shadow_performance(self) -> Dict[str, Any]:
        """Get current shadow mode performance"""
        if not self._active_experiment_id:
            return {}
        
        trades = self.shadow_engine.get_trade_history(self._active_experiment_id)
        active = self.shadow_engine.get_active_trades(self._active_experiment_id)
        metrics = self.experiment_tracker.calculate_metrics_from_trades(trades)
        
        return {
            'experiment_id': self._active_experiment_id,
            'total_trades': metrics.total_trades,
            'active_trades': len(active),
            'win_rate': metrics.win_rate,
            'total_pnl': metrics.total_pnl,
            'profit_factor': metrics.profit_factor,
            'max_drawdown': metrics.max_drawdown
        }
    
    def create_dataset_from_current_data(self, name: str, data: Dict[str, Any],
                                          symbols: List[str], timeframe: str,
                                          start_date: datetime, end_date: datetime) -> DatasetSnapshot:
        """Create an immutable dataset snapshot from current market data"""
        return self.dataset_manager.create_snapshot(
            name=name,
            description=f"Market data snapshot for {', '.join(symbols)}",
            data=data,
            symbols=symbols,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date
        )
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of research/production state"""
        experiments = self.experiment_tracker.list_experiments()
        datasets = self.dataset_manager.list_snapshots()
        
        return {
            'shadow_mode_enabled': self._shadow_mode_enabled,
            'active_experiment_id': self._active_experiment_id,
            'total_experiments': len(experiments),
            'running_experiments': len([e for e in experiments if e.status == ExperimentStatus.RUNNING]),
            'completed_experiments': len([e for e in experiments if e.status == ExperimentStatus.COMPLETED]),
            'promoted_experiments': len([e for e in experiments if e.status == ExperimentStatus.PROMOTED]),
            'total_datasets': len(datasets),
            'shadow_performance': self.get_shadow_performance() if self._shadow_mode_enabled else None
        }


# Global instance
_research_production_manager: Optional[ResearchProductionManager] = None
_lock = threading.Lock()


def get_research_production_manager() -> ResearchProductionManager:
    """Get or create the global research/production manager instance"""
    global _research_production_manager
    with _lock:
        if _research_production_manager is None:
            _research_production_manager = ResearchProductionManager()
        return _research_production_manager


# Convenience functions
def enable_shadow_mode(experiment_id: str = None) -> str:
    """Enable shadow mode"""
    return get_research_production_manager().enable_shadow_mode(experiment_id)


def disable_shadow_mode() -> Optional[ExperimentMetrics]:
    """Disable shadow mode"""
    return get_research_production_manager().disable_shadow_mode()


def is_shadow_mode() -> bool:
    """Check if shadow mode is enabled"""
    return get_research_production_manager().is_shadow_mode


def process_shadow_signal(symbol: str, direction: str, entry_price: float,
                          stop_loss: float, take_profit: float, position_size: float,
                          strategy: str, confidence: float) -> Optional[ShadowTrade]:
    """Process a signal in shadow mode"""
    return get_research_production_manager().process_signal_in_shadow(
        symbol, direction, entry_price, stop_loss, take_profit,
        position_size, strategy, confidence
    )


def update_shadow_prices(current_prices: Dict[str, float]):
    """Update shadow trades with current prices"""
    get_research_production_manager().update_shadow_prices(current_prices)


if __name__ == "__main__":
    # Test the research/production manager
    logging.basicConfig(level=logging.INFO)
    
    manager = get_research_production_manager()
    
    # Enable shadow mode
    exp_id = manager.enable_shadow_mode()
    print(f"Shadow mode enabled: {exp_id}")
    
    # Simulate a shadow trade
    trade = manager.process_signal_in_shadow(
        symbol='EURUSD',
        direction='BUY',
        entry_price=1.1000,
        stop_loss=1.0950,
        take_profit=1.1100,
        position_size=0.1,
        strategy='test_strategy',
        confidence=0.75
    )
    print(f"Shadow trade opened: {trade.trade_id}")
    
    # Update with new price
    manager.update_shadow_prices({'EURUSD': 1.1050})
    
    # Get performance
    perf = manager.get_shadow_performance()
    print(f"Shadow performance: {perf}")
    
    # Disable shadow mode
    metrics = manager.disable_shadow_mode()
    print(f"Final metrics: {metrics}")
    
    # Get summary
    summary = manager.get_summary()
    print(f"Summary: {summary}")
