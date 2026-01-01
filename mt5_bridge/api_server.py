"""
MT5 Bridge API Server
Receives trade signals from the AI trading system and serves them to the MT5 EA
Also receives account info and position updates from MT5
Includes SQLite persistence and trade gating integration
"""

import os
import sys
import json
import uuid
import logging
from collections import OrderedDict
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, field, asdict
from threading import Lock
from flask import Flask, request, Response, jsonify

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from persistence import get_persistence, PersistenceManager
    PERSISTENCE_AVAILABLE = True
except ImportError:
    PERSISTENCE_AVAILABLE = False

try:
    from trade_gating import get_trade_gating, TradeGatingSystem
    TRADE_GATING_AVAILABLE = True
except ImportError:
    TRADE_GATING_AVAILABLE = False

try:
    from monitoring import monitoring, AlertLevel, AlertChannel, realtime_alerts
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False
    realtime_alerts = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

@dataclass
class TradeSignal:
    id: str
    symbol: str
    action: str  # buy, sell, close, modify
    volume: float
    sl: float = 0.0
    tp: float = 0.0
    ticket: int = 0  # For close/modify actions
    comment: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    executed: bool = False
    result: Optional[Dict] = None


class MarketDataBuffer:
    """Ring buffer for storing live market data from MT5 with SQLite persistence"""
    
    def __init__(self, max_bars: int = 500):
        self.max_bars = max_bars
        self.data: Dict[str, Dict] = {}  # symbol -> {quote, bars}
        self.lock = Lock()
        self.last_update: Optional[datetime] = None
        self._load_persisted_bars()
    
    def _load_persisted_bars(self):
        """Load bars from SQLite on startup"""
        if not PERSISTENCE_AVAILABLE:
            return
        try:
            persistence = get_persistence()
            symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD', 'EURGBP']
            for symbol in symbols:
                bars = persistence.load_bars(symbol, self.max_bars)
                if bars:
                    self.data[symbol] = {'quote': {}, 'bars': bars}
                    logger.info(f"Loaded {len(bars)} persisted bars for {symbol}")
        except Exception as e:
            logger.warning(f"Failed to load persisted bars: {e}")
    
    def update(self, symbol: str, quote_data: Dict, bars: List[Dict]):
        with self.lock:
            if symbol not in self.data:
                self.data[symbol] = {'quote': {}, 'bars': []}
            
            self.data[symbol]['quote'] = quote_data
            
            # Merge new bars with existing, keeping most recent max_bars
            existing_bars = self.data[symbol]['bars']
            if bars:
                # Convert bars to dict keyed by timestamp for deduplication
                bar_dict = {b['t']: b for b in existing_bars}
                for bar in bars:
                    bar_dict[bar['t']] = bar
                # Sort by timestamp and keep most recent
                sorted_bars = sorted(bar_dict.values(), key=lambda x: x['t'])
                self.data[symbol]['bars'] = sorted_bars[-self.max_bars:]
                
                # Persist bars to SQLite
                if PERSISTENCE_AVAILABLE:
                    try:
                        persistence = get_persistence()
                        persistence.save_bars(symbol, bars)
                    except Exception as e:
                        logger.warning(f"Failed to persist bars for {symbol}: {e}")
            
            self.last_update = datetime.now()
    
    def get_quote(self, symbol: str) -> Optional[Dict]:
        with self.lock:
            if symbol in self.data:
                return self.data[symbol]['quote'].copy()
            return None
    
    def get_bars(self, symbol: str, count: int = 200) -> List[Dict]:
        with self.lock:
            if symbol in self.data:
                bars = self.data[symbol]['bars']
                return bars[-count:] if len(bars) > count else bars.copy()
            return []
    
    def get_all_symbols(self) -> List[str]:
        with self.lock:
            return list(self.data.keys())
    
    def get_spread(self, symbol: str) -> float:
        quote = self.get_quote(symbol)
        return quote.get('spread', 0.0) if quote else 0.0
    
    def get_bid_ask(self, symbol: str) -> tuple:
        quote = self.get_quote(symbol)
        if quote:
            return quote.get('bid', 0.0), quote.get('ask', 0.0)
        return 0.0, 0.0


class ClosedTradesStore:
    """Store for closed trades used for learning feedback loop with SQLite persistence"""
    
    def __init__(self, max_trades: int = 1000):
        self.max_trades = max_trades
        self.trades: List[Dict] = []
        self.lock = Lock()
        self.last_update: Optional[datetime] = None
        self.total_profit: float = 0.0
        self.win_count: int = 0
        self.loss_count: int = 0
        self._load_persisted_trades()
    
    def _load_persisted_trades(self):
        """Load trades from SQLite on startup"""
        if not PERSISTENCE_AVAILABLE:
            return
        try:
            persistence = get_persistence()
            trades = persistence.load_trades(self.max_trades)
            for trade in trades:
                self.trades.append(trade)
                profit = trade.get('profit', 0.0)
                self.total_profit += profit
                if profit > 0:
                    self.win_count += 1
                elif profit < 0:
                    self.loss_count += 1
            if trades:
                logger.info(f"Loaded {len(trades)} persisted trades, P&L=${self.total_profit:.2f}")
        except Exception as e:
            logger.warning(f"Failed to load persisted trades: {e}")
    
    def add_trades(self, trades: List[Dict]):
        with self.lock:
            for trade in trades:
                # Avoid duplicates by ticket
                if not any(t.get('ticket') == trade.get('ticket') for t in self.trades):
                    self.trades.append(trade)
                    profit = trade.get('profit', 0.0)
                    self.total_profit += profit
                    if profit > 0:
                        self.win_count += 1
                    elif profit < 0:
                        self.loss_count += 1
                    
                    # Persist trade to SQLite
                    if PERSISTENCE_AVAILABLE:
                        try:
                            persistence = get_persistence()
                            persistence.save_trade(trade)
                        except Exception as e:
                            logger.warning(f"Failed to persist trade: {e}")
            
            # Keep only most recent trades
            if len(self.trades) > self.max_trades:
                self.trades = self.trades[-self.max_trades:]
            
            self.last_update = datetime.now()
            logger.info(f"Closed trades updated: {len(trades)} new, total={len(self.trades)}, P&L=${self.total_profit:.2f}")
    
    def get_recent_trades(self, count: int = 50) -> List[Dict]:
        with self.lock:
            return self.trades[-count:] if len(self.trades) > count else self.trades.copy()
    
    def get_stats(self) -> Dict:
        with self.lock:
            total = self.win_count + self.loss_count
            win_rate = self.win_count / total if total > 0 else 0.0
            return {
                'total_trades': len(self.trades),
                'total_profit': self.total_profit,
                'win_count': self.win_count,
                'loss_count': self.loss_count,
                'win_rate': win_rate,
                'last_update': self.last_update.isoformat() if self.last_update else None
            }


class MT5BridgeState:
    """Shared state for MT5 bridge communication"""
    
    def __init__(self):
        self.lock = Lock()
        self.account_info: Dict = {}
        self.positions: List[Dict] = []
        self.pending_signals: List[TradeSignal] = []
        self.executed_signals: List[TradeSignal] = []
        self.last_update: Optional[datetime] = None
        self.is_connected: bool = False
        self.connection_time: Optional[datetime] = None
        self.market_data = MarketDataBuffer()
        self.closed_trades = ClosedTradesStore()
        
    def update_account(self, info: Dict):
        with self.lock:
            self.account_info = info
            self.last_update = datetime.now()
            self.is_connected = True
            if not self.connection_time:
                self.connection_time = datetime.now()
            logger.info(f"Account updated: Balance=${info.get('balance', 0):.2f}, Equity=${info.get('equity', 0):.2f}")
    
    def update_positions(self, positions: List[Dict]):
        with self.lock:
            self.positions = positions
            self.last_update = datetime.now()
    
    def add_signal(self, signal: TradeSignal):
        with self.lock:
            self.pending_signals.append(signal)
            logger.info(f"Signal added: {signal.id} - {signal.symbol} {signal.action} {signal.volume} lots")
    
    def get_pending_signals(self) -> List[OrderedDict]:
        """Get pending signals with id first for EA parsing compatibility"""
        with self.lock:
            signals = []
            for sig in self.pending_signals:
                if not sig.executed:
                    # IMPORTANT: id must be first key for EA JSON parsing
                    # Using OrderedDict to preserve key order
                    signals.append(OrderedDict([
                        ('id', sig.id),
                        ('symbol', sig.symbol),
                        ('action', sig.action),
                        ('volume', sig.volume),
                        ('sl', sig.sl),
                        ('tp', sig.tp),
                        ('ticket', sig.ticket),
                        ('comment', sig.comment)
                    ]))
            return signals
    
    def mark_signal_executed(self, signal_id: str, success: bool, ticket: int, error: str):
        with self.lock:
            for sig in self.pending_signals:
                if sig.id == signal_id:
                    sig.executed = True
                    sig.result = {
                        'success': success,
                        'ticket': ticket,
                        'error': error,
                        'executed_at': datetime.now().isoformat()
                    }
                    self.executed_signals.append(sig)
                    self.pending_signals.remove(sig)
                    logger.info(f"Signal {signal_id} executed: success={success}, ticket={ticket}")
                    break
    
    def get_balance(self) -> float:
        with self.lock:
            return self.account_info.get('balance', 0.0)
    
    def get_equity(self) -> float:
        with self.lock:
            return self.account_info.get('equity', 0.0)
    
    def is_mt5_connected(self) -> bool:
        with self.lock:
            if not self.last_update:
                return False
            return (datetime.now() - self.last_update).seconds < 30


# Global state
bridge_state = MT5BridgeState()


# API Routes

@app.route('/api/register', methods=['POST'])
def register():
    """EA registration endpoint"""
    data = request.get_json()
    logger.info(f"EA registered: Login={data.get('login')}, Server={data.get('server')}")
    bridge_state.update_account(data)
    if realtime_alerts:
        realtime_alerts.record_ea_poll()
    return jsonify({'status': 'ok', 'message': 'Registered successfully'})


@app.route('/api/account', methods=['POST'])
def update_account():
    """Receive account info from EA"""
    data = request.get_json()
    bridge_state.update_account(data)
    return jsonify({'status': 'ok'})


@app.route('/api/account', methods=['GET'])
def get_account():
    """Get current account info"""
    return jsonify({
        'connected': bridge_state.is_mt5_connected(),
        'account': bridge_state.account_info,
        'last_update': bridge_state.last_update.isoformat() if bridge_state.last_update else None
    })


@app.route('/api/signals', methods=['GET'])
def get_signals():
    """EA polls this endpoint for pending trade signals"""
    if realtime_alerts:
        realtime_alerts.record_ea_poll()
    signals = bridge_state.get_pending_signals()
    # Use json.dumps with sort_keys=False to preserve key order for EA parsing
    # EA expects {"id":... to be first in each signal object
    response_data = json.dumps({'signals': signals}, sort_keys=False, separators=(',', ':'))
    return Response(response_data, mimetype='application/json')


@app.route('/api/signals', methods=['POST'])
def add_signal():
    """Add a new trade signal (called by trading system)"""
    data = request.get_json()
    
    signal = TradeSignal(
        id=data.get('id', str(uuid.uuid4())),
        symbol=data['symbol'],
        action=data['action'],
        volume=data.get('volume', 0.01),
        sl=data.get('sl', 0.0),
        tp=data.get('tp', 0.0),
        ticket=data.get('ticket', 0),
        comment=data.get('comment', 'AI_Trade')
    )
    
    bridge_state.add_signal(signal)
    
    return jsonify({'status': 'ok', 'signal_id': signal.id})


@app.route('/api/signal_result', methods=['POST'])
def signal_result():
    """EA reports signal execution result"""
    data = request.get_json()
    
    success = data.get('success', False)
    bridge_state.mark_signal_executed(
        signal_id=data['signal_id'],
        success=success,
        ticket=data.get('ticket', 0),
        error=data.get('error', '')
    )
    
    if realtime_alerts:
        if success:
            realtime_alerts.record_trade_success()
        else:
            realtime_alerts.record_trade_failure()
    
    return jsonify({'status': 'ok'})


@app.route('/api/signal_status/<signal_id>', methods=['GET'])
def get_signal_status(signal_id):
    """Check status of a specific signal"""
    with bridge_state.lock:
        # Check pending signals
        for sig in bridge_state.pending_signals:
            if sig.id == signal_id:
                return jsonify({
                    'found': True,
                    'executed': False,
                    'pending': True,
                    'result': None
                })
        
        # Check executed signals
        for sig in bridge_state.executed_signals:
            if sig.id == signal_id:
                return jsonify({
                    'found': True,
                    'executed': True,
                    'pending': False,
                    'result': sig.result
                })
    
    return jsonify({
        'found': False,
        'executed': False,
        'pending': False,
        'result': None
    })


@app.route('/api/positions', methods=['POST'])
def update_positions():
    """EA sends position updates"""
    data = request.get_json()
    positions = data.get('positions', [])
    bridge_state.update_positions(positions)
    return jsonify({'status': 'ok'})


@app.route('/api/positions', methods=['GET'])
def get_positions():
    """Get current positions"""
    return jsonify({
        'positions': bridge_state.positions,
        'count': len(bridge_state.positions)
    })


@app.route('/api/status', methods=['GET'])
def get_status():
    """Get bridge status"""
    return jsonify({
        'mt5_connected': bridge_state.is_mt5_connected(),
        'connection_time': bridge_state.connection_time.isoformat() if bridge_state.connection_time else None,
        'last_update': bridge_state.last_update.isoformat() if bridge_state.last_update else None,
        'balance': bridge_state.get_balance(),
        'equity': bridge_state.get_equity(),
        'open_positions': len(bridge_state.positions),
        'pending_signals': len(bridge_state.pending_signals),
        'executed_signals': len(bridge_state.executed_signals)
    })


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})


@app.route('/api/market_data', methods=['POST'])
def receive_market_data():
    """Receive live market data from EA (bid/ask/spread + OHLC bars)"""
    data = request.get_json()
    
    symbols_data = data.get('symbols', [])
    is_initial = data.get('initial', False)
    
    # Track which symbols need full history resync
    symbols_needing_resync = []
    
    # Record bars update for real-time alerts
    if realtime_alerts and symbols_data:
        realtime_alerts.record_bars_update()
    
    for sym_data in symbols_data:
        symbol = sym_data.get('symbol', '')
        if not symbol:
            continue
        
        spread_points = sym_data.get('spread', 0.0)
        point = sym_data.get('point', 0.00001)
        spread_pips = spread_points * point * 10000 if 'JPY' not in symbol else spread_points * point * 100
        
        # Check spread for alerts
        if realtime_alerts:
            realtime_alerts.check_spread(symbol, spread_pips)
        
        quote_data = {
            'bid': sym_data.get('bid', 0.0),
            'ask': sym_data.get('ask', 0.0),
            'spread': sym_data.get('spread', 0.0),
            'spread_pips': spread_pips,
            'digits': sym_data.get('digits', 5),
            'point': point,
            'stop_level': sym_data.get('stop_level', 0),
            'freeze_level': sym_data.get('freeze_level', 0),
            'tick_value': sym_data.get('tick_value', 1.0),
            'min_lot': sym_data.get('min_lot', 0.01),
            'lot_step': sym_data.get('lot_step', 0.01),
            'timestamp': data.get('ts', 0)
        }
        
        bars = sym_data.get('bars', [])
        bridge_state.market_data.update(symbol, quote_data, bars)
        
        # Check if we need more bars for this symbol (less than 50 bars = need resync)
        current_bar_count = len(bridge_state.market_data.get_bars(symbol, 500))
        if current_bar_count < 50:
            symbols_needing_resync.append(symbol)
    
    if is_initial:
        logger.info(f"Initial market data received for {len(symbols_data)} symbols")
    
    # Tell EA to resend full history if we're missing bars
    need_resync = len(symbols_needing_resync) > 0
    if need_resync and not is_initial:
        logger.warning(f"Requesting bar resync for {len(symbols_needing_resync)} symbols: {symbols_needing_resync}")
    
    return jsonify({
        'status': 'ok', 
        'symbols_updated': len(symbols_data),
        'need_full_history': need_resync,
        'symbols_needing_resync': symbols_needing_resync
    })


@app.route('/api/market_data', methods=['GET'])
def get_market_data():
    """Get current market data for analysis"""
    symbol = request.args.get('symbol', None)
    count = int(request.args.get('count', 200))
    
    if symbol:
        quote = bridge_state.market_data.get_quote(symbol)
        bars = bridge_state.market_data.get_bars(symbol, count)
        return jsonify({
            'symbol': symbol,
            'quote': quote,
            'bars': bars,
            'bar_count': len(bars)
        })
    else:
        # Return all symbols
        symbols = bridge_state.market_data.get_all_symbols()
        result = {}
        for sym in symbols:
            quote = bridge_state.market_data.get_quote(sym)
            bars = bridge_state.market_data.get_bars(sym, count)
            result[sym] = {
                'quote': quote,
                'bars': bars,
                'bar_count': len(bars)
            }
        return jsonify({
            'symbols': result,
            'symbol_count': len(symbols),
            'last_update': bridge_state.market_data.last_update.isoformat() if bridge_state.market_data.last_update else None
        })


@app.route('/api/closed_trades', methods=['POST'])
def receive_closed_trades():
    """Receive closed trade history from EA for learning feedback loop"""
    data = request.get_json()
    
    deals = data.get('deals', [])
    if deals:
        bridge_state.closed_trades.add_trades(deals)
    
    return jsonify({'status': 'ok', 'trades_received': len(deals)})


@app.route('/api/closed_trades', methods=['GET'])
def get_closed_trades():
    """Get closed trades for learning analysis"""
    count = int(request.args.get('count', 50))
    
    trades = bridge_state.closed_trades.get_recent_trades(count)
    stats = bridge_state.closed_trades.get_stats()
    
    return jsonify({
        'trades': trades,
        'stats': stats
    })


@app.route('/api/dashboard', methods=['GET'])
def get_dashboard():
    """Get comprehensive dashboard data for web UI"""
    from datetime import datetime
    
    # Account info
    account = {
        'balance': bridge_state.account_info.get('balance', 0),
        'equity': bridge_state.account_info.get('equity', 0),
        'margin': bridge_state.account_info.get('margin', 0),
        'free_margin': bridge_state.account_info.get('free_margin', 0),
        'margin_level': bridge_state.account_info.get('margin_level', 0),
        'profit': bridge_state.account_info.get('profit', 0),
    }
    
    # Calculate P&L
    starting_balance = 100.0  # Initial deposit
    total_pnl = account['balance'] - starting_balance
    pnl_pct = (total_pnl / starting_balance) * 100 if starting_balance > 0 else 0
    
    # Positions
    positions = bridge_state.positions.copy()
    
    # Recent signals (last 20)
    recent_signals = []
    for sig in list(bridge_state.executed_signals)[-20:]:
        recent_signals.append({
            'id': sig.id,
            'symbol': sig.symbol,
            'action': sig.action,
            'volume': sig.volume,
            'status': sig.status,
            'result': sig.result,
            'timestamp': sig.timestamp.isoformat() if hasattr(sig, 'timestamp') else None
        })
    
    # Trade stats
    trade_stats = bridge_state.closed_trades.get_stats()
    
    # System health
    ea_connected = bridge_state.is_mt5_connected()
    last_ea_poll = bridge_state.last_poll.isoformat() if bridge_state.last_poll else None
    
    # Market data freshness
    market_data_symbols = bridge_state.market_data.get_all_symbols()
    bars_fresh = len(market_data_symbols) > 0
    
    # Get realtime alerts status if available
    alerts_status = {}
    if realtime_alerts:
        alerts_status = {
            'ea_connected': ea_connected,
            'last_ea_poll': last_ea_poll,
            'bars_fresh': bars_fresh,
            'consecutive_failures': realtime_alerts.consecutive_failures,
        }
    
    # Token usage (NewsAPI.ai)
    token_usage = {
        'newsapi_tokens_used': 0,  # Would need to track this in knowledge_acquisition
        'newsapi_tokens_total': 2000,
        'newsapi_tokens_remaining': 2000,
    }
    
    # Try to get actual token usage from knowledge acquisition
    try:
        import sys
        sys.path.insert(0, '/home/ec2-user/forex_trading_system')
        from knowledge_acquisition import knowledge_system
        if hasattr(knowledge_system, 'newsapi_tokens_used'):
            token_usage['newsapi_tokens_used'] = knowledge_system.newsapi_tokens_used
            token_usage['newsapi_tokens_remaining'] = 2000 - knowledge_system.newsapi_tokens_used
    except:
        pass
    
    # Macro data (DXY, VIX, Treasury)
    macro_data = {}
    try:
        from data import MacroDataFetcher
        macro_fetcher = MacroDataFetcher()
        macro_data = macro_fetcher.get_all_macro_data()
    except:
        macro_data = {'error': 'Macro data not available'}
    
    return jsonify({
        'account': account,
        'pnl': {
            'total': round(total_pnl, 2),
            'percent': round(pnl_pct, 2),
            'starting_balance': starting_balance
        },
        'positions': positions,
        'position_count': len(positions),
        'recent_signals': recent_signals,
        'trade_stats': trade_stats,
        'health': {
            'ea_connected': ea_connected,
            'last_ea_poll': last_ea_poll,
            'bars_fresh': bars_fresh,
            'symbols_with_data': len(market_data_symbols),
            'alerts': alerts_status
        },
        'token_usage': token_usage,
        'macro_data': macro_data,
        'timestamp': datetime.now().isoformat()
    })


# Trading system interface functions

def send_trade_signal(symbol: str, action: str, volume: float, 
                      sl: float = 0.0, tp: float = 0.0, 
                      ticket: int = 0, comment: str = "") -> str:
    """
    Send a trade signal to be executed by MT5
    Returns the signal ID for tracking
    """
    signal = TradeSignal(
        id=str(uuid.uuid4()),
        symbol=symbol,
        action=action,
        volume=volume,
        sl=sl,
        tp=tp,
        ticket=ticket,
        comment=comment
    )
    bridge_state.add_signal(signal)
    return signal.id


def get_mt5_balance() -> float:
    """Get current MT5 account balance"""
    return bridge_state.get_balance()


def get_mt5_equity() -> float:
    """Get current MT5 account equity"""
    return bridge_state.get_equity()


def get_mt5_positions() -> List[Dict]:
    """Get current MT5 positions"""
    return bridge_state.positions.copy()


def is_mt5_connected() -> bool:
    """Check if MT5 EA is connected"""
    return bridge_state.is_mt5_connected()


def get_market_data_for_symbol(symbol: str, count: int = 200) -> tuple:
    """Get market data for a symbol - returns (quote, bars)"""
    quote = bridge_state.market_data.get_quote(symbol)
    bars = bridge_state.market_data.get_bars(symbol, count)
    return quote, bars


def get_spread(symbol: str) -> float:
    """Get current spread for a symbol in points"""
    return bridge_state.market_data.get_spread(symbol)


def get_bid_ask(symbol: str) -> tuple:
    """Get current bid/ask for a symbol"""
    return bridge_state.market_data.get_bid_ask(symbol)


def get_available_symbols() -> List[str]:
    """Get list of symbols with market data"""
    return bridge_state.market_data.get_all_symbols()


def has_market_data(symbol: str) -> bool:
    """Check if we have market data for a symbol"""
    return bridge_state.market_data.get_quote(symbol) is not None


def get_closed_trades_for_learning(count: int = 50) -> List[Dict]:
    """Get recent closed trades for learning feedback"""
    return bridge_state.closed_trades.get_recent_trades(count)


def get_trade_stats() -> Dict:
    """Get trading statistics from closed trades"""
    return bridge_state.closed_trades.get_stats()


def wait_for_signal_result(signal_id: str, timeout: int = 30) -> Optional[Dict]:
    """Wait for a signal to be executed and return the result"""
    import time
    start = datetime.now()
    
    while (datetime.now() - start).seconds < timeout:
        for sig in bridge_state.executed_signals:
            if sig.id == signal_id:
                return sig.result
        time.sleep(0.5)
    
    return None


def create_app():
    """Factory function for Gunicorn"""
    return app


if __name__ == '__main__':
    port = int(os.environ.get('MT5_BRIDGE_PORT', 5000))
    use_gunicorn = os.environ.get('USE_GUNICORN', 'false').lower() == 'true'
    
    if use_gunicorn:
        logger.info(f"Starting MT5 Bridge API Server with Gunicorn on port {port}")
        try:
            import gunicorn.app.base
            
            class StandaloneApplication(gunicorn.app.base.BaseApplication):
                def __init__(self, app, options=None):
                    self.options = options or {}
                    self.application = app
                    super().__init__()
                
                def load_config(self):
                    for key, value in self.options.items():
                        if key in self.cfg.settings and value is not None:
                            self.cfg.set(key.lower(), value)
                
                def load(self):
                    return self.application
            
            options = {
                'bind': f'0.0.0.0:{port}',
                'workers': 2,
                'threads': 4,
                'worker_class': 'gthread',
                'timeout': 120,
                'keepalive': 5,
                'accesslog': '-',
                'errorlog': '-',
                'loglevel': 'info',
            }
            StandaloneApplication(app, options).run()
        except ImportError:
            logger.warning("Gunicorn not available, falling back to Flask dev server")
            app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
    else:
        logger.info(f"Starting MT5 Bridge API Server (Flask dev) on port {port}")
        app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
