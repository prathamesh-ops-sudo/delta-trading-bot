"""
MT5 Bridge API Server
Receives trade signals from the AI trading system and serves them to the MT5 EA
Also receives account info and position updates from MT5
"""

import os
import json
import uuid
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, field, asdict
from threading import Lock
from flask import Flask, request, jsonify

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
    
    def get_pending_signals(self) -> List[Dict]:
        with self.lock:
            signals = []
            for sig in self.pending_signals:
                if not sig.executed:
                    signals.append({
                        'id': sig.id,
                        'symbol': sig.symbol,
                        'action': sig.action,
                        'volume': sig.volume,
                        'sl': sig.sl,
                        'tp': sig.tp,
                        'ticket': sig.ticket,
                        'comment': sig.comment
                    })
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
    signals = bridge_state.get_pending_signals()
    return jsonify({'signals': signals})


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
    
    bridge_state.mark_signal_executed(
        signal_id=data['signal_id'],
        success=data.get('success', False),
        ticket=data.get('ticket', 0),
        error=data.get('error', '')
    )
    
    return jsonify({'status': 'ok'})


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


if __name__ == '__main__':
    port = int(os.environ.get('MT5_BRIDGE_PORT', 5000))
    logger.info(f"Starting MT5 Bridge API Server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
