"""
MT5 Bridge Module
Provides communication between the AI trading system and MT5 via Expert Advisor
"""

from .api_server import (
    send_trade_signal,
    get_mt5_balance,
    get_mt5_equity,
    get_mt5_positions,
    is_mt5_connected,
    wait_for_signal_result,
    bridge_state
)

from .mt5_broker_adapter import MT5BrokerAdapter

__all__ = [
    'send_trade_signal',
    'get_mt5_balance',
    'get_mt5_equity',
    'get_mt5_positions',
    'is_mt5_connected',
    'wait_for_signal_result',
    'bridge_state',
    'MT5BrokerAdapter'
]
