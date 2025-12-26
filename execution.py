"""
Institutional Execution Algorithms Module
Implements VWAP, TWAP, POV, and smart order routing
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import time
import threading
from queue import Queue, PriorityQueue

from config import config

logger = logging.getLogger(__name__)


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    PENDING = "pending"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class ExecutionAlgo(Enum):
    MARKET = "market"
    TWAP = "twap"
    VWAP = "vwap"
    POV = "pov"
    ICEBERG = "iceberg"
    SNIPER = "sniper"


@dataclass
class Order:
    """Order data structure"""
    id: str
    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    algo: ExecutionAlgo = ExecutionAlgo.MARKET
    parent_id: Optional[str] = None
    slippage: float = 0.0
    commission: float = 0.0


@dataclass
class ChildOrder:
    """Child order for algorithmic execution"""
    parent_id: str
    sequence: int
    quantity: float
    scheduled_time: datetime
    executed: bool = False
    fill_price: float = 0.0


@dataclass
class ExecutionReport:
    """Execution report for completed orders"""
    order_id: str
    symbol: str
    side: OrderSide
    requested_quantity: float
    filled_quantity: float
    avg_price: float
    vwap_benchmark: float
    slippage_bps: float
    total_cost: float
    execution_time_seconds: float
    num_child_orders: int
    algo_used: ExecutionAlgo
    timestamp: datetime = field(default_factory=datetime.now)


class SlippageModel:
    """Model for estimating and simulating slippage"""
    
    def __init__(self, base_slippage_bps: float = 1.0, 
                 impact_coefficient: float = 0.1):
        self.base_slippage_bps = base_slippage_bps
        self.impact_coefficient = impact_coefficient
    
    def estimate_slippage(self, quantity: float, avg_volume: float,
                          spread: float, volatility: float) -> float:
        """Estimate slippage for an order"""
        # Base slippage from spread
        spread_cost = spread / 2
        
        # Market impact based on order size relative to volume
        if avg_volume > 0:
            participation_rate = quantity / avg_volume
            market_impact = self.impact_coefficient * np.sqrt(participation_rate) * volatility
        else:
            market_impact = 0
        
        # Total slippage in price terms
        total_slippage = spread_cost + market_impact + (self.base_slippage_bps / 10000)
        
        return total_slippage
    
    def simulate_fill_price(self, mid_price: float, side: OrderSide,
                            quantity: float, avg_volume: float,
                            spread: float, volatility: float) -> float:
        """Simulate fill price with slippage"""
        slippage = self.estimate_slippage(quantity, avg_volume, spread, volatility)
        
        # Add random component
        random_slippage = np.random.normal(0, slippage * 0.2)
        total_slippage = slippage + random_slippage
        
        if side == OrderSide.BUY:
            return mid_price * (1 + total_slippage)
        else:
            return mid_price * (1 - total_slippage)


class TWAPExecutor:
    """Time-Weighted Average Price execution algorithm"""
    
    def __init__(self, duration_minutes: int = 60, num_slices: int = 12):
        self.duration_minutes = duration_minutes
        self.num_slices = num_slices
    
    def generate_schedule(self, order: Order, start_time: datetime = None) -> List[ChildOrder]:
        """Generate TWAP execution schedule"""
        start_time = start_time or datetime.now()
        
        # Calculate slice size and interval
        slice_quantity = order.quantity / self.num_slices
        interval_seconds = (self.duration_minutes * 60) / self.num_slices
        
        schedule = []
        for i in range(self.num_slices):
            scheduled_time = start_time + timedelta(seconds=i * interval_seconds)
            
            # Add small random jitter to avoid predictability
            jitter = np.random.uniform(-interval_seconds * 0.1, interval_seconds * 0.1)
            scheduled_time += timedelta(seconds=jitter)
            
            child = ChildOrder(
                parent_id=order.id,
                sequence=i,
                quantity=slice_quantity,
                scheduled_time=scheduled_time
            )
            schedule.append(child)
        
        return schedule
    
    def calculate_benchmark(self, prices: List[float]) -> float:
        """Calculate TWAP benchmark"""
        if not prices:
            return 0.0
        return np.mean(prices)


class VWAPExecutor:
    """Volume-Weighted Average Price execution algorithm"""
    
    def __init__(self, duration_minutes: int = 60, num_slices: int = 12):
        self.duration_minutes = duration_minutes
        self.num_slices = num_slices
    
    def generate_schedule(self, order: Order, volume_profile: List[float],
                          start_time: datetime = None) -> List[ChildOrder]:
        """Generate VWAP execution schedule based on volume profile"""
        start_time = start_time or datetime.now()
        
        # Normalize volume profile
        if not volume_profile or sum(volume_profile) == 0:
            # Use uniform distribution if no volume data
            volume_profile = [1.0] * self.num_slices
        
        total_volume = sum(volume_profile)
        volume_weights = [v / total_volume for v in volume_profile[:self.num_slices]]
        
        # Pad if necessary
        while len(volume_weights) < self.num_slices:
            volume_weights.append(1.0 / self.num_slices)
        
        interval_seconds = (self.duration_minutes * 60) / self.num_slices
        
        schedule = []
        for i in range(self.num_slices):
            scheduled_time = start_time + timedelta(seconds=i * interval_seconds)
            slice_quantity = order.quantity * volume_weights[i]
            
            # Add jitter
            jitter = np.random.uniform(-interval_seconds * 0.1, interval_seconds * 0.1)
            scheduled_time += timedelta(seconds=jitter)
            
            child = ChildOrder(
                parent_id=order.id,
                sequence=i,
                quantity=slice_quantity,
                scheduled_time=scheduled_time
            )
            schedule.append(child)
        
        return schedule
    
    def calculate_benchmark(self, prices: List[float], volumes: List[float]) -> float:
        """Calculate VWAP benchmark"""
        if not prices or not volumes or len(prices) != len(volumes):
            return np.mean(prices) if prices else 0.0
        
        total_value = sum(p * v for p, v in zip(prices, volumes))
        total_volume = sum(volumes)
        
        return total_value / total_volume if total_volume > 0 else 0.0


class POVExecutor:
    """Percentage of Volume execution algorithm"""
    
    def __init__(self, target_participation: float = 0.1, max_participation: float = 0.25):
        self.target_participation = target_participation
        self.max_participation = max_participation
    
    def calculate_slice_size(self, remaining_quantity: float, 
                             current_volume: float) -> float:
        """Calculate next slice size based on current market volume"""
        target_size = current_volume * self.target_participation
        max_size = current_volume * self.max_participation
        
        # Don't exceed remaining quantity
        slice_size = min(target_size, remaining_quantity)
        slice_size = min(slice_size, max_size)
        
        return max(0, slice_size)
    
    def should_execute(self, elapsed_volume: float, executed_quantity: float,
                       total_quantity: float) -> bool:
        """Determine if we should execute based on volume participation"""
        if executed_quantity >= total_quantity:
            return False
        
        target_executed = elapsed_volume * self.target_participation
        return executed_quantity < target_executed


class IcebergExecutor:
    """Iceberg order execution - shows only small visible quantity"""
    
    def __init__(self, visible_ratio: float = 0.1, min_visible: float = 0.01):
        self.visible_ratio = visible_ratio
        self.min_visible = min_visible
    
    def get_visible_quantity(self, total_quantity: float) -> float:
        """Get the visible (displayed) quantity"""
        visible = total_quantity * self.visible_ratio
        return max(visible, self.min_visible)
    
    def generate_slices(self, order: Order) -> List[ChildOrder]:
        """Generate iceberg slices"""
        visible_qty = self.get_visible_quantity(order.quantity)
        num_slices = int(np.ceil(order.quantity / visible_qty))
        
        slices = []
        remaining = order.quantity
        
        for i in range(num_slices):
            slice_qty = min(visible_qty, remaining)
            slices.append(ChildOrder(
                parent_id=order.id,
                sequence=i,
                quantity=slice_qty,
                scheduled_time=datetime.now()  # Execute immediately when previous fills
            ))
            remaining -= slice_qty
        
        return slices


class SniperExecutor:
    """Sniper algorithm - waits for favorable price then executes quickly"""
    
    def __init__(self, price_improvement_threshold: float = 0.0002,
                 max_wait_seconds: int = 300):
        self.price_improvement_threshold = price_improvement_threshold
        self.max_wait_seconds = max_wait_seconds
    
    def should_execute(self, current_price: float, reference_price: float,
                       side: OrderSide, elapsed_seconds: float) -> bool:
        """Determine if conditions are favorable for execution"""
        # Force execute if max wait exceeded
        if elapsed_seconds >= self.max_wait_seconds:
            return True
        
        # Check for price improvement
        if side == OrderSide.BUY:
            improvement = (reference_price - current_price) / reference_price
        else:
            improvement = (current_price - reference_price) / reference_price
        
        return improvement >= self.price_improvement_threshold


class ExecutionEngine:
    """Main execution engine coordinating all algorithms"""
    
    def __init__(self, broker_adapter=None):
        self.broker = broker_adapter
        self.slippage_model = SlippageModel()
        
        # Executors
        self.twap = TWAPExecutor()
        self.vwap = VWAPExecutor()
        self.pov = POVExecutor()
        self.iceberg = IcebergExecutor()
        self.sniper = SniperExecutor()
        
        # Order tracking
        self.active_orders: Dict[str, Order] = {}
        self.child_orders: Dict[str, List[ChildOrder]] = {}
        self.execution_reports: List[ExecutionReport] = []
        
        # Execution queue
        self.order_queue = PriorityQueue()
        self.is_running = False
        self._execution_thread = None
        
        # Statistics
        self.total_slippage = 0.0
        self.total_commission = 0.0
        self.orders_executed = 0
    
    def submit_order(self, order: Order, algo: ExecutionAlgo = ExecutionAlgo.MARKET,
                     algo_params: Dict = None) -> str:
        """Submit order for execution"""
        algo_params = algo_params or {}
        order.algo = algo
        
        self.active_orders[order.id] = order
        
        if algo == ExecutionAlgo.MARKET:
            # Execute immediately
            self._execute_market_order(order)
        
        elif algo == ExecutionAlgo.TWAP:
            duration = algo_params.get('duration_minutes', 60)
            slices = algo_params.get('num_slices', 12)
            self.twap.duration_minutes = duration
            self.twap.num_slices = slices
            
            schedule = self.twap.generate_schedule(order)
            self.child_orders[order.id] = schedule
            self._queue_child_orders(schedule)
        
        elif algo == ExecutionAlgo.VWAP:
            duration = algo_params.get('duration_minutes', 60)
            volume_profile = algo_params.get('volume_profile', [])
            self.vwap.duration_minutes = duration
            
            schedule = self.vwap.generate_schedule(order, volume_profile)
            self.child_orders[order.id] = schedule
            self._queue_child_orders(schedule)
        
        elif algo == ExecutionAlgo.POV:
            participation = algo_params.get('target_participation', 0.1)
            self.pov.target_participation = participation
            # POV requires continuous monitoring - handled separately
            self._start_pov_execution(order)
        
        elif algo == ExecutionAlgo.ICEBERG:
            slices = self.iceberg.generate_slices(order)
            self.child_orders[order.id] = slices
            # Execute first slice immediately
            if slices:
                self._execute_child_order(slices[0])
        
        elif algo == ExecutionAlgo.SNIPER:
            # Sniper waits for favorable conditions
            self._start_sniper_execution(order, algo_params)
        
        logger.info(f"Order {order.id} submitted with {algo.value} algorithm")
        return order.id
    
    def _execute_market_order(self, order: Order):
        """Execute market order immediately"""
        if self.broker:
            # Real execution through broker
            result = self.broker.execute_order(order)
            self._process_fill(order, result)
        else:
            # Simulated execution
            self._simulate_fill(order, order.quantity)
    
    def _execute_child_order(self, child: ChildOrder):
        """Execute a child order"""
        parent = self.active_orders.get(child.parent_id)
        if not parent:
            return
        
        if self.broker:
            # Create child order for broker
            child_order = Order(
                id=f"{child.parent_id}_{child.sequence}",
                symbol=parent.symbol,
                side=parent.side,
                quantity=child.quantity,
                order_type=OrderType.MARKET,
                parent_id=child.parent_id
            )
            result = self.broker.execute_order(child_order)
            child.fill_price = result.get('fill_price', 0)
        else:
            # Simulated execution
            child.fill_price = self._simulate_child_fill(parent, child.quantity)
        
        child.executed = True
        
        # Update parent order
        parent.filled_quantity += child.quantity
        if parent.avg_fill_price == 0:
            parent.avg_fill_price = child.fill_price
        else:
            # Weighted average
            total_qty = parent.filled_quantity
            parent.avg_fill_price = (
                (parent.avg_fill_price * (total_qty - child.quantity) + 
                 child.fill_price * child.quantity) / total_qty
            )
        
        parent.updated_at = datetime.now()
        
        # Check if fully filled
        if parent.filled_quantity >= parent.quantity:
            parent.status = OrderStatus.FILLED
            self._generate_execution_report(parent)
        else:
            parent.status = OrderStatus.PARTIAL
    
    def _simulate_fill(self, order: Order, quantity: float):
        """Simulate order fill with slippage"""
        # Get market data (simplified)
        mid_price = order.limit_price or 1.1000  # Default price
        spread = 0.00010  # 1 pip spread
        volatility = 0.01
        avg_volume = 1000000
        
        fill_price = self.slippage_model.simulate_fill_price(
            mid_price, order.side, quantity, avg_volume, spread, volatility
        )
        
        order.filled_quantity = quantity
        order.avg_fill_price = fill_price
        order.slippage = abs(fill_price - mid_price) / mid_price
        order.commission = quantity * 0.00001  # Simplified commission
        order.status = OrderStatus.FILLED
        order.updated_at = datetime.now()
        
        self.total_slippage += order.slippage * quantity
        self.total_commission += order.commission
        self.orders_executed += 1
        
        self._generate_execution_report(order)
    
    def _simulate_child_fill(self, parent: Order, quantity: float) -> float:
        """Simulate child order fill"""
        mid_price = parent.limit_price or 1.1000
        spread = 0.00010
        volatility = 0.01
        avg_volume = 1000000
        
        return self.slippage_model.simulate_fill_price(
            mid_price, parent.side, quantity, avg_volume, spread, volatility
        )
    
    def _queue_child_orders(self, schedule: List[ChildOrder]):
        """Add child orders to execution queue"""
        for child in schedule:
            # Priority queue uses scheduled time as priority
            self.order_queue.put((child.scheduled_time.timestamp(), child))
    
    def _start_pov_execution(self, order: Order):
        """Start POV execution in background"""
        def pov_loop():
            remaining = order.quantity
            start_time = datetime.now()
            
            while remaining > 0 and self.is_running:
                # Simulate market volume (in real system, get from market data)
                current_volume = np.random.uniform(10000, 50000)
                
                slice_size = self.pov.calculate_slice_size(remaining, current_volume)
                
                if slice_size > 0:
                    child = ChildOrder(
                        parent_id=order.id,
                        sequence=len(self.child_orders.get(order.id, [])),
                        quantity=slice_size,
                        scheduled_time=datetime.now()
                    )
                    
                    if order.id not in self.child_orders:
                        self.child_orders[order.id] = []
                    self.child_orders[order.id].append(child)
                    
                    self._execute_child_order(child)
                    remaining -= slice_size
                
                time.sleep(5)  # Check every 5 seconds
        
        thread = threading.Thread(target=pov_loop, daemon=True)
        thread.start()
    
    def _start_sniper_execution(self, order: Order, params: Dict):
        """Start sniper execution"""
        reference_price = params.get('reference_price', order.limit_price or 1.1000)
        
        def sniper_loop():
            start_time = datetime.now()
            
            while self.is_running:
                elapsed = (datetime.now() - start_time).total_seconds()
                
                # Simulate current price
                current_price = reference_price * (1 + np.random.normal(0, 0.0005))
                
                if self.sniper.should_execute(current_price, reference_price, 
                                              order.side, elapsed):
                    self._simulate_fill(order, order.quantity)
                    break
                
                time.sleep(1)
        
        thread = threading.Thread(target=sniper_loop, daemon=True)
        thread.start()
    
    def _generate_execution_report(self, order: Order):
        """Generate execution report for completed order"""
        # Calculate VWAP benchmark (simplified)
        vwap_benchmark = order.limit_price or order.avg_fill_price
        
        slippage_bps = (order.avg_fill_price - vwap_benchmark) / vwap_benchmark * 10000
        if order.side == OrderSide.SELL:
            slippage_bps = -slippage_bps
        
        report = ExecutionReport(
            order_id=order.id,
            symbol=order.symbol,
            side=order.side,
            requested_quantity=order.quantity,
            filled_quantity=order.filled_quantity,
            avg_price=order.avg_fill_price,
            vwap_benchmark=vwap_benchmark,
            slippage_bps=slippage_bps,
            total_cost=order.filled_quantity * order.avg_fill_price + order.commission,
            execution_time_seconds=(order.updated_at - order.created_at).total_seconds(),
            num_child_orders=len(self.child_orders.get(order.id, [])),
            algo_used=order.algo
        )
        
        self.execution_reports.append(report)
        logger.info(f"Execution report generated for {order.id}: "
                   f"filled {order.filled_quantity} @ {order.avg_fill_price:.5f}, "
                   f"slippage: {slippage_bps:.2f} bps")
    
    def start(self):
        """Start execution engine"""
        self.is_running = True
        
        def execution_loop():
            while self.is_running:
                try:
                    if not self.order_queue.empty():
                        priority, child = self.order_queue.get(timeout=1)
                        
                        # Wait until scheduled time
                        now = datetime.now().timestamp()
                        if priority > now:
                            time.sleep(priority - now)
                        
                        if not child.executed:
                            self._execute_child_order(child)
                    else:
                        time.sleep(0.1)
                except Exception as e:
                    logger.error(f"Execution loop error: {e}")
        
        self._execution_thread = threading.Thread(target=execution_loop, daemon=True)
        self._execution_thread.start()
        logger.info("Execution engine started")
    
    def stop(self):
        """Stop execution engine"""
        self.is_running = False
        if self._execution_thread:
            self._execution_thread.join(timeout=5)
        logger.info("Execution engine stopped")
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an active order"""
        if order_id in self.active_orders:
            order = self.active_orders[order_id]
            if order.status in [OrderStatus.PENDING, OrderStatus.PARTIAL]:
                order.status = OrderStatus.CANCELLED
                order.updated_at = datetime.now()
                logger.info(f"Order {order_id} cancelled")
                return True
        return False
    
    def get_order_status(self, order_id: str) -> Optional[Order]:
        """Get order status"""
        return self.active_orders.get(order_id)
    
    def get_execution_statistics(self) -> Dict:
        """Get execution statistics"""
        if not self.execution_reports:
            return {}
        
        slippages = [r.slippage_bps for r in self.execution_reports]
        exec_times = [r.execution_time_seconds for r in self.execution_reports]
        
        return {
            'total_orders': len(self.execution_reports),
            'avg_slippage_bps': np.mean(slippages),
            'max_slippage_bps': np.max(slippages),
            'min_slippage_bps': np.min(slippages),
            'avg_execution_time': np.mean(exec_times),
            'total_commission': self.total_commission,
            'algo_breakdown': self._get_algo_breakdown()
        }
    
    def _get_algo_breakdown(self) -> Dict:
        """Get breakdown by algorithm"""
        breakdown = {}
        for report in self.execution_reports:
            algo = report.algo_used.value
            if algo not in breakdown:
                breakdown[algo] = {'count': 0, 'avg_slippage': []}
            breakdown[algo]['count'] += 1
            breakdown[algo]['avg_slippage'].append(report.slippage_bps)
        
        for algo in breakdown:
            breakdown[algo]['avg_slippage'] = np.mean(breakdown[algo]['avg_slippage'])
        
        return breakdown
    
    def select_best_algo(self, order: Order, market_conditions: Dict) -> ExecutionAlgo:
        """Select best execution algorithm based on conditions"""
        urgency = market_conditions.get('urgency', 'normal')
        volatility = market_conditions.get('volatility', 'normal')
        order_size_pct = market_conditions.get('order_size_pct', 0.01)
        
        # Large orders in normal conditions -> VWAP
        if order_size_pct > 0.05 and volatility == 'normal':
            return ExecutionAlgo.VWAP
        
        # High urgency -> Market
        if urgency == 'high':
            return ExecutionAlgo.MARKET
        
        # High volatility -> Sniper (wait for good price)
        if volatility == 'high' and urgency != 'high':
            return ExecutionAlgo.SNIPER
        
        # Medium size orders -> TWAP
        if order_size_pct > 0.02:
            return ExecutionAlgo.TWAP
        
        # Small orders -> Market
        return ExecutionAlgo.MARKET


# Singleton instance
execution_engine = ExecutionEngine()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Execution Algorithms Module...")
    
    # Test slippage model
    slippage = SlippageModel()
    est_slippage = slippage.estimate_slippage(
        quantity=10000,
        avg_volume=1000000,
        spread=0.0001,
        volatility=0.01
    )
    print(f"\nEstimated slippage: {est_slippage:.6f}")
    
    # Test TWAP
    print("\nTesting TWAP...")
    twap = TWAPExecutor(duration_minutes=30, num_slices=6)
    order = Order(
        id="test_001",
        symbol="EURUSD",
        side=OrderSide.BUY,
        quantity=1.0
    )
    schedule = twap.generate_schedule(order)
    print(f"Generated {len(schedule)} TWAP slices:")
    for child in schedule[:3]:
        print(f"  Slice {child.sequence}: {child.quantity:.4f} @ {child.scheduled_time}")
    
    # Test VWAP
    print("\nTesting VWAP...")
    vwap = VWAPExecutor(duration_minutes=30, num_slices=6)
    volume_profile = [100, 150, 200, 180, 120, 80]  # U-shaped volume
    schedule = vwap.generate_schedule(order, volume_profile)
    print(f"Generated {len(schedule)} VWAP slices:")
    for child in schedule:
        print(f"  Slice {child.sequence}: {child.quantity:.4f}")
    
    # Test execution engine
    print("\nTesting Execution Engine...")
    engine = ExecutionEngine()
    engine.start()
    
    # Submit market order
    market_order = Order(
        id="market_001",
        symbol="EURUSD",
        side=OrderSide.BUY,
        quantity=0.1,
        limit_price=1.1000
    )
    engine.submit_order(market_order, ExecutionAlgo.MARKET)
    
    time.sleep(1)
    
    # Check status
    status = engine.get_order_status("market_001")
    print(f"Market order status: {status.status.value if status else 'Not found'}")
    
    # Get statistics
    stats = engine.get_execution_statistics()
    print(f"Execution stats: {stats}")
    
    engine.stop()
    
    print("\nExecution Algorithms Module test complete!")
