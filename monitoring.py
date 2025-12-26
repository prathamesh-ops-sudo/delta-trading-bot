"""
Monitoring, Logging, and Alerting System
Provides comprehensive system monitoring, performance tracking, and notifications
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import logging.handlers
import json
import os
import threading
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests

from config import config

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertChannel(Enum):
    LOG = "log"
    EMAIL = "email"
    SMS = "sms"
    WEBHOOK = "webhook"
    TELEGRAM = "telegram"


@dataclass
class Alert:
    """Alert data structure"""
    id: str
    timestamp: datetime
    level: AlertLevel
    category: str
    message: str
    details: Dict = field(default_factory=dict)
    acknowledged: bool = False
    channels_sent: List[AlertChannel] = field(default_factory=list)


@dataclass
class PerformanceMetric:
    """Performance metric data point"""
    timestamp: datetime
    name: str
    value: float
    unit: str = ""
    tags: Dict = field(default_factory=dict)


class LoggingSetup:
    """Configure logging for the trading system"""
    
    @staticmethod
    def setup(log_dir: str = "./logs", log_level: int = logging.INFO,
              max_bytes: int = 10*1024*1024, backup_count: int = 5):
        """Setup logging configuration"""
        os.makedirs(log_dir, exist_ok=True)
        
        # Root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)
        
        # Clear existing handlers
        root_logger.handlers = []
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_format)
        root_logger.addHandler(console_handler)
        
        # File handler - main log
        main_log_path = os.path.join(log_dir, 'trading.log')
        file_handler = logging.handlers.RotatingFileHandler(
            main_log_path, maxBytes=max_bytes, backupCount=backup_count
        )
        file_handler.setLevel(log_level)
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_format)
        root_logger.addHandler(file_handler)
        
        # Error log - separate file for errors
        error_log_path = os.path.join(log_dir, 'errors.log')
        error_handler = logging.handlers.RotatingFileHandler(
            error_log_path, maxBytes=max_bytes, backupCount=backup_count
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_format)
        root_logger.addHandler(error_handler)
        
        # Trade log - separate file for trades
        trade_logger = logging.getLogger('trades')
        trade_log_path = os.path.join(log_dir, 'trades.log')
        trade_handler = logging.handlers.RotatingFileHandler(
            trade_log_path, maxBytes=max_bytes, backupCount=backup_count
        )
        trade_handler.setLevel(logging.INFO)
        trade_format = logging.Formatter('%(asctime)s - %(message)s')
        trade_handler.setFormatter(trade_format)
        trade_logger.addHandler(trade_handler)
        
        logger.info(f"Logging configured: {log_dir}")
        
        return root_logger


class AlertManager:
    """Manages alerts and notifications"""
    
    def __init__(self):
        self.alerts: List[Alert] = []
        self.alert_rules: List[Dict] = []
        self.channels: Dict[AlertChannel, Dict] = {}
        self.alert_count = 0
        self.cooldown_tracker: Dict[str, datetime] = {}
        self.cooldown_minutes = 15
        
        # Rate limiting
        self.alerts_per_hour: Dict[str, int] = {}
        self.max_alerts_per_hour = 10
    
    def configure_channel(self, channel: AlertChannel, config: Dict):
        """Configure an alert channel"""
        self.channels[channel] = config
        logger.info(f"Alert channel configured: {channel.value}")
    
    def add_rule(self, category: str, condition: Callable, level: AlertLevel,
                 message_template: str, channels: List[AlertChannel] = None):
        """Add an alert rule"""
        self.alert_rules.append({
            'category': category,
            'condition': condition,
            'level': level,
            'message_template': message_template,
            'channels': channels or [AlertChannel.LOG]
        })
    
    def check_rules(self, context: Dict):
        """Check all alert rules against current context"""
        for rule in self.alert_rules:
            try:
                if rule['condition'](context):
                    message = rule['message_template'].format(**context)
                    self.create_alert(
                        level=rule['level'],
                        category=rule['category'],
                        message=message,
                        details=context,
                        channels=rule['channels']
                    )
            except Exception as e:
                logger.error(f"Error checking alert rule: {e}")
    
    def create_alert(self, level: AlertLevel, category: str, message: str,
                     details: Dict = None, channels: List[AlertChannel] = None):
        """Create and send an alert"""
        # Check cooldown
        cooldown_key = f"{category}:{message[:50]}"
        if cooldown_key in self.cooldown_tracker:
            last_alert = self.cooldown_tracker[cooldown_key]
            if datetime.now() - last_alert < timedelta(minutes=self.cooldown_minutes):
                return None
        
        # Check rate limit
        hour_key = datetime.now().strftime('%Y%m%d%H')
        if hour_key not in self.alerts_per_hour:
            self.alerts_per_hour = {hour_key: 0}
        
        if self.alerts_per_hour[hour_key] >= self.max_alerts_per_hour:
            logger.warning("Alert rate limit reached")
            return None
        
        # Create alert
        self.alert_count += 1
        alert = Alert(
            id=f"ALT-{self.alert_count:06d}",
            timestamp=datetime.now(),
            level=level,
            category=category,
            message=message,
            details=details or {}
        )
        
        self.alerts.append(alert)
        self.cooldown_tracker[cooldown_key] = datetime.now()
        self.alerts_per_hour[hour_key] += 1
        
        # Send to channels
        channels = channels or [AlertChannel.LOG]
        for channel in channels:
            self._send_to_channel(alert, channel)
            alert.channels_sent.append(channel)
        
        return alert
    
    def _send_to_channel(self, alert: Alert, channel: AlertChannel):
        """Send alert to specific channel"""
        try:
            if channel == AlertChannel.LOG:
                self._send_to_log(alert)
            elif channel == AlertChannel.EMAIL:
                self._send_to_email(alert)
            elif channel == AlertChannel.WEBHOOK:
                self._send_to_webhook(alert)
            elif channel == AlertChannel.TELEGRAM:
                self._send_to_telegram(alert)
        except Exception as e:
            logger.error(f"Failed to send alert to {channel.value}: {e}")
    
    def _send_to_log(self, alert: Alert):
        """Send alert to log"""
        log_message = f"[{alert.level.value.upper()}] {alert.category}: {alert.message}"
        
        if alert.level == AlertLevel.CRITICAL:
            logger.critical(log_message)
        elif alert.level == AlertLevel.ERROR:
            logger.error(log_message)
        elif alert.level == AlertLevel.WARNING:
            logger.warning(log_message)
        else:
            logger.info(log_message)
    
    def _send_to_email(self, alert: Alert):
        """Send alert via email"""
        if AlertChannel.EMAIL not in self.channels:
            return
        
        config = self.channels[AlertChannel.EMAIL]
        
        try:
            msg = MIMEMultipart()
            msg['From'] = config.get('from_email')
            msg['To'] = config.get('to_email')
            msg['Subject'] = f"[{alert.level.value.upper()}] Trading Alert: {alert.category}"
            
            body = f"""
Trading System Alert
====================
Time: {alert.timestamp}
Level: {alert.level.value}
Category: {alert.category}

Message:
{alert.message}

Details:
{json.dumps(alert.details, indent=2, default=str)}
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(config.get('smtp_server'), config.get('smtp_port', 587))
            server.starttls()
            server.login(config.get('username'), config.get('password'))
            server.send_message(msg)
            server.quit()
            
            logger.debug(f"Email alert sent: {alert.id}")
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
    
    def _send_to_webhook(self, alert: Alert):
        """Send alert to webhook"""
        if AlertChannel.WEBHOOK not in self.channels:
            return
        
        config = self.channels[AlertChannel.WEBHOOK]
        
        try:
            payload = {
                'id': alert.id,
                'timestamp': alert.timestamp.isoformat(),
                'level': alert.level.value,
                'category': alert.category,
                'message': alert.message,
                'details': alert.details
            }
            
            response = requests.post(
                config.get('url'),
                json=payload,
                headers=config.get('headers', {}),
                timeout=10
            )
            
            if response.status_code != 200:
                logger.warning(f"Webhook returned {response.status_code}")
            
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")
    
    def _send_to_telegram(self, alert: Alert):
        """Send alert to Telegram"""
        if AlertChannel.TELEGRAM not in self.channels:
            return
        
        config = self.channels[AlertChannel.TELEGRAM]
        
        try:
            message = f"""
🚨 *Trading Alert*
Level: {alert.level.value.upper()}
Category: {alert.category}

{alert.message}
            """
            
            url = f"https://api.telegram.org/bot{config.get('bot_token')}/sendMessage"
            payload = {
                'chat_id': config.get('chat_id'),
                'text': message,
                'parse_mode': 'Markdown'
            }
            
            response = requests.post(url, json=payload, timeout=10)
            
            if response.status_code != 200:
                logger.warning(f"Telegram API returned {response.status_code}")
            
        except Exception as e:
            logger.error(f"Failed to send Telegram alert: {e}")
    
    def get_recent_alerts(self, hours: int = 24, level: AlertLevel = None) -> List[Alert]:
        """Get recent alerts"""
        cutoff = datetime.now() - timedelta(hours=hours)
        alerts = [a for a in self.alerts if a.timestamp > cutoff]
        
        if level:
            alerts = [a for a in alerts if a.level == level]
        
        return alerts
    
    def acknowledge_alert(self, alert_id: str):
        """Acknowledge an alert"""
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.acknowledged = True
                return True
        return False


class PerformanceMonitor:
    """Monitors system and trading performance"""
    
    def __init__(self):
        self.metrics: List[PerformanceMetric] = []
        self.metric_aggregates: Dict[str, Dict] = {}
        self.thresholds: Dict[str, Dict] = {}
        self._running = False
        self._thread = None
    
    def record_metric(self, name: str, value: float, unit: str = "", 
                      tags: Dict = None):
        """Record a performance metric"""
        metric = PerformanceMetric(
            timestamp=datetime.now(),
            name=name,
            value=value,
            unit=unit,
            tags=tags or {}
        )
        
        self.metrics.append(metric)
        
        # Keep only last 10000 metrics
        if len(self.metrics) > 10000:
            self.metrics = self.metrics[-10000:]
        
        # Update aggregates
        self._update_aggregates(metric)
    
    def _update_aggregates(self, metric: PerformanceMetric):
        """Update metric aggregates"""
        name = metric.name
        
        if name not in self.metric_aggregates:
            self.metric_aggregates[name] = {
                'count': 0,
                'sum': 0,
                'min': float('inf'),
                'max': float('-inf'),
                'values': []
            }
        
        agg = self.metric_aggregates[name]
        agg['count'] += 1
        agg['sum'] += metric.value
        agg['min'] = min(agg['min'], metric.value)
        agg['max'] = max(agg['max'], metric.value)
        agg['values'].append(metric.value)
        
        # Keep only last 1000 values for percentile calculations
        if len(agg['values']) > 1000:
            agg['values'] = agg['values'][-1000:]
    
    def set_threshold(self, metric_name: str, warning: float = None, 
                      critical: float = None, comparison: str = 'gt'):
        """Set threshold for a metric"""
        self.thresholds[metric_name] = {
            'warning': warning,
            'critical': critical,
            'comparison': comparison  # 'gt', 'lt', 'eq'
        }
    
    def check_thresholds(self) -> List[Dict]:
        """Check all thresholds and return violations"""
        violations = []
        
        for name, threshold in self.thresholds.items():
            if name not in self.metric_aggregates:
                continue
            
            current_value = self.metric_aggregates[name]['values'][-1] if self.metric_aggregates[name]['values'] else 0
            
            comparison = threshold['comparison']
            
            # Check critical
            if threshold['critical'] is not None:
                if self._compare(current_value, threshold['critical'], comparison):
                    violations.append({
                        'metric': name,
                        'level': 'critical',
                        'value': current_value,
                        'threshold': threshold['critical']
                    })
                    continue
            
            # Check warning
            if threshold['warning'] is not None:
                if self._compare(current_value, threshold['warning'], comparison):
                    violations.append({
                        'metric': name,
                        'level': 'warning',
                        'value': current_value,
                        'threshold': threshold['warning']
                    })
        
        return violations
    
    def _compare(self, value: float, threshold: float, comparison: str) -> bool:
        """Compare value against threshold"""
        if comparison == 'gt':
            return value > threshold
        elif comparison == 'lt':
            return value < threshold
        elif comparison == 'eq':
            return value == threshold
        return False
    
    def get_metric_stats(self, name: str) -> Dict:
        """Get statistics for a metric"""
        if name not in self.metric_aggregates:
            return {}
        
        agg = self.metric_aggregates[name]
        values = agg['values']
        
        return {
            'count': agg['count'],
            'mean': agg['sum'] / agg['count'] if agg['count'] > 0 else 0,
            'min': agg['min'],
            'max': agg['max'],
            'std': np.std(values) if values else 0,
            'p50': np.percentile(values, 50) if values else 0,
            'p95': np.percentile(values, 95) if values else 0,
            'p99': np.percentile(values, 99) if values else 0
        }
    
    def get_metrics_summary(self) -> Dict:
        """Get summary of all metrics"""
        return {
            name: self.get_metric_stats(name)
            for name in self.metric_aggregates
        }


class SystemMonitor:
    """Monitors system resources and health"""
    
    def __init__(self):
        self.health_checks: List[Callable] = []
        self.last_health_status: Dict = {}
        self._running = False
        self._thread = None
    
    def add_health_check(self, name: str, check_func: Callable):
        """Add a health check function"""
        self.health_checks.append({
            'name': name,
            'check': check_func
        })
    
    def run_health_checks(self) -> Dict:
        """Run all health checks"""
        results = {}
        
        for check in self.health_checks:
            try:
                result = check['check']()
                results[check['name']] = {
                    'status': 'healthy' if result else 'unhealthy',
                    'timestamp': datetime.now().isoformat()
                }
            except Exception as e:
                results[check['name']] = {
                    'status': 'error',
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
        
        self.last_health_status = results
        return results
    
    def get_system_metrics(self) -> Dict:
        """Get system resource metrics"""
        try:
            import psutil
            
            return {
                'cpu_percent': psutil.cpu_percent(interval=1),
                'memory_percent': psutil.virtual_memory().percent,
                'memory_available_mb': psutil.virtual_memory().available / (1024 * 1024),
                'disk_percent': psutil.disk_usage('/').percent,
                'disk_free_gb': psutil.disk_usage('/').free / (1024 * 1024 * 1024),
                'network_connections': len(psutil.net_connections()),
                'process_count': len(psutil.pids()),
                'timestamp': datetime.now().isoformat()
            }
        except ImportError:
            return {
                'error': 'psutil not available',
                'timestamp': datetime.now().isoformat()
            }
    
    def start_monitoring(self, interval: int = 60):
        """Start background monitoring"""
        self._running = True
        
        def monitor_loop():
            while self._running:
                try:
                    self.run_health_checks()
                    time.sleep(interval)
                except Exception as e:
                    logger.error(f"Monitoring error: {e}")
                    time.sleep(interval)
        
        self._thread = threading.Thread(target=monitor_loop, daemon=True)
        self._thread.start()
        logger.info("System monitoring started")
    
    def stop_monitoring(self):
        """Stop background monitoring"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("System monitoring stopped")


class TradingMonitor:
    """Monitors trading-specific metrics"""
    
    def __init__(self, alert_manager: AlertManager, perf_monitor: PerformanceMonitor):
        self.alert_manager = alert_manager
        self.perf_monitor = perf_monitor
        
        # Trading metrics
        self.daily_trades = 0
        self.daily_profit = 0.0
        self.daily_wins = 0
        self.daily_losses = 0
        self.current_drawdown = 0.0
        self.peak_equity = 0.0
        
        # Setup default alert rules
        self._setup_default_rules()
    
    def _setup_default_rules(self):
        """Setup default alert rules"""
        # Drawdown alert
        self.alert_manager.add_rule(
            category='risk',
            condition=lambda ctx: ctx.get('drawdown_pct', 0) > 0.10,
            level=AlertLevel.WARNING,
            message_template="Drawdown exceeds 10%: {drawdown_pct:.2%}",
            channels=[AlertChannel.LOG]
        )
        
        self.alert_manager.add_rule(
            category='risk',
            condition=lambda ctx: ctx.get('drawdown_pct', 0) > 0.20,
            level=AlertLevel.CRITICAL,
            message_template="CRITICAL: Drawdown exceeds 20%: {drawdown_pct:.2%}",
            channels=[AlertChannel.LOG]
        )
        
        # Win rate alert
        self.alert_manager.add_rule(
            category='performance',
            condition=lambda ctx: ctx.get('win_rate', 1) < 0.40 and ctx.get('total_trades', 0) > 10,
            level=AlertLevel.WARNING,
            message_template="Win rate below 40%: {win_rate:.2%} ({total_trades} trades)",
            channels=[AlertChannel.LOG]
        )
        
        # Connection alert
        self.alert_manager.add_rule(
            category='system',
            condition=lambda ctx: not ctx.get('broker_connected', True),
            level=AlertLevel.ERROR,
            message_template="Broker disconnected!",
            channels=[AlertChannel.LOG]
        )
    
    def record_trade(self, trade: Dict):
        """Record a completed trade"""
        profit = trade.get('profit', 0)
        
        self.daily_trades += 1
        self.daily_profit += profit
        
        if profit > 0:
            self.daily_wins += 1
        else:
            self.daily_losses += 1
        
        # Record metrics
        self.perf_monitor.record_metric('trade_profit', profit, 'USD')
        self.perf_monitor.record_metric('trade_duration', trade.get('duration_minutes', 0), 'minutes')
        
        # Log trade
        trade_logger = logging.getLogger('trades')
        trade_logger.info(json.dumps(trade, default=str))
    
    def update_equity(self, equity: float, balance: float):
        """Update equity tracking"""
        if equity > self.peak_equity:
            self.peak_equity = equity
        
        if self.peak_equity > 0:
            self.current_drawdown = (self.peak_equity - equity) / self.peak_equity
        
        # Record metrics
        self.perf_monitor.record_metric('equity', equity, 'USD')
        self.perf_monitor.record_metric('balance', balance, 'USD')
        self.perf_monitor.record_metric('drawdown', self.current_drawdown * 100, '%')
        
        # Check alerts
        win_rate = self.daily_wins / self.daily_trades if self.daily_trades > 0 else 0.5
        
        self.alert_manager.check_rules({
            'drawdown_pct': self.current_drawdown,
            'win_rate': win_rate,
            'total_trades': self.daily_trades,
            'daily_profit': self.daily_profit
        })
    
    def reset_daily(self):
        """Reset daily metrics"""
        self.daily_trades = 0
        self.daily_profit = 0.0
        self.daily_wins = 0
        self.daily_losses = 0
        logger.info("Daily trading metrics reset")
    
    def get_daily_summary(self) -> Dict:
        """Get daily trading summary"""
        win_rate = self.daily_wins / self.daily_trades if self.daily_trades > 0 else 0
        
        return {
            'date': datetime.now().date().isoformat(),
            'total_trades': self.daily_trades,
            'winning_trades': self.daily_wins,
            'losing_trades': self.daily_losses,
            'win_rate': win_rate,
            'daily_profit': self.daily_profit,
            'current_drawdown': self.current_drawdown,
            'peak_equity': self.peak_equity
        }


class MonitoringSystem:
    """Main monitoring system coordinator"""
    
    def __init__(self):
        # Setup logging
        LoggingSetup.setup(log_dir="./logs")
        
        # Initialize components
        self.alert_manager = AlertManager()
        self.perf_monitor = PerformanceMonitor()
        self.system_monitor = SystemMonitor()
        self.trading_monitor = TradingMonitor(self.alert_manager, self.perf_monitor)
        
        # Setup default health checks
        self._setup_health_checks()
        
        # Setup default thresholds
        self._setup_thresholds()
    
    def _setup_health_checks(self):
        """Setup default health checks"""
        # Memory check
        def check_memory():
            try:
                import psutil
                return psutil.virtual_memory().percent < 90
            except:
                return True
        
        self.system_monitor.add_health_check('memory', check_memory)
        
        # Disk check
        def check_disk():
            try:
                import psutil
                return psutil.disk_usage('/').percent < 90
            except:
                return True
        
        self.system_monitor.add_health_check('disk', check_disk)
    
    def _setup_thresholds(self):
        """Setup default metric thresholds"""
        self.perf_monitor.set_threshold('drawdown', warning=10, critical=20, comparison='gt')
        self.perf_monitor.set_threshold('trade_profit', warning=-50, comparison='lt')
    
    def start(self):
        """Start monitoring system"""
        self.system_monitor.start_monitoring(interval=60)
        logger.info("Monitoring system started")
    
    def stop(self):
        """Stop monitoring system"""
        self.system_monitor.stop_monitoring()
        logger.info("Monitoring system stopped")
    
    def get_full_status(self) -> Dict:
        """Get full system status"""
        return {
            'timestamp': datetime.now().isoformat(),
            'health': self.system_monitor.last_health_status,
            'system_metrics': self.system_monitor.get_system_metrics(),
            'trading_summary': self.trading_monitor.get_daily_summary(),
            'performance_metrics': self.perf_monitor.get_metrics_summary(),
            'recent_alerts': [
                {
                    'id': a.id,
                    'level': a.level.value,
                    'category': a.category,
                    'message': a.message,
                    'timestamp': a.timestamp.isoformat()
                }
                for a in self.alert_manager.get_recent_alerts(hours=24)
            ]
        }


# Singleton instance
monitoring = MonitoringSystem()


if __name__ == "__main__":
    print("Testing Monitoring System...")
    
    # Test logging setup
    LoggingSetup.setup(log_dir="./logs", log_level=logging.DEBUG)
    logger.info("Test log message")
    logger.warning("Test warning")
    logger.error("Test error")
    
    # Test alert manager
    print("\nTesting Alert Manager...")
    alert_mgr = AlertManager()
    
    alert = alert_mgr.create_alert(
        level=AlertLevel.WARNING,
        category='test',
        message='Test alert message',
        details={'key': 'value'}
    )
    print(f"Created alert: {alert.id}")
    
    # Test performance monitor
    print("\nTesting Performance Monitor...")
    perf_mon = PerformanceMonitor()
    
    for i in range(100):
        perf_mon.record_metric('test_metric', np.random.normal(50, 10))
    
    stats = perf_mon.get_metric_stats('test_metric')
    print(f"Metric stats: {stats}")
    
    # Test system monitor
    print("\nTesting System Monitor...")
    sys_mon = SystemMonitor()
    metrics = sys_mon.get_system_metrics()
    print(f"System metrics: {json.dumps(metrics, indent=2)}")
    
    # Test trading monitor
    print("\nTesting Trading Monitor...")
    trading_mon = TradingMonitor(alert_mgr, perf_mon)
    
    trading_mon.record_trade({'profit': 10, 'duration_minutes': 30})
    trading_mon.record_trade({'profit': -5, 'duration_minutes': 15})
    trading_mon.update_equity(105, 105)
    
    summary = trading_mon.get_daily_summary()
    print(f"Daily summary: {json.dumps(summary, indent=2)}")
    
    # Test full monitoring system
    print("\nTesting Full Monitoring System...")
    mon_sys = MonitoringSystem()
    status = mon_sys.get_full_status()
    print(f"Full status keys: {list(status.keys())}")
    
    print("\nMonitoring System test complete!")
