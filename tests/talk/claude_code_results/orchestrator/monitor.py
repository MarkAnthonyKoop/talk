"""
Monitoring and observability system for agent orchestration.
"""

import json
import logging
import threading
import time
from collections import deque, defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Deque, Tuple
import statistics

log = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"
    RATE = "rate"


@dataclass
class Metric:
    """Individual metric data point"""
    name: str
    type: MetricType
    value: float
    timestamp: datetime = field(default_factory=datetime.now)
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Alert:
    """Alert definition"""
    id: str
    name: str
    condition: str
    threshold: float
    severity: str  # critical, warning, info
    message: str
    cooldown: int = 300  # seconds
    enabled: bool = True
    last_triggered: Optional[datetime] = None


@dataclass
class HealthCheck:
    """Health check result"""
    component: str
    status: str  # healthy, degraded, unhealthy
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    details: Dict[str, Any] = field(default_factory=dict)


class OrchestrationMonitor:
    """
    Comprehensive monitoring system for the orchestrator.
    
    Features:
    - Real-time metrics collection
    - Performance tracking
    - Health monitoring
    - Alerting system
    - Historical data retention
    - Dashboards and reporting
    """
    
    def __init__(self, retention_hours: int = 24, alert_handler: Optional[Any] = None):
        self.retention_hours = retention_hours
        self.alert_handler = alert_handler
        
        # Metrics storage
        self.metrics: Dict[str, Deque[Metric]] = defaultdict(lambda: deque(maxlen=10000))
        self.aggregated_metrics: Dict[str, Dict[str, float]] = {}
        
        # Performance tracking
        self.performance_windows: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=100))
        self.latency_buckets: Dict[str, List[float]] = defaultdict(list)
        
        # Health monitoring
        self.health_checks: Dict[str, HealthCheck] = {}
        self.component_status: Dict[str, str] = {}
        
        # Alerting
        self.alerts: Dict[str, Alert] = {}
        self.alert_history: Deque[Tuple[Alert, datetime]] = deque(maxlen=1000)
        
        # Event tracking
        self.events: Deque[Dict[str, Any]] = deque(maxlen=5000)
        
        # Statistics
        self.stats = {
            'metrics_collected': 0,
            'alerts_triggered': 0,
            'health_checks_performed': 0,
            'events_recorded': 0
        }
        
        # Background processing
        self.shutdown_event = threading.Event()
        self.aggregation_thread = None
        self.cleanup_thread = None
        self.orchestrator = None
        
        # Synchronization
        self.lock = threading.RLock()
        
        log.info("Orchestration monitor initialized")
    
    def start_monitoring(self, orchestrator: Any):
        """Start monitoring an orchestrator"""
        self.orchestrator = orchestrator
        
        # Start background threads
        self.aggregation_thread = threading.Thread(
            target=self._aggregation_loop,
            daemon=True
        )
        self.aggregation_thread.start()
        
        self.cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            daemon=True
        )
        self.cleanup_thread.start()
        
        # Register default alerts
        self._register_default_alerts()
        
        log.info("Monitoring started")
    
    def record_metric(self, 
                     name: str,
                     value: float,
                     type: MetricType = MetricType.GAUGE,
                     tags: Dict[str, str] = None,
                     metadata: Dict[str, Any] = None):
        """Record a metric"""
        with self.lock:
            metric = Metric(
                name=name,
                type=type,
                value=value,
                tags=tags or {},
                metadata=metadata or {}
            )
            
            self.metrics[name].append(metric)
            self.stats['metrics_collected'] += 1
            
            # Check alerts
            self._check_metric_alerts(metric)
    
    def record_timer(self, name: str, duration: float, tags: Dict[str, str] = None):
        """Record a timing metric"""
        self.record_metric(name, duration, MetricType.TIMER, tags)
        
        # Update latency buckets
        with self.lock:
            self.latency_buckets[name].append(duration)
    
    def record_counter(self, name: str, increment: int = 1, tags: Dict[str, str] = None):
        """Record a counter metric"""
        with self.lock:
            current = self.aggregated_metrics.get(name, {}).get('value', 0)
            self.record_metric(name, current + increment, MetricType.COUNTER, tags)
    
    def record_rate(self, name: str, value: float, tags: Dict[str, str] = None):
        """Record a rate metric"""
        self.record_metric(name, value, MetricType.RATE, tags)
        
        # Update performance window
        with self.lock:
            self.performance_windows[name].append(value)
    
    def record_event(self, 
                    event_type: str,
                    message: str,
                    severity: str = "info",
                    metadata: Dict[str, Any] = None):
        """Record an event"""
        with self.lock:
            event = {
                'type': event_type,
                'message': message,
                'severity': severity,
                'timestamp': datetime.now().isoformat(),
                'metadata': metadata or {}
            }
            
            self.events.append(event)
            self.stats['events_recorded'] += 1
            
            log.debug(f"Event recorded: {event_type} - {message}")
    
    def record_agent_spawn(self, agent_id: str, agent_type: str):
        """Record agent spawn event"""
        self.record_event(
            "agent_spawned",
            f"Agent {agent_id} of type {agent_type} spawned",
            metadata={'agent_id': agent_id, 'agent_type': agent_type}
        )
        self.record_counter("agents.spawned", tags={'type': agent_type})
    
    def record_agent_termination(self, agent_id: str):
        """Record agent termination event"""
        self.record_event(
            "agent_terminated",
            f"Agent {agent_id} terminated",
            metadata={'agent_id': agent_id}
        )
        self.record_counter("agents.terminated")
    
    def record_task_submission(self, task_id: str, task: Dict[str, Any]):
        """Record task submission"""
        self.record_event(
            "task_submitted",
            f"Task {task_id} submitted",
            metadata={'task_id': task_id, 'task_type': task.get('type')}
        )
        self.record_counter("tasks.submitted", tags={'type': task.get('type', 'unknown')})
    
    def record_task_completion(self, task_id: str, result: Any):
        """Record task completion"""
        self.record_event(
            "task_completed",
            f"Task {task_id} completed",
            metadata={'task_id': task_id}
        )
        self.record_counter("tasks.completed")
    
    def record_task_failure(self, task_id: str, error: Exception):
        """Record task failure"""
        self.record_event(
            "task_failed",
            f"Task {task_id} failed: {str(error)}",
            severity="error",
            metadata={'task_id': task_id, 'error': str(error)}
        )
        self.record_counter("tasks.failed")
    
    def perform_health_check(self, component: str, check_func: Any) -> HealthCheck:
        """Perform a health check on a component"""
        try:
            result = check_func()
            
            if isinstance(result, bool):
                status = "healthy" if result else "unhealthy"
                message = f"{component} is {status}"
                details = {}
            elif isinstance(result, dict):
                status = result.get('status', 'unknown')
                message = result.get('message', '')
                details = result.get('details', {})
            else:
                status = "unknown"
                message = str(result)
                details = {}
            
            health_check = HealthCheck(
                component=component,
                status=status,
                message=message,
                details=details
            )
            
        except Exception as e:
            health_check = HealthCheck(
                component=component,
                status="unhealthy",
                message=f"Health check failed: {str(e)}",
                details={'error': str(e)}
            )
        
        with self.lock:
            self.health_checks[component] = health_check
            self.component_status[component] = health_check.status
            self.stats['health_checks_performed'] += 1
        
        # Check health alerts
        self._check_health_alerts(health_check)
        
        return health_check
    
    def register_alert(self, alert: Alert):
        """Register an alert"""
        with self.lock:
            self.alerts[alert.id] = alert
            log.info(f"Alert registered: {alert.name}")
    
    def _register_default_alerts(self):
        """Register default system alerts"""
        default_alerts = [
            Alert(
                id="high_task_failure_rate",
                name="High Task Failure Rate",
                condition="tasks.failed.rate",
                threshold=0.1,  # 10% failure rate
                severity="warning",
                message="Task failure rate exceeds 10%"
            ),
            Alert(
                id="agent_pool_exhausted",
                name="Agent Pool Exhausted",
                condition="agents.available",
                threshold=2,
                severity="critical",
                message="Less than 2 agents available"
            ),
            Alert(
                id="high_latency",
                name="High Task Latency",
                condition="tasks.latency.p99",
                threshold=10.0,  # 10 seconds
                severity="warning",
                message="99th percentile task latency exceeds 10s"
            ),
            Alert(
                id="memory_pressure",
                name="Memory Pressure",
                condition="system.memory.usage",
                threshold=0.9,  # 90% memory usage
                severity="warning",
                message="System memory usage exceeds 90%"
            )
        ]
        
        for alert in default_alerts:
            self.register_alert(alert)
    
    def _check_metric_alerts(self, metric: Metric):
        """Check if metric triggers any alerts"""
        for alert in self.alerts.values():
            if not alert.enabled:
                continue
            
            if alert.condition == metric.name:
                if self._evaluate_alert_condition(alert, metric.value):
                    self._trigger_alert(alert)
    
    def _check_health_alerts(self, health_check: HealthCheck):
        """Check if health check triggers alerts"""
        if health_check.status == "unhealthy":
            alert = Alert(
                id=f"health_{health_check.component}",
                name=f"{health_check.component} Unhealthy",
                condition="health",
                threshold=0,
                severity="critical",
                message=health_check.message
            )
            self._trigger_alert(alert)
    
    def _evaluate_alert_condition(self, alert: Alert, value: float) -> bool:
        """Evaluate if alert condition is met"""
        # Simple threshold comparison
        # Could be extended for more complex conditions
        return value > alert.threshold
    
    def _trigger_alert(self, alert: Alert):
        """Trigger an alert"""
        # Check cooldown
        if alert.last_triggered:
            cooldown_end = alert.last_triggered + timedelta(seconds=alert.cooldown)
            if datetime.now() < cooldown_end:
                return
        
        alert.last_triggered = datetime.now()
        
        with self.lock:
            self.alert_history.append((alert, datetime.now()))
            self.stats['alerts_triggered'] += 1
        
        # Call alert handler if configured
        if self.alert_handler:
            try:
                self.alert_handler(alert)
            except Exception as e:
                log.error(f"Alert handler error: {e}")
        
        # Log alert
        if alert.severity == "critical":
            log.critical(f"ALERT: {alert.message}")
        elif alert.severity == "warning":
            log.warning(f"ALERT: {alert.message}")
        else:
            log.info(f"ALERT: {alert.message}")
        
        # Record alert event
        self.record_event(
            "alert_triggered",
            alert.message,
            severity=alert.severity,
            metadata={'alert_id': alert.id, 'threshold': alert.threshold}
        )
    
    def _aggregation_loop(self):
        """Background metric aggregation"""
        while not self.shutdown_event.is_set():
            try:
                time.sleep(10)  # Aggregate every 10 seconds
                self._aggregate_metrics()
                self._compute_derived_metrics()
                
                # Check system alerts
                if self.orchestrator:
                    self._check_system_alerts()
                
            except Exception as e:
                log.error(f"Error in aggregation loop: {e}")
    
    def _aggregate_metrics(self):
        """Aggregate raw metrics"""
        with self.lock:
            for name, metrics in self.metrics.items():
                if not metrics:
                    continue
                
                values = [m.value for m in metrics]
                
                aggregated = {
                    'value': values[-1],  # Latest value
                    'min': min(values),
                    'max': max(values),
                    'avg': statistics.mean(values),
                    'count': len(values)
                }
                
                # Add percentiles for timers
                if metrics[0].type == MetricType.TIMER:
                    sorted_values = sorted(values)
                    aggregated['p50'] = self._percentile(sorted_values, 50)
                    aggregated['p95'] = self._percentile(sorted_values, 95)
                    aggregated['p99'] = self._percentile(sorted_values, 99)
                
                self.aggregated_metrics[name] = aggregated
    
    def _compute_derived_metrics(self):
        """Compute derived metrics"""
        with self.lock:
            # Task success rate
            completed = self.aggregated_metrics.get('tasks.completed', {}).get('count', 0)
            failed = self.aggregated_metrics.get('tasks.failed', {}).get('count', 0)
            
            if completed + failed > 0:
                success_rate = completed / (completed + failed)
                self.record_metric('tasks.success_rate', success_rate, MetricType.GAUGE)
            
            # Agent utilization
            if self.orchestrator:
                active_agents = len(self.orchestrator.registry.active_agents)
                if active_agents > 0:
                    total_load = sum(self.orchestrator.agent_load.values())
                    utilization = total_load / active_agents
                    self.record_metric('agents.utilization', utilization, MetricType.GAUGE)
    
    def _check_system_alerts(self):
        """Check system-level alerts"""
        # Check memory usage
        try:
            import psutil
            memory_usage = psutil.virtual_memory().percent / 100
            self.record_metric('system.memory.usage', memory_usage, MetricType.GAUGE)
            
            cpu_usage = psutil.cpu_percent() / 100
            self.record_metric('system.cpu.usage', cpu_usage, MetricType.GAUGE)
        except ImportError:
            pass
    
    def _cleanup_loop(self):
        """Clean up old metrics"""
        while not self.shutdown_event.is_set():
            try:
                time.sleep(3600)  # Clean up every hour
                self._cleanup_old_metrics()
            except Exception as e:
                log.error(f"Error in cleanup loop: {e}")
    
    def _cleanup_old_metrics(self):
        """Remove metrics older than retention period"""
        cutoff = datetime.now() - timedelta(hours=self.retention_hours)
        
        with self.lock:
            for name, metrics in self.metrics.items():
                # Remove old metrics
                while metrics and metrics[0].timestamp < cutoff:
                    metrics.popleft()
    
    def _percentile(self, sorted_values: List[float], percentile: int) -> float:
        """Calculate percentile from sorted values"""
        if not sorted_values:
            return 0
        
        index = int(len(sorted_values) * percentile / 100)
        return sorted_values[min(index, len(sorted_values) - 1)]
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics"""
        with self.lock:
            return {
                'aggregated': dict(self.aggregated_metrics),
                'latest': {
                    name: metrics[-1].value if metrics else 0
                    for name, metrics in self.metrics.items()
                },
                'stats': self.stats.copy()
            }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status"""
        with self.lock:
            unhealthy = [c for c, s in self.component_status.items() if s == "unhealthy"]
            degraded = [c for c, s in self.component_status.items() if s == "degraded"]
            
            if unhealthy:
                overall_status = "unhealthy"
            elif degraded:
                overall_status = "degraded"
            else:
                overall_status = "healthy"
            
            return {
                'status': overall_status,
                'components': dict(self.component_status),
                'unhealthy_components': unhealthy,
                'degraded_components': degraded,
                'last_check': max(
                    (hc.timestamp for hc in self.health_checks.values()),
                    default=datetime.now()
                ).isoformat()
            }
    
    def get_recent_events(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent events"""
        with self.lock:
            return list(self.events)[-limit:]
    
    def get_alert_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get alert history"""
        with self.lock:
            history = []
            for alert, timestamp in list(self.alert_history)[-limit:]:
                history.append({
                    'alert_id': alert.id,
                    'name': alert.name,
                    'severity': alert.severity,
                    'message': alert.message,
                    'timestamp': timestamp.isoformat()
                })
            return history
    
    def export_metrics(self, format: str = "json") -> str:
        """Export metrics in specified format"""
        data = {
            'timestamp': datetime.now().isoformat(),
            'metrics': self.get_metrics_summary(),
            'health': self.get_health_status(),
            'events': self.get_recent_events(50),
            'alerts': self.get_alert_history(20)
        }
        
        if format == "json":
            return json.dumps(data, indent=2, default=str)
        else:
            # Could support other formats (Prometheus, etc.)
            return json.dumps(data, default=str)
    
    def shutdown(self):
        """Shutdown the monitor"""
        log.info("Shutting down monitor")
        
        # Signal shutdown
        self.shutdown_event.set()
        
        # Wait for threads
        if self.aggregation_thread:
            self.aggregation_thread.join(timeout=5)
        if self.cleanup_thread:
            self.cleanup_thread.join(timeout=5)
        
        # Export final metrics
        final_metrics = self.export_metrics()
        with open(f"metrics_final_{int(time.time())}.json", 'w') as f:
            f.write(final_metrics)
        
        log.info("Monitor shutdown complete")