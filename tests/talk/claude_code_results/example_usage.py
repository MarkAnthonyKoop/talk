"""
Example usage of the agentic orchestration system.

This demonstrates how to use the orchestrator to manage a multi-agent system
for processing complex tasks with intelligent distribution and monitoring.
"""

import time
import logging
from pathlib import Path
import sys

# Add orchestrator to path
sys.path.insert(0, str(Path(__file__).parent))

from orchestrator import (
    AgentOrchestrator,
    AgentRegistry,
    OrchestrationMonitor
)
from orchestrator.core import OrchestrationConfig
from orchestrator.dispatcher import Task, TaskGroup, DistributionStrategy
from orchestrator.communication import MessageType
from orchestrator.monitor import Alert


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


# Example agent implementations
class DataCollectorAgent:
    """Agent that collects data from various sources"""
    def __init__(self, **kwargs):
        self.name = "DataCollector"
        self.id = f"collector_{id(self)}"
        self.sources = kwargs.get('sources', [])
    
    def initialize(self):
        print(f"{self.name} initialized with sources: {self.sources}")
    
    def run(self, prompt):
        """Simulate data collection"""
        time.sleep(0.5)  # Simulate work
        return {
            'data': [f"Data from {source}" for source in self.sources],
            'timestamp': time.time()
        }
    
    def health_check(self):
        return True


class DataProcessorAgent:
    """Agent that processes collected data"""
    def __init__(self, **kwargs):
        self.name = "DataProcessor"
        self.id = f"processor_{id(self)}"
        self.algorithm = kwargs.get('algorithm', 'default')
    
    def initialize(self):
        print(f"{self.name} initialized with algorithm: {self.algorithm}")
    
    def run(self, prompt):
        """Simulate data processing"""
        time.sleep(0.3)  # Simulate work
        
        # Process the data
        if isinstance(prompt, dict) and 'data' in prompt:
            processed = [f"Processed: {item}" for item in prompt['data']]
        else:
            processed = f"Processed: {prompt}"
        
        return {
            'processed_data': processed,
            'algorithm': self.algorithm,
            'timestamp': time.time()
        }
    
    def health_check(self):
        return True


class AnalysisAgent:
    """Agent that performs analysis on processed data"""
    def __init__(self, **kwargs):
        self.name = "Analyzer"
        self.id = f"analyzer_{id(self)}"
        self.model = kwargs.get('model', 'basic')
    
    def initialize(self):
        print(f"{self.name} initialized with model: {self.model}")
    
    def run(self, prompt):
        """Simulate analysis"""
        time.sleep(0.4)  # Simulate work
        
        return {
            'analysis': {
                'summary': 'Analysis complete',
                'insights': ['Insight 1', 'Insight 2', 'Insight 3'],
                'confidence': 0.95
            },
            'model': self.model,
            'timestamp': time.time()
        }
    
    def health_check(self):
        return True


class ReportGeneratorAgent:
    """Agent that generates reports from analysis"""
    def __init__(self, **kwargs):
        self.name = "ReportGenerator"
        self.id = f"reporter_{id(self)}"
        self.format = kwargs.get('format', 'json')
    
    def initialize(self):
        print(f"{self.name} initialized with format: {self.format}")
    
    def run(self, prompt):
        """Generate report"""
        time.sleep(0.2)  # Simulate work
        
        report = {
            'title': 'Analysis Report',
            'sections': [
                {'title': 'Summary', 'content': 'Executive summary here'},
                {'title': 'Findings', 'content': 'Key findings here'},
                {'title': 'Recommendations', 'content': 'Action items here'}
            ],
            'format': self.format,
            'generated_at': time.time()
        }
        
        return report
    
    def health_check(self):
        return True


def create_data_pipeline_example():
    """
    Example 1: Create a data processing pipeline with multiple agent types.
    """
    print("\n" + "="*60)
    print("EXAMPLE 1: Data Processing Pipeline")
    print("="*60)
    
    # Configure orchestrator
    config = OrchestrationConfig(
        max_agents=20,
        max_concurrent_tasks=5,
        enable_monitoring=True,
        enable_auto_scaling=True,
        load_balancing_policy="least_loaded"
    )
    
    # Create orchestrator
    orchestrator = AgentOrchestrator(config)
    
    # Register agent types
    orchestrator.register_agent(
        "collector",
        DataCollectorAgent,
        capabilities=["data_collection", "api_access"]
    )
    
    orchestrator.register_agent(
        "processor",
        DataProcessorAgent,
        capabilities=["data_processing", "transformation"]
    )
    
    orchestrator.register_agent(
        "analyzer",
        AnalysisAgent,
        capabilities=["analysis", "ml_inference"]
    )
    
    orchestrator.register_agent(
        "reporter",
        ReportGeneratorAgent,
        capabilities=["reporting", "visualization"]
    )
    
    # Spawn agent pool
    print("\nSpawning agent pool...")
    collectors = [
        orchestrator.spawn_agent("collector", {'sources': ['API', 'Database']})
        for _ in range(2)
    ]
    
    processors = [
        orchestrator.spawn_agent("processor", {'algorithm': 'advanced'})
        for _ in range(3)
    ]
    
    analyzers = [
        orchestrator.spawn_agent("analyzer", {'model': 'neural_network'})
        for _ in range(2)
    ]
    
    reporter = orchestrator.spawn_agent("reporter", {'format': 'html'})
    
    print(f"Agent pool created: {len(collectors)} collectors, {len(processors)} processors, "
          f"{len(analyzers)} analyzers, 1 reporter")
    
    # Create pipeline tasks
    print("\nSubmitting pipeline tasks...")
    
    # Task 1: Collect data
    collect_task = {
        'type': 'collect',
        'data': {'sources': ['API', 'Database', 'Files']},
        'required_capabilities': ['data_collection']
    }
    
    collect_id = orchestrator.submit_task(collect_task, priority=1)
    print(f"Submitted collection task: {collect_id}")
    
    # Task 2: Process data (depends on collection)
    process_task = {
        'type': 'process',
        'data': {'input': 'collected_data'},
        'required_capabilities': ['data_processing'],
        'dependencies': [collect_id]
    }
    
    process_id = orchestrator.submit_task(process_task, priority=2)
    print(f"Submitted processing task: {process_id}")
    
    # Task 3: Analyze data (depends on processing)
    analyze_task = {
        'type': 'analyze',
        'data': {'input': 'processed_data'},
        'required_capabilities': ['analysis'],
        'dependencies': [process_id]
    }
    
    analyze_id = orchestrator.submit_task(analyze_task, priority=3)
    print(f"Submitted analysis task: {analyze_id}")
    
    # Task 4: Generate report (depends on analysis)
    report_task = {
        'type': 'report',
        'data': {'input': 'analysis_results'},
        'required_capabilities': ['reporting'],
        'dependencies': [analyze_id]
    }
    
    report_id = orchestrator.submit_task(report_task, priority=4)
    print(f"Submitted report task: {report_id}")
    
    # Monitor execution
    print("\nMonitoring pipeline execution...")
    time.sleep(3)
    
    # Get status
    status = orchestrator.get_status()
    print(f"\nOrchestrator Status:")
    print(f"  Active agents: {status['active_agents']}")
    print(f"  Active tasks: {status['active_tasks']}")
    print(f"  Completed tasks: {status['completed_tasks']}")
    
    # Clean up
    orchestrator.shutdown()
    print("\nPipeline example completed")


def create_parallel_processing_example():
    """
    Example 2: Parallel processing with load balancing.
    """
    print("\n" + "="*60)
    print("EXAMPLE 2: Parallel Processing with Load Balancing")
    print("="*60)
    
    # Configure for parallel processing
    config = OrchestrationConfig(
        max_agents=10,
        max_concurrent_tasks=10,
        load_balancing_policy="adaptive"  # Adaptive load balancing
    )
    
    orchestrator = AgentOrchestrator(config)
    
    # Register a worker agent type
    orchestrator.register_agent(
        "worker",
        DataProcessorAgent,
        capabilities=["compute", "parallel"]
    )
    
    # Spawn worker pool
    print("\nSpawning worker pool...")
    workers = [
        orchestrator.spawn_agent("worker", {'algorithm': f'worker_{i}'})
        for i in range(5)
    ]
    print(f"Created {len(workers)} worker agents")
    
    # Submit parallel tasks
    print("\nSubmitting parallel tasks...")
    task_ids = []
    
    for i in range(20):
        task = {
            'type': 'compute',
            'data': {
                'input': f'dataset_{i}',
                'operation': 'transform'
            },
            'required_capabilities': ['compute']
        }
        
        task_id = orchestrator.submit_task(task, priority=5)
        task_ids.append(task_id)
    
    print(f"Submitted {len(task_ids)} parallel tasks")
    
    # Monitor load distribution
    print("\nMonitoring load distribution...")
    time.sleep(2)
    
    status = orchestrator.get_status()
    print(f"\nLoad Distribution:")
    for agent_id, load in status['agent_load'].items():
        print(f"  {agent_id}: {load} tasks")
    
    orchestrator.shutdown()
    print("\nParallel processing example completed")


def create_monitoring_example():
    """
    Example 3: Monitoring and alerting.
    """
    print("\n" + "="*60)
    print("EXAMPLE 3: Monitoring and Alerting")
    print("="*60)
    
    # Create orchestrator with monitoring
    config = OrchestrationConfig(
        enable_monitoring=True,
        health_check_interval=5
    )
    
    orchestrator = AgentOrchestrator(config)
    
    # Setup custom alerts
    if orchestrator.monitor:
        print("\nSetting up custom alerts...")
        
        # Alert for high task queue
        high_queue_alert = Alert(
            id="high_queue",
            name="High Task Queue",
            condition="queue_size",
            threshold=10,
            severity="warning",
            message="Task queue size exceeds 10"
        )
        orchestrator.monitor.register_alert(high_queue_alert)
        
        # Alert for agent failures
        agent_failure_alert = Alert(
            id="agent_failure",
            name="Agent Failure",
            condition="agent.failures",
            threshold=2,
            severity="critical",
            message="Agent failure rate too high"
        )
        orchestrator.monitor.register_alert(agent_failure_alert)
        
        print("Alerts configured")
    
    # Register and spawn agents
    orchestrator.register_agent("worker", DataProcessorAgent)
    agents = [orchestrator.spawn_agent("worker") for _ in range(3)]
    
    # Record some metrics
    print("\nRecording metrics...")
    if orchestrator.monitor:
        orchestrator.monitor.record_metric("custom.metric", 42.0)
        orchestrator.monitor.record_counter("tasks.custom", 5)
        orchestrator.monitor.record_timer("operation.duration", 1.234)
        
        # Record events
        orchestrator.monitor.record_event(
            "custom_event",
            "Important event occurred",
            severity="info",
            metadata={'details': 'Event details here'}
        )
    
    # Get monitoring summary
    if orchestrator.monitor:
        print("\nMonitoring Summary:")
        summary = orchestrator.monitor.get_metrics_summary()
        print(f"  Metrics collected: {summary['stats']['metrics_collected']}")
        print(f"  Events recorded: {summary['stats']['events_recorded']}")
        
        # Get health status
        health = orchestrator.monitor.get_health_status()
        print(f"\nHealth Status: {health['status']}")
        
        # Export metrics
        metrics_export = orchestrator.monitor.export_metrics()
        print(f"\nExported metrics (first 200 chars):")
        print(metrics_export[:200] + "...")
    
    orchestrator.shutdown()
    print("\nMonitoring example completed")


def create_message_bus_example():
    """
    Example 4: Inter-agent communication via message bus.
    """
    print("\n" + "="*60)
    print("EXAMPLE 4: Inter-Agent Communication")
    print("="*60)
    
    orchestrator = AgentOrchestrator()
    
    # Register agents
    orchestrator.register_agent("collector", DataCollectorAgent)
    orchestrator.register_agent("processor", DataProcessorAgent)
    
    # Spawn agents
    collector_id = orchestrator.spawn_agent("collector")
    processor_id = orchestrator.spawn_agent("processor")
    
    # Register agents with message bus
    orchestrator.message_bus.register_agent(collector_id)
    orchestrator.message_bus.register_agent(processor_id)
    
    print(f"\nAgents registered: {collector_id}, {processor_id}")
    
    # Subscribe to topics
    orchestrator.message_bus.subscribe(processor_id, "data_ready")
    print(f"{processor_id} subscribed to 'data_ready' topic")
    
    # Publish message
    print("\nPublishing message to topic...")
    orchestrator.message_bus.publish(
        collector_id,
        "data_ready",
        {'data': ['item1', 'item2', 'item3']}
    )
    
    # Direct message
    print("Sending direct message...")
    from orchestrator.communication import Message
    
    message = Message(
        sender=collector_id,
        recipient=processor_id,
        content={'command': 'process', 'data': 'test_data'}
    )
    orchestrator.message_bus.send(message)
    
    # Check message statistics
    stats = orchestrator.message_bus.get_statistics()
    print(f"\nMessage Bus Statistics:")
    print(f"  Messages sent: {stats['messages_sent']}")
    print(f"  Messages delivered: {stats['messages_delivered']}")
    print(f"  Registered agents: {stats['registered_agents']}")
    
    orchestrator.shutdown()
    print("\nMessage bus example completed")


def main():
    """Run all examples"""
    print("\n" + "="*80)
    print("AGENTIC ORCHESTRATION SYSTEM - DEMONSTRATION")
    print("="*80)
    
    examples = [
        ("Data Pipeline", create_data_pipeline_example),
        ("Parallel Processing", create_parallel_processing_example),
        ("Monitoring", create_monitoring_example),
        ("Message Bus", create_message_bus_example)
    ]
    
    for name, example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"\nError in {name} example: {e}")
        
        time.sleep(1)  # Pause between examples
    
    print("\n" + "="*80)
    print("ALL EXAMPLES COMPLETED")
    print("="*80)
    print("\nThe agentic orchestration system provides:")
    print("  ✓ Dynamic agent spawning and lifecycle management")
    print("  ✓ Intelligent task distribution with multiple strategies")
    print("  ✓ Load balancing and failover capabilities")
    print("  ✓ Comprehensive monitoring and alerting")
    print("  ✓ Inter-agent communication via message bus")
    print("  ✓ Task dependencies and DAG execution")
    print("  ✓ Health checking and auto-recovery")
    print("  ✓ Checkpoint and restore functionality")


if __name__ == "__main__":
    main()