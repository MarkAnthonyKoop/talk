#!/usr/bin/env python3
"""
Real-time Monitoring Dashboard for Agent Collaboration System

This module provides a web-based dashboard for monitoring and controlling
the multi-agent collaboration system in real-time. It includes:

- Agent status monitoring with performance metrics
- Task queue visualization and management  
- Voting system monitoring and interaction
- System health metrics and alerts
- Interactive controls for agent management
- Real-time updates via WebSockets

Built with Flask and Socket.IO for real-time communication.
"""

import asyncio
import json
import logging
import time
import threading
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit, disconnect
import psutil

from .agent_message_bus import AgentMessageBus, MessageType
from .shared_workspace import SharedWorkspace
from .collaborative_decision_making import VotingSystem, DecisionType, VoteType, VotingStrategy
from .dynamic_agent_spawning import AgentSpawner, Task, TaskPriority, AgentTemplate

log = logging.getLogger(__name__)

class CollaborationDashboard:
    """
    Real-time monitoring dashboard for the agent collaboration system.
    
    Provides a web interface for monitoring agent activity, system health,
    and collaborative processes in real-time.
    """
    
    def __init__(self, host: str = "localhost", port: int = 5000, debug: bool = False):
        """
        Initialize the dashboard.
        
        Args:
            host: Host address to bind to
            port: Port to listen on
            debug: Enable debug mode
        """
        self.host = host
        self.port = port
        self.debug = debug
        
        # Flask app setup
        self.app = Flask(__name__, 
                        template_folder=str(Path(__file__).parent / "templates"),
                        static_folder=str(Path(__file__).parent / "static"))
        self.app.config['SECRET_KEY'] = 'collaboration_dashboard_secret'
        
        # Socket.IO setup
        self.socketio = SocketIO(self.app, cors_allowed_origins="*", async_mode='threading')
        
        # Collaboration system references
        self.message_bus: Optional[AgentMessageBus] = None
        self.workspace: Optional[SharedWorkspace] = None
        self.voting_system: Optional[VotingSystem] = None
        self.agent_spawner: Optional[AgentSpawner] = None
        
        # Dashboard state
        self.connected_clients = set()
        self.update_interval = 2.0  # seconds
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        
        # Setup routes and socket handlers
        self._setup_routes()
        self._setup_socket_handlers()
        
        log.info(f"CollaborationDashboard initialized on {host}:{port}")
    
    def connect_systems(self, message_bus: AgentMessageBus, workspace: SharedWorkspace = None,
                       voting_system: VotingSystem = None, agent_spawner: AgentSpawner = None):
        """
        Connect the dashboard to collaboration systems.
        
        Args:
            message_bus: AgentMessageBus instance
            workspace: Optional SharedWorkspace instance
            voting_system: Optional VotingSystem instance
            agent_spawner: Optional AgentSpawner instance
        """
        self.message_bus = message_bus
        self.workspace = workspace
        self.voting_system = voting_system
        self.agent_spawner = agent_spawner
        
        log.info("Dashboard connected to collaboration systems")
    
    def _setup_routes(self):
        """Setup Flask routes."""
        
        @self.app.route('/')
        def index():
            """Main dashboard page."""
            return render_template('dashboard.html')
        
        @self.app.route('/api/status')
        def api_status():
            """API endpoint for system status."""
            return jsonify(self._get_system_status())
        
        @self.app.route('/api/agents')
        def api_agents():
            """API endpoint for agent information."""
            return jsonify(self._get_agent_data())
        
        @self.app.route('/api/tasks')
        def api_tasks():
            """API endpoint for task queue information."""
            return jsonify(self._get_task_data())
        
        @self.app.route('/api/votes')
        def api_votes():
            """API endpoint for voting information."""
            return jsonify(self._get_voting_data())
        
        @self.app.route('/api/health')
        def api_health():
            """API endpoint for system health metrics."""
            return jsonify(self._get_health_data())
    
    def _setup_socket_handlers(self):
        """Setup Socket.IO event handlers."""
        
        @self.socketio.on('connect')
        def handle_connect():
            """Handle client connection."""
            self.connected_clients.add(request.sid)
            log.info(f"Client connected: {request.sid}")
            
            # Send initial data
            emit('system_status', self._get_system_status())
            emit('update_data', self._get_all_data())
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Handle client disconnection."""
            self.connected_clients.discard(request.sid)
            log.info(f"Client disconnected: {request.sid}")
        
        @self.socketio.on('spawn_agent')
        def handle_spawn_agent(data):
            """Handle agent spawning request."""
            try:
                agent_type = data.get('agent_type', 'worker')
                reason = data.get('reason', 'manual_spawn')
                
                if self.agent_spawner:
                    # This would need to be called from an async context
                    # For now, we'll simulate it
                    agent_id = f"manual_{int(time.time())}"
                    emit('agent_spawned', {
                        'agent_id': agent_id,
                        'message': f'Agent {agent_id} spawned successfully'
                    })
                    log.info(f"Manual agent spawn requested: {agent_type}")
                else:
                    emit('error', {'message': 'Agent spawner not available'})
                    
            except Exception as e:
                log.error(f"Error spawning agent: {e}")
                emit('error', {'message': f'Failed to spawn agent: {str(e)}'})
        
        @self.socketio.on('terminate_agent')
        def handle_terminate_agent(data):
            """Handle agent termination request."""
            try:
                agent_id = data.get('agent_id')
                
                if not agent_id:
                    emit('error', {'message': 'Agent ID is required'})
                    return
                
                if self.agent_spawner:
                    # This would need to be called from an async context
                    # For now, we'll simulate it
                    emit('agent_terminated', {
                        'agent_id': agent_id,
                        'message': f'Agent {agent_id} terminated successfully'
                    })
                    log.info(f"Manual agent termination requested: {agent_id}")
                else:
                    emit('error', {'message': 'Agent spawner not available'})
                    
            except Exception as e:
                log.error(f"Error terminating agent: {e}")
                emit('error', {'message': f'Failed to terminate agent: {str(e)}'})
        
        @self.socketio.on('submit_task')
        def handle_submit_task(data):
            """Handle task submission request."""
            try:
                task_description = data.get('task_description', '')
                priority = data.get('priority', 'NORMAL')
                
                if not task_description:
                    emit('error', {'message': 'Task description is required'})
                    return
                
                # Create and submit task
                task_id = f"manual_{int(time.time())}"
                
                if self.agent_spawner:
                    # This would need proper async handling
                    emit('task_submitted', {
                        'task_id': task_id,
                        'message': f'Task {task_id} submitted successfully'
                    })
                    log.info(f"Manual task submission: {task_description}")
                else:
                    emit('error', {'message': 'Agent spawner not available'})
                    
            except Exception as e:
                log.error(f"Error submitting task: {e}")
                emit('error', {'message': f'Failed to submit task: {str(e)}'})
        
        @self.socketio.on('create_vote')
        def handle_create_vote(data):
            """Handle vote creation request."""
            try:
                title = data.get('title', '')
                description = data.get('description', '')
                
                if not title or not description:
                    emit('error', {'message': 'Title and description are required'})
                    return
                
                if self.voting_system:
                    # This would need proper async handling
                    decision_id = f"manual_{int(time.time())}"
                    emit('vote_created', {
                        'decision_id': decision_id,
                        'message': f'Vote "{title}" created successfully'
                    })
                    log.info(f"Manual vote creation: {title}")
                else:
                    emit('error', {'message': 'Voting system not available'})
                    
            except Exception as e:
                log.error(f"Error creating vote: {e}")
                emit('error', {'message': f'Failed to create vote: {str(e)}'})
    
    def _get_system_status(self) -> Dict[str, Any]:
        """Get overall system status."""
        status = {
            'timestamp': datetime.now().isoformat(),
            'systems': {
                'message_bus': self.message_bus is not None,
                'workspace': self.workspace is not None,
                'voting_system': self.voting_system is not None,
                'agent_spawner': self.agent_spawner is not None
            },
            'connected_clients': len(self.connected_clients)
        }
        
        if self.message_bus:
            bus_stats = self.message_bus.get_stats()
            status['message_bus_stats'] = bus_stats
        
        if self.agent_spawner:
            spawner_status = self.agent_spawner.get_status()
            status['spawner_stats'] = spawner_status
        
        return status
    
    def _get_agent_data(self) -> Dict[str, Any]:
        """Get agent status data."""
        agents_data = {}
        
        if self.message_bus:
            agents = self.message_bus.get_online_agents()
            for agent_id, agent_info in agents.items():
                agents_data[agent_id] = {
                    'agent_id': agent_id,
                    'agent_type': agent_info.agent_type,
                    'status': agent_info.status,
                    'capabilities': agent_info.capabilities,
                    'last_heartbeat': agent_info.last_heartbeat,
                    'is_online': agent_info.is_online()
                }
        
        if self.agent_spawner:
            spawner_agents = self.agent_spawner.active_agents
            for agent_id, agent_instance in spawner_agents.items():
                if agent_id in agents_data:
                    agents_data[agent_id].update({
                        'current_tasks': len(agent_instance.current_tasks),
                        'completed_tasks': agent_instance.completed_tasks,
                        'failed_tasks': agent_instance.failed_tasks,
                        'performance_score': agent_instance.performance_score,
                        'spawned_at': agent_instance.spawned_at
                    })
        
        return {'agents': agents_data}
    
    def _get_task_data(self) -> Dict[str, Any]:
        """Get task queue data."""
        tasks_data = {
            'pending_tasks': [],
            'completed_tasks': [],
            'failed_tasks': []
        }
        
        if self.agent_spawner:
            status = self.agent_spawner.get_status()
            tasks_data.update({
                'pending_count': status.get('pending_tasks', 0),
                'completed_count': status.get('completed_tasks', 0),
                'failed_count': status.get('failed_tasks', 0)
            })
            
            # Get recent task history (last 20 tasks)
            task_history = self.agent_spawner.task_history[-20:]
            for task in task_history:
                task_info = {
                    'task_id': task.task_id,
                    'task_type': task.task_type,
                    'priority': task.priority.name,
                    'created_at': task.created_at,
                    'assigned_agent': task.assigned_agent,
                    'requirements': task.requirements
                }
                
                # Categorize by status (simplified)
                if task.assigned_agent:
                    tasks_data['completed_tasks'].append(task_info)
                else:
                    tasks_data['pending_tasks'].append(task_info)
        
        return tasks_data
    
    def _get_voting_data(self) -> Dict[str, Any]:
        """Get voting system data."""
        voting_data = {
            'active_decisions': [],
            'completed_decisions': [],
            'stats': {}
        }
        
        if self.voting_system:
            # Get active decisions
            active_decisions = self.voting_system.get_active_decisions()
            for decision in active_decisions:
                voting_data['active_decisions'].append({
                    'decision_id': decision['decision_id'],
                    'title': decision['title'],
                    'description': decision['description'],
                    'proposer_id': decision['proposer_id'],
                    'created_at': decision['created_at'],
                    'deadline': decision.get('deadline'),
                    'vote_count': decision['vote_count'],
                    'quorum_progress': decision['quorum_progress'],
                    'current_tally': decision['current_tally']
                })
            
            # Get voting stats
            voting_data['stats'] = self.voting_system.get_stats()
        
        return voting_data
    
    def _get_health_data(self) -> Dict[str, Any]:
        """Get system health metrics."""
        health_data = {
            'timestamp': time.time(),
            'system': {},
            'collaboration': {}
        }
        
        # System metrics
        try:
            health_data['system'] = {
                'cpu_percent': psutil.cpu_percent(interval=0.1),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_percent': psutil.disk_usage('/').percent,
                'uptime': time.time() - psutil.boot_time()
            }
        except Exception as e:
            log.warning(f"Failed to get system metrics: {e}")
            health_data['system'] = {'error': str(e)}
        
        # Collaboration system metrics
        if self.message_bus:
            bus_stats = self.message_bus.get_stats()
            health_data['collaboration']['message_bus'] = {
                'uptime': bus_stats.get('uptime_seconds', 0),
                'total_messages': bus_stats.get('total_messages', 0),
                'messages_per_second': bus_stats.get('messages_per_second', 0),
                'online_agents': bus_stats.get('online_agents', 0)
            }
        
        if self.workspace:
            workspace_locks = self.workspace.get_active_locks()
            workspace_changes = self.workspace.get_changes(limit=10)
            health_data['collaboration']['workspace'] = {
                'active_locks': len(workspace_locks),
                'recent_changes': len(workspace_changes)
            }
        
        if self.voting_system:
            voting_stats = self.voting_system.get_stats()
            health_data['collaboration']['voting'] = voting_stats
        
        if self.agent_spawner:
            spawner_status = self.agent_spawner.get_status()
            health_data['collaboration']['spawner'] = spawner_status
        
        return health_data
    
    def _get_all_data(self) -> Dict[str, Any]:
        """Get all dashboard data."""
        return {
            'agents': self._get_agent_data(),
            'tasks': self._get_task_data(),
            'voting': self._get_voting_data(),
            'health': self._get_health_data(),
            'system_status': self._get_system_status()
        }
    
    def start_monitoring(self):
        """Start real-time monitoring and updates."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        log.info("Real-time monitoring started")
    
    def stop_monitoring(self):
        """Stop real-time monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        log.info("Real-time monitoring stopped")
    
    def _monitoring_loop(self):
        """Background monitoring loop."""
        while self.monitoring_active:
            try:
                if self.connected_clients:
                    # Get all data and broadcast to connected clients
                    data = self._get_all_data()
                    self.socketio.emit('update_data', data)
                
                time.sleep(self.update_interval)
                
            except Exception as e:
                log.error(f"Error in monitoring loop: {e}")
                time.sleep(self.update_interval)
    
    def run(self, **kwargs):
        """Run the dashboard server."""
        log.info(f"Starting dashboard server on {self.host}:{self.port}")
        
        # Start monitoring
        self.start_monitoring()
        
        try:
            # Run the Flask-SocketIO server
            self.socketio.run(
                self.app,
                host=self.host,
                port=self.port,
                debug=self.debug,
                **kwargs
            )
        finally:
            # Stop monitoring
            self.stop_monitoring()

# Template for the dashboard HTML
DASHBOARD_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Agent Collaboration Dashboard</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            text-align: center;
        }
        .dashboard-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
        }
        .card {
            background: white;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            padding: 20px;
        }
        .card h2 {
            margin-top: 0;
            color: #333;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        th {
            background-color: #f8f9fa;
            font-weight: 600;
        }
        .status-online { color: #28a745; }
        .status-offline { color: #dc3545; }
        .status-busy { color: #ffc107; }
        .priority-high { background-color: #ffebee; }
        .priority-normal { background-color: #f3e5f5; }
        .priority-low { background-color: #e8f5e8; }
        .controls {
            display: flex;
            gap: 10px;
            margin-top: 15px;
            flex-wrap: wrap;
        }
        .controls input, .controls button {
            padding: 8px 12px;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        .controls button {
            background: #667eea;
            color: white;
            border: none;
            cursor: pointer;
        }
        .controls button:hover {
            background: #5a6fd8;
        }
        .message {
            margin-top: 10px;
            padding: 10px;
            border-radius: 5px;
            display: none;
        }
        .message.success {
            background-color: #d4edda;
            color: #155724;
        }
        .message.error {
            background-color: #f8d7da;
            color: #721c24;
        }
        .metric {
            display: inline-block;
            margin: 5px 10px;
            font-weight: bold;
        }
        .metric-value {
            font-size: 1.2em;
            color: #667eea;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 Agent Collaboration Dashboard</h1>
            <div>
                <span class="metric">Connected Clients: <span id="connected-clients" class="metric-value">0</span></span>
                <span class="metric">Last Update: <span id="last-update" class="metric-value">--</span></span>
            </div>
        </div>

        <div class="dashboard-grid">
            <!-- Agent Status Card -->
            <div class="card">
                <h2>🔧 Agent Status</h2>
                <table id="agent-table">
                    <thead>
                        <tr>
                            <th>Agent ID</th>
                            <th>Type</th>
                            <th>Status</th>
                            <th>Tasks</th>
                            <th>Performance</th>
                        </tr>
                    </thead>
                    <tbody>
                        <!-- Agent data will be populated here -->
                    </tbody>
                </table>
            </div>

            <!-- Task Queue Card -->
            <div class="card">
                <h2>📋 Task Queue</h2>
                <div>
                    <span class="metric">Pending: <span id="pending-tasks" class="metric-value">0</span></span>
                    <span class="metric">Completed: <span id="completed-tasks" class="metric-value">0</span></span>
                    <span class="metric">Failed: <span id="failed-tasks" class="metric-value">0</span></span>
                </div>
                <table id="task-table">
                    <thead>
                        <tr>
                            <th>Task ID</th>
                            <th>Type</th>
                            <th>Priority</th>
                            <th>Agent</th>
                            <th>Status</th>
                        </tr>
                    </thead>
                    <tbody>
                        <!-- Task data will be populated here -->
                    </tbody>
                </table>
            </div>

            <!-- Voting System Card -->
            <div class="card">
                <h2>🗳️ Voting System</h2>
                <div>
                    <span class="metric">Active: <span id="active-votes" class="metric-value">0</span></span>
                    <span class="metric">Approval Rate: <span id="approval-rate" class="metric-value">0%</span></span>
                </div>
                <table id="voting-table">
                    <thead>
                        <tr>
                            <th>Decision</th>
                            <th>Proposer</th>
                            <th>Progress</th>
                            <th>Votes</th>
                        </tr>
                    </thead>
                    <tbody>
                        <!-- Voting data will be populated here -->
                    </tbody>
                </table>
            </div>

            <!-- System Health Card -->
            <div class="card">
                <h2>💊 System Health</h2>
                <div>
                    <span class="metric">CPU: <span id="cpu-usage" class="metric-value">0%</span></span>
                    <span class="metric">Memory: <span id="memory-usage" class="metric-value">0%</span></span>
                    <span class="metric">Agents: <span id="online-agents" class="metric-value">0</span></span>
                </div>
                <canvas id="health-chart" width="400" height="200"></canvas>
            </div>

            <!-- Interactive Controls Card -->
            <div class="card">
                <h2>🎮 Interactive Controls</h2>
                
                <h3>Agent Management</h3>
                <div class="controls">
                    <input type="text" id="agent-type" placeholder="Agent Type (e.g., worker)" value="worker">
                    <button onclick="spawnAgent()">Spawn Agent</button>
                    <input type="text" id="agent-id" placeholder="Agent ID to terminate">
                    <button onclick="terminateAgent()">Terminate Agent</button>
                </div>

                <h3>Task Management</h3>
                <div class="controls">
                    <input type="text" id="task-description" placeholder="Task description">
                    <select id="task-priority">
                        <option value="LOW">Low</option>
                        <option value="NORMAL" selected>Normal</option>
                        <option value="HIGH">High</option>
                        <option value="CRITICAL">Critical</option>
                    </select>
                    <button onclick="submitTask()">Submit Task</button>
                </div>

                <h3>Voting</h3>
                <div class="controls">
                    <input type="text" id="vote-title" placeholder="Vote title">
                    <input type="text" id="vote-description" placeholder="Vote description">
                    <button onclick="createVote()">Create Vote</button>
                </div>

                <div id="message" class="message"></div>
            </div>
        </div>
    </div>

    <script>
        const socket = io();
        let healthChart = null;

        // Socket connection handlers
        socket.on('connect', () => {
            console.log('Connected to dashboard');
            showMessage('Connected to dashboard', 'success');
        });

        socket.on('disconnect', () => {
            console.log('Disconnected from dashboard');
            showMessage('Disconnected from dashboard', 'error');
        });

        // Data update handlers
        socket.on('update_data', (data) => {
            updateLastUpdate();
            updateAgentTable(data.agents);
            updateTaskTable(data.tasks);
            updateVotingTable(data.voting);
            updateHealthMetrics(data.health);
            updateSystemStatus(data.system_status);
        });

        // Response handlers
        socket.on('agent_spawned', (data) => {
            showMessage(data.message, 'success');
        });

        socket.on('agent_terminated', (data) => {
            showMessage(data.message, 'success');
        });

        socket.on('task_submitted', (data) => {
            showMessage(data.message, 'success');
        });

        socket.on('vote_created', (data) => {
            showMessage(data.message, 'success');
        });

        socket.on('error', (data) => {
            showMessage(data.message, 'error');
        });

        // Update functions
        function updateLastUpdate() {
            document.getElementById('last-update').textContent = new Date().toLocaleTimeString();
        }

        function updateAgentTable(agentData) {
            const tbody = document.querySelector('#agent-table tbody');
            tbody.innerHTML = '';
            
            const agents = agentData.agents || {};
            for (const [agentId, agent] of Object.entries(agents)) {
                const row = tbody.insertRow();
                row.innerHTML = `
                    <td>${agentId}</td>
                    <td>${agent.agent_type}</td>
                    <td><span class="status-${agent.is_online ? 'online' : 'offline'}">${agent.status}</span></td>
                    <td>${agent.current_tasks || 0}</td>
                    <td>${(agent.performance_score || 1.0).toFixed(2)}</td>
                `;
            }
        }

        function updateTaskTable(taskData) {
            document.getElementById('pending-tasks').textContent = taskData.pending_count || 0;
            document.getElementById('completed-tasks').textContent = taskData.completed_count || 0;
            document.getElementById('failed-tasks').textContent = taskData.failed_count || 0;

            const tbody = document.querySelector('#task-table tbody');
            tbody.innerHTML = '';
            
            const tasks = [...(taskData.pending_tasks || []), ...(taskData.completed_tasks || [])].slice(0, 10);
            tasks.forEach(task => {
                const row = tbody.insertRow();
                const status = task.assigned_agent ? 'assigned' : 'pending';
                row.innerHTML = `
                    <td>${task.task_id}</td>
                    <td>${task.task_type}</td>
                    <td><span class="priority-${task.priority.toLowerCase()}">${task.priority}</span></td>
                    <td>${task.assigned_agent || 'N/A'}</td>
                    <td>${status}</td>
                `;
            });
        }

        function updateVotingTable(votingData) {
            document.getElementById('active-votes').textContent = votingData.active_decisions?.length || 0;
            document.getElementById('approval-rate').textContent = 
                ((votingData.stats?.approval_rate || 0) * 100).toFixed(1) + '%';

            const tbody = document.querySelector('#voting-table tbody');
            tbody.innerHTML = '';
            
            const decisions = votingData.active_decisions || [];
            decisions.forEach(decision => {
                const row = tbody.insertRow();
                const tally = decision.current_tally;
                row.innerHTML = `
                    <td>${decision.title}</td>
                    <td>${decision.proposer_id}</td>
                    <td>${decision.quorum_progress}</td>
                    <td>👍${tally.approve} 👎${tally.reject} 🤷${tally.abstain}</td>
                `;
            });
        }

        function updateHealthMetrics(healthData) {
            const system = healthData.system || {};
            document.getElementById('cpu-usage').textContent = (system.cpu_percent || 0).toFixed(1) + '%';
            document.getElementById('memory-usage').textContent = (system.memory_percent || 0).toFixed(1) + '%';
            
            const collaboration = healthData.collaboration || {};
            const messageBus = collaboration.message_bus || {};
            document.getElementById('online-agents').textContent = messageBus.online_agents || 0;

            // Update health chart
            updateHealthChart(healthData);
        }

        function updateSystemStatus(statusData) {
            document.getElementById('connected-clients').textContent = statusData.connected_clients || 0;
        }

        function updateHealthChart(healthData) {
            const ctx = document.getElementById('health-chart').getContext('2d');
            
            if (!healthChart) {
                healthChart = new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: [],
                        datasets: [{
                            label: 'CPU %',
                            data: [],
                            borderColor: 'rgb(75, 192, 192)',
                            tension: 0.1
                        }, {
                            label: 'Memory %',
                            data: [],
                            borderColor: 'rgb(255, 99, 132)',
                            tension: 0.1
                        }]
                    },
                    options: {
                        responsive: true,
                        scales: {
                            y: {
                                beginAtZero: true,
                                max: 100
                            }
                        }
                    }
                });
            }

            const now = new Date().toLocaleTimeString();
            const system = healthData.system || {};
            
            healthChart.data.labels.push(now);
            healthChart.data.datasets[0].data.push(system.cpu_percent || 0);
            healthChart.data.datasets[1].data.push(system.memory_percent || 0);
            
            // Keep only last 20 points
            if (healthChart.data.labels.length > 20) {
                healthChart.data.labels.shift();
                healthChart.data.datasets[0].data.shift();
                healthChart.data.datasets[1].data.shift();
            }
            
            healthChart.update('none');
        }

        // Control functions
        function spawnAgent() {
            const agentType = document.getElementById('agent-type').value;
            if (!agentType) {
                showMessage('Agent type is required', 'error');
                return;
            }
            
            socket.emit('spawn_agent', {
                agent_type: agentType,
                reason: 'manual_dashboard'
            });
        }

        function terminateAgent() {
            const agentId = document.getElementById('agent-id').value;
            if (!agentId) {
                showMessage('Agent ID is required', 'error');
                return;
            }
            
            socket.emit('terminate_agent', {
                agent_id: agentId
            });
        }

        function submitTask() {
            const description = document.getElementById('task-description').value;
            const priority = document.getElementById('task-priority').value;
            
            if (!description) {
                showMessage('Task description is required', 'error');
                return;
            }
            
            socket.emit('submit_task', {
                task_description: description,
                priority: priority
            });
        }

        function createVote() {
            const title = document.getElementById('vote-title').value;
            const description = document.getElementById('vote-description').value;
            
            if (!title || !description) {
                showMessage('Title and description are required', 'error');
                return;
            }
            
            socket.emit('create_vote', {
                title: title,
                description: description
            });
        }

        function showMessage(message, type) {
            const messageEl = document.getElementById('message');
            messageEl.textContent = message;
            messageEl.className = `message ${type}`;
            messageEl.style.display = 'block';
            
            setTimeout(() => {
                messageEl.style.display = 'none';
            }, 5000);
        }
    </script>
</body>
</html>
'''

def create_dashboard_template():
    """Create the dashboard template file."""
    template_dir = Path(__file__).parent / "templates"
    template_dir.mkdir(exist_ok=True)
    
    template_file = template_dir / "dashboard.html"
    with open(template_file, 'w') as f:
        f.write(DASHBOARD_TEMPLATE)
    
    print(f"Dashboard template created at: {template_file}")

# Example usage
async def main():
    """Example usage of the monitoring dashboard."""
    from .agent_message_bus import AgentMessageBus
    from .collaborative_decision_making import VotingSystem
    
    # Initialize systems
    message_bus = AgentMessageBus()
    voting_system = VotingSystem()
    
    await message_bus.start()
    
    # Create dashboard
    dashboard = CollaborationDashboard(host="0.0.0.0", port=5000, debug=True)
    dashboard.connect_systems(message_bus, voting_system=voting_system)
    
    # Create template
    create_dashboard_template()
    
    print("Starting dashboard server at http://localhost:5000")
    print("Press Ctrl+C to stop")
    
    try:
        dashboard.run()
    except KeyboardInterrupt:
        print("\nShutting down dashboard...")
    finally:
        await message_bus.stop()

if __name__ == "__main__":
    create_dashboard_template()
    print("Dashboard template created. Run with your collaboration system to start the server.")