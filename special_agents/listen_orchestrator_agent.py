#!/usr/bin/env python3
"""
Listen Orchestrator Agent

Master coordinator for Listen v7's fully agentic architecture.
Decides which pipeline to use and coordinates agent execution.
"""

import logging
from typing import Dict, List, Optional, Any
from agent.agent import Agent

log = logging.getLogger(__name__)


class ListenOrchestratorAgent(Agent):
    """
    Master orchestrator for Listen pipeline.
    
    Responsibilities:
    - Select appropriate execution plan
    - Monitor agent health
    - Handle fallback strategies
    - Coordinate inter-agent communication
    """
    
    def __init__(self, config: Dict[str, Any] = None, plan_runner=None, **kwargs):
        roles = [
            "You orchestrate voice processing pipelines",
            "You select optimal execution plans based on context",
            "You monitor agent health and handle failures",
            "You coordinate communication between specialized agents",
            "You optimize for speed, accuracy, and cost based on configuration"
        ]
        
        super().__init__(roles=roles, **kwargs)
        
        self.config = config or {}
        self.plan_runner = plan_runner
        self.agent_health = {}
        self.plan_history = []
        
        log.info("ListenOrchestratorAgent initialized")
    
    async def initialize(self):
        """Initialize all managed agents."""
        if self.plan_runner:
            # Check health of all registered agents
            for agent_name, agent in self.plan_runner.agents.items():
                if hasattr(agent, 'health_check'):
                    health = await agent.health_check()
                    self.agent_health[agent_name] = health
                else:
                    self.agent_health[agent_name] = {"status": "unknown"}
            
            log.info(f"Agent health check complete: {len(self.agent_health)} agents")
    
    async def select_plan(self, context: Any) -> str:
        """
        Select the optimal execution plan based on context.
        
        Decision factors:
        - Audio length
        - Service availability
        - Previous intent patterns
        - Performance requirements
        """
        # Simple heuristic for now
        audio_data = getattr(context, 'audio_data', None)
        
        if not audio_data:
            return "conversation"
        
        # Check audio length
        audio_length = len(audio_data) if audio_data else 0
        
        # Quick commands are typically short
        if audio_length < 50000:  # ~3 seconds at 16kHz
            # Check if MCP executor is healthy
            if self.agent_health.get("mcp_executor", {}).get("status") == "healthy":
                return "quick_action"
        
        # Check conversation history for patterns
        history = getattr(context, 'conversation_history', [])
        if history:
            recent_intents = [turn.get("intent_type") for turn in history[-3:]]
            if all(intent == "CONVERSATION" for intent in recent_intents):
                return "conversation"
        
        # Default to full pipeline
        return "voice_command"
    
    async def run(self, prompt: str, context: Dict[str, Any] = None) -> str:
        """
        Main orchestration logic.
        
        This is called when the orchestrator needs to make decisions
        during pipeline execution.
        """
        # Analyze the prompt to determine action
        if "select_plan" in prompt:
            # Plan selection logic
            return await self.select_plan(context)
        
        elif "handle_failure" in prompt:
            # Failure recovery logic
            failed_agent = context.get("failed_agent")
            error = context.get("error")
            
            log.warning(f"Agent {failed_agent} failed: {error}")
            
            # Determine fallback strategy
            if failed_agent == "voice_processor":
                return "use_fallback_transcription"
            elif failed_agent == "mcp_executor":
                return "skip_command_execution"
            else:
                return "continue_pipeline"
        
        elif "optimize_pipeline" in prompt:
            # Pipeline optimization based on performance metrics
            metrics = context.get("metrics", {})
            
            if metrics.get("latency_ms", 0) > 1000:
                return "switch_to_quick_action"
            elif metrics.get("error_rate", 0) > 0.1:
                return "enable_conservative_mode"
            else:
                return "maintain_current_pipeline"
        
        else:
            # Default response
            return f"Orchestrating: {prompt}"
    
    async def monitor_execution(self, plan_name: str, step_results: Dict[str, Any]):
        """
        Monitor plan execution and intervene if needed.
        
        Called by PlanRunner during execution for real-time monitoring.
        """
        # Track execution metrics
        self.plan_history.append({
            "plan": plan_name,
            "timestamp": step_results.get("timestamp"),
            "duration_ms": step_results.get("duration_ms"),
            "success": step_results.get("success")
        })
        
        # Check for issues
        if not step_results.get("success"):
            # Trigger recovery
            recovery_action = await self.run(
                "handle_failure",
                {"failed_agent": step_results.get("agent"), "error": step_results.get("error")}
            )
            return recovery_action
        
        return "continue"
    
    async def coordinate_agents(self, source_agent: str, target_agent: str, message: Any):
        """
        Facilitate communication between agents.
        
        This implements Talk's inter-agent communication protocol.
        """
        if self.plan_runner:
            # Route message through PlanRunner
            if target_agent in self.plan_runner.agents:
                target = self.plan_runner.agents[target_agent]
                if hasattr(target, 'receive_message'):
                    response = await target.receive_message(source_agent, message)
                    return response
        
        log.warning(f"Cannot route message from {source_agent} to {target_agent}")
        return None
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get statistics about pipeline execution."""
        if not self.plan_history:
            return {"total_executions": 0}
        
        # Calculate statistics
        total = len(self.plan_history)
        successful = sum(1 for p in self.plan_history if p.get("success"))
        
        # Average latency by plan
        plan_latencies = {}
        for record in self.plan_history:
            plan = record["plan"]
            duration = record.get("duration_ms", 0)
            if plan not in plan_latencies:
                plan_latencies[plan] = []
            plan_latencies[plan].append(duration)
        
        avg_latencies = {
            plan: sum(durations) / len(durations)
            for plan, durations in plan_latencies.items()
        }
        
        return {
            "total_executions": total,
            "successful_executions": successful,
            "success_rate": successful / total if total > 0 else 0,
            "average_latencies": avg_latencies,
            "agent_health": self.agent_health
        }
    
    async def cleanup(self):
        """Clean up orchestrator resources."""
        log.info("ListenOrchestratorAgent cleanup complete")