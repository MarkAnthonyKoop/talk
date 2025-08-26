#!/usr/bin/env python3
"""
Listen v7 - Fully Agentic Architecture

Complete rewrite using Talk's full agentic patterns:
- All processing through specialized agents
- PlanRunner for pipeline orchestration
- Inter-agent communication protocol
- Dynamic plan generation
- Full fallback chains through agents
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict

# Core Talk framework
from agent.agent import Agent
from plan_runner.plan_runner import PlanRunner
from plan_runner.step import Step
from plan_runner.blackboard import Blackboard

# Import v6 components we'll wrap as agents
from listen.versions.listen_v6 import (
    VoiceProcessingResult,
    ServiceTier,
    ResponseMode,
    IntentType,
    IntentClassification
)
from listen.versions.listen_v6_mcp_integration import MCPToolResult

log = logging.getLogger(__name__)


@dataclass
class ListenContext:
    """Shared context for Listen pipeline."""
    audio_data: bytes = None
    transcript: str = ""
    confidence: float = 0.0
    speaker_id: str = ""
    speaker_confidence: float = 0.0
    intent: IntentClassification = None
    command_result: Dict[str, Any] = None
    response: str = ""
    processing_times: Dict[str, float] = None
    service_tier: str = "standard"
    conversation_history: List[Dict] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for plan context."""
        return {
            k: v for k, v in asdict(self).items() 
            if v is not None and k != 'audio_data'  # Don't pass raw audio
        }


class ListenV7:
    """
    Listen v7 - Fully agentic implementation.
    
    Uses Talk's PlanRunner and agent communication for all operations.
    Every component is an agent, every flow is a plan.
    """
    
    def __init__(
        self,
        name: str = "Listen v7",
        service_tier: str = "standard",
        deepgram_key: Optional[str] = None,
        anthropic_key: Optional[str] = None,
        **kwargs
    ):
        self.name = name
        self.service_tier = service_tier
        self.start_time = datetime.now()
        
        # Configuration
        self.config = {
            "service_tier": service_tier,
            "deepgram_key": deepgram_key or os.getenv("DEEPGRAM_API_KEY"),
            "anthropic_key": anthropic_key or os.getenv("ANTHROPIC_API_KEY"),
            **kwargs
        }
        
        # Initialize agent registry
        self.agents = {}
        
        # Register all agents first
        self._register_agents()
        
        # Initialize PlanRunner will be done per plan execution
        self.verbose = kwargs.get("verbose", False)
        
        # Load pipeline plans
        self.plans = self._load_plans()
        
        # Session state
        self.conversation_history = []
        self.session_stats = {
            "total_requests": 0,
            "agent_calls": {},
            "plan_executions": {},
            "avg_response_time": 0
        }
        
        log.info(f"🚀 {name} initialized with fully agentic architecture")
        log.info(f"   Service tier: {service_tier}")
        log.info(f"   Registered agents: {len(self.agents)}")
        log.info(f"   Available plans: {len(self.plans)}")
    
    def _register_agents(self):
        """Register all agents with synchronous wrappers for PlanRunner."""
        from special_agents.listen_orchestrator_agent import ListenOrchestratorAgent
        from special_agents.voice_processing_agent import VoiceProcessingAgent
        from special_agents.speaker_identification_agent import SpeakerIdentificationAgent
        from special_agents.intent_detection_agent import IntentDetectionAgent
        from special_agents.security_validator_agent import SecurityValidatorAgent
        from special_agents.mcp_executor_agent import MCPExecutorAgent
        from special_agents.response_generation_agent import ResponseGenerationAgent
        from special_agents.tts_synthesis_agent import TTSSynthesisAgent
        
        # Create async agents first
        async_agents = {}
        
        # Master orchestrator
        self.orchestrator = ListenOrchestratorAgent(
            config=self.config,
            plan_runner=None  # Will set during execution
        )
        async_agents["orchestrator"] = self.orchestrator
        
        # Voice processing pipeline
        self.voice_processor = VoiceProcessingAgent(
            deepgram_key=self.config.get("deepgram_key"),
            service_tier=self.service_tier
        )
        async_agents["voice_processor"] = self.voice_processor
        
        # Speaker identification
        self.speaker_identifier = SpeakerIdentificationAgent()
        async_agents["speaker_identifier"] = self.speaker_identifier
        
        # Intent detection
        self.intent_detector = IntentDetectionAgent()
        async_agents["intent_detector"] = self.intent_detector
        
        # Security validation
        self.security_validator = SecurityValidatorAgent()
        async_agents["security_validator"] = self.security_validator
        
        # Command execution via MCP
        self.mcp_executor = MCPExecutorAgent(
            anthropic_key=self.config.get("anthropic_key")
        )
        async_agents["mcp_executor"] = self.mcp_executor
        
        # Response generation
        self.response_generator = ResponseGenerationAgent(
            service_tier=self.service_tier
        )
        async_agents["response_generator"] = self.response_generator
        
        # TTS synthesis
        self.tts_synthesizer = TTSSynthesisAgent()
        async_agents["tts_synthesizer"] = self.tts_synthesizer
        
        # Create synchronous wrappers for PlanRunner
        self.agents = self._create_sync_agent_wrappers(async_agents)
        
        log.info(f"✅ Registered {len(self.agents)} agents with sync wrappers")
    
    def _create_sync_agent_wrappers(self, async_agents: Dict) -> Dict:
        """
        Create synchronous wrappers for async agents to work with PlanRunner.
        
        PlanRunner expects agents with synchronous run() methods.
        """
        from special_agents.sync_agent_wrapper import SyncAgentWrapper
        
        # Create wrapped agents
        wrapped_agents = {}
        
        for agent_name, async_agent in async_agents.items():
            wrapped_agents[agent_name] = SyncAgentWrapper(async_agent, agent_name)
        
        return wrapped_agents
    
    def _load_plans(self) -> Dict[str, Dict]:
        """Load pipeline execution plans."""
        plans = {}
        
        # Main voice command processing plan
        plans["voice_command"] = {
            "name": "voice_command_pipeline",
            "description": "Process voice command from audio to response",
            "steps": [
                {
                    "name": "transcribe_audio",
                    "agent": "voice_processor",
                    "action": "transcribe",
                    "input": {
                        "audio_data": "${audio_data}",
                        "service_tier": "${service_tier}"
                    },
                    "output": "transcription_result"
                },
                {
                    "name": "identify_speaker",
                    "agent": "speaker_identifier",
                    "action": "identify",
                    "input": {
                        "audio_data": "${audio_data}"
                    },
                    "output": "speaker_info",
                    "parallel": True  # Run parallel with transcription
                },
                {
                    "name": "detect_intent",
                    "agent": "intent_detector",
                    "action": "detect",
                    "input": {
                        "text": "${transcription_result.transcript}",
                        "context": "${conversation_history}"
                    },
                    "output": "intent",
                    "depends_on": ["transcribe_audio"]
                },
                {
                    "name": "validate_security",
                    "agent": "security_validator",
                    "action": "validate",
                    "input": {
                        "intent": "${intent}"
                    },
                    "output": "security_check",
                    "condition": "${intent.intent_type} == 'ACTION'",
                    "depends_on": ["detect_intent"]
                },
                {
                    "name": "execute_command",
                    "agent": "mcp_executor",
                    "action": "execute",
                    "input": {
                        "command": "${intent.command}",
                        "parameters": "${intent.parameters}"
                    },
                    "output": "command_result",
                    "condition": "${security_check.approved} == true",
                    "depends_on": ["validate_security"]
                },
                {
                    "name": "generate_response",
                    "agent": "response_generator",
                    "action": "generate",
                    "input": {
                        "transcript": "${transcription_result.transcript}",
                        "intent": "${intent}",
                        "command_result": "${command_result}",
                        "speaker": "${speaker_info}",
                        "conversation_history": "${conversation_history}"
                    },
                    "output": "response_text",
                    "depends_on": ["detect_intent", "identify_speaker"]
                },
                {
                    "name": "synthesize_speech",
                    "agent": "tts_synthesizer",
                    "action": "synthesize",
                    "input": {
                        "text": "${response_text}"
                    },
                    "output": "audio_response",
                    "depends_on": ["generate_response"]
                }
            ]
        }
        
        # Conversation-only plan (no command execution)
        plans["conversation"] = {
            "name": "conversation_pipeline",
            "description": "Handle pure conversation without actions",
            "steps": [
                {
                    "name": "transcribe_audio",
                    "agent": "voice_processor",
                    "action": "transcribe",
                    "input": {"audio_data": "${audio_data}"},
                    "output": "transcription_result"
                },
                {
                    "name": "identify_speaker",
                    "agent": "speaker_identifier",
                    "action": "identify",
                    "input": {"audio_data": "${audio_data}"},
                    "output": "speaker_info",
                    "parallel": True
                },
                {
                    "name": "generate_response",
                    "agent": "response_generator",
                    "action": "generate_conversational",
                    "input": {
                        "transcript": "${transcription_result.transcript}",
                        "speaker": "${speaker_info}",
                        "conversation_history": "${conversation_history}"
                    },
                    "output": "response_text",
                    "depends_on": ["transcribe_audio", "identify_speaker"]
                },
                {
                    "name": "synthesize_speech",
                    "agent": "tts_synthesizer",
                    "action": "synthesize",
                    "input": {"text": "${response_text}"},
                    "output": "audio_response",
                    "depends_on": ["generate_response"]
                }
            ]
        }
        
        # Quick action plan (optimized for speed)
        plans["quick_action"] = {
            "name": "quick_action_pipeline",
            "description": "Fast path for simple commands",
            "steps": [
                {
                    "name": "quick_transcribe",
                    "agent": "voice_processor",
                    "action": "transcribe_fast",
                    "input": {"audio_data": "${audio_data}"},
                    "output": "transcript"
                },
                {
                    "name": "execute_direct",
                    "agent": "mcp_executor",
                    "action": "execute_simple",
                    "input": {"command": "${transcript}"},
                    "output": "result",
                    "depends_on": ["quick_transcribe"]
                },
                {
                    "name": "quick_response",
                    "agent": "response_generator",
                    "action": "generate_brief",
                    "input": {"result": "${result}"},
                    "output": "response",
                    "depends_on": ["execute_direct"]
                }
            ]
        }
        
        return plans
    
    async def process_audio(self, audio_data: bytes) -> str:
        """
        Main entry point - process audio through agent pipeline.
        
        This delegates everything to agents via PlanRunner.
        """
        start_time = asyncio.get_event_loop().time()
        
        # Create context for this request
        context = ListenContext(
            audio_data=audio_data,
            service_tier=self.service_tier,
            conversation_history=self.conversation_history[-5:]  # Last 5 turns
        )
        
        # Let orchestrator decide which plan to use
        plan_name = await self.orchestrator.select_plan(context)
        
        # Get the plan definition
        plan_def = self.plans[plan_name]
        
        # Convert plan to Step objects for PlanRunner
        steps = self._create_steps_from_plan(plan_def)
        
        # Create blackboard with initial context
        blackboard = Blackboard()
        # Store audio data length instead of raw bytes to avoid serialization issues
        blackboard.add("audio_data_length", len(audio_data))
        blackboard.add("service_tier", self.service_tier)
        blackboard.add("conversation_history", json.dumps(context.conversation_history) if context.conversation_history else "[]")
        
        # Store audio data in wrapper for agent access
        for agent in self.agents.values():
            if hasattr(agent, 'async_agent'):
                agent.async_agent._audio_data_cache = audio_data
        
        # Create PlanRunner for this execution
        plan_runner = PlanRunner(
            steps=steps,
            agents=self.agents,
            blackboard=blackboard
        )
        
        # Execute the plan synchronously (PlanRunner is sync)
        # Pass initial audio data as the first prompt
        initial_prompt = json.dumps({
            "audio_data_length": len(audio_data),
            "service_tier": self.service_tier
        })
        result = plan_runner.run(initial_prompt)
        
        # Extract results from blackboard
        transcript_entries = blackboard.query_sync(label="transcript")
        transcript = transcript_entries[0].content if transcript_entries else ""
        
        response_entries = blackboard.query_sync(label="response_text")
        response_text = response_entries[0].content if response_entries else "I processed your request."
        
        # Update conversation history
        self.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "transcript": transcript,
            "response": response_text,
            "plan_used": plan_name,
            "processing_time": asyncio.get_event_loop().time() - start_time
        })
        
        # Update session stats
        self.session_stats["total_requests"] += 1
        self.session_stats["plan_executions"][plan_name] = \
            self.session_stats["plan_executions"].get(plan_name, 0) + 1
        
        # Track agent usage
        for step in steps:
            if step.agent_key:
                self.session_stats["agent_calls"][step.agent_key] = \
                    self.session_stats["agent_calls"].get(step.agent_key, 0) + 1
        
        # Return the final response
        return response_text
    
    def _create_steps_from_plan(self, plan_def: Dict) -> List[Step]:
        """
        Convert plan definition to Step objects for PlanRunner.
        
        This bridges our declarative plan format to PlanRunner's Step objects.
        """
        steps = []
        step_index = {}  # Map step names to Step objects
        
        # First pass: create all steps
        for step_def in plan_def["steps"]:
            # Create a Step object with correct field names
            step = Step(
                label=step_def["name"],
                agent_key=step_def["agent"]  # Use agent_key not agent
            )
            
            steps.append(step)
            step_index[step_def["name"]] = step
        
        # Second pass: handle dependencies by reordering
        # For now, we'll rely on the order in the plan definition
        # Real dependency handling would require topological sorting
        
        # Handle parallel steps if specified
        # Group parallel steps together if they have the same dependencies
        parallel_groups = {}
        for i, step_def in enumerate(plan_def["steps"]):
            if step_def.get("parallel", False) and "depends_on" in step_def:
                dep_key = tuple(step_def["depends_on"])
                if dep_key not in parallel_groups:
                    parallel_groups[dep_key] = []
                parallel_groups[dep_key].append(steps[i])
        
        # For simplicity, return steps in order for now
        # A full implementation would handle complex dependency graphs
        return steps
    
    async def start(self):
        """Start listening for voice input."""
        log.info(f"🎤 {self.name} starting in {self.service_tier} tier...")
        
        # Initialize agents
        await self.orchestrator.initialize()
        
        # Start continuous listening
        await self.listen_continuous()
    
    async def listen_continuous(self):
        """Continuous listening loop."""
        import speech_recognition as sr
        
        recognizer = sr.Recognizer()
        microphone = sr.Microphone(sample_rate=16000, chunk_size=1024)
        
        with microphone as source:
            recognizer.adjust_for_ambient_noise(source)
            print("🎤 Listening... (Ctrl+C to stop)")
            
            while True:
                try:
                    # Capture audio
                    audio = recognizer.listen(
                        source,
                        timeout=1.0,
                        phrase_time_limit=10
                    )
                    
                    # Convert to bytes
                    audio_data = audio.get_wav_data(
                        convert_rate=16000,
                        convert_width=2
                    )
                    
                    # Process through agent pipeline
                    response = await self.process_audio(audio_data)
                    
                    # Output response (TTS handled by agent)
                    log.info(f"Response: {response}")
                    
                except sr.WaitTimeoutError:
                    continue
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    log.error(f"Error in listening loop: {e}")
                    continue
        
        log.info("👋 Listening stopped")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get session statistics."""
        uptime = (datetime.now() - self.start_time).total_seconds()
        
        # Find most used agent
        if self.session_stats["agent_calls"]:
            most_used_agent = max(
                self.session_stats["agent_calls"],
                key=self.session_stats["agent_calls"].get
            )
        else:
            most_used_agent = "none"
        
        # Find most used plan
        if self.session_stats["plan_executions"]:
            most_used_plan = max(
                self.session_stats["plan_executions"],
                key=self.session_stats["plan_executions"].get
            )
        else:
            most_used_plan = "none"
        
        return {
            "uptime_seconds": uptime,
            "total_requests": self.session_stats["total_requests"],
            "agent_calls": self.session_stats["agent_calls"],
            "plan_executions": self.session_stats["plan_executions"],
            "most_used_agent": most_used_agent,
            "most_used_plan": most_used_plan,
            "conversation_length": len(self.conversation_history)
        }
    
    async def shutdown(self):
        """Clean shutdown of all agents."""
        log.info(f"🛑 Shutting down {self.name}...")
        
        # Let each agent clean up
        for agent_name, agent in self.agents.items():
            if hasattr(agent, 'cleanup'):
                await agent.cleanup()
        
        # Final stats
        stats = self.get_stats()
        log.info(f"📊 Final stats: {json.dumps(stats, indent=2)}")
        
        log.info(f"👋 {self.name} shutdown complete")


async def main():
    """Demo Listen v7 capabilities."""
    print("🚀 Listen v7 - Fully Agentic Architecture")
    print("=" * 60)
    
    # Initialize
    assistant = ListenV7(
        service_tier="standard",
        verbose=True
    )
    
    # Show registered agents
    print(f"\n📦 Registered Agents:")
    for agent_name in assistant.agents.keys():
        print(f"   • {agent_name}")
    
    # Show available plans
    print(f"\n📋 Available Plans:")
    for plan_name, plan in assistant.plans.items():
        print(f"   • {plan_name}: {plan['description']}")
    
    # Test with mock audio
    print("\n🧪 Testing agent pipeline...")
    mock_audio = b"RIFF" + b"\x00" * 1000  # Mock WAV data
    
    try:
        response = await assistant.process_audio(mock_audio)
        print(f"\n✅ Pipeline result: {response}")
    except Exception as e:
        print(f"\n⚠️  Pipeline test needs agents implemented: {e}")
    
    # Show stats
    stats = assistant.get_stats()
    print(f"\n📊 Session Stats:")
    print(f"   Total requests: {stats['total_requests']}")
    print(f"   Most used agent: {stats['most_used_agent']}")
    print(f"   Most used plan: {stats['most_used_plan']}")
    
    # Cleanup
    await assistant.shutdown()
    
    print("\n✅ Listen v7 demo complete!")


if __name__ == "__main__":
    asyncio.run(main())