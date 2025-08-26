#!/usr/bin/env python3
"""
Integration test for Listen v7 with PlanRunner.

Tests the full agentic architecture with:
- PlanRunner orchestration
- Async-to-sync agent wrappers
- Blackboard communication
- Multi-agent pipeline execution
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from datetime import datetime

# Add Talk root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from listen.versions.listen_v7 import ListenV7
from plan_runner.blackboard import Blackboard
from tests.utilities.test_output_writer import TestOutputWriter

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)


async def test_planrunner_integration():
    """Test PlanRunner integration with Listen v7."""
    writer = TestOutputWriter("integration", "test_listen_v7_planrunner")
    output_dir = writer.get_output_dir()
    
    results = {
        "test_name": "Listen v7 PlanRunner Integration",
        "timestamp": datetime.now().isoformat(),
        "tests_run": 0,
        "tests_passed": 0,
        "tests_failed": 0,
        "details": []
    }
    
    try:
        # Test 1: Initialize Listen v7 with agents
        log.info("Test 1: Initializing Listen v7 with agents...")
        listen = ListenV7(
            name="Listen v7 Test",
            service_tier="standard"
        )
        
        # Verify agents are registered with sync wrappers
        assert len(listen.agents) > 0, "No agents registered"
        assert "voice_processor" in listen.agents, "Voice processor not found"
        assert "mcp_executor" in listen.agents, "MCP executor not found"
        
        # Verify all agents have run() method
        for agent_name, agent in listen.agents.items():
            assert hasattr(agent, 'run'), f"Agent {agent_name} missing run() method"
        
        results["tests_run"] += 1
        results["tests_passed"] += 1
        results["details"].append({
            "test": "Agent registration",
            "status": "passed",
            "agents_registered": len(listen.agents)
        })
        log.info("✓ Agent registration successful")
        
        # Test 2: Test sync wrapper execution
        log.info("Test 2: Testing synchronous agent wrapper...")
        voice_processor = listen.agents["voice_processor"]
        
        # Test with mock audio data (pass length, not bytes)
        test_prompt = json.dumps({
            "audio_data_length": 100,
            "service_tier": "standard"
        })
        
        result = voice_processor.run(test_prompt)
        result_data = json.loads(result)
        
        assert "transcript" in result_data, "No transcript in result"
        assert "confidence" in result_data, "No confidence in result"
        assert "service_used" in result_data, "No service_used in result"
        
        results["tests_run"] += 1
        results["tests_passed"] += 1
        results["details"].append({
            "test": "Sync wrapper execution",
            "status": "passed",
            "transcript": result_data.get("transcript", ""),
            "service": result_data.get("service_used", "unknown")
        })
        log.info(f"✓ Sync wrapper test: {result_data['transcript']}")
        
        # Test 3: Test plan creation
        log.info("Test 3: Testing plan creation...")
        plans = listen._load_plans()
        
        assert len(plans) > 0, "No plans loaded"
        assert "voice_command" in plans, "Voice command plan not found"
        assert "quick_action" in plans, "Quick action plan not found"
        
        # Verify plan structure
        voice_plan = plans["voice_command"]
        assert "steps" in voice_plan, "No steps in voice command plan"
        assert len(voice_plan["steps"]) > 0, "Empty steps in voice command plan"
        
        results["tests_run"] += 1
        results["tests_passed"] += 1
        results["details"].append({
            "test": "Plan creation",
            "status": "passed",
            "plans_loaded": len(plans),
            "plan_names": list(plans.keys())
        })
        log.info(f"✓ Plan creation successful: {len(plans)} plans")
        
        # Test 4: Test Step conversion
        log.info("Test 4: Testing Step object conversion...")
        steps = listen._create_steps_from_plan(voice_plan)
        
        assert len(steps) > 0, "No steps created"
        for step in steps:
            assert hasattr(step, 'label'), "Step missing label"
            assert hasattr(step, 'agent_key'), "Step missing agent_key"
            assert step.label, "Step has empty label"
            
        results["tests_run"] += 1
        results["tests_passed"] += 1
        results["details"].append({
            "test": "Step conversion",
            "status": "passed",
            "steps_created": len(steps),
            "step_labels": [s.label for s in steps]
        })
        log.info(f"✓ Step conversion successful: {len(steps)} steps")
        
        # Test 5: Test Blackboard integration
        log.info("Test 5: Testing Blackboard communication...")
        blackboard = Blackboard()
        
        # Set initial data
        test_audio = b"test_audio_data"
        blackboard.add("audio_data", test_audio)
        blackboard.add("service_tier", "standard")
        
        # Verify data retrieval
        retrieved_audio = blackboard.query_sync(label="audio_data")
        assert len(retrieved_audio) > 0, "Could not retrieve audio from blackboard"
        assert retrieved_audio[0].content == test_audio, "Audio data mismatch"
        
        results["tests_run"] += 1
        results["tests_passed"] += 1
        results["details"].append({
            "test": "Blackboard communication",
            "status": "passed",
            "entries_added": 2,
            "retrieval_successful": True
        })
        log.info("✓ Blackboard communication successful")
        
        # Test 6: Test full pipeline execution
        log.info("Test 6: Testing full pipeline execution...")
        
        # Create mock audio data
        mock_audio = b"mock_audio_for_testing" * 100  # Simulate real audio
        
        # Process through pipeline
        try:
            response = await listen.process_audio(mock_audio)
            
            assert response, "No response from pipeline"
            assert isinstance(response, str), "Response not a string"
            
            # Check session stats
            stats = listen.get_stats()
            assert stats["total_requests"] > 0, "No requests recorded"
            
            results["tests_run"] += 1
            results["tests_passed"] += 1
            results["details"].append({
                "test": "Full pipeline execution",
                "status": "passed",
                "response": response[:100],
                "total_requests": stats["total_requests"],
                "most_used_agent": stats.get("most_used_agent", "none")
            })
            log.info(f"✓ Pipeline execution: {response[:50]}...")
            
        except Exception as e:
            results["tests_run"] += 1
            results["tests_failed"] += 1
            results["details"].append({
                "test": "Full pipeline execution",
                "status": "failed",
                "error": str(e)
            })
            log.error(f"✗ Pipeline execution failed: {e}")
        
        # Test 7: Test multi-agent coordination
        log.info("Test 7: Testing multi-agent coordination...")
        
        # Test intent detection chain
        intent_detector = listen.agents["intent_detector"]
        test_transcript = "list my files"
        
        intent_result = intent_detector.run(test_transcript)
        intent_data = json.loads(intent_result)
        
        assert "intent_type" in intent_data, "No intent type"
        
        # Test command execution chain
        if intent_data.get("intent_type") == "ACTION":
            mcp_executor = listen.agents["mcp_executor"]
            command_result = mcp_executor.run("ls")
            command_data = json.loads(command_result)
            
            assert "success" in command_data, "No success flag in command result"
            
            # Test response generation chain
            response_gen = listen.agents["response_generator"]
            response_context = json.dumps({
                "transcript": test_transcript,
                "command_result": command_data
            })
            
            final_response = response_gen.run(response_context)
            assert final_response, "No final response generated"
            
        results["tests_run"] += 1
        results["tests_passed"] += 1
        results["details"].append({
            "test": "Multi-agent coordination",
            "status": "passed",
            "intent_detected": intent_data.get("intent_type", "unknown"),
            "chain_completed": True
        })
        log.info("✓ Multi-agent coordination successful")
        
    except Exception as e:
        log.error(f"Test suite failed: {e}")
        results["tests_failed"] += 1
        results["details"].append({
            "test": "Test suite",
            "status": "error",
            "error": str(e)
        })
    
    # Write results
    writer.write_results(results)
    
    # Print summary
    print("\n" + "="*60)
    print(f"Listen v7 PlanRunner Integration Test Results")
    print("="*60)
    print(f"Tests run: {results['tests_run']}")
    print(f"Tests passed: {results['tests_passed']}")
    print(f"Tests failed: {results['tests_failed']}")
    print(f"Success rate: {results['tests_passed']/results['tests_run']*100:.1f}%")
    
    if results['tests_failed'] > 0:
        print("\nFailed tests:")
        for detail in results['details']:
            if detail['status'] == 'failed':
                print(f"  - {detail['test']}: {detail.get('error', 'Unknown error')}")
    
    print(f"\nResults saved to: {output_dir}")
    
    return results['tests_failed'] == 0


def main():
    """Run the test suite."""
    success = asyncio.run(test_planrunner_integration())
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()