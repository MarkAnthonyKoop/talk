#!/usr/bin/env python3
"""
Test suite for Listen v2 components.

Tests the advanced personal assistant architecture including:
- ConversationManager with speaker tracking
- InformationOrganizer with categorization
- InterjectionAgent confidence logic
- Multi-source orchestration
"""

import sys
import json
import asyncio
from pathlib import Path
from datetime import datetime, timedelta
import unittest
from unittest.mock import Mock, MagicMock, patch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from utilities.test_output_writer import TestOutputWriter
from special_agents.conversation_manager import ConversationManager, Speaker
from special_agents.information_organizer import InformationOrganizer
from special_agents.interjection_agent import InterjectionAgent
from special_agents.input_sources import InputSource, MultiSourceOrchestrator


class TestConversationManager(unittest.TestCase):
    """Test conversation tracking and speaker identification."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.writer = TestOutputWriter("unit", "test_conversation_manager")
        self.output_dir = self.writer.get_output_dir()
        self.conv_manager = ConversationManager(
            conversation_id="test_conv",
            save_path=self.output_dir / "conversations"
        )
    
    def test_add_speakers(self):
        """Test adding speakers to conversation."""
        # Add speakers
        speaker1 = self.conv_manager.add_speaker("user", "Alice")
        speaker2 = self.conv_manager.add_speaker("assistant", "Assistant")
        
        # Verify speakers
        self.assertEqual(len(self.conv_manager.speakers), 2)
        self.assertEqual(speaker1.name, "Alice")
        self.assertEqual(speaker2.name, "Assistant")
        
        # Log results
        self.writer.write_log("Added 2 speakers successfully")
    
    def test_conversation_turns(self):
        """Test adding conversation turns."""
        # Add speakers
        self.conv_manager.add_speaker("user", "User")
        self.conv_manager.add_speaker("assistant", "Assistant")
        
        # Add turns
        turn1 = self.conv_manager.add_turn(
            text="Hello, how are you?",
            speaker_id="user"
        )
        
        turn2 = self.conv_manager.add_turn(
            text="I'm doing well, thank you!",
            speaker_id="assistant"
        )
        
        # Verify turns
        self.assertEqual(len(self.conv_manager.turns), 2)
        self.assertEqual(turn1.speaker_id, "user")
        self.assertEqual(turn2.speaker_id, "assistant")
        
        # Test context retrieval
        context = self.conv_manager.get_context(num_turns=2)
        self.assertEqual(len(context), 2)
        self.assertEqual(context[0]["speaker"], "User")
        
        # Write results
        self.writer.write_json("conversation.json", {
            "turns": [turn.to_dict() for turn in self.conv_manager.turns],
            "context": context
        })
    
    def test_speaker_identification(self):
        """Test simplified speaker identification."""
        # Test with audio features
        features_high = {"pitch": 250}
        speaker_id = self.conv_manager.identify_speaker(audio_features=features_high)
        self.assertEqual(speaker_id, "speaker_1")
        
        features_low = {"pitch": 100}
        speaker_id = self.conv_manager.identify_speaker(audio_features=features_low)
        self.assertEqual(speaker_id, "speaker_2")
        
        # Log results
        self.writer.write_log("Speaker identification tests passed")
    
    def test_conversation_analysis(self):
        """Test conversation analysis features."""
        # Create a conversation
        self.conv_manager.add_speaker("alice", "Alice")
        self.conv_manager.add_speaker("bob", "Bob")
        
        # Alice talks more
        for i in range(5):
            self.conv_manager.add_turn(
                f"Alice says something {i}",
                speaker_id="alice"
            )
        
        # Bob asks questions
        self.conv_manager.add_turn(
            "Bob asks: What do you think?",
            speaker_id="bob"
        )
        
        # Analyze
        analysis = self.conv_manager.analyze_conversation()
        
        # Verify analysis
        self.assertEqual(analysis["total_turns"], 6)
        self.assertEqual(analysis["speaker_count"], 2)
        self.assertEqual(analysis["dominant_speaker"], "Alice")
        
        # Check speaker stats
        alice_stats = analysis["speakers"]["Alice"]
        self.assertEqual(alice_stats["utterance_count"], 5)
        self.assertGreater(alice_stats["participation_rate"], 0.8)
        
        bob_stats = analysis["speakers"]["Bob"]
        self.assertEqual(bob_stats["questions_asked"], 1)
        
        # Write analysis
        self.writer.write_json("conversation_analysis.json", analysis)
    
    def test_conversation_persistence(self):
        """Test saving and loading conversations."""
        # Create conversation
        self.conv_manager.add_speaker("user", "User")
        self.conv_manager.add_turn("Test message", speaker_id="user")
        
        # Save
        self.conv_manager.save_conversation()
        
        # Create new manager and load
        new_manager = ConversationManager(save_path=self.output_dir / "conversations")
        new_manager.load_conversation("test_conv")
        
        # Verify loaded data
        self.assertEqual(len(new_manager.turns), 1)
        self.assertEqual(new_manager.turns[0].text, "Test message")
        
        self.writer.write_log("Conversation persistence test passed")


class TestInformationOrganizer(unittest.TestCase):
    """Test information organization and categorization."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.writer = TestOutputWriter("unit", "test_information_organizer")
        self.output_dir = self.writer.get_output_dir()
        self.organizer = InformationOrganizer(save_path=self.output_dir / "organized")
    
    def test_categorization(self):
        """Test automatic categorization."""
        test_cases = [
            ("I need to finish the project by tomorrow", "work"),
            ("Let's meet for coffee this weekend", "social"),
            ("Debug the API endpoint error", "technical"),
            ("Doctor appointment at 3pm", "health"),
            ("Reply to John's email about the proposal", "communication"),
        ]
        
        results = []
        for text, expected_category in test_cases:
            category, confidence = self.organizer.categorize(text)
            results.append({
                "text": text,
                "expected": expected_category,
                "actual": category,
                "confidence": confidence,
                "match": category == expected_category
            })
        
        # Write results
        self.writer.write_json("categorization_results.json", results)
        
        # Verify most categorizations are correct
        correct = sum(1 for r in results if r["match"])
        self.assertGreaterEqual(correct, 3)  # At least 3 out of 5 correct
    
    def test_information_organization(self):
        """Test organizing information items."""
        # Organize various items
        items = [
            "Meeting with client at 2pm",
            "Fix bug in authentication module",
            "Buy groceries for dinner",
            "Study machine learning tutorial",
            "Pay electricity bill"
        ]
        
        organized = []
        for content in items:
            item = self.organizer.organize(
                content=content,
                source="test",
                auto_categorize=True
            )
            organized.append(item.to_dict())
        
        # Verify organization
        self.assertEqual(self.organizer.stats["total_items"], 5)
        self.assertGreater(len(self.organizer.categories), 0)
        
        # Write organized items
        self.writer.write_json("organized_items.json", organized)
    
    def test_tag_extraction(self):
        """Test tag extraction from content."""
        content = "Check #important email from @john about https://example.com project"
        
        item = self.organizer.organize(
            content=content,
            source="test"
        )
        
        # Verify tags
        self.assertIn("#important", item.tags)
        self.assertIn("@john", item.tags)
        self.assertIn("#has_url", item.tags)
        
        self.writer.write_log(f"Extracted tags: {item.tags}")
    
    def test_information_retrieval(self):
        """Test retrieving organized information."""
        # Add test items
        for i in range(10):
            category = "work" if i % 2 == 0 else "personal"
            self.organizer.organize(
                content=f"Test item {i}",
                source="test",
                metadata={"category": category}
            )
        
        # Retrieve by category
        work_items = self.organizer.retrieve(category="work", limit=5)
        self.assertLessEqual(len(work_items), 5)
        
        # Retrieve by query
        query_items = self.organizer.retrieve(query="item 5", limit=3)
        self.assertGreater(len(query_items), 0)
        
        # Get summary
        summary = self.organizer.get_summary()
        self.assertEqual(summary["total_items"], 10)
        
        self.writer.write_json("retrieval_summary.json", summary)


class TestInterjectionAgent(unittest.TestCase):
    """Test interjection agent logic."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.writer = TestOutputWriter("unit", "test_interjection_agent")
        
        # Mock LLM for InterjectionAgent
        with patch('special_agents.interjection_agent.Agent.__init__'):
            self.agent = InterjectionAgent(
                confidence_threshold=0.7,
                cooldown_seconds=5
            )
            # Manually set required attributes
            self.agent.roles = ["Test agent"]
            self.agent.send_message = Mock(return_value="Test interjection")
    
    def test_should_interject_logic(self):
        """Test interjection decision logic."""
        # Test with question trigger
        conversation_turn = {
            "text": "What is the best way to implement this?",
            "speaker": "user"
        }
        available_info = [
            {"relevance_score": 0.8, "content": "Implementation guide"}
        ]
        
        should, confidence, int_type = self.agent.should_interject(
            conversation_turn, available_info
        )
        
        self.assertTrue(should)
        self.assertGreater(confidence, 0.7)
        self.assertIsNotNone(int_type)
        
        # Write results
        self.writer.write_json("interjection_decision.json", {
            "should_interject": should,
            "confidence": confidence,
            "type": int_type
        })
    
    def test_cooldown_mechanism(self):
        """Test interjection cooldown."""
        conversation_turn = {
            "text": "How do I fix this error?",
            "speaker": "user"
        }
        info = [{"relevance_score": 0.9}]
        
        # First interjection should work
        should1, _, _ = self.agent.should_interject(conversation_turn, info)
        self.assertTrue(should1)
        
        # Set last interjection time
        self.agent.last_interjection = datetime.now()
        
        # Second interjection should be blocked by cooldown
        should2, _, _ = self.agent.should_interject(conversation_turn, info)
        self.assertFalse(should2)
        
        self.writer.write_log("Cooldown mechanism test passed")
    
    def test_learning_from_feedback(self):
        """Test confidence adjustment from feedback."""
        initial_threshold = self.agent.confidence_threshold
        
        # Positive feedback should lower threshold
        self.agent.interjection_history.append({"id": 0})
        self.agent.learn_from_feedback(0, was_helpful=True)
        self.assertLess(self.agent.confidence_threshold, initial_threshold)
        
        # Negative feedback should raise threshold
        self.agent.interjection_history.append({"id": 1})
        self.agent.learn_from_feedback(1, was_helpful=False)
        self.assertGreater(self.agent.confidence_threshold, initial_threshold - 0.02)
        
        self.writer.write_log(f"Threshold adjusted: {initial_threshold} -> {self.agent.confidence_threshold}")


class TestMultiSourceOrchestration(unittest.TestCase):
    """Test multi-source input orchestration."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.writer = TestOutputWriter("integration", "test_multi_source")
        self.orchestrator = MultiSourceOrchestrator()
    
    def test_source_management(self):
        """Test adding and removing sources."""
        # Create mock source
        mock_source = Mock(spec=InputSource)
        mock_source.name = "test_source"
        mock_source.priority = 5
        mock_source.validate.return_value = True
        
        # Add source
        self.orchestrator.add_source(mock_source)
        self.assertEqual(len(self.orchestrator.sources), 1)
        
        # Remove source
        self.orchestrator.remove_source("test_source")
        self.assertEqual(len(self.orchestrator.sources), 0)
        
        self.writer.write_log("Source management test passed")
    
    def test_source_prioritization(self):
        """Test source priority ordering."""
        # Create sources with different priorities
        for i, priority in enumerate([3, 8, 5, 1]):
            source = Mock(spec=InputSource)
            source.name = f"source_{i}"
            source.priority = priority
            source.validate.return_value = True
            self.orchestrator.add_source(source)
        
        # Get prioritized list
        prioritized = self.orchestrator.prioritize_sources()
        
        # Verify order (highest priority first)
        self.assertEqual(prioritized[0], "source_1")  # priority 8
        self.assertEqual(prioritized[-1], "source_3")  # priority 1
        
        self.writer.write_json("source_priorities.json", {
            "sources": prioritized,
            "priorities": [self.orchestrator.sources[s].priority for s in prioritized]
        })


def run_tests():
    """Run all Listen v2 tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestConversationManager))
    suite.addTests(loader.loadTestsFromTestCase(TestInformationOrganizer))
    suite.addTests(loader.loadTestsFromTestCase(TestInterjectionAgent))
    suite.addTests(loader.loadTestsFromTestCase(TestMultiSourceOrchestration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Write summary
    writer = TestOutputWriter("integration", "listen_v2_summary")
    writer.write_results({
        "total": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "success": result.wasSuccessful()
    })
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)