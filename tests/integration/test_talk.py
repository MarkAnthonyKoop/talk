#!/usr/bin/env python3

"""
Test Talk orchestrator with multi-iteration tasks.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from talk.talk import TalkOrchestrator


class TestTalk(unittest.TestCase):
    """Test the Talk orchestrator with complex multi-iteration tasks."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        
    def tearDown(self):
        """Clean up test environment."""
        os.chdir(self.original_cwd)
        
    def test_multi_iteration_webapp(self):
        """Test Talk with a multi-iteration web application task."""
        # Change to temp directory for the test
        os.chdir(self.temp_dir)
        
        # Define a complex task that should require multiple iterations
        task = """
        Create a simple web application with the following features:
        1. A Flask web server with a home page
        2. A user registration form with validation
        3. A simple database to store users
        4. Basic CSS styling for the pages
        5. Unit tests for the registration functionality
        
        The application should be fully functional and tested.
        """
        
        # Create the orchestrator (non-interactive mode for testing)
        orchestrator = TalkOrchestrator(
            task=task,
            working_dir=self.temp_dir,
            model="gpt-4o-mini",  # Use a faster model for testing
            timeout_minutes=10,
            interactive=False
        )
        
        # Run the orchestrator
        print(f"\nRunning Talk with task: {task[:100]}...")
        result = orchestrator.run()
        
        # Check that it completed successfully
        self.assertEqual(result, 0, "Talk orchestrator should complete successfully")
        
        # Verify output structure
        session_dir = orchestrator.session_dir
        working_dir = orchestrator.working_dir
        self.assertTrue(session_dir.exists(), "Session directory should exist")
        self.assertTrue(working_dir.exists(), "Working directory should exist")
        
        # Check for expected output files in session directory
        session_info = session_dir / "session_info.json"
        blackboard_file = session_dir / "blackboard.json"
        
        self.assertTrue(session_info.exists(), "Session info file should exist")
        self.assertTrue(blackboard_file.exists(), "Blackboard file should exist")
        
        # Print results for manual inspection
        print(f"\nTalk completed successfully!")
        print(f"Session directory: {session_dir}")
        print(f"Working directory: {working_dir}")
        print(f"Session files: {list(session_dir.glob('*'))}")
        print(f"Working files: {list(working_dir.glob('*')) if working_dir.exists() else 'N/A'}")
        
        # Check if any files were created outside the working directory
        temp_files = list(Path(self.temp_dir).glob('*'))
        # No files should be created in temp_dir anymore since we use centralized output
        
        if temp_files:
            print(f"WARNING: Unexpected files in temp directory: {temp_files}")
        else:
            print("✅ No files leaked to temp directory - output properly centralized!")
        
        return True


if __name__ == "__main__":
    unittest.main()