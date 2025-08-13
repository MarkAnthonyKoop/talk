#!/usr/bin/env python3
"""
Test runner that properly sets up the Python path and runs tests.
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import and run the test
if __name__ == "__main__":
    import unittest
    
    # Import the test module
    from tests.test_advanced_plan import TestAdvancedPlan
    
    # Create a test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestAdvancedPlan)
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Exit with proper code
    sys.exit(0 if result.wasSuccessful() else 1)