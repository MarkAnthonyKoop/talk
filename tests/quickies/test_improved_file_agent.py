#!/usr/bin/env python3
"""Test the improved file agent with SEARCH/REPLACE blocks."""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from special_agents.improved_file_agent import ImprovedFileAgent


def test_create_file():
    """Test creating a new file."""
    print("\n=== Test: Create New File ===")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        agent = ImprovedFileAgent(base_dir=tmpdir)
        
        edit_request = """
test_hello.py
<<<<<<< SEARCH
=======
def say_hello(name):
    \"\"\"Say hello to someone.\"\"\"
    return f"Hello, {name}!"

if __name__ == "__main__":
    print(say_hello("World"))
>>>>>>> REPLACE
"""
        
        result = agent.reply(edit_request)
        print(f"Result: {result}")
        
        # Check if file was created
        file_path = Path(tmpdir) / "test_hello.py"
        if file_path.exists():
            print("✓ File created successfully")
            with open(file_path) as f:
                content = f.read()
                print(f"Content:\n{content}")
                
            # Check indentation
            if '    return f"Hello' in content:
                print("✓ Indentation preserved correctly")
            else:
                print("✗ Indentation issue detected")
        else:
            print("✗ File was not created")
            
        return file_path.exists()


def test_edit_file():
    """Test editing an existing file."""
    print("\n=== Test: Edit Existing File ===")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        agent = ImprovedFileAgent(base_dir=tmpdir)
        
        # First create a file
        test_file = Path(tmpdir) / "calculator.py"
        original_content = """def add(a, b):
    return a + b

def subtract(a, b):
    return a - b"""
        
        with open(test_file, 'w') as f:
            f.write(original_content)
        
        # Now edit it
        edit_request = """
calculator.py
<<<<<<< SEARCH
def add(a, b):
    return a + b
=======
def add(a, b):
    \"\"\"Add two numbers together.\"\"\"
    result = a + b
    print(f"Adding {a} + {b} = {result}")
    return result
>>>>>>> REPLACE
"""
        
        result = agent.reply(edit_request)
        print(f"Result: {result}")
        
        # Check if file was edited
        with open(test_file) as f:
            new_content = f.read()
            
        if "Add two numbers together" in new_content:
            print("✓ Edit applied successfully")
            print(f"New content:\n{new_content}")
            
            # Check that unchanged parts remain
            if "def subtract(a, b):" in new_content:
                print("✓ Unchanged code preserved")
            else:
                print("✗ Unchanged code was affected")
                
            # Check indentation
            if '    result = a + b' in new_content:
                print("✓ Indentation preserved in edit")
            else:
                print("✗ Indentation issue in edit")
        else:
            print("✗ Edit was not applied")
            
        return "SUCCESS" in result


def test_multiple_edits():
    """Test multiple edits to the same file."""
    print("\n=== Test: Multiple Edits ===")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        agent = ImprovedFileAgent(base_dir=tmpdir)
        
        # Create initial file
        test_file = Path(tmpdir) / "config.py"
        with open(test_file, 'w') as f:
            f.write("""DEBUG = False
LOG_LEVEL = "INFO"
DATABASE_URL = "sqlite:///db.sqlite3"
SECRET_KEY = "change-me"
""")
        
        # Apply multiple edits
        edit_request = """
config.py
<<<<<<< SEARCH
DEBUG = False
=======
DEBUG = True
>>>>>>> REPLACE

config.py
<<<<<<< SEARCH
LOG_LEVEL = "INFO"
=======
LOG_LEVEL = "DEBUG"
>>>>>>> REPLACE

config.py
<<<<<<< SEARCH
SECRET_KEY = "change-me"
=======
SECRET_KEY = "super-secret-key-123"
>>>>>>> REPLACE
"""
        
        result = agent.reply(edit_request)
        print(f"Result: {result}")
        
        # Check all edits
        with open(test_file) as f:
            content = f.read()
            
        success = True
        if "DEBUG = True" in content:
            print("✓ First edit applied")
        else:
            print("✗ First edit failed")
            success = False
            
        if 'LOG_LEVEL = "DEBUG"' in content:
            print("✓ Second edit applied")
        else:
            print("✗ Second edit failed")
            success = False
            
        if "super-secret-key-123" in content:
            print("✓ Third edit applied")
        else:
            print("✗ Third edit failed")
            success = False
            
        if "sqlite:///db.sqlite3" in content:
            print("✓ Unchanged content preserved")
        else:
            print("✗ Unchanged content affected")
            success = False
            
        print(f"\nFinal content:\n{content}")
        return success


def test_python_class_indentation():
    """Test that Python class indentation is preserved correctly."""
    print("\n=== Test: Python Class Indentation ===")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        agent = ImprovedFileAgent(base_dir=tmpdir)
        
        # Create a Python class file with proper indentation
        edit_request = """
agent.py
<<<<<<< SEARCH
=======
from enum import Enum
import uuid

class AgentStatus(Enum):
    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"

class Agent:
    def __init__(self, capabilities):
        self.id = str(uuid.uuid4())
        self.capabilities = capabilities
        self.status = AgentStatus.IDLE
        
    def can_handle_task(self, required_capabilities):
        return all(cap in self.capabilities for cap in required_capabilities)
>>>>>>> REPLACE
"""
        
        result = agent.reply(edit_request)
        print(f"Result: {result}")
        
        # Check the file
        file_path = Path(tmpdir) / "agent.py"
        if file_path.exists():
            with open(file_path) as f:
                content = f.read()
                
            # Run syntax check
            try:
                compile(content, file_path, 'exec')
                print("✓ Python syntax is valid")
                
                # Check specific indentation
                lines = content.split('\n')
                checks = [
                    ('    IDLE = "idle"', "Enum member indentation"),
                    ('    def __init__(self, capabilities):', "Method indentation"),
                    ('        self.id = str(uuid.uuid4())', "Method body indentation"),
                ]
                
                for check_line, desc in checks:
                    if any(check_line in line for line in lines):
                        print(f"✓ {desc} correct")
                    else:
                        print(f"✗ {desc} incorrect")
                        
            except SyntaxError as e:
                print(f"✗ Python syntax error: {e}")
                print(f"Content:\n{content}")
                return False
        
        return file_path.exists()


if __name__ == "__main__":
    print("Testing Improved File Agent with SEARCH/REPLACE blocks")
    print("=" * 60)
    
    results = []
    results.append(("Create File", test_create_file()))
    results.append(("Edit File", test_edit_file()))
    results.append(("Multiple Edits", test_multiple_edits()))
    results.append(("Python Indentation", test_python_class_indentation()))
    
    print("\n" + "=" * 60)
    print("SUMMARY:")
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    if all(passed for _, passed in results):
        print("\nAll tests passed!")
        sys.exit(0)
    else:
        print("\nSome tests failed.")
        sys.exit(1)