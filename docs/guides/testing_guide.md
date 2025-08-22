# Testing Guide for Talk/Listen Framework

## Overview

This guide covers testing practices, conventions, and procedures for the Talk/Listen framework.

## Test Organization

Tests are organized by category in the `tests/` directory:

```
tests/
├── unit/           # Single component tests
├── integration/    # Multi-component tests (includes e2e)
├── performance/    # Performance benchmarks
├── quickies/       # Quick one-liner tests
├── playground/     # Experimental tests (gitignored)
├── input/          # Test input data (gitignored)
├── output/         # Test outputs (gitignored)
└── utilities/      # Test helpers (TestOutputWriter, etc.)
```

## Running Tests

### Using Make (Recommended)

```bash
# Run all tests with pytest
make test

# Run linting
make lint

# Run type checking
make typecheck

# Format code
make format
```

### Direct Python Execution

```bash
# Run specific test file
python3 tests/listen/test_listen_v2.py

# Run with unittest discover
python3 -m unittest discover tests/

# Run specific test class
python3 -m unittest tests.listen.test_listen_v2.TestConversationManager
```

### Using pytest

```bash
# Install dev dependencies first
pip install -e ".[dev]"

# Run all tests
pytest

# Run specific category
pytest tests/unit/

# Run with coverage
pytest --cov=talk --cov=listen tests/
```

## Writing Tests

### 1. Use TestOutputWriter for ALL Test Outputs

Every test MUST use the `TestOutputWriter` utility to ensure proper output organization:

```python
from utilities.test_output_writer import TestOutputWriter

class TestMyComponent(unittest.TestCase):
    def setUp(self):
        self.writer = TestOutputWriter("unit", "test_my_component")
        self.output_dir = self.writer.get_output_dir()
    
    def test_something(self):
        # Your test code
        result = {"status": "passed"}
        
        # Write results
        self.writer.write_results(result)
        self.writer.write_log("Test completed")
```

### 2. Output Directory Structure

Test outputs follow this structure:
```
tests/output/<category>/<testname>/<year_month>/
```

Example:
```
tests/output/unit/test_conversation_manager/2025_08/
├── conversation.json
├── conversation_analysis.json
└── test_conversation_manager_20250821_193435.log
```

### 3. Test Categories

Choose the appropriate category for your test:

- **unit/**: Testing single components in isolation
- **integration/**: Testing multiple components together
- **performance/**: Benchmarking and performance tests
- **quickies/**: Simple validation tests
- **playground/**: Experimental tests (not committed to git)

### 4. Mock External Dependencies

Use mocks for external services and LLMs:

```python
from unittest.mock import Mock, patch

class TestInterjectionAgent(unittest.TestCase):
    def setUp(self):
        # Mock the LLM backend
        with patch('special_agents.interjection_agent.Agent.__init__'):
            self.agent = InterjectionAgent()
            self.agent.send_message = Mock(return_value="Mocked response")
```

### 5. Test Data Management

- Input data goes in `tests/input/<category>/<testname>/`
- Output data goes in `tests/output/<category>/<testname>/<year_month>/`
- Both directories are gitignored to avoid committing test data

## Best Practices

### 1. Test Isolation

Each test should be independent and not rely on the state from other tests:

```python
def setUp(self):
    """Set up clean state for each test."""
    self.temp_dir = Path(tempfile.mkdtemp())
    
def tearDown(self):
    """Clean up after each test."""
    shutil.rmtree(self.temp_dir, ignore_errors=True)
```

### 2. Descriptive Test Names

Use clear, descriptive test names that explain what is being tested:

```python
def test_conversation_manager_tracks_multiple_speakers(self):
    """Test that ConversationManager correctly tracks multiple speakers."""
    # Good: Clear what's being tested

def test_1(self):
    # Bad: Unclear test purpose
```

### 3. Test Coverage Goals

Aim for:
- Core functionality: 80%+ coverage
- Special agents: 70%+ coverage
- Utilities: 90%+ coverage

Check coverage with:
```bash
pytest --cov=talk --cov-report=html
open htmlcov/index.html
```

### 4. Integration Test Patterns

For integration tests, use realistic scenarios:

```python
class TestListenV2Integration(unittest.TestCase):
    def test_audio_to_categorization_pipeline(self):
        """Test full pipeline from audio input to categorized information."""
        # 1. Create audio source with mock data
        # 2. Process through conversation manager
        # 3. Organize with information organizer
        # 4. Verify categorization
```

### 5. Performance Testing

For performance tests, measure and assert on metrics:

```python
class TestPerformance(unittest.TestCase):
    def test_conversation_manager_scales_to_1000_turns(self):
        """Test ConversationManager handles 1000 turns efficiently."""
        start = time.time()
        
        manager = ConversationManager()
        for i in range(1000):
            manager.add_turn(f"Turn {i}", speaker_id="user")
        
        duration = time.time() - start
        self.assertLess(duration, 5.0)  # Should complete in < 5 seconds
```

## Continuous Integration

### GitHub Actions

Tests run automatically on:
- Pull requests
- Pushes to main branch
- Daily scheduled runs

### Local Pre-commit

Run tests before committing:
```bash
# Add to .git/hooks/pre-commit
#!/bin/bash
make test || exit 1
make lint || exit 1
```

## Debugging Failed Tests

### 1. Check Test Outputs

Test outputs are saved in `tests/output/`:
```bash
# Find recent test outputs
find tests/output -name "*.log" -mtime -1

# Check specific test results
cat tests/output/unit/test_name/2025_08/results.json
```

### 2. Run with Verbose Output

```bash
# Unittest verbose
python3 -m unittest tests.listen.test_listen_v2 -v

# Pytest verbose
pytest tests/listen/test_listen_v2.py -vv
```

### 3. Interactive Debugging

```python
import pdb

def test_something(self):
    result = my_function()
    pdb.set_trace()  # Debugger stops here
    self.assertEqual(result, expected)
```

## Testing Listen Components

### Audio Testing

Since audio requires hardware, use mocks:

```python
# Mock audio source for testing
mock_audio = Mock(spec=AudioSource)
mock_audio.capture.return_value = AsyncGenerator(...)
```

### Conversation Testing

Test speaker identification and conversation flow:

```python
def test_speaker_diarization(self):
    manager = ConversationManager()
    
    # Simulate different speakers with audio features
    manager.add_turn("Hello", audio_features={"pitch": 200})  # Higher voice
    manager.add_turn("Hi", audio_features={"pitch": 100})     # Lower voice
    
    # Verify different speakers identified
    self.assertEqual(len(manager.speakers), 2)
```

### Information Organization Testing

Test categorization accuracy:

```python
def test_categorization_accuracy(self):
    organizer = InformationOrganizer()
    
    test_cases = [
        ("Meeting at 3pm", "scheduling"),
        ("Fix the bug", "technical"),
        ("Coffee with friends", "social")
    ]
    
    for text, expected_category in test_cases:
        category, confidence = organizer.categorize(text)
        self.assertEqual(category, expected_category)
```

## Common Issues and Solutions

### Issue: ModuleNotFoundError

**Problem**: `ModuleNotFoundError: No module named 'tests.utilities'`

**Solution**: Fix import paths:
```python
sys.path.insert(0, str(Path(__file__).parent.parent))
from utilities.test_output_writer import TestOutputWriter
```

### Issue: Test Output Not Created

**Problem**: Test outputs not appearing in expected location

**Solution**: Always use TestOutputWriter:
```python
writer = TestOutputWriter("unit", "test_name")
writer.write_results({"status": "passed"})
```

### Issue: Async Test Failures

**Problem**: Async tests not running properly

**Solution**: Use proper async test patterns:
```python
class TestAsync(unittest.TestCase):
    def test_async_method(self):
        async def run():
            result = await async_function()
            self.assertEqual(result, expected)
        
        asyncio.run(run())
```

## Summary

- Always use `TestOutputWriter` for test outputs
- Follow the directory structure in `tests/`
- Mock external dependencies
- Write descriptive test names
- Run tests with `make test` or `pytest`
- Check outputs in `tests/output/<category>/<testname>/<year_month>/`