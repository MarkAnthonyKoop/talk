"""
Test Output Writer Utility

Standard utility for writing test outputs following the repository's directory structure.
All tests MUST use this utility to ensure consistent output organization.

Directory structure:
    tests/outputs/<category>/<testname>/<month_year>/

Usage:
    from tests.utilities.test_output_writer import TestOutputWriter
    
    writer = TestOutputWriter("unit", "test_agent")
    output_dir = writer.get_output_dir()  # Creates and returns the directory
    writer.write_file("results.json", {"status": "passed"})
    writer.write_log("Test completed successfully")
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional


class TestOutputWriter:
    """Standard utility for managing test output directories and files."""
    
    def __init__(self, category: str, test_name: str, base_dir: Optional[str] = None):
        """
        Initialize the test output writer.
        
        Args:
            category: Test category (e.g., "unit", "integration", "e2e", "performance")
            test_name: Name of the test (e.g., "test_agent", "test_talk_v17")
            base_dir: Base directory for outputs (defaults to tests/outputs)
        """
        self.category = category
        self.test_name = test_name
        self.timestamp = datetime.now()
        self.month_year = self.timestamp.strftime("%Y_%m")
        
        # Determine base directory
        if base_dir is None:
            # Find tests directory relative to this file
            current_file = Path(__file__)
            tests_dir = current_file.parent.parent
            base_dir = tests_dir / "output"
        else:
            base_dir = Path(base_dir)
            
        self.output_dir = base_dir / category / test_name / self.month_year
        self._ensure_directory()
        
        # Set up logging for this test
        self.log_file = self.output_dir / f"{test_name}_{self.timestamp.strftime('%Y%m%d_%H%M%S')}.log"
        self._setup_logging()
    
    def _ensure_directory(self) -> None:
        """Create the output directory if it doesn't exist."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _setup_logging(self) -> None:
        """Set up logging to file for this test."""
        self.logger = logging.getLogger(f"{self.category}.{self.test_name}")
        handler = logging.FileHandler(self.log_file)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def get_output_dir(self) -> Path:
        """
        Get the output directory path.
        
        Returns:
            Path object for the output directory
        """
        return self.output_dir
    
    def write_file(self, filename: str, content: Any, mode: str = 'w') -> Path:
        """
        Write content to a file in the output directory.
        
        Args:
            filename: Name of the file to write
            content: Content to write (will be JSON-encoded if dict/list)
            mode: File write mode (default 'w')
            
        Returns:
            Path to the written file
        """
        file_path = self.output_dir / filename
        
        if isinstance(content, (dict, list)):
            with open(file_path, mode) as f:
                json.dump(content, f, indent=2, default=str)
        else:
            with open(file_path, mode) as f:
                f.write(str(content))
        
        return file_path
    
    def write_log(self, message: str, level: str = "INFO") -> None:
        """
        Write a message to the test log.
        
        Args:
            message: Log message
            level: Log level (INFO, WARNING, ERROR, DEBUG)
        """
        log_method = getattr(self.logger, level.lower(), self.logger.info)
        log_method(message)
    
    def write_results(self, results: Dict[str, Any]) -> Path:
        """
        Write test results to a standardized results.json file.
        
        Args:
            results: Dictionary of test results
            
        Returns:
            Path to the results file
        """
        # Add metadata to results
        results["_metadata"] = {
            "category": self.category,
            "test_name": self.test_name,
            "timestamp": self.timestamp.isoformat(),
            "output_dir": str(self.output_dir)
        }
        
        return self.write_file("results.json", results)
    
    def create_subdirectory(self, subdir_name: str) -> Path:
        """
        Create a subdirectory within the test output directory.
        
        Args:
            subdir_name: Name of subdirectory to create
            
        Returns:
            Path to the created subdirectory
        """
        subdir = self.output_dir / subdir_name
        subdir.mkdir(parents=True, exist_ok=True)
        return subdir
    
    def __str__(self) -> str:
        """String representation of the output writer."""
        return f"TestOutputWriter({self.category}/{self.test_name}/{self.month_year})"


class TestInputReader:
    """Standard utility for reading test input files."""
    
    def __init__(self, category: str, test_name: str, base_dir: Optional[str] = None):
        """
        Initialize the test input reader.
        
        Args:
            category: Test category
            test_name: Name of the test
            base_dir: Base directory for inputs (defaults to tests/inputs)
        """
        self.category = category
        self.test_name = test_name
        
        if base_dir is None:
            current_file = Path(__file__)
            tests_dir = current_file.parent.parent
            base_dir = tests_dir / "inputs"
        else:
            base_dir = Path(base_dir)
            
        self.input_dir = base_dir / category / test_name
    
    def get_input_dir(self, month_year: Optional[str] = None) -> Path:
        """
        Get the input directory path.
        
        Args:
            month_year: Optional month_year subdirectory (e.g., "2025_08")
            
        Returns:
            Path object for the input directory
        """
        if month_year:
            return self.input_dir / month_year
        return self.input_dir
    
    def read_file(self, filename: str, month_year: Optional[str] = None) -> Any:
        """
        Read a file from the input directory.
        
        Args:
            filename: Name of the file to read
            month_year: Optional month_year subdirectory
            
        Returns:
            File contents (parsed as JSON if .json extension)
        """
        if month_year:
            file_path = self.input_dir / month_year / filename
        else:
            file_path = self.input_dir / filename
            
        if not file_path.exists():
            raise FileNotFoundError(f"Input file not found: {file_path}")
        
        with open(file_path, 'r') as f:
            if filename.endswith('.json'):
                return json.load(f)
            return f.read()


# Example usage in tests
if __name__ == "__main__":
    # Example test using the output writer
    writer = TestOutputWriter("unit", "test_example")
    
    # Write test results
    results = {
        "passed": 10,
        "failed": 2,
        "skipped": 1,
        "duration": 1.234
    }
    writer.write_results(results)
    
    # Log test progress
    writer.write_log("Starting test execution")
    writer.write_log("Test passed", "INFO")
    writer.write_log("Unexpected behavior", "WARNING")
    
    # Write additional output files
    writer.write_file("detailed_output.txt", "Detailed test output here...")
    
    print(f"Test outputs written to: {writer.get_output_dir()}")