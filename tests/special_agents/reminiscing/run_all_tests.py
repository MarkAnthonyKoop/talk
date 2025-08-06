#!/usr/bin/env python3
"""
Comprehensive test runner for ReminiscingAgent test suite.

Runs all test categories and provides detailed coverage report.
"""

import sys
import os
import time
import importlib
from typing import List, Tuple, Dict
from datetime import datetime

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, project_root)


class TestRunner:
    """Runs all tests and generates comprehensive report."""
    
    def __init__(self):
        self.test_modules = [
            ('Basic Functionality', 'test_reminiscing_agent'),
            ('Vector Store', 'test_vector_store'),
            ('Enhanced Vector Store', 'test_enhanced_vector_store'),
            ('Sub-agents', 'test_subagents'),
            ('Semantic Search', 'test_semantic_search_agent'),
            ('Algorithms', 'test_algorithms'),
            ('Performance', 'test_performance'),
            ('Error Recovery', 'test_error_recovery'),
            ('Edge Cases', 'test_edge_cases'),
        ]
        self.results = []
        self.start_time = None
        self.end_time = None
    
    def run_module_tests(self, module_name: str) -> Tuple[int, int, List[str]]:
        """Run tests from a single module."""
        try:
            # Import module
            module = importlib.import_module(f'tests.special_agents.reminiscing.{module_name}')
            
            # Find test functions
            test_functions = [
                (name, func) for name, func in module.__dict__.items()
                if callable(func) and name.startswith('test_')
            ]
            
            passed = 0
            failed = 0
            failures = []
            
            # Run each test
            for test_name, test_func in test_functions:
                try:
                    result = test_func()
                    if result or result is None:  # None means test passed
                        passed += 1
                    else:
                        failed += 1
                        failures.append(test_name)
                except Exception as e:
                    failed += 1
                    failures.append(f"{test_name}: {str(e)[:50]}")
            
            return passed, failed, failures
            
        except ImportError as e:
            return 0, 1, [f"Failed to import {module_name}: {e}"]
        except Exception as e:
            return 0, 1, [f"Module error: {e}"]
    
    def run_all_tests(self):
        """Run all test modules."""
        print("=" * 70)
        print("REMINISCING AGENT - COMPREHENSIVE TEST SUITE")
        print("=" * 70)
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)
        
        self.start_time = time.time()
        total_passed = 0
        total_failed = 0
        
        for category, module_name in self.test_modules:
            print(f"\n{'='*50}")
            print(f"Category: {category}")
            print(f"Module: {module_name}")
            print("-" * 50)
            
            module_start = time.time()
            passed, failed, failures = self.run_module_tests(module_name)
            module_time = time.time() - module_start
            
            total_passed += passed
            total_failed += failed
            
            # Store results
            self.results.append({
                'category': category,
                'module': module_name,
                'passed': passed,
                'failed': failed,
                'failures': failures,
                'time': module_time
            })
            
            # Print module summary
            print(f"Results: {passed} passed, {failed} failed")
            print(f"Time: {module_time:.2f}s")
            
            if failures:
                print("Failures:")
                for failure in failures[:5]:  # Show first 5 failures
                    print(f"  - {failure}")
                if len(failures) > 5:
                    print(f"  ... and {len(failures)-5} more")
        
        self.end_time = time.time()
        
        # Print final summary
        self._print_summary(total_passed, total_failed)
        
        return total_failed == 0
    
    def _print_summary(self, total_passed: int, total_failed: int):
        """Print comprehensive test summary."""
        total_time = self.end_time - self.start_time
        total_tests = total_passed + total_failed
        
        print("\n" + "=" * 70)
        print("TEST SUMMARY")
        print("=" * 70)
        
        # Overall statistics
        print(f"\nTotal Tests: {total_tests}")
        print(f"Passed: {total_passed} ({100*total_passed//max(total_tests,1)}%)")
        print(f"Failed: {total_failed} ({100*total_failed//max(total_tests,1)}%)")
        print(f"Total Time: {total_time:.2f}s")
        
        # Category breakdown
        print("\nCategory Breakdown:")
        print("-" * 50)
        print(f"{'Category':<25} {'Passed':<10} {'Failed':<10} {'Time(s)':<10}")
        print("-" * 50)
        
        for result in self.results:
            print(f"{result['category']:<25} {result['passed']:<10} "
                  f"{result['failed']:<10} {result['time']:.2f}")
        
        # Coverage estimate
        print("\n" + "=" * 70)
        print("COVERAGE ANALYSIS")
        print("=" * 70)
        self._print_coverage_analysis()
        
        # Final verdict
        print("\n" + "=" * 70)
        if total_failed == 0:
            print("✅ ALL TESTS PASSED - TEST SUITE IS COMPREHENSIVE")
        else:
            print(f"❌ {total_failed} TESTS FAILED - REVIEW FAILURES ABOVE")
        print("=" * 70)
    
    def _print_coverage_analysis(self):
        """Print test coverage analysis."""
        coverage_areas = {
            'Core Functionality': ['test_reminiscing_agent', 'test_vector_store', 'test_enhanced_vector_store'],
            'Sub-components': ['test_subagents', 'test_semantic_search_agent'],
            'Algorithms': ['test_algorithms'],
            'Performance': ['test_performance'],
            'Reliability': ['test_error_recovery', 'test_edge_cases'],
        }
        
        print("\nTest Coverage by Area:")
        for area, modules in coverage_areas.items():
            area_results = [r for r in self.results if r['module'] in modules]
            area_passed = sum(r['passed'] for r in area_results)
            area_failed = sum(r['failed'] for r in area_results)
            area_total = area_passed + area_failed
            
            if area_total > 0:
                coverage = 100 * area_passed // area_total
                status = "✅" if coverage >= 80 else "⚠️" if coverage >= 60 else "❌"
                print(f"  {status} {area}: {coverage}% ({area_passed}/{area_total} tests)")
        
        # Component coverage
        print("\nComponent Test Coverage:")
        components = {
            'ReminiscingAgent': 'test_reminiscing_agent',
            'VectorStore': 'test_vector_store',
            'EnhancedVectorStore': 'test_enhanced_vector_store',
            'SemanticSearchAgent': 'test_semantic_search_agent',
            'Sub-agents': 'test_subagents',
        }
        
        for component, module in components.items():
            result = next((r for r in self.results if r['module'] == module), None)
            if result:
                total = result['passed'] + result['failed']
                status = "✅" if result['failed'] == 0 else "❌"
                print(f"  {status} {component}: {result['passed']}/{total} tests")
    
    def generate_report(self, filename: str = "test_report.txt"):
        """Generate detailed test report file."""
        with open(filename, 'w') as f:
            f.write("REMINISCING AGENT TEST REPORT\n")
            f.write(f"Generated: {datetime.now()}\n")
            f.write("=" * 70 + "\n\n")
            
            for result in self.results:
                f.write(f"Category: {result['category']}\n")
                f.write(f"Module: {result['module']}\n")
                f.write(f"Passed: {result['passed']}\n")
                f.write(f"Failed: {result['failed']}\n")
                f.write(f"Time: {result['time']:.2f}s\n")
                
                if result['failures']:
                    f.write("Failures:\n")
                    for failure in result['failures']:
                        f.write(f"  - {failure}\n")
                
                f.write("\n" + "-" * 50 + "\n\n")
            
            # Summary statistics
            total_passed = sum(r['passed'] for r in self.results)
            total_failed = sum(r['failed'] for r in self.results)
            total_time = sum(r['time'] for r in self.results)
            
            f.write("SUMMARY\n")
            f.write(f"Total Tests: {total_passed + total_failed}\n")
            f.write(f"Passed: {total_passed}\n")
            f.write(f"Failed: {total_failed}\n")
            f.write(f"Total Time: {total_time:.2f}s\n")
            f.write(f"Success Rate: {100*total_passed/max(total_passed+total_failed,1):.1f}%\n")
        
        print(f"\nDetailed report saved to: {filename}")


def main():
    """Main entry point."""
    runner = TestRunner()
    
    # Parse arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == '--quick':
            # Run only essential tests
            runner.test_modules = [
                ('Basic Functionality', 'test_reminiscing_agent'),
                ('Vector Store', 'test_vector_store'),
                ('Algorithms', 'test_algorithms'),
            ]
            print("Running QUICK test suite (essential tests only)...\n")
        elif sys.argv[1] == '--help':
            print("Usage: python run_all_tests.py [--quick] [--report]")
            print("  --quick  Run essential tests only")
            print("  --report Generate detailed report file")
            return
    
    # Run tests
    success = runner.run_all_tests()
    
    # Generate report if requested
    if '--report' in sys.argv:
        runner.generate_report()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()