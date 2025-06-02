#!/usr/bin/env python3
"""
Executor module for Talk CLI.

This module takes the output from the planner and applies the improvements to the codebase.
It handles file modification, verification, and error handling.
"""

import os
import re
import sys
import shutil
import logging
import subprocess
import tempfile
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime

from talk.analyzer import (
    AnalysisResult,
    Improvement,
    ImprovementType,
    CodeLocation,
    Language
)
from talk.planner import (
    Plan,
    PlannedImprovement,
    PlanningStrategy,
    PriorityLevel,
    DifficultyLevel
)

logger = logging.getLogger("talk.executor")


class ExecutionStatus(Enum):
    """Status of an improvement execution."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"
    REVERTED = "reverted"


class VerificationType(Enum):
    """Types of verification to perform after applying an improvement."""
    SYNTAX = "syntax"
    TESTS = "tests"
    LINT = "lint"
    COMPILATION = "compilation"


@dataclass
class ExecutedImprovement:
    """Represents an executed improvement with execution metadata."""
    planned_improvement: PlannedImprovement
    status: ExecutionStatus
    execution_time_seconds: float
    modified_files: List[Path]
    diff: Optional[str] = None
    error_message: Optional[str] = None
    verification_results: Dict[VerificationType, bool] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        """String representation of the executed improvement."""
        return (
            f"{self.status.value.upper()}: {self.planned_improvement.improvement.description} "
            f"({self.execution_time_seconds:.2f}s, {len(self.modified_files)} files modified)"
        )


@dataclass
class ExecutionResult:
    """Results of executing a plan."""
    executed_improvements: List[ExecutedImprovement]
    total_execution_time_seconds: float
    successful_count: int
    failed_count: int
    skipped_count: int
    reverted_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_executed_by_status(self, status: ExecutionStatus) -> List[ExecutedImprovement]:
        """Filter executed improvements by status."""
        return [i for i in self.executed_improvements if i.status == status]

    def get_modified_files(self) -> List[Path]:
        """Get all files modified during execution."""
        modified_files = set()
        for improvement in self.executed_improvements:
            if improvement.status == ExecutionStatus.SUCCEEDED:
                modified_files.update(improvement.modified_files)
        return list(modified_files)


class BackupStrategy(Enum):
    """Strategies for backing up files before modification."""
    NONE = "none"
    COPY = "copy"
    GIT = "git"


@dataclass
class ExecutorConfig:
    """Configuration for the executor."""
    backup_strategy: BackupStrategy = BackupStrategy.COPY
    verify_syntax: bool = True
    run_tests: bool = True
    run_lint: bool = False
    test_command: str = "pytest"
    lint_command: str = "ruff"
    max_retries: int = 1
    timeout_seconds: int = 300
    dry_run: bool = False
    verbose: bool = False


class Executor:
    """Executes code improvements based on a plan."""
    
    def __init__(
        self,
        root_path: Path,
        config: ExecutorConfig = None
    ):
        """Initialize the executor with a root path and configuration."""
        self.root_path = root_path
        self.config = config or ExecutorConfig()
        self.backup_dir = None
        self._setup_backup_dir()
    
    def _setup_backup_dir(self):
        """Set up a directory for backups if needed."""
        if self.config.backup_strategy == BackupStrategy.COPY:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.backup_dir = self.root_path / f".talk_backup_{timestamp}"
            if not self.config.dry_run:
                self.backup_dir.mkdir(exist_ok=True)
            logger.info(f"Backup directory created at {self.backup_dir}")
    
    def execute_plan(self, plan: Plan) -> ExecutionResult:
        """Execute a plan of improvements."""
        logger.info(f"Executing improvement plan with {len(plan.improvements)} improvements")
        
        if self.config.dry_run:
            logger.info("DRY RUN MODE: No actual changes will be made")
        
        executed_improvements = []
        start_time = datetime.now()
        
        # Execute improvements in order
        for planned_improvement in plan.improvements:
            executed = self.execute_improvement(planned_improvement)
            executed_improvements.append(executed)
            
            # Stop execution if a critical improvement fails
            if (executed.status == ExecutionStatus.FAILED and 
                planned_improvement.priority == PriorityLevel.CRITICAL):
                logger.error("Critical improvement failed. Stopping execution.")
                break
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        # Count results by status
        successful_count = sum(1 for i in executed_improvements if i.status == ExecutionStatus.SUCCEEDED)
        failed_count = sum(1 for i in executed_improvements if i.status == ExecutionStatus.FAILED)
        skipped_count = sum(1 for i in executed_improvements if i.status == ExecutionStatus.SKIPPED)
        reverted_count = sum(1 for i in executed_improvements if i.status == ExecutionStatus.REVERTED)
        
        # Create and return execution result
        result = ExecutionResult(
            executed_improvements=executed_improvements,
            total_execution_time_seconds=execution_time,
            successful_count=successful_count,
            failed_count=failed_count,
            skipped_count=skipped_count,
            reverted_count=reverted_count,
            metadata={
                'dry_run': self.config.dry_run,
                'backup_strategy': self.config.backup_strategy.value,
                'root_path': str(self.root_path),
                'timestamp': start_time.isoformat()
            }
        )
        
        logger.info(f"Plan execution completed: {successful_count} succeeded, "
                    f"{failed_count} failed, {skipped_count} skipped, "
                    f"{reverted_count} reverted")
        
        return result
    
    def execute_improvement(self, planned_improvement: PlannedImprovement) -> ExecutedImprovement:
        """Execute a single planned improvement."""
        improvement = planned_improvement.improvement
        logger.info(f"Executing improvement: {improvement.description}")
        
        # Initialize result with default values
        result = ExecutedImprovement(
            planned_improvement=planned_improvement,
            status=ExecutionStatus.PENDING,
            execution_time_seconds=0,
            modified_files=[],
            verification_results={}
        )
        
        # Skip if dry run and we can't generate a diff
        if self.config.dry_run and not planned_improvement.after_code:
            logger.info("Skipping in dry run mode (no generated code available)")
            result.status = ExecutionStatus.SKIPPED
            return result
        
        # Start execution
        start_time = datetime.now()
        result.status = ExecutionStatus.IN_PROGRESS
        
        try:
            # Backup files if needed
            if improvement.location and improvement.location.file_path:
                self._backup_file(improvement.location.file_path)
            
            # Apply the improvement
            if self.config.dry_run:
                # In dry run mode, just generate a diff
                diff = self._generate_diff(
                    improvement.location.file_path,
                    planned_improvement.before_code,
                    planned_improvement.after_code
                )
                result.diff = diff
                result.status = ExecutionStatus.SUCCEEDED
                logger.info("Generated diff in dry run mode")
            else:
                # Apply the actual changes
                modified_files = self._apply_improvement(planned_improvement)
                result.modified_files = modified_files
                
                # Verify the changes
                verification_results = self._verify_changes(modified_files, planned_improvement)
                result.verification_results = verification_results
                
                # Check if all verifications passed
                if all(verification_results.values()):
                    result.status = ExecutionStatus.SUCCEEDED
                    logger.info("Improvement applied successfully")
                else:
                    # Revert changes if verification failed
                    logger.warning("Verification failed, reverting changes")
                    self._revert_changes(modified_files)
                    result.status = ExecutionStatus.REVERTED
        
        except Exception as e:
            logger.exception(f"Error executing improvement: {str(e)}")
            result.status = ExecutionStatus.FAILED
            result.error_message = str(e)
            
            # Revert any changes made
            if result.modified_files:
                self._revert_changes(result.modified_files)
        
        # Calculate execution time
        end_time = datetime.now()
        result.execution_time_seconds = (end_time - start_time).total_seconds()
        
        return result
    
    def _backup_file(self, file_path: Path) -> Optional[Path]:
        """Backup a file before modifying it."""
        if self.config.dry_run:
            return None
        
        if self.config.backup_strategy == BackupStrategy.NONE:
            return None
        
        elif self.config.backup_strategy == BackupStrategy.COPY:
            if not self.backup_dir:
                return None
            
            # Create relative path structure in backup dir
            rel_path = file_path.relative_to(self.root_path)
            backup_path = self.backup_dir / rel_path
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy the file
            shutil.copy2(file_path, backup_path)
            logger.debug(f"Backed up {file_path} to {backup_path}")
            return backup_path
        
        elif self.config.backup_strategy == BackupStrategy.GIT:
            # Check if file is in a git repository
            try:
                result = subprocess.run(
                    ["git", "rev-parse", "--is-inside-work-tree"],
                    cwd=file_path.parent,
                    capture_output=True,
                    text=True,
                    check=False
                )
                
                if result.returncode == 0 and result.stdout.strip() == "true":
                    # File is in a git repo, no need for additional backup
                    logger.debug(f"File {file_path} is in a git repository, using git for backup")
                    return None
                else:
                    # Fall back to copy backup
                    logger.debug(f"File {file_path} is not in a git repository, falling back to copy backup")
                    return self._backup_file_copy(file_path)
            
            except Exception as e:
                logger.warning(f"Error checking git status for {file_path}: {e}")
                return self._backup_file_copy(file_path)
        
        return None
    
    def _backup_file_copy(self, file_path: Path) -> Optional[Path]:
        """Backup a file by copying it (fallback method)."""
        backup_path = file_path.with_suffix(file_path.suffix + ".bak")
        try:
            shutil.copy2(file_path, backup_path)
            logger.debug(f"Backed up {file_path} to {backup_path}")
            return backup_path
        except Exception as e:
            logger.warning(f"Failed to create backup of {file_path}: {e}")
            return None
    
    def _generate_diff(self, file_path: Path, before_code: str, after_code: str) -> Optional[str]:
        """Generate a diff between before and after code."""
        if not before_code or not after_code:
            return None
        
        try:
            # Write before and after to temporary files
            with tempfile.NamedTemporaryFile(mode='w', suffix='.before') as before_file:
                before_file.write(before_code)
                before_file.flush()
                
                with tempfile.NamedTemporaryFile(mode='w', suffix='.after') as after_file:
                    after_file.write(after_code)
                    after_file.flush()
                    
                    # Generate diff using external diff tool
                    result = subprocess.run(
                        ["diff", "-u", before_file.name, after_file.name],
                        capture_output=True,
                        text=True,
                        check=False
                    )
                    
                    # diff returns non-zero if files differ, which is expected
                    return result.stdout
        
        except Exception as e:
            logger.warning(f"Failed to generate diff: {e}")
            return None
    
    def _apply_improvement(self, planned_improvement: PlannedImprovement) -> List[Path]:
        """Apply an improvement to the codebase."""
        improvement = planned_improvement.improvement
        modified_files = []
        
        # Check if we have before/after code
        if (improvement.location and 
            improvement.location.file_path and 
            planned_improvement.before_code and 
            planned_improvement.after_code):
            
            file_path = improvement.location.file_path
            before_code = planned_improvement.before_code
            after_code = planned_improvement.after_code
            
            # Apply the change
            self._modify_file(file_path, before_code, after_code)
            modified_files.append(file_path)
        
        # If no before/after code, use improvement type specific handlers
        else:
            handler = self._get_improvement_handler(improvement.type)
            if handler:
                handler_modified_files = handler(planned_improvement)
                if handler_modified_files:
                    modified_files.extend(handler_modified_files)
        
        return modified_files
    
    def _modify_file(self, file_path: Path, before_snippet: str, after_snippet: str) -> bool:
        """Modify a file by replacing a code snippet."""
        try:
            # Read the entire file
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Replace the snippet
            if before_snippet in content:
                new_content = content.replace(before_snippet, after_snippet)
                
                # Write the modified content back
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                
                logger.debug(f"Modified file: {file_path}")
                return True
            else:
                logger.warning(f"Could not find the exact code snippet in {file_path}")
                return False
        
        except Exception as e:
            logger.error(f"Error modifying file {file_path}: {e}")
            raise
    
    def _get_improvement_handler(
        self, 
        improvement_type: ImprovementType
    ) -> Optional[Callable[[PlannedImprovement], List[Path]]]:
        """Get a handler function for a specific improvement type."""
        handlers = {
            ImprovementType.UNUSED_CODE: self._handle_unused_code,
            ImprovementType.DOCUMENTATION: self._handle_documentation,
            ImprovementType.STYLE: self._handle_style,
            # Add more handlers as needed
        }
        
        return handlers.get(improvement_type)
    
    def _handle_unused_code(self, planned_improvement: PlannedImprovement) -> List[Path]:
        """Handle unused code improvements."""
        improvement = planned_improvement.improvement
        modified_files = []
        
        if "Unused import" in improvement.description:
            file_path = improvement.location.file_path
            import_name = improvement.metadata.get('name', '')
            
            if file_path and import_name:
                try:
                    # Read the file
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    
                    # Find and remove the import line
                    new_lines = []
                    for line in lines:
                        if (f"import {import_name}" in line or 
                            f"from " in line and f"import {import_name}" in line):
                            # Skip this line (remove it)
                            continue
                        new_lines.append(line)
                    
                    # Write the file back if changed
                    if len(new_lines) < len(lines):
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.writelines(new_lines)
                        
                        modified_files.append(file_path)
                        logger.debug(f"Removed unused import '{import_name}' from {file_path}")
                
                except Exception as e:
                    logger.error(f"Error removing unused import from {file_path}: {e}")
                    raise
        
        return modified_files
    
    def _handle_documentation(self, planned_improvement: PlannedImprovement) -> List[Path]:
        """Handle documentation improvements."""
        improvement = planned_improvement.improvement
        modified_files = []
        
        if "Missing docstring" in improvement.description:
            file_path = improvement.location.file_path
            node_type = improvement.metadata.get('node_type', '')
            name = improvement.metadata.get('name', '')
            
            if file_path and node_type and name:
                try:
                    # Read the file
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    
                    # Find the definition line
                    for i, line in enumerate(lines):
                        definition_pattern = (
                            f"def {re.escape(name)}\(" if node_type == 'function' 
                            else f"class {re.escape(name)}\(?:"
                        )
                        
                        if re.search(definition_pattern, line):
                            # Check if the next line already has a docstring
                            if i+1 < len(lines) and '"""' in lines[i+1]:
                                logger.debug(f"Docstring already exists for {node_type} {name}")
                                break
                            
                            # Get indentation from the next line or use a default
                            if i+1 < len(lines):
                                indentation = re.match(r'^(\s*)', lines[i+1]).group(1)
                            else:
                                indentation = '    ' if node_type == 'function' else ''
                            
                            # Create docstring
                            if node_type == 'function':
                                docstring = f'{indentation}"""\n{indentation}Description of {name}.\n\n{indentation}Args:\n{indentation}    # TODO: Add parameters\n\n{indentation}Returns:\n{indentation}    # TODO: Add return value\n{indentation}"""\n'
                            else:  # class
                                docstring = f'{indentation}"""\n{indentation}{name} class.\n\n{indentation}Attributes:\n{indentation}    # TODO: Add attributes\n{indentation}"""\n'
                            
                            # Insert docstring after definition line
                            lines.insert(i+1, docstring)
                            
                            # Write the file back
                            with open(file_path, 'w', encoding='utf-8') as f:
                                f.writelines(lines)
                            
                            modified_files.append(file_path)
                            logger.debug(f"Added docstring to {node_type} {name} in {file_path}")
                            break
                
                except Exception as e:
                    logger.error(f"Error adding docstring to {file_path}: {e}")
                    raise
        
        return modified_files
    
    def _handle_style(self, planned_improvement: PlannedImprovement) -> List[Path]:
        """Handle style improvements."""
        improvement = planned_improvement.improvement
        file_path = improvement.location.file_path
        modified_files = []
        
        if not file_path:
            return modified_files
        
        # For style improvements, we can use external tools like black or ruff
        try:
            if self.config.lint_command:
                # Run the formatter
                result = subprocess.run(
                    [self.config.lint_command, "--fix", str(file_path)],
                    capture_output=True,
                    text=True,
                    check=False
                )
                
                if result.returncode == 0:
                    modified_files.append(file_path)
                    logger.debug(f"Applied style fixes to {file_path}")
                else:
                    logger.warning(f"Failed to apply style fixes to {file_path}: {result.stderr}")
        
        except Exception as e:
            logger.error(f"Error applying style fixes to {file_path}: {e}")
            raise
        
        return modified_files
    
    def _verify_changes(
        self, 
        modified_files: List[Path], 
        planned_improvement: PlannedImprovement
    ) -> Dict[VerificationType, bool]:
        """Verify changes by running tests, checking syntax, etc."""
        results = {}
        
        # Skip verification in dry run mode
        if self.config.dry_run:
            return {
                VerificationType.SYNTAX: True,
                VerificationType.TESTS: True,
                VerificationType.LINT: True
            }
        
        # Check syntax
        if self.config.verify_syntax:
            syntax_ok = self._verify_syntax(modified_files)
            results[VerificationType.SYNTAX] = syntax_ok
            
            # If syntax check fails, don't bother with other checks
            if not syntax_ok:
                logger.error("Syntax verification failed")
                return results
        
        # Run tests if configured
        if self.config.run_tests:
            tests_ok = self._run_tests()
            results[VerificationType.TESTS] = tests_ok
            
            if not tests_ok:
                logger.error("Test verification failed")
                return results
        
        # Run linting if configured
        if self.config.run_lint:
            lint_ok = self._run_lint(modified_files)
            results[VerificationType.LINT] = lint_ok
            
            if not lint_ok:
                logger.warning("Lint verification failed")
        
        return results
    
    def _verify_syntax(self, modified_files: List[Path]) -> bool:
        """Verify the syntax of modified files."""
        for file_path in modified_files:
            # For Python files, use the compile function to check syntax
            if file_path.suffix.lower() == '.py':
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    compile(content, str(file_path), 'exec')
                except SyntaxError as e:
                    logger.error(f"Syntax error in {file_path}: {e}")
                    return False
                except Exception as e:
                    logger.error(f"Error checking syntax of {file_path}: {e}")
                    return False
            
            # For other file types, we could add specific syntax checkers
        
        return True
    
    def _run_tests(self) -> bool:
        """Run tests to verify changes."""
        if not self.config.test_command:
            return True
        
        try:
            logger.info(f"Running tests with command: {self.config.test_command}")
            
            # Split the command into parts
            command_parts = self.config.test_command.split()
            
            # Run the tests
            result = subprocess.run(
                command_parts,
                cwd=self.root_path,
                capture_output=True,
                text=True,
                check=False,
                timeout=self.config.timeout_seconds
            )
            
            if result.returncode == 0:
                logger.info("Tests passed")
                return True
            else:
                logger.error(f"Tests failed: {result.stderr}")
                return False
        
        except subprocess.TimeoutExpired:
            logger.error(f"Tests timed out after {self.config.timeout_seconds} seconds")
            return False
        except Exception as e:
            logger.error(f"Error running tests: {e}")
            return False
    
    def _run_lint(self, modified_files: List[Path]) -> bool:
        """Run linting on modified files."""
        if not self.config.lint_command:
            return True
        
        try:
            # Run linting on each modified file
            for file_path in modified_files:
                logger.debug(f"Linting file: {file_path}")
                
                # Split the command into parts
                command_parts = self.config.lint_command.split()
                command_parts.append(str(file_path))
                
                # Run the linter
                result = subprocess.run(
                    command_parts,
                    capture_output=True,
                    text=True,
                    check=False
                )
                
                if result.returncode != 0:
                    logger.warning(f"Linting failed for {file_path}: {result.stderr}")
                    return False
            
            return True
        
        except Exception as e:
            logger.error(f"Error running linter: {e}")
            return False
    
    def _revert_changes(self, modified_files: List[Path]) -> bool:
        """Revert changes to modified files."""
        if self.config.dry_run:
            return True
        
        success = True
        
        for file_path in modified_files:
            try:
                if self.config.backup_strategy == BackupStrategy.COPY:
                    # Restore from backup
                    if self.backup_dir:
                        rel_path = file_path.relative_to(self.root_path)
                        backup_path = self.backup_dir / rel_path
                        
                        if backup_path.exists():
                            shutil.copy2(backup_path, file_path)
                            logger.debug(f"Reverted changes to {file_path} from backup")
                            continue
                
                # Fallback to backup file if it exists
                backup_path = file_path.with_suffix(file_path.suffix + ".bak")
                if backup_path.exists():
                    shutil.copy2(backup_path, file_path)
                    logger.debug(f"Reverted changes to {file_path} from .bak file")
                    continue
                
                # If using git, try to revert with git
                if self.config.backup_strategy == BackupStrategy.GIT:
                    try:
                        result = subprocess.run(
                            ["git", "checkout", "--", str(file_path)],
                            cwd=self.root_path,
                            capture_output=True,
                            text=True,
                            check=False
                        )
                        
                        if result.returncode == 0:
                            logger.debug(f"Reverted changes to {file_path} using git")
                            continue
                    except Exception as e:
                        logger.warning(f"Failed to revert {file_path} using git: {e}")
                
                logger.warning(f"Could not revert changes to {file_path}, no backup found")
                success = False
            
            except Exception as e:
                logger.error(f"Error reverting changes to {file_path}: {e}")
                success = False
        
        return success
    
    def undo_last_execution(self, execution_result: ExecutionResult) -> bool:
        """Undo the changes made in the last execution."""
        logger.info("Undoing last execution")
        
        if self.config.dry_run:
            logger.info("Dry run mode, nothing to undo")
            return True
        
        modified_files = execution_result.get_modified_files()
        if not modified_files:
            logger.info("No files were modified in the last execution")
            return True
        
        return self._revert_changes(modified_files)


def create_executor(
    root_path: Path,
    backup_strategy: str = "copy",
    verify_syntax: bool = True,
    run_tests: bool = True,
    run_lint: bool = False,
    test_command: str = "pytest",
    lint_command: str = "ruff",
    dry_run: bool = False,
    verbose: bool = False
) -> Executor:
    """Factory function to create an executor with the specified configuration."""
    # Map backup strategy name to enum
    strategy_map = {
        "none": BackupStrategy.NONE,
        "copy": BackupStrategy.COPY,
        "git": BackupStrategy.GIT
    }
    
    backup_strategy_enum = strategy_map.get(backup_strategy.lower(), BackupStrategy.COPY)
    
    config = ExecutorConfig(
        backup_strategy=backup_strategy_enum,
        verify_syntax=verify_syntax,
        run_tests=run_tests,
        run_lint=run_lint,
        test_command=test_command,
        lint_command=lint_command,
        dry_run=dry_run,
        verbose=verbose
    )
    
    return Executor(root_path=root_path, config=config)


def execute_plan(
    root_path: Path,
    plan: Plan,
    backup_strategy: str = "copy",
    verify_syntax: bool = True,
    run_tests: bool = True,
    run_lint: bool = False,
    test_command: str = "pytest",
    lint_command: str = "ruff",
    dry_run: bool = False,
    verbose: bool = False
) -> ExecutionResult:
    """Execute a plan of improvements."""
    executor = create_executor(
        root_path=root_path,
        backup_strategy=backup_strategy,
        verify_syntax=verify_syntax,
        run_tests=run_tests,
        run_lint=run_lint,
        test_command=test_command,
        lint_command=lint_command,
        dry_run=dry_run,
        verbose=verbose
    )
    
    return executor.execute_plan(plan)
