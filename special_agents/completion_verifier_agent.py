#!/usr/bin/env python3
"""
CompletionVerifier Agent - Validates task completion against success criteria.

This agent is the quality gatekeeper that determines if a task has been
truly completed or if additional work is needed. It evaluates both
quantitative metrics and qualitative assessments to make completion decisions.
"""

from __future__ import annotations

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from agent.agent import Agent

log = logging.getLogger(__name__)


@dataclass
class CompletionCriteria:
    """Represents completion criteria for a task."""
    min_files: int = 1
    min_lines: int = 10
    required_extensions: List[str] = None
    requires_tests: bool = False
    requires_docs: bool = False
    requires_examples: bool = False
    requires_config: bool = False
    min_packages: int = 1
    functional_requirements: List[str] = None
    
    def __post_init__(self):
        if self.required_extensions is None:
            self.required_extensions = []
        if self.functional_requirements is None:
            self.functional_requirements = []


class CompletionVerifierAgent(Agent):
    """
    Agent that verifies if tasks have been completed according to criteria.
    
    This agent analyzes the output of other agents and determines if the
    task has been sufficiently completed or if more work is needed.
    """
    
    def __init__(self, **kwargs):
        """Initialize the completion verifier."""
        roles = [
            "You are an expert completion verifier for software development tasks.",
            "You evaluate whether tasks have been completed according to their requirements.",
            "You provide detailed analysis of what's missing and what needs improvement.",
            "You make decisive recommendations about whether work is complete or needs continuation."
        ]
        super().__init__(roles=roles, **kwargs)
        
        # Default criteria for different task types
        self.criteria_templates = {
            "simple": CompletionCriteria(
                min_files=1,
                min_lines=5,
                requires_tests=False,
                requires_docs=False
            ),
            "moderate": CompletionCriteria(
                min_files=3,
                min_lines=100,
                requires_tests=True,
                requires_docs=True,
                min_packages=1
            ),
            "complex": CompletionCriteria(
                min_files=10,
                min_lines=500,
                requires_tests=True,
                requires_docs=True,
                requires_examples=True,
                requires_config=True,
                min_packages=3
            ),
            "epic": CompletionCriteria(
                min_files=25,
                min_lines=2000,
                requires_tests=True,
                requires_docs=True,
                requires_examples=True,
                requires_config=True,
                min_packages=5,
                required_extensions=[".py", ".md", ".yml", ".json"],
                functional_requirements=[
                    "Working examples",
                    "Installation instructions",
                    "API documentation",
                    "Configuration files",
                    "Test suites"
                ]
            )
        }
    
    def verify_completion(self, 
                         workspace_path: str, 
                         task_description: str,
                         complexity: str = "moderate",
                         custom_criteria: Optional[CompletionCriteria] = None) -> Dict[str, Any]:
        """
        Verify if a task has been completed satisfactorily.
        
        Args:
            workspace_path: Path to the workspace to analyze
            task_description: Original task description
            complexity: Task complexity level
            custom_criteria: Custom completion criteria
            
        Returns:
            Dictionary with completion analysis
        """
        # Get criteria
        criteria = custom_criteria or self.criteria_templates.get(complexity, 
                                                                 self.criteria_templates["moderate"])
        
        # Analyze workspace
        workspace_analysis = self._analyze_workspace(workspace_path)
        
        # Check quantitative criteria
        quantitative_check = self._check_quantitative_criteria(workspace_analysis, criteria)
        
        # LLM qualitative assessment
        qualitative_check = self._llm_qualitative_assessment(
            workspace_analysis, task_description, criteria
        )
        
        # Combine assessments
        return self._combine_completion_analysis(
            quantitative_check, qualitative_check, criteria
        )
    
    def _analyze_workspace(self, workspace_path: str) -> Dict[str, Any]:
        """Analyze workspace contents."""
        workspace = Path(workspace_path)
        if not workspace.exists():
            return {"error": "Workspace does not exist", "files": [], "total_lines": 0}
        
        analysis = {
            "total_files": 0,
            "total_lines": 0,
            "files_by_type": {},
            "packages": set(),
            "has_tests": False,
            "has_docs": False,
            "has_examples": False,
            "has_config": False,
            "file_list": []
        }
        
        for file_path in workspace.rglob("*"):
            if file_path.is_file():
                analysis["total_files"] += 1
                
                # Count lines
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = len(f.readlines())
                        analysis["total_lines"] += lines
                except:
                    lines = 0
                
                # Categorize file
                ext = file_path.suffix.lower()
                if ext not in analysis["files_by_type"]:
                    analysis["files_by_type"][ext] = 0
                analysis["files_by_type"][ext] += 1
                
                # Track packages (directories with __init__.py or multiple files)
                parent = file_path.parent
                if parent != workspace:
                    analysis["packages"].add(str(parent.relative_to(workspace)))
                
                # Check for special file types
                name_lower = file_path.name.lower()
                if "test" in name_lower or ext in [".test"]:
                    analysis["has_tests"] = True
                if ext in [".md", ".rst", ".txt"] and any(doc in name_lower for doc in ["readme", "doc", "guide"]):
                    analysis["has_docs"] = True
                if "example" in name_lower or "sample" in name_lower:
                    analysis["has_examples"] = True
                if ext in [".yml", ".yaml", ".json", ".ini", ".cfg", ".toml"] or "config" in name_lower:
                    analysis["has_config"] = True
                
                analysis["file_list"].append({
                    "path": str(file_path.relative_to(workspace)),
                    "lines": lines,
                    "type": ext
                })
        
        analysis["packages"] = list(analysis["packages"])
        analysis["num_packages"] = len(analysis["packages"])
        
        return analysis
    
    def _check_quantitative_criteria(self, analysis: Dict, criteria: CompletionCriteria) -> Dict[str, Any]:
        """Check quantitative completion criteria."""
        checks = {
            "min_files": analysis["total_files"] >= criteria.min_files,
            "min_lines": analysis["total_lines"] >= criteria.min_lines,
            "min_packages": analysis["num_packages"] >= criteria.min_packages,
            "requires_tests": not criteria.requires_tests or analysis["has_tests"],
            "requires_docs": not criteria.requires_docs or analysis["has_docs"],
            "requires_examples": not criteria.requires_examples or analysis["has_examples"],
            "requires_config": not criteria.requires_config or analysis["has_config"],
        }
        
        # Check required extensions
        if criteria.required_extensions:
            has_required_exts = all(
                ext in analysis["files_by_type"] 
                for ext in criteria.required_extensions
            )
            checks["required_extensions"] = has_required_exts
        else:
            checks["required_extensions"] = True
        
        return {
            "checks": checks,
            "passed": all(checks.values()),
            "score": sum(checks.values()) / len(checks)
        }
    
    def _llm_qualitative_assessment(self, analysis: Dict, task: str, criteria: CompletionCriteria) -> Dict:
        """Use LLM for qualitative assessment."""
        prompt = f"""Evaluate the completion quality of this development task:

TASK: {task}

WORKSPACE ANALYSIS:
- Total files: {analysis['total_files']}
- Total lines: {analysis['total_lines']}
- Packages: {analysis['num_packages']}
- File types: {analysis['files_by_type']}
- Has tests: {analysis['has_tests']}
- Has docs: {analysis['has_docs']}
- Has examples: {analysis['has_examples']}
- Has config: {analysis['has_config']}

FILES CREATED:
{json.dumps(analysis['file_list'][:10], indent=2)}

COMPLETION CRITERIA:
- Minimum {criteria.min_files} files, {criteria.min_lines} lines
- Tests required: {criteria.requires_tests}
- Documentation required: {criteria.requires_docs}
- Examples required: {criteria.requires_examples}
- Functional requirements: {criteria.functional_requirements}

Provide your assessment in this format:

COMPLETION_STATUS: [COMPLETE|INCOMPLETE|PARTIAL]
QUALITY_SCORE: [0-10]
MISSING_COMPONENTS: [bullet list]
STRENGTHS: [bullet list]
RECOMMENDATIONS: [bullet list]
NEXT_STEPS: [bullet list if incomplete]

Focus on:
1. Does the output match the task's intent and scope?
2. Is the code quality appropriate for the complexity level?
3. Are there any critical missing components?
4. Could this be used by someone else as-is?
"""
        
        response = self.run(prompt)
        return self._parse_qualitative_response(response)
    
    def _parse_qualitative_response(self, response: str) -> Dict:
        """Parse LLM qualitative response."""
        result = {
            "status": "UNKNOWN",
            "quality_score": 5,
            "missing_components": [],
            "strengths": [],
            "recommendations": [],
            "next_steps": []
        }
        
        lines = response.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            
            if line.startswith("COMPLETION_STATUS:"):
                result["status"] = line.split(":", 1)[1].strip()
            elif line.startswith("QUALITY_SCORE:"):
                try:
                    result["quality_score"] = float(line.split(":", 1)[1].strip())
                except:
                    result["quality_score"] = 5
            elif line.startswith("MISSING_COMPONENTS:"):
                current_section = "missing"
            elif line.startswith("STRENGTHS:"):
                current_section = "strengths"
            elif line.startswith("RECOMMENDATIONS:"):
                current_section = "recommendations"
            elif line.startswith("NEXT_STEPS:"):
                current_section = "next_steps"
            elif line.startswith("-") or line.startswith("•"):
                item = line.lstrip("-•").strip()
                if current_section == "missing":
                    result["missing_components"].append(item)
                elif current_section == "strengths":
                    result["strengths"].append(item)
                elif current_section == "recommendations":
                    result["recommendations"].append(item)
                elif current_section == "next_steps":
                    result["next_steps"].append(item)
        
        return result
    
    def _combine_completion_analysis(self, quantitative: Dict, qualitative: Dict, criteria: CompletionCriteria) -> Dict[str, Any]:
        """Combine quantitative and qualitative assessments."""
        # Determine overall completion
        quant_passed = quantitative["passed"]
        qual_complete = qualitative["status"] == "COMPLETE"
        
        if quant_passed and qual_complete:
            overall_status = "COMPLETE"
        elif quantitative["score"] >= 0.7 and qualitative["quality_score"] >= 7:
            overall_status = "ACCEPTABLE"
        else:
            overall_status = "INCOMPLETE"
        
        return {
            "overall_status": overall_status,
            "is_complete": overall_status == "COMPLETE",
            "is_acceptable": overall_status in ["COMPLETE", "ACCEPTABLE"],
            "quantitative": quantitative,
            "qualitative": qualitative,
            "criteria_used": criteria,
            "recommendation": self._generate_recommendation(overall_status, quantitative, qualitative)
        }
    
    def _generate_recommendation(self, status: str, quant: Dict, qual: Dict) -> str:
        """Generate final recommendation."""
        if status == "COMPLETE":
            return "Task completed successfully. Ready for delivery."
        elif status == "ACCEPTABLE":
            return "Task substantially complete with minor gaps. Consider additional polish."
        else:
            missing = qual.get("missing_components", [])
            next_steps = qual.get("next_steps", [])
            
            recommendation = "Task incomplete. Additional work needed:\n"
            if missing:
                recommendation += "\nMissing components:\n" + "\n".join(f"- {item}" for item in missing)
            if next_steps:
                recommendation += "\nNext steps:\n" + "\n".join(f"- {item}" for item in next_steps)
            
            return recommendation