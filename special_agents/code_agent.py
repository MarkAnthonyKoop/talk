#!/usr/bin/env python3
# special_agents/code_agent.py

"""
CodeAgent - Specialized agent for generating code.

This agent takes a task description and generates code content.
It focuses purely on code generation, not file operations.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Dict, List, Optional, Any

from agent.agent import Agent

log = logging.getLogger(__name__)

class CodeAgent(Agent):
    """
    Specialized agent for generating code.
    
    This agent takes a task description and generates code content.
    It returns code along with metadata about dependencies and structure.
    """
    
    def __init__(self, **kwargs):
        """Initialize with specialized system prompt for code generation."""
        roles = kwargs.pop("roles", [])
        
        # Add code-specific system prompts
        code_system_prompts = [
            "You are an expert software developer who generates high-quality code.",
            "Focus on writing clean, maintainable, and well-structured code.",
            "Include proper error handling and follow best practices.",
            "Identify and list any external dependencies the code requires."
        ]
        
        # Combine with any existing roles
        roles = code_system_prompts + roles
        
        # Initialize the base agent
        super().__init__(roles=roles, **kwargs)
    
    def run(self, input_text: str) -> str:
        """
        Generate code based on the task description.
        
        Args:
            input_text: Task description and context
                
        Returns:
            JSON with code content and metadata
        """
        try:
            # Create a focused prompt for code generation
            prompt = self._format_code_prompt(input_text)
            
            # Get the LLM to generate code
            self._append("user", prompt)
            response = self.call_ai()
            self._append("assistant", response)
            
            # Parse and structure the response
            result = self._parse_code_response(response)
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            log.error(f"Error generating code: {str(e)}")
            return json.dumps({"error": str(e)}, indent=2)
    
    def _format_code_prompt(self, input_text: str) -> str:
        """
        Format a prompt focused on code generation.
        
        Args:
            input_text: Task description and context
            
        Returns:
            A formatted prompt for code generation
        """
        prompt = f"""
Task: {input_text.strip()}

Generate the code to accomplish this task.

Respond in this format:

CODE:
```language
[your code here]
```

DEPENDENCIES:
- package1
- package2

DESCRIPTION:
[Brief description of what the code does]

FILENAME_SUGGESTION:
[Suggested filename for this code]
"""
        return prompt
    
    def _parse_code_response(self, response: str) -> Dict[str, Any]:
        """
        Parse the LLM response to extract code and metadata.
        
        Args:
            response: Raw LLM response
            
        Returns:
            Dictionary with code, dependencies, and metadata
        """
        result = {
            "code": "",
            "language": "python",
            "dependencies": [],
            "description": "",
            "filename_suggestion": "main.py"
        }
        
        try:
            # Extract code block
            code_match = re.search(r'```(\w+)?\n([\s\S]*?)```', response)
            if code_match:
                result["language"] = code_match.group(1) or "python"
                result["code"] = code_match.group(2).rstrip()
            
            # Extract dependencies
            if "DEPENDENCIES:" in response:
                deps_section = response.split("DEPENDENCIES:")[1]
                deps_section = deps_section.split("\n\n")[0] if "\n\n" in deps_section else deps_section
                for line in deps_section.split("\n"):
                    line = line.strip()
                    if line.startswith("-"):
                        result["dependencies"].append(line[1:].strip())
                    elif line and not line.startswith("DESCRIPTION:"):
                        result["dependencies"].append(line)
            
            # Extract description
            if "DESCRIPTION:" in response:
                desc_section = response.split("DESCRIPTION:")[1]
                desc_section = desc_section.split("\n\n")[0] if "\n\n" in desc_section else desc_section
                result["description"] = desc_section.strip()
            
            # Extract filename suggestion
            if "FILENAME_SUGGESTION:" in response:
                filename_section = response.split("FILENAME_SUGGESTION:")[1]
                filename = filename_section.split("\n")[0].strip()
                if filename:
                    result["filename_suggestion"] = filename
            
        except Exception as e:
            log.warning(f"Error parsing response sections: {e}")
        
        return result
    
    def generate_file_operations(self, task: str, code_result: Dict[str, Any]) -> str:
        """
        Generate file operations from code result for FileAgent.
        
        This is a helper method that can be used by orchestrators
        to convert CodeAgent output to FileAgent input.
        
        Args:
            task: Original task description
            code_result: Result from run() method
            
        Returns:
            SEARCH/REPLACE format for FileAgent
        """
        if isinstance(code_result, str):
            code_result = json.loads(code_result)
        
        filename = code_result.get("filename_suggestion", "main.py")
        code = code_result.get("code", "")
        
        # Format as SEARCH/REPLACE for new file
        return f"""{filename}
<<<<<<< SEARCH
=======
{code}
>>>>>>> REPLACE"""
