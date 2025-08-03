#!/usr/bin/env python3
# special_agents/code_agent.py

"""
CodeAgent - Specialized agent for generating code changes as unified diffs.

This agent takes a task description and current file content as input,
then uses an LLM to generate appropriate code changes in unified diff format.
"""

from __future__ import annotations

import difflib
import logging
import re
from typing import Dict, List, Optional, Tuple, Union

from agent.agent import Agent
from agent.messages import Message, Role

log = logging.getLogger(__name__)

class CodeAgent(Agent):
    """
    Specialized agent for generating code changes.
    
    This agent takes a task description and current file content,
    then generates a unified diff that can be applied by a FileAgent.
    """
    
    def __init__(self, **kwargs):
        """Initialize with specialized system prompt for code generation."""
        roles = kwargs.pop("roles", [])
        
        # Add code-specific system prompts
        code_system_prompts = [
            "You are an expert software developer. Your task is to generate code changes as unified diffs.",
            "Always respond with valid unified diff format that can be applied with the 'patch' command.",
            "Include the full context in your diffs to ensure they apply cleanly.",
            "Be precise and focused on the requested changes only."
        ]
        
        # Combine with any existing roles
        roles = code_system_prompts + roles
        
        # Initialize the base agent
        super().__init__(roles=roles, **kwargs)
    
    def run(self, input_text: str) -> str:
        """
        Process the input and generate structured file operations.
        
        Args:
            input_text: Project-level context containing:
                - Task description
                - Existing files information
                - Research context (optional)
                
        Returns:
            A structured response with file operations that can be applied by FileAgent
        """
        try:
            # Create a project-level prompt for intelligent code generation
            prompt = self._format_project_prompt(input_text)
            
            # Get the LLM to generate file operations
            self._append("user", prompt)
            response = self.call_ai()
            self._append("assistant", response)
            
            # Parse and structure the response into file operations
            structured_operations = self._parse_file_operations(response)
            
            return structured_operations
            
        except Exception as e:
            log.error(f"Error generating code: {str(e)}")
            return f"ERROR: Failed to generate diff: {str(e)}"
    
    def _format_project_prompt(self, input_text: str) -> str:
        """
        Format a project-level prompt for intelligent code generation.
        
        Args:
            input_text: Raw input from TalkOrchestrator containing task and context
            
        Returns:
            A formatted prompt for the LLM that encourages structured output
        """
        prompt = f"""
You are an expert software developer. Given the following project context, generate the necessary code files.

{input_text.strip()}

IMPORTANT: Respond with a structured format that specifies exactly what files to create or modify.

Use this exact format for your response:

FILE_OPERATIONS:
CREATE: filename.ext
```
[file content here]
```

MODIFY: existing_file.ext
```diff
[unified diff here]
```

CREATE: another_file.ext
```
[file content here]
```

Focus on:
1. Creating well-structured, working code
2. Following best practices for the language/framework
3. Including appropriate error handling
4. Making the code production-ready

Generate only the files needed to complete the task successfully.
"""
        return prompt
    
    def _parse_file_operations(self, response: str) -> str:
        """
        Parse the LLM response and convert it to a format FileAgent can understand.
        
        Args:
            response: Raw LLM response with FILE_OPERATIONS structure
            
        Returns:
            Formatted string with file operations for FileAgent
        """
        try:
            # Look for FILE_OPERATIONS section
            if "FILE_OPERATIONS:" not in response:
                # Fallback: treat the whole response as a single file creation
                return self._fallback_single_file(response)
            
            # Extract the operations section
            operations_start = response.find("FILE_OPERATIONS:") + len("FILE_OPERATIONS:")
            operations_text = response[operations_start:].strip()
            
            # Parse each operation
            formatted_output = ""
            current_operation = None
            current_filename = None
            code_block = ""
            in_code_block = False
            
            for line in operations_text.split('\n'):
                line = line.strip()
                
                if line.startswith("CREATE:") or line.startswith("MODIFY:"):
                    # Save previous operation if exists
                    if current_operation and current_filename:
                        formatted_output += self._format_operation(current_operation, current_filename, code_block)
                    
                    # Start new operation
                    parts = line.split(":", 1)
                    current_operation = parts[0].strip()
                    current_filename = parts[1].strip()
                    code_block = ""
                    in_code_block = False
                
                elif line.startswith("```"):
                    if in_code_block:
                        # End of code block
                        in_code_block = False
                    else:
                        # Start of code block
                        in_code_block = True
                        code_block = ""
                
                elif in_code_block:
                    code_block += line + "\n"
            
            # Save final operation
            if current_operation and current_filename:
                formatted_output += self._format_operation(current_operation, current_filename, code_block)
            
            return formatted_output.strip()
            
        except Exception as e:
            log.error(f"Error parsing file operations: {e}")
            return self._fallback_single_file(response)
    
    def _format_operation(self, operation: str, filename: str, content: str) -> str:
        """Format a single file operation for FileAgent."""
        if operation == "CREATE":
            return f"CREATE_FILE: {filename}\n{content.strip()}\n\n"
        elif operation == "MODIFY":
            return f"MODIFY_FILE: {filename}\n{content.strip()}\n\n"
        else:
            return f"CREATE_FILE: {filename}\n{content.strip()}\n\n"
    
    def _fallback_single_file(self, response: str) -> str:
        """Fallback parser for responses that don't follow the expected format."""
        # Try to extract a Python script from the response
        code_blocks = re.findall(r'```(?:python)?\s*(.*?)```', response, re.DOTALL)
        
        if code_blocks:
            content = code_blocks[0].strip()
            # Guess filename based on content
            if "print(" in content and "hello" in content.lower():
                filename = "hello.py"
            elif "app.run(" in content or "Flask" in content:
                filename = "app.py"
            else:
                filename = "main.py"
            
            return f"CREATE_FILE: {filename}\n{content}\n"
        
        # Last resort: treat entire response as a Python file
        return f"CREATE_FILE: main.py\n{response.strip()}\n"
    
    def _format_code_prompt(self, task: str, file_path: str, content: str) -> str:
        """
        Format a specialized prompt for code generation.
        
        Args:
            task: Description of the code change to make
            file_path: Path to the file being modified
            content: Current content of the file
            
        Returns:
            A formatted prompt for the LLM
        """
        prompt = f"""
Task: {task}

File: {file_path}

Current content:
```
{content}
```

Please generate a unified diff that implements the requested changes.
The diff should be in the standard unified diff format that can be applied with the 'patch' command.
Include only the diff in your response, no explanations or additional text.
"""
        return prompt
    
    def _extract_and_validate_diff(self, diff_text: str, file_path: str) -> str:
        """
        Extract and validate the diff from the LLM response.
        
        Args:
            diff_text: The raw response from the LLM
            file_path: The path to the file being modified
            
        Returns:
            A cleaned, validated diff string
        """
        # Extract the diff from the response (in case there's markdown or explanations)
        # NOTE: escape '-' inside character class so it is not parsed as a range
        diff_pattern = r'(--- .*?\n\+\+\+ .*?(?:\n@@.*?@@.*?)(?:\n@@.*?@@.*?)*(?:\n[+\-\s].*)*)'
        diff_matches = re.findall(diff_pattern, diff_text, re.DOTALL)
        
        if not diff_matches:
            # If no standard diff format found, try to extract code blocks
            code_block_pattern = r'```(?:diff)?\s*(.*?)```'
            code_blocks = re.findall(code_block_pattern, diff_text, re.DOTALL)
            
            if code_blocks:
                for block in code_blocks:
                    # Check if this block looks like a diff
                    if '---' in block and '+++' in block:
                        diff_text = block.strip()
                        break
            else:
                # If still no diff found, try to create one from the response
                return self._create_diff_from_response(diff_text, file_path)
        else:
            diff_text = diff_matches[0]
        
        # Validate that it's a proper unified diff
        if not (diff_text.startswith('---') and '+++' in diff_text and '@@' in diff_text):
            raise ValueError("Generated text is not a valid unified diff")
        
        return diff_text
    
    def _create_diff_from_response(self, response: str, file_path: str) -> str:
        """
        Attempt to create a diff if the LLM didn't generate one properly.
        
        This is a fallback for when the LLM returns complete file content instead of a diff.
        
        Args:
            response: The LLM response
            file_path: The path to the original file
            
        Returns:
            A unified diff string
        """
        # Extract what appears to be code content
        code_pattern = r'```(?:python|java|javascript|typescript|go|rust|cpp|c\+\+|c#|csharp|ruby|php)?\s*(.*?)```'
        code_blocks = re.findall(code_pattern, response, re.DOTALL)
        
        if not code_blocks:
            # If no code blocks with language markers, try without language
            code_blocks = re.findall(r'```\s*(.*?)```', response, re.DOTALL)
        
        if not code_blocks:
            raise ValueError("Could not extract code content from the response")
        
        # Use the first code block as the new content
        new_content = code_blocks[0].strip()
        
        # Get the file content from the last user message
        old_content = ""
        for msg in reversed(self.conversation.messages):
            if msg.role == Role.USER:
                # Try to extract the file content from the user message
                content_match = re.search(r'```\s*(.*?)```', msg.content, re.DOTALL)
                if content_match:
                    old_content = content_match.group(1).strip()
                    break
        
        if not old_content:
            # Fallback: try to find content after "Current content:" in the user message
            for msg in reversed(self.conversation.messages):
                if msg.role == Role.USER and "Current content:" in msg.content:
                    parts = msg.content.split("Current content:")
                    if len(parts) > 1:
                        content_part = parts[1].strip()
                        if content_part.startswith("```") and "```" in content_part[3:]:
                            old_content = content_part.split("```", 2)[1].strip()
                            break
        
        if not old_content:
            raise ValueError("Could not determine original file content")
        
        # Generate a unified diff
        diff = difflib.unified_diff(
            old_content.splitlines(keepends=True),
            new_content.splitlines(keepends=True),
            fromfile=f"a/{file_path}",
            tofile=f"b/{file_path}"
        )
        
        return "".join(diff)
