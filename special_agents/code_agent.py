#!/usr/bin/env python3
"""
CodeAgent - Intelligent code generation specialist.

This agent uses the LLM to generate code implementations based on task descriptions.
It returns the raw LLM completion which includes code, explanations, and metadata.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

from agent.agent import Agent

log = logging.getLogger(__name__)


class CodeAgent(Agent):
    """
    Code generation agent that produces implementation code.
    
    This agent:
    1. Analyzes task requirements and context
    2. Uses the LLM to generate appropriate code
    3. Returns the LLM's completion directly (may include structured output)
    4. Can save code snippets to scratch for other agents
    """
    
    def __init__(self, **kwargs):
        """Initialize the CodeAgent with code-focused roles."""
        roles = [
            "You are an expert code generation agent.",
            "You analyze requirements and generate complete, working code implementations.",
            "You write clean, well-structured, idiomatic code with appropriate error handling.",
            "You consider best practices, performance, and maintainability.",
            "",
            "IMPORTANT GUIDELINES:",
            "1. Generate complete, runnable code - not just snippets",
            "2. Include necessary imports and dependencies",
            "3. Add helpful comments explaining complex logic",
            "4. Consider edge cases and error handling",
            "5. Follow the language's conventions and idioms",
            "",
            "OUTPUT FORMAT:",
            "You should structure your response to include:",
            "- The code implementation",
            "- Brief explanation of the approach",
            "- Any dependencies or requirements",
            "- Suggested filename(s) for the code",
            "",
            "You may output JSON for structured data, but the code itself should be in code blocks."
        ]
        super().__init__(roles=roles, **kwargs)
        
        self.scratch_dir = None
        self.last_generated_files = []
    
    def run(self, input_text: str) -> str:
        """
        Generate code based on the task description.
        
        Args:
            input_text: Task description, requirements, and context
                
        Returns:
            LLM completion containing code and explanations
        """
        try:
            # Build comprehensive prompt
            prompt = self._build_code_prompt(input_text)
            
            # Get the LLM to generate code
            self._append("user", prompt)
            completion = self.call_ai()
            self._append("assistant", completion)
            
            # Optionally save code blocks to scratch
            self._save_code_to_scratch(completion)
            
            # Return the raw LLM completion
            return completion
            
        except Exception as e:
            log.error(f"Error generating code: {str(e)}")
            # Return error as LLM-style response
            return f"""I encountered an error while generating code: {str(e)}

Let me try a simpler approach:

```python
# Error occurred, generating basic template
def main():
    # TODO: Implement the requested functionality
    pass

if __name__ == "__main__":
    main()
```

The error has been logged. Please provide more context if you need specific functionality."""
    
    def _build_code_prompt(self, input_text: str) -> str:
        """Build a comprehensive prompt for code generation."""
        # Check if we have context from previous planning
        context_parts = []
        
        # Try to load planning context from scratch
        planning_context = self._load_planning_context()
        if planning_context:
            context_parts.append("PLANNING CONTEXT:")
            if "todo_hierarchy" in planning_context:
                context_parts.append(f"Task breakdown:\n{planning_context['todo_hierarchy']}")
            if "analysis" in planning_context:
                context_parts.append(f"Analysis: {json.dumps(planning_context['analysis'], indent=2)}")
            context_parts.append("")
        
        # Check for existing code in workspace
        existing_files = self._check_existing_code()
        if existing_files:
            context_parts.append("EXISTING FILES IN WORKSPACE:")
            for file in existing_files[:10]:  # Limit to 10 files
                context_parts.append(f"- {file}")
            if len(existing_files) > 10:
                context_parts.append(f"... and {len(existing_files) - 10} more files")
            context_parts.append("")
        
        # Add the main task
        context_parts.append(f"TASK: {input_text}")
        
        # Add generation instructions
        context_parts.append("""
Generate code to accomplish the task above.

Structure your response as follows:

1. First, briefly explain your approach
2. Then provide the complete code implementation
3. List any dependencies or requirements
4. Suggest appropriate filename(s)

Use markdown code blocks with language specification (```python, ```javascript, etc.)

If the task requires multiple files, generate them all with clear file markers.

Example response format:

I'll create a [description of what you're building].

Here's the implementation:

```python
# filename: example.py
import necessary_modules

def main():
    # Implementation here
    pass
```

Dependencies:
- module1
- module2

This code should be saved as `example.py` and can be run with `python example.py`.
""")
        
        return "\n".join(context_parts)
    
    def _load_planning_context(self) -> Optional[Dict]:
        """Load planning context from scratch if available."""
        try:
            scratch_dir = Path.cwd() / ".talk_scratch"
            planning_file = scratch_dir / "latest_planning.json"
            
            if planning_file.exists():
                with open(planning_file) as f:
                    return json.load(f)
        except Exception as e:
            log.debug(f"Could not load planning context: {e}")
        
        return None
    
    def _check_existing_code(self) -> List[str]:
        """Check for existing code files in the workspace."""
        try:
            workspace = Path.cwd()
            
            # Common code file extensions
            extensions = ['.py', '.js', '.ts', '.java', '.cpp', '.c', '.go', '.rs', '.rb', '.php']
            
            files = []
            for ext in extensions:
                files.extend([str(f.relative_to(workspace)) for f in workspace.rglob(f"*{ext}") 
                             if not str(f).startswith('.') and 'node_modules' not in str(f)])
            
            return sorted(files)[:20]  # Limit to 20 files
            
        except Exception as e:
            log.debug(f"Could not check existing code: {e}")
            return []
    
    def _save_code_to_scratch(self, completion: str):
        """Extract and save code blocks to scratch for other agents."""
        try:
            # Create scratch directory if needed
            if not self.scratch_dir:
                scratch_dir = Path.cwd() / ".talk_scratch"
                scratch_dir.mkdir(exist_ok=True)
                self.scratch_dir = scratch_dir
            
            # Extract code blocks with filenames
            code_blocks = self._extract_code_blocks(completion)
            
            if code_blocks:
                # Save metadata about generated files
                metadata = {
                    "files": [],
                    "summary": self._extract_summary(completion)
                }
                
                for filename, language, code in code_blocks:
                    if filename:
                        # Save individual code file
                        code_file = self.scratch_dir / f"generated_{filename}"
                        with open(code_file, "w") as f:
                            f.write(code)
                        
                        metadata["files"].append({
                            "filename": filename,
                            "language": language,
                            "path": str(code_file),
                            "lines": len(code.splitlines())
                        })
                
                # Save metadata
                metadata_file = self.scratch_dir / "generated_code_metadata.json"
                with open(metadata_file, "w") as f:
                    json.dump(metadata, f, indent=2)
                
                self.last_generated_files = metadata["files"]
                log.info(f"Saved {len(metadata['files'])} code files to scratch")
                
        except Exception as e:
            log.debug(f"Could not save code to scratch: {e}")
    
    def _extract_code_blocks(self, text: str) -> List[Tuple[str, str, str]]:
        """Extract code blocks from markdown text.
        
        Returns:
            List of (filename, language, code) tuples
        """
        blocks = []
        
        # Pattern for code blocks with optional filename comment
        pattern = r'```(\w+)?\n((?:# filename: ([\w\.-]+)\n)?.*?)```'
        
        for match in re.finditer(pattern, text, re.DOTALL):
            language = match.group(1) or "text"
            code = match.group(2)
            
            # Check for filename in first line comment
            filename = None
            lines = code.split('\n')
            if lines and lines[0].startswith(('# filename:', '// filename:', '/* filename:')):
                filename_match = re.search(r'filename:\s*([\w\.-]+)', lines[0])
                if filename_match:
                    filename = filename_match.group(1)
                    # Remove the filename comment from code
                    code = '\n'.join(lines[1:])
            
            # If no filename, generate one based on language and count
            if not filename:
                ext_map = {
                    'python': 'py',
                    'javascript': 'js',
                    'typescript': 'ts',
                    'java': 'java',
                    'cpp': 'cpp',
                    'c': 'c',
                    'go': 'go',
                    'rust': 'rs',
                    'ruby': 'rb',
                    'php': 'php'
                }
                ext = ext_map.get(language, 'txt')
                count = len([b for b in blocks if b[1] == language]) + 1
                filename = f"code_{count}.{ext}"
            
            blocks.append((filename, language, code.strip()))
        
        return blocks
    
    def _extract_summary(self, text: str) -> str:
        """Extract a summary of what was generated."""
        # Look for first paragraph or explanation
        lines = text.split('\n')
        summary_lines = []
        
        for line in lines:
            # Skip code blocks and headers
            if line.startswith('```') or line.startswith('#'):
                if summary_lines:
                    break
                continue
            
            # Add non-empty lines to summary
            if line.strip():
                summary_lines.append(line.strip())
                if len(summary_lines) >= 3:  # Limit to 3 lines
                    break
        
        return ' '.join(summary_lines) if summary_lines else "Code implementation generated"