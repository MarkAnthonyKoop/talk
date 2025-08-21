#!/usr/bin/env python3
# special_agents/file_agent.py

"""
FileAgent - Enhanced file editing using Aider's SEARCH/REPLACE format.

This agent implements Aider's proven SEARCH/REPLACE block format for reliable
code editing that preserves indentation and formatting.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from agent.agent import Agent

log = logging.getLogger(__name__)


class FileAgent(Agent):
    """
    Enhanced file agent using SEARCH/REPLACE blocks for reliable code editing.
    
    Supports:
    - SEARCH/REPLACE blocks for precise edits
    - File creation with proper formatting
    - Multiple edits per file
    - Validation of search blocks
    """
    
    @property
    def brief_description(self) -> str:
        """Brief description for workflow."""
        return "Save code to files, apply changes, manage file operations"
    
    @property
    def triggers(self) -> List[str]:
        """Words that suggest file operations are needed."""
        return ["save", "write", "file", "apply", "persist", "store"]
    
    def __init__(self, base_dir: Optional[str] = None, **kwargs):
        """Initialize the FileAgent."""
        super().__init__(roles=[], **kwargs)
        self.base_dir = Path(base_dir or os.getcwd()).resolve()
        log.info(f"FileAgent initialized with base directory: {self.base_dir}")
        
        # Track files for validation
        self._file_cache = {}
    
    def run(self, input_text: str) -> str:
        """Run method that processes code files."""
        return self.reply(input_text)
    
    def reply(self, input_text: str, **kwargs) -> str:
        """Process SEARCH/REPLACE blocks or file creation requests."""
        # Check if this is coming from CodeAgent (has code blocks)
        if "```" in input_text and "filename:" in input_text:
            log.info("Detected CodeAgent output with code blocks")
            self._append("user", f"Applying code from CodeAgent")
        else:
            self._append("user", f"File edit request:\n{input_text[:200]}...")
        
        try:
            results = self._process_edit_blocks(input_text)
            response = "\n".join(results) if results else "No operations performed"
            self._append("assistant", response)
            return response
        except Exception as e:
            error_msg = f"ERROR: Failed to process edits: {str(e)}"
            log.error(error_msg, exc_info=True)
            self._append("assistant", error_msg)
            return error_msg
    
    def _process_edit_blocks(self, input_text: str) -> List[str]:
        """Process all edit blocks in the input."""
        results = []
        
        # First check if this is CodeAgent output with markdown blocks
        code_files = self._extract_code_files(input_text)
        if code_files:
            return self._write_code_files(code_files)
        
        # Check scratch directory for code files from CodeAgent
        code_files = self._load_from_scratch()
        if code_files:
            return self._write_code_files(code_files)
        
        # Otherwise try to parse SEARCH/REPLACE blocks
        file_blocks = self._parse_file_blocks(input_text)
        
        for file_path, blocks in file_blocks.items():
            try:
                if self._is_new_file(blocks):
                    result = self._create_file(file_path, blocks[0])
                else:
                    result = self._apply_edits(file_path, blocks)
                results.append(f"{file_path}: {result}")
            except Exception as e:
                results.append(f"{file_path}: ERROR - {str(e)}")
        
        return results
    
    def _parse_file_blocks(self, input_text: str) -> Dict[str, List[Dict]]:
        """Parse input text to extract file paths and their SEARCH/REPLACE blocks."""
        file_blocks = {}
        
        # Pattern to match file paths followed by edit blocks
        # Handles both fenced and unfenced formats
        file_pattern = r'^([^\s]+\.(py|js|ts|jsx|tsx|java|cpp|c|h|go|rs|rb|php|swift|kt|scala|r|m|sql|sh|yaml|yml|json|xml|html|css|scss|vue|svelte|txt|md|log|conf|cfg|ini|toml))\s*$'
        
        lines = input_text.split('\n')
        i = 0
        current_file = None
        
        while i < len(lines):
            line = lines[i].strip()
            
            # Check for file path
            match = re.match(file_pattern, line, re.IGNORECASE)
            if match:
                current_file = match.group(1)
                if current_file not in file_blocks:
                    file_blocks[current_file] = []
                i += 1
                continue
            
            # Check for SEARCH/REPLACE block
            if line in ['```', '```python', '```javascript', '```typescript'] or line.startswith('```'):
                # Look for SEARCH block
                search_start = i + 1
                if search_start < len(lines) and 'SEARCH' in lines[search_start]:
                    block = self._extract_search_replace_block(lines, search_start)
                    if block and current_file:
                        file_blocks[current_file].append(block)
                        i = block['end_index']
                        continue
            
            # Also handle direct SEARCH markers without fences
            if line.startswith('<<<<<<< SEARCH'):
                block = self._extract_search_replace_block(lines, i)
                if block and current_file:
                    file_blocks[current_file].append(block)
                    i = block['end_index']
                    continue
            
            i += 1
        
        return file_blocks
    
    def _extract_search_replace_block(self, lines: List[str], start_idx: int) -> Optional[Dict]:
        """Extract a single SEARCH/REPLACE block."""
        search_content = []
        replace_content = []
        in_search = False
        in_replace = False
        i = start_idx
        
        while i < len(lines):
            line = lines[i]
            
            if '<<<<<<< SEARCH' in line:
                in_search = True
                in_replace = False
            elif '=======' in line:
                in_search = False
                in_replace = True
            elif '>>>>>>> REPLACE' in line:
                return {
                    'search': '\n'.join(search_content),
                    'replace': '\n'.join(replace_content),
                    'end_index': i + 1
                }
            elif in_search:
                search_content.append(line)
            elif in_replace:
                replace_content.append(line)
            
            i += 1
        
        return None
    
    def _is_new_file(self, blocks: List[Dict]) -> bool:
        """Check if this is a new file creation (empty SEARCH block)."""
        return len(blocks) == 1 and blocks[0]['search'].strip() == ''
    
    def _create_file(self, file_path: str, block: Dict) -> str:
        """Create a new file with the given content."""
        try:
            full_path = self.base_dir / file_path
            
            # Create parent directories if needed
            full_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write the file
            content = block['replace']
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            log.info(f"Created file: {full_path}")
            return "SUCCESS - File created"
            
        except Exception as e:
            return f"ERROR - {str(e)}"
    
    def _load_from_scratch(self) -> List[Tuple[str, str]]:
        """Load code files from scratch directory created by CodeAgent."""
        files = []
        scratch_dir = self.base_dir / ".talk_scratch"
        
        if not scratch_dir.exists():
            return files
        
        # Look for latest code files
        code_files = list(scratch_dir.glob("*.*"))
        for file_path in code_files:
            if file_path.suffix in ['.py', '.js', '.ts', '.java', '.go', '.rb', '.rs']:
                try:
                    with open(file_path, 'r') as f:
                        content = f.read()
                    files.append((file_path.name, content))
                    log.info(f"Loaded code file from scratch: {file_path.name}")
                except Exception as e:
                    log.error(f"Failed to load {file_path}: {e}")
        
        return files
    
    def _extract_code_files(self, text: str) -> List[Tuple[str, str]]:
        """Extract code files from markdown blocks with filenames."""
        files = []
        lines = text.split('\n')
        i = 0
        
        log.debug(f"Extracting code files from text with {len(lines)} lines")
        
        while i < len(lines):
            line = lines[i]
            # Look for code block with filename comment
            if line.startswith('```'):
                lang = line[3:].strip()
                # Look for filename in next line
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    filename_match = re.match(r'^#\s*filename:\s*(.+)$', next_line)
                    if filename_match:
                        filename = filename_match.group(1).strip()
                        # Collect code until closing ```
                        code_lines = []
                        j = i + 2  # Skip filename line
                        while j < len(lines) and not lines[j].strip() == '```':
                            code_lines.append(lines[j])
                            j += 1
                        if code_lines:
                            files.append((filename, '\n'.join(code_lines)))
                            log.info(f"Found code file: {filename} with {len(code_lines)} lines")
                        i = j
            i += 1
        
        log.info(f"Extracted {len(files)} code files")
        return files
    
    def _write_code_files(self, files: List[Tuple[str, str]]) -> List[str]:
        """Write code files to workspace."""
        results = []
        for filename, content in files:
            try:
                full_path = self.base_dir / filename
                full_path.parent.mkdir(parents=True, exist_ok=True)
                with open(full_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                log.info(f"Created file: {full_path}")
                results.append(f"{filename}: SUCCESS - File created")
            except Exception as e:
                results.append(f"{filename}: ERROR - {str(e)}")
        return results
    
    def _apply_edits(self, file_path: str, blocks: List[Dict]) -> str:
        """Apply multiple SEARCH/REPLACE blocks to an existing file."""
        try:
            full_path = self.base_dir / file_path
            
            if not full_path.exists():
                return "ERROR - File not found"
            
            # Read the file
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            applied_count = 0
            
            # Apply each block
            for block in blocks:
                search_text = block['search']
                replace_text = block['replace']
                
                # Check if search text exists
                if search_text not in content:
                    log.warning(f"Search text not found in {file_path}")
                    continue
                
                # Replace only the first occurrence
                content = content.replace(search_text, replace_text, 1)
                applied_count += 1
            
            # Write back if changes were made
            if content != original_content:
                # Create backup
                self._create_backup(full_path)
                
                with open(full_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                return f"SUCCESS - Applied {applied_count}/{len(blocks)} edits"
            else:
                return "WARNING - No changes applied"
            
        except Exception as e:
            return f"ERROR - {str(e)}"
    
    def _create_backup(self, file_path: Path) -> None:
        """Create a backup of the file before modifying."""
        backup_dir = self.base_dir / '.talk_backups'
        backup_dir.mkdir(exist_ok=True)
        
        # Create timestamped backup
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{file_path.name}.{timestamp}.bak"
        backup_path = backup_dir / backup_name
        
        import shutil
        shutil.copy2(file_path, backup_path)
        log.debug(f"Created backup: {backup_path}")