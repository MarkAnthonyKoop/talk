#!/usr/bin/env python3
# special_agents/improved_file_agent.py

"""
ImprovedFileAgent - Enhanced file editing using Aider's SEARCH/REPLACE format.

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


class ImprovedFileAgent(Agent):
    """
    Enhanced file agent using SEARCH/REPLACE blocks for reliable code editing.
    
    Supports:
    - SEARCH/REPLACE blocks for precise edits
    - File creation with proper formatting
    - Multiple edits per file
    - Validation of search blocks
    """
    
    def __init__(self, base_dir: Optional[str] = None, **kwargs):
        """Initialize the ImprovedFileAgent."""
        super().__init__(roles=[], **kwargs)
        self.base_dir = Path(base_dir or os.getcwd()).resolve()
        log.info(f"ImprovedFileAgent initialized with base directory: {self.base_dir}")
        
        # Track files for validation
        self._file_cache = {}
    
    def reply(self, input_text: str, **kwargs) -> str:
        """Process SEARCH/REPLACE blocks or file creation requests."""
        self._append("user", f"File edit request:\n{input_text}")
        
        try:
            results = self._process_edit_blocks(input_text)
            response = "\n".join(results)
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
        
        # Find all file paths and their associated blocks
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
        file_pattern = r'^([^\s]+\.(py|js|ts|jsx|tsx|java|cpp|c|h|go|rs|rb|php|swift|kt|scala|r|m|sql|sh|yaml|yml|json|xml|html|css|scss|vue|svelte))\s*$'
        
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