#!/usr/bin/env python3
"""
VitalsAgent - System health monitor and rate limit handler.

This agent monitors the health of the workflow loop and implements
protective measures like rate limiting and backoff strategies.
It does NOT call any LLM - it's purely mechanical/administrative.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

from agent.agent import Agent
from plan_runner.step import Step

log = logging.getLogger(__name__)


class VitalsAgent(Agent):
    """
    System vitals monitor that acts as a safety gate in the workflow.
    
    This agent:
    1. Monitors system health metrics (errors, rate limits, loops)
    2. Implements sleep delays to respect rate limits
    3. Can override workflow decisions if system is unhealthy
    4. Passes through normally when system is healthy
    """
    
    # Class-level static tracking (persists across instances)
    _global_stats = {
        'total_calls': 0,
        'start_time': None,
        'start_datetime': None,
        'total_llm_calls': 0,
        'total_errors': 0,
        'total_rate_limits': 0,
        'recent_steps': deque(maxlen=50),
        'step_counts': {},  # Count of each step label
        'last_10_steps': deque(maxlen=10),
        'consecutive_errors': 0,
        'time_since_last_error': None,
        'current_backoff': 0,
        'workspace_files': {},  # filename -> (size, mtime)
        'workspace_dir': None,
        'last_vitals_save': None,
        'vitals_file': None
    }
    
    def __init__(self, 
                 monitored_step: Optional[Step] = None,
                 plan: Optional[list] = None,
                 vitals_step: Optional[Step] = None,
                 workspace_dir: Optional[str] = None,
                 **kwargs):
        """
        Initialize VitalsAgent.
        
        Args:
            monitored_step: The step being monitored (usually select_action)
            plan: The complete workflow plan
            vitals_step: Reference to this agent's own step
            workspace_dir: Path to workspace directory for file tracking
            **kwargs: Additional arguments for base Agent
        """
        # No LLM roles - this agent doesn't call AI
        super().__init__(roles=[], **kwargs)
        
        self.monitored_step = monitored_step  # The branching step we're monitoring
        self.vitals_step = vitals_step  # Our own step
        self.plan = plan or []
        
        # Initialize start time if first instance
        if self._global_stats['start_time'] is None:
            self._global_stats['start_time'] = time.time()
            self._global_stats['start_datetime'] = datetime.now().isoformat()
        
        # Set workspace directory if provided
        if workspace_dir:
            self._global_stats['workspace_dir'] = workspace_dir
        
        # Set vitals file location
        if self._global_stats['vitals_file'] is None and self._global_stats['workspace_dir']:
            # Put vitals.jsonl in the session directory (parent of workspace)
            workspace_path = Path(self._global_stats['workspace_dir'])
            if workspace_path.parent.exists():
                self._global_stats['vitals_file'] = workspace_path.parent / 'vitals.jsonl'
        
        # Gemini free tier limits (conservative)
        self.RATE_LIMIT_SLEEP = 5  # 5 seconds between requests (for 15 RPM limit)
        self.ERROR_BACKOFF_BASE = 10  # Base backoff after errors
        self.MAX_BACKOFF = 120  # Max 2 minutes backoff
        self.CRITICAL_ERROR_THRESHOLD = 10  # Halt after 10 consecutive errors
        
    def run(self, prompt: str) -> str:
        """
        Check system vitals and apply protective measures.
        
        Args:
            prompt: Input from previous step (usually BranchingAgent output)
            
        Returns:
            The prompt (passed through) or emergency message if critical
        """
        try:
            # Increment static call counter
            self._global_stats['total_calls'] += 1
            
            # Extract what step was selected (if we can detect it)
            selected_step = self._extract_selected_step(prompt)
            if selected_step:
                self._global_stats['last_10_steps'].append(selected_step)
                self._global_stats['recent_steps'].append({
                    'step': selected_step,
                    'time': time.time(),
                    'call_num': self._global_stats['total_calls']
                })
                # Track step frequency
                self._global_stats['step_counts'][selected_step] = \
                    self._global_stats['step_counts'].get(selected_step, 0) + 1
            
            # Check for rate limit errors in prompt
            if self._detected_rate_limit(prompt):
                self._handle_rate_limit()
                self._global_stats['total_rate_limits'] += 1
                self._global_stats['consecutive_errors'] += 1
            
            # Check for other errors
            elif self._detected_error(prompt):
                self._handle_error()
                self._global_stats['total_errors'] += 1
                self._global_stats['consecutive_errors'] += 1
            else:
                # Reset consecutive errors on success
                self._global_stats['consecutive_errors'] = 0
                self._global_stats['current_backoff'] = 0
            
            # Always apply minimum sleep to respect rate limits
            self._apply_rate_limit_sleep()
            
            # Check overall system health
            health_status = self._assess_health()
            
            # Log vitals every 5 calls
            if self._global_stats['total_calls'] % 5 == 0:
                log.info(f"System Vitals: {self._get_vitals_summary()}")
            
            # Add vitals info to prompt if things are getting bad
            vitals_info = ""
            if health_status in ['warning', 'critical']:
                vitals_info = f"\n\n[SYSTEM VITALS WARNING]\n{self._get_vitals_summary()}\n"
            elif self._global_stats['total_calls'] % 10 == 0:
                vitals_info = f"\n\n[SYSTEM VITALS]\n{self._get_vitals_summary()}\n"
            
            # Copy the selection from BranchingAgent before health check
            if self.vitals_step:
                # Check if we have access to BranchingAgent (v3b setup)
                selected_step = None
                if hasattr(self, 'branching_agent') and hasattr(self.branching_agent, 'selected_step'):
                    selected_step = self.branching_agent.selected_step
                    log.debug(f"VitalsAgent read selection from BranchingAgent: {selected_step}")
                elif self.monitored_step:
                    # Fallback to reading from step's on_success
                    selected_step = self.monitored_step.on_success
                    log.debug(f"VitalsAgent using step on_success: {selected_step}")
                
                if selected_step and selected_step != "check_vitals":
                    self.vitals_step.on_success = selected_step
                    log.debug(f"VitalsAgent routing to: {selected_step}")
            
            # Critical condition - halt workflow
            if health_status == 'critical':
                log.error("CRITICAL: System health critical, halting workflow")
                if self.vitals_step:
                    # Override to complete
                    self.vitals_step.on_success = "complete"
                return "EMERGENCY HALT: System in critical condition - too many errors"
            
            # Warning condition - increase delays
            elif health_status == 'warning':
                backoff = self._global_stats['current_backoff'] or self.ERROR_BACKOFF_BASE
                log.warning(f"System health warning - applying {backoff}s backoff")
                time.sleep(backoff)
            
            # Track what step was selected from the monitored step
            if self.monitored_step and hasattr(self.monitored_step, 'on_success'):
                monitored_selection = self.monitored_step.on_success
                if monitored_selection and monitored_selection != "check_vitals":
                    self._global_stats['last_10_steps'].append(monitored_selection)
                    self._global_stats['step_counts'][monitored_selection] = \
                        self._global_stats['step_counts'].get(monitored_selection, 0) + 1
            
            # Update workspace files tracking
            self._scan_workspace()
            
            # Save vitals to JSONL
            self._save_vitals()
            
            # Pass through with vitals info if needed
            return prompt + vitals_info if vitals_info else prompt
            
        except Exception as e:
            log.error(f"VitalsAgent error: {e}")
            # Even on error, apply safety sleep
            time.sleep(self.RATE_LIMIT_SLEEP)
            return prompt
    
    def _detected_rate_limit(self, prompt: str) -> bool:
        """Check if prompt contains rate limit error."""
        indicators = [
            "429",
            "rate limit",
            "quota",
            "exceeded",
            "too many requests",
            "retry_delay"
        ]
        prompt_lower = prompt.lower()
        return any(indicator in prompt_lower for indicator in indicators)
    
    def _detected_error(self, prompt: str) -> bool:
        """Check if prompt contains error indicators."""
        indicators = [
            "error",
            "failed",
            "exception",
            "backend error",
            "timeout"
        ]
        prompt_lower = prompt.lower()
        return any(indicator in prompt_lower for indicator in indicators)
    
    def _handle_rate_limit(self):
        """Handle rate limit detection."""
        self._global_stats['time_since_last_error'] = time.time()
        
        # Exponential backoff for rate limits
        if self._global_stats['current_backoff'] == 0:
            self._global_stats['current_backoff'] = self.ERROR_BACKOFF_BASE
        else:
            self._global_stats['current_backoff'] = min(
                self._global_stats['current_backoff'] * 2,
                self.MAX_BACKOFF
            )
        
        log.warning(f"Rate limit detected (#{self._global_stats['total_rate_limits']}), "
                   f"backing off {self._global_stats['current_backoff']}s")
        
        # Apply backoff
        time.sleep(self._global_stats['current_backoff'])
    
    def _handle_error(self):
        """Handle general error detection."""
        self._global_stats['time_since_last_error'] = time.time()
        
        log.warning(f"Error detected (#{self._global_stats['total_errors']}), "
                   f"consecutive: {self._global_stats['consecutive_errors']}")
    
    def _apply_rate_limit_sleep(self):
        """Apply minimum sleep to respect Gemini rate limits."""
        # Always apply a small sleep to avoid hammering the API
        base_sleep = 2.0  # 2 seconds minimum between calls
        
        # Add extra sleep if we're going too fast
        runtime = max(time.time() - self._global_stats['start_time'], 1)
        calls_per_min = (self._global_stats['total_calls'] / runtime) * 60
        
        if calls_per_min > 15:  # Gemini free tier is 15 RPM
            # We're going too fast, add more sleep
            extra_sleep = (calls_per_min / 15) * 2  # Proportional slowdown
            sleep_time = base_sleep + extra_sleep
            log.info(f"Rate limiting: {calls_per_min:.1f} calls/min, sleeping {sleep_time:.1f}s")
        else:
            sleep_time = base_sleep
            log.debug(f"Base rate limit sleep: {sleep_time:.1f}s")
        
        time.sleep(sleep_time)
    
    def _assess_health(self) -> str:
        """
        Assess overall system health.
        
        Returns:
            'healthy', 'warning', or 'critical'
        """
        stats = self._global_stats
        
        # Critical conditions
        if stats['consecutive_errors'] >= self.CRITICAL_ERROR_THRESHOLD:
            return 'critical'
        
        # Check for rate limit death spiral
        if stats['total_rate_limits'] > 20:
            return 'critical'
        
        # Check if stuck in tight loop
        if stats['total_calls'] > 50:
            return 'critical'
        
        # Warning conditions
        if stats['consecutive_errors'] >= 3:
            return 'warning'
        
        if stats['total_errors'] > 5:
            return 'warning'
        
        # Check for loops (same step repeated many times)
        last_10 = list(stats['last_10_steps'])
        if len(last_10) >= 10:
            unique_steps = set(last_10)
            if len(unique_steps) <= 2:  # Only 2 different steps in last 10
                return 'warning'
        
        # Check call rate (too fast might hit rate limits)
        runtime = max(time.time() - stats['start_time'], 1)
        calls_per_min = (stats['total_calls'] / runtime) * 60
        if calls_per_min > 30:  # More than 30 calls/min is concerning
            return 'warning'
        
        return 'healthy'
    
    def _extract_selected_step(self, prompt: str) -> Optional[str]:
        """Extract which step was selected from the prompt."""
        # Look for "Selected: X" pattern
        import re
        match = re.search(r'Selected:\s*(\w+)', prompt)
        if match:
            return match.group(1)
        return None
    
    def _get_vitals_summary(self) -> str:
        """Get a summary of current vitals."""
        runtime = int(time.time() - self._global_stats['start_time'])
        
        # Calculate rates
        calls_per_min = (self._global_stats['total_calls'] / max(runtime, 1)) * 60
        
        # Find most common steps
        top_steps = sorted(self._global_stats['step_counts'].items(), 
                          key=lambda x: x[1], reverse=True)[:3]
        top_steps_str = ', '.join([f"{s[0]}:{s[1]}" for s in top_steps])
        
        # Check for loops
        last_10 = list(self._global_stats['last_10_steps'])
        loop_warning = ""
        if len(last_10) >= 5:
            # Check if same step repeated too much
            from collections import Counter
            counts = Counter(last_10[-5:])
            if any(count >= 3 for count in counts.values()):
                loop_warning = " [LOOP DETECTED]"
        
        return (f"Calls: {self._global_stats['total_calls']}, "
                f"Runtime: {runtime}s, "
                f"Rate: {calls_per_min:.1f}/min, "
                f"Errors: {self._global_stats['total_errors']}, "
                f"RateLimits: {self._global_stats['total_rate_limits']}, "
                f"TopSteps: [{top_steps_str}]{loop_warning}")
    
    @classmethod
    def reset_stats(cls, workspace_dir: Optional[str] = None):
        """Reset global statistics (useful between runs)."""
        cls._global_stats = {
            'total_calls': 0,
            'start_time': time.time(),
            'start_datetime': datetime.now().isoformat(),
            'total_llm_calls': 0,
            'total_errors': 0,
            'total_rate_limits': 0,
            'recent_steps': deque(maxlen=50),
            'step_counts': {},
            'last_10_steps': deque(maxlen=10),
            'consecutive_errors': 0,
            'time_since_last_error': None,
            'current_backoff': 0,
            'workspace_files': {},
            'workspace_dir': workspace_dir,
            'last_vitals_save': None,
            'vitals_file': None
        }
        
        # Set vitals file if workspace provided
        if workspace_dir:
            workspace_path = Path(workspace_dir)
            if workspace_path.parent.exists():
                cls._global_stats['vitals_file'] = workspace_path.parent / 'vitals.jsonl'
        
        log.info("VitalsAgent stats reset")
    
    @classmethod
    def get_stats(cls) -> dict:
        """Get current global statistics."""
        return dict(cls._global_stats)
    
    @classmethod
    def should_pause(cls) -> bool:
        """Check if workflow should pause based on vitals."""
        # High error rate
        if cls._global_stats['consecutive_errors'] >= 5:
            return True
        # Too many calls too fast
        runtime = max(time.time() - cls._global_stats['start_time'], 1)
        calls_per_min = (cls._global_stats['total_calls'] / runtime) * 60
        if calls_per_min > 40:
            return True
        return False
    
    @property
    def brief_description(self) -> str:
        """Brief description for workflow."""
        return "Monitor system health and handle rate limits"
    
    def _scan_workspace(self) -> None:
        """Scan workspace directory for files and sizes."""
        if not self._global_stats['workspace_dir']:
            return
        
        workspace_path = Path(self._global_stats['workspace_dir'])
        if not workspace_path.exists():
            return
        
        new_files = {}
        total_size = 0
        
        try:
            # Recursively scan all files
            for file_path in workspace_path.rglob('*'):
                if file_path.is_file():
                    try:
                        stat = file_path.stat()
                        rel_path = file_path.relative_to(workspace_path)
                        new_files[str(rel_path)] = {
                            'size': stat.st_size,
                            'mtime': stat.st_mtime,
                            'mtime_iso': datetime.fromtimestamp(stat.st_mtime).isoformat()
                        }
                        total_size += stat.st_size
                    except Exception as e:
                        log.debug(f"Error scanning file {file_path}: {e}")
            
            # Update global stats
            self._global_stats['workspace_files'] = new_files
            self._global_stats['workspace_total_size'] = total_size
            self._global_stats['workspace_file_count'] = len(new_files)
            
        except Exception as e:
            log.debug(f"Error scanning workspace: {e}")
    
    def _save_vitals(self) -> None:
        """Save current vitals to JSONL file."""
        if not self._global_stats['vitals_file']:
            return
        
        try:
            # Prepare vitals snapshot
            vitals_snapshot = {
                'timestamp': datetime.now().isoformat(),
                'wall_time': time.time(),
                'elapsed_time': time.time() - self._global_stats['start_time'],
                'call_number': self._global_stats['total_calls'],
                'errors': self._global_stats['total_errors'],
                'rate_limits': self._global_stats['total_rate_limits'],
                'consecutive_errors': self._global_stats['consecutive_errors'],
                'step_counts': dict(self._global_stats['step_counts']),
                'last_10_steps': list(self._global_stats['last_10_steps']),
                'workspace_file_count': self._global_stats.get('workspace_file_count', 0),
                'workspace_total_size': self._global_stats.get('workspace_total_size', 0),
                'workspace_files': self._global_stats.get('workspace_files', {}),
                'calls_per_minute': self._calculate_rate(),
                'health_status': self._assess_health()
            }
            
            # Append to JSONL file
            with open(self._global_stats['vitals_file'], 'a') as f:
                f.write(json.dumps(vitals_snapshot) + '\n')
            
            self._global_stats['last_vitals_save'] = time.time()
            
        except Exception as e:
            log.debug(f"Error saving vitals: {e}")
    
    def _calculate_rate(self) -> float:
        """Calculate current calls per minute rate."""
        runtime = max(time.time() - self._global_stats['start_time'], 1)
        return (self._global_stats['total_calls'] / runtime) * 60
    
    @property
    def triggers(self) -> List[str]:
        """Words that suggest vitals checking is needed."""
        return ["health", "vitals", "monitor", "rate", "limit", "error"]