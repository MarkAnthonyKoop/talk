#!/usr/bin/env python3
"""
Security Validator Agent

Validates commands and actions for safety.
Part of Listen v7's agentic architecture.
"""

import re
import logging
from typing import Dict, Any, Tuple, List
from enum import Enum
from agent.agent import Agent

log = logging.getLogger(__name__)


class SecurityLevel(Enum):
    """Security levels for operations."""
    SAFE = 1
    CAUTION = 2
    DANGEROUS = 3
    FORBIDDEN = 4


class SecurityValidatorAgent(Agent):
    """
    Agent responsible for security validation.
    
    Ensures all actions are safe before execution.
    """
    
    def __init__(self, strict_mode: bool = True, **kwargs):
        roles = [
            "You validate the safety of system operations",
            "You prevent dangerous or destructive commands",
            "You protect system files and user data",
            "You identify security risks and vulnerabilities",
            "You enforce security policies consistently"
        ]
        
        super().__init__(roles=roles, **kwargs)
        
        self.strict_mode = strict_mode
        self.blocked_commands = []
        self.approved_commands = []
        
        # Security rules
        self.forbidden_patterns = [
            r"rm\s+-rf\s+/",
            r":\(\)\{.*:\|:&\s*\};:",  # Fork bomb
            r"dd\s+if=/dev/(zero|random)",
            r"mkfs\.",
            r"format\s+[cC]:",
            r"sudo\s+rm",
            r"chmod\s+777\s+/",
            r"chown\s+.*\s+/"
        ]
        
        self.dangerous_paths = [
            "/etc", "/sys", "/proc", "/boot", "/dev",
            "/usr/bin", "/usr/sbin", "/bin", "/sbin",
            "C:\\Windows", "C:\\System32"
        ]
        
        log.info(f"SecurityValidatorAgent initialized (strict: {strict_mode})")
    
    async def validate(self, intent: Any) -> Dict[str, Any]:
        """
        Validate the safety of an intent.
        
        This is the main action called by PlanRunner.
        """
        # Extract command from intent
        if hasattr(intent, 'command'):
            command = intent.command
        elif isinstance(intent, dict):
            command = intent.get("command", "")
        else:
            command = str(intent)
        
        # Perform validation
        security_level = self._assess_security_level(command)
        is_safe = security_level in [SecurityLevel.SAFE, SecurityLevel.CAUTION]
        
        # Strict mode blocks CAUTION level too
        if self.strict_mode and security_level == SecurityLevel.CAUTION:
            is_safe = False
        
        # Record decision
        if is_safe:
            self.approved_commands.append(command)
        else:
            self.blocked_commands.append(command)
        
        return {
            "approved": is_safe,
            "security_level": security_level.name,
            "command": command,
            "reason": self._get_security_reason(security_level),
            "recommendations": self._get_recommendations(command, security_level)
        }
    
    def _assess_security_level(self, command: str) -> SecurityLevel:
        """Assess the security level of a command."""
        command_lower = command.lower()
        
        # Check forbidden patterns
        for pattern in self.forbidden_patterns:
            if re.search(pattern, command, re.IGNORECASE):
                return SecurityLevel.FORBIDDEN
        
        # Check dangerous paths
        for path in self.dangerous_paths:
            if path.lower() in command_lower and "ls" not in command_lower:
                return SecurityLevel.DANGEROUS
        
        # Check specific dangerous commands
        if any(cmd in command_lower for cmd in ["sudo", "su ", "doas"]):
            return SecurityLevel.DANGEROUS
        
        if any(cmd in command_lower for cmd in ["rm ", "del ", "rmdir", "shred"]):
            return SecurityLevel.CAUTION
        
        if any(cmd in command_lower for cmd in ["mv ", "move ", "cp ", "copy "]):
            return SecurityLevel.CAUTION
        
        # Safe commands
        safe_commands = ["ls", "pwd", "echo", "cat", "grep", "find", "ps", "df", "free"]
        if any(cmd in command_lower for cmd in safe_commands):
            return SecurityLevel.SAFE
        
        # Default to caution for unknown commands
        return SecurityLevel.CAUTION
    
    def _get_security_reason(self, level: SecurityLevel) -> str:
        """Get explanation for security decision."""
        reasons = {
            SecurityLevel.SAFE: "Command is safe to execute",
            SecurityLevel.CAUTION: "Command requires careful review",
            SecurityLevel.DANGEROUS: "Command could damage system or data",
            SecurityLevel.FORBIDDEN: "Command is explicitly forbidden for safety"
        }
        return reasons.get(level, "Unknown security level")
    
    def _get_recommendations(self, command: str, level: SecurityLevel) -> List[str]:
        """Get security recommendations."""
        recommendations = []
        
        if level == SecurityLevel.FORBIDDEN:
            recommendations.append("Do not execute this command under any circumstances")
            recommendations.append("Consider a safer alternative")
        
        elif level == SecurityLevel.DANGEROUS:
            recommendations.append("Requires explicit user confirmation")
            recommendations.append("Consider running in a sandbox environment")
            recommendations.append("Ensure you have backups before proceeding")
        
        elif level == SecurityLevel.CAUTION:
            if "rm" in command.lower():
                recommendations.append("Consider using 'rm -i' for interactive confirmation")
                recommendations.append("Verify the target path is correct")
            elif "mv" in command.lower() or "cp" in command.lower():
                recommendations.append("Check if target already exists to avoid overwriting")
        
        return recommendations
    
    async def health_check(self) -> Dict[str, Any]:
        """Check security validator health."""
        return {
            "status": "healthy",
            "strict_mode": self.strict_mode,
            "commands_blocked": len(self.blocked_commands),
            "commands_approved": len(self.approved_commands),
            "security_rules": len(self.forbidden_patterns)
        }
    
    async def get_security_report(self) -> Dict[str, Any]:
        """Get security activity report."""
        return {
            "total_validations": len(self.approved_commands) + len(self.blocked_commands),
            "approved": len(self.approved_commands),
            "blocked": len(self.blocked_commands),
            "recent_blocked": self.blocked_commands[-5:] if self.blocked_commands else [],
            "strict_mode": self.strict_mode
        }
    
    async def update_policy(self, policy: Dict[str, Any]):
        """Update security policy dynamically."""
        if "strict_mode" in policy:
            self.strict_mode = policy["strict_mode"]
            log.info(f"Security mode updated: strict={self.strict_mode}")
        
        if "additional_forbidden" in policy:
            self.forbidden_patterns.extend(policy["additional_forbidden"])
            log.info(f"Added {len(policy['additional_forbidden'])} forbidden patterns")
    
    async def run(self, prompt: str, context: Dict[str, Any] = None) -> str:
        """
        Agent interface for security analysis.
        """
        if "validate" in prompt.lower():
            command = context.get("command", "") if context else ""
            result = await self.validate({"command": command})
            return f"Security: {result['security_level']} - {result['reason']}"
        
        elif "report" in prompt.lower():
            report = await self.get_security_report()
            return f"Security Report: {report['blocked']} blocked, {report['approved']} approved"
        
        return f"SecurityValidatorAgent: {prompt}"
    
    async def cleanup(self):
        """Clean up security validator resources."""
        if self.blocked_commands:
            log.warning(f"Session blocked {len(self.blocked_commands)} dangerous commands")
        log.info("SecurityValidatorAgent cleanup complete")