#!/usr/bin/env python3.11
"""
Claude's Last Exam - The Ultimate Test of Talk vs Claude Code

This test compares the output of Talk and Claude Code on building an agentic orchestration system.
The test iteratively refines Talk until it clearly outperforms Claude Code.

Usage:
    python claudes_last_exam.py
"""

import subprocess
import json
import time
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

class ClaudesLastExam:
    """
    Head-to-head comparison test between Talk and Claude Code.
    """
    
    def __init__(self, prompt: str = "build an agentic orchestration system"):
        self.prompt = prompt
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(f"tests/data/output/claudes_last_exam_{self.timestamp}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.talk_output_file = self.output_dir / "talk_output.txt"
        self.claude_output_file = self.output_dir / "claude_output.txt"
        self.report_file = self.output_dir / "evaluation_report.md"
        
    def run_talk(self) -> Tuple[str, float]:
        """Run Talk with the prompt and capture output."""
        print(f"\n{'='*60}")
        print("🤖 Running Talk...")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        # Run Talk test mode for quick comparison
        cmd = [
            "/usr/bin/python3.11",
            "/home/xx/code/talk/talk_test_mode.py",
            self.prompt
        ]
        
        try:
            # Talk asks for confirmation, so we pipe "y\n" to it
            result = subprocess.run(
                cmd,
                input="y\n",
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
                cwd="/home/xx/code"
            )
            
            output = result.stdout + "\n\n--- STDERR ---\n" + result.stderr
            elapsed = time.time() - start_time
            
            # Save output
            with open(self.talk_output_file, 'w') as f:
                f.write(output)
            
            print(f"✓ Talk completed in {elapsed:.2f} seconds")
            print(f"  Output saved to: {self.talk_output_file}")
            
            return output, elapsed
            
        except subprocess.TimeoutExpired:
            print("✗ Talk timed out after 5 minutes")
            return "TIMEOUT", 300.0
        except Exception as e:
            print(f"✗ Talk failed: {e}")
            return f"ERROR: {e}", 0.0
    
    def run_claude_code(self) -> Tuple[str, float]:
        """Run Claude Code with the prompt and capture output."""
        print(f"\n{'='*60}")
        print("🤖 Running Claude Code...")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        # Run Claude Code in non-interactive mode
        cmd = [
            "claude",
            "-p",
            self.prompt
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
                cwd="/home/xx/code"
            )
            
            output = result.stdout + "\n\n--- STDERR ---\n" + result.stderr
            elapsed = time.time() - start_time
            
            # Save output
            with open(self.claude_output_file, 'w') as f:
                f.write(output)
            
            print(f"✓ Claude Code completed in {elapsed:.2f} seconds")
            print(f"  Output saved to: {self.claude_output_file}")
            
            return output, elapsed
            
        except subprocess.TimeoutExpired:
            print("✗ Claude Code timed out after 5 minutes")
            return "TIMEOUT", 300.0
        except Exception as e:
            print(f"✗ Claude Code failed: {e}")
            return f"ERROR: {e}", 0.0
    
    def analyze_outputs(self, talk_output: str, claude_output: str, 
                       talk_time: float, claude_time: float) -> Dict[str, Any]:
        """Analyze and compare the outputs."""
        
        analysis = {
            "timestamp": self.timestamp,
            "prompt": self.prompt,
            "execution_times": {
                "talk": talk_time,
                "claude_code": claude_time
            },
            "output_sizes": {
                "talk": len(talk_output),
                "claude_code": len(claude_output)
            },
            "metrics": {}
        }
        
        # Count lines of code generated (rough estimate)
        talk_code_lines = len([l for l in talk_output.split('\n') 
                              if l.strip() and not l.strip().startswith('#')])
        claude_code_lines = len([l for l in claude_output.split('\n') 
                               if l.strip() and not l.strip().startswith('#')])
        
        analysis["metrics"]["code_lines"] = {
            "talk": talk_code_lines,
            "claude_code": claude_code_lines
        }
        
        # Check for key components of an agentic orchestration system
        key_components = [
            "agent", "orchestrat", "task", "queue", "message", "plan",
            "execut", "monitor", "schedul", "workflow", "pipeline",
            "async", "parallel", "distribut", "scale", "fault"
        ]
        
        talk_components = sum(1 for comp in key_components 
                             if comp.lower() in talk_output.lower())
        claude_components = sum(1 for comp in key_components 
                               if comp.lower() in claude_output.lower())
        
        analysis["metrics"]["key_components"] = {
            "talk": talk_components,
            "claude_code": claude_components
        }
        
        # Check for actual Python files created
        talk_files = talk_output.count("```python")
        claude_files = claude_output.count("```python")
        
        analysis["metrics"]["python_blocks"] = {
            "talk": talk_files,
            "claude_code": claude_files
        }
        
        return analysis
    
    def generate_report(self, analysis: Dict[str, Any]) -> str:
        """Generate evaluation report."""
        
        report = f"""# Claude's Last Exam - Evaluation Report

**Date:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Prompt:** "{self.prompt}"

## Executive Summary

This report compares Talk and Claude Code on building an agentic orchestration system.

## Execution Metrics

| Metric | Talk | Claude Code | Winner |
|--------|------|-------------|--------|
| Execution Time | {analysis['execution_times']['talk']:.2f}s | {analysis['execution_times']['claude_code']:.2f}s | {'Talk' if analysis['execution_times']['talk'] < analysis['execution_times']['claude_code'] else 'Claude Code'} |
| Output Size | {analysis['output_sizes']['talk']:,} chars | {analysis['output_sizes']['claude_code']:,} chars | {'Talk' if analysis['output_sizes']['talk'] > analysis['output_sizes']['claude_code'] else 'Claude Code'} |
| Code Lines | {analysis['metrics']['code_lines']['talk']:,} | {analysis['metrics']['code_lines']['claude_code']:,} | {'Talk' if analysis['metrics']['code_lines']['talk'] > analysis['metrics']['code_lines']['claude_code'] else 'Claude Code'} |
| Key Components | {analysis['metrics']['key_components']['talk']}/{len(['agent', 'orchestrat', 'task', 'queue', 'message', 'plan', 'execut', 'monitor', 'schedul', 'workflow', 'pipeline', 'async', 'parallel', 'distribut', 'scale', 'fault'])} | {analysis['metrics']['key_components']['claude_code']}/{len(['agent', 'orchestrat', 'task', 'queue', 'message', 'plan', 'execut', 'monitor', 'schedul', 'workflow', 'pipeline', 'async', 'parallel', 'distribut', 'scale', 'fault'])} | {'Talk' if analysis['metrics']['key_components']['talk'] > analysis['metrics']['key_components']['claude_code'] else 'Claude Code'} |
| Python Blocks | {analysis['metrics']['python_blocks']['talk']} | {analysis['metrics']['python_blocks']['claude_code']} | {'Talk' if analysis['metrics']['python_blocks']['talk'] > analysis['metrics']['python_blocks']['claude_code'] else 'Claude Code'} |

## Qualitative Analysis

### Talk Output Characteristics
- Focus: {'Large-scale code generation' if analysis['metrics']['code_lines']['talk'] > 1000 else 'Targeted implementation'}
- Approach: {'Multi-agent orchestration' if 'v15' in str(analysis) or 'v16' in str(analysis) else 'Single-agent generation'}
- Completeness: {'Full system' if analysis['metrics']['key_components']['talk'] >= 12 else 'Partial implementation'}

### Claude Code Output Characteristics  
- Focus: {'Comprehensive implementation' if analysis['metrics']['code_lines']['claude_code'] > 500 else 'Conceptual design'}
- Approach: {'Direct implementation' if analysis['metrics']['python_blocks']['claude_code'] > 0 else 'Planning/design'}
- Completeness: {'Full system' if analysis['metrics']['key_components']['claude_code'] >= 12 else 'Partial implementation'}

## Winner Determination

"""
        
        # Count wins
        talk_wins = 0
        claude_wins = 0
        
        if analysis['output_sizes']['talk'] > analysis['output_sizes']['claude_code']:
            talk_wins += 1
        else:
            claude_wins += 1
            
        if analysis['metrics']['code_lines']['talk'] > analysis['metrics']['code_lines']['claude_code']:
            talk_wins += 2  # Double weight for code generation
        else:
            claude_wins += 2
            
        if analysis['metrics']['key_components']['talk'] > analysis['metrics']['key_components']['claude_code']:
            talk_wins += 1
        else:
            claude_wins += 1
            
        if analysis['metrics']['python_blocks']['talk'] > analysis['metrics']['python_blocks']['claude_code']:
            talk_wins += 1
        else:
            claude_wins += 1
        
        if talk_wins > claude_wins:
            winner = "🏆 **TALK WINS**"
            report += f"{winner} ({talk_wins}-{claude_wins})\n\n"
            report += "Talk demonstrated superior code generation capabilities.\n"
        elif claude_wins > talk_wins:
            winner = "🏆 **CLAUDE CODE WINS**"
            report += f"{winner} ({claude_wins}-{talk_wins})\n\n"
            report += "Claude Code provided a more comprehensive solution.\n"
        else:
            winner = "🤝 **TIE**"
            report += f"{winner} ({talk_wins}-{claude_wins})\n\n"
            report += "Both systems performed comparably.\n"
        
        # Add recommendations
        report += "\n## Recommendations for Talk Improvements\n\n"
        
        if analysis['metrics']['code_lines']['talk'] < analysis['metrics']['code_lines']['claude_code']:
            report += "1. **Increase code generation volume** - Talk should generate more actual code\n"
        
        if analysis['metrics']['key_components']['talk'] < analysis['metrics']['key_components']['claude_code']:
            report += "2. **Add missing components** - Ensure all key orchestration components are included\n"
        
        if analysis['execution_times']['talk'] > analysis['execution_times']['claude_code']:
            report += "3. **Optimize execution time** - Reduce generation latency\n"
        
        if analysis['metrics']['python_blocks']['talk'] < analysis['metrics']['python_blocks']['claude_code']:
            report += "4. **Generate more complete files** - Create more standalone Python modules\n"
        
        if talk_wins <= claude_wins:
            report += "\n### Critical Improvements Needed:\n"
            report += "- Talk must leverage its v15/v16/v17 architecture to generate massive codebases\n"
            report += "- Enable parallel agent execution for faster generation\n"
            report += "- Ensure each agent generates complete, runnable code\n"
            report += "- Target 100,000+ lines of code to showcase Talk's true capabilities\n"
        
        report += f"\n---\n*Report generated at {datetime.now()}*\n"
        
        return report
    
    def run_exam(self) -> Dict[str, Any]:
        """Run the complete exam."""
        print("\n" + "="*60)
        print("🎓 CLAUDE'S LAST EXAM - STARTING")
        print("="*60)
        print(f"Prompt: '{self.prompt}'")
        print(f"Output directory: {self.output_dir}")
        
        # Run both systems
        talk_output, talk_time = self.run_talk()
        claude_output, claude_time = self.run_claude_code()
        
        # Analyze outputs
        print(f"\n{'='*60}")
        print("📊 Analyzing outputs...")
        print(f"{'='*60}")
        
        analysis = self.analyze_outputs(talk_output, claude_output, talk_time, claude_time)
        
        # Generate report
        report = self.generate_report(analysis)
        
        # Save report
        with open(self.report_file, 'w') as f:
            f.write(report)
        
        print(f"\n✓ Report saved to: {self.report_file}")
        print("\n" + report)
        
        return analysis


def main():
    """Run Claude's Last Exam."""
    exam = ClaudesLastExam()
    analysis = exam.run_exam()
    
    # Return success/failure based on winner
    talk_score = (
        (1 if analysis['output_sizes']['talk'] > analysis['output_sizes']['claude_code'] else 0) +
        (2 if analysis['metrics']['code_lines']['talk'] > analysis['metrics']['code_lines']['claude_code'] else 0) +
        (1 if analysis['metrics']['key_components']['talk'] > analysis['metrics']['key_components']['claude_code'] else 0) +
        (1 if analysis['metrics']['python_blocks']['talk'] > analysis['metrics']['python_blocks']['claude_code'] else 0)
    )
    
    if talk_score >= 3:
        print("\n🎉 TALK PASSES THE EXAM!")
        return 0
    else:
        print("\n📚 TALK NEEDS IMPROVEMENT - See report for recommendations")
        return 1


if __name__ == "__main__":
    sys.exit(main())