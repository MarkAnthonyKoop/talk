#!/usr/bin/env python3.11
"""
Test comparing Talk's output vs Claude Code's output for building an agentic orchestration system.
"""

import subprocess
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, '/home/xx/code')


def run_talk_orchestration():
    """Run Talk with the orchestration task."""
    print("\n" + "="*60)
    print("RUNNING TALK WITH ORCHESTRATION TASK")
    print("="*60)
    
    task = "build an agentic orchestration system"
    
    # Create output directory for Talk results
    output_dir = Path("/home/xx/code/tests/talk/orchestrator_comparison/talk_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run Talk command
    cmd = [
        "talk",
        "--task", task,
        "--dir", str(output_dir),
        "--skip-validation",  # Skip validation for speed
        "--timeout", "10"  # 10 minute timeout
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print(f"Output directory: {output_dir}")
    
    start_time = time.time()
    
    try:
        # Run Talk and capture output
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )
        
        elapsed = time.time() - start_time
        
        # Save output
        with open(output_dir / "talk_output.log", "w") as f:
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write(f"Elapsed time: {elapsed:.2f} seconds\n")
            f.write(f"Return code: {result.returncode}\n")
            f.write("\n--- STDOUT ---\n")
            f.write(result.stdout)
            f.write("\n--- STDERR ---\n")
            f.write(result.stderr)
        
        print(f"\nCompleted in {elapsed:.2f} seconds")
        print(f"Return code: {result.returncode}")
        
        return {
            "success": result.returncode == 0,
            "elapsed_time": elapsed,
            "output_dir": str(output_dir),
            "stdout": result.stdout,
            "stderr": result.stderr
        }
        
    except subprocess.TimeoutExpired:
        print("\n[ERROR] Talk execution timed out after 10 minutes")
        return {
            "success": False,
            "error": "Timeout after 10 minutes",
            "output_dir": str(output_dir)
        }
    except Exception as e:
        print(f"\n[ERROR] Failed to run Talk: {e}")
        return {
            "success": False,
            "error": str(e),
            "output_dir": str(output_dir)
        }


def analyze_claude_code_results():
    """Analyze Claude Code's orchestrator implementation."""
    print("\n" + "="*60)
    print("ANALYZING CLAUDE CODE RESULTS")
    print("="*60)
    
    claude_dir = Path("/home/xx/code/tests/talk/claude_code_results/orchestrator")
    
    analysis = {
        "files_created": [],
        "total_lines": 0,
        "key_features": [],
        "design_patterns": [],
        "complexity_indicators": []
    }
    
    # Count files and lines
    for file_path in claude_dir.glob("*.py"):
        if file_path.name != "__init__.py":
            with open(file_path) as f:
                lines = len(f.readlines())
            analysis["files_created"].append({
                "name": file_path.name,
                "lines": lines
            })
            analysis["total_lines"] += lines
    
    # Analyze core.py for key features
    core_file = claude_dir / "core.py"
    if core_file.exists():
        with open(core_file) as f:
            content = f.read()
            
        # Key features
        if "ThreadPoolExecutor" in content:
            analysis["key_features"].append("Thread-based parallelism")
        if "ProcessPoolExecutor" in content:
            analysis["key_features"].append("Process-based parallelism")
        if "async" in content or "await" in content:
            analysis["key_features"].append("Async/await support")
        if "checkpoint" in content.lower():
            analysis["key_features"].append("Checkpointing and recovery")
        if "health_check" in content:
            analysis["key_features"].append("Health monitoring")
        if "load_balanc" in content.lower():
            analysis["key_features"].append("Load balancing")
        if "retry" in content.lower():
            analysis["key_features"].append("Retry policies")
        if "MessageBus" in content:
            analysis["key_features"].append("Message bus communication")
        
        # Design patterns
        if "Registry" in content:
            analysis["design_patterns"].append("Registry pattern")
        if "Dispatcher" in content:
            analysis["design_patterns"].append("Dispatcher pattern")
        if "Monitor" in content:
            analysis["design_patterns"].append("Observer/Monitor pattern")
        if "@dataclass" in content:
            analysis["design_patterns"].append("Dataclass configuration")
        if "Enum" in content:
            analysis["design_patterns"].append("Enum for type safety")
        
        # Complexity indicators
        import_count = content.count("import ")
        class_count = content.count("class ")
        method_count = content.count("def ")
        
        analysis["complexity_indicators"] = {
            "imports": import_count,
            "classes": class_count,
            "methods": method_count,
            "lines_in_core": len(content.splitlines())
        }
    
    print(f"\nFiles created: {len(analysis['files_created'])}")
    for file_info in analysis["files_created"]:
        print(f"  - {file_info['name']}: {file_info['lines']} lines")
    print(f"\nTotal lines of code: {analysis['total_lines']}")
    
    print(f"\nKey features ({len(analysis['key_features'])}):")
    for feature in analysis["key_features"]:
        print(f"  - {feature}")
    
    print(f"\nDesign patterns ({len(analysis['design_patterns'])}):")
    for pattern in analysis["design_patterns"]:
        print(f"  - {pattern}")
    
    print(f"\nComplexity indicators:")
    for key, value in analysis["complexity_indicators"].items():
        print(f"  - {key}: {value}")
    
    return analysis


def analyze_talk_results(output_dir):
    """Analyze Talk's orchestrator implementation."""
    print("\n" + "="*60)
    print("ANALYZING TALK RESULTS")
    print("="*60)
    
    output_path = Path(output_dir)
    
    analysis = {
        "files_created": [],
        "total_lines": 0,
        "key_features": [],
        "design_patterns": [],
        "session_info": {}
    }
    
    # Find the Talk session directory
    talk_dir = Path("/home/xx/code/.talk")
    if talk_dir.exists():
        # Get most recent session
        sessions = sorted([d for d in talk_dir.iterdir() if d.is_dir()], 
                         key=lambda x: x.stat().st_mtime, reverse=True)
        if sessions:
            latest_session = sessions[0]
            print(f"Latest session: {latest_session.name}")
            
            # Check workspace for generated files
            workspace = latest_session / "workspace"
            if workspace.exists():
                for file_path in workspace.rglob("*.py"):
                    with open(file_path) as f:
                        lines = len(f.readlines())
                    rel_path = file_path.relative_to(workspace)
                    analysis["files_created"].append({
                        "name": str(rel_path),
                        "lines": lines
                    })
                    analysis["total_lines"] += lines
                    
                    # Analyze content
                    with open(file_path) as f:
                        content = f.read()
                    
                    # Check for key features
                    if "Agent" in content and "class" in content:
                        analysis["key_features"].append("Agent-based architecture")
                    if "blackboard" in content.lower():
                        analysis["key_features"].append("Blackboard pattern")
                    if "plan" in content.lower() and "step" in content.lower():
                        analysis["key_features"].append("Step-based planning")
            
            # Load session info
            session_info_file = latest_session / "session_info.json"
            if session_info_file.exists():
                with open(session_info_file) as f:
                    analysis["session_info"] = json.load(f)
    
    # Check files in the provided output directory
    for file_path in output_path.rglob("*.py"):
        if file_path.name not in [f["name"] for f in analysis["files_created"]]:
            with open(file_path) as f:
                lines = len(f.readlines())
            analysis["files_created"].append({
                "name": file_path.name,
                "lines": lines
            })
            analysis["total_lines"] += lines
    
    print(f"\nFiles created: {len(analysis['files_created'])}")
    for file_info in analysis["files_created"]:
        print(f"  - {file_info['name']}: {file_info['lines']} lines")
    print(f"\nTotal lines of code: {analysis['total_lines']}")
    
    if analysis["key_features"]:
        print(f"\nKey features detected:")
        for feature in set(analysis["key_features"]):
            print(f"  - {feature}")
    
    if analysis["session_info"]:
        print(f"\nSession info:")
        print(f"  - Task: {analysis['session_info'].get('task', 'N/A')}")
        print(f"  - Model: {analysis['session_info'].get('model', 'N/A')}")
        print(f"  - Version: {analysis['session_info'].get('version', 'N/A')}")
    
    return analysis


def compare_results(claude_analysis, talk_analysis):
    """Compare Claude Code and Talk results."""
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    
    comparison = {
        "metrics": {
            "claude_code": {
                "files": len(claude_analysis["files_created"]),
                "lines": claude_analysis["total_lines"],
                "features": len(claude_analysis["key_features"]),
                "patterns": len(claude_analysis["design_patterns"])
            },
            "talk": {
                "files": len(talk_analysis["files_created"]),
                "lines": talk_analysis["total_lines"],
                "features": len(talk_analysis["key_features"]),
                "patterns": len(talk_analysis["design_patterns"])
            }
        },
        "observations": []
    }
    
    # Print comparison table
    print("\n┌─────────────────┬──────────────┬──────────────┐")
    print("│ Metric          │ Claude Code  │ Talk         │")
    print("├─────────────────┼──────────────┼──────────────┤")
    print(f"│ Files Created   │ {comparison['metrics']['claude_code']['files']:^12} │ {comparison['metrics']['talk']['files']:^12} │")
    print(f"│ Lines of Code   │ {comparison['metrics']['claude_code']['lines']:^12} │ {comparison['metrics']['talk']['lines']:^12} │")
    print(f"│ Key Features    │ {comparison['metrics']['claude_code']['features']:^12} │ {comparison['metrics']['talk']['features']:^12} │")
    print(f"│ Design Patterns │ {comparison['metrics']['claude_code']['patterns']:^12} │ {comparison['metrics']['talk']['patterns']:^12} │")
    print("└─────────────────┴──────────────┴──────────────┘")
    
    # Observations
    if claude_analysis["total_lines"] > talk_analysis["total_lines"] * 2:
        comparison["observations"].append("Claude Code produced significantly more code")
    elif talk_analysis["total_lines"] > claude_analysis["total_lines"] * 2:
        comparison["observations"].append("Talk produced significantly more code")
    else:
        comparison["observations"].append("Similar code volume produced")
    
    # Feature comparison
    claude_features = set(claude_analysis["key_features"])
    talk_features = set(talk_analysis["key_features"])
    
    if claude_features and not talk_features:
        comparison["observations"].append("Claude Code implemented more advanced features")
    elif talk_features and not claude_features:
        comparison["observations"].append("Talk implemented unique architectural features")
    
    # Unique features
    claude_unique = claude_features - talk_features if talk_features else claude_features
    talk_unique = talk_features - claude_features if claude_features else talk_features
    
    if claude_unique:
        print(f"\nClaude Code unique features:")
        for feature in claude_unique:
            print(f"  - {feature}")
    
    if talk_unique:
        print(f"\nTalk unique features:")
        for feature in talk_unique:
            print(f"  - {feature}")
    
    print(f"\nKey Observations:")
    for obs in comparison["observations"]:
        print(f"  • {obs}")
    
    # Save comparison report
    report_file = Path("/home/xx/code/tests/talk/orchestrator_comparison/comparison_report.json")
    report_file.parent.mkdir(parents=True, exist_ok=True)
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "task": "build an agentic orchestration system",
        "claude_code_analysis": claude_analysis,
        "talk_analysis": talk_analysis,
        "comparison": comparison
    }
    
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\nFull report saved to: {report_file}")
    
    return comparison


def main():
    """Main test execution."""
    print("\n" + "="*80)
    print(" ORCHESTRATOR COMPARISON TEST: TALK vs CLAUDE CODE ")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Analyze Claude Code results (already generated)
    claude_analysis = analyze_claude_code_results()
    
    # Run Talk with the same task
    talk_result = run_talk_orchestration()
    
    if talk_result["success"]:
        # Analyze Talk results
        talk_analysis = analyze_talk_results(talk_result["output_dir"])
    else:
        print("\n[WARNING] Talk execution failed or timed out")
        talk_analysis = {
            "files_created": [],
            "total_lines": 0,
            "key_features": [],
            "design_patterns": [],
            "error": talk_result.get("error", "Unknown error")
        }
    
    # Compare results
    comparison = compare_results(claude_analysis, talk_analysis)
    
    print("\n" + "="*80)
    print(" TEST COMPLETE ")
    print("="*80)
    
    return 0 if talk_result.get("success", False) else 1


if __name__ == "__main__":
    sys.exit(main())