#!/usr/bin/env python3.11
"""
Run Talk in background for orchestration task with periodic status checks.
"""

import subprocess
import json
import os
import sys
import time
import signal
from pathlib import Path
from datetime import datetime

def run_talk_background():
    """Run Talk in background and monitor until completion."""
    print("\n" + "="*60)
    print("RUNNING TALK IN BACKGROUND")
    print("="*60)
    
    task = "build an agentic orchestration system"
    output_dir = Path("/home/xx/code/tests/talk/orchestrator_comparison/talk_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run Talk command in background
    cmd = [
        "talk",
        "--task", task,
        "--dir", str(output_dir),
        "--skip-validation"
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print(f"Output directory: {output_dir}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Start the process
    log_file = output_dir / "talk_background.log"
    with open(log_file, "w") as f:
        process = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True
        )
    
    print(f"Process PID: {process.pid}")
    print(f"Log file: {log_file}")
    print("\nMonitoring progress (checking every minute)...")
    print("-" * 40)
    
    start_time = time.time()
    check_count = 0
    
    # Monitor the process
    while True:
        check_count += 1
        time.sleep(60)  # Wait 1 minute
        
        # Check if process is still running
        poll_result = process.poll()
        elapsed = time.time() - start_time
        elapsed_min = elapsed / 60
        
        print(f"[Check #{check_count}] {datetime.now().strftime('%H:%M:%S')} - Elapsed: {elapsed_min:.1f} min", end="")
        
        if poll_result is None:
            # Still running
            print(" - Status: RUNNING")
            
            # Check Talk session directory for progress
            talk_dir = Path("/home/xx/code/.talk")
            if talk_dir.exists():
                sessions = sorted([d for d in talk_dir.iterdir() if d.is_dir()], 
                                key=lambda x: x.stat().st_mtime, reverse=True)
                if sessions:
                    latest_session = sessions[0]
                    workspace = latest_session / "workspace"
                    if workspace.exists():
                        py_files = list(workspace.rglob("*.py"))
                        if py_files:
                            print(f"  Files created so far: {len(py_files)}")
                            for f in py_files[:3]:  # Show first 3 files
                                print(f"    - {f.name}")
                            if len(py_files) > 3:
                                print(f"    ... and {len(py_files) - 3} more")
        else:
            # Process finished
            print(f" - Status: COMPLETED (exit code: {poll_result})")
            break
    
    # Process completed
    print("-" * 40)
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total elapsed time: {elapsed_min:.1f} minutes")
    print(f"Exit code: {poll_result}")
    
    # Read the log file
    with open(log_file, "r") as f:
        log_content = f.read()
    
    # Save summary
    summary = {
        "task": task,
        "start_time": datetime.fromtimestamp(start_time).isoformat(),
        "end_time": datetime.now().isoformat(),
        "elapsed_minutes": elapsed_min,
        "exit_code": poll_result,
        "success": poll_result == 0,
        "output_dir": str(output_dir),
        "log_file": str(log_file)
    }
    
    summary_file = output_dir / "execution_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nExecution summary saved to: {summary_file}")
    
    # Show last 50 lines of output
    print("\n" + "="*60)
    print("LAST 50 LINES OF OUTPUT:")
    print("="*60)
    lines = log_content.splitlines()
    for line in lines[-50:]:
        print(line)
    
    return summary

if __name__ == "__main__":
    try:
        result = run_talk_background()
        sys.exit(0 if result["success"] else 1)
    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Monitoring stopped by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)