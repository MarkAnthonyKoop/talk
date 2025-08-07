#!/usr/bin/env python3.11
"""
Monitor Talk execution progress until completion.
"""

import subprocess
import time
import sys
from pathlib import Path
from datetime import datetime

def monitor_talk(pid):
    """Monitor Talk process until completion."""
    start_time = time.time()
    check_count = 0
    
    print(f"\nMonitoring Talk process (PID: {pid})")
    print("=" * 60)
    
    while True:
        check_count += 1
        
        # Check if process is running
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "pid,etime"],
            capture_output=True,
            text=True
        )
        
        elapsed = time.time() - start_time
        elapsed_min = elapsed / 60
        
        if str(pid) not in result.stdout:
            # Process completed
            print(f"\n[Check #{check_count}] Process COMPLETED after {elapsed_min:.1f} minutes")
            break
        
        # Get elapsed time from ps output
        lines = result.stdout.strip().split('\n')
        if len(lines) > 1:
            ps_elapsed = lines[1].split()[-1]
        else:
            ps_elapsed = "unknown"
        
        print(f"[Check #{check_count}] {datetime.now().strftime('%H:%M:%S')} - Elapsed: {ps_elapsed} - Status: RUNNING")
        
        # Check log file for recent activity
        log_file = Path("tests/talk/orchestrator_comparison/talk_direct.log")
        if log_file.exists():
            result = subprocess.run(
                ["tail", "-5", str(log_file)],
                capture_output=True,
                text=True
            )
            
            # Look for interesting patterns
            if "generate_code" in result.stdout:
                print("  → Currently generating code")
            elif "apply_files" in result.stdout:
                print("  → Applying files")
            elif "run_tests" in result.stdout:
                print("  → Running tests")
            elif "research" in result.stdout:
                print("  → Researching")
            elif "plan_next" in result.stdout:
                print("  → Planning next steps")
            elif "complete" in result.stdout.lower():
                print("  → Task appears complete")
            
            # Check for code generation
            if "Saved" in result.stdout and "files to scratch" in result.stdout:
                import re
                match = re.search(r'Saved (\d+) code files', result.stdout)
                if match:
                    print(f"  → Generated {match.group(1)} code files")
        
        # Check Talk workspace for files
        talk_dir = Path("/home/xx/code/.talk")
        if talk_dir.exists():
            sessions = sorted([d for d in talk_dir.iterdir() if d.is_dir()], 
                            key=lambda x: x.stat().st_mtime, reverse=True)
            if sessions:
                workspace = sessions[0] / "workspace"
                if workspace.exists():
                    py_files = list(workspace.rglob("*.py"))
                    if py_files:
                        print(f"  → Files in workspace: {len(py_files)}")
        
        # Wait for next check
        time.sleep(60)
    
    # Process completed - show final status
    print("\n" + "=" * 60)
    print("EXECUTION COMPLETE")
    print("=" * 60)
    
    # Show last 30 lines of log
    log_file = Path("tests/talk/orchestrator_comparison/talk_direct.log")
    if log_file.exists():
        print("\nLast 30 lines of output:")
        print("-" * 40)
        result = subprocess.run(
            ["tail", "-30", str(log_file)],
            capture_output=True,
            text=True
        )
        print(result.stdout)
    
    return elapsed_min

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: monitor_loop.py <pid>")
        sys.exit(1)
    
    pid = int(sys.argv[1])
    elapsed = monitor_talk(pid)
    print(f"\nTotal execution time: {elapsed:.1f} minutes")