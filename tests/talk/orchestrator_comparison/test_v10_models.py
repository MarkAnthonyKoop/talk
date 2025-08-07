#!/usr/bin/env python3.11
"""
Test Talk v10 with different models for the orchestration task.
"""

import subprocess
import json
import time
import sys
from pathlib import Path
from datetime import datetime


def test_model(model: str, rate_limit: int = 30000) -> dict:
    """Test Talk v10 with a specific model."""
    print(f"\n{'='*60}")
    print(f"TESTING WITH {model.upper()}")
    print(f"{'='*60}")
    
    task = "build an agentic orchestration system"
    output_dir = Path(f"/home/xx/code/tests/talk/orchestrator_comparison/v10_{model.replace('-', '_')}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Adjust rate limits based on model
    if "gemini" in model.lower():
        rate_limit = 30000  # Conservative for free tier
    elif "sonnet" in model.lower():
        rate_limit = 40000  # Anthropic limits
    elif "opus" in model.lower():
        rate_limit = 40000  # Anthropic limits
    
    cmd = [
        "talk",
        "--task", task,
        "--dir", str(output_dir),
        "--model", model,
        "--rate-limit", str(rate_limit),
        "--max-tokens", "8000",
        "--timeout", "15"  # 15 minutes max
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print(f"Output directory: {output_dir}")
    print(f"Rate limit: {rate_limit} tokens/min")
    
    log_file = output_dir / "execution.log"
    start_time = time.time()
    
    try:
        # Run Talk v10
        with open(log_file, "w") as f:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=900  # 15 minute timeout
            )
        
        elapsed = time.time() - start_time
        
        # Save output
        with open(output_dir / "output.txt", "w") as f:
            f.write(f"Model: {model}\n")
            f.write(f"Elapsed: {elapsed:.1f} seconds\n")
            f.write(f"Return code: {result.returncode}\n")
            f.write("\n--- STDOUT ---\n")
            f.write(result.stdout)
            f.write("\n--- STDERR ---\n")
            f.write(result.stderr)
        
        # Count generated files
        py_files = list(output_dir.rglob("*.py"))
        
        # Check session directory for actual output
        talk_dir = Path("/home/xx/code/.talk")
        if talk_dir.exists():
            sessions = sorted([d for d in talk_dir.iterdir() if d.is_dir()], 
                            key=lambda x: x.stat().st_mtime, reverse=True)
            if sessions:
                workspace = sessions[0] / "workspace"
                if workspace.exists():
                    workspace_files = list(workspace.rglob("*.py"))
                    if workspace_files and not py_files:
                        py_files = workspace_files
        
        print(f"\nResults:")
        print(f"  - Elapsed: {elapsed:.1f} seconds")
        print(f"  - Success: {'Yes' if result.returncode == 0 else 'No'}")
        print(f"  - Files generated: {len(py_files)}")
        
        if py_files:
            print(f"  - Sample files:")
            for f in py_files[:3]:
                print(f"    • {f.name}")
        
        # Extract key metrics from output
        metrics = {
            "model": model,
            "success": result.returncode == 0,
            "elapsed_seconds": elapsed,
            "files_generated": len(py_files),
            "file_list": [f.name for f in py_files],
            "rate_limit_hits": result.stdout.count("Rate limit:"),
            "errors": result.stderr.count("ERROR") if result.stderr else 0
        }
        
        # Check for specific v10 features
        if "Context pruned" in result.stdout:
            metrics["context_pruning_used"] = True
        if "Auto-persisted" in result.stdout:
            metrics["auto_persistence_used"] = True
        if "Using fast path" in result.stdout:
            metrics["fast_path_used"] = True
        
        return metrics
        
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        print(f"\n[TIMEOUT] Execution exceeded 15 minutes")
        return {
            "model": model,
            "success": False,
            "elapsed_seconds": elapsed,
            "error": "Timeout after 15 minutes",
            "files_generated": 0
        }
    except Exception as e:
        print(f"\n[ERROR] {e}")
        return {
            "model": model,
            "success": False,
            "error": str(e),
            "files_generated": 0
        }


def analyze_file_content(file_path: Path) -> dict:
    """Analyze generated file for quality metrics."""
    try:
        with open(file_path) as f:
            content = f.read()
        
        return {
            "lines": len(content.splitlines()),
            "has_classes": "class " in content,
            "has_functions": "def " in content,
            "has_docstrings": '"""' in content or "'''" in content,
            "has_type_hints": "->" in content or ": " in content,
            "imports": content.count("import ")
        }
    except:
        return {}


def compare_models(results: list) -> None:
    """Compare results across models."""
    print(f"\n{'='*60}")
    print("MODEL COMPARISON")
    print(f"{'='*60}")
    
    # Create comparison table
    print("\n┌─────────────────┬──────────┬─────────┬───────┬────────────┐")
    print("│ Model           │ Success  │ Time(s) │ Files │ Rate Limits│")
    print("├─────────────────┼──────────┼─────────┼───────┼────────────┤")
    
    for r in results:
        model = r["model"][:15].ljust(15)
        success = "✓" if r["success"] else "✗"
        time_str = f"{r['elapsed_seconds']:.1f}" if "elapsed_seconds" in r else "N/A"
        files = str(r.get("files_generated", 0))
        rate_limits = str(r.get("rate_limit_hits", 0))
        
        print(f"│ {model} │ {success:^8} │ {time_str:^7} │ {files:^5} │ {rate_limits:^10} │")
    
    print("└─────────────────┴──────────┴─────────┴───────┴────────────┘")
    
    # Feature usage
    print("\nV10 Features Used:")
    for r in results:
        print(f"\n{r['model']}:")
        if r.get("context_pruning_used"):
            print("  ✓ Context pruning")
        if r.get("auto_persistence_used"):
            print("  ✓ Auto file persistence")
        if r.get("fast_path_used"):
            print("  ✓ Fast path optimization")
        if not any([r.get("context_pruning_used"), 
                   r.get("auto_persistence_used"),
                   r.get("fast_path_used")]):
            print("  (No v10 features detected)")
    
    # Save comparison report
    report = {
        "test_date": datetime.now().isoformat(),
        "task": "build an agentic orchestration system",
        "results": results,
        "summary": {
            "models_tested": len(results),
            "successful": sum(1 for r in results if r["success"]),
            "avg_time": sum(r.get("elapsed_seconds", 0) for r in results) / len(results) if results else 0,
            "total_files": sum(r.get("files_generated", 0) for r in results)
        }
    }
    
    report_file = Path("/home/xx/code/tests/talk/orchestrator_comparison/v10_model_comparison.json")
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\nFull report saved to: {report_file}")


def main():
    """Run tests with different models."""
    print("="*80)
    print(" TALK V10 MODEL COMPARISON TEST ")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Models to test (in order)
    models = [
        "gemini-2.0-flash",      # Free, fast
        "claude-3-5-sonnet-20241022",  # Paid, balanced
        # "claude-3-opus-20240229"  # Paid, powerful (optional)
    ]
    
    results = []
    
    for model in models:
        try:
            result = test_model(model)
            results.append(result)
            
            # Wait between models to avoid rate limits
            if model != models[-1]:
                print("\nWaiting 30 seconds before next model...")
                time.sleep(30)
                
        except Exception as e:
            print(f"\nFailed to test {model}: {e}")
            results.append({
                "model": model,
                "success": False,
                "error": str(e),
                "files_generated": 0
            })
    
    # Compare results
    compare_models(results)
    
    # Determine outcome
    print("\n" + "="*80)
    print(" ANALYSIS ")
    print("="*80)
    
    successful = [r for r in results if r["success"]]
    if not successful:
        print("\n❌ All models failed - Likely an architectural issue in Talk")
    elif len(successful) == len(models):
        print("\n✅ All models succeeded - V10 improvements working!")
    else:
        failed_models = [r["model"] for r in results if not r["success"]]
        print(f"\n⚠️ Mixed results - Failed models: {', '.join(failed_models)}")
        print("This suggests model-specific limitations or rate limit issues")
    
    print("\n" + "="*80)
    print(" TEST COMPLETE ")
    print("="*80)
    
    return 0 if any(r["success"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())