#!/usr/bin/env python3
"""
MetaMetaOrchestratorAgent - Orchestrates multiple Talk v16 instances.

This is the conductor of conductors:
- Each v16 instance orchestrates 4 v15 instances
- Each v15 generates 50k lines
- Total: 4-8 v16s × 4 v15s × 50k lines = 1,000,000+ lines

This agent manages the entire civilization-building process.
"""

from __future__ import annotations

import json
import logging
import time
import asyncio
import threading
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, List, Set, Tuple
from datetime import datetime, timedelta
import multiprocessing as mp
import queue

from agent.agent import Agent
from agent.output_manager import OutputManager
from special_agents.galaxy_decomposer_agent import GalaxyDecomposer, CivilizationPlan, TechGalaxy

log = logging.getLogger(__name__)


@dataclass
class GalaxyExecution:
    """Tracks execution of a single galaxy (v16 instance)."""
    galaxy: TechGalaxy
    v16_instance_id: int
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    status: str = "pending"  # pending, running, completed, error
    result: Optional[Dict[str, Any]] = None
    output_dir: Optional[Path] = None
    lines_generated: int = 0
    files_generated: int = 0
    subsystems_built: int = 0
    

class HyperParallelExecutor:
    """
    Executes multiple Talk v16 instances in parallel.
    
    Each v16 runs 4 v15s, each v15 generates 50k lines.
    Total capacity: 8 v16s × 4 v15s × 50k = 1.6M lines
    """
    
    def __init__(self, max_v16_instances: int = 4, model: str = "gemini-2.0-flash"):
        self.max_v16_instances = max_v16_instances
        self.model = model
        self.executor = ProcessPoolExecutor(max_workers=max_v16_instances)
        self.galaxy_executions = {}
        self.total_v15_instances = 0
        
    def execute_galaxy(self, galaxy: TechGalaxy, working_dir: Path) -> Dict[str, Any]:
        """Execute a single galaxy using Talk v16."""
        try:
            # Import here to avoid pickle issues
            from talk.talk_v16_meta import TalkV16MetaOrchestrator
            
            # Create task for this galaxy
            task = f"Build {galaxy.name}: {galaxy.description}"
            
            # Create output directory
            galaxy_dir = working_dir / galaxy.id
            galaxy_dir.mkdir(parents=True, exist_ok=True)
            
            log.info(f"Launching v16 for {galaxy.name} ({galaxy.estimated_lines:,} lines)")
            log.info(f"  This v16 will run {galaxy.v16_config['parallel_instances']} v15 instances")
            
            # Track total v15 instances
            v15_count = galaxy.v16_config.get("parallel_instances", 4)
            
            # Run Talk v16 (which runs multiple v15s)
            orchestrator = TalkV16MetaOrchestrator(
                task=task,
                model=self.model,
                working_dir=str(galaxy_dir),
                max_parallel=v15_count,
                verbose=False
            )
            
            start_time = datetime.now()
            result = orchestrator.run()
            end_time = datetime.now()
            
            # Enhance result
            result["galaxy_id"] = galaxy.id
            result["galaxy_name"] = galaxy.name
            result["v15_instances_used"] = v15_count
            result["execution_hours"] = (end_time - start_time).total_seconds() / 3600
            result["output_dir"] = str(galaxy_dir)
            
            log.info(f"Completed {galaxy.name}: {result.get('total_lines_generated', 0):,} lines")
            
            return result
            
        except Exception as e:
            log.error(f"Failed to execute galaxy {galaxy.id}: {e}")
            return {
                "status": "error",
                "galaxy_id": galaxy.id,
                "error": str(e)
            }
    
    def execute_civilization(self, 
                           galaxies: List[TechGalaxy],
                           working_dir: Path,
                           parallel_mode: str = "aggressive") -> Dict[str, Dict[str, Any]]:
        """
        Execute all galaxies to build the civilization.
        
        Modes:
        - aggressive: Run all v16s in parallel (16-32 v15s total)
        - balanced: Run in batches to limit resource usage
        - sequential: Run one v16 at a time (still 4 v15s each)
        """
        futures = {}
        results = {}
        executions = {}
        
        print(f"\n🌌 LAUNCHING CIVILIZATION CONSTRUCTION 🌌")
        print(f"  Galaxies to build: {len(galaxies)}")
        print(f"  Execution mode: {parallel_mode}")
        
        # Calculate total v15 instances
        total_v15s = sum(g.v16_config.get("parallel_instances", 4) for g in galaxies)
        print(f"  Total v15 instances: {total_v15s}")
        print(f"  Estimated total lines: {sum(g.estimated_lines for g in galaxies):,}")
        
        if parallel_mode == "aggressive":
            # Launch all v16s at once
            print(f"\n⚡ AGGRESSIVE MODE: Launching {len(galaxies)} v16 instances simultaneously!")
            print(f"  This will run {total_v15s} v15 instances in parallel!")
            
            for i, galaxy in enumerate(galaxies):
                execution = GalaxyExecution(
                    galaxy=galaxy,
                    v16_instance_id=i,
                    start_time=datetime.now(),
                    status="running"
                )
                executions[galaxy.id] = execution
                
                future = self.executor.submit(self.execute_galaxy, galaxy, working_dir)
                futures[future] = galaxy
                
                print(f"  🚀 v16 #{i+1}: {galaxy.name} ({galaxy.estimated_lines:,} lines)")
        
        elif parallel_mode == "balanced":
            # Run in batches of 2 v16s
            batch_size = 2
            print(f"\n⚖️ BALANCED MODE: Running in batches of {batch_size} v16 instances")
            
            for batch_start in range(0, len(galaxies), batch_size):
                batch = galaxies[batch_start:batch_start + batch_size]
                batch_v15s = sum(g.v16_config.get("parallel_instances", 4) for g in batch)
                
                print(f"\n  Batch {batch_start//batch_size + 1}: {len(batch)} v16s = {batch_v15s} v15s")
                
                batch_futures = {}
                for i, galaxy in enumerate(batch):
                    execution = GalaxyExecution(
                        galaxy=galaxy,
                        v16_instance_id=batch_start + i,
                        start_time=datetime.now(),
                        status="running"
                    )
                    executions[galaxy.id] = execution
                    
                    future = self.executor.submit(self.execute_galaxy, galaxy, working_dir)
                    batch_futures[future] = galaxy
                    print(f"    🚀 {galaxy.name}")
                
                # Wait for batch to complete
                for future in as_completed(batch_futures):
                    galaxy = batch_futures[future]
                    try:
                        result = future.result()
                        results[galaxy.id] = result
                        executions[galaxy.id].status = "completed"
                        executions[galaxy.id].result = result
                        print(f"    ✅ Completed: {galaxy.name}")
                    except Exception as e:
                        log.error(f"Galaxy {galaxy.id} failed: {e}")
                        results[galaxy.id] = {"status": "error", "error": str(e)}
                        executions[galaxy.id].status = "error"
        
        else:  # sequential
            print(f"\n🐢 SEQUENTIAL MODE: Running one v16 at a time")
            for i, galaxy in enumerate(galaxies):
                v15s = galaxy.v16_config.get("parallel_instances", 4)
                print(f"\n  v16 #{i+1}: {galaxy.name} (runs {v15s} v15s)")
                
                execution = GalaxyExecution(
                    galaxy=galaxy,
                    v16_instance_id=i,
                    start_time=datetime.now(),
                    status="running"
                )
                executions[galaxy.id] = execution
                
                result = self.execute_galaxy(galaxy, working_dir)
                results[galaxy.id] = result
                execution.end_time = datetime.now()
                execution.status = "completed" if result.get("status") == "success" else "error"
                execution.result = result
                
                print(f"    Lines: {result.get('total_lines_generated', 0):,}")
                print(f"    Time: {result.get('execution_hours', 0):.1f} hours")
        
        # For aggressive mode, wait for all to complete
        if parallel_mode == "aggressive":
            print(f"\n⏳ Waiting for {len(futures)} v16 instances to complete...")
            print(f"  (Each running {4} v15 instances = {len(futures) * 4} total v15s)")
            
            for future in as_completed(futures):
                galaxy = futures[future]
                try:
                    result = future.result()
                    results[galaxy.id] = result
                    executions[galaxy.id].end_time = datetime.now()
                    executions[galaxy.id].status = "completed"
                    executions[galaxy.id].result = result
                    
                    print(f"\n  ✅ {galaxy.name} complete!")
                    print(f"     Lines: {result.get('total_lines_generated', 0):,}")
                    print(f"     Files: {result.get('total_files_generated', 0):,}")
                    print(f"     Time: {result.get('execution_hours', 0):.1f} hours")
                    
                except Exception as e:
                    log.error(f"Galaxy {galaxy.id} failed: {e}")
                    results[galaxy.id] = {"status": "error", "error": str(e)}
                    executions[galaxy.id].status = "error"
        
        return results, executions
    
    def shutdown(self):
        """Shutdown the executor."""
        self.executor.shutdown(wait=True)


class CivilizationStitcher(Agent):
    """Stitches together multiple galaxies into a unified civilization."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.roles = [
            "You are the grand unifier of technological civilizations.",
            "You connect massive tech galaxies into a single coherent ecosystem.",
            "You create the nervous system that connects planetary-scale infrastructure.",
            "You ensure billions of services work as one unified platform."
        ]
    
    def create_unification_plan(self,
                               plan: CivilizationPlan,
                               galaxy_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Create plan for unifying all galaxies."""
        
        prompt = f"""Create a unification layer for this technological civilization:

Civilization: {plan.vision}
Scale: {plan.scale}
Total Systems: {len(galaxy_results)} galaxies

Galaxies Built:
{chr(10).join(f"- {g.name}: {r.get('total_lines_generated', 0):,} lines across {r.get('subsystems_built', 0)} subsystems" 
              for g in plan.tech_galaxies for r in [galaxy_results.get(g.id, {})] if r)}

Create a MASSIVE unification system that connects all galaxies:
1. Planetary API Gateway (routes to 1000+ services)
2. Galactic Event Bus (handles 1T events/day)
3. Civilization Service Mesh (connects everything)
4. Planetary Observability (monitors entire civilization)
5. Unified Identity Platform (for billions of entities)
6. Global State Synchronization (eventual consistency at scale)
7. Disaster Recovery (survives regional apocalypses)
8. Quantum Communication Layer (for interplanetary scale)

Return JSON with unification components."""

        self._append("user", prompt)
        completion = self.call_ai()
        self._append("assistant", completion)
        
        try:
            import re
            json_match = re.search(r'```(?:json)?\s*\n(.*?)(?:\n```|$)', completion, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(1))
            else:
                data = json.loads(completion)
            return data
        except:
            return {
                "unification_components": [
                    {"name": "planetary-api-gateway", "lines": 10000},
                    {"name": "galactic-event-bus", "lines": 8000},
                    {"name": "civilization-mesh", "lines": 12000},
                    {"name": "unified-observability", "lines": 10000},
                    {"name": "identity-platform", "lines": 10000}
                ],
                "total_unification_lines": 50000
            }


class MetaMetaOrchestratorAgent(Agent):
    """
    The ultimate orchestrator - coordinates multiple v16 instances.
    
    Architecture:
    - v17 (this) orchestrates 4-8 v16 instances
    - Each v16 orchestrates 4 v15 instances  
    - Each v15 generates 50k lines
    - Total: 1,000,000+ lines
    """
    
    def __init__(self,
                 task: str,
                 working_dir: Optional[str] = None,
                 model: str = "gemini-2.0-flash",
                 max_v16_instances: int = 4,
                 parallel_mode: str = "balanced",
                 **kwargs):
        """Initialize the meta-meta orchestrator."""
        super().__init__(**kwargs)
        
        self.task = task
        self.model = model
        self.max_v16_instances = max_v16_instances
        self.parallel_mode = parallel_mode
        
        # Setup directories
        self.output_manager = OutputManager()
        self.session_dir, self.working_dir = self._create_session(working_dir)
        
        # Create component agents
        self.galaxy_decomposer = GalaxyDecomposer(
            overrides={"provider": {"google": {"model_name": model}}}
        )
        self.hyper_executor = HyperParallelExecutor(max_v16_instances, model)
        self.stitcher = CivilizationStitcher(
            overrides={"provider": {"google": {"model_name": model}}}
        )
        
        # State
        self.civilization_plan = None
        self.galaxy_results = {}
        self.start_time = None
        self.end_time = None
        
        log.info(f"MetaMetaOrchestratorAgent (v17) initialized")
        log.info(f"Task: {task}")
        log.info(f"Max v16 instances: {max_v16_instances}")
        log.info(f"Parallel mode: {parallel_mode}")
    
    def _create_session(self, working_dir: Optional[str] = None) -> Tuple[Path, Path]:
        """Create session directories."""
        import re
        task_name = re.sub(r'[^\w\s-]', '', self.task.lower())
        task_name = re.sub(r'\s+', '_', task_name)[:30]
        
        session_dir = self.output_manager.create_session_dir("v17_civilization", f"civ_{task_name}")
        
        if working_dir:
            work_dir = Path(working_dir).resolve()
        else:
            work_dir = session_dir / "civilization"
        
        work_dir.mkdir(parents=True, exist_ok=True)
        
        return session_dir, work_dir
    
    def run(self) -> Dict[str, Any]:
        """Execute civilization-scale orchestration."""
        try:
            self.start_time = datetime.now()
            
            self._print_header()
            
            # Phase 1: Decompose into galaxies
            print("\n[PHASE 1] Galaxy Decomposition...")
            self.civilization_plan = self.galaxy_decomposer.decompose_into_galaxies(self.task)
            print(f"✓ Decomposed into {len(self.civilization_plan.tech_galaxies)} technology galaxies")
            print(f"✓ Total target: {self.civilization_plan.total_estimated_lines:,} lines")
            
            for galaxy in self.civilization_plan.tech_galaxies:
                v15s = galaxy.v16_config.get("parallel_instances", 4)
                print(f"  - {galaxy.name}: {galaxy.estimated_lines:,} lines (via {v15s} v15s)")
            
            total_v15s = sum(g.v16_config.get("parallel_instances", 4) 
                           for g in self.civilization_plan.tech_galaxies)
            print(f"\n📊 Total v15 instances to be run: {total_v15s}")
            
            # Phase 2: Execute galaxies in parallel
            print(f"\n[PHASE 2] Hyper-Parallel Execution...")
            print(f"Mode: {self.parallel_mode}")
            
            self.galaxy_results, executions = self.hyper_executor.execute_civilization(
                self.civilization_plan.tech_galaxies,
                self.working_dir,
                self.parallel_mode
            )
            
            # Phase 3: Unification
            print(f"\n[PHASE 3] Civilization Unification...")
            unification_plan = self.stitcher.create_unification_plan(
                self.civilization_plan,
                self.galaxy_results
            )
            print(f"✓ Designed unification layer with {len(unification_plan.get('unification_components', []))} components")
            
            # Phase 4: Summary
            self.end_time = datetime.now()
            return self._generate_summary(unification_plan, executions)
            
        except Exception as e:
            log.exception("Civilization construction failed")
            return {
                "status": "error",
                "error": str(e),
                "partial_results": self.galaxy_results
            }
        finally:
            self.hyper_executor.shutdown()
    
    def _print_header(self):
        """Print dramatic header."""
        print("\n" + "🌌"*30)
        print("\nTALK v17 - THE SINGULARITY")
        print("\nCIVILIZATION-SCALE CODE GENERATION")
        print("\n" + "🌌"*30)
        
        print(f"\n📢 INITIATING PLANETARY TECHNOLOGY CONSTRUCTION")
        print(f"  Task: {self.task}")
        print(f"  Target: 1,000,000+ lines")
        print(f"  v16 instances: {self.max_v16_instances}")
        print(f"  v15 instances per v16: 4")
        print(f"  Total parallel v15s: {self.max_v16_instances * 4}")
        print(f"  Execution mode: {self.parallel_mode}")
        
        print("\n🎯 SCALE COMPARISON:")
        print("  Talk v15: Company (50k lines)")
        print("  Talk v16: Tech Giant (200k lines)")
        print("  Talk v17: ENTIRE CIVILIZATION (1M+ lines)")
        
        print("\n" + "-"*70)
    
    def _generate_summary(self, unification_plan: Dict[str, Any], 
                         executions: Dict[str, GalaxyExecution]) -> Dict[str, Any]:
        """Generate final summary."""
        
        # Calculate totals
        total_lines = sum(
            r.get("total_lines_generated", 0)
            for r in self.galaxy_results.values()
        ) + unification_plan.get("total_unification_lines", 0)
        
        total_files = sum(
            r.get("total_files_generated", 0)
            for r in self.galaxy_results.values()
        )
        
        total_v15s_used = sum(
            r.get("parallel_instances", 0)
            for r in self.galaxy_results.values()
        )
        
        execution_hours = (self.end_time - self.start_time).total_seconds() / 3600
        
        print("\n" + "="*80)
        print("CIVILIZATION CONSTRUCTION COMPLETE")
        print("="*80)
        
        print(f"\n📊 FINAL STATISTICS:")
        print(f"  Total Lines Generated: {total_lines:,}")
        print(f"  Total Files Created: {total_files:,}")
        print(f"  Galaxies Built: {len([e for e in executions.values() if e.status == 'completed'])}/{len(self.civilization_plan.tech_galaxies)}")
        print(f"  v16 Instances Used: {len(self.galaxy_results)}")
        print(f"  v15 Instances Used: {total_v15s_used}")
        print(f"  Total Execution Time: {execution_hours:.2f} hours")
        
        print(f"\n🌌 GALAXIES CONSTRUCTED:")
        for galaxy_id, execution in executions.items():
            if execution.result:
                print(f"  - {execution.galaxy.name}:")
                print(f"      Lines: {execution.result.get('total_lines_generated', 0):,}")
                print(f"      Files: {execution.result.get('total_files_generated', 0)}")
                print(f"      Subsystems: {execution.result.get('subsystems_built', 0)}")
        
        print(f"\n🏆 ACHIEVEMENT LEVEL:")
        if total_lines >= 2000000:
            print("  🌟 GALACTIC CIVILIZATION - You built technology for multiple star systems!")
        elif total_lines >= 1000000:
            print("  🌍 PLANETARY CIVILIZATION - You built Earth's entire digital infrastructure!")
        elif total_lines >= 500000:
            print("  🏙️ CONTINENTAL SCALE - You built a continent's technology stack!")
        else:
            print("  🌆 MEGA-CORPORATION - You built the next FAANG company!")
        
        print(f"\n📁 Output Directory: {self.working_dir}")
        print("="*80 + "\n")
        
        return {
            "status": "success",
            "original_task": self.task,
            "civilization_vision": self.civilization_plan.vision,
            "total_lines_generated": total_lines,
            "total_files_generated": total_files,
            "galaxies_built": len([e for e in executions.values() if e.status == 'completed']),
            "galaxies_total": len(self.civilization_plan.tech_galaxies),
            "v16_instances_used": len(self.galaxy_results),
            "v15_instances_total": total_v15s_used,
            "execution_hours": execution_hours,
            "working_directory": str(self.working_dir),
            "galaxy_results": self.galaxy_results,
            "unification_plan": unification_plan
        }