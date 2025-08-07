#!/usr/bin/env python3
"""
MetaOrchestratorAgent - Coordinates multiple Talk v15 instances to build massive systems.

This agent:
1. Decomposes massive tasks into subsystem domains
2. Runs up to 4 Talk v15 instances in parallel
3. Monitors and coordinates their progress
4. Runs a final v15 to stitch everything together
5. Generates 200,000+ lines of integrated code
"""

from __future__ import annotations

import json
import logging
import time
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, List, Set, Tuple
from datetime import datetime, timedelta
import multiprocessing as mp
import queue

from agent.agent import Agent
from agent.output_manager import OutputManager

log = logging.getLogger(__name__)


@dataclass
class SubsystemDomain:
    """Represents a major subsystem that can be built independently."""
    id: str
    name: str
    type: str  # backend, frontend, infrastructure, data, ml, mobile
    description: str
    estimated_lines: int
    dependencies_on: List[str]  # Other domain IDs
    interfaces: List[Dict[str, Any]]  # API contracts with other domains
    assigned_to_instance: Optional[int] = None
    status: str = "planned"
    result: Optional[Dict[str, Any]] = None
    output_dir: Optional[Path] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    def execution_time(self) -> float:
        """Get execution time in hours."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds() / 3600
        return 0.0


@dataclass
class MetaOrchestrationPlan:
    """Plan for building a massive system using parallel v15 instances."""
    original_task: str
    interpreted_scale: str  # google, meta, amazon, microsoft
    total_estimated_lines: int
    subsystem_domains: List[SubsystemDomain]
    integration_points: List[Dict[str, Any]]
    parallel_instances: int = 4
    
    def get_parallel_batches(self) -> List[List[SubsystemDomain]]:
        """Group domains into batches for parallel execution."""
        # Sort by dependencies - domains with no deps first
        sorted_domains = sorted(
            self.subsystem_domains,
            key=lambda d: (len(d.dependencies_on), -d.estimated_lines)
        )
        
        # Create batches
        batches = []
        current_batch = []
        assigned = set()
        
        for domain in sorted_domains:
            # Check if dependencies are satisfied
            deps_satisfied = all(dep in assigned for dep in domain.dependencies_on)
            
            if deps_satisfied:
                current_batch.append(domain)
                
                if len(current_batch) >= self.parallel_instances:
                    batches.append(current_batch)
                    assigned.update(d.id for d in current_batch)
                    current_batch = []
        
        if current_batch:
            batches.append(current_batch)
        
        return batches


class SystemDecomposer(Agent):
    """Decomposes massive tasks into parallel subsystem domains."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.roles = [
            "You are a principal architect at Google/Meta scale.",
            "You decompose MASSIVE systems into independent subsystems.",
            "You think in terms of entire company platforms.",
            "You design for billions of users and exabytes of data.",
            "",
            "When decomposing, you create:",
            "- Independent subsystem domains",
            "- Clear interface contracts",
            "- Minimal coupling between domains",
            "- Maximum parallelization potential",
            "",
            "Each subsystem should be 30,000-50,000 lines minimum."
        ]
    
    def decompose(self, task: str) -> MetaOrchestrationPlan:
        """Decompose task into massive parallel subsystems."""
        prompt = f"""Decompose this task into a MASSIVE multi-subsystem platform:

Task: "{task}"

Think GOOGLE/META/AMAZON scale. This should be a platform that could run a trillion-dollar company.

Create 4-8 major subsystem domains that can be built in parallel. Each should be a complete platform itself.

Example for "build a social media platform":
1. Core Social Graph System (Facebook-scale social network)
2. Content Delivery Platform (YouTube-scale video/media)
3. Real-time Messaging Infrastructure (WhatsApp-scale)
4. Advertising & Monetization Platform (Google Ads-scale)
5. ML/AI Services Platform (recommendation, moderation, analysis)
6. Enterprise Tools Suite (Creator Studio, Business Manager, Analytics)

Return JSON:
{{
    "interpreted_scale": "meta|google|amazon|microsoft",
    "platform_name": "MegaPlatform",
    "vision": "Description of the trillion-dollar platform",
    "total_estimated_lines": 200000,
    "subsystem_domains": [
        {{
            "id": "social-core",
            "name": "Core Social Graph Platform",
            "type": "backend",
            "description": "Facebook-scale social networking core with 100M+ users",
            "estimated_lines": 50000,
            "dependencies_on": [],
            "interfaces": [
                {{"type": "GraphQL", "name": "SocialGraphAPI", "endpoints": 100}},
                {{"type": "gRPC", "name": "InternalUserService", "methods": 50}}
            ],
            "components": [
                "user-service", "friend-service", "feed-service", "group-service",
                "event-service", "page-service", "notification-service", "presence-service"
            ]
        }},
        {{
            "id": "content-platform",
            "name": "Content Delivery Platform",
            "type": "backend",
            "description": "YouTube-scale video and media platform",
            "estimated_lines": 60000,
            "dependencies_on": ["social-core"],
            "interfaces": [
                {{"type": "REST", "name": "ContentAPI", "endpoints": 150}},
                {{"type": "WebSocket", "name": "LiveStreamingAPI", "channels": 20}}
            ],
            "components": [
                "upload-service", "transcoding-service", "cdn-service", "streaming-service",
                "storage-service", "analytics-service", "monetization-service"
            ]
        }},
        // ... more massive subsystems
    ],
    "integration_points": [
        {{
            "from": "content-platform",
            "to": "social-core",
            "type": "event-bus",
            "description": "Content events flow to social graph"
        }},
        // ... more integration points
    ],
    "infrastructure_requirements": {{
        "compute": "10,000+ servers",
        "storage": "100+ PB",
        "databases": ["Spanner", "Bigtable", "Cassandra", "MongoDB"],
        "regions": 20,
        "cdn_pops": 200
    }}
}}

Make this ABSOLUTELY MASSIVE! Each subsystem should be enterprise-scale on its own!"""

        self._append("user", prompt)
        completion = self.call_ai()
        self._append("assistant", completion)
        
        try:
            import re
            json_match = re.search(r'```(?:json)?\s*\n(.*?)(?:\n```|$)', completion, re.DOTALL)
            if json_match:
                json_text = json_match.group(1)
            else:
                json_text = completion
            
            data = json.loads(json_text)
            
            # Convert to plan
            domains = []
            for d in data.get("subsystem_domains", []):
                domain = SubsystemDomain(
                    id=d["id"],
                    name=d["name"],
                    type=d["type"],
                    description=d["description"],
                    estimated_lines=d["estimated_lines"],
                    dependencies_on=d.get("dependencies_on", []),
                    interfaces=d.get("interfaces", [])
                )
                domains.append(domain)
            
            plan = MetaOrchestrationPlan(
                original_task=task,
                interpreted_scale=data.get("interpreted_scale", "meta"),
                total_estimated_lines=data.get("total_estimated_lines", 200000),
                subsystem_domains=domains,
                integration_points=data.get("integration_points", [])
            )
            
            log.info(f"Decomposed into {len(domains)} subsystem domains, "
                    f"{plan.total_estimated_lines:,} total lines")
            
            return plan
            
        except (json.JSONDecodeError, KeyError) as e:
            log.error(f"Failed to parse decomposition: {e}")
            return self._create_default_plan(task)
    
    def _create_default_plan(self, task: str) -> MetaOrchestrationPlan:
        """Create default mega-scale plan."""
        domains = [
            SubsystemDomain(
                id="core-platform",
                name="Core Platform Services",
                type="backend",
                description="Main platform with user, auth, content services",
                estimated_lines=50000,
                dependencies_on=[],
                interfaces=[{"type": "REST", "endpoints": 100}]
            ),
            SubsystemDomain(
                id="data-platform",
                name="Big Data & Analytics Platform",
                type="data",
                description="Data lake, warehousing, real-time analytics",
                estimated_lines=40000,
                dependencies_on=["core-platform"],
                interfaces=[{"type": "Kafka", "topics": 50}]
            ),
            SubsystemDomain(
                id="ml-platform",
                name="ML/AI Services Platform",
                type="ml",
                description="Recommendation, NLP, computer vision services",
                estimated_lines=45000,
                dependencies_on=["data-platform"],
                interfaces=[{"type": "gRPC", "services": 20}]
            ),
            SubsystemDomain(
                id="frontend-suite",
                name="Frontend Applications Suite",
                type="frontend",
                description="Web, mobile, desktop, TV applications",
                estimated_lines=55000,
                dependencies_on=["core-platform"],
                interfaces=[{"type": "GraphQL", "schemas": 30}]
            )
        ]
        
        return MetaOrchestrationPlan(
            original_task=task,
            interpreted_scale="meta",
            total_estimated_lines=190000,
            subsystem_domains=domains,
            integration_points=[]
        )


class ParallelV15Executor:
    """Executes multiple Talk v15 instances in parallel."""
    
    def __init__(self, max_parallel: int = 4, model: str = "gemini-2.0-flash"):
        self.max_parallel = max_parallel
        self.model = model
        self.executor = ProcessPoolExecutor(max_workers=max_parallel)
        self.results = {}
        self.progress_queue = mp.Queue()
        
    def execute_domain(self, domain: SubsystemDomain, working_dir: Path) -> Dict[str, Any]:
        """Execute a single domain using Talk v15."""
        try:
            # Import here to avoid pickle issues
            from talk.talk_v15_enterprise import TalkV15Orchestrator
            
            # Create task description for this domain
            task = f"Build {domain.name}: {domain.description}"
            
            # Create output directory
            domain_dir = working_dir / domain.id
            domain_dir.mkdir(parents=True, exist_ok=True)
            
            # Run Talk v15 in big mode
            orchestrator = TalkV15Orchestrator(
                task=task,
                big_mode=True,
                model=self.model,
                working_dir=str(domain_dir),
                target_lines=domain.estimated_lines,
                minimum_hours=1.0,  # Minimum 1 hour per domain
                verbose=False
            )
            
            domain.start_time = datetime.now()
            result = orchestrator.run()
            domain.end_time = datetime.now()
            
            # Add domain info to result
            result["domain_id"] = domain.id
            result["domain_name"] = domain.name
            result["output_dir"] = str(domain_dir)
            
            return result
            
        except Exception as e:
            log.error(f"Failed to execute domain {domain.id}: {e}")
            return {
                "status": "error",
                "domain_id": domain.id,
                "error": str(e)
            }
    
    def execute_parallel_batch(self, domains: List[SubsystemDomain], 
                             working_dir: Path) -> Dict[str, Dict[str, Any]]:
        """Execute a batch of domains in parallel."""
        futures = {}
        results = {}
        
        print(f"\n🚀 Launching {len(domains)} parallel v15 instances...")
        
        for i, domain in enumerate(domains):
            domain.assigned_to_instance = i
            future = self.executor.submit(self.execute_domain, domain, working_dir)
            futures[future] = domain
            print(f"  Instance {i+1}: {domain.name} ({domain.estimated_lines:,} lines)")
        
        print("\n⏳ Parallel execution in progress...")
        print("  (Each instance running Talk v15 --big independently)")
        
        # Monitor progress
        for future in as_completed(futures):
            domain = futures[future]
            try:
                result = future.result()
                results[domain.id] = result
                domain.result = result
                domain.status = "completed" if result.get("status") == "success" else "error"
                
                print(f"\n  ✓ Completed: {domain.name}")
                print(f"    Lines: {result.get('lines_generated', 0):,}")
                print(f"    Files: {result.get('files_generated', 0)}")
                print(f"    Time: {domain.execution_time():.1f} hours")
                
            except Exception as e:
                log.error(f"Domain {domain.id} failed: {e}")
                results[domain.id] = {"status": "error", "error": str(e)}
                domain.status = "error"
        
        return results
    
    def shutdown(self):
        """Shutdown the executor."""
        self.executor.shutdown(wait=True)


class IntegrationStitcher(Agent):
    """Stitches together results from parallel v15 executions."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.roles = [
            "You are a master system integrator.",
            "You stitch together massive subsystems into unified platforms.",
            "You create seamless integration layers.",
            "You ensure all parts work as one cohesive system."
        ]
    
    def create_integration_plan(self, plan: MetaOrchestrationPlan, 
                               results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Create plan for stitching subsystems together."""
        prompt = f"""Create integration layer for these completed subsystems:

Platform: {plan.interpreted_scale}-scale system
Total Components: {len(results)}

Subsystems Built:
{chr(10).join(f"- {d.name}: {d.result.get('lines_generated', 0):,} lines in {d.result.get('files_generated', 0)} files" 
              for d in plan.subsystem_domains if d.result)}

Integration Points:
{json.dumps(plan.integration_points, indent=2)}

Create a comprehensive integration system with:
1. API Gateway that routes between all subsystems
2. Event bus for inter-service communication
3. Service mesh for internal networking
4. Distributed tracing and monitoring
5. Centralized authentication/authorization
6. Data synchronization services
7. Orchestration workflows
8. Testing harnesses for integration tests
9. Documentation portal
10. DevOps pipeline for the entire system

Return JSON with integration components to build:
{{
    "integration_components": [
        {{
            "name": "api-gateway-orchestrator",
            "description": "Master API gateway routing to all subsystems",
            "estimated_lines": 5000,
            "connects_to": ["all-subsystems"]
        }},
        // ... more integration components
    ],
    "total_integration_lines": 20000,
    "deployment_strategy": "kubernetes-federation",
    "monitoring_stack": ["prometheus", "grafana", "jaeger", "elk"]
}}"""

        self._append("user", prompt)
        completion = self.call_ai()
        self._append("assistant", completion)
        
        try:
            import re
            json_match = re.search(r'```(?:json)?\s*\n(.*?)(?:\n```|$)', completion, re.DOTALL)
            if json_match:
                json_text = json_match.group(1)
            else:
                json_text = completion
            
            return json.loads(json_text)
            
        except json.JSONDecodeError:
            return {
                "integration_components": [
                    {
                        "name": "api-gateway",
                        "description": "Central API gateway",
                        "estimated_lines": 5000
                    },
                    {
                        "name": "event-bus",
                        "description": "Kafka-based event bus",
                        "estimated_lines": 3000
                    },
                    {
                        "name": "service-mesh",
                        "description": "Istio service mesh config",
                        "estimated_lines": 2000
                    }
                ],
                "total_integration_lines": 10000
            }
    
    def build_integration_layer(self, integration_plan: Dict[str, Any],
                               working_dir: Path) -> Dict[str, Any]:
        """Build the integration layer using Talk v15."""
        try:
            from talk.talk_v15_enterprise import TalkV15Orchestrator
            
            # Create integration task
            task = f"Build integration layer with: {', '.join(c['name'] for c in integration_plan['integration_components'])}"
            
            integration_dir = working_dir / "integration"
            integration_dir.mkdir(parents=True, exist_ok=True)
            
            # Run Talk v15 for integration
            orchestrator = TalkV15Orchestrator(
                task=task,
                big_mode=True,
                model="gemini-2.0-flash",
                working_dir=str(integration_dir),
                target_lines=integration_plan.get("total_integration_lines", 10000),
                minimum_hours=0.5,
                verbose=False
            )
            
            result = orchestrator.run()
            result["type"] = "integration"
            
            return result
            
        except Exception as e:
            log.error(f"Failed to build integration layer: {e}")
            return {"status": "error", "error": str(e)}


class MetaOrchestratorAgent(Agent):
    """
    Master orchestrator that coordinates multiple Talk v15 instances.
    
    This agent:
    1. Decomposes massive tasks into subsystems
    2. Runs up to 4 Talk v15 instances in parallel
    3. Monitors and coordinates progress
    4. Builds integration layer to stitch everything together
    5. Generates 200,000+ lines of integrated code
    """
    
    def __init__(self,
                 task: str,
                 working_dir: Optional[str] = None,
                 model: str = "gemini-2.0-flash",
                 max_parallel: int = 4,
                 **kwargs):
        """Initialize meta orchestrator."""
        super().__init__(**kwargs)
        
        self.task = task
        self.model = model
        self.max_parallel = max_parallel
        
        # Setup directories
        self.output_manager = OutputManager()
        self.session_dir, self.working_dir = self._create_session(working_dir)
        
        # Create components
        self.decomposer = SystemDecomposer(
            overrides={"provider": {"google": {"model_name": model}}}
        )
        self.executor = ParallelV15Executor(max_parallel, model)
        self.stitcher = IntegrationStitcher(
            overrides={"provider": {"google": {"model_name": model}}}
        )
        
        # State
        self.plan = None
        self.results = {}
        self.start_time = None
        self.end_time = None
        
        log.info(f"MetaOrchestratorAgent initialized")
        log.info(f"Task: {task}")
        log.info(f"Max Parallel: {max_parallel}")
    
    def _create_session(self, working_dir: Optional[str] = None) -> Tuple[Path, Path]:
        """Create session directories."""
        import re
        task_name = re.sub(r'[^\w\s-]', '', self.task.lower())
        task_name = re.sub(r'\s+', '_', task_name)[:30]
        
        session_dir = self.output_manager.create_session_dir("meta_orchestrator", f"mega_{task_name}")
        
        if working_dir:
            work_dir = Path(working_dir).resolve()
        else:
            work_dir = session_dir / "workspace"
        
        work_dir.mkdir(parents=True, exist_ok=True)
        
        return session_dir, work_dir
    
    def run(self) -> Dict[str, Any]:
        """Execute meta-orchestration of multiple v15 instances."""
        try:
            self.start_time = datetime.now()
            
            print(f"\n{'='*80}")
            print(f"TALK v16 META-ORCHESTRATOR")
            print(f"{'='*80}")
            print(f"Task: {self.task}")
            print(f"Scale: GOOGLE/META/AMAZON")
            print(f"Target: 200,000+ lines")
            print(f"Parallel Instances: {self.max_parallel}")
            print(f"{'='*80}\n")
            
            # Phase 1: Decompose into subsystems
            print("\n[PHASE 1] System Decomposition...")
            self.plan = self.decomposer.decompose(self.task)
            print(f"✓ Decomposed into {len(self.plan.subsystem_domains)} subsystem domains")
            print(f"✓ Total target: {self.plan.total_estimated_lines:,} lines")
            
            for domain in self.plan.subsystem_domains:
                print(f"  - {domain.name}: {domain.estimated_lines:,} lines")
            
            # Phase 2: Execute in parallel batches
            print(f"\n[PHASE 2] Parallel Execution...")
            batches = self.plan.get_parallel_batches()
            print(f"✓ Organized into {len(batches)} execution batches")
            
            all_results = {}
            for i, batch in enumerate(batches, 1):
                print(f"\n🎯 Batch {i}/{len(batches)}")
                batch_results = self.executor.execute_parallel_batch(batch, self.working_dir)
                all_results.update(batch_results)
                
                # Update domains with results
                for domain in batch:
                    if domain.id in batch_results:
                        domain.result = batch_results[domain.id]
            
            self.results = all_results
            
            # Phase 3: Build integration layer
            print(f"\n[PHASE 3] Integration & Stitching...")
            integration_plan = self.stitcher.create_integration_plan(self.plan, self.results)
            print(f"✓ Designed integration layer with {len(integration_plan['integration_components'])} components")
            
            integration_result = self.stitcher.build_integration_layer(integration_plan, self.working_dir)
            print(f"✓ Built integration layer: {integration_result.get('lines_generated', 0):,} lines")
            
            # Phase 4: Final summary
            self.end_time = datetime.now()
            return self._generate_summary(integration_result)
            
        except Exception as e:
            log.exception("Meta-orchestration failed")
            return {
                "status": "error",
                "error": str(e),
                "partial_results": self.results
            }
        finally:
            self.executor.shutdown()
    
    def _generate_summary(self, integration_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate final summary of the mega-build."""
        total_lines = sum(
            r.get("lines_generated", 0) 
            for r in self.results.values()
        ) + integration_result.get("lines_generated", 0)
        
        total_files = sum(
            r.get("files_generated", 0)
            for r in self.results.values()
        ) + integration_result.get("files_generated", 0)
        
        execution_hours = (self.end_time - self.start_time).total_seconds() / 3600
        
        print(f"\n{'='*80}")
        print(f"MEGA-BUILD COMPLETE")
        print(f"{'='*80}")
        print(f"Original Task: {self.task}")
        print(f"Interpreted As: {self.plan.interpreted_scale.upper()}-scale platform")
        print(f"")
        print(f"📊 Final Statistics:")
        print(f"  Total Lines: {total_lines:,}")
        print(f"  Total Files: {total_files:,}")
        print(f"  Subsystems Built: {len([d for d in self.plan.subsystem_domains if d.status == 'completed'])}/{len(self.plan.subsystem_domains)}")
        print(f"  Parallel Instances Used: {self.max_parallel}")
        print(f"  Total Execution Time: {execution_hours:.2f} hours")
        print(f"")
        print(f"📁 Subsystem Directories:")
        for domain in self.plan.subsystem_domains:
            if domain.result and domain.result.get("status") == "success":
                print(f"  - {domain.id}/: {domain.result.get('lines_generated', 0):,} lines")
        print(f"  - integration/: {integration_result.get('lines_generated', 0):,} lines")
        print(f"")
        print(f"🏗️ Architecture:")
        print(f"  Scale: {self.plan.interpreted_scale.upper()}")
        print(f"  Subsystems: {len(self.plan.subsystem_domains)}")
        print(f"  Integration Points: {len(self.plan.integration_points)}")
        print(f"")
        print(f"📁 Output: {self.working_dir}")
        print(f"{'='*80}\n")
        
        return {
            "status": "success",
            "original_task": self.task,
            "interpreted_scale": self.plan.interpreted_scale,
            "total_lines_generated": total_lines,
            "total_files_generated": total_files,
            "subsystems_built": len([d for d in self.plan.subsystem_domains if d.status == 'completed']),
            "subsystems_total": len(self.plan.subsystem_domains),
            "parallel_instances": self.max_parallel,
            "execution_hours": execution_hours,
            "working_directory": str(self.working_dir),
            "subsystem_results": self.results,
            "integration_result": integration_result
        }


def main():
    """Test meta orchestrator."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Meta Orchestrator - Coordinate multiple v15 instances")
    parser.add_argument("task", help="Massive task description")
    parser.add_argument("--model", default="gemini-2.0-flash")
    parser.add_argument("--working-dir", help="Working directory")
    parser.add_argument("--parallel", type=int, default=4, help="Max parallel instances")
    
    args = parser.parse_args()
    
    agent = MetaOrchestratorAgent(
        task=args.task,
        working_dir=args.working_dir,
        model=args.model,
        max_parallel=args.parallel
    )
    
    result = agent.run()
    
    # Save result
    import json
    with open("meta_orchestrator_result.json", "w") as f:
        json.dump(result, f, indent=2)
    
    return 0 if result.get("status") == "success" else 1


if __name__ == "__main__":
    exit(main())