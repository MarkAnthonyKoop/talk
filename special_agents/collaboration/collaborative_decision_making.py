#!/usr/bin/env python3
"""
CollaborativeDecisionMaking - Voting and consensus system for agent collaboration.

This module provides mechanisms for agents to make collective decisions through
voting, consensus building, and priority-based decision making. It supports
multiple voting strategies and decision types.

Features:
- Multiple voting types (simple majority, weighted, consensus)
- Decision proposal and deliberation system
- Priority-based conflict resolution
- Quorum requirements and participation tracking
- Time-limited voting windows
- Decision history and audit trail
"""

import asyncio
import time
import json
import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Set, Callable
from enum import Enum
import uuid

log = logging.getLogger(__name__)

class VoteType(Enum):
    """Types of votes that can be cast."""
    APPROVE = "approve"
    REJECT = "reject"
    ABSTAIN = "abstain"
    DELEGATE = "delegate"  # Delegate vote to another agent

class DecisionType(Enum):
    """Types of decisions that can be made."""
    CODE_MERGE = "code_merge"
    ARCHITECTURE_CHANGE = "architecture_change"
    TASK_ASSIGNMENT = "task_assignment"
    CONFLICT_RESOLUTION = "conflict_resolution"
    PRIORITY_CHANGE = "priority_change"
    RESOURCE_ALLOCATION = "resource_allocation"
    WORKFLOW_MODIFICATION = "workflow_modification"

class VotingStrategy(Enum):
    """Voting strategies for decision making."""
    SIMPLE_MAJORITY = "simple_majority"  # >50% approval
    SUPERMAJORITY = "supermajority"      # ≥66% approval
    CONSENSUS = "consensus"              # 100% approval (excluding abstains)
    WEIGHTED = "weighted"                # Weighted by agent capabilities/role
    PRIORITY_BASED = "priority_based"    # Higher priority agents have more weight

@dataclass
class Vote:
    """A vote cast by an agent."""
    vote_id: str
    decision_id: str
    agent_id: str
    vote_type: VoteType
    weight: float
    reasoning: Optional[str] = None
    timestamp: float = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
        if self.vote_id is None:
            self.vote_id = str(uuid.uuid4())

@dataclass
class Decision:
    """A decision that requires voting."""
    decision_id: str
    title: str
    description: str
    decision_type: DecisionType
    proposer_id: str
    voting_strategy: VotingStrategy
    created_at: float
    deadline: Optional[float] = None
    quorum_required: int = 1
    context: Dict[str, Any] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.decision_id is None:
            self.decision_id = str(uuid.uuid4())
        if self.created_at is None:
            self.created_at = time.time()

@dataclass
class DecisionResult:
    """Result of a decision."""
    decision_id: str
    status: str  # "approved", "rejected", "timeout", "insufficient_quorum"
    total_votes: int
    approve_votes: int
    reject_votes: int
    abstain_votes: int
    weighted_score: float
    quorum_met: bool
    finalized_at: float
    reasoning: str = ""

class VotingSystem:
    """
    Collaborative decision making system with voting mechanisms.
    
    Enables agents to propose decisions, vote on them, and reach consensus
    through various voting strategies and mechanisms.
    """
    
    def __init__(self, default_deadline_minutes: float = 30.0):
        """
        Initialize the voting system.
        
        Args:
            default_deadline_minutes: Default voting deadline in minutes
        """
        self.default_deadline_minutes = default_deadline_minutes
        
        # Decision and vote storage
        self.decisions: Dict[str, Decision] = {}
        self.votes: Dict[str, List[Vote]] = {}  # decision_id -> [votes]
        self.decision_results: Dict[str, DecisionResult] = {}
        
        # Agent information for weighted voting
        self.agent_weights: Dict[str, float] = {}  # agent_id -> weight
        self.agent_roles: Dict[str, str] = {}      # agent_id -> role
        self.agent_capabilities: Dict[str, List[str]] = {}  # agent_id -> [capabilities]
        
        # Callbacks for decision events
        self.decision_callbacks: List[Callable] = []
        
        # Active voting sessions
        self.active_sessions: Set[str] = set()
        
        log.info("VotingSystem initialized")
    
    def register_agent(self, agent_id: str, role: str, capabilities: List[str], 
                      weight: float = 1.0) -> bool:
        """
        Register an agent for voting.
        
        Args:
            agent_id: Unique agent identifier
            role: Agent role (affects voting weight)
            capabilities: List of agent capabilities
            weight: Base voting weight
            
        Returns:
            True if registration successful
        """
        self.agent_weights[agent_id] = weight
        self.agent_roles[agent_id] = role
        self.agent_capabilities[agent_id] = capabilities
        
        log.info(f"Agent registered for voting: {agent_id} ({role}, weight: {weight})")
        return True
    
    def unregister_agent(self, agent_id: str) -> bool:
        """Unregister an agent from voting."""
        removed = False
        
        if agent_id in self.agent_weights:
            del self.agent_weights[agent_id]
            removed = True
        
        if agent_id in self.agent_roles:
            del self.agent_roles[agent_id]
        
        if agent_id in self.agent_capabilities:
            del self.agent_capabilities[agent_id]
        
        if removed:
            log.info(f"Agent unregistered from voting: {agent_id}")
        
        return removed
    
    async def create_decision(self, title: str, description: str, 
                            decision_type: DecisionType, proposer_id: str,
                            voting_strategy: VotingStrategy = VotingStrategy.SIMPLE_MAJORITY,
                            deadline_minutes: float = None,
                            quorum_required: int = None,
                            context: Dict[str, Any] = None) -> str:
        """
        Create a new decision for voting.
        
        Args:
            title: Short title for the decision
            description: Detailed description
            decision_type: Type of decision being made
            proposer_id: Agent proposing the decision
            voting_strategy: Strategy for determining outcome
            deadline_minutes: Voting deadline in minutes
            quorum_required: Minimum number of votes required
            context: Additional context data
            
        Returns:
            Decision ID
        """
        if deadline_minutes is None:
            deadline_minutes = self.default_deadline_minutes
        
        if quorum_required is None:
            # Default quorum is majority of registered agents
            quorum_required = max(1, len(self.agent_weights) // 2 + 1)
        
        decision_id = str(uuid.uuid4())
        deadline = time.time() + (deadline_minutes * 60) if deadline_minutes > 0 else None
        
        decision = Decision(
            decision_id=decision_id,
            title=title,
            description=description,
            decision_type=decision_type,
            proposer_id=proposer_id,
            voting_strategy=voting_strategy,
            created_at=time.time(),
            deadline=deadline,
            quorum_required=quorum_required,
            context=context or {}
        )
        
        self.decisions[decision_id] = decision
        self.votes[decision_id] = []
        self.active_sessions.add(decision_id)
        
        log.info(f"Decision created: {title} ({decision_id}) by {proposer_id}")
        
        # Notify callbacks
        await self._notify_callbacks("decision_created", decision)
        
        return decision_id
    
    async def cast_vote(self, decision_id: str, agent_id: str, vote_type: VoteType,
                       reasoning: str = None, metadata: Dict[str, Any] = None) -> bool:
        """
        Cast a vote on a decision.
        
        Args:
            decision_id: ID of decision to vote on
            agent_id: Agent casting the vote
            vote_type: Type of vote being cast
            reasoning: Optional reasoning for the vote
            metadata: Additional vote metadata
            
        Returns:
            True if vote was accepted
        """
        if decision_id not in self.decisions:
            log.warning(f"Vote attempted on unknown decision: {decision_id}")
            return False
        
        decision = self.decisions[decision_id]
        
        # Check if voting is still open
        if decision.deadline and time.time() > decision.deadline:
            log.warning(f"Vote attempted after deadline: {decision_id}")
            return False
        
        if decision_id not in self.active_sessions:
            log.warning(f"Vote attempted on closed session: {decision_id}")
            return False
        
        # Check if agent is registered
        if agent_id not in self.agent_weights:
            log.warning(f"Vote attempted by unregistered agent: {agent_id}")
            return False
        
        # Remove any existing vote from this agent
        existing_votes = self.votes[decision_id]
        self.votes[decision_id] = [v for v in existing_votes if v.agent_id != agent_id]
        
        # Calculate vote weight
        weight = self._calculate_vote_weight(agent_id, decision)
        
        # Create and store vote
        vote = Vote(
            vote_id=str(uuid.uuid4()),
            decision_id=decision_id,
            agent_id=agent_id,
            vote_type=vote_type,
            weight=weight,
            reasoning=reasoning,
            metadata=metadata or {}
        )
        
        self.votes[decision_id].append(vote)
        
        log.info(f"Vote cast: {vote_type.value} on {decision_id} by {agent_id} (weight: {weight})")
        
        # Check if decision can be finalized
        if await self._check_decision_ready(decision_id):
            await self._finalize_decision(decision_id)
        
        return True
    
    def _calculate_vote_weight(self, agent_id: str, decision: Decision) -> float:
        """Calculate the weight of an agent's vote for a specific decision."""
        base_weight = self.agent_weights.get(agent_id, 1.0)
        
        if decision.voting_strategy == VotingStrategy.WEIGHTED:
            # Weight based on role and capabilities
            role = self.agent_roles.get(agent_id, "")
            capabilities = self.agent_capabilities.get(agent_id, [])
            
            # Role-based multipliers
            role_multipliers = {
                "lead": 2.0,
                "senior": 1.5,
                "specialist": 1.3,
                "regular": 1.0
            }
            role_multiplier = role_multipliers.get(role, 1.0)
            
            # Capability-based bonus
            capability_bonus = len(capabilities) * 0.1
            
            return base_weight * role_multiplier + capability_bonus
        
        elif decision.voting_strategy == VotingStrategy.PRIORITY_BASED:
            # Weight based on decision type and agent capabilities
            decision_type = decision.decision_type
            capabilities = self.agent_capabilities.get(agent_id, [])
            
            # Decision type relevance
            relevance_map = {
                DecisionType.CODE_MERGE: ["python", "javascript", "code_review"],
                DecisionType.ARCHITECTURE_CHANGE: ["architecture", "design", "system_design"],
                DecisionType.TASK_ASSIGNMENT: ["project_management", "coordination"],
                DecisionType.CONFLICT_RESOLUTION: ["mediation", "leadership"],
            }
            
            relevant_capabilities = relevance_map.get(decision_type, [])
            relevance_bonus = sum(1.0 for cap in capabilities if cap in relevant_capabilities)
            
            return base_weight + relevance_bonus
        
        else:
            # Simple/consensus voting - equal weights
            return 1.0
    
    async def _check_decision_ready(self, decision_id: str) -> bool:
        """Check if a decision is ready to be finalized."""
        decision = self.decisions[decision_id]
        votes = self.votes[decision_id]
        
        # Check quorum
        if len(votes) < decision.quorum_required:
            return False
        
        # Check deadline
        if decision.deadline and time.time() > decision.deadline:
            return True
        
        # For consensus, check if all registered agents have voted
        if decision.voting_strategy == VotingStrategy.CONSENSUS:
            voted_agents = {vote.agent_id for vote in votes}
            registered_agents = set(self.agent_weights.keys())
            return voted_agents >= registered_agents
        
        # For other strategies, can finalize early if outcome is clear
        result = self._calculate_result(decision_id)
        if result:
            total_possible_weight = sum(self.agent_weights.values())
            votes_cast_weight = sum(vote.weight for vote in votes)
            
            # If we have >50% participation and clear outcome, can finalize
            if votes_cast_weight > total_possible_weight * 0.5:
                return True
        
        return False
    
    async def _finalize_decision(self, decision_id: str):
        """Finalize a decision and calculate the result."""
        if decision_id not in self.active_sessions:
            return
        
        self.active_sessions.remove(decision_id)
        result = self._calculate_result(decision_id)
        
        if result:
            self.decision_results[decision_id] = result
            log.info(f"Decision finalized: {decision_id} - {result.status}")
            
            # Notify callbacks
            await self._notify_callbacks("decision_finalized", result)
    
    def _calculate_result(self, decision_id: str) -> Optional[DecisionResult]:
        """Calculate the result of a decision based on votes."""
        if decision_id not in self.decisions:
            return None
        
        decision = self.decisions[decision_id]
        votes = self.votes[decision_id]
        
        if not votes:
            return None
        
        # Count votes
        approve_votes = sum(1 for v in votes if v.vote_type == VoteType.APPROVE)
        reject_votes = sum(1 for v in votes if v.vote_type == VoteType.REJECT)
        abstain_votes = sum(1 for v in votes if v.vote_type == VoteType.ABSTAIN)
        total_votes = len(votes)
        
        # Calculate weighted score
        approve_weight = sum(v.weight for v in votes if v.vote_type == VoteType.APPROVE)
        reject_weight = sum(v.weight for v in votes if v.vote_type == VoteType.REJECT)
        total_weight = approve_weight + reject_weight  # Exclude abstains from weight calculation
        
        weighted_score = approve_weight / total_weight if total_weight > 0 else 0.0
        
        # Check quorum
        quorum_met = total_votes >= decision.quorum_required
        
        # Determine status based on voting strategy
        status = "insufficient_quorum"
        reasoning = ""
        
        if quorum_met:
            if decision.voting_strategy == VotingStrategy.SIMPLE_MAJORITY:
                if weighted_score > 0.5:
                    status = "approved"
                    reasoning = f"Simple majority achieved ({weighted_score:.1%})"
                else:
                    status = "rejected"
                    reasoning = f"Simple majority not achieved ({weighted_score:.1%})"
            
            elif decision.voting_strategy == VotingStrategy.SUPERMAJORITY:
                if weighted_score >= 0.66:
                    status = "approved"
                    reasoning = f"Supermajority achieved ({weighted_score:.1%})"
                else:
                    status = "rejected"
                    reasoning = f"Supermajority not achieved ({weighted_score:.1%})"
            
            elif decision.voting_strategy == VotingStrategy.CONSENSUS:
                if reject_votes == 0 and approve_votes > 0:
                    status = "approved"
                    reasoning = "Consensus achieved (no rejections)"
                else:
                    status = "rejected"
                    reasoning = f"Consensus not achieved ({reject_votes} rejections)"
            
            elif decision.voting_strategy in [VotingStrategy.WEIGHTED, VotingStrategy.PRIORITY_BASED]:
                threshold = 0.6  # Higher threshold for weighted voting
                if weighted_score >= threshold:
                    status = "approved"
                    reasoning = f"Weighted threshold achieved ({weighted_score:.1%})"
                else:
                    status = "rejected"
                    reasoning = f"Weighted threshold not achieved ({weighted_score:.1%})"
        else:
            reasoning = f"Quorum not met ({total_votes}/{decision.quorum_required})"
        
        # Check for timeout
        if decision.deadline and time.time() > decision.deadline:
            if status == "insufficient_quorum":
                status = "timeout"
                reasoning = "Voting deadline exceeded with insufficient quorum"
        
        return DecisionResult(
            decision_id=decision_id,
            status=status,
            total_votes=total_votes,
            approve_votes=approve_votes,
            reject_votes=reject_votes,
            abstain_votes=abstain_votes,
            weighted_score=weighted_score,
            quorum_met=quorum_met,
            finalized_at=time.time(),
            reasoning=reasoning
        )
    
    async def get_result(self, decision_id: str) -> Optional[Dict[str, Any]]:
        """Get the result of a decision."""
        if decision_id in self.decision_results:
            return asdict(self.decision_results[decision_id])
        
        # If decision is still active, return current status
        if decision_id in self.active_sessions:
            result = self._calculate_result(decision_id)
            if result:
                return asdict(result)
        
        return None
    
    async def close_decision(self, decision_id: str, force: bool = False) -> bool:
        """
        Close a decision (stop accepting votes).
        
        Args:
            decision_id: Decision to close
            force: Force close even if deadline hasn't passed
            
        Returns:
            True if decision was closed
        """
        if decision_id not in self.active_sessions:
            return False
        
        decision = self.decisions[decision_id]
        
        # Check if decision can be closed
        if not force and decision.deadline and time.time() < decision.deadline:
            # Only allow early close if quorum is met and outcome is clear
            if not await self._check_decision_ready(decision_id):
                return False
        
        await self._finalize_decision(decision_id)
        return True
    
    def get_active_decisions(self) -> List[Dict[str, Any]]:
        """Get all active decisions that are still accepting votes."""
        active = []
        
        for decision_id in self.active_sessions:
            if decision_id in self.decisions:
                decision = self.decisions[decision_id]
                votes = self.votes.get(decision_id, [])
                
                decision_info = asdict(decision)
                decision_info['vote_count'] = len(votes)
                decision_info['quorum_progress'] = f"{len(votes)}/{decision.quorum_required}"
                
                # Add current vote tally
                approve_count = sum(1 for v in votes if v.vote_type == VoteType.APPROVE)
                reject_count = sum(1 for v in votes if v.vote_type == VoteType.REJECT)
                abstain_count = sum(1 for v in votes if v.vote_type == VoteType.ABSTAIN)
                
                decision_info['current_tally'] = {
                    'approve': approve_count,
                    'reject': reject_count,
                    'abstain': abstain_count
                }
                
                active.append(decision_info)
        
        return sorted(active, key=lambda d: d['created_at'])
    
    def get_vote_history(self, agent_id: str = None, decision_id: str = None) -> List[Dict[str, Any]]:
        """Get vote history with optional filtering."""
        all_votes = []
        
        for d_id, votes in self.votes.items():
            if decision_id and d_id != decision_id:
                continue
            
            for vote in votes:
                if agent_id and vote.agent_id != agent_id:
                    continue
                
                vote_info = asdict(vote)
                
                # Add decision context
                if d_id in self.decisions:
                    decision = self.decisions[d_id]
                    vote_info['decision_title'] = decision.title
                    vote_info['decision_type'] = decision.decision_type.value
                
                all_votes.append(vote_info)
        
        return sorted(all_votes, key=lambda v: v['timestamp'], reverse=True)
    
    async def add_decision_callback(self, callback: Callable):
        """Add a callback for decision events."""
        self.decision_callbacks.append(callback)
    
    async def _notify_callbacks(self, event_type: str, data: Any):
        """Notify all registered callbacks of an event."""
        for callback in self.decision_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event_type, data)
                else:
                    callback(event_type, data)
            except Exception as e:
                log.error(f"Error in decision callback: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get voting system statistics."""
        total_decisions = len(self.decisions)
        active_decisions = len(self.active_sessions)
        completed_decisions = len(self.decision_results)
        
        total_votes = sum(len(votes) for votes in self.votes.values())
        
        # Decision type breakdown
        decision_types = {}
        for decision in self.decisions.values():
            dt = decision.decision_type.value
            decision_types[dt] = decision_types.get(dt, 0) + 1
        
        # Approval rate
        approved = sum(1 for result in self.decision_results.values() if result.status == "approved")
        approval_rate = approved / completed_decisions if completed_decisions > 0 else 0
        
        return {
            "total_decisions": total_decisions,
            "active_decisions": active_decisions,
            "completed_decisions": completed_decisions,
            "total_votes": total_votes,
            "registered_voters": len(self.agent_weights),
            "decision_types": decision_types,
            "approval_rate": approval_rate,
            "average_votes_per_decision": total_votes / total_decisions if total_decisions > 0 else 0
        }

# Example usage
async def main():
    """Example usage of the VotingSystem."""
    voting_system = VotingSystem()
    
    # Register agents
    await voting_system.register_agent("agent1", "lead", ["python", "architecture"], 2.0)
    await voting_system.register_agent("agent2", "senior", ["javascript", "testing"], 1.5)
    await voting_system.register_agent("agent3", "regular", ["python"], 1.0)
    
    # Create a decision
    decision_id = await voting_system.create_decision(
        title="Merge feature branch",
        description="Should we merge the new authentication feature?",
        decision_type=DecisionType.CODE_MERGE,
        proposer_id="agent1",
        voting_strategy=VotingStrategy.SIMPLE_MAJORITY,
        deadline_minutes=1.0  # 1 minute for demo
    )
    
    print(f"Created decision: {decision_id}")
    
    # Cast votes
    await voting_system.cast_vote(decision_id, "agent1", VoteType.APPROVE, "Feature looks good to me")
    await voting_system.cast_vote(decision_id, "agent2", VoteType.APPROVE, "Tests are passing")
    await voting_system.cast_vote(decision_id, "agent3", VoteType.REJECT, "Needs more testing")
    
    # Get result
    result = await voting_system.get_result(decision_id)
    print(f"Decision result: {result}")
    
    # Get stats
    stats = voting_system.get_stats()
    print(f"Voting stats: {stats}")

if __name__ == "__main__":
    asyncio.run(main())