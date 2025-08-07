"""
Policies for load balancing and failover management.
"""

import logging
from enum import Enum
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta

log = logging.getLogger(__name__)


class LoadBalancingPolicy(Enum):
    """Load balancing policies"""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    RANDOM = "random"
    LEAST_RESPONSE_TIME = "least_response_time"
    HASH_BASED = "hash_based"


class FailoverPolicy(Enum):
    """Failover policies"""
    IMMEDIATE = "immediate"
    DELAYED = "delayed"
    GRADUAL = "gradual"
    MANUAL = "manual"


@dataclass
class PolicyConfig:
    """Configuration for policies"""
    load_balancing: LoadBalancingPolicy = LoadBalancingPolicy.ROUND_ROBIN
    failover: FailoverPolicy = FailoverPolicy.IMMEDIATE
    health_check_interval: int = 30
    failover_threshold: int = 3
    recovery_timeout: int = 60
    weights: Dict[str, float] = field(default_factory=dict)