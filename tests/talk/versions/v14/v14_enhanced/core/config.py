"""Configuration management for core components."""

import os
import yaml
from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class CoreConfig:
    """Configuration settings for core components."""
    
    # Redis settings
    redis_host: str = os.getenv("REDIS_HOST", "localhost")
    redis_port: int = int(os.getenv("REDIS_PORT", "6379"))
    redis_password: Optional[str] = os.getenv("REDIS_PASSWORD")
    redis_ssl: bool = bool(os.getenv("REDIS_SSL", "False"))
    
    # Rate limiting
    rate_limit_calls: int = int(os.getenv("RATE_LIMIT_CALLS", "100"))
    rate_limit_period: int = int(os.getenv("RATE_LIMIT_PERIOD", "60"))
    
    # Threading
    max_workers: int = int(os.getenv("MAX_WORKERS", "4"))
    
    # Retry settings
    max_retries: int = int(os.getenv("MAX_RETRIES", "3"))
    retry_delay: float = float(os.getenv("RETRY_DELAY", "0.5"))
    
    # Security
    encryption_key: Optional[str] = os.getenv("ENCRYPTION_KEY")
    api_key: Optional[str] = os.getenv("API_KEY")
    
    # Monitoring
    enable_metrics: bool = bool(os.getenv("ENABLE_METRICS", "True"))
    metrics_port: int = int(os.getenv("METRICS_PORT", "8000"))
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "CoreConfig":
        """Create config from dictionary."""
        return cls(**config_dict)
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "CoreConfig":
        """Create config from YAML file."""
        with open(yaml_path) as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    def validate(self) -> None:
        """Validate configuration settings."""
        if self.rate_limit_calls <= 0:
            raise ValueError("Rate limit calls must be positive")
        if self.rate_limit_period <= 0:
            raise ValueError("Rate limit period must be positive")
        if self.max_workers <= 0:
            raise ValueError("Max workers must be positive")