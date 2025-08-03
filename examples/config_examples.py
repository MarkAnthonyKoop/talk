#!/usr/bin/env python3
"""
Talk Agent Framework - Configuration Examples
============================================

This file demonstrates different ways to configure the Talk Agent Framework,
including environment variables, config files, and programmatic settings.

Run specific examples with:
    python examples/config_examples.py --example=env
    python examples/config_examples.py --example=file
    python examples/config_examples.py --example=programmatic
    python examples/config_examples.py --example=all
"""

import argparse
import json
import os
import tempfile
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

# Import the agent components
from agent.agent import Agent
from agent.settings import Settings
from agent.llm_backends import get_backend, LLMBackend


def print_section(title):
    """Helper to print formatted section headers"""
    print(f"\n{'=' * 80}")
    print(f" {title}")
    print(f"{'=' * 80}\n")


def print_agent_settings(agent):
    """Print the key settings of an agent"""
    print(f"Agent ID: {agent.id}")
    print(f"Provider Type: {agent.settings.provider.type}")
    
    # Print provider-specific settings if available
    provider = agent.settings.provider
    if hasattr(provider, "model_name"):
        print(f"Model: {provider.model_name}")
    if hasattr(provider, "temperature"):
        print(f"Temperature: {provider.temperature}")
    if hasattr(provider, "max_tokens"):
        print(f"Max Tokens: {provider.max_tokens}")
    
    # Print conversation settings
    print(f"Log Path: {agent.settings.conversation.log_path}")
    print(f"Log Format: {agent.settings.conversation.log_format}")


def example_environment_config():
    """
    Demonstrates how to configure the agent using environment variables.
    
    The Talk framework checks for environment variables like:
    - OPENAI_API_KEY, CLAUDE_API_TOKEN, GEMINI_API_KEY, etc.
    - TALK_PROVIDER_TYPE can override the default provider
    - TALK_LOG_PATH can set the conversation log location
    """
    print_section("1. Environment-Based Configuration")
    
    # Save original environment to restore later
    original_env = os.environ.copy()
    
    try:
        # Set environment variables for configuration
        os.environ["TALK_PROVIDER_TYPE"] = "stub"  # Use stub to avoid needing real API keys
        os.environ["TALK_MODEL_NAME"] = "env-configured-model"
        os.environ["TALK_TEMPERATURE"] = "0.8"
        os.environ["TALK_MAX_TOKENS"] = "2000"
        os.environ["TALK_LOG_PATH"] = "./logs/env_config"
        
        # Create an agent - it will automatically use environment variables
        print("Creating agent with environment variables:")
        agent = Agent()
        print_agent_settings(agent)
        
        # Demonstrate a simple interaction
        print("\nTesting agent configured via environment:")
        response = agent.run("Hello! You were configured using environment variables.")
        print(f"Response: {response}")
        
    finally:
        # Restore original environment
        os.environ.clear()
        os.environ.update(original_env)


def example_file_config():
    """
    Demonstrates how to configure the agent using configuration files.
    
    The Talk framework can load settings from:
    - YAML files
    - JSON files
    - .env files (for environment variables)
    """
    print_section("2. File-Based Configuration")
    
    # Create a temporary YAML config file
    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w+", delete=False) as yaml_file:
        yaml_config = {
            "provider": {
                "type": "stub",
                "model_name": "yaml-configured-model",
                "temperature": 0.5,
                "max_tokens": 1500
            },
            "conversation": {
                "log_path": "./logs/yaml_config",
                "log_format": "jsonl"
            }
        }
        yaml.dump(yaml_config, yaml_file)
        yaml_path = yaml_file.name
    
    # Create a temporary JSON config file
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w+", delete=False) as json_file:
        json_config = {
            "provider": {
                "type": "stub",
                "model_name": "json-configured-model",
                "temperature": 0.3,
                "max_tokens": 3000
            },
            "conversation": {
                "log_path": "./logs/json_config",
                "log_format": "jsonl"
            }
        }
        json.dump(json_config, json_file)
        json_path = json_file.name
    
    try:
        # Load settings from YAML file
        print(f"Loading configuration from YAML file: {yaml_path}")
        yaml_settings = Settings.from_file(yaml_path)
        yaml_agent = Agent(settings=yaml_settings)
        print_agent_settings(yaml_agent)
        
        # Test the YAML-configured agent
        print("\nTesting agent configured via YAML:")
        response = yaml_agent.run("Hello! You were configured using a YAML file.")
        print(f"Response: {response}")
        
        # Load settings from JSON file
        print(f"\nLoading configuration from JSON file: {json_path}")
        json_settings = Settings.from_file(json_path)
        json_agent = Agent(settings=json_settings)
        print_agent_settings(json_agent)
        
        # Test the JSON-configured agent
        print("\nTesting agent configured via JSON:")
        response = json_agent.run("Hello! You were configured using a JSON file.")
        print(f"Response: {response}")
        
    finally:
        # Clean up temporary files
        for path in [yaml_path, json_path]:
            try:
                os.unlink(path)
            except:
                pass


def example_programmatic_config():
    """
    Demonstrates how to configure the agent programmatically using Python dictionaries.
    
    This is the most flexible approach and allows for runtime configuration changes.
    """
    print_section("3. Programmatic Configuration")
    
    # Create an agent with direct settings override
    basic_overrides = {
        "provider": {
            "type": "stub",
            "model_name": "programmatic-model",
            "temperature": 0.7,
            "max_tokens": 2500
        },
        "conversation": {
            "log_path": "./logs/programmatic",
            "log_format": "jsonl"
        }
    }
    
    print("Creating agent with basic programmatic configuration:")
    basic_agent = Agent(overrides=basic_overrides)
    print_agent_settings(basic_agent)
    
    # Test the basic programmatically configured agent
    print("\nTesting agent with basic programmatic configuration:")
    response = basic_agent.run("Hello! You were configured programmatically.")
    print(f"Response: {response}")
    
    # Create an agent with more complex configuration
    advanced_overrides = {
        "provider": {
            "type": "stub",
            "model_name": "advanced-programmatic-model",
            "temperature": 0.4,
            "max_tokens": 4000,
            "top_p": 0.95,
            "frequency_penalty": 0.5,
            "presence_penalty": 0.5,
        },
        "conversation": {
            "log_path": "./logs/advanced_programmatic",
            "log_format": "jsonl",
            "log_enabled": True,
            "system_prompt": "You are a helpful assistant specialized in programming."
        }
    }
    
    print("\nCreating agent with advanced programmatic configuration:")
    advanced_agent = Agent(overrides=advanced_overrides)
    print_agent_settings(advanced_agent)
    
    # Test the advanced programmatically configured agent
    print("\nTesting agent with advanced programmatic configuration:")
    response = advanced_agent.run("Hello! You were configured with advanced settings.")
    print(f"Response: {response}")


def example_llm_provider_configs():
    """
    Demonstrates configuration for different LLM providers.
    
    The Talk framework supports multiple LLM backends:
    - OpenAI (GPT models)
    - Anthropic (Claude models)
    - Google (Gemini models)
    - Perplexity
    - Fireworks
    - OpenRouter (gateway to multiple models)
    - Shell (for testing)
    """
    print_section("4. Different LLM Provider Configurations")
    
    # Configuration examples for different providers
    # Note: Using stub for all to avoid needing real API keys
    provider_configs = {
        "openai": {
            "provider": {
                "type": "stub",  # Would be "openai" in real usage
                "model_name": "gpt-4-turbo",
                "temperature": 0.7,
                "max_tokens": 2000,
                "top_p": 1.0,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0,
            }
        },
        "anthropic": {
            "provider": {
                "type": "stub",  # Would be "anthropic" in real usage
                "model_name": "claude-3-opus-20240229",
                "temperature": 0.5,
                "max_tokens": 4000,
            }
        },
        "gemini": {
            "provider": {
                "type": "stub",  # Would be "gemini" in real usage
                "model_name": "gemini-1.5-pro",
                "temperature": 0.9,
                "max_tokens": 8192,
                "top_p": 0.95,
            }
        },
        "openrouter": {
            "provider": {
                "type": "stub",  # Would be "openrouter" in real usage
                "model_name": "anthropic/claude-3-opus",
                "temperature": 0.7,
                "max_tokens": 4000,
                "route_prefix": "my-app",  # OpenRouter specific
            }
        }
    }
    
    # Create and test agents with different provider configurations
    for provider_name, config in provider_configs.items():
        print(f"\n--- {provider_name.upper()} Configuration Example ---")
        agent = Agent(overrides=config)
        print_agent_settings(agent)
        
        # Test the agent
        print(f"\nTesting {provider_name} configured agent:")
        response = agent.run(f"Hello! You're configured as a {provider_name} agent.")
        print(f"Response: {response}")


def example_custom_use_cases():
    """
    Demonstrates custom configurations for different use cases.
    
    Different applications might need different configurations:
    - Code generation (high precision, lower temperature)
    - Creative writing (higher temperature)
    - Question answering (balanced settings)
    - etc.
    """
    print_section("5. Custom Settings for Different Use Cases")
    
    use_case_configs = {
        "code_generation": {
            "provider": {
                "type": "stub",
                "model_name": "gpt-4-turbo",
                "temperature": 0.2,  # Lower temperature for more deterministic outputs
                "max_tokens": 4000,  # Longer context for complex code
                "top_p": 0.95,
            },
            "conversation": {
                "system_prompt": "You are a senior software engineer. Generate clean, efficient, and well-documented code.",
                "log_path": "./logs/code_generation",
            }
        },
        "creative_writing": {
            "provider": {
                "type": "stub",
                "model_name": "claude-3-opus",
                "temperature": 0.9,  # Higher temperature for more creative outputs
                "max_tokens": 8000,  # Very long context for stories
                "top_p": 1.0,
            },
            "conversation": {
                "system_prompt": "You are a creative writer with a unique voice. Create engaging and imaginative content.",
                "log_path": "./logs/creative_writing",
            }
        },
        "customer_support": {
            "provider": {
                "type": "stub",
                "model_name": "gpt-3.5-turbo",
                "temperature": 0.4,  # Balanced temperature
                "max_tokens": 1500,  # Shorter responses for customer support
            },
            "conversation": {
                "system_prompt": "You are a helpful customer support agent. Provide clear, concise, and accurate information.",
                "log_path": "./logs/customer_support",
            }
        },
        "data_analysis": {
            "provider": {
                "type": "stub",
                "model_name": "gemini-1.5-pro",
                "temperature": 0.1,  # Very low temperature for factual analysis
                "max_tokens": 4000,
            },
            "conversation": {
                "system_prompt": "You are a data analyst. Provide detailed, accurate analysis with clear explanations.",
                "log_path": "./logs/data_analysis",
            }
        }
    }
    
    # Create and test agents for different use cases
    for use_case, config in use_case_configs.items():
        print(f"\n--- {use_case.upper()} Use Case Configuration ---")
        agent = Agent(overrides=config)
        print_agent_settings(agent)
        
        # Print the system prompt
        if "conversation" in config and "system_prompt" in config["conversation"]:
            print(f"System Prompt: {config['conversation']['system_prompt']}")
        
        # Test the agent
        print(f"\nTesting {use_case} configured agent:")
        response = agent.run(f"Hello! You're configured for {use_case}.")
        print(f"Response: {response}")


def example_production_best_practices():
    """
    Demonstrates best practices for production deployment.
    
    Production environments need:
    - Secure API key handling
    - Fallback mechanisms
    - Logging configuration
    - Error handling
    - Performance optimization
    """
    print_section("6. Best Practices for Production Deployment")
    
    # Example of a production-ready configuration
    production_config = {
        "provider": {
            "type": "stub",  # In production, use your preferred provider
            "model_name": "gpt-4-turbo",
            "temperature": 0.5,
            "max_tokens": 2000,
            # Fallback configuration
            "fallback_providers": ["anthropic", "openai", "gemini"],
            "retry_attempts": 3,
            "timeout_seconds": 30,
        },
        "conversation": {
            "log_path": "/var/log/talk/conversations",
            "log_format": "jsonl",
            "log_enabled": True,
            "log_rotation": {
                "max_size_mb": 100,
                "backup_count": 10,
            },
        },
        "security": {
            "api_key_source": "environment",  # Don't hardcode API keys
            "pii_detection": True,  # Enable PII detection
            "content_filtering": "medium",  # Filter inappropriate content
        },
        "caching": {
            "enabled": True,
            "ttl_seconds": 3600,
            "max_size_mb": 1000,
        },
        "monitoring": {
            "enabled": True,
            "metrics_endpoint": "http://metrics.example.com/ingest",
            "log_level": "INFO",
        }
    }
    
    print("Production-Ready Configuration Best Practices:")
    for category, settings in production_config.items():
        print(f"\n{category.upper()}:")
        for key, value in settings.items():
            print(f"  - {key}: {value}")
    
    print("\nKey Production Deployment Recommendations:")
    print("1. Use environment variables or a secure vault for API keys")
    print("2. Implement fallback mechanisms between different providers")
    print("3. Set up comprehensive logging and monitoring")
    print("4. Configure appropriate timeouts and retry logic")
    print("5. Use a caching layer to reduce API costs and improve performance")
    print("6. Implement content filtering for safety")
    print("7. Set up automated testing for your agent workflows")
    print("8. Use a CI/CD pipeline for deployment")
    
    # Example of creating a production agent with fallback
    print("\nExample of creating a production agent with fallback mechanisms:")
    
    def create_production_agent_with_fallback():
        """Example function demonstrating production agent creation with fallbacks"""
        primary_config = {"provider": {"type": "stub", "model_name": "gpt-4"}}
        fallback_configs = [
            {"provider": {"type": "stub", "model_name": "claude-3"}},
            {"provider": {"type": "stub", "model_name": "gemini-1.5"}},
        ]
        
        # Try to create the primary agent
        try:
            agent = Agent(overrides=primary_config)
            print("Successfully created primary agent")
            return agent
        except Exception as e:
            print(f"Primary agent creation failed: {e}")
            
            # Try fallbacks
            for i, fallback_config in enumerate(fallback_configs):
                try:
                    agent = Agent(overrides=fallback_config)
                    print(f"Successfully created fallback agent #{i+1}")
                    return agent
                except Exception as e:
                    print(f"Fallback #{i+1} failed: {e}")
            
            # Last resort - use shell backend
            print("All fallbacks failed, using shell backend")
            return Agent(overrides={"provider": {"type": "shell"}})
    
    # Create a production agent
    production_agent = create_production_agent_with_fallback()
    print_agent_settings(production_agent)


def main():
    """Run the configuration examples"""
    parser = argparse.ArgumentParser(description="Talk Agent Framework Configuration Examples")
    parser.add_argument("--example", choices=["env", "file", "programmatic", "providers", "usecases", "production", "all"],
                        default="all", help="Which configuration example to run")
    args = parser.parse_args()
    
    # Create logs directory
    os.makedirs("./logs", exist_ok=True)
    
    # Run the selected example(s)
    if args.example in ["env", "all"]:
        example_environment_config()
    
    if args.example in ["file", "all"]:
        example_file_config()
    
    if args.example in ["programmatic", "all"]:
        example_programmatic_config()
    
    if args.example in ["providers", "all"]:
        example_llm_provider_configs()
    
    if args.example in ["usecases", "all"]:
        example_custom_use_cases()
    
    if args.example in ["production", "all"]:
        example_production_best_practices()


if __name__ == "__main__":
    main()

