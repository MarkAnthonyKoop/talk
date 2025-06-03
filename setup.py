#!/usr/bin/env python3
from setuptools import setup, find_packages
import os

# Read the contents of README.md
with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

# Core requirements
REQUIRES = [
    "pydantic>=2.0.0",
    "pydantic-settings>=2.0.0",
    "typing-extensions>=4.5.0",
]

# Optional dependencies
EXTRAS = {
    # LLM backend dependencies
    "openai": ["openai>=1.0.0"],
    "anthropic": ["anthropic>=0.5.0"],
    "gemini": ["google-generativeai>=0.3.0"],
    "all": [
        "openai>=1.0.0",
        "anthropic>=0.5.0",
        "google-generativeai>=0.3.0",
        "python-dotenv>=1.0.0",
    ],
    # Development dependencies
    "dev": [
        "pytest>=7.0.0",
        "black>=23.0.0",
        "ruff>=0.1.0",
        "mypy>=1.0.0",
    ],
}

setup(
    name="talk",
    version="0.1.0",
    description="Agent framework for building autonomous, LLM-powered workflows",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Mark Anthony Koop",
    author_email="mark@example.com",  # Placeholder
    url="https://github.com/MarkAnthonyKoop/talk",
    packages=find_packages(exclude=["tests*"]),
    python_requires=">=3.9",
    install_requires=REQUIRES,
    extras_require=EXTRAS,
    entry_points={
        "console_scripts": [
            "talk=talk.talk:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="llm, agent, ai, chat, framework, openai, anthropic, gemini",
)
