#!/usr/bin/env python3
"""
Setup script for YouTube Research CLI
"""

from setuptools import setup

setup(
    name="youtube-research-cli",
    version="1.0.0",
    description="YouTube Research CLI - Analyze viewing history with transcript fetching and web research",
    py_modules=[
        "youtube_research_cli",
    ],
    install_requires=[
        "youtube-transcript-api",
        "duckduckgo-search",
        "beautifulsoup4",
        "requests",
    ],
    entry_points={
        "console_scripts": [
            "youtube-research=youtube_research_cli:main",
        ],
    },
    python_requires=">=3.8",
)