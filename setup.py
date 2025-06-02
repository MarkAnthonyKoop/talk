#!/usr/bin/env python3
"""
Setup script for Talk - Autonomous CLI for continuous, unattended improvement of any codebase.
"""

import os
import re
from setuptools import setup, find_packages

# Get the version from talk/__init__.py
with open(os.path.join("talk", "__init__.py"), "r", encoding="utf-8") as f:
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M)
    if version_match:
        version = version_match.group(1)
    else:
        raise RuntimeError("Unable to find version string in talk/__init__.py")

# Get long description from README.md
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

# Core dependencies required for basic functionality
install_requires = [
    "typer>=0.9.0",
    "rich>=13.0.0",
    "libcst>=1.0.0",
]

# Optional dependencies for enhanced functionality
extras_require = {
    "dev": [
        "pytest>=7.0.0",
        "black>=23.0.0",
        "ruff>=0.1.0",
        "pre-commit>=3.0.0",
    ],
    "analysis": [
        "radon>=6.0.0",
        "networkx>=3.0.0",
    ],
}

# All optional dependencies combined
extras_require["all"] = list(set(sum(extras_require.values(), [])))

setup(
    name="talk",
    version=version,
    author="Mark Anthony Koop",
    author_email="mark@example.com",  # Replace with actual email if available
    description="Autonomous CLI for continuous, unattended improvement of any codebase",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MarkAnthonyKoop/talk",
    packages=find_packages(include=["talk", "talk.*"]),
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=install_requires,
    extras_require=extras_require,
    entry_points={
        "console_scripts": [
            "talk=talk.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Software Development :: Quality Assurance",
        "Topic :: Software Development :: Code Generators",
    ],
    keywords="code, automation, refactoring, improvement, cli, development",
    project_urls={
        "Bug Reports": "https://github.com/MarkAnthonyKoop/talk/issues",
        "Source": "https://github.com/MarkAnthonyKoop/talk",
    },
)
