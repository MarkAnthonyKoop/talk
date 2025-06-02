#!/usr/bin/env python3
"""
Talk CLI - Autonomous codebase improvement tool.

This module provides the command-line interface for the Talk tool,
which analyzes, plans, and rewrites source code without supervision.
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn

# Setup application
app = typer.Typer(
    name="talk",
    help="Autonomous CLI for continuous, unattended improvement of any codebase.",
    add_completion=True,
)

# Setup console for rich output
console = Console()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, console=console)]
)
logger = logging.getLogger("talk")


def setup_logging(verbose: bool):
    """Configure logging level based on verbosity."""
    if verbose:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    else:
        logger.setLevel(logging.INFO)


def validate_path(path: Path) -> Path:
    """Validate that the provided path exists."""
    if not path.exists():
        console.print(f"[bold red]Error:[/] Path '{path}' does not exist.")
        raise typer.Exit(code=1)
    return path


@app.callback()
def callback(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
):
    """
    Talk - Autonomous CLI for continuous, unattended improvement of any codebase.
    
    Point it at a repository and Talk will analyze, plan, and rewrite your code
    without supervision, applying best practices and improvements automatically.
    """
    setup_logging(verbose)


@app.command()
def run(
    path: Path = typer.Argument(
        Path("."),
        help="Path to the codebase to improve",
        callback=validate_path,
    ),
    dry: bool = typer.Option(
        False,
        "--dry",
        "-d",
        help="Dry run - prints proposed patches without applying",
    ),
    branch: Optional[str] = typer.Option(
        None,
        "--branch",
        "-b",
        help="Target a specific branch",
    ),
    include: Optional[List[str]] = typer.Option(
        None,
        "--include",
        "-i",
        help="Restrict to specific files or globs",
    ),
    iterations: int = typer.Option(
        5,
        "--iterations",
        "-n",
        help="Maximum number of improvement iterations",
    ),
):
    """
    Iterate on codebase until exhausted or stopped.
    
    This command analyzes the codebase, plans improvements, and applies them
    in an iterative process until no further actionable improvements remain
    or the maximum number of iterations is reached.
    """
    path = path.resolve()
    logger.info(f"Starting Talk on codebase: {path}")
    
    if dry:
        logger.info("Running in dry-run mode - changes will not be applied")
    
    if branch:
        logger.info(f"Targeting branch: {branch}")
    
    if include:
        logger.info(f"Including only: {', '.join(include)}")
    
    logger.info(f"Maximum iterations: {iterations}")
    
    # This would be where we'd call into the actual implementation
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Analyzing codebase...", total=None)
        # Placeholder for actual implementation
        progress.update(task, description="Planning improvements...")
        progress.update(task, description="Applying changes...")
        progress.update(task, description="Running tests...")
        progress.update(task, description="Committing changes...")
    
    console.print("[bold green]✓[/] Codebase improvement completed successfully!")
    console.print("Run [bold]talk undo[/] to revert the most recent changes if needed.")


@app.command()
def plan(
    path: Path = typer.Argument(
        Path("."),
        help="Path to the codebase to analyze",
        callback=validate_path,
    ),
    include: Optional[List[str]] = typer.Option(
        None,
        "--include",
        "-i",
        help="Restrict to specific files or globs",
    ),
):
    """
    Generate and display the next set of improvements without applying them.
    
    This command analyzes the codebase and generates a plan of improvements
    that would be applied by the 'run' command, but does not actually make
    any changes to the code.
    """
    path = path.resolve()
    logger.info(f"Planning improvements for codebase: {path}")
    
    if include:
        logger.info(f"Including only: {', '.join(include)}")
    
    # Placeholder for actual implementation
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Analyzing codebase...", total=None)
        # Placeholder for actual implementation
        progress.update(task, description="Generating improvement plan...")
    
    # Example output
    console.print("\n[bold]Proposed Improvements:[/]")
    console.print("1. [yellow]Refactor:[/] Extract duplicate code in `src/utils.py`")
    console.print("2. [yellow]Performance:[/] Replace list comprehension with generator in `src/data.py`")
    console.print("3. [yellow]Testing:[/] Add unit tests for `src/api.py`")
    
    console.print("\nRun [bold]talk run .[/] to apply these improvements.")


@app.command()
def undo():
    """
    Revert the most recent Talk-generated commit.
    
    This command identifies and reverts the most recent commit made by Talk,
    effectively undoing the last set of automated changes.
    """
    logger.info("Reverting the most recent Talk-generated commit")
    
    # Placeholder for actual implementation
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Identifying last Talk commit...", total=None)
        # Placeholder for actual implementation
        progress.update(task, description="Reverting changes...")
    
    console.print("[bold green]✓[/] Successfully reverted the most recent Talk-generated commit.")


@app.command()
def config(
    edit: bool = typer.Option(
        False,
        "--edit",
        "-e",
        help="Open the configuration file in your default editor",
    ),
    reset: bool = typer.Option(
        False,
        "--reset",
        "-r",
        help="Reset the configuration to default values",
    ),
):
    """
    Edit or print current configuration file.
    
    This command allows you to view, edit, or reset the Talk configuration file
    that controls the behavior of the tool.
    """
    config_path = Path(".talk.yaml")
    
    if reset:
        logger.info("Resetting configuration to default values")
        # Placeholder for actual implementation
        console.print("[bold green]✓[/] Configuration reset to default values.")
        return
    
    if edit:
        logger.info("Opening configuration file in editor")
        # Placeholder for actual implementation
        console.print("[bold green]✓[/] Configuration file opened in editor.")
        return
    
    logger.info("Displaying current configuration")
    
    if not config_path.exists():
        console.print("[yellow]No configuration file found.[/] Using default settings.")
        console.print("Run [bold]talk config --edit[/] to create and edit a configuration file.")
        return
    
    # Example output
    console.print("\n[bold]Current Configuration:[/]")
    console.print("```yaml")
    console.print("strategy: conservative")
    console.print("max_iterations: 5")
    console.print("lint: true")
    console.print("tests:")
    console.print("  enabled: true")
    console.print("  command: pytest")
    console.print("backup: git")
    console.print("```")


@app.command()
def plugins(
    list_plugins: bool = typer.Option(
        True,
        "--list",
        "-l",
        help="List installed plugins",
    ),
    add: Optional[str] = typer.Option(
        None,
        "--add",
        "-a",
        help="Add a plugin from a path or URL",
    ),
    remove: Optional[str] = typer.Option(
        None,
        "--remove",
        "-r",
        help="Remove an installed plugin",
    ),
):
    """
    List, add, or remove strategy plugins.
    
    This command allows you to manage the plugins that extend Talk's
    functionality with additional analysis, planning, or execution strategies.
    """
    if add:
        logger.info(f"Adding plugin from: {add}")
        # Placeholder for actual implementation
        console.print(f"[bold green]✓[/] Plugin added successfully: {add}")
        return
    
    if remove:
        logger.info(f"Removing plugin: {remove}")
        # Placeholder for actual implementation
        console.print(f"[bold green]✓[/] Plugin removed successfully: {remove}")
        return
    
    if list_plugins:
        logger.info("Listing installed plugins")
        # Example output
        console.print("\n[bold]Installed Plugins:[/]")
        console.print("1. [blue]core[/] - Core improvement strategies (built-in)")
        console.print("2. [blue]python[/] - Python-specific improvements (built-in)")
        console.print("3. [blue]javascript[/] - JavaScript/TypeScript improvements (built-in)")


def main():
    """Entry point for the CLI application."""
    try:
        app()
    except Exception as e:
        logger.exception("An unexpected error occurred")
        console.print(f"[bold red]Error:[/] {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
