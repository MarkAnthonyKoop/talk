#
# Changes from Original & Architectural Theory:
# This is a new file created to solve a specific, critical problem.
#
# Architectural Theory:
# The primary challenge identified early on was the conflict between `asyncio.run()`
# and environments that already manage an event loop (like Jupyter or pytest-asyncio).
# The chosen architecture avoids this by making the core agent/plan execution
# synchronous and providing this "compatibility shim."
#
# `run_sync_in_thread` acts as a safeguard. When called, it checks if an asyncio
# event loop is active.
# - If NOT, it runs the given synchronous function directly.
# - If YES, it moves the execution of the synchronous function to a new, separate
#   worker thread. This prevents the synchronous, potentially blocking function
#   from freezing the event loop of the calling environment.
#
# The `ShellAgent.run` method uses this shim to wrap its call to `PlanRunner.run`,
# making it safe to call the agent from any environment without causing conflicts.
#

"""
Provides a compatibility shim to safely run synchronous code from an async context.
"""
from __future__ import annotations
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, TypeVar

# A generic TypeVar to preserve type hints of the wrapped function.
_T = TypeVar("_T")

def run_sync_in_thread(fn: Callable[[], _T]) -> _T:
    """
    Executes a synchronous function `fn` safely.

    If an asyncio event loop is already running in the current thread, `fn` is
    executed in a new temporary worker thread to avoid blocking the event loop.
    This is essential for compatibility with environments like Jupyter notebooks
    or async test frameworks.

    If no event loop is running, `fn` is called directly for maximum efficiency.

    Args:
        fn: The synchronous, parameter-less function to execute.

    Returns:
        The result of the function call `fn()`.
    """
    try:
        # Check if an event loop is active in the current thread.
        asyncio.get_running_loop()
    except RuntimeError:
        # No loop is running. It's safe to call the function directly.
        return fn()

    # An event loop IS running. To avoid blocking it, execute `fn` in a
    # separate thread and wait for its result.
    # We use a one-shot executor to create and tear down the thread for this call.
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(fn)
        return fut.result()
