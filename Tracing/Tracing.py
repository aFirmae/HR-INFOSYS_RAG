import os
from langsmith import traceable
from langchain_core.tracers.context import tracing_v2_enabled
from contextlib import nullcontext

# Auto-setup (only runs if not already set via .env)
os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
os.environ.setdefault("LANGCHAIN_PROJECT", "productivity-rag")

def is_tracing_enabled() -> bool:
    return os.environ.get("LANGCHAIN_TRACING_V2", "false").lower() == "true"

def trace_block():
    """
    Use this as a context manager to trace specific blocks of code.
    """
    return tracing_v2_enabled() if is_tracing_enabled() else nullcontext()

def traceable_fn(name: str = None):
    """
    Decorator to trace a specific function.
    Optional: Pass a name to override function name in LangSmith dashboard.
    """
    return traceable(name=name)
