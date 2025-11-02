"""Core utilities for inferai template."""

def greet(name: str) -> str:
    """Return a greeting for the given name.

    Raises ValueError if name is empty.
    """
    if not name:
        raise ValueError("name must be non-empty")
    return f"Hello, {name}!"
