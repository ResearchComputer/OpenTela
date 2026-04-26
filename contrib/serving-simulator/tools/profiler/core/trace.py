"""
Chrome trace event generation for HTTP request profiling.
"""

from dataclasses import dataclass
from typing import Optional, Union


@dataclass
class TraceEvent:
    """A single trace event for Chrome tracing format."""
    name: str
    cat: str
    ph: str
    pid: Union[int, str]
    tid: Union[int, str]
    ts: int  # in microseconds
    args: Optional[dict] = None
    dur: Optional[int] = None
