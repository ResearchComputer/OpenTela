from abc import ABC, abstractmethod
from typing import List, Optional
import random
import threading


class Backend:
    """Represents a backend server."""

    def __init__(self, url: str, name: str, weight: int = 1):
        self.url = url
        self.name = name
        self.weight = weight
        self.healthy = True
        self.active_connections = 0
        self._lock = threading.Lock()

    def increment_connections(self):
        """Increment active connection count."""
        with self._lock:
            self.active_connections += 1

    def decrement_connections(self):
        """Decrement active connection count."""
        with self._lock:
            self.active_connections = max(0, self.active_connections - 1)

    def __repr__(self):
        return f"Backend(name={self.name}, url={self.url}, healthy={self.healthy}, connections={self.active_connections})"


class Scheduler(ABC):
    """Base scheduler interface."""

    def __init__(self, backends: List[Backend]):
        self.backends = backends
        self._lock = threading.Lock()

    @abstractmethod
    def select_backend(self) -> Optional[Backend]:
        """Select a backend server for the next request."""
        pass

    def get_healthy_backends(self) -> List[Backend]:
        """Get list of healthy backends."""
        return [b for b in self.backends if b.healthy]

    def mark_backend_unhealthy(self, backend: Backend):
        """Mark a backend as unhealthy."""
        backend.healthy = False

    def mark_backend_healthy(self, backend: Backend):
        """Mark a backend as healthy."""
        backend.healthy = True


class RoundRobinScheduler(Scheduler):
    """Round Robin scheduler - distributes requests evenly across all servers."""

    def __init__(self, backends: List[Backend]):
        super().__init__(backends)
        self.current_index = 0

    def select_backend(self) -> Optional[Backend]:
        """Select next backend in round-robin fashion."""
        healthy_backends = self.get_healthy_backends()

        if not healthy_backends:
            return None

        with self._lock:
            backend = healthy_backends[self.current_index % len(healthy_backends)]
            self.current_index += 1
            return backend


class RandomScheduler(Scheduler):
    """Random scheduler - randomly selects a server for each request."""

    def select_backend(self) -> Optional[Backend]:
        """Randomly select a healthy backend."""
        healthy_backends = self.get_healthy_backends()

        if not healthy_backends:
            return None

        return random.choice(healthy_backends)


class LeastConnectionsScheduler(Scheduler):
    """
    Least Connections scheduler - routes to server with fewest active connections.
    This is a placeholder implementation for future use.
    """

    def select_backend(self) -> Optional[Backend]:
        """Select backend with least active connections."""
        healthy_backends = self.get_healthy_backends()

        if not healthy_backends:
            return None

        # Sort by active connections and return the one with least connections
        return min(healthy_backends, key=lambda b: b.active_connections)


def create_scheduler(scheduler_type: str, backends: List[Backend]) -> Scheduler:
    """Factory function to create scheduler based on type."""
    schedulers = {
        'round_robin': RoundRobinScheduler,
        'random': RandomScheduler,
        'least_connections': LeastConnectionsScheduler,
    }

    scheduler_class = schedulers.get(scheduler_type.lower())
    if not scheduler_class:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}. Available: {list(schedulers.keys())}")

    return scheduler_class(backends)
