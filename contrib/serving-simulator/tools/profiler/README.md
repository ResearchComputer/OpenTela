# OpenAI Load Balancer

A load balancing server that distributes requests across multiple OpenAI-compatible API servers.

## Features

- **Multiple Scheduling Algorithms**:
  - **Round Robin**: Distributes requests evenly across all servers in rotation
  - **Random**: Randomly selects a server for each request
  - **Least Connections**: Routes to the server with fewest active connections (placeholder implementation)

- **OpenAI API Compatibility**: Supports standard OpenAI endpoints like `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, etc.

- **Health Checking**: Automatic periodic health checks to detect and route around unhealthy backends

- **Connection Tracking**: Tracks active connections per backend server

- **Configurable**: YAML-based configuration for easy customization

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

Edit `config.yaml` to configure your backend servers and scheduler:

```yaml
backend_servers:
  - url: "http://localhost:8001"
    name: "server-1"
    weight: 1
  - url: "http://localhost:8002"
    name: "server-2"
    weight: 1

# Scheduler options: round_robin, random, least_connections
scheduler_type: "round_robin"

host: "0.0.0.0"
port: 8000

health_check:
  enabled: true
  interval_seconds: 30
  timeout_seconds: 5
  endpoint: "/health"

request_timeout: 300
```

## Usage

### Start the load balancer:

```bash
python server.py
```

The server will start on `http://0.0.0.0:8000` by default.

### Check health status:

```bash
curl http://localhost:8000/health
```

### Make requests:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## Architecture

### Files

- **server.py**: Main FastAPI application and load balancer implementation
- **schedulers.py**: Scheduler implementations (Round Robin, Random, Least Connections)
- **config.yaml**: Configuration file for backends and settings
- **requirements.txt**: Python dependencies

### Schedulers

All schedulers implement the base `Scheduler` interface:

```python
class Scheduler(ABC):
    @abstractmethod
    def select_backend(self) -> Optional[Backend]:
        """Select a backend server for the next request."""
        pass
```

You can easily add new scheduling algorithms by extending the `Scheduler` class.

## How It Works

1. The load balancer receives an OpenAI API request
2. The scheduler selects a healthy backend server based on the configured algorithm
3. The request is proxied to the selected backend
4. The response (including streaming responses) is returned to the client
5. Health checks run periodically in the background to monitor backend availability

## Adding New Schedulers

To add a new scheduler:

1. Create a new class in `schedulers.py` that extends `Scheduler`
2. Implement the `select_backend()` method
3. Add it to the `create_scheduler()` factory function
4. Update the config file to use your new scheduler

Example:

```python
class WeightedRandomScheduler(Scheduler):
    def select_backend(self) -> Optional[Backend]:
        healthy_backends = self.get_healthy_backends()
        if not healthy_backends:
            return None

        weights = [b.weight for b in healthy_backends]
        return random.choices(healthy_backends, weights=weights, k=1)[0]
```
