"""
HTTP request profiler for benchmarking LLM serving endpoints.
"""

import asyncio
import time
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import aiohttp
from .trace import TraceEvent


@dataclass
class RequestSpec:
    """Specification for a single HTTP request."""
    request_id: str
    model: str
    prompt: str
    max_tokens: int
    min_tokens: int
    temperature: float = 0.0
    ignore_eos: bool = True
    api_key: Optional[str] = None


class HTTPProfiler:
    """Profiles HTTP requests to an LLM server and generates Chrome traces."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self.trace_events: List[TraceEvent] = []
        self.request_stats: List[Dict[str, Any]] = []
        self.start_time: Optional[float] = None
        self.server_to_tid: Dict[str, int] = {}  # Map server names to thread IDs
        self.next_tid: int = 0  # Counter for assigning thread IDs

    def add_trace_event(self, event: TraceEvent):
        """Add a trace event to the collection."""
        self.trace_events.append(event)

    def get_server_tid(self, server_name: str) -> int:
        """Get or assign a thread ID for a server."""
        if server_name not in self.server_to_tid:
            self.server_to_tid[server_name] = self.next_tid
            self.next_tid += 1
        return self.server_to_tid[server_name]

    def _time_to_microseconds(self, timestamp: float) -> int:
        """Convert timestamp to microseconds relative to start time."""
        if self.start_time is None:
            return 0
        return int((timestamp - self.start_time) * 1_000_000)

    async def _send_request(
        self,
        session: aiohttp.ClientSession,
        request_spec: RequestSpec,
        request_index: int
    ) -> Dict[str, Any]:
        """Send a single HTTP request and track timing."""

        # Record arrival event
        arrival_time = time.perf_counter()
        arrival_ts = self._time_to_microseconds(arrival_time)

        self.add_trace_event(TraceEvent(
            name=f"Request arrived :: {request_index}",
            cat="request.lifecycle",
            ph="X",
            pid="profiler",
            tid=request_index,
            ts=arrival_ts,
            args={
                "request_id": request_spec.request_id,
                "model": request_spec.model,
                "prompt_length": len(request_spec.prompt),
                "max_tokens": request_spec.max_tokens,
            },
            dur=0
        ))

        # Prepare request payload
        payload = {
            "model": request_spec.model,
            "prompt": request_spec.prompt,
            "max_tokens": request_spec.max_tokens,
            "min_tokens": request_spec.min_tokens,
            "temperature": request_spec.temperature,
            "ignore_eos": request_spec.ignore_eos,
            "stream": True,
            "stream_options": {"include_usage": True}
        }

        headers = {
            "Content-Type": "application/json"
        }
        if request_spec.api_key:
            headers["Authorization"] = f"Bearer {request_spec.api_key}"

        request_start_time = time.perf_counter()
        first_token_time: Optional[float] = None
        last_token_time: Optional[float] = None
        token_timestamps: List[float] = []  # Track all token arrival times
        token_count: int = 0  # Count tokens/chunks as they arrive
        generated_text: str = ""  # Collect the full generated response
        completion_tokens: int = 0
        prompt_tokens: int = 0
        error: Optional[str] = None
        backend_server: str = "unknown"  # Track which server handled this

        try:
            async with session.post(
                f"{self.base_url}/v1/completions",
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=300)
            ) as response:
                if response.status == 200:
                    # Get which backend server handled this request
                    backend_server = response.headers.get('X-Backend-Server', 'unknown')
                    server_tid = self.get_server_tid(backend_server)
                    async for line in response.content:
                        if not line:
                            continue

                        line_str = line.decode('utf-8').strip()
                        if not line_str.startswith('data: '):
                            continue

                        data_str = line_str[6:]
                        if data_str == '[DONE]':
                            break

                        try:
                            chunk = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue

                        current_time = time.perf_counter()

                        # Check for token content
                        choices = chunk.get("choices", [])
                        if choices:
                            delta = choices[0]
                            text_piece = delta.get("text", "")
                            if text_piece:
                                generated_text += text_piece  # Collect generated text

                                # Calculate inter-token latency in milliseconds
                                if first_token_time is None:
                                    first_token_time = current_time
                                    inter_token_latency_ms = (current_time - request_start_time) * 1000  # TTFT in ms
                                else:
                                    inter_token_latency_ms = (current_time - last_token_time) * 1000

                                token_timestamps.append(current_time)
                                last_token_time = current_time

                                # Create per-token trace event
                                token_ts = self._time_to_microseconds(current_time)
                                self.add_trace_event(TraceEvent(
                                    name=f"Token {token_count}",
                                    cat="token.generation",
                                    ph="i",  # Instant event
                                    pid=backend_server,  # Use server name as process ID
                                    tid=server_tid,  # Use server-specific thread ID
                                    ts=token_ts,
                                    args={
                                        "request_id": request_spec.request_id,
                                        "token_index": token_count,
                                        "inter_token_latency_ms": round(inter_token_latency_ms, 3),
                                        "text_chunk": text_piece[:50],  # Preview of text
                                    },
                                    dur=0
                                ))

                                token_count += 1

                        # Check for usage information
                        usage = chunk.get("usage")
                        if usage:
                            if "completion_tokens" in usage:
                                completion_tokens = usage["completion_tokens"]
                            if "prompt_tokens" in usage:
                                prompt_tokens = usage["prompt_tokens"]
                else:
                    error = f"HTTP {response.status}: {await response.text()}"
        except asyncio.TimeoutError:
            error = "Request timeout"
        except Exception as e:
            error = str(e)

        request_end_time = time.perf_counter()

        # Calculate timings
        total_latency = request_end_time - request_start_time
        prefill_time = (first_token_time - request_start_time) if first_token_time else None  # TTFT
        decode_time = (last_token_time - first_token_time) if (last_token_time and first_token_time) else None

        # Calculate average inter-token latency
        avg_token_latency: Optional[float] = None
        if len(token_timestamps) > 1:
            inter_token_latencies = [
                token_timestamps[i] - token_timestamps[i-1]
                for i in range(1, len(token_timestamps))
            ]
            avg_token_latency = sum(inter_token_latencies) / len(inter_token_latencies)

        # Add completion trace event
        completion_ts = self._time_to_microseconds(request_end_time)
        duration_us = int(total_latency * 1_000_000)

        self.add_trace_event(TraceEvent(
            name=f"Request completed :: {request_index}",
            cat="request.completion",
            ph="X",
            pid="profiler",
            tid=request_index,
            ts=arrival_ts,
            args={
                "request_id": request_spec.request_id,
                "model": request_spec.model,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_latency_s": total_latency,
                "prefill_time_s": prefill_time,
                "decode_time_s": decode_time,
                "error": error,
            },
            dur=duration_us
        ))

        # Add prefill event if available
        if prefill_time is not None:
            prefill_start_ts = arrival_ts
            prefill_dur_us = int(prefill_time * 1_000_000)
            server_tid = self.get_server_tid(backend_server)

            self.add_trace_event(TraceEvent(
                name=f"Prefill :: {request_index}",
                cat="request.prefill",
                ph="X",
                pid=backend_server,  # Use server name as PID
                tid=server_tid,  # Use server-specific thread ID
                ts=prefill_start_ts,
                args={
                    "request_id": request_spec.request_id,
                    "prompt_tokens": prompt_tokens,
                    "prefill_time_s": prefill_time,
                    "prompt_text": request_spec.prompt[:200],  # Include prompt preview
                },
                dur=prefill_dur_us
            ))

        # Add decode event if available
        if decode_time is not None and first_token_time is not None:
            decode_start_ts = self._time_to_microseconds(first_token_time)
            decode_dur_us = int(decode_time * 1_000_000)
            server_tid = self.get_server_tid(backend_server)

            self.add_trace_event(TraceEvent(
                name=f"Decode :: {request_index}",
                cat="request.decode",
                ph="X",
                pid=backend_server,  # Use server name as PID
                tid=server_tid,  # Use server-specific thread ID
                ts=decode_start_ts,
                args={
                    "request_id": request_spec.request_id,
                    "completion_tokens": completion_tokens,
                    "decode_time_s": decode_time,
                    "tokens_per_second": completion_tokens / decode_time if decode_time > 0 else 0,
                    "generated_text": generated_text[:200],  # Include generated text preview
                },
                dur=decode_dur_us
            ))

        # Return stats
        stat = {
            "req_id": request_spec.request_id,
            "model": request_spec.model,
            "backend_server": backend_server,  # Which server handled this request
            "prompt_length": len(request_spec.prompt),
            "max_tokens": request_spec.max_tokens,
            "arrive_at": arrival_time - self.start_time if self.start_time else 0,
            "prefill_time": prefill_time,  # Time to first token (TTFT)
            "decode_time": decode_time,
            "avg_token_latency": avg_token_latency,  # Average inter-token latency
            "generated_tokens": completion_tokens,
            "prompt_tokens": prompt_tokens,
            "total_latency": total_latency,
            "prefill_finished_at": (first_token_time - self.start_time) if (first_token_time and self.start_time) else None,
            "generation_finished_at": (request_end_time - self.start_time) if self.start_time else None,
            "generated_text": generated_text,  # Save the actual response text
            "error": error,
        }

        return stat

    async def profile_requests(
        self,
        requests: List[RequestSpec],
        arrival_rate: Optional[float] = None
    ):
        """
        Profile a list of HTTP requests.

        Args:
            requests: List of request specifications
            arrival_rate: If specified, requests arrive at this rate (requests/second)
                         If None, all requests are sent immediately
        """
        self.start_time = time.perf_counter()

        # Create connector for session
        connector = aiohttp.TCPConnector(
            limit=len(requests) + 10,
            limit_per_host=len(requests) + 5,
            keepalive_timeout=300,
            enable_cleanup_closed=True
        )

        async with aiohttp.ClientSession(connector=connector) as session:
            if arrival_rate is None:
                # Send all requests immediately
                tasks = [
                    self._send_request(session, req, i)
                    for i, req in enumerate(requests)
                ]
                self.request_stats = await asyncio.gather(*tasks, return_exceptions=False)
            else:
                # Send requests at specified arrival rate
                interval = 1.0 / arrival_rate
                tasks = []

                for i, req in enumerate(requests):
                    task = asyncio.create_task(self._send_request(session, req, i))
                    tasks.append(task)

                    if i < len(requests) - 1:
                        await asyncio.sleep(interval)

                self.request_stats = await asyncio.gather(*tasks, return_exceptions=False)

    def export_trace(self, output_file: str):
        """Export trace events to Chrome trace format."""
        trace_data = {
            "traceEvents": [asdict(event) for event in self.trace_events]
        }

        with open(output_file, 'w') as f:
            json.dump(trace_data, f, indent=4)

    def export_stats(self, output_file: str):
        """Export request statistics."""
        stats_data = {
            "summary": self.request_stats,
            "failed": [s for s in self.request_stats if s.get("error") is not None],
            "config": {
                "base_url": self.base_url,
                "total_requests": len(self.request_stats),
            }
        }

        with open(output_file, 'w') as f:
            json.dump(stats_data, f, indent=4)
