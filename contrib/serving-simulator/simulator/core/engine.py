from collections import deque
from transformers import AutoConfig
from typing import Any, Deque, Dict, List, Optional, Set
import numpy as np

from simulator.configs.hardware import hardware_params
from .model_analyzer import ModelAnalyzer
from .memory import MemoryPlanner
from .config import ParallelConfig
from .request import GenerationRequest, REQ_STATUS
from .trace import TraceEvent
from .events import Event, EventType, EventPriority


class Batch:
    """Represents a batch of requests being processed together."""

    def __init__(self, batch_id: str, requests: List[GenerationRequest]):
        self.batch_id = batch_id
        self.requests = requests
        self.created_at = 0.0
        self.status = "forming"  # forming, prefill, decode, completed
        self.memory_usage = 0.0

    def add_request(self, request: GenerationRequest):
        """Add a request to the batch."""
        if request not in self.requests:
            self.requests.append(request)

    def remove_request(self, request: GenerationRequest):
        """Remove a completed request from the batch."""
        if request in self.requests:
            self.requests.remove(request)

        # Mark batch as completed if empty
        if len(self.requests) == 0:
            self.status = "completed"

    def is_empty(self) -> bool:
        """Check if batch is empty (all requests completed)."""
        return len(self.requests) == 0

    def get_batch_size(self) -> int:
        """Get current batch size."""
        return len(self.requests)


class ServingEngine:
    """Event-driven serving engine for local node simulation."""

    def __init__(
            self,
            engine_id: str,
            model_id: str,
            model_instance: Any,
            hardware: str,
            parallel_config=None,
            max_batch_size: int = 8
        ):
        """
        Initialize the serving engine with model and hardware configurations.

        Args:
            engine_id: Unique identifier for this engine
            model_id: Identifier for the model to be served
            model_instance: Configuration object for the model
            hardware: Hardware type identifier
            parallel_config: Optional parallel configuration for the engine
            max_batch_size: Maximum batch size for processing
        """
        self.engine_id = engine_id
        self.model_id = model_id
        self.model_params = AutoConfig.from_pretrained(model_id)
        self.model_instance = model_instance
        self.hardware = hardware
        self.parallel_config = parallel_config
        self.max_batch_size = max_batch_size

        # Hardware specifications
        self.hardware_spec = hardware_params[hardware]
        self.gpu_memory_capacity = self.hardware_spec['vmemory']
        self.memory_bandwidth = self.hardware_spec['bandwidth']
        self.compute_throughput = self.hardware_spec['FP16']

        # Memory planning - simplified to avoid MemoryPlanner complexity
        self.memory_planner = MemoryPlanner(
            model_params = self.model_params,
            model_config = model_instance,
            w_bit=16,
            a_bit=16,
            kv_bit=16,
            hardware_params=self.hardware_spec,
            parallel_config=parallel_config,
            block_size=16,
        )

        # Request management
        self.request_queue: Deque[GenerationRequest] = deque()
        self.active_requests: Set[GenerationRequest] = set()
        self.completed_requests: List[GenerationRequest] = []

        # Separate queues for different processing stages
        self.prefill_queue: Deque[GenerationRequest] = deque()  # Requests waiting for prefill
        self.decode_ready_requests: Set[GenerationRequest] = set()  # Requests ready for decode batching

        # Batch management (for decode phase only)
        self.current_decode_batch: Optional[Batch] = None
        self.batch_counter = 0

        # Current prefill request (processed individually)
        self.current_prefill_request: Optional[GenerationRequest] = None

        # Tracing
        self.trace_events: List[TraceEvent] = []
        self.current_time = 0.0  # Last wall-clock time passed in from cluster
        self.time_cursor = 0.0   # Engine's internal notion of GPU time progression

        # Model analyzer for accurate timing
        # Use appropriate config based on model_id
        analyzer_config = model_instance
        if analyzer_config is None:
            # Try to provide appropriate config based on model_id
            if "llama" in self.model_id.lower() or "Llama-2" in self.model_id:
                try:
                    from simulator.configs.models import llama
                    analyzer_config = llama
                except ImportError:
                    analyzer_config = None
            else:
                analyzer_config = None

        self.model_analyzer = ModelAnalyzer(
            model_id=self.model_id,
            config=analyzer_config,
            hardware=self.hardware
        )

        # Model loading state
        self.model_loaded = False
        self.model_memory_usage = 0.0
        self.kv_cache_memory = 0.0
        self.kv_cache_capacity = 0.0

        # Performance tracking
        self.statistics: Dict[str, Any] = {
            'requests_processed': 0,
            'total_tokens_generated': 0,
            'total_prefill_time': 0.0,
            'total_decode_time': 0.0,
            'memory_peak': 0.0,
            'batch_sizes': []
        }

        # Event callback (set by cluster manager)
        self.event_callback = None

    def set_event_callback(self, callback):
        """Set callback function to emit events to cluster manager."""
        self.event_callback = callback

    def emit_event(self, event_type: EventType, data: Dict[str, Any],
                   priority: EventPriority = EventPriority.MEDIUM):
        """Emit an event to the cluster manager."""
        if self.event_callback:
            event = Event(
                timestamp=0.0,  # Will be set by cluster manager
                event_type=event_type,
                target=self.engine_id,
                data=data,
                priority=priority
            )
            self.event_callback(event)

    def _refresh_memory_usage(self):
        """Update cached KV memory usage and peak tracking."""
        try:
            kv_memory = self.memory_planner.get_allocated_kv_memory_per_shard()
            kv_capacity = self.memory_planner.get_total_kv_memory_capacity_per_shard()
        except Exception:
            kv_memory = 0.0
            kv_capacity = 0.0

        self.kv_cache_memory = kv_memory
        self.kv_cache_capacity = kv_capacity
        total_used = self.model_memory_usage + self.kv_cache_memory
        self.statistics['memory_peak'] = max(self.statistics['memory_peak'], total_used)

    def _get_memory_snapshot(self) -> Dict[str, Any]:
        """Return current memory stats formatted for trace annotations."""
        info = self.get_memory_info()
        kv_blocks_in_use = int(info.get('kv_blocks_used', 0))
        kv_blocks_capacity = int(info.get('kv_blocks_capacity', 0))
        kv_blocks_remaining = max(kv_blocks_capacity - kv_blocks_in_use, 0)
        return {
            'memory_total_bytes': int(info.get('total', 0.0)),
            'memory_used_bytes': int(info.get('used', 0.0)),
            'memory_available_bytes': int(info.get('available', 0.0)),
            'memory_utilization': round(info.get('utilization', 0.0), 6),
            'model_weights_bytes': int(info.get('model_weights', 0.0)),
            'kv_cache_bytes': int(info.get('kv_cache', 0.0)),
            'kv_cache_capacity_bytes': int(info.get('kv_cache_capacity', 0.0)),
            'kv_blocks_in_use_total': kv_blocks_in_use,
            'kv_blocks_remaining': kv_blocks_remaining,
            'kv_blocks_capacity': kv_blocks_capacity,
        }

    def load_model(self) -> bool:
        """Load the model into GPU memory."""
        if self.model_loaded:
            return True

        try:
            model_memory = self.memory_planner.get_weights_memory_per_shard()
        except Exception:
            # Fallback to simplified estimate if memory planner is unavailable
            hidden_size = getattr(self.model_params, 'hidden_size', 4096)
            num_layers = getattr(self.model_params, 'num_hidden_layers', 32)
            vocab_size = getattr(self.model_params, 'vocab_size', 50257)

            total_params = (
                hidden_size * hidden_size * 4 * num_layers +
                hidden_size * hidden_size * 3 * num_layers +
                hidden_size * vocab_size * 2
            )
            model_memory = total_params * 2

        if model_memory > self.gpu_memory_capacity:
            return False

        self.model_memory_usage = model_memory
        self.model_loaded = True
        self._refresh_memory_usage()

        self.emit_event(
            EventType.MODEL_LOAD,
            {'model_id': self.model_id, 'memory_usage': model_memory},
            EventPriority.HIGH
        )

        return True

    def unload_model(self):
        """Unload the model from GPU memory."""
        if self.model_loaded:
            self.model_loaded = False
            self.model_memory_usage = 0.0
            self.kv_cache_memory = 0.0
            self.kv_cache_capacity = 0.0

            self.emit_event(
                EventType.MODEL_UNLOAD,
                {'model_id': self.model_id},
                EventPriority.HIGH
            )

    def can_accommodate_request(self, request: GenerationRequest,
                               safety_margin: float = 0.1) -> bool:
        """Check if the engine can accommodate a new request."""
        if not self.model_loaded:
            if not self.load_model():
                return False

        if not self.memory_planner.can_allocate_request(request):
            return False

        current_kv = self.memory_planner.get_allocated_kv_memory_per_shard()
        additional_kv = self.memory_planner.estimate_additional_kv_memory_per_shard(request)
        total_memory_needed = self.model_memory_usage + current_kv + additional_kv
        available_memory = self.gpu_memory_capacity * (1 - safety_margin)

        return total_memory_needed <= available_memory

    def add_request(self, request: GenerationRequest):
        """Add a new request to the processing queue."""
        if not self.can_accommodate_request(request):
            return False

        # Reserve KV cache memory for this request
        self.memory_planner.allocate(request)
        self._refresh_memory_usage()

        # New requests go directly to prefill queue since prefill has priority
        self.prefill_queue.append(request)
        request.status = REQ_STATUS.SCHEDULED

        # Don't emit REQUEST_ARRIVAL event here since the cluster manager
        # already knows about this request and is the one adding it

        return True

    def get_current_load(self) -> int:
        """Get current number of active requests."""
        return len(self.active_requests) + len(self.decode_ready_requests) + (1 if self.current_prefill_request else 0)

    def has_model_loaded(self, model_id: str) -> bool:
        """Check if a specific model is loaded."""
        return self.model_loaded and self.model_id == model_id

    def get_memory_info(self) -> Dict[str, float]:
        """Get current memory usage information."""
        self._refresh_memory_usage()
        total_used = self.model_memory_usage + self.kv_cache_memory
        available = max(self.gpu_memory_capacity - total_used, 0.0)
        utilization = (total_used / self.gpu_memory_capacity) if self.gpu_memory_capacity else 0.0

        return {
            'total': self.gpu_memory_capacity,
            'model_weights': self.model_memory_usage,
            'kv_cache': self.kv_cache_memory,
            'kv_cache_capacity': self.kv_cache_capacity,
            'kv_blocks_used': self.memory_planner.get_allocated_block_count(),
            'kv_blocks_capacity': self.memory_planner.get_max_block_count(),
            'available': available,
            'used': total_used,
            'utilization': utilization
        }

    def step(self, current_time: Optional[float] = None) -> List[Event]:
        """Execute one step of the serving engine with priority-based processing."""
        # Update current time if provided
        if current_time is not None:
            self.current_time = current_time
            # Ensure GPU timeline never goes backwards
            if self.time_cursor < self.current_time:
                self.time_cursor = self.current_time

        events = []

        # Priority 1: Process prefill requests (individual processing, no batching)
        if self.current_prefill_request is None and self.prefill_queue:
            # Start prefill for next waiting request
            self._start_next_prefill()

        if self.current_prefill_request:
            # Continue processing current prefill request
            prefill_complete = self._process_individual_prefill(self.current_time)
            if prefill_complete:
                # Move completed prefill request to decode-ready queue
                self.decode_ready_requests.add(self.current_prefill_request)
                self.current_prefill_request = None

        # Priority 2: Handle decode batch formation and processing
        # Form decode batch whenever we have requests ready for decode and no current decode batch
        # Continuous batching: requests can join decode as soon as they complete prefill
        if self.current_decode_batch is None and self.decode_ready_requests:
            self._form_decode_batch()
        elif self.current_decode_batch and self.decode_ready_requests:
            # Add new ready requests to existing decode batch (continuous batching)
            self._add_to_decode_batch()

        if self.current_decode_batch and not self.current_decode_batch.is_empty():
            decode_events = self._process_decode_batch(self.current_time)
            events.extend(decode_events)

        # Clean up empty decode batch
        if self.current_decode_batch and self.current_decode_batch.is_empty():
            self.current_decode_batch = None

        return events

    def _start_next_prefill(self):
        """Start prefill for the next request in the prefill queue."""
        if not self.prefill_queue:
            return

        # Get the next request
        request = self.prefill_queue.popleft()

        if not self.memory_planner.has_allocation(request.req_id):
            if not self.can_accommodate_request(request):
                # Put it back in the queue for later
                self.prefill_queue.appendleft(request)
                return
            self.memory_planner.allocate(request)
            self._refresh_memory_usage()

        # Start prefill for this request
        self.current_prefill_request = request
        self.active_requests.add(request)
        request._prefill()  # Mark as in prefill phase

        # Emit prefill start event aligned to the current GPU timeline
        if self.event_callback:
            start_time = max(self.time_cursor, self.current_time)
            event = Event(
                timestamp=start_time,
                event_type=EventType.PREFILL_START,
                target=self.engine_id,
                data={'request_id': request.req_id},
                priority=EventPriority.HIGH
            )
            self.event_callback(event)

    def _process_individual_prefill(self, current_time: float = 0.0) -> bool:
        """Process prefill for an individual request (no batching)."""
        if not self.current_prefill_request:
            return False

        request = self.current_prefill_request

        # Calculate prefill time for this single request
        prefill_duration = self._get_prefill_time(request.input_length, 1)

        # Ensure we have a reasonable minimum duration for visualization
        if prefill_duration <= 0:
            prefill_duration = 0.001  # 1ms minimum

        # Align start time with current GPU timeline
        prefill_start_time = max(self.time_cursor, self.current_time)
        prefill_end_time = prefill_start_time + prefill_duration
        self.time_cursor = prefill_end_time

        # Update statistics
        self.statistics['total_prefill_time'] += prefill_duration

        # Create trace event for this prefill including memory snapshot at completion
        memory_snapshot = self._get_memory_snapshot()
        prefill_events = self.create_detailed_events(
            phase="prefill",
            handled_requests=[request],
            start_at=prefill_start_time,
            end_at=prefill_end_time,
            memory_info=memory_snapshot
        )
        self.trace_events.extend(prefill_events)

        # Mark prefill as complete
        request.set_prefill_finished_at(prefill_end_time)

        # Emit prefill complete event
        if self.event_callback:
            event = Event(
                timestamp=prefill_end_time,
                event_type=EventType.PREFILL_COMPLETE,
                target=self.engine_id,
                data={
                    'request_id': request.req_id,
                    'prefill_time': prefill_duration
                },
                priority=EventPriority.HIGH
            )
            self.event_callback(event)

        return True

    def _form_decode_batch(self):
        """Form a decode batch from all ready requests."""
        if not self.decode_ready_requests or self.current_decode_batch is not None:
            return

        # Get all ready requests up to max_batch_size
        batch_requests = list(self.decode_ready_requests)[:self.max_batch_size]

        # Remove selected requests from ready set
        for request in batch_requests:
            self.decode_ready_requests.remove(request)

        # Create decode batch
        self.batch_counter += 1
        self.current_decode_batch = Batch(f"decode_batch_{self.batch_counter}", batch_requests)
        self.current_decode_batch.status = "decode"

        # Update statistics
        self.statistics['batch_sizes'].append(len(batch_requests))

    def _add_to_decode_batch(self):
        """Add new ready requests to existing decode batch (continuous batching)."""
        if not self.current_decode_batch or not self.decode_ready_requests:
            return

        current_batch_size = self.current_decode_batch.get_batch_size()
        available_slots = self.max_batch_size - current_batch_size

        if available_slots <= 0:
            return  # Batch is full

        # Add requests to existing batch up to max batch size
        requests_to_add = list(self.decode_ready_requests)[:available_slots]

        for request in requests_to_add:
            self.decode_ready_requests.remove(request)
            self.current_decode_batch.add_request(request)

        # Update statistics to reflect the new batch size
        self.statistics['batch_sizes'].append(self.current_decode_batch.get_batch_size())

    def _process_decode_batch(self, current_time: float = 0.0) -> List[Event]:
        """Process one decode step for the current decode batch."""
        if not self.current_decode_batch:
            return []

        events = []
        completed_requests = []
        completed_request_ids: List[str] = []

        # Calculate decode time per token using ModelAnalyzer
        batch_size = len(self.current_decode_batch.requests)

        # Calculate current sequence length during decode (prompt + generated tokens)
        if self.current_decode_batch.requests:
            current_seq_length = max(
                req.input_length + req.generated_tokens for req in self.current_decode_batch.requests
            )
        else:
            current_seq_length = 1024  # fallback

        decode_duration = self._get_decode_time(current_seq_length, batch_size)

        # Ensure we have a reasonable minimum duration for visualization
        if decode_duration <= 0:
            decode_duration = 0.001  # 1ms minimum

        # Align decode step with current GPU timeline
        decode_start_time = max(self.time_cursor, self.current_time)
        decode_end_time = decode_start_time + decode_duration
        self.time_cursor = decode_end_time

        # Preserve handled requests for tracing after memory updates
        handled_requests = [
            req for req in self.current_decode_batch.requests if req.status != REQ_STATUS.EXIT
        ]

        # Process decode step for each request
        for request in list(self.current_decode_batch.requests):
            is_complete = request._decode()
            # Ensure KV cache allocations keep pace with generation
            self.memory_planner.allocate(request)
            if is_complete:
                completed_requests.append(request)
                completed_request_ids.append(request.req_id)

        # Update statistics
        self.statistics['total_decode_time'] += decode_duration
        self.statistics['total_tokens_generated'] += batch_size

        # Update KV cache memory
        self._update_kv_cache_memory()

        # Handle completed requests
        for request in completed_requests:
            self.active_requests.remove(request)
            self.completed_requests.append(request)
            self.current_decode_batch.remove_request(request)
            self.statistics['requests_processed'] += 1

            events.append(Event(
                timestamp=decode_end_time,
                event_type=EventType.REQUEST_COMPLETE,
                target=self.engine_id,
                data={
                    'request_id': request.req_id,
                    'total_tokens': request.generated_tokens,
                    'completion_time': decode_end_time
                },
                priority=EventPriority.HIGH
            ))

        if completed_request_ids:
            self.memory_planner.free(completed_request_ids)
            self._update_kv_cache_memory()

        # Create detailed trace events for decode step with memory snapshot post-update
        if handled_requests:
            memory_snapshot = self._get_memory_snapshot()
            decode_events = self.create_detailed_events(
                phase="decode",
                handled_requests=handled_requests,
                start_at=decode_start_time,
                end_at=decode_end_time,
                memory_info=memory_snapshot
            )
            self.trace_events.extend(decode_events)

        return events

    
    def _update_kv_cache_memory(self):
        """Update total KV cache memory usage."""
        self._refresh_memory_usage()

    def _get_prefill_time(self, sequence_length: int, batch_size: int) -> float:
        """Get accurate prefill time using ModelAnalyzer."""
        try:
            results = self.model_analyzer.analyze(
                seqlen=sequence_length,
                batchsize=batch_size,
                w_bit=16,  # Default to 16-bit weights
                a_bit=16,  # Default to 16-bit activations
                kv_bit=16  # Default to 16-bit KV cache
            )
            return results["total_results"]["prefill"]["inference_time"]
        except Exception as e:
            # Fallback to simplified estimate if ModelAnalyzer fails
            print(f"Warning: ModelAnalyzer prefill failed, using fallback: {e}")
            attention_ops = sequence_length * sequence_length * batch_size
            ops_per_second = self.compute_throughput
            return attention_ops / ops_per_second

    def _get_decode_time(self, sequence_length: int, batch_size: int) -> float:
        """Get accurate decode time using ModelAnalyzer."""
        try:
            results = self.model_analyzer.analyze(
                seqlen=sequence_length,
                batchsize=batch_size,
                w_bit=16,  # Default to 16-bit weights
                a_bit=16,  # Default to 16-bit activations
                kv_bit=16  # Default to 16-bit KV cache
            )
            return results["total_results"]["decode"]["inference_time"]
        except Exception as e:
            # Fallback to simplified estimate if ModelAnalyzer fails
            print(f"Warning: ModelAnalyzer decode failed, using fallback: {e}")
            avg_seq_length = 1024  # Placeholder
            attention_ops = avg_seq_length * batch_size
            ops_per_second = self.compute_throughput
            return attention_ops / ops_per_second

    def set_current_time(self, current_time: float):
        """Set the current simulation time for tracing purposes."""
        self.current_time = current_time
        if self.time_cursor < current_time:
            self.time_cursor = current_time

    def add_trace_event(self, name: str, category: str, phase: str,
                       timestamp: float, duration: Optional[float] = None,
                       args: Optional[Dict[str, Any]] = None):
        """Add a trace event for Chrome tracing visualization."""
        trace_event = TraceEvent(
            name=name,
            cat=category,
            ph=phase,
            pid=self.engine_id,
            tid="engine",
            ts=int(timestamp * 1e6),  # Convert to microseconds
            args=args or {},
            dur=int(duration * 1e6) if duration is not None else None
        )
        self.trace_events.append(trace_event)

    def create_detailed_events(self, phase: str, handled_requests: List[GenerationRequest],
                              start_at: float, end_at: float,
                              memory_info: Optional[Dict[str, Any]] = None) -> List[TraceEvent]:
        """
        Create Chrome trace format events for detailed performance visualization.

        Args:
            phase: Either "prefill" or "decode"
            handled_requests: List of requests processed in this phase
            start_at: Start time in seconds
            end_at: End time in seconds

        Returns:
            List of TraceEvent objects compatible with Chrome tracing format
        """
        complete_events = []
        start_us = int(max(start_at, 0) * 1_000_000)
        duration_s = max(end_at - start_at, 0.0)
        duration_us = max(int(duration_s * 1_000_000), 1)

        for req in handled_requests:
            event_args = {
                "request_id": req.req_id,
                "requested_model": req.model,
                "engine_id": str(self.engine_id),
                "engine_model": self.model_id,
                "hardware": self.hardware,
                "phase": phase,
                "start_time_s": round(start_at, 6),
                "end_time_s": round(end_at, 6),
                "duration_s": round(duration_s, 6),
            }

            if phase == "prefill":
                event_args.update(
                    {
                        "prompt_tokens": req.input_length,
                        "target_output_tokens": req.output_length,
                    }
                )
            elif phase == "decode":
                event_args.update(
                    {
                        "target_output_tokens": req.output_length,
                        "generated_tokens_total": req.generated_tokens,
                        "tokens_emitted_this_step": 1,
                    }
                )

            if memory_info:
                event_args.update(memory_info)

            complete_events.append(
                TraceEvent(
                    name=f"{phase.upper()[0]}:{req.req_id}",
                    cat=f"request.{phase}",
                    ph="X",  # Complete event (duration event)
                    pid=str(self.engine_id),
                    tid=0,   # Single thread for the engine
                    ts=start_us,
                    dur=duration_us,
                    args=event_args,
                )
            )

        return complete_events

    def get_statistics(self) -> Dict[str, Any]:
        """Get engine performance statistics."""
        avg_batch_size = np.mean(self.statistics['batch_sizes']) if self.statistics['batch_sizes'] else 0

        return {
            **self.statistics,
            'engine_id': self.engine_id,
            'model_id': self.model_id,
            'hardware': self.hardware,
            'current_load': len(self.active_requests) + len(self.decode_ready_requests) + (1 if self.current_prefill_request else 0),
            'queue_length': len(self.prefill_queue),
            'prefill_queue_length': len(self.prefill_queue),
            'decode_ready_count': len(self.decode_ready_requests),
            'current_prefill_request': self.current_prefill_request is not None,
            'current_decode_batch_size': len(self.current_decode_batch.requests) if self.current_decode_batch else 0,
            'avg_batch_size': avg_batch_size,
            'memory_info': self.get_memory_info(),
            'model_loaded': self.model_loaded,
            'trace_events': self.trace_events
        }