"""
Server-Timing instrumentation for the FoodAI inference service.

Adds a `Server-Timing` response header:
    Server-Timing: download;dur=85.2, decode;dur=12.3, preprocess;dur=4.1, inference;dur=180.7, total;dur=290.5

All durations in milliseconds. Spring Boot forwards these to the phone with an `ai_`
prefix (see FoodAI-API's ServerTiming.mergeUpstream), so the client sees one combined
header covering the whole request chain.

Added for the latency benchmark requested by ICACIT Reviewer 1 (rigor: N>=100 trials,
device, network, percentiles, CPU/GPU). See paper/benchmark_latencia/PROTOCOLO.md.
"""

from __future__ import annotations

import contextlib
import contextvars
import time
from typing import Dict, Iterator

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

# Per-request accumulator of stage durations. contextvars keeps it isolated per request
# even with async handlers.
_stages: contextvars.ContextVar[Dict[str, float] | None] = contextvars.ContextVar(
    "server_timing_stages", default=None
)


@contextlib.contextmanager
def timed_stage(name: str) -> Iterator[None]:
    """Measure a block of the handler and store it under `name` (milliseconds)."""
    stages = _stages.get()
    t0 = time.perf_counter()
    try:
        yield
    finally:
        if stages is not None:
            dt_ms = (time.perf_counter() - t0) * 1000.0
            stages[name] = stages.get(name, 0.0) + dt_ms


class ServerTimingMiddleware(BaseHTTPMiddleware):
    """Attach `Server-Timing` with the recorded stages plus the total handler time."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        token = _stages.set({})
        t0 = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            stages = _stages.get() or {}
            _stages.reset(token)
        total_ms = (time.perf_counter() - t0) * 1000.0

        parts = [f"{k};dur={v:.2f}" for k, v in stages.items()]
        parts.append(f"total;dur={total_ms:.2f}")
        response.headers["Server-Timing"] = ", ".join(parts)
        return response
