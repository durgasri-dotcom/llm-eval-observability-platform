from contextlib import contextmanager
from dataclasses import dataclass, field
from time import time
from typing import Optional
import uuid


@dataclass
class Span:
    name: str
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    start_time: float
    end_time: Optional[float] = None
    attributes: dict = field(default_factory=dict)

    @property
    def duration_ms(self):
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time) * 1000


class Tracer:
    def __init__(self):
        self.spans = []
        self._current_trace_id = None
        self._span_stack = []

    def start_trace(self):
        self._current_trace_id = str(uuid.uuid4())
        self._span_stack = []
        return self._current_trace_id

    @contextmanager
    def span(self, name, **attributes):
        if self._current_trace_id is None:
            self.start_trace()
        span_id = str(uuid.uuid4())[:8]
        parent = self._span_stack[-1] if self._span_stack else None
        s = Span(
            name=name,
            trace_id=self._current_trace_id,
            span_id=span_id,
            parent_span_id=parent,
            start_time=time(),
            attributes=attributes,
        )
        self._span_stack.append(span_id)
        try:
            yield s
        finally:
            s.end_time = time()
            self._span_stack.pop()
            self.spans.append(s)

    def export_trace_tree(self, trace_id=None):
        tid = trace_id or self._current_trace_id
        return [
            {
                "span_id": s.span_id,
                "parent_span_id": s.parent_span_id,
                "name": s.name,
                "duration_ms": s.duration_ms,
                "attributes": s.attributes,
            }
            for s in self.spans
            if s.trace_id == tid
        ]