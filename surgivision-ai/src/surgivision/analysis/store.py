"""Small thread-safe in-memory registry for API analysis sessions."""

from __future__ import annotations

from collections import OrderedDict
from threading import Lock

from surgivision.types import AnalysisResult


class AnalysisStore:
    """Bounded process-local analysis storage with least-recently-used eviction."""

    def __init__(self, capacity: int = 8) -> None:
        if capacity <= 0:
            raise ValueError("Store capacity must be greater than zero")
        self.capacity = capacity
        self._items: OrderedDict[str, AnalysisResult] = OrderedDict()
        self._lock = Lock()

    def put(self, analysis: AnalysisResult) -> None:
        with self._lock:
            self._items[analysis.analysis_id] = analysis
            self._items.move_to_end(analysis.analysis_id)
            while len(self._items) > self.capacity:
                self._items.popitem(last=False)

    def get(self, analysis_id: str) -> AnalysisResult | None:
        with self._lock:
            result = self._items.get(analysis_id)
            if result is not None:
                self._items.move_to_end(analysis_id)
            return result

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
