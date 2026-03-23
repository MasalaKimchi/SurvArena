from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass


@dataclass(slots=True)
class TimerResult:
    start: float
    end: float | None = None

    @property
    def elapsed(self) -> float:
        if self.end is None:
            return time.perf_counter() - self.start
        return self.end - self.start


@contextmanager
def timer() -> TimerResult:
    result = TimerResult(start=time.perf_counter())
    try:
        yield result
    finally:
        result.end = time.perf_counter()
