"""轻量阶段计时工具。

measure 是上下文管理器，用于记录 SearchService 中 plan、candidate search、rerank 等阶段耗时。
Elapsed 同时提供 seconds 和 ms，方便内部统计和 API 返回。
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass


@dataclass
class Elapsed:
    seconds: float = 0.0

    @property
    def ms(self) -> float:
        return self.seconds * 1000.0


@contextmanager
def measure() -> Elapsed:
    elapsed = Elapsed()
    start = time.perf_counter()
    try:
        yield elapsed
    finally:
        elapsed.seconds = time.perf_counter() - start
