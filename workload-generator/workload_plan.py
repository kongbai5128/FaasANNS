"""Utilities for replaying workload-generator plan.bin schedules."""

from __future__ import annotations

import pickle
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class WorkloadPlan:
    path: Path
    timestamps: list[float]
    offset: float = 0.0

    @property
    def duration(self) -> float:
        return self.timestamps[-1] if self.timestamps else 0.0


class QueryProgress:
    def __init__(self, total: int, *, enabled: bool = True, stream=None) -> None:
        self.total = total
        self.enabled = enabled and total > 0
        self.stream = stream or sys.stderr
        self.sent = 0
        self.received = 0
        self.started_at = time.perf_counter()
        self.lock = threading.Lock()
        self.closed = False
        self.is_tty = bool(getattr(self.stream, "isatty", lambda: False)())
        self.interval_s = 0.5 if self.is_tty else 5.0
        self.last_render_at = 0.0
        self.last_line_len = 0
        if self.enabled:
            self._render(force=True)

    def mark_sent(self) -> None:
        if not self.enabled:
            return
        with self.lock:
            self.sent += 1
            self._render(force=self.sent == self.total)

    def mark_received(self) -> None:
        if not self.enabled:
            return
        with self.lock:
            self.received += 1
            self._render(force=self.received == self.total)

    def close(self) -> None:
        if not self.enabled:
            return
        with self.lock:
            if self.closed:
                return
            if self.is_tty or self.sent < self.total or self.received < self.total:
                self._render(force=True)
            if self.is_tty:
                self.stream.write("\n")
                self.stream.flush()
            self.closed = True

    def _render(self, *, force: bool = False) -> None:
        now = time.perf_counter()
        if not force and now - self.last_render_at < self.interval_s:
            return
        self.last_render_at = now
        elapsed = now - self.started_at
        in_flight = max(self.sent - self.received, 0)
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        line = (
            f"Progress: time={timestamp}, "
            f"sent={self.sent}/{self.total} ({self.sent / self.total * 100:.1f}%), "
            f"received={self.received}/{self.total} ({self.received / self.total * 100:.1f}%), "
            f"in_flight={in_flight}, elapsed={elapsed:.1f}s"
        )
        if self.is_tty:
            padding = " " * max(0, self.last_line_len - len(line))
            self.stream.write(f"\r{line}{padding}")
            self.stream.flush()
            self.last_line_len = len(line)
        else:
            print(line, file=self.stream, flush=True)


class _ModeT(Enum):
    get_distribution = auto()
    generate = auto()


class _PlanParameters:
    pass


class _PlanDump:
    pass


class _PlanUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if module == "common":
            if name == "dump_t":
                return _PlanDump
            if name == "parameters_t":
                return _PlanParameters
            if name == "mode_t":
                return _ModeT
        return super().find_class(module, name)


def load_workload_plan(path: str | Path, *, root: Path) -> WorkloadPlan:
    """Load workload-generator/plan.bin.

    The generator pickles objects from module name ``common``. Use a small
    custom unpickler so replaying a plan does not require importing scipy from
    workload-generator/common.py.
    """

    plan_path = Path(path)
    if not plan_path.is_absolute():
        plan_path = root / plan_path
    plan_path = plan_path.resolve()

    with plan_path.open("rb") as fp:
        dump = _PlanUnpickler(fp).load()

    raw_timestamps = getattr(dump, "plan", None)
    if raw_timestamps is None:
        raise ValueError(f"workload plan {plan_path} does not contain a plan list")

    timestamps = [float(value) for value in raw_timestamps]
    if any(value < 0 for value in timestamps):
        raise ValueError(f"workload plan {plan_path} contains negative timestamps")

    parameters = getattr(dump, "parameters", None)
    offset = float(getattr(parameters, "offset", 0.0) or 0.0)
    return WorkloadPlan(path=plan_path, timestamps=timestamps, offset=offset)


def submit_with_plan(
    executor: ThreadPoolExecutor,
    count: int,
    submit: Callable[[int], object],
    plan: WorkloadPlan | None,
    *,
    spin_threshold_s: float = 0.03,
    on_submit: Callable[[Future], None] | None = None,
) -> tuple[list[Future], float]:
    """Submit query tasks immediately or at timestamps from ``plan``.

    Returns the submitted futures and the wall-clock time immediately before
    query scheduling starts. Cold-start accounting should use this wall time,
    not process startup time.
    """

    def track(future: Future) -> Future:
        if on_submit is not None:
            on_submit(future)
        return future

    if plan is None:
        batch_start_wall_time = time.time()
        return [track(executor.submit(submit, i)) for i in range(count)], batch_start_wall_time

    if count > len(plan.timestamps):
        raise ValueError(f"query count {count} exceeds workload plan length {len(plan.timestamps)}")

    if plan.offset > 0:
        time.sleep(plan.offset)

    batch_start_wall_time = time.time()
    t0 = time.perf_counter()
    futures: list[Future] = []
    items = list(enumerate(plan.timestamps[:count]))
    pos = 0
    while pos < len(items):
        target = items[pos][1]
        group: list[int] = []
        while pos < len(items) and items[pos][1] == target:
            group.append(items[pos][0])
            pos += 1

        _wait_until(t0, target, spin_threshold_s)
        if len(group) == 1:
            futures.append(track(executor.submit(submit, group[0])))
            continue

        start_gate = threading.Event()
        futures.extend(track(executor.submit(_run_after_gate, start_gate, submit, i)) for i in group)
        start_gate.set()
    return futures, batch_start_wall_time


def _wait_until(t0: float, target: float, spin_threshold_s: float) -> None:
    while True:
        elapsed = time.perf_counter() - t0
        remaining = target - elapsed
        if remaining <= 0:
            return
        if remaining <= spin_threshold_s:
            while time.perf_counter() - t0 < target:
                pass
            return
        time.sleep(max(0.0, remaining - spin_threshold_s))


def _run_after_gate(gate: threading.Event, submit: Callable[[int], object], i: int) -> object:
    gate.wait()
    return submit(i)


def resolve_request_count(query_num: int, plan: WorkloadPlan | None) -> int:
    """Resolve how many requests to submit.

    ``query_num <= 0`` means replay the whole workload plan. Without a plan we
    require an explicit positive count so accidental empty runs fail early.
    """

    if plan is None:
        if query_num <= 0:
            raise ValueError("--query-num must be positive when --plan-file is not set")
        return query_num
    if query_num <= 0:
        return len(plan.timestamps)
    return min(query_num, len(plan.timestamps))


def describe_workload_plan(plan: WorkloadPlan | None, count: int) -> str:
    if plan is None:
        return "Workload plan: disabled"
    last_timestamp = plan.timestamps[count - 1] if count > 0 else 0.0
    return (
        f"Workload plan: {plan.path} "
        f"(using {count}/{len(plan.timestamps)} timestamps, "
        f"offset={plan.offset:.3f}s, last={last_timestamp:.3f}s)"
    )
