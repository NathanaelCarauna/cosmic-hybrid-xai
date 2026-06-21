"""Timing utilities for performance measurement and reporting."""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Generator

import numpy as np


@contextmanager
def timer() -> Generator[dict, None, None]:
    """Context manager that records elapsed wall-clock seconds.

    Usage::

        with timer() as t:
            do_work()
        print(t["elapsed"])
    """
    record: dict = {"elapsed": 0.0}
    t0 = time.perf_counter()
    try:
        yield record
    finally:
        record["elapsed"] = time.perf_counter() - t0


def compute_timing_stats(durations: list[float]) -> dict:
    """Compute descriptive statistics over a list of durations (seconds).

    Returns a dict with: total, mean, median, min, max, p95, count.
    """
    if not durations:
        return {
            "total_s": 0.0,
            "mean_s": 0.0,
            "median_s": 0.0,
            "min_s": 0.0,
            "max_s": 0.0,
            "p95_s": 0.0,
            "count": 0,
        }
    arr = np.array(durations)
    return {
        "total_s": float(arr.sum()),
        "mean_s": float(arr.mean()),
        "median_s": float(np.median(arr)),
        "min_s": float(arr.min()),
        "max_s": float(arr.max()),
        "p95_s": float(np.percentile(arr, 95)),
        "count": len(durations),
    }


def compute_throughput(
    n_items: int,
    total_tokens: int,
    wall_clock_s: float,
) -> dict:
    """Compute throughput metrics."""
    if wall_clock_s <= 0:
        return {"fp_per_min": 0.0, "tokens_per_s": 0.0}
    return {
        "fp_per_min": round(n_items / wall_clock_s * 60, 2),
        "tokens_per_s": round(total_tokens / wall_clock_s, 2),
    }
