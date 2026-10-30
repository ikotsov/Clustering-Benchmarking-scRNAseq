"""
Runtime and memory measurement of a block of code.

Memory is the resident set size (RSS) of the whole process: the RAM it occupies, including what
C extensions (igraph, leidenalg, hdbscan) allocate outside of Python.
"""
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import asdict, dataclass

import psutil

# The OS reports memory in bytes. 1 MB = 1024 × 1024 bytes, so dividing by _MB converts bytes to megabytes.
_MB = 1024 ** 2


@dataclass
class ResourceUsage:
    """
    wall_time_s : elapsed (wall clock) time of the block
    cpu_time_s : user + system CPU time of all threads of the process during the block;
        above wall_time_s when the block runs in parallel
    rss_mb : RSS of the process when the block ended (data loaded before it included)
    rss_increase_mb : rss_mb minus the RSS when the block started. Memory the block used and freed
        again before it ended is not counted, so this is a lower bound of what the block needed.
    """
    wall_time_s: float = float("nan")
    cpu_time_s: float = float("nan")
    rss_mb: float = float("nan")
    rss_increase_mb: float = float("nan")

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


@contextmanager
def measure_resources() -> Generator[ResourceUsage]:
    """Measures the runtime and memory of the with block; the yielded ResourceUsage is filled in on exit."""
    process = psutil.Process()
    usage = ResourceUsage()

    rss_before = process.memory_info().rss
    cpu_before = _cpu_time(process)
    wall_before = time.perf_counter()
    try:
        yield usage
    finally:
        usage.wall_time_s = time.perf_counter() - wall_before
        usage.cpu_time_s = _cpu_time(process) - cpu_before
        rss_after = process.memory_info().rss
        usage.rss_mb = rss_after / _MB
        usage.rss_increase_mb = (rss_after - rss_before) / _MB


def _cpu_time(process: psutil.Process) -> float:
    times = process.cpu_times()
    return times.user + times.system
