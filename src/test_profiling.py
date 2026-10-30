import time

import numpy as np

from src.profiling import measure_resources


def test_measure_resources_records_runtime():
    with measure_resources() as usage:
        time.sleep(0.05)

    assert usage.wall_time_s >= 0.05
    assert usage.cpu_time_s < usage.wall_time_s


def test_measure_resources_records_memory_kept_by_the_block():
    with measure_resources() as usage:
        block = np.ones(200 * 1024 ** 2 // 8)  # 200 MB, touched so it is resident

    assert usage.rss_increase_mb > 0
    assert usage.rss_mb >= usage.rss_increase_mb
    del block


def test_measure_resources_fills_usage_when_the_block_raises():
    try:
        with measure_resources() as usage:
            raise RuntimeError
    except RuntimeError:
        pass

    assert usage.wall_time_s >= 0
    assert set(usage.as_dict()) == {"wall_time_s", "cpu_time_s", "rss_mb", "rss_increase_mb"}
