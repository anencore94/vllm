# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import copy

import pytest

from tests.plugins.vllm_add_dummy_stat_logger.dummy_stat_logger.dummy_stat_logger import (  # noqa E501
    DummyStatLogger,
)
from tests.utils import wait_for_memory_to_settle
from vllm.config import CacheConfig
from vllm.v1.engine.async_llm import AsyncEngineArgs, AsyncLLM
from vllm.v1.metrics.loggers import PerEngineStatLoggerAdapter, StatLoggerBase
from vllm.v1.metrics.ray_wrappers import RayPrometheusStatLogger


def test_priority_stats_through_llm(vllm_runner):
    """Real engine/IPC snapshots reach external loggers without raw labels."""
    from vllm import SamplingParams

    snapshots = []

    class RecordingLogger(StatLoggerBase):
        def __init__(self, vllm_config, engine_index=0):
            pass

        def log_engine_initialized(self):
            pass

        def record(
            self, scheduler_stats, iteration_stats, mm_cache_stats=None, engine_idx=0
        ):
            if scheduler_stats is not None:
                snapshots.append(scheduler_stats)

    with vllm_runner(
        "distilbert/distilgpt2",
        scheduling_policy="priority",
        max_num_seqs=1,
        enforce_eager=True,
        disable_log_stats=False,
        stat_loggers=[RecordingLogger],
    ) as runner:
        outputs = runner.llm.generate(
            ["Hello"] * 4,
            SamplingParams(temperature=0, max_tokens=4),
            priority=[50, 30, 10, -1],
        )
    assert len(outputs) == 4 and all(output.finished for output in outputs)
    assert snapshots
    assert any(stats.num_running_reqs for stats in snapshots)
    assert any(stats.num_waiting_reqs for stats in snapshots)
    for stats in snapshots:
        priority = stats.priority_stats
        assert priority is not None
        assert sum(priority.running) == stats.num_running_reqs
        assert sum(priority.waiting) == (
            stats.num_waiting_reqs + stats.num_skipped_waiting_reqs
        )
    assert {stats.priority_stats.worst_running_priority for stats in snapshots} >= {
        -1,
        10,
        30,
        50,
    }


def _required_utilization(engine_args: AsyncEngineArgs) -> float:
    # Unset resolves to 1.0 under the extensible KV cache, which does not need
    # that memory free up front; wait for the standard fraction instead.
    return (
        engine_args.gpu_memory_utilization or CacheConfig.DEFAULT_GPU_MEMORY_UTILIZATION
    )


@pytest.fixture
def log_stats_enabled_engine_args():
    """Shared fixture providing common AsyncEngineArgs configuration
    used across multiple tests.
    """
    return AsyncEngineArgs(
        model="distilbert/distilgpt2",
        dtype="half",
        disable_log_stats=False,
        enforce_eager=True,
    )


@pytest.mark.asyncio
async def test_async_llm_replace_default_loggers(log_stats_enabled_engine_args):
    """RayPrometheusStatLogger should replace the default PrometheusStatLogger."""
    engine = AsyncLLM.from_engine_args(
        log_stats_enabled_engine_args, stat_loggers=[RayPrometheusStatLogger]
    )
    try:
        assert engine.logger_manager is not None
        assert isinstance(
            engine.logger_manager.stat_loggers[0], RayPrometheusStatLogger
        )
    finally:
        engine.shutdown()
        wait_for_memory_to_settle(
            threshold_ratio=1.0 - _required_utilization(log_stats_enabled_engine_args)
        )


@pytest.mark.asyncio
async def test_async_llm_add_to_default_loggers(log_stats_enabled_engine_args):
    """It's still possible to use custom stat loggers exclusively by passing
    disable_log_stats=True in addition to a list of custom stat loggers.
    """
    # Create engine_args with disable_log_stats=True for this test
    disabled_log_engine_args = copy.deepcopy(log_stats_enabled_engine_args)
    disabled_log_engine_args.disable_log_stats = True

    # Disable default loggers; pass custom stat logger to the constructor
    engine = AsyncLLM.from_engine_args(
        disabled_log_engine_args, stat_loggers=[DummyStatLogger]
    )

    try:
        assert engine.logger_manager is not None
        assert len(engine.logger_manager.stat_loggers) == 2
        # DummyStatLogger is a per-engine logger, so the manager wraps it.
        per_engine_adapter = engine.logger_manager.stat_loggers[0]
        assert isinstance(per_engine_adapter, PerEngineStatLoggerAdapter)
        assert len(per_engine_adapter.per_engine_stat_loggers) == 1
        assert isinstance(
            per_engine_adapter.per_engine_stat_loggers[0],
            DummyStatLogger,
        )

        # log_stats is still True, since custom stat loggers are used
        assert engine.log_stats
    finally:
        engine.shutdown()
        wait_for_memory_to_settle(
            threshold_ratio=1.0 - _required_utilization(disabled_log_engine_args)
        )
