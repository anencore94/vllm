# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise priority series through the real logger and Prometheus registry."""

import random

import pytest
from prometheus_client import REGISTRY, generate_latest
from prometheus_client.parser import text_string_to_metric_families

from vllm.config import ModelConfig, SchedulerConfig, VllmConfig
from vllm.config.device import DeviceConfig
from vllm.v1.metrics.loggers import PrometheusStatLogger
from vllm.v1.metrics.stats import SchedulerStats

pytestmark = pytest.mark.cpu_test

METRICS = {
    "vllm:num_requests_running_by_tier",
    "vllm:num_requests_waiting_by_tier",
    "vllm:lowest_running_priority_value",
    "vllm:preemptions_total",
}


def _samples():
    return [
        sample
        for family in text_string_to_metric_families(generate_latest(REGISTRY).decode())
        for sample in family.samples
        if sample.name in METRICS
    ]


def _logger(policy="priority"):
    return PrometheusStatLogger(
        VllmConfig(
            model_config=ModelConfig(model="facebook/opt-125m"),
            device_config=DeviceConfig(device="cpu"),
            scheduler_config=SchedulerConfig(
                policy=policy, max_model_len=2048, is_encoder_decoder=False
            ),
        ),
        engine_indexes=[0, 1],
    )


def test_priority_metrics_registered():
    logger = _logger()
    logger.record(SchedulerStats(), None)
    samples = _samples()
    assert len(samples) == 2 * (4 + 4 + 1 + 16)
    assert all(sample.value == 0 for sample in samples)


def test_priority_registry_cardinality_and_reset():
    from vllm.v1.metrics.stats import (
        PRIORITY_TIERS,
        PriorityStats,
        priority_tier_index,
    )

    logger = _logger()
    priorities = random.Random(42).sample(range(-(10**9), 10**9), 10_000)
    priorities += [-1, 0, 9, 10, 29, 30, 49, 50]
    for index, priority in enumerate(priorities):
        stats = PriorityStats(worst_running_priority=priority)
        stats.running[priority_tier_index(priority)] = 1
        stats.waiting[priority_tier_index(priority)] = 2
        stats.preemptions[index % 16] = 1
        logger.record(
            SchedulerStats(
                num_running_reqs=1,
                num_waiting_reqs=2,
                priority_stats=stats,
            ),
            None,
        )
        if index % 100 == 0:
            samples = _samples()
            assert len(samples) == 2 * (4 + 4 + 1 + 16)
            for name, count in (("running", 1), ("waiting", 2)):
                for sample in samples:
                    if sample.name == f"vllm:num_requests_{name}_by_tier":
                        expected = (
                            count
                            if sample.labels["engine"] == "0"
                            and sample.labels["tier"]
                            == PRIORITY_TIERS[priority_tier_index(priority)]
                            else 0
                        )
                        assert sample.value == expected

    samples = _samples()
    assert all(
        sample.labels.get("tier", PRIORITY_TIERS[0]) in PRIORITY_TIERS
        for sample in samples
    )
    for label in ("victim_tier", "trigger_tier"):
        assert all(
            sample.labels.get(label, PRIORITY_TIERS[0]) in PRIORITY_TIERS
            for sample in samples
        )
    assert not any("priority" in sample.labels for sample in samples)
    assert sum(s.value for s in samples if s.name == "vllm:preemptions_total") == len(
        priorities
    )
    assert (
        next(
            s.value
            for s in samples
            if s.name == "vllm:lowest_running_priority_value"
            and s.labels["engine"] == "0"
        )
        == 50
    )

    # Default/older stats reset stale tiers, but never erase accumulated counters.
    logger.record(SchedulerStats(), None)
    reset = _samples()
    assert all(s.value == 0 for s in reset if s.name != "vllm:preemptions_total")
    assert sum(s.value for s in reset if s.name == "vllm:preemptions_total") == len(
        priorities
    )


def test_fcfs_keeps_priority_metrics_absent():
    logger = _logger("fcfs")
    logger.record(SchedulerStats(), None)
    assert _samples() == []
