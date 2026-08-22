import time
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from astrbot.dashboard.services.stat_service import StatService


def _make_service(db) -> StatService:
    """Build a StatService with a real DB and a mocked core lifecycle."""
    core_lifecycle = MagicMock()
    core_lifecycle.star_context.get_all_stars.return_value = []
    core_lifecycle.platform_manager.get_insts.return_value = []
    core_lifecycle.start_time = int(time.time()) - 100
    return StatService(db_helper=db, core_lifecycle=core_lifecycle, config={})


@pytest.mark.asyncio
async def test_get_stat_aggregates_platform_stats(temp_db):
    """Seeded rows must aggregate into windowed platform sums and a global total."""
    now = datetime.now()
    seed = [
        ("aiocqhttp", 3, now - timedelta(hours=1)),
        ("aiocqhttp", 5, now - timedelta(hours=1, minutes=30)),
        ("qqofficial", 2, now - timedelta(hours=2)),
        ("webchat", 7, now - timedelta(minutes=10)),
        # Outside the 24h window: counted in the total but not in window stats.
        ("aiocqhttp", 4, now - timedelta(hours=26)),
    ]
    for platform_id, count, ts in seed:
        await temp_db.insert_platform_stats(platform_id, platform_id, count, ts)

    result = await _make_service(temp_db).get_stat(86400)

    # Global total counts every row, including the one outside the window.
    assert result["message_count"] == 21

    # Windowed per-platform sums, serialized with the legacy response keys.
    platform = {entry["name"]: entry["count"] for entry in result["platform"]}
    assert platform == {"aiocqhttp": 8, "qqofficial": 2, "webchat": 7}
    for entry in result["platform"]:
        assert set(entry) == {"name", "count", "timestamp"}

    # Hourly buckets cover [now - offset, now) in ascending order.
    series = result["message_time_series"]
    assert len(series) == 24
    bucket_ends = [bucket_end for bucket_end, _ in series]
    assert bucket_ends == sorted(bucket_ends)
    assert all(count >= 0 for _, count in series)
    # Rows within the current partial hour are not bucketed yet, so the
    # series sum never exceeds the windowed total of 17.
    assert sum(count for _, count in series) <= 17

    assert set(result) == {
        "platform",
        "message_count",
        "platform_count",
        "plugin_count",
        "plugins",
        "message_time_series",
        "running",
        "memory",
        "cpu_percent",
        "thread_count",
        "start_time",
    }


@pytest.mark.asyncio
async def test_get_stat_empty_window(temp_db):
    """A window with no rows yields empty platform stats but keeps the total."""
    old_ts = datetime.now() - timedelta(hours=2)
    await temp_db.insert_platform_stats("aiocqhttp", "aiocqhttp", 4, old_ts)

    result = await _make_service(temp_db).get_stat(1)

    assert result["platform"] == []
    assert result["message_count"] == 4
    assert all(count == 0 for _, count in result["message_time_series"])


@pytest.mark.asyncio
async def test_provider_token_stats_include_internal_and_provider_records(temp_db):
    await temp_db.insert_provider_stat(
        umo="platform:internal",
        provider_id="standard",
        provider_model="standard-model",
        status="completed",
        stats={
            "token_usage": {
                "input_other": 3,
                "input_cached": 1,
                "output": 4,
            },
            "start_time": 1.0,
            "end_time": 2.0,
            "time_to_first_token": 0.1,
        },
        agent_type="internal",
    )
    await temp_db.insert_provider_stat(
        umo="provider:oauth:text",
        provider_id="oauth",
        provider_model="oauth-model",
        status="error",
        stats={
            "token_usage": {
                "input_other": 5,
                "input_cached": 2,
                "output": 6,
            },
            "start_time": 3.0,
            "end_time": 4.0,
            "time_to_first_token": 0.0,
        },
        agent_type="provider",
    )
    await temp_db.insert_provider_stat(
        umo="third-party",
        provider_id="excluded",
        provider_model="excluded-model",
        status="completed",
        stats={
            "token_usage": {
                "input_other": 100,
                "input_cached": 0,
                "output": 0,
            },
            "start_time": 5.0,
            "end_time": 6.0,
            "time_to_first_token": 0.0,
        },
        agent_type="third_party",
    )
    await temp_db.insert_provider_stat(
        umo="provider:oauth:test",
        provider_id="oauth",
        provider_model="oauth-model",
        status="completed",
        stats={
            "token_usage": {
                "input_other": 1,
                "input_cached": 0,
                "output": 2,
            },
            "start_time": 7.0,
            "end_time": 8.0,
            "time_to_first_token": 0.0,
        },
        agent_type="test",
    )

    service = StatService(temp_db, SimpleNamespace(), {})
    stats = await service.get_provider_token_stats(1)

    assert stats["range_total_calls"] == 3
    assert stats["range_total_tokens"] == 24
    assert stats["range_success_rate"] == pytest.approx(2 / 3)
    assert stats["range_call_counts"] == {
        "agent": 1,
        "provider": 1,
        "test": 1,
    }
    assert stats["range_token_totals"] == {
        "agent": 8,
        "provider": 13,
        "test": 3,
    }
    assert stats["today_total_calls"] == 3
    assert stats["today_total_tokens"] == 24
    assert stats["range_by_provider"] == [
        {"provider_id": "oauth", "tokens": 16},
        {"provider_id": "standard", "tokens": 8},
    ]
    assert stats["today_by_model"] == [
        {"provider_model": "oauth-model", "tokens": 16},
        {"provider_model": "standard-model", "tokens": 8},
    ]


@pytest.mark.asyncio
async def test_aborted_provider_call_is_not_counted_as_success(temp_db):
    await temp_db.insert_provider_stat(
        umo="platform:aborted",
        provider_id="standard",
        status="aborted",
        stats={"token_usage": {"output": 1}},
        agent_type="internal",
    )

    service = StatService(temp_db, SimpleNamespace(), {})
    stats = await service.get_provider_token_stats(1)

    assert stats["range_total_calls"] == 1
    assert stats["range_success_rate"] == 0
