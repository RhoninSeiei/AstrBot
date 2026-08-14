from types import SimpleNamespace

import pytest

from astrbot.dashboard.services.stat_service import StatService


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
