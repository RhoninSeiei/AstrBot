import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from astrbot.core.provider import manager as manager_module
from astrbot.core.provider.entities import ProviderType
from astrbot.core.provider.manager import ProviderManager
from astrbot.core.provider.provider import (
    DecisionProvider,
    EmbeddingProvider,
    Provider,
    RerankProvider,
    STTProvider,
    TTSProvider,
)

PROVIDER_KINDS = [
    (Provider, "provider_insts", "curr_provider_inst", ProviderType.CHAT_COMPLETION),
    (
        STTProvider,
        "stt_provider_insts",
        "curr_stt_provider_inst",
        ProviderType.SPEECH_TO_TEXT,
    ),
    (
        TTSProvider,
        "tts_provider_insts",
        "curr_tts_provider_inst",
        ProviderType.TEXT_TO_SPEECH,
    ),
    (EmbeddingProvider, "embedding_provider_insts", None, None),
    (RerankProvider, "rerank_provider_insts", None, None),
    (
        DecisionProvider,
        "decision_provider_insts",
        "curr_decision_provider_inst",
        ProviderType.DECISION,
    ),
]
POINTERS = [
    "curr_provider_inst",
    "curr_stt_provider_inst",
    "curr_tts_provider_inst",
    "curr_decision_provider_inst",
]


@pytest.fixture
def provider_manager(monkeypatch):
    class Config(dict):
        saved = 0

        def save_config(self):
            self.saved += 1

    config = Config(
        provider=[{"id": "provider", "enable": True}],
        provider_sources=[],
        provider_settings={},
        provider_stt_settings={"enable": True, "provider_id": "provider"},
        provider_tts_settings={"enable": True, "provider_id": "provider"},
        agent_runner={
            "runner_type": "local",
            "config": {"model": {"provider_id": "provider"}},
        },
    )
    acm = SimpleNamespace(
        confs={"default": config}, default_conf=config, get_conf=lambda _: config
    )
    manager = ProviderManager(acm, MagicMock(), SimpleNamespace(default_persona=""))
    monkeypatch.setattr(manager_module, "astrbot_config", config)
    return manager


def make_provider(kind=Provider, provider_id="provider"):
    provider = MagicMock(spec=kind)
    provider.meta.return_value = SimpleNamespace(id=provider_id)
    provider.terminate = AsyncMock()
    return provider


@pytest.mark.asyncio
@pytest.mark.parametrize("kind,list_name,pointer,provider_type", PROVIDER_KINDS)
@pytest.mark.parametrize("failure", [RuntimeError, asyncio.CancelledError])
async def test_failed_cleanup_cannot_leave_a_selectable_provider(
    provider_manager, kind, list_name, pointer, provider_type, failure
):
    manager = provider_manager
    provider = make_provider(kind)
    manager.inst_map["provider"] = provider
    getattr(manager, list_name).append(provider)
    if pointer:
        setattr(manager, pointer, provider)
    provider.terminate.side_effect = failure("cleanup stopped after closing")

    with pytest.raises(failure):
        await manager.terminate_provider("provider")

    assert await manager.get_provider_by_id("provider") is None
    assert not getattr(manager, list_name)
    if pointer:
        assert getattr(manager, pointer) is None
        assert await manager.get_using_provider_async(provider_type) is None
    await manager.terminate_provider("provider")
    provider.terminate.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind,list_name,pointer,provider_type", PROVIDER_KINDS)
async def test_detach_removes_duplicate_identity_references_but_not_equal_instances(
    provider_manager, kind, list_name, pointer, provider_type
):
    manager = provider_manager
    old = make_provider(kind)
    replacement = make_provider(kind, "replacement")
    old.__eq__.return_value = True
    manager.inst_map.update(provider=old, alias=old, replacement=replacement)
    instances = getattr(manager, list_name)
    instances.extend([replacement, old, old])
    if pointer:
        setattr(manager, pointer, replacement)

    await manager.terminate_provider("provider")

    assert manager.inst_map == {"replacement": replacement}
    assert len(instances) == 1 and instances[0] is replacement
    assert getattr(manager, list_name) is instances
    if pointer:
        assert getattr(manager, pointer) is replacement
    old.terminate.assert_awaited_once()
    replacement.terminate.assert_not_awaited()
    await manager.terminate_provider("alias")
    old.terminate.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [None, RuntimeError, asyncio.CancelledError])
async def test_suspended_cleanup_detaches_before_await_and_preserves_same_id_replacement(
    provider_manager, failure
):
    manager = provider_manager
    old = make_provider()
    replacement = make_provider()
    entered = asyncio.Event()
    release = asyncio.Event()

    async def cleanup():
        entered.set()
        await release.wait()
        if failure:
            raise failure("cleanup stopped")

    old.terminate.side_effect = cleanup
    manager.inst_map["provider"] = old
    manager.provider_insts.append(old)
    manager.curr_provider_inst = old
    task = asyncio.create_task(manager.terminate_provider("provider"))
    await asyncio.wait_for(entered.wait(), 1)
    try:
        assert await manager.get_provider_by_id("provider") is None
        assert (
            await manager.get_using_provider_async(ProviderType.CHAT_COMPLETION) is None
        )
        await manager.terminate_provider("provider")
        old.terminate.assert_awaited_once()
        manager.inst_map["provider"] = replacement
        manager.provider_insts.append(replacement)
        manager.curr_provider_inst = replacement
    finally:
        release.set()
        if failure:
            with pytest.raises(failure):
                await task
        else:
            await task
    assert await manager.get_provider_by_id("provider") is replacement
    assert manager.provider_insts == [replacement]
    assert manager.curr_provider_inst is replacement
    replacement.terminate.assert_not_awaited()


@pytest.mark.asyncio
async def test_actual_task_cancellation_keeps_termination_final(provider_manager):
    manager = provider_manager
    provider = make_provider()
    entered = asyncio.Event()

    async def cleanup():
        entered.set()
        await asyncio.Event().wait()

    provider.terminate.side_effect = cleanup
    manager.inst_map["provider"] = provider
    manager.provider_insts.extend([provider, provider])
    manager.curr_provider_inst = provider
    task = asyncio.create_task(manager.terminate_provider("provider"))
    await asyncio.wait_for(entered.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert await manager.get_provider_by_id("provider") is None
    assert manager.provider_insts == []
    assert manager.curr_provider_inst is None
    await manager.terminate_provider("provider")
    provider.terminate.assert_awaited_once()


@pytest.mark.asyncio
async def test_all_stale_current_pointers_detach_by_identity(provider_manager):
    manager = provider_manager
    provider = make_provider()
    manager.inst_map["provider"] = provider
    for name in POINTERS:
        setattr(manager, name, provider)
    await manager.terminate_provider("provider")
    assert all(getattr(manager, name) is None for name in POINTERS)


@pytest.mark.asyncio
async def test_normal_update_reload_and_delete_keep_existing_lock_order(
    provider_manager,
):
    manager = provider_manager
    old = make_provider()
    replacement = make_provider()
    manager.inst_map["provider"] = old
    manager.provider_insts.append(old)
    manager.curr_provider_inst = old

    async def load(config):
        assert manager.resource_lock.locked()
        assert manager.reload_lock.locked()
        assert config["id"] not in manager.inst_map
        old.terminate.assert_awaited_once()
        manager.inst_map["provider"] = replacement
        manager.provider_insts.append(replacement)

    manager.load_provider = AsyncMock(side_effect=load)
    await asyncio.wait_for(
        manager.update_provider("provider", {"id": "provider", "enable": True}), 1
    )
    assert (
        await manager.get_using_provider_async(ProviderType.CHAT_COMPLETION)
        is replacement
    )
    assert manager.curr_provider_inst is replacement
    assert not manager.resource_lock.locked() and not manager.reload_lock.locked()

    async def cleanup():
        assert manager.resource_lock.locked()
        assert not manager.reload_lock.locked()
        assert await manager.get_provider_by_id("provider") is None

    replacement.terminate.side_effect = cleanup
    await asyncio.wait_for(manager.delete_provider(provider_id="provider"), 1)
    assert manager.acm.default_conf["provider"] == []
    assert manager.providers_config == []
    assert manager.inst_map == {}
    assert manager.provider_insts == []
    assert manager.curr_provider_inst is None
    assert not manager.resource_lock.locked() and not manager.reload_lock.locked()
    replacement.terminate.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [RuntimeError, asyncio.CancelledError])
async def test_failed_reload_does_not_restore_closed_provider(
    provider_manager, failure
):
    manager = provider_manager
    provider = make_provider()
    manager.inst_map["provider"] = provider
    manager.provider_insts.append(provider)
    manager.curr_provider_inst = provider
    provider.terminate.side_effect = failure("cannot finish cleanup")
    manager.load_provider = AsyncMock()

    with pytest.raises(failure):
        await manager.update_provider("provider", {"id": "provider", "enable": True})

    assert await manager.get_using_provider_async(ProviderType.CHAT_COMPLETION) is None
    assert manager.inst_map == {}
    assert manager.provider_insts == []
    manager.load_provider.assert_not_awaited()
    assert not manager.resource_lock.locked() and not manager.reload_lock.locked()
