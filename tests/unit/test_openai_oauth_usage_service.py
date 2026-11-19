import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import httpx

from astrbot.core.config.default import CONFIG_METADATA_3, DEFAULT_CONFIG
from astrbot.core.provider.oauth.openai_oauth_shared_state import OpenAIOAuthSharedState
from astrbot.core.provider.oauth.openai_oauth_usage import QuotaReader
from astrbot.core.provider.oauth.openai_oauth_usage_service import (
    OpenAIOAuthUsageService,
)


def make_source(source_id="oauth-one", **changes):
    source = {
        "id": source_id,
        "type": "openai_oauth_chat_completion",
        "enable": True,
        "auth_mode": "openai_oauth",
        "api_base": "https://chatgpt.com/backend-api/codex",
        "proxy": "http://source-proxy:8080",
        "oauth_access_token": "secret-token",
        "oauth_account_id": "account-one",
    }
    source.update(changes)
    return source


class Manager:
    def __init__(self):
        self._sources = [make_source()]
        self.sources_reads = 0
        self.providers_config = []
        self.states = {}

    @property
    def provider_sources_config(self):
        self.sources_reads += 1
        return self._sources

    def get_openai_oauth_shared_state(self, source_id, source):
        if source_id not in self.states:
            self.states[source_id] = OpenAIOAuthSharedState(source_id, source)
        return self.states[source_id]


class UsageServiceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.manager = Manager()
        self.profiles = {
            "test:FriendMessage:1": {
                "admins_id": ["admin-1"],
                "provider_settings": {
                    "codex_oauth_usage": {"enabled": True, "source_id": ""}
                },
            },
            "test:GroupMessage:2": {
                "admins_id": ["admin-2"],
                "provider_settings": {
                    "codex_oauth_usage": {"enabled": True, "source_id": ""}
                },
            },
        }
        self.context = SimpleNamespace(
            provider_manager=self.manager,
            get_config=Mock(side_effect=lambda *, umo: self.profiles[umo]),
            get_using_provider_async=AsyncMock(
                side_effect=AssertionError("chat model queried")
            ),
            get_provider_by_id=Mock(side_effect=AssertionError("chat model queried")),
        )
        self.reader = SimpleNamespace(
            read_source=AsyncMock(return_value={"status": "success", "windows": []}),
            close=AsyncMock(),
        )
        self.service = OpenAIOAuthUsageService(self.context, self.reader)
        self.sender_id = "admin-1"
        self.event = SimpleNamespace(
            is_admin=lambda: True,
            get_sender_id=lambda: self.sender_id,
            unified_msg_origin="test:FriendMessage:1",
        )

    def settings(self, origin="test:FriendMessage:1"):
        return self.profiles[origin]["provider_settings"]["codex_oauth_usage"]

    def test_hidden_source_setting_replaces_visible_model_setting(self):
        self.assertEqual(
            DEFAULT_CONFIG["provider_settings"]["codex_oauth_usage"],
            {"enabled": True, "source_id": ""},
        )
        items = CONFIG_METADATA_3["ai_group"]["metadata"]["ai"]["items"]
        self.assertFalse(any("codex_oauth_usage" in key for key in items))

    async def test_single_source_works_without_any_chat_model(self):
        self.assertEqual((await self.service.run(self.event))["status"], "success")
        self.reader.read_source.assert_awaited_once()
        source, state = self.reader.read_source.await_args.args
        self.assertIs(source, self.manager._sources[0])
        self.assertIs(state, self.manager.states["oauth-one"])
        self.context.get_using_provider_async.assert_not_awaited()
        self.context.get_provider_by_id.assert_not_called()

    async def test_multiple_sources_require_explicit_source_id(self):
        self.manager._sources.append(make_source("oauth-two"))
        self.assertEqual(
            (await self.service.run(self.event))["status"], "source_ambiguous"
        )
        self.reader.read_source.assert_not_awaited()
        self.settings()["source_id"] = "oauth-two"
        self.assertEqual((await self.service.run(self.event))["status"], "success")
        self.assertIs(
            self.reader.read_source.await_args.args[0], self.manager._sources[1]
        )

    async def test_disabled_source_is_ignored_for_auto_and_rejected_if_selected(self):
        self.manager._sources.append(make_source("oauth-two", enable=False))
        self.assertEqual((await self.service.run(self.event))["status"], "success")
        self.settings()["source_id"] = "oauth-two"
        self.assertEqual(
            (await self.service.run(self.event))["status"], "source_unavailable"
        )

    async def test_malformed_enable_values_fail_closed_and_missing_defaults_enabled(
        self,
    ):
        for value in (0, None, "false"):
            self.manager._sources[0]["enable"] = value
            self.assertEqual(
                (await self.service.run(self.event))["status"], "source_not_found"
            )
            self.settings()["source_id"] = "oauth-one"
            self.assertEqual(
                (await self.service.run(self.event))["status"], "source_unavailable"
            )
            self.settings()["source_id"] = ""
        self.manager._sources[0].pop("enable")
        self.assertEqual((await self.service.run(self.event))["status"], "success")

    async def test_missing_wrong_type_and_duplicate_ids_fail_closed(self):
        self.settings()["source_id"] = "missing"
        self.assertEqual(
            (await self.service.run(self.event))["status"], "source_not_found"
        )
        self.settings()["source_id"] = "oauth-one"
        self.manager._sources[0]["type"] = "openai_chat_completion"
        self.assertEqual(
            (await self.service.run(self.event))["status"], "source_unavailable"
        )
        self.manager._sources[0]["type"] = "openai_oauth_chat_completion"
        self.manager._sources.append(make_source("oauth-one", enable=False))
        self.assertEqual(
            (await self.service.run(self.event))["status"], "source_ambiguous"
        )
        self.reader.read_source.assert_not_awaited()

    async def test_legacy_model_id_uses_only_persisted_source_mapping(self):
        self.settings().pop("source_id")
        self.settings()["provider_id"] = "legacy/model"
        self.manager.providers_config = [
            {"id": "legacy/model", "provider_source_id": "oauth-one"}
        ]
        self.assertEqual((await self.service.run(self.event))["status"], "success")
        self.assertIs(
            self.reader.read_source.await_args.args[0], self.manager._sources[0]
        )
        self.manager.providers_config = []
        self.assertEqual(
            (await self.service.run(self.event))["status"], "source_not_found"
        )

    async def test_explicit_source_id_overrides_stale_legacy_model_id(self):
        self.settings()["source_id"] = "oauth-one"
        self.settings()["provider_id"] = "missing/model"
        self.assertEqual((await self.service.run(self.event))["status"], "success")

    async def test_non_admin_cannot_enumerate_sources(self):
        self.event.is_admin = lambda: False
        self.assertEqual((await self.service.run(self.event))["status"], "not_admin")
        self.assertEqual(self.manager.sources_reads, 0)
        self.reader.read_source.assert_not_awaited()

    async def test_profile_admin_lists_isolate_group_and_private_chat(self):
        self.assertEqual((await self.service.run(self.event))["status"], "success")
        self.event.unified_msg_origin = "test:GroupMessage:2"
        self.assertEqual((await self.service.run(self.event))["status"], "not_admin")
        self.sender_id = "admin-2"
        self.assertEqual((await self.service.run(self.event))["status"], "success")

    async def test_source_removed_or_disabled_during_request_hides_old_result(self):
        for change in ("remove", "disable"):
            self.manager._sources = [make_source()]
            self.manager.states = {}

            async def read_source(_source, _state):
                if change == "remove":
                    self.manager._sources.clear()
                else:
                    self.manager._sources[0]["enable"] = False
                return {"status": "success"}

            self.reader.read_source.side_effect = read_source
            self.assertEqual(
                (await self.service.run(self.event))["status"], "authorization_changed"
            )

    async def test_account_rotation_during_request_hides_old_result(self):
        async def read_source(_source, state):
            state.apply(
                {
                    "oauth_access_token": "rotated-secret",
                    "oauth_account_id": "account-two",
                }
            )
            return {"status": "success"}

        self.reader.read_source.side_effect = read_source
        self.assertEqual(
            (await self.service.run(self.event))["status"], "authorization_changed"
        )

    async def test_token_refresh_for_same_account_keeps_result(self):
        async def read_source(_source, state):
            state.apply({"oauth_access_token": "refreshed-token"})
            return {"status": "success"}

        self.reader.read_source.side_effect = read_source
        self.assertEqual((await self.service.run(self.event))["status"], "success")

    async def test_source_proxy_change_during_request_hides_old_result(self):
        async def read_source(_source, _state):
            self.manager._sources[0]["proxy"] = "http://new-proxy:8080"
            return {"status": "success"}

        self.reader.read_source.side_effect = read_source
        self.assertEqual(
            (await self.service.run(self.event))["status"], "authorization_changed"
        )

    async def test_same_id_source_replacement_during_request_hides_old_result(self):
        async def read_source(_source, _state):
            self.manager._sources[0] = make_source("oauth-one")
            return {"status": "success"}

        self.reader.read_source.side_effect = read_source
        self.assertEqual(
            (await self.service.run(self.event))["status"], "authorization_changed"
        )

    async def test_shared_state_replacement_during_request_hides_old_result(self):
        async def read_source(source, _state):
            self.manager.states["oauth-one"] = OpenAIOAuthSharedState(
                "oauth-one", source
            )
            return {"status": "success"}

        self.reader.read_source.side_effect = read_source
        self.assertEqual(
            (await self.service.run(self.event))["status"], "authorization_changed"
        )

    async def test_source_setting_change_during_request_hides_old_result(self):
        async def read_source(_source, _state):
            self.settings()["source_id"] = "oauth-two"
            return {"status": "success"}

        self.reader.read_source.side_effect = read_source
        self.assertEqual(
            (await self.service.run(self.event))["status"], "authorization_changed"
        )

    async def test_profile_admin_revoked_during_request_hides_old_result(self):
        async def read_source(_source, _state):
            self.profiles["test:FriendMessage:1"]["admins_id"] = []
            return {"status": "success"}

        self.reader.read_source.side_effect = read_source
        self.assertEqual((await self.service.run(self.event))["status"], "not_admin")

    async def test_disabled_and_closed_do_not_query_sources(self):
        self.settings()["enabled"] = False
        self.assertEqual((await self.service.run(self.event))["status"], "disabled")
        self.settings()["enabled"] = True
        await self.service.close()
        self.assertEqual((await self.service.run(self.event))["status"], "closed")
        self.assertEqual(self.manager.sources_reads, 0)
        self.reader.close.assert_awaited_once()

    async def test_reader_failure_status_is_forwarded_without_body(self):
        self.reader.read_source.return_value = {
            "status": "rate_limited",
            "http_status": 429,
        }
        self.assertEqual(
            await self.service.run(self.event),
            {"status": "rate_limited", "http_status": 429},
        )

    async def exercise_401(
        self, second=200, rotate=None, exchange_error=None, before_second_response=None
    ):
        source = self.manager._sources[0]
        source.update(oauth_refresh_token="refresh-one")
        state = self.manager.get_openai_oauth_shared_state(source["id"], source)
        headers = []

        async def handler(request):
            headers.append(request.headers["authorization"])
            if len(headers) == 1:
                if rotate:
                    rotate(state)
                return httpx.Response(401)
            if before_second_response:
                before_second_response(state)
            return httpx.Response(second, json={"plan_type": "pro"})

        transport = httpx.MockTransport(handler)
        self.service.reader = QuotaReader(
            client_factory=lambda **kwargs: httpx.AsyncClient(transport=transport)
        )

        async def persist(_id, patch, **guard):
            self.assertIs(guard["expected_source"], source)
            source.update(patch)

        self.manager._persist_openai_oauth_provider_source_patch = AsyncMock(
            side_effect=persist
        )
        exchange = AsyncMock(
            return_value={"access_token": "fresh", "refresh_token": "refresh-two"},
            side_effect=exchange_error,
        )
        with patch(
            "astrbot.core.provider.oauth.openai_oauth.refresh_access_token",
            exchange,
        ):
            result = await self.service.run(self.event)
        return result, headers, exchange

    async def test_401_refreshes_persists_and_retries_once(self):
        result, headers, exchange = await self.exercise_401()
        self.assertEqual(result["status"], "success")
        self.assertEqual(headers, ["Bearer secret-token", "Bearer fresh"])
        exchange.assert_awaited_once_with("refresh-one", "http://source-proxy:8080")
        self.assertEqual(self.manager._sources[0]["oauth_refresh_token"], "refresh-two")

    async def test_second_401_keeps_public_error(self):
        result, headers, exchange = await self.exercise_401(second=401)
        self.assertEqual(result, {"status": "reauth_required", "http_status": 401})
        self.assertEqual(len(headers), 2)
        exchange.assert_awaited_once()

    async def test_401_with_concurrent_refresh_reuses_new_token(self):
        result, headers, exchange = await self.exercise_401(
            rotate=lambda state: state.apply({"oauth_access_token": "concurrent"})
        )
        self.assertEqual(result["status"], "success")
        self.assertEqual(headers, ["Bearer secret-token", "Bearer concurrent"])
        exchange.assert_not_awaited()

    async def test_401_after_admin_revoked_never_refreshes(self):
        result, headers, exchange = await self.exercise_401(
            rotate=lambda state: self.profiles[self.event.unified_msg_origin].update(
                admins_id=[]
            )
        )
        self.assertEqual(result, {"status": "not_admin"})
        exchange.assert_not_awaited()
        self.assertEqual(len(headers), 1)

    async def test_exchange_failure_hides_secret_and_never_retries(self):
        result, headers, exchange = await self.exercise_401(
            exchange_error=ValueError("private upstream details")
        )
        self.assertEqual(result, {"status": "unavailable"})
        self.assertEqual(len(headers), 1)

    async def test_retry_rotation_discards_response_without_third_request(self):
        result, headers, exchange = await self.exercise_401(
            before_second_response=lambda state: state.apply(
                {"oauth_access_token": "another-token"}
            )
        )
        self.assertEqual(result, {"status": "authorization_changed"})
        self.assertEqual(len(headers), 2)
        exchange.assert_awaited_once()
