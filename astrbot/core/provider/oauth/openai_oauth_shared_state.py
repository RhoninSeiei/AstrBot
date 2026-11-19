import asyncio
import inspect
from typing import Any

OPENAI_OAUTH_CREDENTIAL_FIELDS = (
    "auth_mode",
    "oauth_provider",
    "oauth_access_token",
    "oauth_refresh_token",
    "oauth_expires_at",
    "oauth_account_email",
    "oauth_account_id",
)

_VERSION_FIELDS = {
    "oauth_access_token",
    "oauth_refresh_token",
    "oauth_expires_at",
    "oauth_account_id",
}


class OpenAIOAuthSharedState:
    """Runtime credentials shared by models from one OpenAI OAuth source."""

    def __init__(self, source_id: str, source_config: dict[str, Any]) -> None:
        self.source_id = source_id
        self.refresh_lock = asyncio.Lock()
        self.models_lock = asyncio.Lock()
        self.models_cache: list[str] = []
        self.models_metadata: dict[str, dict[str, int]] = {}
        self.models_expires_at = 0.0
        self.models_cache_key: tuple[str, str] | None = None
        self.version = 0
        self._credentials: dict[str, Any] = dict.fromkeys(
            OPENAI_OAUTH_CREDENTIAL_FIELDS,
            "",
        )
        self.replace(source_config)

    def apply(self, patch: dict[str, Any]) -> int:
        if (
            "oauth_account_id" in patch
            and self._credentials.get("oauth_account_id") != patch["oauth_account_id"]
        ):
            self.clear_models_cache()
        version_changed = False
        for key in OPENAI_OAUTH_CREDENTIAL_FIELDS:
            if key not in patch:
                continue
            value = patch[key]
            if key in _VERSION_FIELDS and self._credentials.get(key) != value:
                version_changed = True
            self._credentials[key] = value
        if version_changed:
            self.version += 1
        return self.version

    def replace(self, source_config: dict[str, Any]) -> int:
        next_credentials = {
            key: source_config.get(key, "") for key in OPENAI_OAUTH_CREDENTIAL_FIELDS
        }
        if (
            self._credentials.get("oauth_account_id")
            != next_credentials["oauth_account_id"]
        ):
            self.clear_models_cache()
        if any(
            self._credentials.get(key) != next_credentials[key]
            for key in _VERSION_FIELDS
        ):
            self.version += 1
        self._credentials = next_credentials
        return self.version

    def apply_token(self, token: dict[str, Any]) -> None:
        patch = {
            "oauth_expires_at": str(token.get("expires_at") or ""),
            "oauth_account_email": token.get("email")
            or self._credentials["oauth_account_email"],
            "oauth_account_id": token.get("account_id")
            or self._credentials["oauth_account_id"],
        }
        for field in ("access_token", "refresh_token"):
            value = str(token.get(field) or "").strip()
            if value:
                patch[f"oauth_{field}"] = value
        self.apply(patch)

    async def refresh_locked(self, exchange, proxy, persist) -> bool:
        """Exchange the current refresh token while the caller holds refresh_lock."""
        version, credentials = self.versioned_snapshot()
        refresh_token = str(credentials.get("oauth_refresh_token") or "").strip()
        if not refresh_token:
            return False
        token = await exchange(refresh_token, proxy)
        if self.version != version:
            return True
        self.apply_token(token)
        if callable(persist):
            patch = self.snapshot()
            patch.update(auth_mode="openai_oauth", oauth_provider="openai")
            result = persist(patch)
            if inspect.isawaitable(result):
                await result
        return True

    def clear_models_cache(self) -> None:
        """Discard account/backend-specific model names and numeric metadata."""
        self.models_cache = []
        self.models_metadata = {}
        self.models_expires_at = 0.0
        self.models_cache_key = None

    def snapshot(self) -> dict[str, Any]:
        return dict(self._credentials)

    def versioned_snapshot(self) -> tuple[int, dict[str, Any]]:
        return self.version, dict(self._credentials)
