"""Tests for resolve_dashboard_dist() when an explicit WebUI directory is used."""

import logging

import pytest

from astrbot import logger
from astrbot.core.config.default import VERSION
from astrbot.core.dashboard_assets import resolve_dashboard_dist

WARNING_FRAGMENT = "does not declare a version matching core"


@pytest.fixture
def astrbot_caplog(caplog):
    added_handler = caplog.handler not in logger.handlers
    if added_handler:
        logger.addHandler(caplog.handler)
    try:
        yield caplog
    finally:
        if added_handler:
            logger.removeHandler(caplog.handler)


def _make_dist(root, version: str | None) -> str:
    assets = root / "assets"
    assets.mkdir(parents=True)
    (root / "index.html").write_text(
        '<html><script type="module" src="/assets/app.js"></script></html>',
        encoding="utf-8",
    )
    (assets / "app.js").write_text("export {};", encoding="utf-8")
    if version is not None:
        (assets / "version").write_text(version, encoding="utf-8")
    return str(root)


class TestExplicitWebuiDir:
    def test_matching_version_is_served_quietly(self, tmp_path, astrbot_caplog):
        """The happy path must not add startup noise."""
        dist = _make_dist(tmp_path / "webui", f"v{VERSION}")

        with astrbot_caplog.at_level(logging.WARNING):
            resolved = resolve_dashboard_dist(dist)

        assert resolved is not None
        assert str(resolved) == str(tmp_path / "webui")
        assert WARNING_FRAGMENT not in astrbot_caplog.text

    def test_non_desktop_mismatched_version_warns_but_is_still_served(
        self, monkeypatch, tmp_path, astrbot_caplog
    ):
        """A custom WebUI remains supported outside the managed desktop app."""
        monkeypatch.delenv("ASTRBOT_DESKTOP_MANAGED", raising=False)
        dist = _make_dist(tmp_path / "webui", "v0.0.1")

        with astrbot_caplog.at_level(logging.WARNING):
            resolved = resolve_dashboard_dist(dist)

        assert resolved is not None  # behaviour unchanged: still served
        assert WARNING_FRAGMENT in astrbot_caplog.text
        assert "v0.0.1" in astrbot_caplog.text
        assert VERSION in astrbot_caplog.text

    def test_desktop_missing_version_marker_is_rejected(
        self, monkeypatch, tmp_path, astrbot_caplog
    ):
        """Desktop assets without a verifiable version must not be served."""
        monkeypatch.setenv("ASTRBOT_DESKTOP_MANAGED", "1")
        dist = _make_dist(tmp_path / "webui", None)

        with astrbot_caplog.at_level(logging.WARNING):
            resolved = resolve_dashboard_dist(dist)

        assert resolved is None
        assert "refusing" in astrbot_caplog.text.lower()
        assert "unknown" in astrbot_caplog.text

    def test_non_desktop_missing_version_marker_warns_as_unknown(
        self, monkeypatch, tmp_path, astrbot_caplog
    ):
        monkeypatch.delenv("ASTRBOT_DESKTOP_MANAGED", raising=False)
        dist = _make_dist(tmp_path / "webui", None)

        with astrbot_caplog.at_level(logging.WARNING):
            resolved = resolve_dashboard_dist(dist)

        assert resolved == (tmp_path / "webui").absolute()
        assert WARNING_FRAGMENT in astrbot_caplog.text
        assert "unknown" in astrbot_caplog.text

    def test_desktop_missing_entry_asset_is_rejected(
        self, monkeypatch, tmp_path, astrbot_caplog
    ):
        """A marker cannot make a partially installed bundle compatible."""
        monkeypatch.setenv("ASTRBOT_DESKTOP_MANAGED", "1")
        dist = _make_dist(tmp_path / "webui", f"v{VERSION}")
        (tmp_path / "webui" / "assets" / "app.js").unlink()

        with astrbot_caplog.at_level(logging.WARNING):
            resolved = resolve_dashboard_dist(dist)

        assert resolved is None
        assert "incomplete" in astrbot_caplog.text.lower()

    def test_desktop_malformed_entry_url_is_rejected(
        self, monkeypatch, tmp_path, astrbot_caplog
    ):
        """A malformed asset URL must not abort backend startup."""
        monkeypatch.setenv("ASTRBOT_DESKTOP_MANAGED", "1")
        dist = _make_dist(tmp_path / "webui", f"v{VERSION}")
        (tmp_path / "webui" / "index.html").write_text(
            '<script src="http://["></script>',
            encoding="utf-8",
        )

        with astrbot_caplog.at_level(logging.WARNING):
            resolved = resolve_dashboard_dist(dist)

        assert resolved is None
        assert "incomplete" in astrbot_caplog.text.lower()

    def test_desktop_mismatched_version_uses_matching_managed_dist(
        self, monkeypatch, tmp_path, astrbot_caplog
    ):
        """A managed desktop backend must replace a known stale explicit dist."""
        monkeypatch.setenv("ASTRBOT_DESKTOP_MANAGED", "1")
        explicit_dist = _make_dist(tmp_path / "webui", "v0.0.1")
        data_dir = tmp_path / "data"
        managed_dist = _make_dist(data_dir / "dist", f"v{VERSION}")
        monkeypatch.setattr(
            "astrbot.core.dashboard_assets.get_astrbot_data_path",
            lambda: str(data_dir),
        )
        monkeypatch.setattr(
            "astrbot.core.dashboard_assets._get_bundled_dist_path",
            lambda: tmp_path / "missing-bundled-dist",
        )

        with astrbot_caplog.at_level(logging.WARNING):
            resolved = resolve_dashboard_dist(explicit_dist)

        assert resolved == (tmp_path / "data" / "dist").absolute()
        assert str(managed_dist) == str(resolved)
        assert "v0.0.1" in astrbot_caplog.text
        assert "refusing" in astrbot_caplog.text.lower()

    def test_desktop_mismatched_version_without_fallback_is_not_served(
        self, monkeypatch, tmp_path, astrbot_caplog
    ):
        """Known stale desktop assets must never become the final fallback."""
        monkeypatch.setenv("ASTRBOT_DESKTOP_MANAGED", "1")
        explicit_dist = _make_dist(tmp_path / "webui", "v0.0.1")
        data_dir = tmp_path / "data"
        _make_dist(data_dir / "dist", "v0.0.2")
        monkeypatch.setattr(
            "astrbot.core.dashboard_assets.get_astrbot_data_path",
            lambda: str(data_dir),
        )
        monkeypatch.setattr(
            "astrbot.core.dashboard_assets._get_bundled_dist_path",
            lambda: tmp_path / "missing-bundled-dist",
        )

        with astrbot_caplog.at_level(logging.WARNING):
            resolved = resolve_dashboard_dist(explicit_dist)

        assert resolved is None
        assert "refusing" in astrbot_caplog.text.lower()

    def test_nonexistent_dir_falls_through(self, tmp_path, astrbot_caplog):
        """A path that does not exist must not be reported as a stale dist."""
        with astrbot_caplog.at_level(logging.WARNING):
            resolve_dashboard_dist(str(tmp_path / "does-not-exist"))

        assert WARNING_FRAGMENT not in astrbot_caplog.text

    @pytest.mark.parametrize("empty", ["", None])
    def test_no_explicit_dir_falls_through(self, empty, astrbot_caplog):
        """Without --webui-dir the managed/bundled resolution path is used."""
        with astrbot_caplog.at_level(logging.WARNING):
            resolve_dashboard_dist(empty)

        assert WARNING_FRAGMENT not in astrbot_caplog.text
