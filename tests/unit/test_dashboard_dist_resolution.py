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
    (root / "index.html").write_text("<html></html>", encoding="utf-8")
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

    def test_mismatched_version_warns_but_is_still_served(
        self, tmp_path, astrbot_caplog
    ):
        """A stale packaged WebUI must not be swapped in silently."""
        dist = _make_dist(tmp_path / "webui", "v0.0.1")

        with astrbot_caplog.at_level(logging.WARNING):
            resolved = resolve_dashboard_dist(dist)

        assert resolved is not None  # behaviour unchanged: still served
        assert WARNING_FRAGMENT in astrbot_caplog.text
        assert "v0.0.1" in astrbot_caplog.text
        assert VERSION in astrbot_caplog.text

    def test_missing_version_marker_warns_as_unknown(self, tmp_path, astrbot_caplog):
        """Assets without a version marker cannot be verified, so say so."""
        dist = _make_dist(tmp_path / "webui", None)

        with astrbot_caplog.at_level(logging.WARNING):
            resolved = resolve_dashboard_dist(dist)

        assert resolved is not None
        assert WARNING_FRAGMENT in astrbot_caplog.text
        assert "unknown" in astrbot_caplog.text

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
