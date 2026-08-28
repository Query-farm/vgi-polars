# Copyright 2026 Query Farm LLC - https://query.farm

"""`attach()`'s oauth=/oauth_refresh_token=/oauth_flow=/... kwargs reach `Client.from_http` correctly.

The OAuth mechanism itself (device-code polling, token refresh, the
VgiOAuthAuth state machine) is exhaustively tested where it actually lives —
vgi-rpc-python's tests/test_oauth_client.py (mock-IdP-driven) and
vgi-python's tests/test_client_oauth.py. This file's job is narrower: prove
`attach()` forwards these parameters to `Client.from_http(...)` unchanged,
by patching `Client.from_http` and inspecting the call it received rather
than standing up a real OAuth-gated worker + mock IdP.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import vgi_polars as vp


def test_oauth_kwargs_reach_client_from_http() -> None:
    """attach(oauth=True, oauth_flow=..., ...) forwards every oauth kwarg to Client.from_http."""
    with patch("vgi_polars.catalog.Client") as mock_client_cls:
        mock_client_cls.from_http.return_value = MagicMock()
        cat = vp.attach(
            "https://worker.example.com",
            name="example",
            oauth=True,
            oauth_flow="device_code",
            oauth_timeout_seconds=42.0,
            oauth_prompt="consent",
        )
        cat.detach()

    assert mock_client_cls.from_http.call_count == 1
    _, kwargs = mock_client_cls.from_http.call_args
    assert kwargs["oauth"] is True
    assert kwargs["oauth_refresh_token"] is None
    assert kwargs["oauth_flow"] == "device_code"
    assert kwargs["oauth_timeout_seconds"] == 42.0
    assert kwargs["oauth_prompt"] == "consent"
    assert kwargs["bearer_token"] is None


def test_oauth_refresh_token_reaches_client_from_http() -> None:
    """oauth_refresh_token= forwards through unchanged (Client itself infers oauth=True from it)."""
    with patch("vgi_polars.catalog.Client") as mock_client_cls:
        mock_client_cls.from_http.return_value = MagicMock()
        cat = vp.attach(
            "https://worker.example.com",
            name="example",
            oauth_refresh_token="pre-seeded-token",
        )
        cat.detach()

    _, kwargs = mock_client_cls.from_http.call_args
    assert kwargs["oauth_refresh_token"] == "pre-seeded-token"
    assert kwargs["oauth"] is False  # attach() forwards the literal flag; Client does the implying


def test_oauth_defaults_are_inert() -> None:
    """With no oauth kwargs passed, attach() still forwards the (harmless) defaults."""
    with patch("vgi_polars.catalog.Client") as mock_client_cls:
        mock_client_cls.from_http.return_value = MagicMock()
        cat = vp.attach("https://worker.example.com", name="example")
        cat.detach()

    _, kwargs = mock_client_cls.from_http.call_args
    assert kwargs["oauth"] is False
    assert kwargs["oauth_refresh_token"] is None
    assert kwargs["oauth_flow"] == "auto"
    assert kwargs["oauth_prompt"] == "none"


def test_client_property_exposes_oauth_identity() -> None:
    """catalog.client.oauth_identity() is reachable -- no VgiCatalog-level wrapper needed."""
    with patch("vgi_polars.catalog.Client") as mock_client_cls:
        fake_client = MagicMock()
        fake_client.oauth_identity.return_value = None
        mock_client_cls.from_http.return_value = fake_client
        cat = vp.attach("https://worker.example.com", name="example", oauth=True)
        assert cat.client.oauth_identity() is None
        cat.detach()
