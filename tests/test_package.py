"""The package imports and exposes its version."""

from __future__ import annotations

from qlora_extraction import __version__


def test_version() -> None:
    assert __version__
