"""Tests for Streamlit frontend smoke tests (Phase 15).

Streamlit apps are difficult to unit-test directly; these tests verify
the module imports cleanly and the core logic functions exist.
"""

import importlib

import pytest


class TestFrontendImport:
    def test_import_app(self):
        mod = importlib.import_module("frontend.app")
        assert mod is not None

    def test_streamlit_available(self):
        import streamlit
        assert hasattr(streamlit, "set_page_config")
