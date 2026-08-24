"""Regression test for Production Hardening Bug #1.

run.py:_build_confidence accessed `options.available`, but `options` is an
`OptionsFlowProfile` which has no `available` attribute. The authoritative
availability marker on that object is `source` ("none" == unavailable,
otherwise the provider name). This test exercises the confidence-building path
with a real OptionsFlowProfile so it fails on the old code and passes on the fix.
"""
import types

from run import HunterOrchestrator
from models.options import OptionsFlowProfile
from core.data_confidence import DataQuality


def _build_confidence_for(opts):
    # _build_confidence does not reference `self`, so we pass None as self to
    # avoid constructing the full orchestrator (providers/memory/telegram).
    data = types.SimpleNamespace(
        ticker="AAPL",
        current_price=100.0,
        previous_close=99.0,
        premarket=types.SimpleNamespace(is_complete=False),
        regular=types.SimpleNamespace(is_complete=False),
    )
    event = types.SimpleNamespace(
        primary_source=types.SimpleNamespace(published_at="2026-08-24T00:00:00Z")
    )
    technical = types.SimpleNamespace(ma20=100.0)
    return HunterOrchestrator._build_confidence(None, data, event, technical, opts)


def _options_chain_field(report):
    return [f for f in report.fields if f.name == "options_chain"][0]


def test_build_confidence_options_available_no_crash():
    opts = OptionsFlowProfile(source="yfinance", confidence=70)
    report = _build_confidence_for(opts)
    assert _options_chain_field(report).quality == DataQuality.PROXY


def test_build_confidence_options_unavailable_no_crash():
    opts = OptionsFlowProfile(source="none", confidence=0, notes=["Options chain unavailable"])
    report = _build_confidence_for(opts)
    assert _options_chain_field(report).quality == DataQuality.MISSING


def test_build_confidence_options_missing_snapshot_no_crash():
    # Current architecture: OptionsEngine always returns a profile (never None)
    # for missing data, with source="none". This represents the missing/None case.
    opts = OptionsFlowProfile(source="none")
    report = _build_confidence_for(opts)
    assert _options_chain_field(report).quality == DataQuality.MISSING
