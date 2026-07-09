"""Tests for app/models/result_models.py"""

from app.models.result_models import DiscoveryResult, GenerateResult


class TestGenerateResult:
    def test_create_minimal(self):
        result = GenerateResult(text="Hello")
        assert result.text == "Hello"
        assert result.cost_usd == 0.0
        assert result.cached is False
        assert result.model_family == "unknown"
        assert result.sentiment == "NEUTRAL"

    def test_create_full(self):
        result = GenerateResult(
            text="Analysis complete",
            tokens={"input": 100, "output": 50},
            cost_usd=0.005,
            model_family="gpt-4",
            cached=True,
            sentiment="POSITIVE",
        )
        assert result.tokens == {"input": 100, "output": 50}
        assert result.cost_usd == 0.005
        assert result.cached is True
        assert result.sentiment == "POSITIVE"

    def test_to_dict(self):
        result = GenerateResult(text="Test", cost_usd=0.01)
        d = result.to_dict()
        assert isinstance(d, dict)
        assert d["text"] == "Test"
        assert d["cost_usd"] == 0.01

    def test_sentiment_negative(self):
        result = GenerateResult(text="Bad output", sentiment="NEGATIVE")
        assert result.sentiment == "NEGATIVE"


class TestDiscoveryResult:
    def test_create_ok(self):
        result = DiscoveryResult(ok=True, site="farfetch", query="gucci bag")
        assert result.ok is True
        assert result.site == "farfetch"
        assert result.query == "gucci bag"
        assert result.proposal == {}
        assert result.message is None

    def test_create_failure(self):
        result = DiscoveryResult(
            ok=False,
            site="moncler",
            query="jacket",
            message="Timeout error",
        )
        assert result.ok is False
        assert result.message == "Timeout error"

    def test_create_with_ai_analysis(self):
        ai = GenerateResult(text="Found 5 products", cost_usd=0.002)
        result = DiscoveryResult(
            ok=True,
            site="gucci",
            query="belt",
            ai_analysis=ai,
        )
        assert result.ai_analysis is not None
        assert result.ai_analysis.text == "Found 5 products"

    def test_to_dict_basic(self):
        result = DiscoveryResult(ok=True, site="test", query="q")
        d = result.to_dict()
        assert isinstance(d, dict)
        assert d["ok"] is True
        assert d["site"] == "test"

    def test_to_dict_with_ai_analysis(self):
        ai = GenerateResult(text="Analysis", cost_usd=0.001)
        result = DiscoveryResult(ok=True, site="s", query="q", ai_analysis=ai)
        d = result.to_dict()
        assert "ai_analysis" in d
        assert d["ai_analysis"]["text"] == "Analysis"

    def test_proposal_dict(self):
        result = DiscoveryResult(
            ok=True,
            site="test",
            query="q",
            proposal={"selector": ".product-card", "count": 10},
        )
        assert result.proposal["selector"] == ".product-card"
