"""Issue #65: pricing rules.py カバレッジ 79→95%+

Tests for uncovered paths:
- L88-89: config_path None returns default
- L91-93: non-existent file returns default
- L95-96: JSON file loading
- L98-101: backward compat buyma_fee_rate alias
- L115-117: exception fallback
"""

from __future__ import annotations

import json

from app.core.pricing.rules import PricingConfig, load_pricing_config


class TestLoadPricingConfigDefault:
    def test_none_path_returns_default(self):
        result = load_pricing_config(None)
        assert isinstance(result, PricingConfig)
        assert result.additional_fee_rate == 0.0

    def test_empty_string_returns_default(self):
        result = load_pricing_config("")
        assert isinstance(result, PricingConfig)


class TestLoadPricingConfigMissingFile:
    def test_nonexistent_file_returns_default(self, tmp_path):
        result = load_pricing_config(str(tmp_path / "no_such_file.json"))
        assert isinstance(result, PricingConfig)
        assert result.additional_fee_rate == 0.0


class TestLoadPricingConfigValidJson:
    def test_valid_json_overrides_defaults(self, tmp_path):
        config_file = tmp_path / "pricing.json"
        config_file.write_text(
            json.dumps(
                {
                    "buyma_platform_fee_rate": 0.08,
                    "buyma_effective_fee_rate": 0.15,
                    "additional_fee_rate": 0.02,
                    "domestic_commission_rate": 0.05,
                    "overseas_commission_rate": 0.06,
                    "transfer_fee": 300,
                }
            ),
            encoding="utf-8",
        )
        result = load_pricing_config(str(config_file))
        assert result.buyma_platform_fee_rate == 0.08
        assert result.buyma_effective_fee_rate == 0.15
        assert result.additional_fee_rate == 0.02
        assert result.transfer_fee == 300

    def test_partial_json_uses_defaults(self, tmp_path):
        config_file = tmp_path / "partial.json"
        config_file.write_text(
            json.dumps({"transfer_fee": 500}),
            encoding="utf-8",
        )
        result = load_pricing_config(str(config_file))
        assert result.transfer_fee == 500
        assert result.additional_fee_rate == 0.0


class TestLoadPricingConfigBackwardCompat:
    def test_buyma_fee_rate_alias(self, tmp_path):
        config_file = tmp_path / "legacy.json"
        config_file.write_text(
            json.dumps({"buyma_fee_rate": 0.20}),
            encoding="utf-8",
        )
        result = load_pricing_config(str(config_file))
        assert result.buyma_effective_fee_rate == 0.20

    def test_effective_rate_takes_priority_over_legacy(self, tmp_path):
        config_file = tmp_path / "both.json"
        config_file.write_text(
            json.dumps(
                {
                    "buyma_fee_rate": 0.20,
                    "buyma_effective_fee_rate": 0.16,
                }
            ),
            encoding="utf-8",
        )
        result = load_pricing_config(str(config_file))
        assert result.buyma_effective_fee_rate == 0.16


class TestLoadPricingConfigExceptionFallback:
    def test_invalid_json_returns_default(self, tmp_path):
        config_file = tmp_path / "bad.json"
        config_file.write_text("not json at all{{{", encoding="utf-8")
        result = load_pricing_config(str(config_file))
        assert isinstance(result, PricingConfig)
        assert result.additional_fee_rate == 0.0

    def test_non_numeric_value_returns_default(self, tmp_path):
        config_file = tmp_path / "bad_type.json"
        config_file.write_text(
            json.dumps({"transfer_fee": "not_a_number"}),
            encoding="utf-8",
        )
        result = load_pricing_config(str(config_file))
        assert isinstance(result, PricingConfig)
