from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.core.pricing.rules import (
    CUSTOMS_RATE_TABLE,
    PricingConfig,
    _DEFAULT_CONFIG,
    load_pricing_config,
    resolve_customs_rate,
)


# =====================================================================
# resolve_customs_rate
# =====================================================================

class TestResolveCustomsRateMaterial:
    """material ベースの判定（優先度高）"""

    @pytest.mark.parametrize(
        "material",
        ["leather", "Leather", "LEATHER", "genuine leather", "Faux Leather"],
    )
    def test_material_leather_english(self, material: str) -> None:
        assert resolve_customs_rate(None, material) == CUSTOMS_RATE_TABLE["leather"]

    @pytest.mark.parametrize("material", ["革", "合革", "本革"])
    def test_material_leather_kanji(self, material: str) -> None:
        assert resolve_customs_rate(None, material) == CUSTOMS_RATE_TABLE["leather"]

    @pytest.mark.parametrize("material", ["レザー", "合レザー"])
    def test_material_leather_katakana(self, material: str) -> None:
        assert resolve_customs_rate(None, material) == CUSTOMS_RATE_TABLE["leather"]

    def test_material_priority_over_category(self) -> None:
        """素材とカテゴリが両方ある場合、素材が優先される"""
        assert resolve_customs_rate("shoes", "leather") == CUSTOMS_RATE_TABLE["leather"]


class TestResolveCustomsRateCategory:
    """category ベースの判定"""

    @pytest.mark.parametrize(
        "category, expected_key",
        [
            ("bag", "bag"),
            ("Bag", "bag"),
            ("BAG", "bag"),
            ("shoulder bag", "bag"),
            ("バッグ", "bag"),
            ("ハンドバッグ", "bag"),
            ("shoes", "shoes"),
            ("Shoes", "shoes"),
            ("靴", "shoes"),
            ("シューズ", "shoes"),
            ("wallet", "wallet"),
            ("Wallet", "wallet"),
            ("財布", "wallet"),
            ("長財布", "wallet"),
            ("watch", "watch"),
            ("Watch", "watch"),
            ("腕時計", "watch"),
            ("wrist watch", "watch"),
        ],
    )
    def test_category_lookups(self, category: str, expected_key: str) -> None:
        assert resolve_customs_rate(category, None) == CUSTOMS_RATE_TABLE[expected_key]


class TestResolveCustomsRateFallback:
    """デフォルトフォールバック"""

    @pytest.mark.parametrize(
        "category, material",
        [
            ("accessory", None),
            ("clothing", None),
            ("Accessories", None),
            ("jacket", None),
            (None, "cotton"),
            ("unknown", "metal"),
        ],
    )
    def test_default_returned(self, category: str | None, material: str | None) -> None:
        assert resolve_customs_rate(category, material) == CUSTOMS_RATE_TABLE["default"]

    def test_both_none(self) -> None:
        assert resolve_customs_rate(None, None) == CUSTOMS_RATE_TABLE["default"]

    def test_default_rate_value(self) -> None:
        assert CUSTOMS_RATE_TABLE["default"] == 0.10


# =====================================================================
# load_pricing_config – ファイルパス系
# =====================================================================

class TestLoadPricingConfigFilePaths:
    """config_path が未指定・None・空文字・存在しないパスの場合"""

    def test_none_returns_default(self) -> None:
        config = load_pricing_config(None)
        assert config == _DEFAULT_CONFIG

    def test_empty_string_returns_default(self) -> None:
        config = load_pricing_config("")
        assert config == _DEFAULT_CONFIG

    def test_nonexistent_path_returns_default(self, tmp_path: Path) -> None:
        config = load_pricing_config(str(tmp_path / "no_such_file.json"))
        assert config == _DEFAULT_CONFIG


# =====================================================================
# load_pricing_config – JSON パース正常系
# =====================================================================

class TestLoadPricingConfigJsonParsing:
    """正常な JSON 設定ファイルの読み込み"""

    def _write_json(self, path: Path, obj: dict) -> str:
        path.write_text(json.dumps(obj), encoding="utf-8")
        return str(path)

    def test_full_config(self, tmp_path: Path) -> None:
        fp = self._write_json(tmp_path / "full.json", {
            "buyma_platform_fee_rate": 0.05,
            "buyma_effective_fee_rate": 0.10,
            "additional_fee_rate": 0.02,
            "domestic_commission_rate": 0.06,
            "overseas_commission_rate": 0.04,
            "transfer_fee": 300.0,
        })
        cfg = load_pricing_config(fp)
        assert cfg == PricingConfig(
            buyma_platform_fee_rate=0.05,
            buyma_effective_fee_rate=0.10,
            additional_fee_rate=0.02,
            domestic_commission_rate=0.06,
            overseas_commission_rate=0.04,
            transfer_fee=300.0,
        )

    def test_partial_config_uses_defaults(self, tmp_path: Path) -> None:
        fp = self._write_json(tmp_path / "partial.json", {
            "buyma_platform_fee_rate": 0.08,
        })
        cfg = load_pricing_config(fp)
        assert cfg.buyma_platform_fee_rate == pytest.approx(0.08)
        assert cfg.buyma_effective_fee_rate == pytest.approx(
            _DEFAULT_CONFIG.buyma_effective_fee_rate
        )
        assert cfg.transfer_fee == pytest.approx(_DEFAULT_CONFIG.transfer_fee)

    def test_empty_json_object(self, tmp_path: Path) -> None:
        fp = self._write_json(tmp_path / "empty.json", {})
        cfg = load_pricing_config(fp)
        assert cfg == _DEFAULT_CONFIG


# =====================================================================
# load_pricing_config – 後方互換 (buyma_fee_rate → buyma_effective_fee_rate)
# =====================================================================

class TestLoadPricingConfigBackwardCompat:
    """旧キー buyma_fee_rate の後方互換性"""

    def _write_json(self, path: Path, obj: dict) -> str:
        path.write_text(json.dumps(obj), encoding="utf-8")
        return str(path)

    def test_old_key_only(self, tmp_path: Path) -> None:
        """buyma_fee_rate だけある場合 → buyma_effective_fee_rate にマップ"""
        fp = self._write_json(tmp_path / "old.json", {
            "buyma_fee_rate": 0.20,
        })
        cfg = load_pricing_config(fp)
        assert cfg.buyma_effective_fee_rate == pytest.approx(0.20)

    def test_new_key_takes_precedence(self, tmp_path: Path) -> None:
        """新旧両方ある場合 → 新キーが優先"""
        fp = self._write_json(tmp_path / "both.json", {
            "buyma_fee_rate": 0.20,
            "buyma_effective_fee_rate": 0.15,
        })
        cfg = load_pricing_config(fp)
        assert cfg.buyma_effective_fee_rate == pytest.approx(0.15)

    def test_neither_key_uses_default(self, tmp_path: Path) -> None:
        """両方無い場合 → デフォルト値"""
        fp = self._write_json(tmp_path / "neither.json", {
            "transfer_fee": 500.0,
        })
        cfg = load_pricing_config(fp)
        assert cfg.buyma_effective_fee_rate == pytest.approx(
            _DEFAULT_CONFIG.buyma_effective_fee_rate
        )


# =====================================================================
# load_pricing_config – 例外フォールバック
# =====================================================================

class TestLoadPricingConfigExceptionFallback:
    """壊れた設定ファイルや JSON エラーでデフォルトにフォールバック"""

    def test_invalid_json_syntax(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text("{invalid json!!", encoding="utf-8")
        cfg = load_pricing_config(str(bad))
        assert cfg == _DEFAULT_CONFIG

    def test_non_numeric_value_causes_fallback(self, tmp_path: Path) -> None:
        """float() への変換で TypeError/ValueError → fallback"""
        bad = tmp_path / "non_numeric.json"
        bad.write_text(
            json.dumps({"buyma_platform_fee_rate": "not_a_number"}),
            encoding="utf-8",
        )
        cfg = load_pricing_config(str(bad))
        assert cfg == _DEFAULT_CONFIG

    def test_encoding_error_fallback(self, tmp_path: Path) -> None:
        """UTF-8 として読めないバイナリ → fallback"""
        bad = tmp_path / "binary.bin"
        bad.write_bytes(b"\xff\xfe\x80\x81\x82")
        cfg = load_pricing_config(str(bad))
        assert cfg == _DEFAULT_CONFIG
