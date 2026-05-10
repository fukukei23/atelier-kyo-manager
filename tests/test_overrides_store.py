"""
overrides_store のテスト — セレクタ提案管理のビジネスロジック検証

tmp_path で実際のファイル操作を検証（モック不使用）。
"""

from __future__ import annotations

import json

import pytest

from app.utils.overrides_store import (
    _deep_merge_safe,
    _normalize_proposal,
    _sanitize_selector_list,
    _strip_empty_lists,
    _unique_extend,
    apply_selector_proposal,
    extract_suggested_selectors,
    update_site_selectors,
)


# ==============================
# 内部ユーティリティのテスト
# ==============================


class TestSanitizeSelectorList:
    def test_filters_invalid_types(self):
        assert _sanitize_selector_list(["a", 1, None, "b", 3.14]) == ["a", "b"]

    def test_keeps_dicts(self):
        data = [{"css": ".product"}, "div.item", 42]
        assert _sanitize_selector_list(data) == [{"css": ".product"}, "div.item"]

    def test_empty_input(self):
        assert _sanitize_selector_list([]) == []


class TestUniqueExtend:
    def test_new_items_prepended(self):
        base = ["a", "b"]
        new = ["c", "d"]
        result = _unique_extend(base, new)
        # new が先、base が後。重複なし
        assert result == ["c", "d", "a", "b"]

    def test_dedup_strings(self):
        base = ["a", "b"]
        new = ["b", "c"]
        result = _unique_extend(base, new)
        # "b" は new 側が優先され base の "b" は除去
        assert result == ["b", "c", "a"]

    def test_dedup_dicts(self):
        base = [{"css": ".a"}]
        new = [{"css": ".a"}, {"css": ".b"}]
        result = _unique_extend(base, new)
        assert len(result) == 2
        assert {"css": ".b"} in result

    def test_mixed_types(self):
        base = ["a", {"x": 1}]
        new = ["b", {"x": 1}]
        result = _unique_extend(base, new)
        assert result == ["b", {"x": 1}, "a"]


class TestStripEmptyLists:
    def test_removes_empty_lists(self):
        data = {"a": [], "b": ["item"]}
        assert _strip_empty_lists(data) == {"b": ["item"]}

    def test_removes_empty_nested_dicts(self):
        data = {"a": {"b": {}}, "c": {"d": ["x"]}}
        assert _strip_empty_lists(data) == {"c": {"d": ["x"]}}

    def test_keeps_non_empty_values(self):
        data = {"key": "value", "num": 42}
        assert _strip_empty_lists(data) == data

    def test_filters_invalid_items_in_lists(self):
        data = {"a": ["valid", 123, None], "b": [456]}
        assert _strip_empty_lists(data) == {"a": ["valid"]}


class TestDeepMergeSafe:
    def test_merges_nested_dicts(self):
        a = {"site1": {"selectors": {"x": ["1"]}}}
        b = {"site1": {"selectors": {"y": ["2"]}}}
        result = _deep_merge_safe(a, b)
        assert result["site1"]["selectors"]["x"] == ["1"]
        assert result["site1"]["selectors"]["y"] == ["2"]

    def test_deduplicates_lists(self):
        a = {"items": ["a", "b"]}
        b = {"items": ["b", "c"]}
        result = _deep_merge_safe(a, b)
        assert "c" in result["items"]
        assert "a" in result["items"]
        assert result["items"].count("b") == 1

    def test_overwrites_scalar(self):
        a = {"key": "old"}
        b = {"key": "new"}
        result = _deep_merge_safe(a, b)
        assert result["key"] == "new"

    def test_does_not_mutate_inputs(self):
        a = {"x": ["a"]}
        b = {"x": ["b"]}
        _deep_merge_safe(a, b)
        assert a == {"x": ["a"]}
        assert b == {"x": ["b"]}


class TestNormalizeProposal:
    def test_with_site_key(self):
        proposal = {"gucci": {"selectors": {"a": ["1"]}}}
        result = _normalize_proposal(proposal, site="gucci")
        assert "gucci" in result

    def test_single_key_auto_infer(self):
        proposal = {"gucci": {"a": ["1"]}}
        result = _normalize_proposal(proposal)
        assert "gucci" in result

    def test_multi_key_without_site_raises(self):
        proposal = {"gucci": {}, "prada": {}}
        with pytest.raises(ValueError, match="site"):
            _normalize_proposal(proposal)

    def test_empty_proposal(self):
        assert _normalize_proposal({}) == {}
        assert _normalize_proposal(None) == {}

    def test_suggested_selectors_key(self):
        proposal = {"gucci": {"suggested_selectors": {"a": ["1"]}}}
        result = _normalize_proposal(proposal, site="gucci")
        assert "selectors" in result["gucci"]


# ==============================
# 公開API のテスト
# ==============================


class TestExtractSuggestedSelectors:
    def test_extracts_from_nested_path(self):
        evidence = {"suggested_selectors": {"pdp": {"pdp_link_selectors": [".product-link", "a.item"]}}}
        result = extract_suggested_selectors(evidence, "pdp.pdp_link_selectors")
        assert result == [".product-link", "a.item"]

    def test_returns_defaults_on_missing_path(self):
        evidence = {"suggested_selectors": {"pdp": {}}}
        result = extract_suggested_selectors(evidence, "pdp.nonexistent", defaults=["fallback"])
        assert result == ["fallback"]

    def test_none_evidence(self):
        assert extract_suggested_selectors(None, defaults=["x"]) == ["x"]

    def test_empty_evidence(self):
        assert extract_suggested_selectors({}, defaults=["x"]) == ["x"]

    def test_no_suggested_selectors_key(self):
        assert extract_suggested_selectors({"other": "data"}, defaults=["y"]) == ["y"]

    def test_non_dict_at_path(self):
        evidence = {"suggested_selectors": {"pdp": "not_a_dict"}}
        result = extract_suggested_selectors(evidence, "pdp.sub", defaults=["z"])
        assert result == ["z"]

    def test_filters_invalid_items(self):
        evidence = {"suggested_selectors": {"pdp": {"links": [".valid", 123, None, ".also-valid"]}}}
        result = extract_suggested_selectors(evidence, "pdp.links")
        assert result == [".valid", ".also-valid"]


class TestApplySelectorProposal:
    def test_merges_into_existing_file(self, tmp_path):
        overrides = tmp_path / "overrides.json"
        overrides.write_text(json.dumps({"gucci": {"selectors": {"old": ["a"]}}}), encoding="utf-8")

        proposal = tmp_path / "proposal.json"
        proposal.write_text(json.dumps({"gucci": {"selectors": {"new": ["b"]}}}), encoding="utf-8")

        success, diff = apply_selector_proposal(proposal_path=proposal, overrides_path=overrides, site="gucci")

        assert success is True
        assert diff is not None
        result = json.loads(overrides.read_text(encoding="utf-8"))
        assert "old" in result["gucci"]["selectors"]
        assert "new" in result["gucci"]["selectors"]

    def test_creates_new_file_when_absent(self, tmp_path):
        overrides = tmp_path / "new_overrides.json"
        proposal = tmp_path / "proposal.json"
        proposal.write_text(json.dumps({"prada": {"selectors": {"x": ["1"]}}}), encoding="utf-8")

        success, diff = apply_selector_proposal(proposal_path=proposal, overrides_path=overrides, site="prada")

        assert success is True
        result = json.loads(overrides.read_text(encoding="utf-8"))
        assert result["prada"]["selectors"]["x"] == ["1"]

    def test_no_change_returns_false(self, tmp_path):
        """内容が完全同一（json.dumps出力も同一）の場合はFalse"""
        overrides = tmp_path / "overrides.json"
        data = {"gucci": {"selectors": {"x": ["1"]}}}
        # json.dumps を通して同じキー順序・インデントで書き込む
        content = json.dumps(data, indent=2, ensure_ascii=False)
        overrides.write_text(content, encoding="utf-8")

        proposal = tmp_path / "proposal.json"
        proposal.write_text(json.dumps({"gucci": {"selectors": {"x": ["1"]}}}), encoding="utf-8")

        success, diff = apply_selector_proposal(proposal_path=proposal, overrides_path=overrides, site="gucci")

        assert success is False

    def test_missing_proposal_returns_false(self, tmp_path):
        overrides = tmp_path / "overrides.json"
        overrides.write_text("{}", encoding="utf-8")
        proposal = tmp_path / "nonexistent.json"

        success, diff = apply_selector_proposal(proposal_path=proposal, overrides_path=overrides, site="gucci")

        assert success is False

    def test_corrupted_overrides_recovers(self, tmp_path):
        overrides = tmp_path / "overrides.json"
        overrides.write_text("{invalid json}", encoding="utf-8")

        proposal = tmp_path / "proposal.json"
        proposal.write_text(json.dumps({"gucci": {"selectors": {"x": ["1"]}}}), encoding="utf-8")

        success, diff = apply_selector_proposal(proposal_path=proposal, overrides_path=overrides, site="gucci")

        assert success is True
        result = json.loads(overrides.read_text(encoding="utf-8"))
        assert result["gucci"]["selectors"]["x"] == ["1"]

    def test_empty_selectors_skipped(self, tmp_path):
        overrides = tmp_path / "overrides.json"
        overrides.write_text("{}", encoding="utf-8")

        proposal = tmp_path / "proposal.json"
        proposal.write_text(json.dumps({"gucci": {"selectors": {}}}), encoding="utf-8")

        success, diff = apply_selector_proposal(proposal_path=proposal, overrides_path=overrides, site="gucci")

        assert success is False

    def test_creates_parent_directory(self, tmp_path):
        overrides = tmp_path / "sub" / "dir" / "overrides.json"
        proposal = tmp_path / "proposal.json"
        proposal.write_text(json.dumps({"gucci": {"selectors": {"x": ["1"]}}}), encoding="utf-8")

        success, diff = apply_selector_proposal(proposal_path=proposal, overrides_path=overrides, site="gucci")

        assert success is True
        assert overrides.exists()


class TestUpdateSiteSelectors:
    def test_updates_existing_site(self, tmp_path):
        overrides = tmp_path / "overrides.json"
        overrides.write_text(json.dumps({"gucci": {"selectors": {"old": ["a"]}}}), encoding="utf-8")

        success, diff = update_site_selectors(
            site="gucci",
            new_selectors={"new": ["b"]},
            overrides_path=overrides,
        )

        assert success is True
        result = json.loads(overrides.read_text(encoding="utf-8"))
        assert "old" in result["gucci"]["selectors"]
        assert "new" in result["gucci"]["selectors"]

    def test_adds_new_site(self, tmp_path):
        overrides = tmp_path / "overrides.json"
        overrides.write_text("{}", encoding="utf-8")

        success, diff = update_site_selectors(
            site="prada",
            new_selectors={"x": ["1"]},
            overrides_path=overrides,
        )

        assert success is True
        result = json.loads(overrides.read_text(encoding="utf-8"))
        assert result["prada"]["selectors"]["x"] == ["1"]

    def test_empty_selectors_returns_false(self, tmp_path):
        overrides = tmp_path / "overrides.json"
        overrides.write_text("{}", encoding="utf-8")

        success, diff = update_site_selectors(
            site="gucci",
            new_selectors={},
            overrides_path=overrides,
        )

        assert success is False

    def test_no_change_returns_false(self, tmp_path):
        """内容が完全同一（json.dumps 出力も同一）の場合はFalse"""
        data = {"gucci": {"selectors": {"x": ["1"]}}}
        overrides = tmp_path / "overrides.json"
        # json.dumps を通して同じキー順序・インデントで書き込む
        content = json.dumps(data, indent=2, ensure_ascii=False)
        overrides.write_text(content, encoding="utf-8")

        success, diff = update_site_selectors(
            site="gucci",
            new_selectors={"x": ["1"]},
            overrides_path=overrides,
        )

        assert success is False

    def test_corrupted_file_recovers(self, tmp_path):
        overrides = tmp_path / "overrides.json"
        overrides.write_text("not valid json {{{", encoding="utf-8")

        success, diff = update_site_selectors(
            site="gucci",
            new_selectors={"x": ["1"]},
            overrides_path=overrides,
        )

        assert success is True
        result = json.loads(overrides.read_text(encoding="utf-8"))
        assert result["gucci"]["selectors"]["x"] == ["1"]

    def test_creates_parent_directory(self, tmp_path):
        overrides = tmp_path / "deep" / "nested" / "overrides.json"

        success, diff = update_site_selectors(
            site="gucci",
            new_selectors={"x": ["1"]},
            overrides_path=overrides,
        )

        assert success is True
        assert overrides.exists()

    def test_atomic_write_no_partial_file(self, tmp_path):
        """tmpファイルが最終ファイルに置換され、.tmpは残らない"""
        overrides = tmp_path / "overrides.json"
        overrides.write_text("{}", encoding="utf-8")

        update_site_selectors(
            site="gucci",
            new_selectors={"x": ["1"]},
            overrides_path=overrides,
        )

        assert overrides.exists()
        # .tmp ファイルが残っていないことを確認
        tmp_files = list(tmp_path.glob("*.tmp"))
        assert len(tmp_files) == 0
