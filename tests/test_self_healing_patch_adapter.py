# -*- coding: utf-8 -*-
"""
CR-ATELIER-003 Phase D-7: SelfHealingPatchAdapter のテスト

patch_candidate を diff 形式に変換するロジックをテストする。
"""

import pytest
from app.agents.self_healing_patch_adapter import SelfHealingPatchAdapter


@pytest.fixture
def adapter():
    """SelfHealingPatchAdapter インスタンス"""
    return SelfHealingPatchAdapter()


@pytest.fixture
def site_config():
    """テスト用 site_config"""
    return {
        "discovery_settings": {
            "timeout_sec": 60,
        },
        "selectors": {
            "pdp": {
                "title": ["h1.product-title"],
                "price": [".product-price"],
            },
        },
    }


def test_to_diff_timeout_increase(adapter, site_config):
    """timeout_sec increase の場合、replace 操作が生成されることを確認"""
    patch_candidate = {
        "target_site": "TEST_SITE",
        "run_id": "test_run_123",
        "changes": [
            {
                "path": "discovery_settings.timeout_sec",
                "action": "increase",
                "value_delta": 30,
                "current_value": 60,
                "proposed_value": 90,
            }
        ],
    }
    
    diff_ops = adapter.to_diff(
        patch_candidate=patch_candidate,
        site_config=site_config,
    )
    
    assert len(diff_ops) == 1
    assert diff_ops[0]["op"] == "replace"
    assert diff_ops[0]["path"] == "/discovery_settings/timeout_sec"
    assert diff_ops[0]["value"] == 90


def test_to_diff_timeout_increase_without_current_value(adapter, site_config):
    """current_value が指定されていない場合、site_config から取得することを確認"""
    patch_candidate = {
        "target_site": "TEST_SITE",
        "run_id": "test_run_123",
        "changes": [
            {
                "path": "discovery_settings.timeout_sec",
                "action": "increase",
                "value_delta": 30,
            }
        ],
    }
    
    diff_ops = adapter.to_diff(
        patch_candidate=patch_candidate,
        site_config=site_config,
    )
    
    assert len(diff_ops) == 1
    assert diff_ops[0]["op"] == "replace"
    assert diff_ops[0]["path"] == "/discovery_settings/timeout_sec"
    # site_config の timeout_sec (60) + value_delta (30) = 90
    assert diff_ops[0]["value"] == 90


def test_to_diff_selector_review_required(adapter, site_config):
    """selector 要再確認フラグの場合、add 操作が生成されることを確認"""
    patch_candidate = {
        "target_site": "TEST_SITE",
        "run_id": "test_run_123",
        "changes": [
            {
                "path": "selectors.pdp",
                "action": "review_required",
                "reason": "Selector may be outdated or incorrect",
                "current_selectors": site_config["selectors"]["pdp"],
            }
        ],
    }
    
    diff_ops = adapter.to_diff(
        patch_candidate=patch_candidate,
        site_config=site_config,
    )
    
    assert len(diff_ops) == 1
    assert diff_ops[0]["op"] == "add"
    assert diff_ops[0]["path"] == "/selectors/pdp/__meta_flags/need_review"
    assert diff_ops[0]["value"] is True


def test_to_diff_set_action(adapter, site_config):
    """set action の場合、replace または add 操作が生成されることを確認"""
    patch_candidate = {
        "target_site": "TEST_SITE",
        "run_id": "test_run_123",
        "changes": [
            {
                "path": "discovery_settings.max_retries",
                "action": "set",
                "value": 5,
            }
        ],
    }
    
    diff_ops = adapter.to_diff(
        patch_candidate=patch_candidate,
        site_config=site_config,
    )
    
    assert len(diff_ops) == 1
    assert diff_ops[0]["op"] in ("add", "replace")
    assert diff_ops[0]["path"] == "/discovery_settings/max_retries"
    assert diff_ops[0]["value"] == 5


def test_to_diff_empty_changes(adapter, site_config):
    """changes が空の場合、空のリストを返すことを確認"""
    patch_candidate = {
        "target_site": "TEST_SITE",
        "run_id": "test_run_123",
        "changes": [],
    }
    
    diff_ops = adapter.to_diff(
        patch_candidate=patch_candidate,
        site_config=site_config,
    )
    
    assert diff_ops == []


def test_to_diff_multiple_changes(adapter, site_config):
    """複数の changes が含まれる場合、すべて変換されることを確認"""
    patch_candidate = {
        "target_site": "TEST_SITE",
        "run_id": "test_run_123",
        "changes": [
            {
                "path": "discovery_settings.timeout_sec",
                "action": "increase",
                "value_delta": 30,
                "current_value": 60,
            },
            {
                "path": "selectors.pdp",
                "action": "review_required",
                "reason": "Selector may be outdated",
            },
        ],
    }
    
    diff_ops = adapter.to_diff(
        patch_candidate=patch_candidate,
        site_config=site_config,
    )
    
    assert len(diff_ops) == 2
    
    # timeout_sec の replace
    timeout_op = next((op for op in diff_ops if "timeout_sec" in op["path"]), None)
    assert timeout_op is not None
    assert timeout_op["op"] == "replace"
    assert timeout_op["value"] == 90
    
    # selector review flag の add
    review_op = next((op for op in diff_ops if "need_review" in op["path"]), None)
    assert review_op is not None
    assert review_op["op"] == "add"
    assert review_op["value"] is True


def test_to_json_pointer(adapter):
    """dot 区切りのパスを JSON Pointer 形式に変換することを確認"""
    assert adapter._to_json_pointer("discovery_settings.timeout_sec") == "/discovery_settings/timeout_sec"
    assert adapter._to_json_pointer("selectors.pdp") == "/selectors/pdp"
    assert adapter._to_json_pointer("navigation.trap_url_patterns") == "/navigation/trap_url_patterns"


def test_get_nested_value(adapter, site_config):
    """ネストされた値を取得することを確認"""
    assert adapter._get_nested_value(site_config, "discovery_settings.timeout_sec") == 60
    assert adapter._get_nested_value(site_config, "selectors.pdp.title") == ["h1.product-title"]
    assert adapter._get_nested_value(site_config, "nonexistent.path", default=0) == 0


def test_path_exists(adapter, site_config):
    """パスが存在するかチェックすることを確認"""
    assert adapter._path_exists(site_config, "discovery_settings.timeout_sec") is True
    assert adapter._path_exists(site_config, "selectors.pdp") is True
    assert adapter._path_exists(site_config, "nonexistent.path") is False

