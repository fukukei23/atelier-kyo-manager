# -*- coding: utf-8 -*-
"""
CR-ATELIER-003 Phase D-8: SelfHealingSandbox のテスト

メモリ上でパッチを適用する機能をテストする。
"""

import pytest
from app.agents.self_healing_sandbox import SelfHealingSandbox


@pytest.fixture
def sandbox():
    """SelfHealingSandbox インスタンス"""
    return SelfHealingSandbox()


@pytest.fixture
def sample_overrides():
    """テスト用 overrides.local.json の内容"""
    return {
        "MONCLER_OFFICIAL": {
            "home_url": "https://www.moncler.com/en-int",
            "discovery_settings": {
                "timeout_sec": 60,
            },
            "selectors": {
                "pdp": {
                    "title": ["h1.product-title"],
                    "price": [".product-price"],
                },
            },
        },
    }


@pytest.fixture
def sample_patch_candidate():
    """テスト用 patch_candidate_self_healing.json の内容"""
    return {
        "target_site": "MONCLER_OFFICIAL",
        "run_id": "test_run_123",
        "generated_at": "2025-12-10T00:00:00Z",
        "strategy": "heuristic_v1",
        "changes": [
            {
                "path": "discovery_settings.timeout_sec",
                "action": "increase",
                "value_delta": 30,
                "current_value": 60,
                "proposed_value": 90,
            }
        ],
        "notes": {
            "summary": "Test patch",
            "root_causes": [],
            "suggested_fixes": [],
        },
    }


def test_apply_patch_in_memory_success(sandbox, sample_overrides, sample_patch_candidate):
    """正常系: パッチ候補が正常に適用されることを確認"""
    # 元の overrides をコピー（変更されないことを確認するため）
    original_overrides = sample_overrides.copy()
    
    # パッチを適用
    virtual_overrides = sandbox.apply_patch_in_memory(
        overrides_json=sample_overrides,
        patch_candidate=sample_patch_candidate,
    )
    
    # 元の overrides が変更されていないことを確認
    assert sample_overrides["MONCLER_OFFICIAL"]["discovery_settings"]["timeout_sec"] == 60
    assert original_overrides["MONCLER_OFFICIAL"]["discovery_settings"]["timeout_sec"] == 60
    
    # 仮想 overrides が更新されたことを確認
    assert virtual_overrides["MONCLER_OFFICIAL"]["discovery_settings"]["timeout_sec"] == 90


def test_apply_patch_in_memory_empty_changes(sandbox, sample_overrides):
    """changes が空の場合、変更されないことを確認"""
    patch_candidate_empty = {
        "target_site": "MONCLER_OFFICIAL",
        "run_id": "test_run_123",
        "changes": [],
    }
    
    virtual_overrides = sandbox.apply_patch_in_memory(
        overrides_json=sample_overrides,
        patch_candidate=patch_candidate_empty,
    )
    
    # 変更されていないことを確認
    assert virtual_overrides["MONCLER_OFFICIAL"]["discovery_settings"]["timeout_sec"] == 60


def test_apply_patch_in_memory_site_not_found(sandbox, sample_overrides, sample_patch_candidate):
    """対象サイトが存在しない場合、変更されないことを確認"""
    patch_candidate_unknown = {
        "target_site": "UNKNOWN_SITE",
        "run_id": "test_run_123",
        "changes": [
            {
                "path": "discovery_settings.timeout_sec",
                "action": "increase",
                "value_delta": 30,
            }
        ],
    }
    
    virtual_overrides = sandbox.apply_patch_in_memory(
        overrides_json=sample_overrides,
        patch_candidate=patch_candidate_unknown,
    )
    
    # 変更されていないことを確認
    assert virtual_overrides["MONCLER_OFFICIAL"]["discovery_settings"]["timeout_sec"] == 60


def test_apply_patch_in_memory_multiple_changes(sandbox, sample_overrides):
    """複数の changes が含まれる場合、すべて適用されることを確認"""
    patch_candidate_multiple = {
        "target_site": "MONCLER_OFFICIAL",
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
    
    virtual_overrides = sandbox.apply_patch_in_memory(
        overrides_json=sample_overrides,
        patch_candidate=patch_candidate_multiple,
    )
    
    # timeout_sec が更新されたことを確認
    assert virtual_overrides["MONCLER_OFFICIAL"]["discovery_settings"]["timeout_sec"] == 90
    
    # selector review flag が追加されたことを確認
    assert "__meta_flags" in virtual_overrides["MONCLER_OFFICIAL"]["selectors"]["pdp"]
    assert virtual_overrides["MONCLER_OFFICIAL"]["selectors"]["pdp"]["__meta_flags"]["need_review"] is True


def test_get_site_config_from_overrides(sandbox, sample_overrides):
    """get_site_config_from_overrides が正しく動作することを確認"""
    site_config = sandbox.get_site_config_from_overrides(
        overrides_json=sample_overrides,
        site_code="MONCLER_OFFICIAL",
    )
    
    assert site_config is not None
    assert site_config["discovery_settings"]["timeout_sec"] == 60
    assert "selectors" in site_config


def test_get_site_config_from_overrides_not_found(sandbox, sample_overrides):
    """存在しないサイトコードの場合、None を返すことを確認"""
    site_config = sandbox.get_site_config_from_overrides(
        overrides_json=sample_overrides,
        site_code="UNKNOWN_SITE",
    )
    
    assert site_config is None


def test_apply_patch_in_memory_original_unchanged(sandbox, sample_overrides, sample_patch_candidate):
    """元の overrides_json が変更されないことを確認（ディープコピーの確認）"""
    original_timeout = sample_overrides["MONCLER_OFFICIAL"]["discovery_settings"]["timeout_sec"]
    
    virtual_overrides = sandbox.apply_patch_in_memory(
        overrides_json=sample_overrides,
        patch_candidate=sample_patch_candidate,
    )
    
    # 元の overrides が変更されていないことを確認
    assert sample_overrides["MONCLER_OFFICIAL"]["discovery_settings"]["timeout_sec"] == original_timeout
    
    # 仮想 overrides が変更されたことを確認
    assert virtual_overrides["MONCLER_OFFICIAL"]["discovery_settings"]["timeout_sec"] != original_timeout

