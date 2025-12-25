# -*- coding: utf-8 -*-
"""
CR-ATELIER-003 Phase D-7: SelfHealingPatchApplier のテスト

パッチ候補を overrides.local.json に適用するロジックをテストする。
"""

import json
import pytest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock

from app.agents.self_healing_patch_applier import SelfHealingPatchApplier
from app.agents.self_healing_patch_adapter import SelfHealingPatchAdapter


@pytest.fixture
def applier():
    """SelfHealingPatchApplier インスタンス"""
    return SelfHealingPatchApplier()


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


def test_apply_patch_candidate_success(applier, sample_overrides, sample_patch_candidate):
    """正常系: パッチ候補が正常に適用されることを確認"""
    with TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        # テスト用ファイルを作成
        overrides_path = tmp_path / "overrides.local.json"
        candidate_path = tmp_path / "patch_candidate_self_healing.json"
        
        with open(overrides_path, "w", encoding="utf-8") as f:
            json.dump(sample_overrides, f, indent=2)
        
        with open(candidate_path, "w", encoding="utf-8") as f:
            json.dump(sample_patch_candidate, f, indent=2)
        
        # パッチを適用
        result = applier.apply_patch_candidate(
            candidate_path=candidate_path,
            overrides_path=overrides_path,
        )
        
        # 適用が成功したことを確認
        assert result["applied"] is True
        assert result["target_site"] == "MONCLER_OFFICIAL"
        assert len(result["diff_ops"]) > 0
        assert result["backup_path"] is not None
        assert result["backup_path"].exists()
        assert result["error"] is None
        
        # overrides.local.json が更新されたことを確認
        with open(overrides_path, "r", encoding="utf-8") as f:
            updated_overrides = json.load(f)
        
        assert updated_overrides["MONCLER_OFFICIAL"]["discovery_settings"]["timeout_sec"] == 90
        
        # バックアップファイルが作成されたことを確認
        assert result["backup_path"].exists()
        
        # バックアップの内容を確認
        with open(result["backup_path"], "r", encoding="utf-8") as f:
            backup_overrides = json.load(f)
        
        assert backup_overrides["MONCLER_OFFICIAL"]["discovery_settings"]["timeout_sec"] == 60


def test_apply_patch_candidate_empty_changes(applier, sample_overrides):
    """changes が空の場合、適用されないことを確認"""
    with TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        # テスト用ファイルを作成
        overrides_path = tmp_path / "overrides.local.json"
        candidate_path = tmp_path / "patch_candidate_self_healing.json"
        
        with open(overrides_path, "w", encoding="utf-8") as f:
            json.dump(sample_overrides, f, indent=2)
        
        patch_candidate_empty = {
            "target_site": "MONCLER_OFFICIAL",
            "run_id": "test_run_123",
            "changes": [],
        }
        
        with open(candidate_path, "w", encoding="utf-8") as f:
            json.dump(patch_candidate_empty, f, indent=2)
        
        # パッチを適用
        result = applier.apply_patch_candidate(
            candidate_path=candidate_path,
            overrides_path=overrides_path,
        )
        
        # 適用されなかったことを確認
        assert result["applied"] is False
        assert result["error"] == "No changes to apply"
        assert result["backup_path"] is None
        
        # overrides.local.json が変更されていないことを確認
        with open(overrides_path, "r", encoding="utf-8") as f:
            updated_overrides = json.load(f)
        
        assert updated_overrides["MONCLER_OFFICIAL"]["discovery_settings"]["timeout_sec"] == 60


def test_apply_patch_candidate_candidate_not_found(applier, sample_overrides):
    """パッチ候補ファイルが存在しない場合、エラーを返すことを確認"""
    with TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        overrides_path = tmp_path / "overrides.local.json"
        candidate_path = tmp_path / "nonexistent.json"
        
        with open(overrides_path, "w", encoding="utf-8") as f:
            json.dump(sample_overrides, f, indent=2)
        
        result = applier.apply_patch_candidate(
            candidate_path=candidate_path,
            overrides_path=overrides_path,
        )
        
        assert result["applied"] is False
        assert "not found" in result["error"].lower()


def test_apply_patch_candidate_overrides_not_found(applier, sample_patch_candidate):
    """overrides.local.json が存在しない場合、エラーを返すことを確認"""
    with TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        overrides_path = tmp_path / "nonexistent.json"
        candidate_path = tmp_path / "patch_candidate_self_healing.json"
        
        with open(candidate_path, "w", encoding="utf-8") as f:
            json.dump(sample_patch_candidate, f, indent=2)
        
        result = applier.apply_patch_candidate(
            candidate_path=candidate_path,
            overrides_path=overrides_path,
        )
        
        assert result["applied"] is False
        assert "not found" in result["error"].lower()


def test_apply_patch_candidate_site_not_found(applier, sample_overrides, sample_patch_candidate):
    """対象サイトが overrides に存在しない場合、エラーを返すことを確認"""
    with TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        overrides_path = tmp_path / "overrides.local.json"
        candidate_path = tmp_path / "patch_candidate_self_healing.json"
        
        # 対象サイトを含まない overrides を作成
        overrides_without_site = {}
        
        with open(overrides_path, "w", encoding="utf-8") as f:
            json.dump(overrides_without_site, f, indent=2)
        
        with open(candidate_path, "w", encoding="utf-8") as f:
            json.dump(sample_patch_candidate, f, indent=2)
        
        result = applier.apply_patch_candidate(
            candidate_path=candidate_path,
            overrides_path=overrides_path,
        )
        
        assert result["applied"] is False
        assert "not found" in result["error"].lower() or "not found" in result["error"]


def test_apply_patch_candidate_invalid_json(applier, sample_overrides):
    """無効な JSON ファイルの場合、エラーを返すことを確認"""
    with TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        overrides_path = tmp_path / "overrides.local.json"
        candidate_path = tmp_path / "patch_candidate_self_healing.json"
        
        with open(overrides_path, "w", encoding="utf-8") as f:
            json.dump(sample_overrides, f, indent=2)
        
        # 無効な JSON を書き込む
        with open(candidate_path, "w", encoding="utf-8") as f:
            f.write("{ invalid json }")
        
        result = applier.apply_patch_candidate(
            candidate_path=candidate_path,
            overrides_path=overrides_path,
        )
        
        assert result["applied"] is False
        assert "json" in result["error"].lower() or "invalid" in result["error"].lower()


def test_apply_patch_candidate_multiple_changes(applier, sample_overrides):
    """複数の changes が含まれる場合、すべて適用されることを確認"""
    with TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        overrides_path = tmp_path / "overrides.local.json"
        candidate_path = tmp_path / "patch_candidate_self_healing.json"
        
        with open(overrides_path, "w", encoding="utf-8") as f:
            json.dump(sample_overrides, f, indent=2)
        
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
        
        with open(candidate_path, "w", encoding="utf-8") as f:
            json.dump(patch_candidate_multiple, f, indent=2)
        
        result = applier.apply_patch_candidate(
            candidate_path=candidate_path,
            overrides_path=overrides_path,
        )
        
        assert result["applied"] is True
        assert len(result["diff_ops"]) == 2
        
        # overrides.local.json が更新されたことを確認
        with open(overrides_path, "r", encoding="utf-8") as f:
            updated_overrides = json.load(f)
        
        assert updated_overrides["MONCLER_OFFICIAL"]["discovery_settings"]["timeout_sec"] == 90
        # selector review flag が追加されたことを確認
        assert "__meta_flags" in updated_overrides["MONCLER_OFFICIAL"]["selectors"]["pdp"]
        assert updated_overrides["MONCLER_OFFICIAL"]["selectors"]["pdp"]["__meta_flags"]["need_review"] is True

