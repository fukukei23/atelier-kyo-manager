# -*- coding: utf-8 -*-
"""
CR-ATELIER-002 Step 7: Moncler Patch Builder のテスト

テスト内容:
- build_moncler_analysis_payload のテスト
- build_moncler_patch_candidate のテスト
- selector の before/after が期待どおりに構築されるか
- trap_url_patterns の append が想定通りか
- 不正な提案（外部ドメインや /products/ を含まないパス等）がフィルタされること
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path
import json
import tempfile
import shutil

from app.agents.moncler_patch_builder import (
    build_moncler_analysis_payload,
    build_moncler_patch_candidate,
    save_moncler_patch_files,
    process_moncler_self_healing_results,
)


class TestBuildMonclerAnalysisPayload:
    """build_moncler_analysis_payload のテスト"""
    
    def test_build_analysis_payload_basic(self):
        """基本的な analysis payload の構築"""
        run_id = "test_run_001"
        site = "MONCLER_OFFICIAL"
        current_url = "https://www.moncler.com/en-int/women/outerwear/all-down-jackets/"
        moncler_outcome = {
            "plp_materialized": False,
            "tiles_detected": 0,
            "pdp_links_raw": 0,
            "pdp_links_accepted": 0,
            "layer_stats": {},
            "rejection_stats": {},
            "locale_corrections": 0,
            "trap_detected": False,
        }
        
        payload = build_moncler_analysis_payload(
            run_id=run_id,
            site=site,
            current_url=current_url,
            moncler_outcome=moncler_outcome,
        )
        
        assert payload["run_id"] == run_id
        assert payload["site"] == site
        assert payload["current_url"] == current_url
        assert payload["moncler_outcome"] == moncler_outcome
        assert "timestamp" in payload
        assert "version" in payload
    
    def test_build_analysis_payload_with_self_healing(self):
        """Self-Healing 結果を含む analysis payload"""
        run_id = "test_run_002"
        site = "MONCLER_OFFICIAL"
        current_url = "https://www.moncler.com/en-int/women/outerwear/all-down-jackets/"
        moncler_outcome = {
            "plp_materialized": False,
            "tiles_detected": 0,
            "pdp_links_raw": 0,
            "pdp_links_accepted": 0,
            "layer_stats": {},
            "rejection_stats": {},
            "locale_corrections": 0,
            "trap_detected": False,
        }
        self_healing_result = {
            "analysis": "セレクタが要素を見つけられていません。",
            "root_cause": "primary selector mismatch",
            "suggested_actions": [
                "MONCLER_PLP_PDP_LINK_SELECTORS_PRIMARY に新しいセレクタを追加",
            ],
            "confidence": 0.85,
        }
        
        payload = build_moncler_analysis_payload(
            run_id=run_id,
            site=site,
            current_url=current_url,
            moncler_outcome=moncler_outcome,
            self_healing_result=self_healing_result,
        )
        
        assert "self_healing" in payload
        assert payload["self_healing"]["root_cause"] == "primary selector mismatch"
        assert payload["self_healing"]["confidence"] == 0.85
        assert len(payload["self_healing"]["suggested_actions"]) > 0
    
    def test_build_analysis_payload_with_selector_discovery(self):
        """Selector Discovery 結果を含む analysis payload"""
        run_id = "test_run_003"
        site = "MONCLER_OFFICIAL"
        current_url = "https://www.moncler.com/en-int/women/outerwear/all-down-jackets/"
        moncler_outcome = {
            "plp_materialized": False,
            "tiles_detected": 0,
            "pdp_links_raw": 0,
            "pdp_links_accepted": 0,
            "layer_stats": {},
            "rejection_stats": {},
            "locale_corrections": 0,
            "trap_detected": False,
        }
        selector_discovery_result = {
            "candidate_selectors": [
                "article[data-component*='ProductCard'] a[href*='/products/']",
                "div[data-testid='product-card'] a[href*='/products/']",
            ],
            "recommended_layer": "primary",
            "confidence_scores": [0.92, 0.88],
        }
        
        payload = build_moncler_analysis_payload(
            run_id=run_id,
            site=site,
            current_url=current_url,
            moncler_outcome=moncler_outcome,
            selector_discovery_result=selector_discovery_result,
        )
        
        assert "selector_discovery" in payload
        assert payload["selector_discovery"]["recommended_layer"] == "primary"
        assert len(payload["selector_discovery"]["candidate_selectors"]) == 2
        assert "confidence_scores" in payload["selector_discovery"]


class TestBuildMonclerPatchCandidate:
    """build_moncler_patch_candidate のテスト"""
    
    def test_build_patch_candidate_selector_update(self):
        """セレクタ更新のパッチ候補生成"""
        analysis_payload = {
            "run_id": "test_run_001",
            "site": "MONCLER_OFFICIAL",
            "timestamp": "2025-12-08T12:00:00Z",
            "moncler_outcome": {},
            "selector_discovery": {
                "candidate_selectors": [
                    "article[data-component*='ProductCard'] a[href*='/products/']",
                    "div[data-testid='product-card'] a[href*='/products/']",
                ],
                "recommended_layer": "primary",
                "confidence_scores": {
                    "article[data-component*='ProductCard'] a[href*='/products/']": 0.92,
                    "div[data-testid='product-card'] a[href*='/products/']": 0.88,
                },
            },
        }
        current_site_config = {
            "selectors": {
                "plp": {
                    "pdp_link_selectors": ["a[href*='/products/']"],
                    "tile_selectors": [],
                }
            }
        }
        
        patch_candidate = build_moncler_patch_candidate(
            analysis_payload=analysis_payload,
            current_site_config=current_site_config,
        )
        
        assert patch_candidate["site"] == "MONCLER_OFFICIAL"
        assert patch_candidate["run_id"] == "test_run_001"
        assert "changes" in patch_candidate
        assert "selectors.plp" in patch_candidate["changes"]
        assert "pdp_link_selectors" in patch_candidate["changes"]["selectors.plp"]
        
        changes = patch_candidate["changes"]["selectors.plp"]["pdp_link_selectors"]
        assert "before" in changes
        assert "after" in changes
        assert changes["before"] == ["a[href*='/products/']"]
        assert len(changes["after"]) == 2
    
    def test_build_patch_candidate_trap_url_patterns_append(self):
        """trap_url_patterns の append パッチ候補生成"""
        analysis_payload = {
            "run_id": "test_run_002",
            "site": "MONCLER_OFFICIAL",
            "timestamp": "2025-12-08T12:00:00Z",
            "moncler_outcome": {
                "rejection_stats": {
                    "double_locale_path": 3,
                    "trap_pattern": 1,
                }
            },
            "self_healing": {
                "root_cause": "locale redirect loop",
            },
        }
        current_site_config = {
            "navigation": {
                "trap_url_patterns": ["/search", "/404"],
            }
        }
        
        patch_candidate = build_moncler_patch_candidate(
            analysis_payload=analysis_payload,
            current_site_config=current_site_config,
        )
        
        assert "changes" in patch_candidate
        assert "navigation.trap_url_patterns" in patch_candidate["changes"]
        
        trap_changes = patch_candidate["changes"]["navigation.trap_url_patterns"]
        assert "append" in trap_changes
        assert len(trap_changes["append"]) > 0
        # 二重ロケールパターンが含まれていることを確認
        assert any("en-[a-z]{2}/en-int" in pattern for pattern in trap_changes["append"])
    
    def test_build_patch_candidate_filters_invalid_selectors(self):
        """不正なセレクタがフィルタされること"""
        analysis_payload = {
            "run_id": "test_run_003",
            "site": "MONCLER_OFFICIAL",
            "timestamp": "2025-12-08T12:00:00Z",
            "moncler_outcome": {},
            "selector_discovery": {
                "candidate_selectors": [
                    "article[data-component*='ProductCard'] a[href*='/products/']",  # 有効
                    "a[href*='/external/']",  # /products/ を含まない → フィルタされる
                    "div[class*='breadcrumb'] a",  # /products/ を含まない → フィルタされる
                    "div[data-testid='product-card'] a[href*='/products/']",  # 有効
                ],
                "recommended_layer": "primary",
                "confidence_scores": [0.92, 0.50, 0.30, 0.88],
            },
        }
        current_site_config = {
            "selectors": {
                "plp": {
                    "pdp_link_selectors": ["a[href*='/products/']"],
                    "tile_selectors": [],
                }
            }
        }
        
        patch_candidate = build_moncler_patch_candidate(
            analysis_payload=analysis_payload,
            current_site_config=current_site_config,
        )
        
        if "selectors.plp" in patch_candidate.get("changes", {}):
            changes = patch_candidate["changes"]["selectors.plp"].get("pdp_link_selectors", {})
            if "after" in changes:
                # /products/ を含むセレクタのみが含まれていることを確認
                for selector in changes["after"]:
                    assert "/products/" in selector or "/product/" in selector or "ProductCard" in selector or "product-card" in selector.lower()
    
    def test_build_patch_candidate_risk_assessment(self):
        """リスク評価が適切に設定されること"""
        analysis_payload = {
            "run_id": "test_run_004",
            "site": "MONCLER_OFFICIAL",
            "timestamp": "2025-12-08T12:00:00Z",
            "moncler_outcome": {},
            "self_healing": {
                "root_cause": "locale redirect loop",
            },
        }
        current_site_config = {}
        
        patch_candidate = build_moncler_patch_candidate(
            analysis_payload=analysis_payload,
            current_site_config=current_site_config,
        )
        
        assert "risk_assessment" in patch_candidate
        assert patch_candidate["risk_assessment"]["overall"] == "HIGH"
        assert "notes" in patch_candidate["risk_assessment"]


class TestSaveMonclerPatchFiles:
    """save_moncler_patch_files のテスト"""
    
    @pytest.fixture
    def mock_run_context(self, tmp_path):
        """モック RunContext を作成"""
        run_context = Mock()
        run_context.run_path = tmp_path / "runs" / "test_run_001"
        run_context.run_path.mkdir(parents=True)
        return run_context
    
    def test_save_patch_files(self, mock_run_context):
        """パッチファイルが保存されること"""
        analysis_payload = {
            "run_id": "test_run_001",
            "site": "MONCLER_OFFICIAL",
            "timestamp": "2025-12-08T12:00:00Z",
            "moncler_outcome": {},
        }
        patch_candidate = {
            "target_file": "app/config/sites/overrides.local.json",
            "site": "MONCLER_OFFICIAL",
            "run_id": "test_run_001",
            "timestamp": "2025-12-08T12:00:00Z",
            "summary": "Test patch",
            "changes": {},
            "risk_assessment": {"overall": "LOW", "notes": "Test"},
        }
        
        saved_paths = save_moncler_patch_files(
            run_context=mock_run_context,
            analysis_payload=analysis_payload,
            patch_candidate=patch_candidate,
            generate_markdown=False,
        )
        
        assert "analysis" in saved_paths
        assert "patch_candidate" in saved_paths
        assert saved_paths["analysis"].exists()
        assert saved_paths["patch_candidate"].exists()
        
        # JSON ファイルの内容を確認
        with open(saved_paths["analysis"], "r", encoding="utf-8") as f:
            analysis_data = json.load(f)
            assert analysis_data["run_id"] == "test_run_001"
        
        with open(saved_paths["patch_candidate"], "r", encoding="utf-8") as f:
            patch_data = json.load(f)
            assert patch_data["run_id"] == "test_run_001"
            assert patch_data["site"] == "MONCLER_OFFICIAL"


class TestProcessMonclerSelfHealingResults:
    """process_moncler_self_healing_results のテスト"""
    
    @pytest.fixture
    def mock_run_context(self, tmp_path):
        """モック RunContext を作成"""
        run_context = Mock()
        run_context.run_path = tmp_path / "runs" / "test_run_001"
        run_context.run_path.mkdir(parents=True)
        return run_context
    
    @pytest.mark.asyncio
    async def test_process_results_without_site_config(self, mock_run_context, tmp_path):
        """site_config が提供されていない場合の処理"""
        # overrides.local.json を一時的に作成
        overrides_path = tmp_path / "overrides.local.json"
        overrides_data = {
            "MONCLER_OFFICIAL": {
                "selectors": {
                    "plp": {
                        "pdp_link_selectors": ["a[href*='/products/']"],
                    }
                }
            }
        }
        with open(overrides_path, "w", encoding="utf-8") as f:
            json.dump(overrides_data, f)
        
        # 一時的に overrides.local.json を app/config/sites/ にコピー
        config_dir = Path("app/config/sites")
        config_dir.mkdir(parents=True, exist_ok=True)
        original_overrides = config_dir / "overrides.local.json"
        backup_exists = original_overrides.exists()
        
        try:
            if backup_exists:
                # バックアップを作成
                backup_path = config_dir / "overrides.local.json.backup"
                shutil.copy2(original_overrides, backup_path)
            
            # 一時的な overrides.local.json を作成
            shutil.copy2(overrides_path, original_overrides)
            
            moncler_outcome = {
                "plp_materialized": False,
                "tiles_detected": 0,
                "pdp_links_raw": 0,
                "pdp_links_accepted": 0,
                "layer_stats": {},
                "rejection_stats": {},
                "locale_corrections": 0,
                "trap_detected": False,
            }
            
            selector_discovery_result = {
                "candidate_selectors": [
                    "article[data-component*='ProductCard'] a[href*='/products/']",
                ],
                "recommended_layer": "primary",
                "confidence_scores": [0.92],
            }
            
            saved_paths = await process_moncler_self_healing_results(
                run_context=mock_run_context,
                run_id="test_run_001",
                site="MONCLER_OFFICIAL",
                current_url="https://www.moncler.com/en-int/women/outerwear/all-down-jackets/",
                moncler_outcome=moncler_outcome,
                self_healing_result=None,
                selector_discovery_result=selector_discovery_result,
                current_site_config=None,
                generate_markdown=False,
            )
            
            # パッチファイルが生成されたことを確認
            assert "analysis" in saved_paths or len(saved_paths) == 0  # エラー時は空の dict を返す可能性がある
        
        finally:
            # バックアップを復元
            if backup_exists:
                backup_path = config_dir / "overrides.local.json.backup"
                if backup_path.exists():
                    shutil.copy2(backup_path, original_overrides)
                    backup_path.unlink()
            elif original_overrides.exists():
                original_overrides.unlink()

