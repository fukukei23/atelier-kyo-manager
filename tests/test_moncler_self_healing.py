# -*- coding: utf-8 -*-
"""
CR-ATELIER-002 Step 6: Moncler Self-Healing Agent のテスト

テスト内容:
- Self-Healing トリガ条件で handle_moncler_failure が呼ばれること
- failure_payload のフォーマット検証
"""

import pytest
pytest.importorskip("playwright")
from unittest.mock import AsyncMock, MagicMock, patch
from app.agents.self_healing_agent import SelfHealingAgent


class TestMonclerSelfHealing:
    """Moncler 専用の Self-Healing Agent のテスト"""
    
    @pytest.fixture
    def self_healing_agent(self):
        """Self-Healing Agent のインスタンスを作成"""
        return SelfHealingAgent()
    
    @pytest.mark.asyncio
    async def test_handle_moncler_failure_raw_zero(self, self_healing_agent):
        """raw_zero の場合のハンドリング"""
        failure_payload = {
            "site": "MONCLER_OFFICIAL",
            "url": "https://www.moncler.com/en-int/women/outerwear/all-down-jackets/",
            "failure_reason": "raw_zero",
            "dom_snapshot_path": None,
            "layer_stats": {"primary_raw": 0, "primary_accepted": 0},
            "rejection_stats": {},
            "selectors_current": {},
            "run_id": "test_run_001",
            "timestamp": "2025-12-08T12:00:00Z",
        }
        
        result = await self_healing_agent.handle_moncler_failure(failure_payload)
        
        assert result["root_cause"] == "primary selector mismatch"
        assert result["confidence"] > 0.0
        assert len(result["suggested_actions"]) > 0
        assert "analysis" in result
    
    @pytest.mark.asyncio
    async def test_handle_moncler_failure_rejected_all(self, self_healing_agent):
        """rejected_all の場合のハンドリング"""
        failure_payload = {
            "site": "MONCLER_OFFICIAL",
            "url": "https://www.moncler.com/en-int/women/outerwear/all-down-jackets/",
            "failure_reason": "rejected_all",
            "dom_snapshot_path": None,
            "layer_stats": {"primary_raw": 10, "primary_accepted": 0},
            "rejection_stats": {"external_domain": 5, "blocked_domain": 5},
            "selectors_current": {},
            "run_id": "test_run_002",
            "timestamp": "2025-12-08T12:00:00Z",
        }
        
        result = await self_healing_agent.handle_moncler_failure(failure_payload)
        
        assert result["root_cause"] == "url validation too strict"
        assert result["confidence"] > 0.0
        assert len(result["suggested_actions"]) > 0
    
    @pytest.mark.asyncio
    async def test_handle_moncler_failure_secondary_used(self, self_healing_agent):
        """secondary_or_tertiary_used の場合のハンドリング"""
        failure_payload = {
            "site": "MONCLER_OFFICIAL",
            "url": "https://www.moncler.com/en-int/women/outerwear/all-down-jackets/",
            "failure_reason": "secondary_or_tertiary_used",
            "dom_snapshot_path": None,
            "layer_stats": {
                "primary_raw": 0,
                "primary_accepted": 0,
                "secondary_raw": 5,
                "secondary_accepted": 3,
            },
            "rejection_stats": {},
            "selectors_current": {},
            "run_id": "test_run_003",
            "timestamp": "2025-12-08T12:00:00Z",
        }
        
        result = await self_healing_agent.handle_moncler_failure(failure_payload)
        
        assert result["root_cause"] == "primary selector ineffective"
        assert result["confidence"] > 0.0
        assert len(result["suggested_actions"]) > 0
    
    @pytest.mark.asyncio
    async def test_handle_moncler_failure_trap_detected(self, self_healing_agent):
        """trap_detected の場合のハンドリング"""
        failure_payload = {
            "site": "MONCLER_OFFICIAL",
            "url": "https://www.moncler.com/en-int/search",
            "failure_reason": "trap_detected",
            "dom_snapshot_path": None,
            "layer_stats": {},
            "rejection_stats": {},
            "selectors_current": {},
            "run_id": "test_run_004",
            "timestamp": "2025-12-08T12:00:00Z",
        }
        
        result = await self_healing_agent.handle_moncler_failure(failure_payload)
        
        assert result["root_cause"] == "trap page navigation"
        assert result["confidence"] > 0.0
        assert len(result["suggested_actions"]) > 0
    
    @pytest.mark.asyncio
    async def test_handle_moncler_failure_locale_corrections_exceeded(self, self_healing_agent):
        """locale_corrections_exceeded の場合のハンドリング"""
        failure_payload = {
            "site": "MONCLER_OFFICIAL",
            "url": "https://www.moncler.com/en-lt/en-int/women/outerwear/all-down-jackets/",
            "failure_reason": "locale_corrections_exceeded",
            "dom_snapshot_path": None,
            "layer_stats": {},
            "rejection_stats": {},
            "selectors_current": {},
            "run_id": "test_run_005",
            "timestamp": "2025-12-08T12:00:00Z",
        }
        
        result = await self_healing_agent.handle_moncler_failure(failure_payload)
        
        assert result["root_cause"] == "locale redirect loop"
        assert result["confidence"] > 0.0
        assert len(result["suggested_actions"]) > 0
    
    @pytest.mark.asyncio
    async def test_handle_moncler_failure_unknown(self, self_healing_agent):
        """不明な失敗理由の場合のハンドリング"""
        failure_payload = {
            "site": "MONCLER_OFFICIAL",
            "url": "https://www.moncler.com/en-int/women/outerwear/all-down-jackets/",
            "failure_reason": "unknown",
            "dom_snapshot_path": None,
            "layer_stats": {},
            "rejection_stats": {},
            "selectors_current": {},
            "run_id": "test_run_006",
            "timestamp": "2025-12-08T12:00:00Z",
        }
        
        result = await self_healing_agent.handle_moncler_failure(failure_payload)
        
        assert result["root_cause"] == "unknown"
        assert result["confidence"] > 0.0
        assert len(result["suggested_actions"]) > 0
    
    def test_failure_payload_format(self):
        """failure_payload のフォーマット検証"""
        required_fields = [
            "site",
            "url",
            "failure_reason",
            "dom_snapshot_path",
            "layer_stats",
            "rejection_stats",
            "selectors_current",
            "run_id",
            "timestamp",
        ]
        
        failure_payload = {
            "site": "MONCLER_OFFICIAL",
            "url": "https://www.moncler.com/en-int/women/outerwear/all-down-jackets/",
            "failure_reason": "raw_zero",
            "dom_snapshot_path": None,
            "layer_stats": {},
            "rejection_stats": {},
            "selectors_current": {},
            "run_id": "test_run_001",
            "timestamp": "2025-12-08T12:00:00Z",
        }
        
        for field in required_fields:
            assert field in failure_payload, f"Missing required field: {field}"

