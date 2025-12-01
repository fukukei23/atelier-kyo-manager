# Stage 4 追加タスク: BrowserUseAgent 薄い統合テスト - 差分形式

## tests/test_browser_use_agent_plp_integration.py の変更

### 追加テスト: `test_run_plp_flow_saves_plp_navigation_result`

```diff
+@pytest.mark.asyncio
+async def test_run_plp_flow_saves_plp_navigation_result(
+    browser_use_agent, mock_page, mock_context, site_config, run_context
+):
+    """
+    Stage 4: BrowserUseAgent._run_plp_flow() を実際に通す薄い統合テスト
+    
+    Given:
+    - BrowserUseAgent instance
+    - Mocked Page, BrowserContext, RunContext
+    - PlpDriver が特定の PlpNavigationResult を返す
+    
+    When:
+    - BrowserUseAgent._run_plp_flow() が実行される
+    
+    Then:
+    - PlpDriver.navigate_to_pdp() が新シグネチャ（target_url, timeout_ms）で呼ばれる
+    - RunContext.save_json("plp_navigation_result.json", ...) が呼ばれる
+    - 保存内容に新フィールドが含まれている
+    """
+    # PlpNavigationResult をモック
+    nav_result = PlpNavigationResult(
+        pdp_url="https://example.com/product/123",
+        pdp_opened_in_new_tab=False,
+        plp_url="https://example.com/category",
+        tiles_seen=7,
+        trap_detected=False,
+        trap_reason=None,
+        recovery_attempted=False,
+        recovery_successful=False,
+        overlays_handled=["cookie"],
+        navigation_method="same_tab",
+        errors=[],
+    )
+    
+    # RunContext.save_json を Spy 化
+    saved_data = {}
+    original_save_json = run_context.save_json
+    
+    def spy_save_json(name: str, data: Dict[str, Any]) -> None:
+        if name == "plp_navigation_result.json":
+            saved_data.clear()
+            saved_data.update(data)
+        else:
+            original_save_json(name, data)
+    
+    run_context.save_json = spy_save_json
+    
+    # NavigationDriver とその他の依存関係をモック
+    from app.agents.browser.navigation_driver import NavigationOutcome, NavigationContext
+    
+    mock_nav_outcome = NavigationOutcome(
+        plp_materialized=False,
+        trap_detected=False,
+        pdp_links=[],  # 空リストで PlpDriver が呼ばれるようにする
+    )
+    
+    # settings を準備
+    settings = {
+        "timeout_sec": 60,
+        "overall_plp_budget_ms": 120000,
+    }
+    
+    # PlpDriver と NavigationDriver をモック
+    with patch('app.agents.browser_use_agent.PlpDriver') as mock_plp_driver_class, \
+         patch('app.agents.browser_use_agent.NavigationDriver') as mock_nav_driver_class, \
+         patch('app.agents.browser_use_agent.NavigationContext') as mock_nav_ctx_class, \
+         patch.object(browser_use_agent, '_run_pdp_flow', new_callable=AsyncMock) as mock_run_pdp_flow, \
+         patch.object(browser_use_agent, '_ensure_telemetry', return_value=MagicMock()):
+        
+        # _run_pdp_flow が DiscoveryResult を返すようにモック
+        from app.models.result_models import DiscoveryResult
+        mock_run_pdp_flow.return_value = DiscoveryResult(
+            ok=True,
+            site="example",
+            query="test",
+            message="Success",
+        )
+        
+        # PlpDriver をモック
+        mock_plp_driver = AsyncMock()
+        mock_plp_driver.navigate_to_pdp = AsyncMock(return_value=nav_result)
+        mock_plp_driver.page = mock_page
+        mock_plp_driver_class.return_value = mock_plp_driver
+        
+        # NavigationDriver をモック
+        mock_nav_driver = AsyncMock()
+        mock_nav_driver.run_plp_flow = AsyncMock(return_value=mock_nav_outcome)
+        mock_nav_driver.collect_pdp_links = AsyncMock(return_value=[])
+        mock_nav_driver_class.return_value = mock_nav_driver
+        mock_nav_ctx_class.return_value = MagicMock()
+        
+        # _run_plp_flow を実行
+        import time
+        result = await browser_use_agent._run_plp_flow(
+            page=mock_page,
+            context=mock_context,
+            site="example",
+            query="test",
+            site_config=site_config,
+            settings=settings,
+            run_context=run_context,
+            target_url="https://example.com/category",
+            start_t=time.time(),
+            budget_ms=60000,
+        )
+        
+        # PlpDriver.navigate_to_pdp が新シグネチャで呼ばれたことを確認
+        mock_plp_driver.navigate_to_pdp.assert_called_once()
+        call_kwargs = mock_plp_driver.navigate_to_pdp.call_args.kwargs
+        assert "target_url" in call_kwargs
+        assert "timeout_ms" in call_kwargs
+        assert call_kwargs["target_url"] == "https://example.com/category"
+        assert isinstance(call_kwargs["timeout_ms"], int)
+        
+        # RunContext.save_json が呼ばれたことを確認
+        assert len(saved_data) > 0, "plp_navigation_result.json が保存されていません"
+        
+        # 新フィールドが保存されていることを確認
+        assert saved_data.get("pdp_url") == "https://example.com/product/123"
+        assert saved_data.get("plp_url") == "https://example.com/category"
+        assert saved_data.get("tiles_seen") == 7
+        assert saved_data.get("overlays_handled") == ["cookie"]
+        assert saved_data.get("navigation_method") == "same_tab"
+        
+        # 既存フィールドも確認
+        assert saved_data.get("trap_detected") is False
+        assert saved_data.get("recovery_attempted") is False
+        assert saved_data.get("recovery_successful") is False
+        assert saved_data.get("errors") == []
```

## テストの目的

1. **BrowserUseAgent._run_plp_flow() を実際に通す**
   - 実際のメソッド呼び出しフローをテスト
   - NavigationDriver や PlpDriver の統合を確認

2. **新シグネチャの確認**
   - `PlpDriver.navigate_to_pdp()` が `target_url`, `timeout_ms` で呼ばれることを確認

3. **RunContext への保存確認**
   - `plp_navigation_result.json` に新フィールドが保存されることを確認

4. **既存テストとの整合性**
   - 既存の fixture（`browser_use_agent`, `mock_page`, `run_context` 等）を再利用

