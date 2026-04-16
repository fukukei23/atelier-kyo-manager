# Stage 4: BrowserUseAgent 更新 - 差分形式

## 1. app/agents/browser_use_agent.py の変更

### 変更箇所: `_run_plp_flow()` メソッド内（1983-2013行目付近）

```diff
                    # --- V88.5.5: 早期失敗ロジック ---
                    # Task C: PlpDriver を使用してタイルクリック → PDP遷移
+                   # Stage 4: 拡張版 PlpDriver に対応
                    if not pdp_links:
                        self.logger.warning("[Fallback] No hrefs after search. Clicking first card using PlpDriver...")
                        try:
                            plp_driver = PlpDriver(
                                page=page,
                                context=context,
                                site_config=site_config,
                                run_context=run_context,
                                logger=self.logger,
                                telemetry=self._ensure_telemetry(),
                            )
-                           nav_result = await plp_driver.navigate_to_pdp(
-                               start_t=start_t,
-                               budget_ms=budget_ms,
-                               target_url=target_url,
-                           )
+                           # Stage 4: 新しいシグネチャを使用（target_url, timeout_ms を優先）
+                           timeout_ms = min(budget_ms, int(settings.get("timeout_sec", 60)) * 1000)
+                           nav_result = await plp_driver.navigate_to_pdp(
+                               target_url=target_url,
+                               timeout_ms=timeout_ms,
+                               # 後方互換性のため、旧パラメータも渡す（内部で fallback として使用）
+                               start_t=start_t,
+                               budget_ms=budget_ms,
+                           )
+                           
+                           # Stage 4: 新しいフィールドを RunContext に保存
+                           run_context.save_json("plp_navigation_result.json", {
+                               "pdp_url": nav_result.pdp_url,
+                               "plp_url": nav_result.plp_url,
+                               "tiles_seen": nav_result.tiles_seen,
+                               "trap_detected": nav_result.trap_detected,
+                               "trap_reason": nav_result.trap_reason,
+                               "recovery_attempted": nav_result.recovery_attempted,
+                               "recovery_successful": nav_result.recovery_successful,
+                               "overlays_handled": nav_result.overlays_handled,
+                               "navigation_method": nav_result.navigation_method,
+                               "errors": nav_result.errors,
+                               "pdp_opened_in_new_tab": nav_result.pdp_opened_in_new_tab,
+                           })
+                           
+                           # Stage 4: 詳細なログ出力
+                           if nav_result.trap_detected:
+                               if nav_result.recovery_successful:
+                                   self.logger.info(
+                                       f"[PlpDriver] Trap detected and recovered: {nav_result.trap_reason} "
+                                       f"(overlays: {nav_result.overlays_handled}, method: {nav_result.navigation_method})"
+                                   )
+                               else:
+                                   self.logger.warning(
+                                       f"[PlpDriver] Trap detected but recovery failed: {nav_result.trap_reason} "
+                                       f"(errors: {nav_result.errors})"
+                                   )
+                           else:
+                               self.logger.info(
+                                   f"[PlpDriver] Successfully navigated to PDP: {nav_result.pdp_url} "
+                                   f"(tiles_seen: {nav_result.tiles_seen}, "
+                                   f"overlays: {nav_result.overlays_handled}, "
+                                   f"method: {nav_result.navigation_method})"
+                               )
+                           
+                           # エラーがある場合は警告
+                           if nav_result.errors:
+                               for error in nav_result.errors:
+                                   self.logger.warning(f"[PlpDriver] Error: {error}")
+                           
                            # PlpDriver が PDP に遷移した場合、そのページで PDP 抽出を実行
                            # PlpDriver 内で既にページ遷移が完了しているので、plp_driver.page を使用
                            pdp_page = plp_driver.page
                            return await self._run_pdp_flow(pdp_page, site, query, settings, run_context, site_config)
```

## 2. tests/test_browser_use_agent_plp_integration.py の変更

### 変更1: 既存テストの更新（Stage 4 フィールド対応）

```diff
    # PlpDriver.navigate_to_pdp の戻り値をモック（Stage 4: 新フィールドを含む）
    expected_result = PlpNavigationResult(
        pdp_url="https://example.com/product/123",
        pdp_opened_in_new_tab=False,
        plp_url="https://example.com/category",
        tiles_seen=5,
        trap_detected=False,
        trap_reason=None,
        recovery_attempted=False,
+       recovery_successful=False,  # Stage 4: 追加
+       overlays_handled=[],  # Stage 4: 追加
+       navigation_method="same_tab",  # Stage 4: 追加
+       errors=[],  # Stage 4: 追加
    )
```

```diff
        import time
-       result = await plp_driver.navigate_to_pdp(
-           start_t=time.time(),
-           budget_ms=30000,
-       )
+       # Stage 4: 新しいシグネチャで呼び出し
+       result = await plp_driver.navigate_to_pdp(
+           target_url="https://example.com/category",
+           timeout_ms=30000,
+       )
        
        # PlpDriver が正しく動作することを確認
        assert isinstance(result, PlpNavigationResult)
        assert result.pdp_url == expected_result.pdp_url
        assert result.tiles_seen == expected_result.tiles_seen
+       # Stage 4: 新しいフィールドも確認
+       assert result.navigation_method == expected_result.navigation_method
+       assert isinstance(result.overlays_handled, list)
```

### 変更2: 新しいテストの追加

**追加テスト1**: `test_browser_use_agent_handles_trap_detection`
- Trap 検出とリカバリ失敗のケースをテスト
- `trap_detected=True, recovery_successful=False` の場合の処理を確認

**追加テスト2**: `test_browser_use_agent_saves_overlays_handled`
- Overlay 処理結果が RunContext に保存されることを確認
- `overlays_handled=["cookie", "geo"]` が保存されることを確認

## 変更のまとめ

### 主な変更点

1. **PlpDriver 呼び出しの更新**
   - 旧シグネチャ（`start_t`, `budget_ms`）から新シグネチャ（`target_url`, `timeout_ms`）に変更
   - 後方互換性のため、旧パラメータも引き続き渡す

2. **PlpNavigationResult の新フィールドを保存**
   - `recovery_successful`
   - `overlays_handled`
   - `navigation_method`
   - `errors`
   - これらを `plp_navigation_result.json` に保存

3. **ログ出力の改善**
   - Trap 検出時の詳細情報（リカバリ成功/失敗、オーバーレイ、ナビゲーション方法）
   - エラーメッセージの詳細な記録

4. **テストの拡張**
   - 既存テストを新フィールドに対応
   - 新しいテストケースを追加（Trap 検出、Overlay 処理）

### 後方互換性

- 既存のパラメータ（`start_t`, `budget_ms`）は引き続きサポート
- 既存の public メソッド名やクラス名は変更なし
- 既存のコードが動作し続けることを確認

