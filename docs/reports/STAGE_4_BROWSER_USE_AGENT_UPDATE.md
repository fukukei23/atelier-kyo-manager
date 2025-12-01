# Stage 4: BrowserUseAgent 更新案（拡張版 PlpDriver 対応）

## 概要

Stage 4 の拡張版 PlpDriver に対応するため、BrowserUseAgent を更新します。

## 主な変更点

### 1. PlpDriver 呼び出しの更新

**変更箇所**: `app/agents/browser_use_agent.py` の `_run_plp_flow()` メソッド内（1988-2000行目付近）

**変更前**:
```python
nav_result = await plp_driver.navigate_to_pdp(
    start_t=start_t,
    budget_ms=budget_ms,
    target_url=target_url,
)
```

**変更後**:
```python
# Stage 4: 新しいシグネチャを使用（target_url, timeout_ms を優先）
timeout_ms = min(budget_ms, int(settings.get("timeout_sec", 60)) * 1000)
nav_result = await plp_driver.navigate_to_pdp(
    target_url=target_url,
    timeout_ms=timeout_ms,
    # 後方互換性のため、旧パラメータも渡す（内部で fallback として使用）
    start_t=start_t,
    budget_ms=budget_ms,
)
```

### 2. PlpNavigationResult の新フィールドを RunContext に保存

**変更箇所**: PlpDriver の呼び出し直後

**追加内容**:
```python
# Stage 4: 新しいフィールドを RunContext に保存
run_context.save_json("plp_navigation_result.json", {
    "pdp_url": nav_result.pdp_url,
    "plp_url": nav_result.plp_url,
    "tiles_seen": nav_result.tiles_seen,
    "trap_detected": nav_result.trap_detected,
    "trap_reason": nav_result.trap_reason,
    "recovery_attempted": nav_result.recovery_attempted,
    "recovery_successful": nav_result.recovery_successful,  # Stage 4: 追加
    "overlays_handled": nav_result.overlays_handled,  # Stage 4: 追加
    "navigation_method": nav_result.navigation_method,  # Stage 4: 追加
    "errors": nav_result.errors,  # Stage 4: 追加
    "pdp_opened_in_new_tab": nav_result.pdp_opened_in_new_tab,
})
```

### 3. ログ出力の改善

**変更箇所**: PlpDriver の呼び出し直後

**追加内容**:
```python
# Stage 4: 詳細なログ出力
if nav_result.trap_detected:
    if nav_result.recovery_successful:
        self.logger.info(
            f"[PlpDriver] Trap detected and recovered: {nav_result.trap_reason} "
            f"(overlays: {nav_result.overlays_handled}, method: {nav_result.navigation_method})"
        )
    else:
        self.logger.warning(
            f"[PlpDriver] Trap detected but recovery failed: {nav_result.trap_reason} "
            f"(errors: {nav_result.errors})"
        )
else:
    self.logger.info(
        f"[PlpDriver] Successfully navigated to PDP: {nav_result.pdp_url} "
        f"(tiles_seen: {nav_result.tiles_seen}, "
        f"overlays: {nav_result.overlays_handled}, "
        f"method: {nav_result.navigation_method})"
    )

# エラーがある場合は警告
if nav_result.errors:
    for error in nav_result.errors:
        self.logger.warning(f"[PlpDriver] Error: {error}")
```

### 4. 例外処理の改善

**変更箇所**: PlpDriver 呼び出しの例外ハンドリング

**変更内容**:
- Trap 検出やリカバリ失敗などの詳細なエラー情報をログに記録
- PlpNavigationResult の errors フィールドを活用

## テスト更新

### 1. 既存テストの更新

**変更箇所**: `tests/test_browser_use_agent_plp_integration.py`

**変更内容**:
- 新しいシグネチャ（target_url, timeout_ms）での呼び出しを確認
- PlpNavigationResult の新フィールド（recovery_successful, overlays_handled, navigation_method, errors）をチェック

### 2. 新しいテストの追加

1. **Trap 検出とリカバリのテスト**
   - trap_detected=True, recovery_successful=False の場合の処理を確認
   - RunContext への JSON 保存を確認

2. **Overlay 処理のテスト**
   - overlays_handled=["cookie", "geo"] が RunContext に保存されることを確認

