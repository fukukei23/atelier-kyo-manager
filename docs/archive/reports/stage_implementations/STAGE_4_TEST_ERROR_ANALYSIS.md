# Stage 4: PlpDriver テストエラー分析

## テスト結果サマリー

**実行日時**: 2025-01-XX  
**結果**: 4 failed, 3 passed

### 失敗したテストの詳細分析

#### 1. test_plp_driver_materialize_tiles

**エラー**: `'coroutine' object has no attribute 'count'`

**原因**: 
- `page.locator(selector).count()` が呼ばれている
- `locator()` が返す Locator オブジェクトの `count()` メソッドは async メソッド
- モックで `loc.count = AsyncMock(return_value=12)` としているが、`locator()` 自体が AsyncMock を返しているため、`locator(selector)` の結果が coroutine になっている

**修正方法**:
- `page.locator` 自体をモックして、呼び出しごとに適切な Locator オブジェクトを返すようにする
- `locator()` が返すオブジェクトの `count()` メソッドが正しく動作するようにする

#### 2. test_plp_driver_trap_detection

**エラー**: `trap_detected is False` (期待値: True)

**原因**:
- `_materialize_plp_tiles()` がモックされており、その後に trap 判定が実行される
- しかし、`navigate_to_pdp()` の最初の処理で `_handle_overlays()` が呼ばれ、その中で `_is_trap_page()` が呼ばれる可能性がある
- または、trap 判定が実行される前に URL が変更されている

**ログから**:
```
WARNING [PlpDriver] Trap/legal page detected: https://example.com/products
```
- 最初の URL は "https://example.com/legal" だが、その後 "https://example.com/products" に変更されている
- しかし、trap 判定時には既に "products" URL になっている

**修正方法**:
- trap 判定のタイミングを確認
- `_is_trap_page()` の side_effect を最初の URL で True を返すようにする
- または、trap 判定前に URL を保存しておく

#### 3. test_plp_driver_click_tile

**エラー**: `assert None is not None`

**原因**:
- `_click_tile_and_navigate_to_pdp()` が `None` を返している
- ログ: `WARNING [PlpDriver] Could not find any clickable link or card.`
- `locator()` が正しく動作していないか、リンクが見つからない

**修正方法**:
- `locator()` のモック設定を修正
- `locator(selector).count()` と `locator(selector).nth(i)` が正しく動作するようにする

#### 4. test_plp_driver_handle_overlays

**エラー**: `cookie_locator_first.click.called is False`

**原因**:
- `_handle_cookie_banner()` が `page.locator(sel).first` を使用している
- モックでは `locator().first` が正しく設定されていない
- または、セレクタがマッチしていない

**修正方法**:
- `locator().first` が正しく動作するようにモックを修正
- セレクタのマッチングを確認

## 修正が必要な箇所

1. **`locator()` のモック方法を修正**
   - `page.locator(selector)` が返す Locator オブジェクトを正しくモックする
   - `locator().count()` が async メソッドとして正しく動作するようにする
   - `locator().first` が Locator オブジェクトを返すようにする

2. **trap 判定のタイミングを修正**
   - `navigate_to_pdp()` 内での trap 判定の順序を確認
   - 最初の URL で trap 判定が実行されるようにする

3. **overlay 処理のモックを修正**
   - `locator().first.count()` が正しく動作するようにする

## 次のステップ

1. テストファイルのモック設定を修正
2. テストを再実行して結果を確認
3. すべてのテストが成功するまで繰り返し修正

