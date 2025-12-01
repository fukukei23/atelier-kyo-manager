# Stage 4: PlpDriver テスト修正レポート

## テスト結果

**実行日時**: 2025-01-XX  
**結果**: 4 failed, 3 passed

### 成功したテスト

1. ✅ `test_plp_driver_trap_detection_no_recovery` - PASSED
2. ✅ `test_plp_driver_navigate_to_pdp_happy_path` - PASSED
3. ✅ `test_plp_driver_navigate_to_pdp_same_tab` - PASSED

### 失敗したテスト（修正済み）

1. ❌ `test_plp_driver_materialize_tiles` - モック設定の問題
2. ❌ `test_plp_driver_trap_detection` - trap 判定の問題
3. ❌ `test_plp_driver_click_tile` - リンク要素のモック設定の問題
4. ❌ `test_plp_driver_handle_overlays` - locator().first のモック設定の問題

## 修正内容

### 1. test_plp_driver_materialize_tiles

**問題**: `locator()` が返すオブジェクトの `.count()` メソッドが正しくモックされていない

**修正前**:
```python
mock_page.locator.return_value.count = AsyncMock(return_value=12)
```

**修正後**:
```python
def locator_side_effect(selector):
    loc = AsyncMock()
    loc.count = AsyncMock(return_value=12)
    return loc

mock_page.locator.side_effect = locator_side_effect
mock_page.evaluate = AsyncMock()  # スクロール用
mock_page.wait_for_timeout = AsyncMock()
mock_page.wait_for_load_state = AsyncMock()
```

**理由**: `locator()` が呼ばれるたびに新しい Locator オブジェクトを返す必要がある

### 2. test_plp_driver_trap_detection

**問題**: リカバリ後の trap 判定が正しく動作していない。モックの `return_value=True` が効かない

**修正前**:
```python
patch.object(driver, '_is_trap_page', return_value=True)
```

**修正後**:
```python
def is_trap_page_side_effect(url, trap_config):
    # リカバリ後の URL では False を返す
    if "products" in url:
        return False
    return True  # リカバリ前は True

patch.object(driver, '_is_trap_page', side_effect=is_trap_page_side_effect)
patch.object(driver, '_click_tile_and_navigate_to_pdp', return_value=mock_page)
```

**理由**: `_is_trap_page()` が URL に基づいて動作するように、`side_effect` を使用して URL によって返り値を変える

### 3. test_plp_driver_click_tile

**問題**: リンク要素が見つからず、`_click_tile_and_navigate_to_pdp()` が `None` を返す

**修正前**:
```python
with patch.object(driver, '_click_and_capture_navigation', return_value=new_page):
```

**修正後**:
```python
# リンク要素のモック
link_element = AsyncMock()
link_element.scroll_into_view_if_needed = AsyncMock()
link_element.get_attribute = AsyncMock(return_value="https://example.com/product/123")
link_element.click = AsyncMock()

def locator_side_effect(selector):
    loc = AsyncMock()
    loc.count = AsyncMock(return_value=1)
    loc.nth = MagicMock(return_value=link_element)
    loc.first = AsyncMock()
    loc.first.scroll_into_view_if_needed = AsyncMock()
    loc.first.count = AsyncMock(return_value=1)
    loc.first.click = AsyncMock()
    return loc

mock_page.locator.side_effect = locator_side_effect

with patch.object(driver, '_click_and_wait_for_navigation', return_value=new_page):
```

**理由**: `_click_tile_and_navigate_to_pdp()` の内部実装に合わせて、`locator()`, `count()`, `nth()`, `get_attribute()` が正しく動作するようにモックを設定

### 4. test_plp_driver_handle_overlays

**問題**: `locator().first` が正しくモックされていない

**修正前**:
```python
cookie_locator = AsyncMock()
cookie_locator.count = AsyncMock(return_value=1)
cookie_locator.click = AsyncMock()
```

**修正後**:
```python
# Cookie バナーのモック
cookie_locator_first = AsyncMock()
cookie_locator_first.count = AsyncMock(return_value=1)
cookie_locator_first.click = AsyncMock()

cookie_locator = AsyncMock()
cookie_locator.first = cookie_locator_first
cookie_locator.count = AsyncMock(return_value=1)

# Geo モーダルのモック
geo_locator_first = AsyncMock()
geo_locator_first.count = AsyncMock(return_value=1)
geo_locator_first.click = AsyncMock()

geo_locator = AsyncMock()
geo_locator.first = geo_locator_first
geo_locator.count = AsyncMock(return_value=1)

def locator_side_effect(selector):
    selector_lower = selector.lower()
    if "cookie" in selector_lower or "#cookie-accept" in selector:
        return cookie_locator
    elif "geo" in selector_lower or "close-modal" in selector_lower:
        return geo_locator
    else:
        default = AsyncMock()
        default.first = AsyncMock()
        default.first.count = AsyncMock(return_value=0)
        default.count = AsyncMock(return_value=0)
        return default
```

**理由**: `_handle_cookie_banner()` と `_handle_geo_modal()` が `locator(sel).first` を使用しているため、`first` 属性が正しく設定されている必要がある

## 期待される結果

修正後、すべてのテストが成功するはずです：

1. ✅ `test_plp_driver_materialize_tiles` - locator のモック設定を修正
2. ✅ `test_plp_driver_trap_detection` - trap 判定を side_effect で修正
3. ✅ `test_plp_driver_trap_detection_no_recovery` - 既に成功
4. ✅ `test_plp_driver_click_tile` - リンク要素のモック設定を修正
5. ✅ `test_plp_driver_navigate_to_pdp_happy_path` - 既に成功
6. ✅ `test_plp_driver_navigate_to_pdp_same_tab` - 既に成功
7. ✅ `test_plp_driver_handle_overlays` - locator().first のモック設定を修正

## テスト実行方法

```bash
cd /home/yn441611/atelier-kyo-manager
source venv/bin/activate 2>/dev/null || source myenv/Scripts/activate 2>/dev/null || true
python -m pytest tests/test_plp_driver.py -v
```

## 次のステップ

テストがすべて成功したら：

1. Task D: 新しいテストケースの追加
   - 新タブ遷移テスト
   - SPA遷移テスト（URL変更検知）
   - Trap → Recovery → PDP成功テスト
   - Overlayが2種類以上出るケース
   - PLP → PDP の URL正規化テスト

2. Task E: 最終成果物の生成
   - BrowserUseAgent への差分パッチ
   - site_config テンプレート
   - 移行ガイド

