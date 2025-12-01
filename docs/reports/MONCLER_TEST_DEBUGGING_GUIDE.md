# Moncler テストデバッグガイド

**作成日時**: 2025-11-30  
**対象テスト**: `test_product_extractor_moncler_pdp_sample`

---

## 現在の状況

テストが失敗している（「1 failed, 13 warnings」）が、詳細なエラーメッセージが確認できていません。

---

## 実施済みの修正

### 1. `images` 設定の正規化

`_get_pdp_config()` で `images` が辞書形式の場合、`selectors` キーからリストを取得するように修正。

### 2. Locator モックの修正

`_extract_title`, `_extract_price`, `_extract_currency`, `_extract_description` が `page.locator(selector).first` を使用するため、すべての Locator モックを2段階の構造に変更：

- `title_locator_base.first` → `title_locator_first`
- `price_locator_base.first` → `price_locator_first`
- `currency_locator_base.first` → `currency_locator_first`
- `description_locator_base.first` → `description_locator_first`

---

## 確認が必要な点

### 1. `availability` 設定の形式

テストコードでは `availability` が辞書形式になっています：

```python
"availability": {
    "selectors": [...],
    "patterns": [...]
}
```

しかし、`_get_pdp_config()` ではリスト形式を期待しています：

```python
"availability": pdp_cfg.get("availability") or [],
```

**対応**: `_get_pdp_config()` で辞書形式にも対応するか、テストコードをリスト形式に変更する必要があります。

### 2. 価格正規化

テストでは価格が `€1,234.56` で、期待値が `1234.56` です。`price_rules` に `strip_chars: ["€", ",", " ", ...]` が設定されているため、正しく正規化されるはずです。

**確認**: `_normalize_price_to_float()` が正しく動作しているか確認が必要です。

### 3. テスト実行の問題

テスト結果ファイルでは「実行されたテスト数: 0」となっていますが、ターミナル出力では「1 failed」となっています。これは `conftest.py` の `pytest_runtest_logreport` フックが正しく動作していない可能性があります。

---

## 次のステップ

### 推奨される確認方法

1. **WSL環境に直接ログインしてテストを実行**:

```bash
cd /home/yn441611/atelier-kyo-manager
source venv/bin/activate
pytest tests/test_product_extractor.py::test_product_extractor_moncler_pdp_sample -xvs --tb=long
```

2. **エラーメッセージを確認**: 実際のエラーメッセージを共有してください。

3. **デバッグ出力を追加**: テストコードに `print()` を追加して、各段階での値を確認する。

---

## 修正候補

### 修正1: `availability` 設定の正規化

`_get_pdp_config()` に以下のようなヘルパーメソッドを追加：

```python
def _normalize_availability_config(self, availability_cfg: Any) -> Tuple[List[str], List[str]]:
    """availability 設定を正規化"""
    if not availability_cfg:
        return [], []
    
    if isinstance(availability_cfg, list):
        return availability_cfg, []
    
    if isinstance(availability_cfg, dict):
        selectors = availability_cfg.get("selectors", [])
        patterns = availability_cfg.get("patterns", [])
        return selectors, patterns
    
    return [], []
```

### 修正2: テストコードの `availability` 設定をリスト形式に変更

テストコードで `availability` をリスト形式に変更するか、辞書形式の `patterns` を `availability_patterns` として別途設定する。

---

## 関連ファイル

- `tests/test_product_extractor.py`: テストコード
- `app/agents/browser/product_extractor.py`: ProductExtractor の実装
- `docs/reports/TEST_RESULTS_*.txt`: テスト結果ファイル

---

**ステータス**: 🔄 デバッグ中（エラーメッセージの確認待ち）

