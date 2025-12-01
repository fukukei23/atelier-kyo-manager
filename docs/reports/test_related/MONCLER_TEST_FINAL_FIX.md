# Moncler テスト最終修正案

**作成日時**: 2025-11-30  
**対象テスト**: `test_product_extractor_moncler_pdp_sample`

---

## 問題点の分析

テスト結果ファイルでは「実行されたテスト数: 0」となっていますが、ターミナル出力では「1 failed, 13 warnings」と表示されています。

これは、テストが実行されているが、`conftest.py` の `pytest_runtest_logreport` フックが正しく結果を記録していない可能性があります。

---

## 実施済みの修正

1. ✅ `images` 設定の正規化（辞書形式 → リスト形式）
2. ✅ Locator モックの2段階構造化（`.first` プロパティ対応）
3. ✅ タイトル、価格、通貨、説明の Locator モック修正

---

## 残っている問題

### 1. `availability` 設定の形式

テストコードでは `availability` が辞書形式になっていますが、`_get_pdp_config()` ではリスト形式を期待しています。

**テストコード**:
```python
"availability": {
    "selectors": [...],
    "patterns": [...]
}
```

**実装**:
```python
"availability": pdp_cfg.get("availability") or [],
```

**対応**: `_get_pdp_config()` で辞書形式にも対応するか、テストコードをリスト形式に変更する必要があります。

---

## 推奨される修正

### 修正1: `availability` 設定の正規化を追加

`_get_pdp_config()` に以下のような処理を追加：

```python
# availability が辞書形式の場合、selectors を取得
availability_cfg = pdp_cfg.get("availability")
if isinstance(availability_cfg, dict):
    availability_selectors = availability_cfg.get("selectors", [])
    availability_patterns = availability_cfg.get("patterns", [])
    if availability_patterns:
        pdp_config["availability_patterns"] = availability_patterns
else:
    availability_selectors = availability_cfg or []

"availability": availability_selectors,
```

### 修正2: テストコードの `availability` 設定を簡素化

テストコードで `availability` をリスト形式に変更し、`patterns` は `availability_patterns` として別途設定：

```python
"availability": [
    "[data-testid='stock-status']",
    "[itemprop='availability']",
    "meta[property='product:availability']"
],
"availability_patterns": [
    "out of stock",
    "in stock",
    "pre-order",
    "在庫なし",
    "在庫あり"
],
```

---

## 次のステップ

1. **エラーメッセージの確認**: WSL環境に直接ログインして、実際のエラーメッセージを確認してください。
2. **修正の適用**: 上記の修正案のいずれかを適用してください。
3. **テストの再実行**: 修正後、テストを再実行して結果を確認してください。

---

**ステータス**: 🔄 修正案提示（エラーメッセージ確認待ち）

