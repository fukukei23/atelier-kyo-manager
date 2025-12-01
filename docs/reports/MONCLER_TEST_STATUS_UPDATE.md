# Moncler テスト状況更新

**更新日時**: 2025-11-30 14:47  
**状況**: テストが失敗しているが、詳細なエラーメッセージが確認できない

---

## 現在の状況

- ✅ **テストは実行されている**: ターミナル出力で「1 failed, 13 warnings」と表示
- ❌ **エラーメッセージが不明**: テスト結果ファイルに詳細が記録されていない
- ❓ **原因特定が困難**: WSL環境の制約でターミナル出力を直接確認できない

---

## 実施済みの修正

### 1. ✅ `images` 設定の正規化
- `_normalize_images_config()` メソッド追加
- `_get_image_attr()` メソッド追加
- `_get_image_base_url()` メソッド追加

### 2. ✅ Locator モックの2段階構造化
- すべての Locator を `.first` プロパティに対応する2段階構造に変更
- `title_locator_base.first` → `title_locator_first`
- `price_locator_base.first` → `price_locator_first`
- `currency_locator_base.first` → `currency_locator_first`
- `description_locator_base.first` → `description_locator_first`

### 3. ✅ `availability` 設定の正規化
- `_normalize_availability_config()` メソッド追加
- `_get_availability_patterns()` メソッド追加

---

## 推測される問題点

### 1. 価格正規化の問題

テストコードでは価格が `€1,234.56` で、期待値が `1234.56` です。

`price_rules` には `strip_chars: ["€", ",", " ", ...]` が設定されているため、以下の処理が行われます：

1. `_normalize_price()`: `strip_chars` を削除 → `€` と `,` が削除される
2. 正規表現パターンで数値部分を抽出 → `1234.56`
3. `_normalize_price_to_float()`: `thousands_separator` を削除 → 既に削除済みなので問題なし
4. `decimal_separator` を "." に統一 → 既に "." なので問題なし
5. float に変換 → `1234.56`

**理論的には正しく動作するはずです。**

### 2. モックの問題

`locator_side_effect` 関数で、セレクタに応じて Locator を返していますが、すべてのセレクタパターンが正しくマッチしていない可能性があります。

### 3. テスト結果ファイルの記録問題

`conftest.py` の `pytest_runtest_logreport` フックが正しく動作していない可能性があります。テストは実行されているが、結果ファイルに記録されていない可能性があります。

---

## 次のステップ

### 推奨される確認方法

1. **WSL環境に直接ログインしてテストを実行**:

```bash
cd /home/yn441611/atelier-kyo-manager
source venv/bin/activate
pytest tests/test_product_extractor.py::test_product_extractor_moncler_pdp_sample -xvs --tb=long
```

2. **エラーメッセージを共有**: 実際のエラーメッセージを共有してください。

3. **デバッグ出力を追加**: テストコードに `print()` を追加して、各段階での値を確認する。

---

## 修正候補

実際のエラーメッセージを確認してから、適切な修正を提案します。

---

**ステータス**: 🔄 デバッグ中（エラーメッセージ確認待ち）

