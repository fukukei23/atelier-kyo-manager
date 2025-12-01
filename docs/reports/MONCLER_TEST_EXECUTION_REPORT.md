# Moncler 用 site_config テスト実行レポート

**実行日時**: 2025-01-28  
**テスト対象**: `test_product_extractor_moncler_pdp_sample`

---

## テスト実行状況

### ✅ テストコード追加完了

`tests/test_product_extractor.py` に以下のテストケースを追加しました:

- **テスト名**: `test_product_extractor_moncler_pdp_sample`
- **目的**: Moncler 用 PDP fixture を使ったサイト固有テスト
- **行数**: 約180行（833行目〜1096行目）

### テスト内容

1. **Moncler 用 site_config 準備**
   - Stage 5 新スキーマに準拠した設定
   - `selectors.pdp.*` の全フィールドを含む
   - `price_rules` を含む

2. **Mock Page の設定**
   - タイトル、価格、通貨、画像、サイズ、カラー、説明のモック設定

3. **アサーション**
   - 基本フィールド（title, price, currency, images, sizes, colors, description）
   - 価格正規化（EUR形式: "€1,234.56" → 1234.56）
   - metadata（has_title, has_price, has_currency, image_count, size_count, color_count）

---

## 実行方法

### 単体テスト実行

```bash
cd /home/yn441611/atelier-kyo-manager
source venv/bin/activate
pytest tests/test_product_extractor.py::test_product_extractor_moncler_pdp_sample -v
```

### Moncler 関連テストをすべて実行

```bash
pytest tests/test_product_extractor.py -k moncler -v
```

---

## 注意事項

WSL環境でのターミナル出力表示に問題があるため、テスト実行後は以下のいずれかで結果を確認してください：

1. **自動生成されるテスト結果ファイル**: `docs/reports/TEST_RESULTS_<timestamp>.txt`
2. **WSL環境に直接ログイン**: `wsl` コマンドで直接ログインして実行
3. **Cursor の統合ターミナル**: ターミナルタブで Ubuntu プロファイルを選択して実行

---

## 次のステップ

1. テストを実行して動作確認
2. E2E テスト: 実際の Moncler サイトで PDP 抽出を実行
3. 結果確認: `pdp_extracted_data.json` と `pdp_raw.html` を確認

---

**ステータス**: ✅ テストコード追加完了（実行待ち）

