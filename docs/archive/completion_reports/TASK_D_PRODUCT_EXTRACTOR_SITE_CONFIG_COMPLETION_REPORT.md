# Task D: Stage 5 – Extractor の site_config 駆動化 完了レポート

## 実装日時
2025年11月28日

## 概要

PDP 抽出ロジックを「サイトごとにハードコード」するのではなく、
site_config JSON で定義されたセレクタ・正規化ルールに基づいて抽出する汎用 Extractor を実装しました。

## 実装ステップ

### 1. ProductExtractor クラスの作成

**新規ファイル**: `app/agents/browser/product_extractor.py`

**主要なクラスとデータ構造**:
- `ProductInfo`: 抽出された商品情報を格納するデータクラス
  - `title`, `price`, `currency`, `images`, `sizes`, `colors`, `description`, `brand`, `list_price`, `discount_pct`, `url`, `raw_html_path`
  
- `PriceRules`: 価格正規化ルールを格納するデータクラス
  - `strip_chars`, `decimal_separator`, `thousands_separator`

- `ProductExtractor`: PDP から商品情報を抽出する汎用 Extractor
  - `extract()`: メインの抽出メソッド
  - `_extract_title()`: タイトル抽出
  - `_extract_price()`: 価格抽出
  - `_extract_price_with_size_option()`: 価格抽出（サイズ選択を試行）
  - `_extract_currency()`: 通貨抽出
  - `_extract_images()`: 画像抽出
  - `_extract_sizes()`: サイズ抽出
  - `_extract_colors()`: カラー抽出
  - `_extract_description()`: 説明抽出
  - `_extract_brand()`: ブランド抽出
  - `_extract_list_price_and_discount()`: 定価・割引率抽出
  - `_click_size_to_reveal_price()`: サイズクリックで価格表示
  - `_extract_from_json_ld_or_meta()`: JSON-LD / Meta タグからのフォールバック抽出
  - `_normalize_price()`: 価格正規化

### 2. BrowserExtractionService の修正

**変更ファイル**: `app/agents/browser/extractor.py`

**主な変更点**:
- `ProductExtractor` のインポートを追加
- `_extract_from_pdp()` メソッドを修正して `ProductExtractor` を使用
- 既存の Moncler 専用抽出やフォールバックロジックは維持（互換性維持）
- `run_context` パラメータを `_extract_from_pdp()` に追加

### 3. site_config スキーマの拡張

**対応する site_config キー**:
- `selectors.pdp.title`: タイトルセレクタ
- `selectors.pdp.price`: 価格セレクタ
- `selectors.pdp.currency`: 通貨セレクタ
- `selectors.pdp.images`: 画像セレクタ
- `selectors.pdp.size`: サイズセレクタ
- `selectors.pdp.color`: カラーセレクタ
- `selectors.pdp.description`: 説明セレクタ
- `selectors.pdp.brand`: ブランドセレクタ
- `selectors.pdp.list_price`: 定価セレクタ
- `selectors.pdp.size_button`: サイズボタンセレクタ
- `selectors.pdp.size_select_policy`: サイズ選択ポリシー
  - `mode`: "off" | "first_instock" | "by_label"
  - `prefer_labels`: 優先ラベルリスト
  - `price_wait_ms`: 価格表示待機時間
- `price_rules`: 価格正規化ルール
  - `strip_chars`: 削除する文字リスト
  - `decimal_separator`: 小数点区切り文字
  - `thousands_separator`: 千の位区切り文字

### 4. 既存ロジックの統合

- 既存の `extract_title_price()` のロジックを `ProductExtractor` に統合
- 既存の `_extract_price_with_size_option()` のロジックを `ProductExtractor` に統合
- 既存の `_extract_ld_json_price()` と `_extract_meta_price()` のロジックを `ProductExtractor` に統合
- Moncler 専用抽出は既存の `MonclerPDPExtractor` を維持（フォールバック）

## 変更ファイル一覧

### 新規作成ファイル

1. **`app/agents/browser/product_extractor.py`**
   - ProductExtractor クラス
   - ProductInfo データクラス
   - PriceRules データクラス

### 変更ファイル

1. **`app/agents/browser/extractor.py`**
   - `ProductExtractor` のインポートを追加
   - `_extract_from_pdp()` を修正して `ProductExtractor` を使用
   - `run_context` パラメータを追加

## 動作確認結果

### 静的解析結果
- リンター警告: Playwright のインポート解決警告（実行時には問題なし）
- 型チェック: 問題なし（`Tuple` の型ヒントを修正）

### コードレビュー結果
- ProductExtractor クラスの実装: ✅ 完了
- BrowserExtractionService の統合: ✅ 完了
- site_config への依存: ✅ 完了
- 既存のフォールバック処理: ✅ 維持

### テスト結果
- ユニットテスト: Task E で実装予定
- 統合テスト: Task E で実装予定

## 設計上の改善点

### アーキテクチャの改善
1. **責務の分離**
   - PDP 抽出ロジックが ProductExtractor に集約
   - BrowserExtractionService は ProductExtractor を呼び出すだけの薄いハブに

2. **サイト非依存化**
   - サイト固有のロジックは `site_config` から参照
   - コード側は汎用的な抽出ロジックのみ

3. **拡張性の向上**
   - 新しいサイトを追加する際、コード改変ではなく `site_config` の追加だけで対応可能
   - 新しい抽出フィールド（例: レビュー、在庫状況）を追加しやすい

### 将来の拡張性への配慮
- `ProductInfo` データクラスに新しいフィールドを追加可能
- `site_config` に新しいセレクタを追加可能
- 価格正規化ルールを `price_rules` で柔軟に設定可能

### コード品質の向上
- テスト容易性の向上（ProductExtractor を独立してテスト可能）
- 可読性の向上（BrowserExtractionService が簡潔に）
- 型安全性の向上（ProductInfo データクラスで型が明確に）

## 既知の制約・注意事項

### 既存コードとの互換性
- 既存の `MonclerPDPExtractor` はフォールバックとして維持
- 既存の `extract_title_price()` はフォールバックとして使用
- 既存の `_extract_ld_json_price()` と `_extract_meta_price()` はフォールバックとして使用

### 制限事項やトレードオフ
- 現在の実装では、ProductExtractor が失敗した場合に既存のフォールバックロジックを使用
- 将来的には、すべての抽出を ProductExtractor に統一することを推奨

### 移行時の注意点
- 既存の実行スクリプトは変更不要
- `site_config` に必要なキーが定義されていることを確認
- 新しいサイトを追加する際は、`site_config` に適切なセレクタを定義

## 次のステップ

### 推奨されるフォローアップアクション

1. **Task E: テストの追加**
   - ProductExtractor のユニットテスト（PDP HTML fixture）
   - BrowserUseAgent の E2E テスト（Moncler 統合テスト）

2. **site_config スキーマの標準化**
   - ProductExtractor が使用する `site_config` キーの標準化
   - ドキュメント化

3. **フォールバックロジックの削減**
   - すべての抽出を ProductExtractor に統一
   - 既存のフォールバックロジックを段階的に削除

4. **新しい抽出フィールドの追加**
   - レビュー、在庫状況、配送情報などの抽出
   - `ProductInfo` データクラスに新しいフィールドを追加

## 関連ファイル

- `app/agents/browser/product_extractor.py` - ProductExtractor クラス
- `app/agents/browser/extractor.py` - BrowserExtractionService クラス
- `app/extractors/product_info_extractor.py` - 既存の抽出ロジック（フォールバック）
- `app/extractors/moncler_extractor.py` - Moncler 専用抽出（フォールバック）

