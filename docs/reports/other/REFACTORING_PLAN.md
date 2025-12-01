# BrowserUseAgent リファクタリング計画

## 完了したモジュール

✅ **settings.py** - 設定解決、タイムバジェット管理、コンテキストオプション構築
✅ **ui_helpers.py** - UI操作ヘルパー（オーバーレイ削除、Cookie、Geoモーダル等）
✅ **moncler_patch.py** - Moncler固有のURL正規化と回復ロジック

## 残りのモジュール（要作成）

### plp_flow.py
主要メソッド:
- `ensure_plp_materialized()` - PLPのマテリアライズ（スクロール、タイル検出）
- `collect_pdp_links()` - PLPからPDPリンクを収集
- `run_deep_extraction_phase2()` - 深い抽出フォールバック
- `normalize_abs_url()` - 絶対URL正規化
- `plp_header_search_fallback()` - ヘッダー検索フォールバック
- `click_and_capture_navigation()` - クリックとナビゲーションキャプチャ
- `click_first_card_or_link()` - 最初のカード/リンクをクリック

### pdp_flow.py
主要メソッド:
- `read_price_or_none()` - 価格読み取り
- `build_pdp_prepare_hook()` - PDP準備フック構築

### observability_hooks.py
主要メソッド:
- `perform_vrt()` - 視覚的回帰テスト
- `handle_run_failure()` - 実行失敗処理
- `run_learning_flow()` - 学習フロー
- `save_learned_selectors()` - 学習済みセレクタ保存

## 次のステップ

1. 残りの3つのモジュールを作成
2. BrowserUseAgent をリファクタリングして新しいモジュールを使用
3. 公開インターフェース（`run()`メソッド）は変更しない
4. テストと動作確認

