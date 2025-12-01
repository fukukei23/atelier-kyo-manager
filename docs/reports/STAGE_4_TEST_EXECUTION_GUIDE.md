# Stage 4: PlpDriver テスト実行ガイド

## テスト実行コマンド

以下のコマンドでテストを実行してください：

```bash
cd /home/yn441611/atelier-kyo-manager
source venv/bin/activate 2>/dev/null || source myenv/Scripts/activate 2>/dev/null || true
python -m pytest tests/test_plp_driver.py -v
```

## 期待される結果

すべてのテストが成功するはずです。以下の7つのテストケースが実行されます：

1. ✅ `test_plp_driver_materialize_tiles` - タイルマテリアライズ
2. ✅ `test_plp_driver_trap_detection` - Trap検出とリカバリ成功
3. ✅ `test_plp_driver_trap_detection_no_recovery` - Trap検出とリカバリ失敗
4. ✅ `test_plp_driver_click_tile` - タイルクリック
5. ✅ `test_plp_driver_navigate_to_pdp_happy_path` - Happy path（新タブ）
6. ✅ `test_plp_driver_navigate_to_pdp_same_tab` - 同タブ遷移
7. ✅ `test_plp_driver_handle_overlays` - Overlay処理

## テスト内容の確認

### 1. 拡張版 PlpNavigationResult の新しいフィールド

以下の新しいフィールドが正しく動作することを確認：

- `recovery_successful`: リカバリの成功/失敗
- `overlays_handled`: 処理したオーバーレイの種類（例: `["cookie", "geo"]`）
- `navigation_method`: ナビゲーション方法（`"new_tab"`, `"same_tab"`, `"spa"`）
- `errors`: エラーメッセージのリスト

### 2. 後方互換性

既存のAPIが引き続き動作することを確認：

- `navigate_to_pdp(start_t=..., budget_ms=...)` - 既存のシグネチャ
- `navigate_to_pdp(timeout_ms=...)` - 新しいシグネチャ
- 既存のメソッド（`_materialize_plp_tiles()`, `_looks_like_trap_or_legal()`, etc.）はすべて保持

### 3. site_config ベースの設定取得

以下のメソッドが正しく動作することを確認：

- `_get_plp_config()`: `selectors.plp.*` から設定を取得
- `_get_overlay_config()`: `navigation.overlays.*` から設定を取得
- `_get_trap_config()`: `navigation.trap.*` から設定を取得

## トラブルシューティング

### テストが失敗する場合

1. **インポートエラー**
   - 仮想環境が正しく有効化されているか確認
   - `pip install -r requirements.txt` で依存関係をインストール

2. **モックのエラー**
   - `AsyncMock` が正しく設定されているか確認
   - テストファイルのモック設定を確認

3. **シグネチャの不一致**
   - `_handle_overlays()` は `overlays_handled` リストをパラメータとして受け取る
   - `_recover_from_trap()` は `bool` を返す

### デバッグモードで実行

より詳細な出力を得るには：

```bash
python -m pytest tests/test_plp_driver.py -v -s --tb=long
```

### 特定のテストのみ実行

```bash
python -m pytest tests/test_plp_driver.py::test_plp_driver_navigate_to_pdp_happy_path -v
```

## テストカバレッジ

現在のテストカバレッジを確認するには：

```bash
python -m pytest tests/test_plp_driver.py --cov=app.agents.browser.plp_driver --cov-report=term-missing
```

## 次のステップ

テストが成功したら：

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

