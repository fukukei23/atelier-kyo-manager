# 実ブラウザテスト実行手順

## 問題

WSL環境では、Cursor のターミナルから直接コマンド出力を取得できない場合があります。

## 解決方法

以下のいずれかの方法で実ブラウザテストを実行してください。

### 方法1: シェルスクリプトを使用（推奨）

WSLターミナルで直接実行：

```bash
cd /home/yn441611/atelier-kyo-manager
source venv/bin/activate
./check_browser_test.sh
```

このスクリプトは：
- テストを実行
- ログを `browser_test_YYYYMMDD_HHMMSS.log` に保存
- 重要なログメッセージを抽出して表示
- 実行結果ディレクトリと `result.json` を表示

### 方法2: 直接コマンド実行

```bash
cd /home/yn441611/atelier-kyo-manager
source venv/bin/activate
python tools/run_browser_use.py \
  --site "MONCLER_OFFICIAL" \
  --url "https://www.moncler.com/en-int/women/outerwear/all-down-jackets/?forceLocale=en-int&shipToCountry=GB" \
  --query "down jacket" \
  --headful \
  --timeout 120
```

### 方法3: ログファイルを確認

テスト実行後、以下のコマンドでログを確認：

```bash
# 最新のログファイルを確認
ls -lt browser_test_*.log | head -1 | awk '{print $NF}' | xargs tail -100

# 最新の実行結果を確認
ls -td instance/runs/* | head -1 | xargs -I {} cat {}/result.json | python -m json.tool
```

## 確認ポイント

テストが成功した場合、以下のログが表示されるはずです：

1. **NavigationDriver のログ**
   - `[NavigationDriver]` で始まるログ
   - `[PLP→PDP]` で始まるログ

2. **PDP リンク収集**
   - `[PLP→PDP][1a]` または `[PLP→PDP][1b]` でリンクが見つかる
   - `[PLP→PDP] collected X PDP-like links` が表示される

3. **結果**
   - `result.ok=true` が表示される
   - `instance/runs/YYYYMMDD_HHMMSS_XXX/result.json` に結果が保存される

## トラブルシューティング

### エラー: `NameError: name 'TelemetryClient' is not defined`
→ 既に修正済み。`app/agents/browser_use_agent.py` のインポートを確認してください。

### エラー: `No PDP links found`
→ `app/config/sites/overrides.local.json` の `selectors.plp.pdp_link_selectors` を確認してください。
→ 既に `"a[href*='/products/']"` などのセレクタを追加済みです。

### エラー: `Timeout after 180s`
→ プロキシの問題やネットワークの問題の可能性があります。
→ `--use-proxy` フラグを外して試してください。

### ブラウザが起動しない
→ Playwright の依存関係を確認：
```bash
playwright install-deps chromium
playwright install chromium
```

