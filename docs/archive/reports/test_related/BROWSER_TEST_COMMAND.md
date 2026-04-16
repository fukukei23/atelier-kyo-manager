# 実ブラウザテストコマンド

## 正しいコマンド

`run_orchestrator.py` は `brand` という位置引数が必要で、`--site` や `--query` などの引数は受け付けていません。

代わりに `tools/run_browser_use.py` を使用してください：

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

## 確認ポイント

実行後、以下のログが出力されることを確認：

1. **NavigationDriver のログ**
   - `[NavigationDriver]` で始まるログ
   - `[PLP→PDP]` で始まるログ

2. **site_config の使用**
   - `selectors.plp` を使用しているメッセージ
   - `navigation.header_search` を使用しているメッセージ
   - `navigation.overlays` を使用しているメッセージ

3. **動作確認**
   - PLP が表示・スクロールされる
   - いくつかの PDP が開かれる
   - エラーが発生しない

## 代替コマンド（run_orchestrator.py を使用する場合）

`run_orchestrator.py` を使用する場合は、以下の形式：

```bash
python run_orchestrator.py "down jacket" --headful --items 5
```

ただし、この場合は site や URL を指定できないため、`tools/run_browser_use.py` の方が適切です。

