# CursorからWSLコマンド出力を取得する方法

## 問題

WSL環境で長時間実行されるコマンド（特にブラウザテスト）の出力が、Cursorのターミナルから取得できない場合があります。

## 解決方法

### 方法1: Pythonスクリプトでリアルタイム出力（推奨）

`run_test_with_immediate_output.py` を使用：

```bash
cd /home/yn441611/atelier-kyo-manager
source venv/bin/activate
python -u run_test_with_immediate_output.py
```

このスクリプトは：
- `PYTHONUNBUFFERED=1` でバッファリングを無効化
- 行バッファリング（`bufsize=1`）で即座に出力
- リアルタイムで出力を読み取って表示

### 方法2: 環境変数でバッファリング無効化

```bash
cd /home/yn441611/atelier-kyo-manager
source venv/bin/activate
PYTHONUNBUFFERED=1 python -u tools/run_browser_use.py \
  --site "MONCLER_OFFICIAL" \
  --url "https://www.moncler.com/en-int/women/outerwear/all-down-jackets/?forceLocale=en-int&shipToCountry=GB" \
  --query "down jacket" \
  --headful \
  --timeout 120
```

**ポイント:**
- `PYTHONUNBUFFERED=1`: Pythonのバッファリングを無効化
- `-u`: Pythonの標準出力/エラー出力のバッファリングを無効化

### 方法3: ログファイルに出力して読み取る

```bash
cd /home/yn441611/atelier-kyo-manager
source venv/bin/activate

# テスト実行（ログファイルに保存）
LOG_FILE="browser_test_$(date +%Y%m%d_%H%M%S).log"
python tools/run_browser_use.py \
  --site "MONCLER_OFFICIAL" \
  --url "https://www.moncler.com/en-int/women/outerwear/all-down-jackets/?forceLocale=en-int&shipToCountry=GB" \
  --query "down jacket" \
  --headful \
  --timeout 120 \
  2>&1 | tee "$LOG_FILE"

# ログを確認
tail -f "$LOG_FILE"  # リアルタイムで確認
# または
cat "$LOG_FILE" | grep -E "(NavigationDriver|PLP→PDP|ERROR|result\.ok)"
```

### 方法4: コマンドを短く分割

長時間実行されるコマンドを短いコマンドに分割：

```bash
# 1. インポートテスト（短時間）
python -c "from app.agents.browser_use_agent import BrowserUseAgent; print('✓ OK')"

# 2. 設定確認（短時間）
python -c "import json; print(json.dumps({'test': 'ok'}, indent=2))"

# 3. 実際のテスト（長時間）
python -u tools/run_browser_use.py ...
```

### 方法5: `stdbuf` を使用（Linux環境）

```bash
stdbuf -oL -eL python tools/run_browser_use.py ...
```

- `-oL`: 標準出力を行バッファリング
- `-eL`: 標準エラー出力を行バッファリング

## 推奨設定

### `.bashrc` または `.zshrc` に追加

```bash
# Pythonのバッファリングを無効化（開発環境用）
export PYTHONUNBUFFERED=1
```

### Cursor設定（`.cursor/settings.json`）

既に設定済み：
```json
{
  "terminal.integrated.defaultProfile.windows": "WSL",
  "terminal.integrated.profiles.windows": {
    "WSL": {
      "path": "wsl.exe",
      "args": []
    }
  }
}
```

## トラブルシューティング

### 出力が全く表示されない

1. **WSLターミナルで直接実行**
   ```bash
   wsl
   cd /home/yn441611/atelier-kyo-manager
   source venv/bin/activate
   python -u tools/run_browser_use.py ...
   ```

2. **ログファイルを確認**
   ```bash
   ls -lt browser_test_*.log | head -1 | awk '{print $NF}' | xargs tail -100
   ```

### 出力が途中で切れる

- `timeout` コマンドを使用：
  ```bash
  timeout 180 python -u tools/run_browser_use.py ...
  ```

### バッファリングが効いている

- `PYTHONUNBUFFERED=1` を設定
- `python -u` を使用
- `sys.stdout.flush()` をコード内で呼び出す

## テスト方法

以下のコマンドで出力取得をテスト：

```bash
# テスト1: 基本的な出力
python -u -c "import time; print('開始'); time.sleep(1); print('1秒後'); print('完了')"

# テスト2: エラー出力
python -u -c "import sys; print('標準出力'); sys.stderr.write('エラー出力\n'); sys.stderr.flush()"

# テスト3: リアルタイム出力
python -u -c "import time, sys; [print(f'{i}秒', flush=True) or time.sleep(1) for i in range(5)]"
```

すべてのテストで出力が即座に表示されれば、設定は正しく動作しています。

