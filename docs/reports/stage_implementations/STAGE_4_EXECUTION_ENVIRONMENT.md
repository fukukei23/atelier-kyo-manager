# Stage 4: 動作確認の実行環境について

**作成日時**: 2025-11-28  
**目的**: MONCLER_OFFICIALの動作確認をどこで実行すべきか明確化

---

## 1. 実行環境の選択肢

### 1.1 WSL環境（現在の作業環境）

**メリット**:
- ✅ 既にWSL環境で作業している
- ✅ ヘッドレスモード（`--headless`）なら問題なく動作
- ✅ PlaywrightはWSLでも動作する
- ✅ コードの変更確認が容易

**デメリット**:
- ⚠️ ヘッドフルモード（`--headful`）はX11 forwardingが必要（設定がやや複雑）
- ⚠️ ブラウザの表示を直接確認できない（ヘッドレスモードの場合）

**推奨**: **ヘッドレスモードでの動作確認ならWSLで十分**

### 1.2 Windows環境

**メリット**:
- ✅ ヘッドフルモード（`--headful`）が簡単に使える
- ✅ ブラウザの表示を直接確認できる
- ✅ デバッグが容易

**デメリット**:
- ⚠️ Windows環境への切り替えが必要
- ⚠️ コードの変更確認がWSLとWindows間で必要

**推奨**: **ヘッドフルモードで詳細な動作確認が必要な場合のみ**

---

## 2. 実行方法

### 2.1 WSL環境での実行（推奨）

#### ヘッドレスモードでの動作確認

```bash
# WSL環境で実行
cd /home/yn441611/atelier-kyo-manager
source venv/bin/activate  # または source myenv/Scripts/activate

# MONCLER_OFFICIALの動作確認（--headlessはデフォルトなので省略可能）
python tools/run_browser_use.py \
    --site MONCLER_OFFICIAL \
    --url "https://www.moncler.com/en-int/women/outerwear/all-down-jackets/?forceLocale=en-int&shipToCountry=GB" \
    --query "down jacket"

# または、Orchestrator経由で実行
python run_orchestrator.py \
    "down jacket" \
    --site MONCLER_OFFICIAL \
    --headless
```

#### ヘッドフルモードでの実行（X11 forwardingが必要）

```bash
# WSL環境でX11 forwardingを有効化（Windows側でXサーバーが必要）
export DISPLAY=:0  # または Windows側のXサーバーのアドレス

# ヘッドフルモードで実行
python tools/run_browser_use.py \
    --site MONCLER_OFFICIAL \
    --url "https://www.moncler.com/en-int/women/outerwear/all-down-jackets/?forceLocale=en-int&shipToCountry=GB" \
    --query "down jacket" \
    --headful
```

### 2.2 Windows環境での実行

#### PowerShell経由での実行

```powershell
# Windows環境で実行
cd C:\Users\USER\tools\atelier-kyo-manager
.\.venv\Scripts\activate

# ヘッドレスモード
python tools\run_browser_use.py `
    --site MONCLER_OFFICIAL `
    --url "https://www.moncler.com/en-int/women/outerwear/all-down-jackets/?forceLocale=en-int&shipToCountry=GB" `
    --query "down jacket" `
    --headless

# ヘッドフルモード（ブラウザ表示あり）
python tools\run_browser_use.py `
    --site MONCLER_OFFICIAL `
    --url "https://www.moncler.com/en-int/women/outerwear/all-down-jackets/?forceLocale=en-int&shipToCountry=GB" `
    --query "down jacket" `
    --headful
```

#### 既存のPowerShellスクリプトを使用

```powershell
# scripts/run_moncler_diagnostics_windows.ps1 を使用
.\scripts\run_moncler_diagnostics_windows.ps1 `
    -Query "down jacket" `
    -Headless
```

---

## 3. 推奨アプローチ

### Phase 1: 基本的な動作確認（WSL環境で十分）

**目的**: Phase 1の実装が正常に動作するか確認

**実行環境**: **WSL環境（ヘッドレスモード）**

**確認項目**:
- ✅ エラーが発生しないか
- ✅ PLP → PDP動作が正常に動作するか
- ✅ ログが適切に出力されているか
- ✅ スクリーンショットが保存されているか

**実行コマンド**:
```bash
cd /home/yn441611/atelier-kyo-manager
source venv/bin/activate
python tools/run_browser_use.py \
    --site MONCLER_OFFICIAL \
    --url "https://www.moncler.com/en-int/women/outerwear/all-down-jackets/?forceLocale=en-int&shipToCountry=GB" \
    --query "down jacket" \
    --headless
```

### Phase 2: 詳細な動作確認（必要に応じてWindows環境）

**目的**: ブラウザの表示を直接確認して、細かい動作を検証

**実行環境**: **Windows環境（ヘッドフルモード）**

**確認項目**:
- ✅ ブラウザの表示が正常か
- ✅ オーバーレイ（Cookieバナー、ジオモーダル）の処理が正常か
- ✅ スクロールやリンククリックが正常か

**実行コマンド**:
```powershell
python tools\run_browser_use.py `
    --site MONCLER_OFFICIAL `
    --url "https://www.moncler.com/en-int/women/outerwear/all-down-jackets/?forceLocale=en-int&shipToCountry=GB" `
    --query "down jacket" `
    --headful
```

---

## 4. 結論

### **MONCLER_OFFICIALの動作確認はWSL環境で実行可能**

- ✅ **ヘッドレスモード**: WSL環境で問題なく動作
- ✅ **基本的な動作確認**: WSL環境で十分
- ⚠️ **ヘッドフルモード**: Windows環境の方が簡単（WSLでも可能だがX11 forwardingが必要）

### **推奨フロー**

1. **Phase 1（基本確認）**: WSL環境でヘッドレスモードで実行
   - エラーの有無
   - ログとスクリーンショットの確認
   - 既存動作の維持確認

2. **Phase 2（詳細確認）**: 必要に応じてWindows環境でヘッドフルモードで実行
   - ブラウザの表示確認
   - 細かい動作の検証

---

## 5. 実行結果の確認場所

実行結果は以下の場所に保存されます：

- **実行ログ**: `instance/logs/runner_YYYYMMDD.log`
- **実行結果**: `instance/runs/<RUN_ID>/`
  - `run.json`: 実行結果のJSON
  - `screenshots/`: スクリーンショット
  - `videos/`: ビデオ（有効な場合）
  - `failure_dom.html`: 失敗時のDOM
  - `fail_snapshot.md`: 失敗時のスナップショット

---

## 6. トラブルシューティング

### WSL環境でPlaywrightが動作しない場合

```bash
# Playwrightのブラウザをインストール
playwright install chromium
```

### ヘッドフルモードでX11 forwardingが必要な場合

```bash
# Windows側でXサーバー（VcXsrv等）を起動
# WSL側でDISPLAYを設定
export DISPLAY=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}'):0
```

### Windows環境でasyncioエラーが発生する場合

`run_orchestrator.py`にWindows用のasyncio policy設定が含まれているため、通常は問題ありません。

