# WSL セットアップスクリプト実行トラブルシューティング 完了レポート

## 実装日時

2025-12-02 23:00

## 概要

`setup_wsl_env.sh` スクリプトの実行時に発生した問題（NexusCoreのvenv作成エラー）のトラブルシューティングと解決を行いました。また、完了レポートの改善パッチを適用し、ドキュメントの品質を向上させました。

### 目的

- setup_wsl_env.sh の正常実行
- NexusCore と atelier-kyo-manager の両プロジェクトでvenv環境を構築
- 完了レポートの内容改善

### ゴール

- venv作成エラーの原因特定と解決
- スクリプトのパス修正
- 完了レポートへのパッチ適用

### 原則

- 既存のファイル構造を尊重
- ユーザーデータの保護（バックアップ作成）
- 問題の根本原因を特定してから修正

## 実装ステップ

### ステップ1: setup_wsl_env.sh のパス修正

#### 1-1. 問題の発見

初回実行時に以下の警告が発生：
```
>>> [WARN] プロジェクトディレクトリが存在しません: /home/yn441611/dev/nexuscore
>>> [WARN] プロジェクトディレクトリが存在しません: /home/yn441611/dev/atelier-kyo-manager
```

#### 1-2. 原因の特定

スクリプト内のプロジェクトパスが実際のディレクトリ構造と一致していませんでした：

**スクリプトの設定:**
```bash
PROJECTS=(
  "$HOME/dev/nexuscore"
  "$HOME/dev/atelier-kyo-manager"
)
```

**実際のパス:**
- `/home/yn441611/NexusCore`
- `/home/yn441611/atelier-kyo-manager`

#### 1-3. 修正内容

`write` ツールでの修正が反映されなかったため、sedコマンドを使用してパスを修正：

```bash
cd ~/dev
cp setup_wsl_env.sh setup_wsl_env.sh.bak
sed 's|/dev/nexuscore|/NexusCore|g; s|/dev/atelier-kyo-manager|/atelier-kyo-manager|g' setup_wsl_env.sh.bak > setup_wsl_env.sh
chmod +x setup_wsl_env.sh
```

**変更理由:**
- 実際のディレクトリ構造に合わせるため
- 既存運用との整合性を優先

### ステップ2: venv作成エラーの解決

#### 2-1. 問題の発見

NexusCoreでvenv作成時にエラーが発生：
```
>>> プロジェクト設定: /home/yn441611/NexusCore
    - venv を作成します...
Error: Unable to create directory '/home/yn441611/NexusCore/venv'
```

#### 2-2. 原因の特定

`venv` という名前のディレクトリではなく、**シェルスクリプトファイル**が既に存在していました：

```bash
$ file ~/NexusCore/venv
/home/yn441611/NexusCore/venv: Bourne-Again shell script, Unicode text, UTF-8 text executable, with CRLF line terminators
```

**ファイルの内容:**
仮想環境を有効化するためのヘルパースクリプト（activate のエイリアス）

#### 2-3. 修正内容

既存の `venv` ファイルをリネームしてから再実行：

```bash
cd ~/NexusCore
mv venv activate_venv.sh
```

**変更理由:**
- venv ディレクトリを作成するため、同名のファイルを移動
- ユーザーのヘルパースクリプトを削除せず、より明確な名前にリネーム
- activate_venv.sh という名前で機能を維持

### ステップ3: 完了レポートへのパッチ適用

#### 3-1. パッチ内容

以下の5つの修正ポイントを適用：

1. **実行ログを明示** - セットアップスクリプト実行ログの抜粋を追加
2. **Python 3.12 前提のリスク** - Ubuntu 22.04/24.04での対応方法を追記
3. **~/dev ではなく ~/ を採用した理由** - 設計判断の根拠を明確化
4. **Makefile 上書き時のバックアップ注意** - バックアップ手順を追加
5. **OS全体への変更の注意** - システム影響範囲の警告を追加

#### 3-2. 適用箇所

`docs/completion_reports/WSL_SETUP_AND_MAKEFILE_COMPLETION_REPORT.md` に対して、unified diff形式のパッチを適用しました。

**変更理由:**
- レポートの曖昧さや弱点を補完
- 第三者が検証できるようにログ情報を追加
- リスクと制約を明確化

## 変更ファイル一覧

### 変更ファイル

1. **`/home/yn441611/dev/setup_wsl_env.sh`**
   - プロジェクトパスを修正
   - `/dev/nexuscore` → `/NexusCore`
   - `/dev/atelier-kyo-manager` → `/atelier-kyo-manager`

2. **`docs/completion_reports/WSL_SETUP_AND_MAKEFILE_COMPLETION_REPORT.md`**
   - 5つの修正ポイントを適用
   - 実行ログ、リスク説明、設計判断の根拠を追加

3. **`/home/yn441611/NexusCore/venv` → `activate_venv.sh`**
   - ファイル名変更（リネーム）
   - venvディレクトリ作成のための準備

### バックアップファイル

1. **`/home/yn441611/dev/setup_wsl_env.sh.bak`**
   - 修正前のスクリプトのバックアップ

## 動作確認結果

### スクリプトの実行状況

#### 初回実行（パス修正前）
```
=== WSL 共通ベース環境セットアップ (NexusCore / atelier-kyo-manager) ===
>>> apt パッケージを更新・インストールします...
[sudo] password for yn441611:
Hit:1 http://archive.ubuntu.com/ubuntu noble InRelease
...
python3 is already the newest version (3.12.3-0ubuntu2.1).
>>> Git LFS を初期化します...
Git LFS initialized.
>>> Git グローバル設定を更新します...
>>> [WARN] プロジェクトディレクトリが存在しません: /home/yn441611/dev/nexuscore
>>> [WARN] プロジェクトディレクトリが存在しません: /home/yn441611/dev/atelier-kyo-manager
```

#### パス修正後の実行（venv作成エラー）
```
>>> プロジェクト設定: /home/yn441611/NexusCore
    - venv を作成します...
Error: Unable to create directory '/home/yn441611/NexusCore/venv'
```

#### venvファイル確認
```bash
$ ls -la ~/NexusCore/ | grep venv
drwxr-xr-x  7 yn441611 yn441611        4096 Nov 14 15:23 .venv
-rw-r--r--  1 yn441611 yn441611         903 Nov 27 15:46 venv

$ file ~/NexusCore/venv
/home/yn441611/NexusCore/venv: Bourne-Again shell script
```

#### リネーム後
```bash
$ cd ~/NexusCore && mv venv activate_venv.sh
```

### 最終状態

- パス修正: ✅ 完了
- venvファイルリネーム: ✅ 完了
- 完了レポートパッチ適用: ✅ 完了

### 次回実行時の期待される動作

ユーザーがターミナルで以下を実行すると、正常に完了するはずです：

```bash
cd ~/dev
./setup_wsl_env.sh
```

**期待される出力:**
```
=== WSL 共通ベース環境セットアップ (NexusCore / atelier-kyo-manager) ===
>>> apt パッケージを更新・インストールします...
[sudo] password for yn441611:
[パッケージは既にインストール済み]
>>> Git LFS を初期化します...
Git LFS initialized.
>>> Git グローバル設定を更新します...
>>> プロジェクト設定: /home/yn441611/NexusCore
    - venv を作成します...
    [venv 作成中...]
    - requirements.txt をインストールします...
    [pip install実行...]
    - .cursor/config.yaml を作成しました: /home/yn441611/NexusCore/.cursor/config.yaml
>>> プロジェクト設定完了: /home/yn441611/NexusCore
>>> プロジェクト設定: /home/yn441611/atelier-kyo-manager
    - 既存 venv を検出: /home/yn441611/atelier-kyo-manager/venv
    - requirements.txt をインストールします...
    [pip install実行...]
    - .cursor/config.yaml を作成しました: /home/yn441611/atelier-kyo-manager/.cursor/config.yaml
>>> プロジェクト設定完了: /home/yn441611/atelier-kyo-manager
=== セットアップ完了 ===
```

## 設計上の改善点

### トラブルシューティングプロセスの確立

1. **段階的な問題特定**
   - まずログを確認してエラーの種類を特定
   - ファイルシステムの状態を確認
   - 根本原因を特定してから修正

2. **データ保護**
   - 修正前に必ずバックアップ作成
   - 既存のユーザーファイルを削除せずリネーム

3. **確認と検証**
   - 各修正後に状態を確認
   - 期待される動作を明確化

### ドキュメント品質の向上

1. **実行ログの明示**
   - 第三者が検証できるようにログを記録
   - 実行結果の抜粋を追加

2. **リスクの明確化**
   - Python 3.12 非標準搭載の対応方法
   - OS全体への影響範囲の警告

3. **設計判断の根拠**
   - なぜ現行パスを採用したかの説明
   - 将来的な改善案の提示

## 既知の制約・注意事項

### 実行上の制約

1. **sudo パスワードが必要**
   - apt-get によるパッケージインストールで sudo 権限が必要
   - 自動化には sudoers の設定が必要（セキュリティ上推奨しない）

2. **既存のvenvファイル**
   - NexusCoreに `venv` という名前のシェルスクリプトが存在していた
   - 今回 `activate_venv.sh` にリネーム済み
   - 他のプロジェクトでも同様の問題が発生する可能性がある

3. **パスの不一致**
   - スクリプトと実際のディレクトリ構造が異なっていた
   - 今後、プロジェクトを追加する際は注意が必要

### 制限事項やトレードオフ

1. **ディレクトリ構造の不統一**
   - 本来は `~/dev/` 配下に統一することが望ましい
   - 既存運用との整合性を優先し、現行パスを維持

2. **write ツールの制約**
   - write ツールでのファイル更新が反映されない場合があった
   - WSL環境では sed コマンドの直接使用が確実

### 移行時の注意点

1. **既存ヘルパースクリプトの扱い**
   - `venv` ファイルは便利なヘルパースクリプトだった
   - リネーム後も `activate_venv.sh` として機能を維持
   - ユーザーは `source activate_venv.sh` で引き続き使用可能

2. **バックアップの重要性**
   - `setup_wsl_env.sh.bak` が作成済み
   - 問題が発生した場合はバックアップから復元可能

## 次のステップ

### 推奨されるフォローアップアクション

1. **スクリプトの最終実行**
   ```bash
   cd ~/dev
   ./setup_wsl_env.sh
   ```
   - パスワードを入力して実行
   - venv作成とrequirements.txtインストールを確認

2. **動作確認**
   ```bash
   # NexusCore
   cd ~/NexusCore
   source venv/bin/activate
   python --version
   pytest --version
   
   # atelier-kyo-manager
   cd ~/atelier-kyo-manager
   source venv/bin/activate
   python --version
   pytest --version
   ```

3. **ディレクトリ整理の検討**
   - 将来的に `~/dev/` 配下にプロジェクトを移動することを検討
   - 新規プロジェクトは `~/dev/` 配下に作成
   - オンボーディング効率の向上

4. **activate_venv.sh の活用**
   - リネームしたヘルパースクリプトを引き続き使用
   - 必要に応じて `.bashrc` や `.zshrc` にエイリアスを追加

5. **BrowserUseAgent リファクタリング（次のタスク）**
   - 例外処理とretry統一のリファクタリング
   - Playwright操作の共通ラッパー導入
   - FailureAnalysisAgent / SelfHealingAgent との連携強化

## 関連ファイル

### 修正したファイル

- `/home/yn441611/dev/setup_wsl_env.sh` - WSL環境セットアップスクリプト
- `docs/completion_reports/WSL_SETUP_AND_MAKEFILE_COMPLETION_REPORT.md` - 完了レポート
- `/home/yn441611/NexusCore/activate_venv.sh` - リネームしたヘルパースクリプト

### バックアップファイル

- `/home/yn441611/dev/setup_wsl_env.sh.bak` - スクリプトのバックアップ

### 関連ドキュメント

- `docs/completion_reports/WSL_SETUP_AND_MAKEFILE_COMPLETION_REPORT.md` - 元の完了レポート
- `docs/completion_reports/OFFICIAL_DOCS_ADDITION_COMPLETION_REPORT.md` - 公式ドキュメント追加レポート

## まとめ

setup_wsl_env.sh の実行時に発生した2つの問題（パスの不一致とvenvファイル名の競合）を特定し、解決しました。また、完了レポートの品質を向上させるパッチを適用しました。

次回ユーザーがスクリプトを実行する際には、正常に両プロジェクトの環境構築が完了するはずです。

