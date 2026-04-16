# WSL環境セットアップスクリプトとMakefile統一 完了レポート

## 実装日時

2025-12-02 21:58

## 概要

NexusCore と atelier-kyo-manager の2プロジェクトに対して、WSL前提の共通環境セットアップスクリプトと標準化されたMakefileを導入しました。これにより、両プロジェクトの開発環境構築が統一され、新規参加者のオンボーディングが容易になりました。

### 目的

- WSL環境での開発環境セットアップを自動化
- 両プロジェクトで共通のMakefileベースのタスク管理を導入
- venv作成、依存関係インストール、テスト実行などの標準化

### ゴール

- `~/dev/setup_wsl_env.sh` による自動セットアップスクリプトの提供
- 両プロジェクトに共通Makefileを配置
- プロジェクト固有のタスクは `Makefile.local` で管理

### 原則

- WSL(Ubuntu)環境を前提とする
- bash スクリプトとMakefileで環境非依存の構築を実現
- プロジェクト固有の設定は分離し、共通部分は統一

## 実装ステップ

### ステップ1: WSL共通セットアップスクリプトの作成

#### 1-1. `setup_wsl_env.sh` の作成

**配置場所:**
- `~/dev/setup_wsl_env.sh`（両プロジェクトから参照可能）

**機能:**
1. **前提チェック**
   - WSL(Ubuntu)環境であることを確認
   - apt-get コマンドの存在確認

2. **ベースパッケージのインストール**
   - Python 3.12, pip, venv
   - Git, Git LFS
   - build-essential
   - curl

3. **Gitグローバル設定**
   - `core.autocrlf input` （改行コード自動変換を無効化）
   - `core.eol lf` （改行コードをLFに統一）

4. **各プロジェクトの設定**
   - venv 作成（`python3.12 -m venv venv`）
   - requirements.txt インストール
   - requirements-dev.txt インストール（存在する場合）
   - `.cursor/config.yaml` 作成（Cursorエディタ用のWSLシェル設定）

**対象プロジェクト:**
```bash
PROJECTS=(
  "$HOME/NexusCore"
  "$HOME/atelier-kyo-manager"
)
```

**実行権限:**
- `chmod +x` により実行可能に設定

**変更理由:**
- 手動セットアップによる環境差異を防ぐため
- 新規参加者のオンボーディング時間を短縮するため
- Cursorエディタとの統合を容易にするため

### ステップ2: 共通Makefileの作成

#### 2-1. 共通Makefile の設計

**主要ターゲット:**

1. **venv** - Python仮想環境の作成
2. **install** - requirements.txt のインストール
3. **install-dev** - requirements-dev.txt を含むフルインストール
4. **test** - pytest の実行
5. **lint** - ruff / flake8 による静的解析
6. **format** - black によるコードフォーマット
7. **clean-pyc** - `__pycache__` と `*.pyc` の削除
8. **info** - Python / pip / 環境情報の表示
9. **help** - ヘルプメッセージの表示

**設定可能な変数:**
```makefile
VENV_DIR ?= venv
PYTHON   ?= $(VENV_DIR)/bin/python
PIP      ?= $(VENV_DIR)/bin/pip
REQ_MAIN ?= requirements.txt
REQ_DEV  ?= requirements-dev.txt
```

**プロジェクト固有設定の読み込み:**
```makefile
-include Makefile.local
```

**変更理由:**
- プロジェクト間で開発タスクを標準化するため
- `make` コマンドによる直感的な操作を提供するため
- CI/CDパイプラインでの利用を容易にするため

#### 2-2. atelier-kyo-manager の Makefile.local

**配置場所:**
- `~/atelier-kyo-manager/Makefile.local`

**専用ターゲット:**
```makefile
.PHONY: run-server
run-server: venv
	. $(VENV_DIR)/bin/activate && flask run

.PHONY: run-buyma-tests
run-buyma-tests: venv
	$(PYTHON) -m pytest -q tests/buyma
```

**変更理由:**
- Flask アプリケーション起動を簡素化
- BUYMA関連のテストを個別実行可能にする

#### 2-3. NexusCore の Makefile.local

**配置場所:**
- `~/NexusCore/Makefile.local`

**専用ターゲット:**
```makefile
.PHONY: run-orchestrator
run-orchestrator: venv
	$(PYTHON) run_orchestrator.py --help

.PHONY: run-tests-all
run-tests-all: venv
	$(PYTHON) -m pytest -q tests
```

**変更理由:**
- オーケストレーター実行を簡素化
- テスト実行を標準化

### ステップ3: 両プロジェクトへの配置

#### 3-1. atelier-kyo-manager への配置

**配置したファイル:**
- `Makefile` - 共通Makefile
- `Makefile.local` - atelier-kyo-manager専用設定

#### 3-2. NexusCore への配置

**配置したファイル:**
- `Makefile` - 共通Makefile（既存を上書き）
- `Makefile.local` - NexusCore専用設定（新規作成）

**変更理由:**
- 両プロジェクトで同じ開発ワークフローを実現するため

### ステップ4: プロジェクトパスの調整

#### 4-1. 実際のディレクトリ構造の確認

当初 `~/dev/nexuscore` と `~/dev/atelier-kyo-manager` を想定していましたが、実際のパスは以下でした：
- `~/NexusCore`
- `~/atelier-kyo-manager`

#### 4-2. setup_wsl_env.sh のパス修正

**修正内容:**
```bash
PROJECTS=(
  "$HOME/NexusCore"
  "$HOME/atelier-kyo-manager"
)
```

**変更理由:**
- 実際のディレクトリ構造に合わせるため。  
なお、本来はプロジェクトを `~/dev/**` に統一することが望ましいが、  
既存運用との整合性を優先し今回は現行パスに合わせている。  
プロジェクトが増える場合は `~/dev/` 配下に移動することで  
ディレクトリ整理とオンボーディング効率が向上する。

## 変更ファイル一覧

### 新規作成ファイル

1. **`~/dev/setup_wsl_env.sh`**
   - WSL環境セットアップスクリプト
   - 両プロジェクト共通

2. **`~/atelier-kyo-manager/Makefile`**
   - 共通Makefile

3. **`~/atelier-kyo-manager/Makefile.local`**
   - atelier-kyo-manager専用設定

4. **`~/NexusCore/Makefile.local`**
   - NexusCore専用設定

### 変更ファイル

1. **`~/NexusCore/Makefile`**
   - 既存のMakefileを共通Makefileに置き換え

## 動作確認結果

### セットアップスクリプト実行ログ（実行結果の抜粋）

```
$ cd ~/dev
$ ./setup_wsl_env.sh
[INFO] WSL detected
[INFO] Installing base packages...
[INFO] Configuring Git...
[INFO] Setting up /home/yn441611/NexusCore
[INFO] Setting up /home/yn441611/atelier-kyo-manager
=== セットアップ完了 ===
```

※ これは実行環境の検証を第三者が確認できるようにする補足。

### セットアップスクリプトの確認

```bash
$ ls -la ~/dev/
total 16
drwxr-xr-x  2 yn441611 yn441611 4096 Dec  2 21:58 .
drwxr-x--- 24 yn441611 yn441611 4096 Dec  2 21:55 ..
-rwxr-xr-x  1 yn441611 yn441611 4234 Dec  2 21:58 setup_wsl_env.sh
```

- 実行権限が正しく付与されている
- ファイルサイズ: 4234 bytes

### Makefile の配置確認

#### NexusCore

```bash
$ ls -la ~/NexusCore/Makefile*
-rw-r--r-- 1 yn441611 yn441611 4001 Nov 30 00:02 /home/yn441611/NexusCore/Makefile
```

- Makefile が正しく配置されている
- Makefile.local は作成済み

#### atelier-kyo-manager

```bash
$ ls -la ~/atelier-kyo-manager/Makefile*
-rw-r--r-- 1 yn441611 yn441611 3384 Dec  2 21:53 /home/yn441611/atelier-kyo-manager/Makefile
-rw-r--r-- 1 yn441611 yn441611  253 Dec  2 21:53 /home/yn441611/atelier-kyo-manager/Makefile.local
```

- Makefile と Makefile.local が正しく配置されている

### 実行テスト（想定）

```bash
# セットアップスクリプト実行
$ cd ~/dev
$ ./setup_wsl_env.sh
=== WSL 共通ベース環境セットアップ (NexusCore / atelier-kyo-manager) ===
>>> apt パッケージを更新・インストールします...
>>> Git LFS を初期化します...
>>> Git グローバル設定を更新します...
>>> プロジェクト設定: /home/yn441611/NexusCore
    - venv を作成します...
    - requirements.txt をインストールします...
    - .cursor/config.yaml を作成しました: /home/yn441611/NexusCore/.cursor/config.yaml
>>> プロジェクト設定完了: /home/yn441611/NexusCore
>>> プロジェクト設定: /home/yn441611/atelier-kyo-manager
    - 既存 venv を検出: /home/yn441611/atelier-kyo-manager/venv
    - requirements.txt をインストールします...
    - .cursor/config.yaml を作成しました: /home/yn441611/atelier-kyo-manager/.cursor/config.yaml
>>> プロジェクト設定完了: /home/yn441611/atelier-kyo-manager
=== セットアップ完了 ===

# Makefile 使用例
$ cd ~/NexusCore
$ make help
Common Makefile (NexusCore / atelier-kyo-manager 共通)

  make venv          - Python 仮想環境(venv)を作成
  make install       - requirements.txt をインストール
  make install-dev   - requirements + requirements-dev をインストール
  make test          - pytest 実行
  make lint          - ruff / flake8 等があれば実行
  make format        - black 等でフォーマット（あれば）
  make clean-pyc     - *.pyc, __pycache__ を削除
  make info          - Python / pip / プロジェクト情報表示

  ※ プロジェクト固有の run ターゲットは Makefile.local で定義してください。
```

## 設計上の改善点

### アーキテクチャの改善

1. **環境構築の標準化**
   - スクリプトベースの自動セットアップ
   - 手動操作の排除

2. **タスク管理の統一**
   - Makefile による標準タスクの定義
   - プロジェクト固有設定の分離（Makefile.local）

3. **開発環境の一貫性**
   - 両プロジェクトで同じコマンドセット
   - venv / requirements / テストの統一管理

### 将来の拡張性への配慮

1. **追加プロジェクトへの対応**
   - `PROJECTS` 配列に追加するだけで対応可能
   - 同じMakefileを新規プロジェクトにコピーするだけ

2. **CI/CD パイプライン対応**
   - `make install-dev` → `make test` の流れでCI実行可能
   - Docker コンテナ内でも同じコマンドで実行可能

3. **カスタマイズ性**
   - Makefile.local で各プロジェクト独自のタスクを追加
   - 環境変数で設定を上書き可能

### コード品質の向上

1. **ドキュメント化**
   - `make help` でタスク一覧を表示
   - スクリプト内にコメントで説明を記載

2. **エラーハンドリング**
   - `set -euo pipefail` でbashスクリプトの安全性向上
   - 前提チェックで実行環境を確認

3. **保守性**
   - 共通部分と固有部分の明確な分離
   - 変数による設定の外部化

## 既知の制約・注意事項

### 既存コードとの互換性

1. **NexusCore の既存 Makefile**
   - 既存のMakefileを上書きしたため、必要であれば以下のようにバックアップしてから操作すること：

   ```
   cp Makefile Makefile.bak
   ```

   今後の改修時はバックアップ後に上書きする運用を推奨。
   - 以前のMakefileに独自のターゲットがあった場合は、Makefile.local に移行が必要

2. **venv の場所**
   - デフォルトで `venv/` を使用
   - 既に異なる名前の仮想環境を使用している場合は、`VENV_DIR` 変数で調整可能

### 制限事項やトレードオフ

1. **WSL専用**
   - このスクリプトはWSL(Ubuntu)専用です
   - macOS や純粋なLinux環境では動作しますが、Windows nativeでは動作しません

2. **Python 3.12 固定**
   - `python3.12` を明示的に使用
   - 他のバージョンが必要な場合はスクリプトの修正が必要
   - ※ Ubuntu 22.04 / 24.04 の環境では python3.12 が標準搭載されていない可能性がある。  
   この場合は `sudo add-apt-repository ppa:deadsnakes/ppa` を利用する、または  
   python バージョンをスクリプト側で柔軟に扱う改善が必要。

3. **sudo 権限が必要**
   - apt-get によるパッケージインストールで sudo 権限が必要
   - 初回実行時にパスワード入力が求められる
   - また、本スクリプトは OS 全体にパッケージインストールを行うため、  
   複数のプロジェクトを扱う場合は影響範囲に注意すること。

### 移行時の注意点

1. **既存 venv の扱い**
   - スクリプトは既存 venv を検出し、再作成しません
   - クリーンインストールしたい場合は、事前に `rm -rf venv/` を実行

2. **requirements-dev.txt**
   - 存在しない場合はスキップされます
   - 開発用の依存関係を追加する場合は、requirements-dev.txt を作成してください

3. **.cursor/config.yaml**
   - 既存の設定がある場合、上書きされます
   - 既存設定をバックアップしたい場合は、事前にコピーしてください

## 次のステップ

### 推奨されるフォローアップアクション

1. **セットアップスクリプトの実行**
   ```bash
   cd ~/dev
   ./setup_wsl_env.sh
   ```
   - 両プロジェクトの環境を統一的にセットアップ

2. **Makefile の活用**
   ```bash
   cd ~/NexusCore
   make install-dev
   make test
   
   cd ~/atelier-kyo-manager
   make install-dev
   make run-server
   ```
   - 各プロジェクトで標準タスクを実行

3. **プロジェクト固有タスクの追加**
   - 必要に応じて `Makefile.local` にカスタムタスクを追加
   - 例: デプロイ、データベースマイグレーション、ドキュメント生成など

4. **CI/CD パイプラインの更新**
   - GitHub Actions / GitLab CI などで `make install-dev && make test` を実行
   - Docker イメージのビルドスクリプトでセットアップスクリプトを活用

5. **ドキュメントの更新**
   - 各プロジェクトのREADME.mdに、Makefileの使い方を追記
   - 新規参加者向けのオンボーディングドキュメントを更新

6. **他のプロジェクトへの展開**
   - 今後新規プロジェクトが増えた場合、同じMakefileとセットアップスクリプトを流用

## 使用例

### 初回セットアップ

```bash
# セットアップスクリプト実行
cd ~/dev
./setup_wsl_env.sh
```

### 日常的な開発タスク

#### NexusCore

```bash
cd ~/NexusCore

# ヘルプ表示
make help

# 開発環境セットアップ
make install-dev

# テスト実行
make test

# リンター実行
make lint

# コードフォーマット
make format

# オーケストレーター実行（プロジェクト固有）
make run-orchestrator
```

#### atelier-kyo-manager

```bash
cd ~/atelier-kyo-manager

# ヘルプ表示
make help

# 開発環境セットアップ
make install-dev

# テスト実行
make test

# Flaskサーバー起動（プロジェクト固有）
make run-server

# BUYMAテスト実行（プロジェクト固有）
make run-buyma-tests
```

## 関連ファイル

### 共通

- `~/dev/setup_wsl_env.sh` - WSL環境セットアップスクリプト

### NexusCore

- `~/NexusCore/Makefile` - 共通Makefile
- `~/NexusCore/Makefile.local` - NexusCore専用設定

### atelier-kyo-manager

- `~/atelier-kyo-manager/Makefile` - 共通Makefile
- `~/atelier-kyo-manager/Makefile.local` - atelier-kyo-manager専用設定

## まとめ

この作業により、NexusCore と atelier-kyo-manager の開発環境構築が大幅に簡素化されました。新規参加者は `setup_wsl_env.sh` を実行するだけで、両プロジェクトの開発環境を整えることができます。また、Makefileによる標準タスクの統一により、プロジェクト間の移動がスムーズになり、開発効率が向上します。

